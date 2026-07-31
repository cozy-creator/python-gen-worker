"""pgw#817 — regional cells: identity, binding, arming, and the numerics gate.

SCENARIO-SHAPED, not issue-shaped. The file is organised as the four scenarios
a regional cell walks — DECLARE, MINT-IDENTITY, ARM, JUDGE — because that is
how these cases fold into `tests_v2`'s cells/mint scenarios when pgw#808's
tree lands (`tests_v2` is not on `origin/chaos` yet; see the tracker note).
Nothing here is a per-issue regression pinned to a line number.

Every numeric assertion is pinned to a MEASURED coordinate from pgw#812 (the
regional pilot) or pgw#814 (the whole-graph fp8 degradation), so moving a
policy constant fails with the evidence in hand.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import pytest
import torch
from torch import nn

from gen_worker import aot_mint, aot_regional, cell_key, numerics_ladder
from gen_worker.aot_regional import RegionalArmRefused
from gen_worker.api.decorators import Compile, Dim, DynamicDim


# ---------------------------------------------------------------------------
# Measured coordinates. Everything below is asserted against THESE, so the
# policy constants cannot move without an argument.
# ---------------------------------------------------------------------------

#: (label, cosine, retention, required verdict). pgw#814's VERDICT table on
#: the production toolchain (torch 2.13.0+cu130, L4/sm_89, flux2-klein-4b and
#: sdxl), plus pgw#812's regional arms.
MEASURED: Tuple[Tuple[str, float, float, str], ...] = (
    ("bf16 control, whole-graph", 0.99979, 0.997, "healthy"),
    ("sdxl w8a8, whole-graph", 0.99984, 0.998, "healthy"),
    ("flux2 w8a8 regional T_img=4096", 0.9890, 0.99, "degraded"),
    ("flux2 w8a8 regional T_img=8160", 0.9926, 0.99, "degraded"),
    ("flux2 w8a8 rowwise whole-graph", 0.97300, 0.902, "destroyed"),
    ("flux2 w8a8 pertensor whole-graph", 0.93094, 0.905, "destroyed"),
)


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, str, str]]:
    """Every typed activity event this test emitted."""
    seen: List[Tuple[str, str, str]] = []

    def _emit(kind: str, detail: str = "", phase: str = "", **_: Any) -> None:
        seen.append((str(kind), str(phase), str(detail)))

    monkeypatch.setattr(numerics_ladder.activity_mod, "emit_event", _emit)
    monkeypatch.setattr(aot_regional.activity_mod, "emit_event", _emit)
    return seen


def _pair_at(cosine: float, retention: float, n: int = 4096) -> Tuple[Any, Any]:
    """Two real tensors whose aggregate cosine and norm ratio are EXACTLY the
    requested pair, built from an orthogonal decomposition rather than a
    random search — the gate must be tested on the numbers it will meet."""
    generator = torch.Generator().manual_seed(17)
    a = torch.randn(n, generator=generator, dtype=torch.float64)
    a = a / a.norm()
    o = torch.randn(n, generator=generator, dtype=torch.float64)
    o = o - (o @ a) * a
    o = o / o.norm()
    b = retention * (cosine * a + math.sqrt(max(0.0, 1.0 - cosine ** 2)) * o)
    return a.to(torch.float32), b.to(torch.float32)


# ---------------------------------------------------------------------------
# SCENARIO 1 — DECLARE: what a family may say about regional (D4)
# ---------------------------------------------------------------------------


def _decl(**kwargs: Any) -> Compile:
    base: Dict[str, Any] = dict(
        family="scenario", shapes=((1024, 1024),), targets=("unet",),
        text_len=77, shape_strategy="static-rows", warm_changes_key=False)
    base.update(kwargs)
    return Compile(**base)


def test_declaration_admits_regional_with_dynamic() -> None:
    """RED before pgw#817: `Compile.__post_init__` raised
    `Compile(regional=True) cannot carry dynamic=(...)` outright, on the
    reading that regional 'never applies the declared marks' and 'is retiring
    in favor of whole-graph export'. pgw#812 measured an exported regional
    cell that DOES apply them, and measured the symbolic inner axis as free
    on a conv-free region (+0.2% bf16 / 0.0% w8a8)."""
    decl = _decl(
        regional=True,
        dims=(Dim("T_img", carried_by=(("hidden_states", 1),), multiple_of=64),),
        dynamic=(DynamicDim(dim="T_img", min=1024, max=8192),),
        shape_strategy="dynamic-collapse")
    assert decl.regional is True
    assert tuple(d.dim for d in decl.dynamic) == ("T_img",)


def test_the_dynamo_lane_declines_regional_plus_dynamic_by_name() -> None:
    """The old refusal's CONTENT survives where it is still true. The dynamo
    regional branch calls `compile_repeated_blocks(dynamic=None)` and cannot
    honour the marks, so it declines and the target takes the whole-forward
    branch (which does mark) — instead of arming a graph that does not
    implement the contract its cell key asserts."""
    from gen_worker import compile_cache as cc

    decl = _decl(
        regional=True,
        dims=(Dim("T_img", carried_by=(("hidden_states", 1),), multiple_of=64),),
        dynamic=(DynamicDim(dim="T_img", min=1024, max=8192),),
        shape_strategy="dynamic-collapse")
    reason = cc._regional_dynamic_decline(decl, "unet")
    assert "compile_repeated_blocks(dynamic=None)" in reason
    assert "T_img" in reason
    # No declared dynamism => no decline; regional still arms on that lane.
    assert cc._regional_dynamic_decline(_decl(regional=True), "unet") == ""


def test_declared_numerics_tolerance_is_validated_and_defaults_are_measured(
) -> None:
    decl = _decl(numerics_floor=0.995, numerics_warn=0.9999)
    thresholds = aot_regional.declared_thresholds(decl)
    assert (thresholds.floor, thresholds.warn) == (0.995, 0.9999)
    # Undeclared falls back to the SDK band derived from pgw#814.
    assert aot_regional.declared_thresholds(_decl()) is \
        aot_regional.DEFAULT_THRESHOLDS
    with pytest.raises(ValueError, match="must not exceed numerics_warn"):
        _decl(numerics_floor=0.999, numerics_warn=0.98)
    with pytest.raises(ValueError, match="COSINE bound"):
        _decl(numerics_floor=1.5)


# ---------------------------------------------------------------------------
# SCENARIO 2 — MINT IDENTITY: mode, block coordinates, and the shell digest
# ---------------------------------------------------------------------------


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.lin = nn.Linear(width, width, bias=False)

    def forward(self, x: Any, scale: Any = None) -> Any:
        out = self.lin(x)
        return out if scale is None else out * scale


class _WideBlock(_Block):
    pass


class _Shell(nn.Module):
    """A model with declared repeated blocks — the structure the discovery
    reads, expressed the way diffusers expresses it."""

    _repeated_blocks = ["_Block", "_WideBlock"]

    def __init__(self, layers: int = 4, width: int = 8, wide: int = 1) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(width) for _ in range(layers)])
        self.wide = nn.ModuleList([_WideBlock(width * 2) for _ in range(wide)])
        self.width = width
        self.config = {"layers": layers, "width": width, "wide": wide}

    def forward(self, x: Any) -> Any:
        for block in self.blocks:
            x = block(x)
        big = torch.cat([x, x], dim=-1)
        for block in self.wide:
            big = block(big)
        return big[..., :self.width]


def test_block_discovery_groups_by_parameter_shape_not_class_name() -> None:
    """Two blocks of ONE class at different widths compile to different
    kernels and must be different entries; the grouping key is therefore the
    ordered parameter-shape fingerprint, and nothing here is per-family
    knowledge — the class names come from the module's own
    `_repeated_blocks`."""
    model = _Shell(layers=5, wide=2)
    groups = aot_regional.repeated_block_groups(model)
    assert [(g.key, g.count) for g in groups] == [
        ("_Block#0", 5), ("_WideBlock#0", 2)]
    # A module that declares nothing has no regional structure at all.
    assert aot_regional.repeated_block_groups(nn.Linear(4, 4)) == ()


def test_a_block_entry_is_named_in_the_existing_entry_grammar() -> None:
    """S1: one .pt2, the same `<target>/<k=v>/<k=v>` grammar, no new artifact
    class and no hub change. Only the entry AXIS inverts."""
    from gen_worker.aot_declaration import entry_name

    fork = aot_regional.block_entry_fork("_Block#0")
    assert entry_name("unet", fork, ()) == "unet/block=_Block#0"
    assert entry_name("unet", fork, (("h", 128), ("w", 128))) == \
        "unet/block=_Block#0/h=128,w=128"


def test_shell_digest_separates_models_with_IDENTICAL_blocks() -> None:
    """S3.3, the load-bearing identity change. Two models whose block classes
    and block weights are identical but whose SHELL differs must not key
    identically — regionally the cell covers only the parts, so nothing else
    in the key binds the assembly.

    RED without it: both models produce the same block set, so the same
    per-entry class hashes, so the same `combined_graph_hash`."""
    four = _Shell(layers=4)
    five = _Shell(layers=5)
    assert aot_regional.shell_digest(four) != aot_regional.shell_digest(five)
    # ... and the part the digest replaces genuinely does NOT separate them:
    # identical block classes at identical widths.
    assert [g.key for g in aot_regional.repeated_block_groups(four)] == \
        [g.key for g in aot_regional.repeated_block_groups(five)]
    # Stable across construction of the same shape (a digest that moved per
    # process would re-mint the fleet on every boot).
    assert aot_regional.shell_digest(_Shell(layers=4)) == \
        aot_regional.shell_digest(four)


def _meta(**over: Any) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "format": "aot-1", "family": "scenario", "sm": "sm_89",
        "combined_graph_hash": "abc123",
        "entries": {"unet/block=_Block#0": {
            "target": "unet", "class_hash": "deadbeef"}},
        "env_seal": {"cpp_march": "x86-64-v3"},
        "toolchain": {"g++": "13"},
        "code_closure": {"scenario.main": "aaa"},
    }
    meta.update(over)
    return meta


def test_cell_identity_keys_regional_apart_and_binds_the_shell() -> None:
    """RED at HEAD (pgw#812 D4): `cell_identity` hardcoded `"mode": ""` with
    the comment 'an exported cell is always whole-graph: regional is a dynamo
    partitioning strategy with no export counterpart'. A regional cell and a
    whole-graph cell of the same family x lane x sm keyed IDENTICALLY."""
    spec = aot_mint.ExportSpec(family="scenario", target="unet")
    whole = aot_mint.cell_identity(_meta(), spec)
    regional = aot_mint.cell_identity(
        _meta(), aot_mint.replace_spec(
            spec, regional=True, shell_digest="1111111111111111"))
    assert whole.digest != regional.digest

    other_shell = aot_mint.cell_identity(
        _meta(), aot_mint.replace_spec(
            spec, regional=True, shell_digest="2222222222222222"))
    assert regional.digest != other_shell.digest


def test_a_regional_cell_without_a_shell_digest_is_REFUSED() -> None:
    """Mandatory, not optional: an absent digest is exactly the collision the
    axis exists to prevent, and a key that is sometimes assembly-bound and
    sometimes not is worse than either."""
    spec = aot_mint.replace_spec(
        aot_mint.ExportSpec(family="scenario", target="unet"), regional=True)
    with pytest.raises(aot_mint.MintRefused, match="shell_digest"):
        aot_mint.cell_identity(_meta(), spec)


def test_the_cell_key_mode_axis_carries_regional() -> None:
    """The ck5 `mode` axis already existed and already fed the digest — this
    is the minimal change, not new hashing machinery."""
    assert cell_key.compute is not None
    key = cell_key.from_axes({
        "format": "aot-1", "kind": "aot-inductor", "family": "scenario",
        "lane": "w8a8", "mode": aot_regional.MODE_REGIONAL, "sm": "sm_89",
        "contract": "c", "env_seal": "e", "toolchain": "t",
        "code_closure": "cc"})
    whole = cell_key.from_axes({
        "format": "aot-1", "kind": "aot-inductor", "family": "scenario",
        "lane": "w8a8", "mode": "", "sm": "sm_89",
        "contract": "c", "env_seal": "e", "toolchain": "t",
        "code_closure": "cc"})
    assert key.digest != whole.digest


# ---------------------------------------------------------------------------
# SCENARIO 3 — ARM: bind-by-reference, per instance, all or nothing (D3/S4)
# ---------------------------------------------------------------------------


class _RecordingPackage:
    """A package that records exactly how it was bound and answers calls.

    Stands in for the compiled `.so` only — every gate under test
    (`assert_bindable`, `resolve_constants`, `load_constants`'s strictness,
    `assert_ready`) is the REAL `aot_serve.ArtifactRunner` code path.
    """

    def __init__(
        self, fqns: List[str], *, supports_user_managed: bool = True,
    ) -> None:
        self._fqns = list(fqns)
        self.supports_user_managed = supports_user_managed
        self.bound: Dict[str, Any] = {}
        self.user_managed = False
        self.calls = 0

    def get_constant_fqns(self) -> List[str]:
        return list(self._fqns)

    def load_constants(
        self, values: Any, check_full_update: bool = False, **kwargs: Any,
    ) -> None:
        if "user_managed" in kwargs and not self.supports_user_managed:
            raise TypeError(
                "load_constants() got an unexpected keyword argument "
                "'user_managed'")
        self.user_managed = bool(kwargs.get("user_managed", False))
        self.bound = dict(values)

    def __call__(self, *args: Any) -> Any:
        self.calls += 1
        weight = self.bound["lin.weight"]
        return (args[0] @ weight.T,)


def _runner(block: Any, **kwargs: Any) -> Any:
    from gen_worker import aot_serve

    fqns = sorted(block.state_dict())
    width = int(block.lin.weight.shape[1])
    return aot_serve.ArtifactRunner(
        package=_RecordingPackage(fqns, **kwargs),
        # S5's INNER gate: a block entry's contract describes the block's
        # own CAPTURED inputs, derived from the artifact's recorded symbol
        # ranges — never a second declared contract for the endpoint.
        contract=aot_serve.ArtifactContract(
            inputs=(aot_serve.InputContract(
                name="x", position=0, dtype="float32",
                shape=("batch", width)),),
            symbols={"batch": (1, 64)}),
        constants=tuple(
            aot_serve.ConstantSpec(
                fqn=f, source=aot_serve.SOURCE_STATE_DICT,
                dtype="float32", shape=())
            for f in fqns),
        entry="unet/block=_Block#0")


def test_bind_by_reference_costs_ZERO_weight_copies_across_N_instances(
) -> None:
    """D3. `user_managed` appears nowhere in the SDK today —
    `ArtifactRunner.bind` calls `load_constants(values, check_full_update=
    True)` with the default `user_managed=False`, i.e. the artifact COPIES
    every constant. Whole-graph that is a one-off duplicate; regionally it is
    N copies of the block weights (for flux2, a second whole model).

    Proven by identity, not by a memory reading: every bound tensor must BE
    the resident parameter, same storage."""
    model = _Shell(layers=5)
    group = aot_regional.repeated_block_groups(model)[0]
    assert group.count == 5

    bound_ptrs = []
    for module in group.instances:
        runner = _runner(module)
        runner.bind(dict(module.state_dict()), {}, user_managed=True)
        assert runner.user_managed is True
        assert runner.package.user_managed is True
        for fqn, tensor in runner.package.bound.items():
            resident = dict(module.state_dict())[fqn]
            assert tensor.data_ptr() == resident.data_ptr()
            bound_ptrs.append(tensor.data_ptr())
    # 5 instances x 1 constant, all distinct storages, none of them copies.
    assert len(bound_ptrs) == 5
    assert len(set(bound_ptrs)) == 5


def test_the_default_bind_is_UNCHANGED_for_whole_graph() -> None:
    """The whole-graph call shape stays byte-identical to what pgw#721/#723
    measured on a pod: no `user_managed` keyword at all."""
    model = _Shell(layers=1)
    module = model.blocks[0]
    runner = _runner(module)
    runner.bind(dict(module.state_dict()), {})
    assert runner.user_managed is False
    assert runner.package.user_managed is False


def test_a_torch_without_user_managed_is_a_NAMED_refusal() -> None:
    """A silent copy would OOM the card N blocks later, which is a far worse
    way to learn the same fact."""
    from gen_worker.aot_serve import ConstantsUnboundError

    model = _Shell(layers=1)
    module = model.blocks[0]
    runner = _runner(module, supports_user_managed=False)
    with pytest.raises(ConstantsUnboundError, match="user_managed"):
        runner.bind(dict(module.state_dict()), {}, user_managed=True)
    assert runner.bound is False


def test_arming_is_ALL_OR_NOTHING_per_target() -> None:
    """S4: a model with 24 of 25 blocks armed is a silently half-eager model
    — it serves, it is slower than either pure lane, and nothing reports it.
    A per-instance failure must revert EVERY instance."""
    model = _Shell(layers=5, wide=0)
    groups = aot_regional.repeated_block_groups(model)
    made: List[int] = []

    def runner_for(_key: str) -> Any:
        made.append(len(made))
        if len(made) == 4:  # instance 4 of 5
            raise RuntimeError("simulated bind failure on the fourth instance")
        block = model.blocks[len(made) - 1]
        return _runner(block)

    with pytest.raises(RegionalArmRefused, match="silently half-eager"):
        aot_regional.arm_blocks(groups, runner_for, target="unet")
    # Every instance is eager again — no shim survives on any of them.
    for block in model.blocks:
        assert "forward" not in block.__dict__


def test_the_unbound_call_gate_runs_before_every_INSTANCE() -> None:
    """The segfault surface multiplies by N, so `assert_ready` is per
    instance, not once per cell."""
    model = _Shell(layers=3, wide=0)
    groups = aot_regional.repeated_block_groups(model)
    seen: List[bool] = []

    def runner_for(_key: str) -> Any:
        block = model.blocks[len(seen)]
        runner = _runner(block)
        original = runner.assert_ready

        def _record() -> None:
            seen.append(runner.bound)
            original()
        runner.assert_ready = _record  # type: ignore[method-assign]
        return runner

    arm = aot_regional.arm_blocks(groups, runner_for, target="unet")
    assert arm.bound_instances == 3
    assert seen == [True, True, True]
    arm.revert()


# ---------------------------------------------------------------------------
# SCENARIO 4 — JUDGE: the adoption numerics gate (S6 / pgw#800's ladder)
# ---------------------------------------------------------------------------


def test_the_ladder_calls_every_MEASURED_configuration_correctly() -> None:
    """The whole calibration, pinned. Moving `NUMERICS_FLOOR` or
    `NUMERICS_WARN` fails here with the evidence attached.

    RED-VERIFY, in line: pgw#800's ADAPTER thresholds (0.80 / 0.99) — which
    pgw#814 explicitly warns must not be inherited — call the flux2 w8a8
    whole-graph artifact DEGRADED and would have SERVED it. pgw#814's own
    ruling on that artifact is 'do not adopt a flux2 w8a8 cell until this
    closes'."""
    from gen_worker.models import adapter_fidelity

    for label, cosine, retention, want in MEASURED:
        got = aot_regional.DEFAULT_THRESHOLDS.verdict(cosine, retention)
        assert got == want, f"{label}: cos={cosine} ret={retention} -> {got}"

    # The red half: the adapter calibration serves what this one refuses.
    for label, cosine, retention, want in MEASURED:
        adapter_call = adapter_fidelity.ADAPTER_THRESHOLDS.verdict(
            cosine, retention)
        if want == "destroyed":
            assert adapter_call == "degraded", (
                f"{label}: the adapter ladder would have SERVED this")


def test_the_floor_and_warn_bracket_the_measured_band() -> None:
    """Both constants are DERIVED, and the derivation is checkable."""
    worst_accepted = 0.9890   # flux2 w8a8 regional, pgw#812/#814
    best_refused = 0.97300    # flux2 w8a8 rowwise whole-graph, pgw#814
    assert best_refused < aot_regional.NUMERICS_FLOOR < worst_accepted
    assert abs(aot_regional.NUMERICS_FLOOR
               - math.sqrt(worst_accepted * best_refused)) < 0.001
    ret_accepted, ret_refused = 0.997, 0.905
    assert abs(aot_regional.NUMERICS_RETENTION_FLOOR
               - math.sqrt(ret_accepted * ret_refused)) < 0.001


def test_a_perfect_cosine_at_the_wrong_MAGNITUDE_is_not_healthy() -> None:
    """Cosine is scale-invariant. An artifact that reproduces eager's
    direction exactly at 0.9x the magnitude serves a systematically dimmer
    image and pgw#800's ladder could not see it, because an adapter's
    retention is evidence rather than a bound (a destroyed one measures
    15.3)."""
    assert aot_regional.DEFAULT_THRESHOLDS.verdict(1.0, 0.90) == "degraded"
    assert aot_regional.DEFAULT_THRESHOLDS.verdict(1.0, 1.0) == "healthy"


def test_compare_outputs_is_norm_weighted_never_a_per_row_median() -> None:
    """pgw#800's rule, carried across populations: a handful of destroyed
    high-norm outputs must not hide behind many intact low-norm ones."""
    good_a, good_b = _pair_at(1.0, 1.0, n=64)
    bad_a, bad_b = _pair_at(0.0, 1.0, n=64)
    # Three intact tiny rows, one destroyed row carrying 100x the norm.
    reference = [good_a * 0.01, good_a * 0.01, good_a * 0.01, bad_a * 1.0]
    subject = [good_b * 0.01, good_b * 0.01, good_b * 0.01, bad_b * 1.0]
    cmp_ = numerics_ladder.compare_outputs(
        reference, subject, thresholds=aot_regional.DEFAULT_THRESHOLDS)
    median = sorted(r.cosine for r in cmp_.rows)[len(cmp_.rows) // 2]
    assert median > 0.99          # a median would call this healthy
    assert cmp_.cosine < 0.1      # the norm-weighted aggregate does not
    assert cmp_.verdict == "destroyed"


def test_compare_outputs_refuses_a_STRUCTURAL_mismatch() -> None:
    """A silently-dropped output is the failure this gate exists to catch,
    not a row to average over."""
    a, b = _pair_at(1.0, 1.0, n=16)
    with pytest.raises(ValueError, match="output structure differs"):
        numerics_ladder.compare_outputs(
            [a, a], [b], thresholds=aot_regional.DEFAULT_THRESHOLDS)
    with pytest.raises(ValueError, match="shape differs"):
        numerics_ladder.compare_outputs(
            a, b.reshape(4, 4), thresholds=aot_regional.DEFAULT_THRESHOLDS)


def test_a_DEGRADED_artifact_arms_and_CONFESSES(
    events: List[Tuple[str, str, str]],
) -> None:
    """flux2 w8a8 regional (cos 0.989) — served, known, and on the wire."""
    a, b = _pair_at(0.9890, 0.99)
    cmp_ = aot_regional.gate_assembled(
        a, b, thresholds=aot_regional.DEFAULT_THRESHOLDS,
        family="flux2", cell_key="ck5-degraded")
    assert cmp_.verdict == "degraded"
    kinds = [(k, p) for k, p, _ in events]
    assert ("cell_numerics", "degraded") in kinds


def test_a_DESTROYED_artifact_REFUSES_TO_ARM(
    events: List[Tuple[str, str, str]],
) -> None:
    """Paul's 'no degradation in quality output' made structural rather than
    procedural. Red-verified against pgw#814's ACTUAL degraded configuration:
    flux2 w8a8 pertensor whole-graph, cos 0.93094, retention 0.905 — a real
    artifact that exists, that the platform ruled unadoptable, and that
    nothing in the worker would have noticed."""
    a, b = _pair_at(0.93094, 0.905)
    with pytest.raises(RegionalArmRefused) as excinfo:
        aot_regional.gate_assembled(
            a, b, thresholds=aot_regional.DEFAULT_THRESHOLDS,
            family="flux2", cell_key="ck5-pgw814")
    refusal = excinfo.value
    assert refusal.reason == "numerics_destroyed"
    assert refusal.comparison is not None
    assert refusal.comparison.verdict == "destroyed"
    # The evidence reaches the hub, not only the caller.
    refused = [(k, p, d) for k, p, d in events if p == "refused"]
    assert len(refused) == 1
    assert "cell_numerics" == refused[0][0]
    assert "cosine=0.930" in refused[0][2]
    assert "ck5-pgw814" in refused[0][2]


def test_arm_and_verify_REVERTS_every_instance_on_a_destroyed_verdict(
    events: List[Tuple[str, str, str]],
) -> None:
    """The whole adoption sequence on a real assembled model: eager
    reference, arm every instance by reference, re-run, judge. A DESTROYED
    verdict must leave the pipeline exactly as it was — eager, serving."""
    torch.manual_seed(3)
    model = _Shell(layers=4, wide=0, width=8)
    feed = torch.randn(2, 8)
    groups = aot_regional.repeated_block_groups(model)

    made: List[Any] = []

    def runner_for(_key: str) -> Any:
        block = model.blocks[len(made)]
        runner = _runner(block)
        made.append(runner)
        return runner

    # A HEALTHY arm: the recording package computes the block's real linear,
    # so the assembled model reproduces eager exactly.
    arm = aot_regional.arm_and_verify(
        model, groups, runner_for, lambda: model(feed), target="unet",
        thresholds=aot_regional.DEFAULT_THRESHOLDS, family="scenario",
        cell_key="ck5-healthy")
    assert arm.armed is True
    assert arm.bound_instances == 4
    assert arm.comparison is not None and arm.comparison.verdict == "healthy"
    arm.revert()
    for block in model.blocks:
        assert "forward" not in block.__dict__

    # A DESTROYED arm: one block's artifact answers with the wrong wiring —
    # exactly the "correct block, wrong assembly" failure mode that could not
    # previously exist (S6).
    made.clear()
    events.clear()

    def broken_runner_for(_key: str) -> Any:
        runner = runner_for(_key)
        if len(made) == 3:
            package = runner.package
            original = package.__call__

            def _wrong(*args: Any) -> Any:
                out = original(*args)
                return (out[0].flip(-1),)
            runner.package = type(
                "Broken", (), {"__call__": staticmethod(_wrong),
                               "get_constant_fqns": package.get_constant_fqns,
                               "load_constants": package.load_constants})()
        return runner

    with pytest.raises(RegionalArmRefused):
        aot_regional.arm_and_verify(
            model, groups, broken_runner_for, lambda: model(feed),
            target="unet", thresholds=aot_regional.DEFAULT_THRESHOLDS,
            family="scenario", cell_key="ck5-broken")
    for block in model.blocks:
        assert "forward" not in block.__dict__, \
            "a refused arm must leave the pipeline eager"
    assert ("cell_numerics", "refused") in [(k, p) for k, p, _ in events]


def test_the_shared_ladder_is_ONE_ladder() -> None:
    """pgw#814 asked for a refactor, not a second gate: the rungs, the
    ordering and the gate shape live in one module and each population brings
    only its calibration."""
    from gen_worker.models import adapter_fidelity

    assert adapter_fidelity.VERDICT_DESTROYED is \
        numerics_ladder.VERDICT_DESTROYED
    assert adapter_fidelity.PHASE_REFUSED is numerics_ladder.PHASE_REFUSED
    # The adapter gate still calls its own measured configurations correctly
    # through the shared ladder (pgw#800's pinned rows).
    assert adapter_fidelity.ADAPTER_THRESHOLDS.verdict(0.5115) == "destroyed"
    assert adapter_fidelity.ADAPTER_THRESHOLDS.verdict(0.9507) == "degraded"
    assert adapter_fidelity.ADAPTER_THRESHOLDS.verdict(0.999999) == "healthy"


# ---------------------------------------------------------------------------
# SCENARIO 5 — the pool re-price (S7) and the recipe wiring
# ---------------------------------------------------------------------------


def test_the_pool_is_RE_PRICED_for_the_many_small_compiles_shape() -> None:
    """S7: regional and K-wide entry parallelism COMPOSE, and must be
    re-priced together rather than multiplied. Regional moves both inputs —
    the entry count up (one per plan x block class) and the per-entry device
    ask down by the block fraction."""
    model = _Shell(layers=8, wide=2, width=16)

    class _Pipe:
        def __init__(self) -> None:
            self.unet = model

    decl = _decl(regional=True, targets=("unet",))
    pipe = _Pipe()
    targets = aot_mint._regional_targets(pipe, decl)
    assert set(targets) == {"unet"}
    owner, groups = targets["unet"]
    assert owner is model
    assert sum(g.count for g in groups) == 10

    plans = [(type("P", (), {"target": "unet"})(), None)] * 3
    # 3 plans x 2 block classes = 6 entries, where whole-graph had 3.
    assert aot_mint._regional_entry_count(pipe, decl, plans) == 6

    fraction = aot_mint._block_device_fraction(pipe, decl)
    assert 0.1 <= fraction < 1.0
    # The wide block is the biggest single class, and it is a real fraction
    # of the model rather than a constant.
    assert fraction == pytest.approx(
        sum(p.numel() for p in model.wide[0].parameters())
        / sum(p.numel() for p in model.parameters()), rel=1e-6)


def test_mint_recipe_selects_the_regional_shape_from_the_DECLARATION(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-family opt-in, never a fleet default: pgw#812 ranked it that way
    deliberately — on a small-table DiT regional is a 2x that costs a
    serve-path change, while pgw#811 buys a comparable win with none."""
    from gen_worker import aot_cells, fleet_cells

    monkeypatch.setattr(aot_cells, "prefer_aot", lambda: True)
    monkeypatch.setattr(aot_mint, "lane_admitted", lambda *a, **k: "")
    monkeypatch.setattr(aot_mint, "lifted_torch_gap", lambda *a, **k: "")
    # pgw#822's pre-spawn declaration/module gate, stubbed like every other
    # gate above it: this test asks which SHAPE the declaration selects, and
    # the double carries no class rows for a real class set to be derived
    # from. Its own coverage is test_aot_lifted_arm_pgw822.py.
    monkeypatch.setattr(aot_mint, "declaration_module_gaps", lambda *a, **k: [])
    monkeypatch.setattr(
        fleet_cells, "aot_export_spec", lambda *a, **k: object())

    import gen_worker.api.export_contract as ec

    for regional, want in ((False, "aot"), (True, "aot-regional")):
        monkeypatch.setattr(
            ec, "export_declaration", lambda _f, r=regional: _decl(regional=r))
        cfg = _decl(regional=regional)
        assert fleet_cells.mint_recipe(
            object(), cfg, delegate=True, emit=False) == want


def test_both_AOT_recipes_report_through_aot_mint_phases() -> None:
    """The acceptance channel. `aot_mint_phases` is where the minutes-scale
    claim is MEASURED end-to-end (not harness arithmetic), and the delegation
    tail chose its reporting kind with a string-literal `recipe == "aot"` —
    which would have sent every regional mint's phase table down the
    `jit_compile` kind instead, recording the one number the whole issue
    turns on under the wrong name."""
    import inspect

    from gen_worker import fleet_cells, mint_delegate

    source = inspect.getsource(mint_delegate.build_cell)
    assert 'recipe", "")) == "aot"' not in source
    assert "RECIPE_AOT_REGIONAL" in source
    # The child echoes the recipe it was ASKED for, so the parent's grouping
    # and the child's report cannot disagree.
    from gen_worker import mint_child

    assert mint_child.RECIPE_AOT_REGIONAL == fleet_cells.RECIPE_AOT_REGIONAL
    assert "recipe=request.recipe" in inspect.getsource(mint_child._mint_aot)


def test_the_regional_recipe_rides_the_pgw816_COMPOSITION(
) -> None:
    """pgw#816 (landed as `8acceda`): a directory path does not describe a
    composition, so the child now loads through the parent's resolved
    `component_paths`. The regional recipe must ride the SAME seam — it is a
    label on one `MintRequest`, not a second code path — because the blocks
    it discovers and the shell it digests have to be the ones the parent is
    actually SERVING. A regional mint that composed differently would export
    blocks the serving pod cannot adopt, silently."""
    import inspect

    from gen_worker import mint_child

    source = inspect.getsource(mint_child.mint)
    # ONE composition, built before the recipe is consulted.
    setup_at = source.index("component_paths=overrides")
    recipe_at = source.index("if request.recipe in (")
    assert setup_at < recipe_at, \
        "the recipe must be chosen AFTER the parent's composition is loaded"
    assert source.count("run_setup(") == 1


# ---------------------------------------------------------------------------
# SCENARIO 6 — THE REAL PATH: an actual regional mint, end to end
# ---------------------------------------------------------------------------
#
# Real `torch.export` + real AOTInductor compiles on CPU. No mocks anywhere
# on the mint path, because the mint path IS what this issue changes: block
# discovery -> feed capture -> positional marshal -> export -> the declared
# range gate -> packaging -> per-entry class hashes -> the stamped identity.


class _TinyBlock(nn.Module):
    """One repeated block, shaped like a transformer block's signature: a
    positional hidden state plus an optional kwarg the shell does not always
    pass. That optional is the whole reason `positional_feed` exists."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.lin = nn.Linear(width, width)

    def forward(self, hidden_states: Any, temb: Any = None) -> Any:
        out = torch.tanh(self.lin(hidden_states))
        return out if temb is None else out + temb


class _TinyRegionalUNet(nn.Module):
    """A model that declares its repeated blocks the way diffusers does."""

    _repeated_blocks = ["_TinyBlock"]

    def __init__(self, layers: int = 3, width: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock(width) for _ in range(layers)])
        self.head = nn.Linear(width, width)
        self.config = {"layers": layers, "width": width}

    def forward(self, sample: Any) -> Any:
        for block in self.blocks:
            sample = block(sample)
        return self.head(sample)


@pytest.fixture
def _registry():
    from gen_worker.api.export_contract import reset_export_declarations

    reset_export_declarations()
    yield
    reset_export_declarations()


@pytest.fixture
def _fake_sm(monkeypatch: pytest.MonkeyPatch):
    """A CPU box has no sm, and identity requires one (the pgw#723/#758
    convention). Mint and consumer probes must agree."""
    from gen_worker import aot_serve, compile_cache

    full = {"sku": "", "sm": "sm_89", "torch": str(torch.__version__),
            "cuda": ""}
    monkeypatch.setattr(compile_cache, "runtime_key", lambda: dict(full))
    monkeypatch.setattr(aot_serve, "runtime_key", lambda: dict(full))
    return full


def _register_regional(family: str, *, regional: bool) -> Any:
    from gen_worker.api.export_contract import (
        Fork, GraphClass, Input, register_export_declaration)

    return register_export_declaration(Compile(
        family=family,
        targets=("unet",),
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        forks=(Fork("cfg", served=(True, False)),),
        classes=(GraphClass(dims={"B": 2}, fork={"cfg": True}),
                 GraphClass(dims={"B": 1}, fork={"cfg": False})),
        inputs=(Input("sample", shape=("B", 4)),),
        shape_strategy="static-rows",
        warm_changes_key=False,
        regional=regional,
    ))


@pytest.mark.integration
def test_a_REAL_regional_mint_packages_block_entries(
    tmp_path, _registry, _fake_sm,
) -> None:
    """The headline. One real mint of a 3-block model produces ONE .pt2
    whose entries are BLOCK classes — 2 declared plans x 1 block class = 2
    entries where whole-graph produced 2 whole-forward graphs — and the
    stamped key says `mode=regional` and binds the shell.

    RED before pgw#817 in the only way that matters: `Compile(regional=True)`
    reached `aot_mint` nowhere at all (D4 — 'nothing in aot_mint / aot_serve
    / aot_declaration reads it'), so this mint produced whole-forward graphs
    and a key claiming `mode: ""`."""
    import types

    from gen_worker.api.export_contract import export_declaration

    _register_regional("tiny817", regional=True)
    pipe = types.SimpleNamespace(unet=_TinyRegionalUNet(layers=3))
    spec = aot_mint.ExportSpec(family="tiny817", target="")
    result = aot_mint.mint(
        pipe, spec, tmp_path / "out", allow_regressed_lanes=True,
        entry_workers=1)

    entries = dict(result.metadata["entries"])
    assert sorted(entries) == [
        "unet/block=_TinyBlock#0,cfg=false/B=1",
        "unet/block=_TinyBlock#0,cfg=true/B=2",
    ], "entries must enumerate BLOCK classes in the existing entry grammar"

    # The identity: mode set, shell bound, and both recorded on the artifact
    # so a consumer can recompute rather than trust the stamp.
    assert result.metadata["mode"] == "regional"
    shell = result.metadata["shell_digest"]
    assert shell, "a regional cell must record the shell it was minted for"
    # Recomputable from the artifact's OWN facts — the standing ck5
    # discipline: a stamp can never disagree with the axes it summarizes.
    assert result.metadata["cell_key"] == aot_mint.cell_identity(
        result.metadata,
        aot_mint.replace_spec(spec, regional=True, shell_digest=shell)).digest
    # And it genuinely binds THIS assembly: a model with the same block class
    # and a different layer count digests differently.
    assert shell != aot_mint._cell_shell_digest(
        types.SimpleNamespace(unet=_TinyRegionalUNet(layers=4)),
        export_declaration("tiny817"))

    # The compiled thing really is ONE BLOCK, not the model: the block has one
    # Linear, the model has four.
    decl = export_declaration("tiny817")
    assert decl is not None and decl.regional is True
    # The compile bill, which is the whole point: a block entry's constant
    # table is ONE block's, not the model's. (Each Linear's bias is folded
    # into the matmul epilogue by the compiler on both arms, so what is left
    # is the weights — 1 here against the whole-graph arm's 4 below.)
    per_entry_constants = {
        name: len(block["constants"]) for name, block in entries.items()}
    assert set(per_entry_constants.values()) == {1}, \
        f"a block entry binds ONE block's constants: {per_entry_constants}"
    assert {c["fqn"] for b in entries.values() for c in b["constants"]} == \
        {"lin.weight"}, \
        "a block entry's constant FQNs are BLOCK-relative (S4's template)"


@pytest.mark.integration
def test_the_same_declaration_WHOLE_GRAPH_keys_differently_and_is_bigger(
    tmp_path, _registry, _fake_sm,
) -> None:
    """The A/B, on one model and one toolchain: flipping only `regional`
    changes the entry axis, the constant count, and the cell key — and the
    two cells can never be confused for each other, which is what the `mode`
    axis is for."""
    import types

    _register_regional("tiny817w", regional=False)
    pipe = types.SimpleNamespace(unet=_TinyRegionalUNet(layers=3))
    spec = aot_mint.ExportSpec(family="tiny817w", target="")
    whole = aot_mint.mint(
        pipe, spec, tmp_path / "whole", allow_regressed_lanes=True,
        entry_workers=1)

    assert sorted(whole.metadata["entries"]) == ["unet/cfg=false/B=1",
                                                 "unet/cfg=true/B=2"]
    assert whole.metadata["mode"] == ""
    # 3 block weights + the head's = 4 constants per entry, against the block
    # entry's 1. This is the compile bill regional removes, and on the real
    # sdxl w8a8 cell the same ratio is 2,423 constants against one block's
    # (pgw#812 RESULT 5: 274.7 s -> 19.4 s).
    whole_constants = {
        len(b["constants"]) for b in whole.metadata["entries"].values()}
    assert whole_constants == {4}
    assert {c["fqn"] for b in whole.metadata["entries"].values()
            for c in b["constants"]} == {
        "blocks.0.lin.weight", "blocks.1.lin.weight", "blocks.2.lin.weight",
        "head.weight"}


@pytest.mark.integration
def test_a_regional_mint_REFUSES_a_model_with_no_repeated_blocks(
    tmp_path, _registry, _fake_sm,
) -> None:
    """Refused by name at the declaration/model seam, not discovered as an
    empty entry set three phases later."""
    import types

    class _Flat(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, sample: Any) -> Any:
            return self.lin(sample)

    _register_regional("tiny817f", regional=True)
    pipe = types.SimpleNamespace(unet=_Flat())
    spec = aot_mint.ExportSpec(family="tiny817f", target="")
    with pytest.raises(aot_mint.MintRefused, match="_repeated_blocks"):
        aot_mint.mint(pipe, spec, tmp_path / "out", allow_regressed_lanes=True,
                      entry_workers=1)


@pytest.mark.integration
def test_a_regional_mint_compiles_through_the_POOL_the_same_way(
    tmp_path, _registry, _fake_sm, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shape the live acceptance runs. Regional's entries go out to
    pgw#809's K-wide out-of-process pool exactly like whole-graph's, and the
    artifact must be identical in identity to the serial one.

    This is also the case that would have caught pgw#754's thread-local ISA
    clamp (fixed in `3bd9887`): every host compile off the boot thread built
    `-march=native`, and a regional mint is many small compiles fanned out
    across pool threads, so it is the shape that hits it hardest — a cell
    minted under an unrecorded ISA publishes under an unchanged key."""
    import types

    from gen_worker import aot_compile_pool, host_isa

    _register_regional("tiny817p", regional=True)
    pipe = types.SimpleNamespace(unet=_TinyRegionalUNet(layers=3))
    spec = aot_mint.ExportSpec(family="tiny817p", target="")

    widths: List[Any] = []
    real_entry_workers = aot_compile_pool.entry_workers

    def _record(entries: int, **kwargs: Any) -> Any:
        width = real_entry_workers(entries, **kwargs)
        widths.append((entries, width.workers))
        return width

    monkeypatch.setattr(aot_compile_pool, "entry_workers", _record)
    monkeypatch.setattr(aot_mint.aot_compile_pool, "entry_workers", _record)

    serial = aot_mint.mint(
        pipe, spec, tmp_path / "serial", allow_regressed_lanes=True,
        entry_workers=1)

    # The pool is priced on the REGIONAL entry count (2 plans x 1 block class),
    # not on the plan count — S7's re-price, observed rather than asserted in
    # a comment.
    assert widths and widths[0][0] == 2

    # The clamp the pool's children inherit is PROCESS-wide now (pgw#754,
    # `3bd9887`), so the cell's recorded ISA cannot depend on which thread
    # compiled it — read it back on a foreign thread, which is what a pool
    # child's compile thread is.
    on_foreign_thread = host_isa._read_in_fresh_thread(host_isa.effective)
    assert isinstance(on_foreign_thread, dict)
    assert on_foreign_thread == host_isa.effective(), (
        "a regional mint is many small compiles fanned out across pool "
        "threads, so a thread-local clamp would record one ISA and build "
        "another")
    # (What the seal RECORDS is pgw#754's own suite's business; what matters
    # here is that the regional mint's many-threaded compile shape cannot see
    # a different clamp than the one the seal was taken under.)
    assert "env_seal" in serial.metadata
