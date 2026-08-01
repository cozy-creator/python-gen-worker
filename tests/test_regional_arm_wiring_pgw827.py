"""pgw#827 — a REGIONAL cell is adopted through the REGIONAL arm.

Measured on a real L4 (gen-worker 0.85.0, release ``4f32ea53dd81adc1d8501e14``,
sdxl 0.2.105, lane ``w8a8-lora64``, recipe ``aot-regional``, pod
``o7y87kfunc3rmm``): **the first successful AOT mint in platform history** —
``aot_mint_phases phase=minted n_entries=72 total_s=354.45`` — and then, one
stage later, at the mint's own self-adopt verification:

    aot_adopt  constants_constant_unresolved
      entry 'unet/adapter=false,block=BasicTransformerBlock#0,cfg=false/
      B=1,H_lat=104,T_txt=77,W_lat=152': 30 declared constant(s) have no
      value: ['attn1.to_k.weight (source=state_dict)', ...]
    self_mint_abort   delegated_adopt_failed
    self_mint_compile seal_publish state=failed

``models.provision.arm_aot`` DETECTED the regional cell (pgw#825 added that,
to skip the lifted install) and then handed it to ``aot_serve.enable``
anyway — the WHOLE-GRAPH arm, which builds ONE bind table per TARGET from
``resident_constants(unet)``. A regional entry's declared FQNs are
block-relative (``attn1.to_k.weight``), the denoiser carries them under their
full path (``down_blocks.…transformer_blocks.0.attn1.to_k.weight``), and
``resolve_constants`` is a direct FQN lookup by design — so none of them
resolve, for any entry. ``aot_regional.arm_blocks`` builds the RIGHT table
(``resident_constants(block)``, per instance) and had no caller.

Because the mint's self-adopt verification runs the same ``arm_aot``, an
unwired regional arm did not mean "regional cells cannot serve". It meant
**regional cells cannot be PUBLISHED** — a cell the minting runtime cannot
adopt is refused at ``seal_publish``.

Every test here runs a REAL regional mint (torch.export + AOTI on CPU) and a
REAL adopt. RED at base: :func:`test_a_regional_cell_ADOPTS` reproduces the
pod's sentence verbatim (``constants_constant_unresolved``, the block's own
weight FQNs unresolved) and every assertion below it is unreachable.
"""

from __future__ import annotations

import copy
import types
from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker import (  # noqa: E402
    aot_mint,
    aot_regional,
    aot_serve,
    compile_cache,
    fleet_cells,
)
from gen_worker.api.decorators import Compile  # noqa: E402
from gen_worker.api.export_contract import (  # noqa: E402
    Dim,
    GraphClass,
    Input,
    register_export_declaration,
    reset_export_declarations,
)
from gen_worker.models import provision  # noqa: E402

FAMILY = "tiny827"
WIDTH = 8
DEPTH = 3


class TinyBlock(nn.Module):
    """A repeated block whose constants are BLOCK-RELATIVE — the whole point.

    ``attn1.weight`` on the block is ``blocks.0.attn1.weight`` on the shell,
    which is why a bind table built at shell scope resolves none of them.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attn1 = nn.Linear(WIDTH, WIDTH, bias=False)
        self.ff = nn.Linear(WIDTH, WIDTH, bias=False)

    def forward(self, hidden: Any) -> Any:
        return self.ff(torch.tanh(self.attn1(hidden)))


class TinyUNet(nn.Module):
    _repeated_blocks = ("TinyBlock",)

    def __init__(self, depth: int = DEPTH) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock() for _ in range(depth)])

    def forward(self, sample: Any) -> Any:
        for block in self.blocks:
            sample = block(sample)
        return sample


def _declare() -> Any:
    reset_export_declarations()
    return register_export_declaration(Compile(
        family=FAMILY,
        targets=("unet",),
        regional=True,
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        classes=(GraphClass(dims={"B": 2}),),
        inputs=(Input("sample", shape=("B", WIDTH)),),
        shape_strategy="static-rows",
        warm_changes_key=False,
    ))


def _pipe() -> Any:
    return types.SimpleNamespace(unet=TinyUNet().eval())


def _cfg() -> Any:
    return types.SimpleNamespace(family=FAMILY, lora_bucket=0)


def _fake_sm(mp) -> None:
    full = {"sku": "", "sm": "sm_89", "torch": str(torch.__version__), "cuda": ""}
    mp.setattr(compile_cache, "runtime_key", lambda: dict(full))
    mp.setattr(aot_serve, "runtime_key", lambda: dict(full))


@pytest.fixture(autouse=True)
def _fresh_registry():
    reset_export_declarations()
    yield
    reset_export_declarations()


@pytest.fixture(scope="module")
def cell(tmp_path_factory, request) -> Dict[str, Any]:
    """ONE real regional mint: torch.export + a real AOTI pack, on CPU."""
    from _pytest.monkeypatch import MonkeyPatch

    from gen_worker import host_isa

    # The boot precondition every real host compile has (conftest's
    # `_boot_isa_clamp` is function-scoped and this fixture is not).
    try:
        host_isa.impose()
    except Exception:  # pragma: no cover — torchless/non-x86 runner
        pass
    mp = MonkeyPatch()
    request.addfinalizer(mp.undo)
    _fake_sm(mp)
    _declare()
    tmp = tmp_path_factory.mktemp("cell827")
    pipe = _pipe()
    result = aot_mint.mint(
        pipe, aot_mint.ExportSpec(family=FAMILY, target=""), tmp / "out",
        allow_regressed_lanes=True)
    reset_export_declarations()
    return {"result": result, "tmp": tmp, "pipe": pipe}


def _minted_pipe(cell) -> Any:
    """A private copy of the pipeline this cell was minted FROM."""
    return copy.deepcopy(cell["pipe"])


# ---------------------------------------------------------------------------
# The artifact really is regional, and its FQNs really are block-relative
# ---------------------------------------------------------------------------


def test_the_cell_declares_regional_mode_and_block_relative_constants(cell) -> None:
    meta = cell["result"].metadata
    assert meta["mode"] == aot_regional.MODE_REGIONAL
    entries = meta["entries"]
    assert entries
    for name, block in entries.items():
        assert aot_regional.entry_block_key(name, block), name
        declared = [row["fqn"] for row in block["constants"]
                    if row["source"] == aot_serve.SOURCE_STATE_DICT]
        assert declared, name
        # Block-relative, every one of them: no `blocks.<i>.` prefix.
        assert all(not fqn.startswith("blocks.") for fqn in declared), declared


def test_the_whole_graph_table_resolves_NONE_of_them(cell) -> None:
    """The pod's sentence, derived rather than quoted: the denoiser-scope
    table — what ``aot_serve.load_and_wrap`` builds — cannot resolve a single
    block-relative FQN, so all 30 (here: all of them) go missing."""
    pipe = _pipe()
    block_meta = next(iter(cell["result"].metadata["entries"].values()))
    specs = aot_serve.constants_from_meta(block_meta)
    denoiser_table = aot_serve.resident_constants(pipe.unet)
    with pytest.raises(aot_serve.ConstantsUnboundError) as err:
        aot_serve.resolve_constants(specs, denoiser_table, {})
    assert err.value.reason == "constant_unresolved"
    # …and the per-INSTANCE table resolves every one.
    per_instance = aot_serve.resident_constants(pipe.unet.blocks[0])
    assert aot_serve.resolve_constants(specs, per_instance, {})


# ---------------------------------------------------------------------------
# THE RED TEST — the whole reason nothing has ever been published
# ---------------------------------------------------------------------------


def test_a_regional_cell_ADOPTS(cell, monkeypatch, tmp_path) -> None:
    """RED at base with the pod's own reason, ``constants_constant_unresolved``.

    This is the mint's SELF-ADOPT verification, which is why it is a publish
    blocker and not a serving one: ``fleet_cells.adopt_delegated_mint`` calls
    exactly this ``provision.arm_aot``.
    """
    _fake_sm(monkeypatch)
    pipe = _minted_pipe(cell)
    events: List[Dict[str, Any]] = []
    from gen_worker import activity as activity_mod

    monkeypatch.setattr(
        activity_mod, "emit_event",
        lambda kind, detail, **kw: events.append(
            {"kind": kind, "detail": detail, **kw}))

    armed = provision.arm_aot(
        pipe, _cfg(), tmp_path, cell["result"].artifact, 0,
        cell["result"].metadata)

    adopt = [e for e in events if e["kind"] == aot_serve.ADOPT_EVENT]
    assert armed, (
        "the regional cell did not adopt: "
        + "; ".join(f"{e['phase']}: {e['detail']}" for e in adopt))
    assert adopt and adopt[-1]["phase"] == "armed"
    assert f"mode={aot_regional.MODE_REGIONAL}" in adopt[-1]["detail"]


def test_the_armed_cell_SERVES_and_reproduces_eager(cell, monkeypatch, tmp_path) -> None:
    """EVERY instance, not just the prototype the entry was exported from —
    the three blocks carry different weights, so a single artifact serving
    all three can only reproduce eager if each instance bound its OWN."""
    _fake_sm(monkeypatch)
    pipe = _minted_pipe(cell)
    sample = torch.randn(2, WIDTH)
    with torch.no_grad():
        eager = pipe.unet(sample).clone()

    assert provision.arm_aot(
        pipe, _cfg(), tmp_path, cell["result"].artifact, 0,
        cell["result"].metadata)
    before = aot_serve.execution_count(pipe)
    with torch.no_grad():
        served = pipe.unet(sample)
    # Every instance served through the artifact, not just one.
    assert aot_serve.execution_count(pipe) == before + DEPTH
    assert aot_serve.is_armed(pipe)
    assert aot_serve.proven_since(pipe, before)
    assert torch.allclose(served, eager, atol=1e-5), (served - eager).abs().max()


def test_the_arm_binds_BY_REFERENCE_on_every_instance(cell, monkeypatch, tmp_path) -> None:
    """pgw#812 D3: N instances must not mean N copies of the block's weights.

    Proven by mutation rather than by a flag: writing through the RESIDENT
    weight changes what the artifact computes, which is only true of a
    by-reference bind.
    """
    _fake_sm(monkeypatch)
    pipe = _minted_pipe(cell)
    assert provision.arm_aot(
        pipe, _cfg(), tmp_path, cell["result"].artifact, 0,
        cell["result"].metadata)
    sample = torch.randn(2, WIDTH)
    with torch.no_grad():
        first = pipe.unet(sample).clone()
        pipe.unet.blocks[0].attn1.weight.mul_(0.5)
        second = pipe.unet(sample).clone()
    assert not torch.allclose(first, second), (
        "mutating the resident weight did not change the artifact's output — "
        "the constants were bound by COPY")


def test_the_marker_is_the_SAME_shape_the_whole_graph_arm_publishes(
    cell, monkeypatch, tmp_path,
) -> None:
    """Without this the executor's adoption PROOF (pgw#735: it executed, and
    it is still armed) can never pass for a regional cell — the mint would
    publish and then be rolled back as unproven."""
    _fake_sm(monkeypatch)
    pipe = _minted_pipe(cell)
    assert provision.arm_aot(
        pipe, _cfg(), tmp_path, cell["result"].artifact, 0,
        cell["result"].metadata)
    assert aot_serve.is_armed(pipe)
    assert aot_serve.set_guard_failure_callback(pipe, lambda _d: None)
    assert aot_serve.unwrap(pipe)
    assert not aot_serve.is_armed(pipe)
    # Eager again, on every instance.
    for block in pipe.unet.blocks:
        assert not isinstance(
            block.__dict__.get("forward"), aot_regional.BlockShim)
    with torch.no_grad():
        assert pipe.unet(torch.randn(2, WIDTH)) is not None


def test_an_artifact_failure_serves_EAGER_and_disarms(cell, monkeypatch, tmp_path) -> None:
    """Parity with the whole-graph lane's fail-soft contract: ANY artifact
    problem is one eager request plus a disarmed cell, never a failed one."""
    _fake_sm(monkeypatch)
    pipe = _minted_pipe(cell)
    assert provision.arm_aot(
        pipe, _cfg(), tmp_path, cell["result"].artifact, 0,
        cell["result"].metadata)
    revoked: List[str] = []
    aot_serve.set_guard_failure_callback(pipe, revoked.append)

    shim = pipe.unet.blocks[0].__dict__["forward"]

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("simulated AOTI failure")

    monkeypatch.setattr(shim, "runner", _boom)
    with torch.no_grad():
        out = pipe.unet(torch.randn(2, WIDTH))
    assert out is not None            # served, eagerly
    assert revoked                    # and the scheduler was told
    assert not aot_serve.is_armed(pipe)


# ---------------------------------------------------------------------------
# The pre-pay half — ask before the compile, not after
# ---------------------------------------------------------------------------


def test_an_unwired_mode_DECLINES_by_name_instead_of_defaulting() -> None:
    assert provision.arm_route("") == "aot_serve.enable"
    assert provision.arm_route(aot_regional.MODE_REGIONAL) == \
        "aot_regional.enable"
    assert provision.arm_route("some-future-recipe") is None


def test_a_cell_whose_mode_has_no_arm_is_REFUSED_not_guessed(
    cell, monkeypatch, tmp_path,
) -> None:
    _fake_sm(monkeypatch)
    meta = dict(cell["result"].metadata)
    meta["mode"] = "some-future-recipe"
    assert not provision.arm_aot(
        _minted_pipe(cell), _cfg(), tmp_path, cell["result"].artifact, 0, meta)


def test_the_mint_DECLINES_regional_when_the_arm_is_unwired(monkeypatch) -> None:
    """pgw#827's second ask. Attempt nine paid 354.45 s of L4 and a complete
    72-entry artifact to learn this; it is answerable at
    ``self_mint_started``, from the arm's own dispatch table."""
    from gen_worker import aot_cells

    _declare()
    monkeypatch.setattr(aot_cells, "prefer_aot", lambda: True)
    monkeypatch.setattr(fleet_cells, "delegation_refusal", lambda *_a: "")
    # A lane the AOT ladder does not hold on dynamo, so the regional question
    # is the one this decline answers.
    monkeypatch.setattr(
        fleet_cells.loading, "pipeline_weight_lane", lambda _p: "w8a8")
    monkeypatch.setattr(fleet_cells.cc, "cxx_toolchain_present", lambda: True)
    monkeypatch.setattr(provision, "arm_route", lambda _mode: None)
    events: List[Dict[str, Any]] = []
    from gen_worker import activity as activity_mod

    monkeypatch.setattr(
        activity_mod, "emit_event",
        lambda kind, detail, **kw: events.append(
            {"kind": kind, "detail": detail, **kw}))

    recipe = fleet_cells.mint_recipe(_pipe(), _cfg(), delegate=True)
    assert recipe == fleet_cells.RECIPE_DYNAMO
    assert any(e.get("phase") == "regional_arm_unwired" for e in events), events


# ---------------------------------------------------------------------------
# All-or-nothing, per target (S4) — a partly armed model is never served
# ---------------------------------------------------------------------------


def test_a_block_class_the_CELL_does_not_cover_refuses_the_whole_arm(
    cell, monkeypatch, tmp_path,
) -> None:
    """A resident block class with no entry would arm into a silently
    half-eager model, so it refuses by name and nothing is installed."""
    _fake_sm(monkeypatch)

    class OtherBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(WIDTH, WIDTH)

        def forward(self, hidden: Any) -> Any:
            return self.lin(hidden)

    class MixedUNet(TinyUNet):
        _repeated_blocks = ("TinyBlock", "OtherBlock")

        def __init__(self) -> None:
            super().__init__()
            self.extra = OtherBlock()

    pipe = types.SimpleNamespace(unet=MixedUNet().eval())
    with pytest.raises(compile_cache.AdoptError) as err:
        aot_regional.load_and_arm(
            pipe, _cfg(), cell["result"].artifact, cache_dir=tmp_path)
    assert err.value.reason == "regional_block_undeclared"
    assert not aot_serve.is_armed(pipe)
    for block in pipe.unet.blocks:
        assert "forward" not in block.__dict__


def test_a_bind_failure_reverts_EVERY_instance(cell, monkeypatch, tmp_path) -> None:
    _fake_sm(monkeypatch)
    pipe = _minted_pipe(cell)
    real = aot_serve.ArtifactRunner.bind
    seen: List[str] = []

    def bind_third_fails(self, state_dict, literals, **kw):
        seen.append(self.entry)
        if len(seen) == 3:
            raise aot_serve.ConstantsUnboundError("constant_unresolved", "boom")
        return real(self, state_dict, literals, **kw)

    monkeypatch.setattr(aot_serve.ArtifactRunner, "bind", bind_third_fails)
    assert not aot_regional.enable(
        pipe, _cfg(), tmp_path, cell["result"].artifact)
    assert not aot_serve.is_armed(pipe)
    assert not hasattr(pipe, "_cozy_aot")
    for block in pipe.unet.blocks:
        assert "forward" not in block.__dict__


# ---------------------------------------------------------------------------
# Found by wiring the arm: a FOLDED constant bakes the prototype's weights
# into every other instance (a new refusal, and the reason it must exist)
# ---------------------------------------------------------------------------


class BiasBlock(nn.Module):
    """The same block with a bias — which the compiler folds away."""

    def __init__(self) -> None:
        super().__init__()
        self.attn1 = nn.Linear(WIDTH, WIDTH, bias=False)
        self.ff = nn.Linear(WIDTH, WIDTH, bias=True)

    def forward(self, hidden: Any) -> Any:
        return self.ff(torch.tanh(self.attn1(hidden)))


class BiasUNet(nn.Module):
    _repeated_blocks = ("BiasBlock",)

    def __init__(self, depth: int = DEPTH) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([BiasBlock() for _ in range(depth)])

    def forward(self, sample: Any) -> Any:
        for block in self.blocks:
            sample = block(sample)
        return sample


def test_a_folded_state_dict_constant_REFUSES_a_regional_mint(
    tmp_path, monkeypatch,
) -> None:
    """RED at base: the mint SUCCEEDS and publishes a cell that is correct for
    instance 0 and silently wrong for every other one.

    Measured on this box (torch 2.13.0+cu130, CPU) while wiring pgw#827's arm:
    with ``ff.bias`` folded, block 0 reproduces eager to 0.0 and block 1 is
    off by 0.53. ``eliminated_constants``' standing verdict — "routine
    compiler fusion, recorded, never fatal" — is right for a whole-graph cell,
    which is minted from the weights it serves, and wrong for a regional one,
    which is reused across instances that do NOT share weights.
    """
    _fake_sm(monkeypatch)
    reset_export_declarations()
    register_export_declaration(Compile(
        family="tiny827bias", targets=("unet",), regional=True,
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        classes=(GraphClass(dims={"B": 2}),),
        inputs=(Input("sample", shape=("B", WIDTH)),),
        shape_strategy="static-rows", warm_changes_key=False))
    pipe = types.SimpleNamespace(unet=BiasUNet().eval())
    with pytest.raises(aot_mint.MintRefused) as err:
        aot_mint.mint(
            pipe, aot_mint.ExportSpec(family="tiny827bias", target=""),
            tmp_path / "out", allow_regressed_lanes=True)
    assert "ff.bias" in str(err.value)
    assert "use_runtime_constant_folding" in str(err.value)


def test_a_folded_constant_is_still_ROUTINE_for_a_whole_graph_cell(
    tmp_path, monkeypatch,
) -> None:
    """The refusal is scoped to REGIONAL. A whole-graph cell serves the weights
    it was minted from, so folding is the observability line it has always
    been — tightening it there would refuse cells that serve correctly."""
    _fake_sm(monkeypatch)
    reset_export_declarations()
    register_export_declaration(Compile(
        family="tiny827whole", targets=("unet",), regional=False,
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        classes=(GraphClass(dims={"B": 2}),),
        inputs=(Input("sample", shape=("B", WIDTH)),),
        shape_strategy="static-rows", warm_changes_key=False))
    pipe = types.SimpleNamespace(unet=BiasUNet().eval())
    result = aot_mint.mint(
        pipe, aot_mint.ExportSpec(family="tiny827whole", target=""),
        tmp_path / "out", allow_regressed_lanes=True)
    fused = [row for block in result.metadata["entries"].values()
             for row in block["graph"]["fused_constants"]]
    assert any("bias" in name for name in fused), fused


# ---------------------------------------------------------------------------
# Also found by wiring the arm: the ADAPTER fork is not ingress-discriminable
# for a block, so pgw#790's routing rule needs its regional form
# ---------------------------------------------------------------------------


def _block_contract() -> Any:
    return aot_serve.contract_from_meta({
        "inputs": [{"name": "hidden", "position": 0, "dtype": "float32",
                    "shape": [2, WIDTH]}],
        "symbols": {},
    })


class _Pkg:
    def __init__(self, tag: float) -> None:
        self.tag = tag

    def get_constant_fqns(self) -> List[str]:
        return []

    def load_constants(self, values, check_full_update=True) -> None:
        pass

    def __call__(self, *feeds: Any) -> Any:
        return [feeds[0] + self.tag]


def _entry(name: str, tag: float, excluded: tuple = ()) -> Any:
    contract = _block_contract()
    if excluded:
        contract = aot_serve.ArtifactContract(
            inputs=contract.inputs, symbols=contract.symbols,
            excluded=excluded)
    runner = aot_serve.ArtifactRunner(
        package=_Pkg(tag), contract=contract, constants=(), entry=name)
    runner.bind({}, {})
    return aot_serve.EntryDispatch(((name, runner),))


def test_two_regional_arms_share_one_ingress_contract() -> None:
    """The premise, measured rather than argued: a block never carries the
    lifted pair (pgw#825), so the branchless class's ``excluded_inputs`` never
    fires and BOTH arms admit every block call — ``entry_ambiguous`` forever."""
    from gen_worker.models import lora_lifted

    bearing = _entry("unet/adapter=true,block=B#0/B=2", 1.0)
    branchless = _entry(
        "unet/adapter=false,block=B#0/B=2", 2.0,
        excluded=tuple(lora_lifted.LIFTED_INPUT_NAMES))
    both = aot_serve.EntryDispatch(bearing.runners + branchless.runners)
    with pytest.raises(aot_serve.IngressContractError) as err:
        both(torch.zeros(2, WIDTH))
    assert err.value.reason == "entry_ambiguous"


def test_the_regional_dispatch_routes_by_the_DENOISERS_adapter_state() -> None:
    """The discriminator regional actually has — the same one the whole-graph
    lane reads for the same decision (``lora_lifted.adapter_active``)."""
    owner = types.SimpleNamespace()
    dispatch = aot_regional.BlockDispatch(
        by_arm={
            True: _entry("unet/adapter=true,block=B#0/B=2", 1.0),
            False: _entry("unet/adapter=false,block=B#0/B=2", 2.0),
        },
        adapter_owner=owner)
    x = torch.zeros(2, WIDTH)

    from gen_worker.models import lora_lifted

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(lora_lifted, "adapter_active", lambda m: False)
        assert torch.allclose(dispatch(x)[0], x + 2.0)   # branchless
        mp.setattr(lora_lifted, "adapter_active", lambda m: True)
        assert torch.allclose(dispatch(x)[0], x + 1.0)   # branch-bearing
    assert dispatch.calls == 2
    assert sorted(dispatch.entry_calls()) == [
        "unet/adapter=false,block=B#0/B=2", "unet/adapter=true,block=B#0/B=2"]


def test_an_unforked_regional_block_needs_no_adapter_state() -> None:
    """A bucket-0 family's cell has one arm; asking for adapter state there
    would be a read with no question behind it."""
    dispatch = aot_regional.BlockDispatch(
        by_arm={None: _entry("unet/block=B#0/B=2", 3.0)}, adapter_owner=None)
    x = torch.zeros(2, WIDTH)
    assert torch.allclose(dispatch(x)[0], x + 3.0)


def test_the_real_cell_arms_through_the_regional_dispatch(
    cell, monkeypatch, tmp_path,
) -> None:
    _fake_sm(monkeypatch)
    pipe = _minted_pipe(cell)
    assert provision.arm_aot(
        pipe, _cfg(), tmp_path, cell["result"].artifact, 0,
        cell["result"].metadata)
    shim = pipe.unet.blocks[0].__dict__["forward"]
    assert isinstance(shim.runner, aot_regional.BlockDispatch)
    assert shim.runner.user_managed
