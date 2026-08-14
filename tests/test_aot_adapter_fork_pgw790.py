"""The adapter FORK: one cell, two graph classes, routed by the declared
ingress contract.

Every branch-capable leaf carries a canonical zeroed rank-bucket branch so a
curated attach is a buffer copy instead of a recompile. That branch is not free
(RTX 4090 + RTX 5090, SDXL UNet, batch-2 CFG, 1024x1024, n=28 warm):

    card    arm                base w8a8   w8a8-lora64   branch tax
    5090    AOT aoti_static    62.20 ms    90.75 ms      +45.9%
    5090    JIT dynamo         66.40 ms    96.18 ms      +44.9%
    4090    JIT dynamo         86.38 ms    113.87 ms     +31.8%
    launches/fwd (5090)        1,638       3,507         +114%

The branches are ZEROED — every request that names no adapter pays that to
compute zeros, and ~95% of sdxl denoiser forwards name no adapter.

The fix is a fork, not a flag: a bucket-bearing family mints BOTH classes into
the one cell, and the serve path routes per request. Real mint (torch.export +
AOTI on CPU) and real serve — the packaging and dispatch paths are exactly
what this issue changes, so neither is stood in for.
"""

from __future__ import annotations

import copy
import json
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from gen_worker import aot_flatten

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker import (  # noqa: E402
    aot_declaration,
    aot_mint,
    aot_serve,
    cell_key,
    compile_cache,
)
from gen_worker.api.decorators import Compile  # noqa: E402
from gen_worker.api.export_contract import (  # noqa: E402
    Dim,
    GraphClass,
    Input,
    register_export_declaration,
    reset_export_declarations,
)
from gen_worker.models import lora_lifted, w8a8_lora  # noqa: E402

FAMILY = "tiny790"
BUCKET = 16      # RANK_BUCKETS' floor — the cheapest real branch there is


class TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(BUCKET, BUCKET)

    def forward(self, sample: Any) -> Any:
        return torch.tanh(self.lin(sample))


def _declare() -> Any:
    """The DECLARED path, not the hand-registered escape hatch: the
    lifted pair is derived by `aot_declaration.declared_inputs` and bound to
    the positional slots `install_lifted_lora_forward` now appends to the
    denoiser's signature."""
    reset_export_declarations()
    return register_export_declaration(Compile(
        family=FAMILY,
        targets=("unet",),
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        classes=(GraphClass(dims={"B": 2}),),
        inputs=(Input("sample", shape=("B", BUCKET), dtype="model"),),
        shape_strategy="static-rows",
        warm_changes_key=False,
    ))


def _armed_pipe() -> Any:
    pipe = types.SimpleNamespace(unet=TinyUNet().eval())
    w8a8_lora.enable_branch_execution_lanes(pipe, BUCKET)
    lora_lifted.install_lifted_lora_execution_lanes(pipe, BUCKET)
    return pipe


def _spec() -> aot_mint.ExportSpec:
    return aot_mint.ExportSpec(
        family=FAMILY, target="", lora_bucket=BUCKET,
        lifted_inputs=lora_lifted.LIFTED_INPUT_NAMES)


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
    """ONE real two-arm mint, shared (an AOTI compile costs ~10s per entry).

    the mint yields TWO independently keyed artifacts rather than
    one two-entry cell. ``by_entry`` indexes them by the class each one names,
    which is the addressing every row below wants — an entry NAMES its class,
    and that is what makes a per-class refusal bisectable.
    """
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    request.addfinalizer(mp.undo)
    _fake_sm(mp)
    _declare()
    tmp = tmp_path_factory.mktemp("cell790")
    pipe = _armed_pipe()
    result = aot_mint.mint(
        pipe, _spec(), tmp / "out")
    reset_export_declarations()
    by_entry = {row.entry: row for row in result.entries}
    assert len(by_entry) == len(result.entries), "two entries collided on one name"
    return {"pipe": pipe, "result": result, "tmp": tmp, "by_entry": by_entry}


#: The two graph classes this declaration forks into.
LEAN = "unet/adapter=false/B=2"
FAT = "unet/adapter=true/B=2"


def _entry_block(cell: Dict[str, Any], name: str) -> Dict[str, Any]:
    """One class's recorded entry block, off ITS OWN artifact's metadata.

    pgw#1176 replaced the ``entries`` MAP with one ``entry`` block per
    artifact, so a class is addressed by picking its artifact rather than by
    indexing a map that could silently be missing the key.
    """
    row = cell["by_entry"][name]
    block = dict(row.metadata[cell_key.ENTRY_BLOCK_KEY])
    assert block["name"] == name, block.get("name")
    return block


# ---------------------------------------------------------------------------
# The plan fan-out — declaration-derived, composed-truth-scoped
# ---------------------------------------------------------------------------


def test_a_bucket_family_forks_every_branch_capable_target() -> None:
    decl = _declare()
    pipe = _armed_pipe()
    rows = aot_mint.adapter_arm_plans(
        aot_declaration.cell_plans(decl), pipe, _spec())
    names = [aot_declaration.plan_entry_name(p) for p, _arm in rows]
    assert names == ["unet/adapter=true/B=2", "unet/adapter=false/B=2"]
    # Adapter-bearing FIRST: the composed pipeline arrives lifted, so the
    # branch machinery is toggled off exactly once.
    assert [arm for _p, arm in rows] == [True, False]


def test_a_bucket_zero_family_forks_into_nothing() -> None:
    decl = _declare()
    pipe = types.SimpleNamespace(unet=TinyUNet().eval())
    spec = aot_mint.ExportSpec(family=FAMILY, target="")
    rows = aot_mint.adapter_arm_plans(
        aot_declaration.cell_plans(decl), pipe, spec)
    assert [arm for _p, arm in rows] == [None]
    assert aot_declaration.plan_entry_name(rows[0][0]) == "unet/B=2"


def test_a_target_with_no_branch_capable_module_does_not_fork() -> None:
    """wan's ``vae.decode`` in a bucket-128 cell: there is no adapter for it
    to carry, so a second identical graph would be pure mint bill."""
    decl = _declare()
    pipe = _armed_pipe()
    plans = aot_declaration.cell_plans(decl)
    plans = tuple(
        aot_declaration.replace(p, target="vae") for p in plans)  # type: ignore[attr-defined]
    pipe.vae = TinyUNet().eval()
    rows = aot_mint.adapter_arm_plans(plans, pipe, _spec())
    assert [arm for _p, arm in rows] == [None]


def test_the_arm_is_a_fork_coordinate_and_lands_in_the_class_hash() -> None:
    entry = {
        "target": "unet",
        "fork": [["adapter", True]],
        "class_dims": [["B", 2]],
        "range_digest": "abc",
        "graph": {"specialization": {"fork.adapter": True}},
    }
    with_adapter = aot_serve.class_hash(entry, strict=True, lora_bucket=BUCKET)
    branchless = copy.deepcopy(entry)
    branchless["fork"] = [["adapter", False]]
    branchless["graph"] = {"specialization": {"fork.adapter": False}}
    assert aot_serve.class_hash(
        branchless, strict=True, lora_bucket=BUCKET) != with_adapter


# ---------------------------------------------------------------------------
# The mint — two classes, one cell, contracts that discriminate
# ---------------------------------------------------------------------------


def test_one_mint_packages_both_arms(cell) -> None:
    """pgw#1176: "one cell, two graph classes" became "one mint, two
    independently keyed ARTIFACTS" — the fork is unchanged, what it packages
    into is not. Each artifact carries one class and is keyed by it, so the
    two must also be keyed DIFFERENTLY."""
    result = cell["result"]
    assert sorted(row.entry for row in result.entries) == [LEAN, FAT]
    assert len({row.key for row in result.entries}) == 2
    assert len({row.artifact for row in result.entries}) == 2


def test_the_branchless_arm_has_no_adapter_inputs_and_says_so(cell) -> None:
    lean = _entry_block(cell, LEAN)
    fat = _entry_block(cell, FAT)
    lean_names = {row["name"] for row in lean["inputs"]}
    fat_names = {row["name"] for row in fat["inputs"]}
    assert set(lora_lifted.LIFTED_INPUT_NAMES) <= fat_names
    assert not set(lora_lifted.LIFTED_INPUT_NAMES) & lean_names
    # The NEGATIVE half of the contract — without it the branchless class
    # silently admits an adapter-bearing call (see the ambiguity test below).
    assert lean["excluded_inputs"] == list(lora_lifted.LIFTED_INPUT_NAMES)
    assert "excluded_inputs" not in fat


def test_the_branchless_graph_actually_dropped_the_branch(cell) -> None:
    """Not just the signature: the adapter must be gone from the traced
    program, or the fork bought nothing."""
    lean = _entry_block(cell, LEAN)["graph"]
    fat = _entry_block(cell, FAT)["graph"]
    assert lean["lifted_inputs"] == []
    assert fat["lifted_inputs"] == sorted(lora_lifted.LIFTED_INPUT_NAMES)
    assert lean["specialization"]["fork.adapter"] is False
    assert lean["specialization"]["lora_bucket"] == 0
    assert fat["specialization"]["fork.adapter"] is True
    assert fat["specialization"]["lora_bucket"] == BUCKET


def test_the_pipeline_is_left_armed_after_the_branchless_exports(cell) -> None:
    """The mint disarms the branch containers to trace the lean class; a
    pipeline left branchless would silently be a different graph family for
    everything that follows."""
    unet = cell["pipe"].unet
    assert w8a8_lora.branch_bucket(unet) == BUCKET
    assert lora_lifted.lifted_binding(unet) is not None


# ---------------------------------------------------------------------------
# CELL IDENTITY — a declaration function, not a runtime one
# ---------------------------------------------------------------------------


def test_the_stamped_key_is_recomputable_from_the_recorded_facts(cell) -> None:
    """pgw#1176: recomputable PER ENTRY, off that entry's own artifact. The
    two classes must also recompute to two DIFFERENT keys, or the fork is not
    addressable and a serve path could fetch either for either."""
    rows = cell["result"].entries
    assert len(rows) == 2
    for row in rows:
        meta = dict(row.metadata)
        assert aot_mint.cell_identity(meta).digest == meta["cell_key"] == row.key
    assert len({row.key for row in rows}) == 2


def test_identity_is_a_function_of_the_declaration_not_of_adapter_state(
        tmp_path, monkeypatch) -> None:
    """The same declaration mints the same cell whether or not an adapter
    happens to be ACTIVE on the composed pipeline when the mint runs.

    This is the property Paul's rule protects: variability lives in tensor
    values and declared dims, never in program structure keyed on Python
    state. An active adapter changes the VALUES in the flat pair and nothing
    else, so it must not reach the key.
    """
    _fake_sm(monkeypatch)
    keys = []
    names = []
    for active in (False, True):
        _declare()
        pipe = _armed_pipe()
        if active:
            # A real placement through the real swap path, not a flag poke.
            adapter = {
                "lin.lora_A.weight": torch.randn(BUCKET, BUCKET) * 0.01,
                "lin.lora_B.weight": torch.randn(BUCKET, BUCKET) * 0.01,
            }
            lora_lifted.lifted_binding(pipe.unet).swap([(adapter, 1.0, "t")])
            assert lora_lifted.adapter_active(pipe.unet)
        out = tmp_path / f"active{int(active)}"
        result = aot_mint.mint(
            pipe, _spec(), out)
        # The key SET, per entry name. One comparison now carries
        # both halves this row used to assert apart (same classes, same keys),
        # and the length assert is what stops an empty mint passing it
        # vacuously.
        keys.append({row.entry: row.key for row in result.entries})
        names.append(sorted(row.entry for row in result.entries))
        reset_export_declarations()
    assert names[0] == names[1] == [LEAN, FAT]
    assert keys[0] == keys[1]


def test_an_exclusion_free_contract_digests_exactly_as_before() -> None:
    """The published fleet is not re-keyed by this change: a contract that
    excludes nothing hashes to what it always hashed."""
    block = {
        "inputs": [{"name": "sample", "position": 0, "dtype": "float32",
                    "shape": [2, 8]}],
        "symbols": {},
        "constants": [],
    }
    import hashlib

    canon = {
        "inputs": [{"name": "sample", "position": 0, "dtype": "float32",
                    "shape": [2, 8], "optional": False}],
        "symbols": {},
    }
    want = hashlib.sha256(json.dumps(
        canon, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]
    assert aot_serve.range_digest(block) == want
    # ...and a contract that DOES exclude something digests differently.
    excluding = dict(block, excluded_inputs=list(lora_lifted.LIFTED_INPUT_NAMES))
    assert aot_serve.range_digest(excluding) != want


# ---------------------------------------------------------------------------
# SERVE — routing by the declared contract, on the real dispatch
# ---------------------------------------------------------------------------


@pytest.fixture
def armed(cell, monkeypatch) -> Dict[str, Any]:
    """Both classes armed on a fresh pipeline through the real
    `aot_serve.arm_entry` — stage -> verify -> load -> bind -> register, ONE
    graph class per call.

    this is the accretion loop, and it is the production shape
    rather than a test convenience. Each entry arms whole or not at all and
    JOINS the target's registry, so the two classes below reach the dispatch
    through two independent arms — which is exactly what makes a single
    class's refusal cost one class instead of the whole fork.
    """
    _fake_sm(monkeypatch)
    pipe = _armed_pipe()
    cfg = types.SimpleNamespace(family=FAMILY)
    metas = {}
    for row in cell["result"].entries:
        metas[row.entry] = aot_serve.arm_entry(
            pipe, cfg, Path(row.artifact),
            cache_dir=cell["tmp"] / "stage")
    assert set(metas) == {LEAN, FAT}
    # Both classes are SERVING — the rows below distinguish which one served,
    # so an arm that silently dropped one would make them pass for the wrong
    # reason.
    assert set(aot_serve.entry_states(pipe)) == {LEAN, FAT}
    assert all(s["state"] == "armed"
               for s in aot_serve.entry_states(pipe).values())
    return {"pipe": pipe, "metas": metas}


def _call(pipe: Any) -> Any:
    with torch.no_grad():
        return pipe.unet(torch.randn(2, BUCKET))


def test_an_adapter_free_request_is_served_by_the_branchless_class(
        armed) -> None:
    pipe = armed["pipe"]
    assert not lora_lifted.adapter_active(pipe.unet)
    _call(pipe)
    assert aot_serve.served_entry_calls(pipe) == {
        "unet/adapter=false/B=2": 1, "unet/adapter=true/B=2": 0}
    assert aot_serve.ingress_refusals(pipe) == 0


def test_an_active_adapter_is_served_by_the_branch_bearing_class(armed) -> None:
    pipe = armed["pipe"]
    adapter = {
        "lin.lora_A.weight": torch.randn(BUCKET, BUCKET) * 0.05,
        "lin.lora_B.weight": torch.randn(BUCKET, BUCKET) * 0.05,
    }
    lora_lifted.lifted_binding(pipe.unet).swap([(adapter, 1.0, "turbo")])
    _call(pipe)
    assert aot_serve.served_entry_calls(pipe) == {
        "unet/adapter=false/B=2": 0, "unet/adapter=true/B=2": 1}
    assert aot_serve.ingress_refusals(pipe) == 0


def test_the_active_adapter_actually_changes_the_output(armed) -> None:
    """The routing is only worth anything if the branch-bearing class still
    APPLIES the adapter — a fast path that quietly serves the base model is
    the pgw#704 S9 defect wearing a performance hat."""
    pipe = armed["pipe"]
    sample = torch.randn(2, BUCKET)
    with torch.no_grad():
        base = pipe.unet(sample)
    binding = lora_lifted.lifted_binding(pipe.unet)
    adapter = {
        "lin.lora_A.weight": torch.eye(BUCKET) * 0.5,
        "lin.lora_B.weight": torch.eye(BUCKET) * 0.5,
    }
    binding.swap([(adapter, 1.0, "turbo")])
    with torch.no_grad():
        attached = pipe.unet(sample)
    assert not torch.allclose(base, attached, atol=1e-4)
    # And it matches EAGER with the same adapter placed — same numerics, one
    # graph class over.
    eager = TinyUNet().eval()
    eager.load_state_dict(pipe.unet.state_dict(), strict=False)
    w8a8_lora.enable_lora_branches(eager, BUCKET)
    ref_binding = lora_lifted.install_lifted_lora_forward(eager, BUCKET)
    ref_binding.swap([(adapter, 1.0, "turbo")])
    with torch.no_grad():
        want = eager(sample, **ref_binding.call_kwargs())
    assert torch.allclose(attached, want, atol=1e-3)
    # Clearing routes back to the branchless class, and back to the base value.
    binding.clear()
    with torch.no_grad():
        cleared = pipe.unet(sample)
    assert torch.allclose(base, cleared, atol=1e-4)
    assert aot_serve.served_entry_calls(pipe)["unet/adapter=false/B=2"] >= 2


def test_without_the_exclusion_the_dispatch_cannot_discriminate(cell) -> None:
    """RED for the negative declaration: strip ``excluded_inputs`` and an
    adapter-bearing call is admitted by BOTH classes, which the dispatch
    refuses as ``entry_ambiguous`` — i.e. the whole attach lane would fall
    back to eager."""
    lean_block = json.loads(json.dumps(_entry_block(cell, LEAN)))
    fat_block = json.loads(json.dumps(_entry_block(cell, FAT)))
    lean = dict(lean_block)
    lean.pop("excluded_inputs")
    lean_contract = aot_serve.contract_from_meta(lean)
    fat_contract = aot_serve.contract_from_meta(fat_block)
    dispatch = aot_serve.EntryDispatch((
        ("unet/adapter=false/B=2", aot_serve.ArtifactRunner(
            package=None, contract=lean_contract, constants=())),
        ("unet/adapter=true/B=2", aot_serve.ArtifactRunner(
            package=None, contract=fat_contract, constants=())),
    ))
    call = (torch.randn(2, BUCKET),)
    pair = {"lora_a": torch.zeros(BUCKET * BUCKET),
            "lora_b": torch.zeros(BUCKET * BUCKET)}
    with pytest.raises(aot_serve.IngressContractError) as err:
        dispatch.select(call, pair)
    assert err.value.reason == "entry_ambiguous"

    # With the declaration as minted, the same call routes deterministically.
    good = aot_serve.EntryDispatch((
        (LEAN, aot_serve.ArtifactRunner(
            package=None,
            contract=aot_serve.contract_from_meta(lean_block),
            constants=())),
        (FAT, aot_serve.ArtifactRunner(
            package=None, contract=fat_contract, constants=())),
    ))
    assert good.select(call, pair)[0] == "unet/adapter=true/B=2"
    assert good.select(call, {})[0] == "unet/adapter=false/B=2"


def test_a_single_class_cell_still_receives_the_mandatory_pair() -> None:
    """A cell with no branchless class declares the adapter pair MANDATORY;
    withholding it would refuse every request by name. The routing rule reads
    the declaration, so it does not."""
    module = TinyUNet().eval()
    w8a8_lora.enable_lora_branches(module, BUCKET)
    lora_lifted.install_lifted_lora_forward(module, BUCKET)
    contract = aot_serve.contract_from_meta({
        "inputs": [
            {"name": "sample", "position": 0, "dtype": "float32",
             "shape": [2, BUCKET]},
            {"name": "lora_a", "position": 1, "dtype": "float32",
             "shape": [BUCKET * BUCKET]},
            {"name": "lora_b", "position": 2, "dtype": "float32",
             "shape": [BUCKET * BUCKET]},
        ],
        "symbols": {}, "constants": [],
    })
    sole = aot_serve.ArtifactRunner(
        package=None, contract=contract, constants=())
    artifact_kwargs, eager_kwargs = aot_serve.adapter_call_kwargs(module, sole)
    assert set(artifact_kwargs) == set(lora_lifted.LIFTED_INPUT_NAMES)
    assert set(eager_kwargs) == set(lora_lifted.LIFTED_INPUT_NAMES)


def test_the_eager_fallback_always_receives_the_pair() -> None:
    """``bind_views`` refuses a ``None`` half by name, so an adapter-free
    request routed to a branchless artifact must still hand the EAGER forward
    its pair — the fallback is not on the lean call set."""
    module = TinyUNet().eval()
    w8a8_lora.enable_lora_branches(module, BUCKET)
    lora_lifted.install_lifted_lora_forward(module, BUCKET)
    lean = aot_serve.contract_from_meta({
        "inputs": [{"name": "sample", "position": 0, "dtype": "float32",
                    "shape": [2, BUCKET]}],
        "symbols": {}, "constants": [],
        "excluded_inputs": list(lora_lifted.LIFTED_INPUT_NAMES),
    })
    runner = aot_serve.ArtifactRunner(
        package=None, contract=lean, constants=())
    artifact_kwargs, eager_kwargs = aot_serve.adapter_call_kwargs(module, runner)
    assert artifact_kwargs == {}
    assert set(eager_kwargs) == set(lora_lifted.LIFTED_INPUT_NAMES)
    # The eager forward accepts exactly that call.
    with torch.no_grad():
        module(torch.randn(2, BUCKET), **eager_kwargs)


# ---------------------------------------------------------------------------
# The contract's NAMES — a flattened container must not shift them (found on
# a real sdxl UNet while proving this issue; see `aot_mint.flat_input_names`)
# ---------------------------------------------------------------------------


class NestedUNet(nn.Module):
    """A denoiser shaped like SDXL's: a container argument sits between the
    declared tensors and the parameters that follow it."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(BUCKET, BUCKET)

    def forward(self, sample, timestep=None, class_labels=None,
                added_cond_kwargs=None, extra_residuals=None):
        out = torch.tanh(self.lin(sample))
        if added_cond_kwargs is not None:
            out = out + added_cond_kwargs["time_ids"].sum()
            out = out + added_cond_kwargs["text_embeds"].sum()
        return out


def test_a_flattened_container_does_not_shift_the_declared_names() -> None:
    module = NestedUNet().eval()
    args = (torch.randn(2, BUCKET), torch.tensor(1.0), None,
            {"text_embeds": torch.randn(2, 4), "time_ids": torch.randn(2, 6)})
    flat = tuple(leaf.name for leaf in
                 aot_mint.flat_input_leaves(module, args, {}))
    # One name per EXPORTED user input, in export's own (insertion-order) order —
    # not one name per PARAMETER, which is what shifted `text_embeds` onto
    # `added_cond_kwargs` and `time_ids` onto the parameter after it.
    assert flat == ("sample", "timestep", "class_labels",
                    "text_embeds", "time_ids")
    assert aot_mint._input_names(module, args, {}) == (
        "sample", "timestep", "class_labels", "added_cond_kwargs")


def test_the_recorded_contract_names_the_flattened_leaves(tmp_path) -> None:
    from gen_worker import aot_package

    module = NestedUNet().eval()
    args = (torch.randn(2, BUCKET), torch.tensor(1.0), None,
            {"text_embeds": torch.randn(2, 4), "time_ids": torch.randn(2, 6)})
    with torch.no_grad():
        program = torch.export.export(module, args, {}, strict=False)
    rows, _symbols = aot_package.input_contract(
        program, aot_mint.flat_input_leaves(module, args, {}))
    by_name = {r["name"]: r for r in rows}
    assert by_name["text_embeds"]["shape"] == [2, 4]
    assert by_name["time_ids"]["shape"] == [2, 6]
    assert "added_cond_kwargs" not in by_name

    # RED: the pre-fix derivation (parameter names) mislabels both.
    stale, _ = aot_package.input_contract(program, [
        aot_flatten.Leaf(param=name, param_position=index, path=())
        for index, name in enumerate(aot_mint._input_names(module, args, {}))
    ])
    stale_names = {r["name"] for r in stale}
    assert "time_ids" not in stale_names
    assert "added_cond_kwargs" in stale_names


def test_the_serve_path_binds_the_flattened_names_from_a_nested_call() -> None:
    """The names the mint records must be the ones a diffusers-shaped call can
    actually be resolved against.

    REWRITTEN by pgw#994. This used to pass because `bind_call_inputs`
    SEARCHED every mapping-valued kwarg for the bare leaf key — which worked
    only while no other kwarg carried the same key, and could not help the
    list case at all. The contract now records where each leaf lives and the
    serve side replays it; the search is deleted.
    """
    contract = aot_serve.contract_from_meta({
        "inputs": [
            {"name": "sample", "position": 0, "dtype": "float32",
             "shape": [2, BUCKET]},
            {"name": "text_embeds", "position": 3, "dtype": "float32",
             "shape": [2, 4], "param": "added_cond_kwargs",
             "param_position": 3, "path": ["text_embeds"]},
            {"name": "time_ids", "position": 4, "dtype": "float32",
             "shape": [2, 6], "param": "added_cond_kwargs",
             "param_position": 3, "path": ["time_ids"]},
        ],
        "symbols": {}, "constants": [],
    })
    call_kwargs = {"added_cond_kwargs": {
        "text_embeds": torch.randn(2, 4), "time_ids": torch.randn(2, 6)}}
    feeds = aot_serve.marshal_positional(
        contract, (torch.randn(2, BUCKET),), call_kwargs)
    assert [tuple(f.shape) for f in feeds] == [(2, BUCKET), (2, 4), (2, 6)]
