"""pgw#822 — the mint must hand ``torch.export`` the module it declared.

Measured on a real L4 (gen-worker 0.82.0, release ``b0b592812227228604c2237f``,
sdxl 0.2.100, lane ``w8a8``/``lora64``, pod ``2cuc4oituz1wyi``): the delegated
child loaded the pipeline, reached ``trace_graph``, and was refused —

    aot mint refused: family 'sdxl': declared input(s) ['lora_a', 'lora_b']
    are not parameters of 'forward' on UNet2DConditionModel — the declaration
    does not fit this module

The declaration was right. The ``lora64`` bucket lifts the adapter to graph
INPUTS on purpose (pgw#725 option 2: an adapter that is a call argument can
never be baked). The MODULE was wrong: the child armed the branch CONTAINERS
(``compile_cache.apply_lora_lane``, the dynamo lane's end state) and stopped,
so the object handed to export was the bare denoiser, whose forward takes no
such parameters.

Two ends, both here:

* the ARM — one function (``lora_lifted.arm_lifted_lora_lanes``) owned by the
  mint, so an adapter-bearing class is exported from the lifted forward and a
  branchless one from the plain module (the pgw#790 fork, per class);
* the PRE-SPAWN check — ``aot_mint.declaration_module_gaps``, which asks the
  same signature question on the PARENT, before a child is spawned or a pod is
  rented, and declines by name.
"""

from __future__ import annotations

import types
from typing import Any, Dict

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker import (  # noqa: E402
    aot_declaration,
    aot_mint,
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
from gen_worker.models import lora_lifted, w8a8_lora  # noqa: E402

FAMILY = "tiny822"
BUCKET = 16      # RANK_BUCKETS' floor — the cheapest real branch there is


class TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(BUCKET, BUCKET)

    def forward(self, sample: Any) -> Any:
        return torch.tanh(self.lin(sample))


class NoBranchVAE(nn.Module):
    """No Linear/Conv leaf — nothing a branch could ride."""

    def forward(self, sample: Any) -> Any:
        return torch.tanh(sample)


def _declare(**changes: Any) -> Any:
    reset_export_declarations()
    fields: Dict[str, Any] = dict(
        family=FAMILY,
        targets=("unet",),
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        classes=(GraphClass(dims={"B": 2}),),
        inputs=(Input("sample", shape=("B", BUCKET)),),
        shape_strategy="static-rows",
        warm_changes_key=False,
    )
    fields.update(changes)
    return register_export_declaration(Compile(**fields))


def _container_only_pipe() -> Any:
    """Exactly what ``compile_cache.apply_lora_lane`` leaves behind — the
    state the mint child handed to ``aot_mint.mint`` in the measured run.
    Branch containers allocated, denoiser forward untouched."""
    pipe = types.SimpleNamespace(unet=TinyUNet().eval())
    w8a8_lora.enable_branch_lanes(pipe, BUCKET)
    assert lora_lifted.lifted_binding(pipe.unet) is None
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


# ---------------------------------------------------------------------------
# The arm — one function, both halves, per class
# ---------------------------------------------------------------------------


def test_the_arm_installs_both_halves_over_a_container_only_pipeline() -> None:
    pipe = _container_only_pipe()
    lora_lifted.arm_lifted_lora_lanes(pipe, BUCKET)
    binding = lora_lifted.lifted_binding(pipe.unet)
    assert binding is not None
    import inspect

    params = list(inspect.signature(pipe.unet.forward).parameters)
    assert params[-2:] == list(lora_lifted.LIFTED_INPUT_NAMES)


def test_the_arm_is_idempotent_and_a_noop_at_bucket_zero() -> None:
    pipe = _container_only_pipe()
    first = lora_lifted.arm_lifted_lora_lanes(pipe, BUCKET)["unet"]
    again = lora_lifted.arm_lifted_lora_lanes(pipe, BUCKET)["unet"]
    assert again is first
    plain = types.SimpleNamespace(unet=TinyUNet().eval())
    assert lora_lifted.arm_lifted_lora_lanes(plain, 0) == {}
    assert lora_lifted.lifted_binding(plain.unet) is None


def test_the_declared_feed_binds_once_the_lane_is_armed() -> None:
    """The measured refusal, at its exact site: ``declared_inputs`` binding
    the declaration to the module's own signature."""
    decl = _declare()
    pipe = _container_only_pipe()
    spec = aot_mint.replace_spec(_spec(), target="unet")

    with pytest.raises(aot_mint.MintRefused, match="are not parameters of"):
        aot_declaration.declared_inputs(pipe.unet, spec, decl)

    lora_lifted.arm_lifted_lora_lanes(pipe, BUCKET)
    args, kwargs = aot_declaration.declared_inputs(pipe.unet, spec, decl)
    assert kwargs == {}
    assert len(args) == 3          # sample + the lifted pair, all positional
    assert all(isinstance(a, torch.Tensor) for a in args)


# ---------------------------------------------------------------------------
# The mint, end to end: RED here at base, both classes here after the fix
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cell(tmp_path_factory, request) -> Dict[str, Any]:
    """ONE real two-arm mint (torch.export + AOTI) from a CONTAINER-ONLY
    pipeline — the state the child actually produced. At base this raises
    ``MintRefused: ... ['lora_a', 'lora_b'] are not parameters of 'forward'``
    and every assertion below is unreachable."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    request.addfinalizer(mp.undo)
    _fake_sm(mp)
    _declare()
    tmp = tmp_path_factory.mktemp("cell822")
    pipe = _container_only_pipe()
    result = aot_mint.mint(
        pipe, _spec(), tmp / "out", allow_regressed_lanes=True)
    reset_export_declarations()
    return {"pipe": pipe, "result": result}


def test_a_container_only_pipeline_mints_both_graph_classes(cell) -> None:
    assert sorted(cell["result"].metadata["entries"]) == [
        "unet/adapter=false/B=2", "unet/adapter=true/B=2"]


def test_the_two_classes_were_prepared_DIFFERENTLY(cell) -> None:
    """The pgw#790 fork survives the pgw#822 arm: the adapter-bearing class
    carries the lifted pair, the branchless one is exported from the PLAIN
    module and says so. A one-size wrapper would break this."""
    entries = cell["result"].metadata["entries"]
    fat = entries["unet/adapter=true/B=2"]
    lean = entries["unet/adapter=false/B=2"]
    assert set(lora_lifted.LIFTED_INPUT_NAMES) <= {
        row["name"] for row in fat["inputs"]}
    assert not set(lora_lifted.LIFTED_INPUT_NAMES) & {
        row["name"] for row in lean["inputs"]}
    assert lean["excluded_inputs"] == list(lora_lifted.LIFTED_INPUT_NAMES)
    assert fat["graph"]["lifted_inputs"] == sorted(
        lora_lifted.LIFTED_INPUT_NAMES)
    assert lean["graph"]["lifted_inputs"] == []


def test_the_pipeline_is_left_lifted_after_the_mint(cell) -> None:
    """The branchless exports disarm the pipeline; a mint that returned it
    branchless would leave the process serving a different graph family."""
    assert lora_lifted.lifted_binding(cell["pipe"].unet) is not None
    assert w8a8_lora.branch_bucket(cell["pipe"].unet) == BUCKET


def test_an_unarmed_export_refuses_naming_the_ARM_not_the_declaration() -> None:
    """The permanent guard. If a future caller reaches ``_export_entry`` with
    the lift missing, the sentence must point at the pipeline's PREPARATION —
    the measured run's sentence pointed at the endpoint's contract, which is
    where the first investigation went."""
    decl = _declare()
    pipe = _container_only_pipe()
    plan, arm = aot_mint.adapter_arm_plans(
        aot_declaration.cell_plans(decl), pipe, _spec())[0]
    assert arm is True
    with pytest.raises(aot_mint.MintRefused, match="carries no lifted forward"):
        aot_mint._export_entry(pipe, _spec(), plan, decl, compile_now=False)


# ---------------------------------------------------------------------------
# The pre-spawn check — the same question, before anything is rented
# ---------------------------------------------------------------------------


def test_no_gaps_when_the_declaration_fits() -> None:
    decl = _declare()
    assert aot_mint.declaration_module_gaps(
        _container_only_pipe(), _spec(), decl) == []


def test_the_lifted_pair_is_admitted_on_a_lift_CAPABLE_target() -> None:
    """The parent is not lifted — the mint lifts it. The check must predict
    the mint, not describe the parent, or it would decline every correct
    bucket-bearing mint on the fleet."""
    decl = _declare()
    pipe = _container_only_pipe()
    assert lora_lifted.lifted_binding(pipe.unet) is None
    assert aot_mint.declaration_module_gaps(pipe, _spec(), decl) == []


def test_a_declared_input_the_module_does_not_take_is_named() -> None:
    decl = _declare(inputs=(
        Input("sample", shape=("B", BUCKET)),
        Input("encoder_hidden_states", shape=("B", BUCKET)),
    ))
    gaps = aot_mint.declaration_module_gaps(
        _container_only_pipe(), _spec(), decl)
    assert len(gaps) == 2                      # both fork arms
    assert all("'encoder_hidden_states'" in g for g in gaps)
    assert all("are not parameters of 'forward'" in g for g in gaps)


def test_a_lifted_declaration_on_a_target_that_cannot_carry_one_is_named() -> None:
    """A bucket declared over a target with no branch-capable leaf: no lifted
    forward can ever be installed there, so admitting the pair "because the
    mint lifts it" would be a lie the pod would pay to discover."""
    decl = _declare(targets=("vae",))
    pipe = _container_only_pipe()
    pipe.vae = NoBranchVAE().eval()
    gaps = aot_mint.declaration_module_gaps(pipe, _spec(), decl)
    assert len(gaps) == 1
    assert "no branch-capable module" in gaps[0]
    assert "'lora_a'" in gaps[0]


def test_the_parent_declines_the_mint_by_name_instead_of_renting(monkeypatch) -> None:
    """The filing's second ask: this refusal was knowable locally, so it must
    cost a log line and not a pod. Serving is untouched — the decline falls
    back to the dynamo recipe exactly as every other named decline does."""
    decl = _declare(inputs=(
        Input("sample", shape=("B", BUCKET)),
        Input("encoder_hidden_states", shape=("B", BUCKET)),
    ))
    assert decl is not None
    pipe = _container_only_pipe()
    cfg = types.SimpleNamespace(
        family=FAMILY, lora_bucket=BUCKET, shapes=(), text_lens=(),
        guidance_scales=(), targets=("unet",))

    monkeypatch.setattr(fleet_cells.aot_cells, "prefer_aot", lambda: True)
    monkeypatch.setattr(
        aot_mint, "lane_admitted",
        lambda spec, allow_regressed_lanes=False: "")
    monkeypatch.setattr(aot_mint, "lifted_torch_gap", lambda spec: "")
    events: list = []
    monkeypatch.setattr(
        fleet_cells.activity_mod, "emit_event",
        lambda kind, detail, **kw: events.append((kind, detail, kw)))

    recipe = fleet_cells.mint_recipe(pipe, cfg, delegate=True)
    assert recipe == fleet_cells.RECIPE_DYNAMO
    assert len(events) == 1
    kind, detail, kw = events[0]
    assert kind == "self_mint_skipped"
    assert kw["phase"] == "declaration_module_mismatch"
    assert "encoder_hidden_states" in detail
