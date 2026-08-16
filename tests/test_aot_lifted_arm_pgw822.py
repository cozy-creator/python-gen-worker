"""The mint must hand ``torch.export`` the module it declared.

The failure it prevents: the delegated child reaches ``trace_graph`` and is
refused with *"declared input(s) ['lora_a', 'lora_b'] are not parameters of
'forward'"*. The declaration is right — the ``lora64`` bucket lifts the adapter
to graph INPUTS on purpose, since an adapter that is a call argument can never
be baked. The MODULE is wrong: arming the branch CONTAINERS
(``compile_cache.apply_lora_lane``, the dynamo lane's end state) and stopping
there hands export the bare denoiser, whose forward takes no such parameters.

Two ends, both here:

* the ARM — one function (``lora_lifted.arm_lifted_lora_lanes``) owned by the
  mint, so an adapter-bearing class is exported from the lifted forward and a
  branchless one from the plain module (per class);
* the PRE-SPAWN check — ``aot_mint.declaration_module_gaps``, which asks the
  same signature question on the PARENT, before a child is spawned or a pod is
  rented, and declines by name.
"""

from __future__ import annotations

import dataclasses
import types
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from torch import nn
from gen_worker._vendor.torch_compiled_graphs import CallIngress

from gen_worker import (
    aot_declaration,
    aot_mint,
    fleet_cells,
)
from gen_worker.api.decorators import Compile
from gen_worker.api.export_contract import (
    Dim,
    GraphClass,
    Input,
    register_export_declaration,
    reset_export_declarations,
)
from gen_worker.models import lora_lifted, w8a8_lora

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
    fields: dict[str, Any] = {
        "family": FAMILY,
        "targets": ("unet",),
        "dims": (Dim("B", carried_by=(("sample", 0),)),),
        "classes": (GraphClass(dims={"B": 2}),),
        "inputs": (Input("sample", shape=("B", BUCKET), dtype="model"),),
        "shape_strategy": "static-rows",
        "warm_changes_key": False,
    }
    fields.update(changes)
    return register_export_declaration(Compile(**fields))


def _container_only_pipe() -> Any:
    """Exactly what ``compile_cache.apply_lora_lane`` leaves behind — the
    state the compile child hands to ``trace_for_key``.
    Branch containers allocated, denoiser forward untouched."""
    pipe = types.SimpleNamespace(unet=TinyUNet().eval())
    w8a8_lora.enable_branch_execution_lanes(pipe, BUCKET)
    assert lora_lifted.lifted_binding(pipe.unet) is None
    return pipe


def _spec() -> aot_mint.ExportSpec:
    return aot_mint.ExportSpec(
        family=FAMILY, target="", lora_bucket=BUCKET,
        lifted_inputs=lora_lifted.LIFTED_INPUT_NAMES)


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
    lora_lifted.arm_lifted_lora_execution_lanes(pipe, BUCKET)
    binding = lora_lifted.lifted_binding(pipe.unet)
    assert binding is not None
    import inspect

    params = list(inspect.signature(pipe.unet.forward).parameters)
    assert params[-2:] == list(lora_lifted.LIFTED_INPUT_NAMES)


def test_the_arm_is_idempotent_and_a_noop_at_bucket_zero() -> None:
    pipe = _container_only_pipe()
    first = lora_lifted.arm_lifted_lora_execution_lanes(pipe, BUCKET)["unet"]
    again = lora_lifted.arm_lifted_lora_execution_lanes(pipe, BUCKET)["unet"]
    assert again is first
    plain = types.SimpleNamespace(unet=TinyUNet().eval())
    assert lora_lifted.arm_lifted_lora_execution_lanes(plain, 0) == {}
    assert lora_lifted.lifted_binding(plain.unet) is None


def test_the_declared_feed_binds_once_the_execution_lane_is_armed() -> None:
    """The measured refusal, at its exact site: ``declared_inputs`` binding
    the declaration to the module's own signature."""
    decl = _declare()
    pipe = _container_only_pipe()
    spec = dataclasses.replace(_spec(), target="unet")

    with pytest.raises(aot_mint.MintRefused, match="are not parameters of"):
        aot_declaration.declared_inputs(pipe.unet, spec, decl)

    lora_lifted.arm_lifted_lora_execution_lanes(pipe, BUCKET)
    args, kwargs = aot_declaration.declared_inputs(pipe.unet, spec, decl)
    assert kwargs == {}
    assert len(args) == 3          # sample + the lifted pair, all positional
    assert all(isinstance(a, torch.Tensor) for a in args)


# ---------------------------------------------------------------------------
# The trace into TCG declarations: both classes, no AOTI compile
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def traced_classes() -> dict[str, Any]:
    """One two-arm export from a CONTAINER-ONLY pipeline, stopped at TCG's
    public declaration boundary. At base this raises
    ``MintRefused: ... ['lora_a', 'lora_b'] are not parameters of 'forward'``
    and every assertion below is unreachable. ``Engine.compile`` is never
    constructed or called."""
    decl = _declare()
    pipe = _container_only_pipe()
    rows = list(aot_mint.trace_for_key(pipe, _spec(), decl))
    declarations = {
        row.name: aot_mint.tcg_graph_class_spec(row, _spec()).declare()
        for row in rows
    }
    for row in rows:
        row.release()
    reset_export_declarations()
    by_entry = {row.name: row for row in rows}
    assert len(by_entry) == len(rows), "two traces collided on one name"
    return {"pipe": pipe, "by_entry": by_entry, "declarations": declarations}


#: The two graph classes this declaration forks into.
LEAN = "unet/adapter=false/B=2"
FAT = "unet/adapter=true/B=2"


def test_a_container_only_pipeline_declares_both_graph_classes(
    traced_classes: dict[str, Any],
) -> None:
    assert sorted(traced_classes["by_entry"]) == [LEAN, FAT]
    declarations = traced_classes["declarations"]
    assert len({row.class_hash for row in declarations.values()}) == 2


def test_the_two_classes_were_prepared_DIFFERENTLY(
    traced_classes: dict[str, Any],
) -> None:
    """The pgw#790 fork survives the pgw#822 arm: the adapter-bearing class
    carries the lifted pair, the branchless one is exported from the PLAIN
    module and says so. A one-size wrapper would break this."""
    fat = traced_classes["declarations"][FAT]
    lean = traced_classes["declarations"][LEAN]
    fat_ingress = CallIngress.from_graph(fat.graph)
    lean_ingress = CallIngress.from_graph(lean.graph)
    assert set(lora_lifted.LIFTED_INPUT_NAMES) <= {
        row.name for row in fat_ingress.inputs}
    assert not set(lora_lifted.LIFTED_INPUT_NAMES) & {
        row.name for row in lean_ingress.inputs}
    assert lean_ingress.excluded_inputs == tuple(sorted(lora_lifted.LIFTED_INPUT_NAMES))
    assert fat.graph["lifted_inputs"] == sorted(
        lora_lifted.LIFTED_INPUT_NAMES)
    assert lean.graph["lifted_inputs"] == []


def test_the_pipeline_is_left_lifted_after_the_trace(
    traced_classes: dict[str, Any],
) -> None:
    """The branchless exports disarm the pipeline; a mint that returned it
    branchless would leave the process serving a different graph family."""
    assert lora_lifted.lifted_binding(traced_classes["pipe"].unet) is not None
    assert w8a8_lora.branch_bucket(traced_classes["pipe"].unet) == BUCKET


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
        aot_mint._export_entry(pipe, _spec(), plan, decl)


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
        Input("sample", shape=("B", BUCKET), dtype="model"),
        Input("encoder_hidden_states", shape=("B", BUCKET), dtype="model"),
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
        Input("sample", shape=("B", BUCKET), dtype="model"),
        Input("encoder_hidden_states", shape=("B", BUCKET), dtype="model"),
    ))
    assert decl is not None
    pipe = _container_only_pipe()
    cfg = types.SimpleNamespace(
        family=FAMILY, lora_bucket=BUCKET, shapes=(), text_lens=(),
        guidance_scales=(), targets=("unet",))

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
