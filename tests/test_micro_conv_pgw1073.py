"""pgw#1073 — the conv (static-rows) and second-bucket gauntlet members,
asserted where CI can see them.

The full mint cycle needs a compiler and is LOCAL-ONLY (the rig). What does
NOT need one is every structural claim the new members make — the strategy,
the compiled graph multiplicity, the int64 input, the persistent buffer, and the
determinism of the derived conv weights. Each test goes RED if the property
it names stops holding.
"""

from __future__ import annotations

import sys

import pytest
import torch

from harness.rig_vehicles import (
    MICRO_LORA16_BUCKET,
    MICRO_LORA_BUCKET,
    MICRO_SRC,
)

if str(MICRO_SRC) not in sys.path:
    sys.path.insert(0, str(MICRO_SRC))

from micro_diffusion import aot_declaration_conv as decl_mod  # noqa: E402
from micro_diffusion.main_conv import GenerateConv  # noqa: E402
from micro_diffusion.model import MicroConfig  # noqa: E402
from micro_diffusion.model_conv import (  # noqa: E402
    MicroConvDenoiser,
    build_conv_denoiser,
)
from micro_diffusion.pipeline import MicroConvPipeline  # noqa: E402
from micro_diffusion.weights import materialize  # noqa: E402

from gen_worker import aot_declaration as ad  # noqa: E402
from gen_worker import compile_cache as cc  # noqa: E402
from gen_worker.aot_mint import ExportSpec  # noqa: E402


@pytest.fixture(scope="module")
def declaration():
    return decl_mod.build_declaration()


# ---------------------------------------------------------------------------
# The static-rows claims — sdxl's class, at micro scale
# ---------------------------------------------------------------------------


def test_static_rows_yields_four_static_compiled_graphs(declaration) -> None:
    """2 rows x the cfg fork = 4 compiled graphs, each a STATIC graph. Under
    static-rows an compiled graph's dims ARE its coordinate — no dynamic dims, which
    is the difference from every other micro member and the surface pgw#1058
    broke on (per-row compiled graph labels vs serve-side asks)."""
    assert declaration.shape_strategy == "static-rows"
    plans = ad.compiled_graph_plans(declaration)
    assert len(plans) == 4
    names = sorted(ad.plan_compiled_graph_name(p) for p in plans)
    assert len(set(names)) == 4
    for plan in plans:
        assert plan.dynamic == (), (
            f"{ad.plan_compiled_graph_name(plan)} carries dynamic dims under "
            f"static-rows — the strategy's whole point is that it does not")


def test_the_graph_is_conv_bearing(declaration) -> None:
    """The mirror image of micro's conv-free test: #730 ratified static-rows
    FOR conv-bearing graphs, so a later edit that deletes the convs would
    make the declared strategy unjustified, silently."""
    convs = [m for m in MicroConvDenoiser(MicroConfig()).modules()
             if isinstance(m, torch.nn.modules.conv._ConvNd)]
    assert len(convs) >= 4


def test_the_timestep_is_declared_and_fed_int64(declaration) -> None:
    """The mixed-dtype signature (wan-2.2's shape, the pgw#1058 axis): dtype
    is a declared per-input fact, and the int64 input must be STRUCTURAL —
    it indexes an nn.Embedding, so a float would fail eagerly, not just
    drift."""
    by_name = {i.name: i for i in declaration.inputs}
    assert by_name["timestep"].dtype == "int64"
    assert by_name["sample"].dtype == "float32"
    module = build_conv_denoiser(MicroConfig())
    spec = ExportSpec(family=decl_mod.FAMILY, target="unet",
                      fork=(("cfg", True),),
                      class_dims=(("B", 2), ("H_lat", 24), ("W_lat", 24)))
    args, kwargs = ad.declared_inputs(module, spec, declaration)
    assert kwargs == {}
    assert args[1].dtype == torch.int64
    with torch.no_grad():
        out = module(*args)
    assert tuple(out.shape) == (2, MicroConfig().in_channels, 24, 24)


def test_the_class_table_is_a_persistent_named_buffer() -> None:
    """The H3 pattern — a config-derived table that is CHECKPOINT STATE,
    bound like a weight. Contrast micro's rope ``freqs`` (non-persistent, the
    literal half of pgw#857): this one must appear in state_dict."""
    module = MicroConvDenoiser(MicroConfig())
    assert "class_table" in dict(module.named_buffers())
    assert "class_table" in module.state_dict()


def test_conv_weights_are_a_pure_function_of_the_tree(tmp_path) -> None:
    """Two processes rebuilding from the same tree must agree byte-for-byte —
    the determinism claim the adopt leg's parity number stands on."""
    tree = materialize(tmp_path / "w")
    first = MicroConvPipeline.from_pretrained(str(tree))
    second = MicroConvPipeline.from_pretrained(str(tree))
    s1, s2 = first.unet.state_dict(), second.unet.state_dict()
    assert sorted(s1) == sorted(s2)
    for name, tensor in s1.items():
        assert torch.equal(tensor, s2[name]), name


def test_the_slot_is_a_catalog_slot_with_no_code_default() -> None:
    spec = GenerateConv.__gen_worker_endpoint__
    slot = spec.slots["pipeline"]
    assert slot.selected_by == "model"
    assert slot.default_checkpoint is None
    assert slot.pipeline_cls is MicroConvPipeline


# ---------------------------------------------------------------------------
# The second bucket — the lane label follows the NUMBER
# ---------------------------------------------------------------------------


def test_two_buckets_are_two_lanes() -> None:
    """The bucket is a KEY axis: lora16 and lora64 must be disjoint lanes, so
    a compiled graph minted at one rank can never be armed at the other."""
    assert MICRO_LORA16_BUCKET != MICRO_LORA_BUCKET
    lane16 = cc.execution_lane_label("", MICRO_LORA16_BUCKET)
    lane64 = cc.execution_lane_label("", MICRO_LORA_BUCKET)
    assert lane16 != lane64
    assert str(MICRO_LORA16_BUCKET) in lane16
    assert str(MICRO_LORA_BUCKET) in lane64
