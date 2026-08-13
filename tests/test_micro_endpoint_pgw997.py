"""pgw#997 — the micro-diffusion endpoint, asserted where CI can see it.

The full mint cycle needs a compiler and is LOCAL-ONLY (pgw#978's rig). What
does NOT need one is every claim this family makes about its own SHAPE, and
those are the claims that would rot silently: the compiled graph count, the container
ordering that keeps pgw#993/pgw#994 under test, the determinism of the
generated weights, and the two model properties the declaration's strategy
depends on.

Each test below is written so it goes RED if the property it names stops
holding — not so it passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from harness.rig_vehicles import MICRO_SRC

if str(MICRO_SRC) not in sys.path:
    sys.path.insert(0, str(MICRO_SRC))

from micro_diffusion import aot_declaration as decl_mod  # noqa: E402
from micro_diffusion.main import Generate, Size  # noqa: E402
from micro_diffusion.model import (  # noqa: E402
    MicroConfig,
    MicroDecoder,
    MicroDenoiser,
)
from micro_diffusion.pipeline import MicroPipeline  # noqa: E402
from micro_diffusion.weights import (  # noqa: E402
    MAX_TREE_BYTES,
    SEED,
    load_state,
    materialize,
    state_dict,
    tree_bytes,
)

from gen_worker import aot_declaration as ad  # noqa: E402
from gen_worker import aot_flatten  # noqa: E402
from gen_worker.aot_mint import ExportSpec  # noqa: E402

EXPECTED_COMPILED_GRAPHS = ["decoder", "transformer/cfg=false", "transformer/cfg=true"]


@pytest.fixture(scope="module")
def declaration():
    return decl_mod.build_declaration()


@pytest.fixture(scope="module")
def tree(tmp_path_factory) -> Path:
    return materialize(tmp_path_factory.mktemp("micro-weights"))


# ---------------------------------------------------------------------------
# The size claim — three compiled graphs, and the reason it is three
# ---------------------------------------------------------------------------


def test_declares_exactly_three_compiled_graphs(declaration) -> None:
    """The whole premise. sdxl declares 36 and costs ~95 min a cycle."""
    names = sorted(ad.plan_compiled_graph_name(p) for p in ad.compiled_graph_plans(declaration))
    assert names == EXPECTED_COMPILED_GRAPHS


def test_the_two_latent_rows_collapse_rather_than_multiply(declaration) -> None:
    """Two declared rows, still one artifact per arm — with a DERIVED range.

    If this collapsed the other way the family would have 6 compiled graphs, and if
    the range were not derived the artifact would silently serve only its
    seed row.
    """
    assert len(declaration.shapes) == 2
    for plan in ad.compiled_graph_plans(declaration):
        assert len(plan.rows) == 2, ad.plan_compiled_graph_name(plan)
        spans = {(d.input_name, d.axis): (d.min, d.max) for d in plan.dynamic}
        assert spans, (
            f"{ad.plan_compiled_graph_name(plan)} collapsed two rows with NO dynamic "
            f"dim — the artifact would silently serve only its seed row")
        assert set(spans.values()) == {(1024, 2304)}


def test_every_traced_extent_is_linear_in_one_symbol(declaration) -> None:
    """pgw#998, kept RED-able. A matmul extent that is a PRODUCT of two
    dynamic axes cannot be lowered after the mint's own
    ``torch.export.save``/``load`` hand-off to its compile child. The
    declaration avoids it by taking token sequences, and this test fails if a
    later edit puts two dynamic axes back on one input."""
    for plan in ad.compiled_graph_plans(declaration):
        per_input: dict = {}
        for row in plan.dynamic:
            per_input.setdefault(row.input_name, []).append(row.axis)
        for name, axes in per_input.items():
            assert len(axes) == 1, (
                f"{ad.plan_compiled_graph_name(plan)}/{name} has SYMBOLIC axes {axes} "
                f"— their product is a nonlinear traced extent, which "
                f"inductor cannot lower after the mint's export save/load "
                f"(pgw#998)")


def test_the_decoder_is_a_second_target_with_its_own_dims(declaration) -> None:
    """A dim carried through a CONTAINER for one target and a PLAIN tensor for
    the other — pgw#993's seam, declared."""
    assert declaration.targets == ("transformer", "decoder")
    by_name = {d.name: d for d in declaration.dims}
    assert ("x", 0) in by_name["T"].carried_by
    assert ("latent", 1) in by_name["T"].carried_by
    denoiser = [d for d in ad.compiled_graph_plans(declaration)
                if d.target == "transformer"]
    decoder = [d for d in ad.compiled_graph_plans(declaration) if d.target == "decoder"]
    assert len(denoiser) == 2 and len(decoder) == 1


# ---------------------------------------------------------------------------
# The pgw#993 / pgw#994 shape — present, not merely intended
# ---------------------------------------------------------------------------


def _spec(target: str, cfg=None) -> ExportSpec:
    fork = () if cfg is None else (("cfg", cfg),)
    return ExportSpec(family=decl_mod.FAMILY, target=target, fork=fork,
                      class_dims=())


@pytest.mark.parametrize("cfg,arity", [(True, 2), (False, 1)])
def test_container_arity_follows_the_fork(declaration, cfg, arity) -> None:
    module = MicroDenoiser(MicroConfig())
    got = ad.container_arities(declaration, _spec("transformer", cfg), module)
    assert got == {"x": arity, "cond": arity}


def test_a_plain_input_sits_after_a_container_and_its_leaf_shifts(
    declaration,
) -> None:
    """THE pgw#994 shape, asserted structurally rather than believed.

    ``t`` is argument 1 of the traced call. Once ``x`` expands to two leaves,
    ``t``'s FLATTENED position is 2. A serve side that matched the contract's
    flattened position against pre-flattening args would bind the whole list
    to element 0 and shift ``t`` — which is the defect. This test fails the
    day the declaration stops reproducing that divergence.
    """
    module = MicroDenoiser(MicroConfig())
    args, kwargs = ad.declared_inputs(module, _spec("transformer", True),
                                      declaration)
    assert kwargs == {}
    assert isinstance(args[0], list) and len(args[0]) == 2, "x is a container"
    assert isinstance(args[1], torch.Tensor), "t is plain, and it is next"
    leaves = aot_flatten.flatten_call(("x", "t", "cond"), args, {})
    by_name = {leaf.name: leaf for leaf in leaves}
    t_leaf = by_name["t"]
    assert t_leaf.param_position == 1, "argument position"
    assert leaves.index(t_leaf) == 2, "flattened position — the divergence"
    assert by_name["x.0"].exported_name == "x_0"


def test_the_turbo_arm_is_a_one_element_container_not_a_bare_tensor(
    declaration,
) -> None:
    """A one-element container is ``x_0``, never ``x`` — the arity-1 case is
    where a "container support" implementation usually quietly degrades."""
    module = MicroDenoiser(MicroConfig())
    args, _ = ad.declared_inputs(module, _spec("transformer", False), declaration)
    assert isinstance(args[0], list) and len(args[0]) == 1
    leaves = aot_flatten.flatten_call(("x", "t", "cond"), args, {})
    assert leaves[0].exported_name == "x_0"


def test_every_declared_arm_runs_eagerly(declaration, tree) -> None:
    """The declared example feed is a LEGAL call of the real module — the
    cheapest possible catch for a declaration that drifted from its model."""
    pipe = MicroPipeline.from_pretrained(str(tree))
    for target, cfg in (("transformer", True), ("transformer", False),
                        ("decoder", None)):
        module = getattr(pipe, target)
        args, _ = ad.declared_inputs(module, _spec(target, cfg), declaration)
        with torch.no_grad():
            out = module(*args)
        assert out.shape[0] > 0


# ---------------------------------------------------------------------------
# The model properties the declaration DEPENDS on
# ---------------------------------------------------------------------------


def test_the_graph_is_conv_free(declaration) -> None:
    """``dynamic-collapse`` is ratified for conv-free graphs (#730). A conv
    added later would make the declared strategy wrong, silently."""
    assert declaration.shape_strategy == "dynamic-collapse"
    config = MicroConfig()
    for module in (MicroDenoiser(config), MicroDecoder(config)):
        convs = [type(m).__name__ for m in module.modules()
                 if isinstance(m, torch.nn.modules.conv._ConvNd)]
        assert convs == []


def test_the_frequency_table_is_a_buffer_not_a_plain_attribute() -> None:
    """pgw#857: a plain attribute is lifted as a graph LITERAL and its bytes
    ship inside every compiled graph."""
    embed = MicroDenoiser(MicroConfig()).time_embed
    assert "freqs" in dict(embed.named_buffers())
    assert "freqs" not in embed.__dict__


# ---------------------------------------------------------------------------
# Generated weights — the locality claim
# ---------------------------------------------------------------------------


def test_weights_are_deterministic_and_small(tree, tmp_path) -> None:
    again = materialize(tmp_path / "again")
    first, second = load_state(tree), load_state(again)
    assert sorted(first) == sorted(second)
    for name, tensor in first.items():
        assert torch.equal(tensor, second[name]), name
    size = tree_bytes(tree)
    assert 0 < size <= MAX_TREE_BYTES
    assert size < 5_000_000, f"{size} bytes — the family must stay tiny"


def test_regenerating_reproduces_the_written_bytes(tree) -> None:
    """The claim the Dockerfile's ``--verify`` step enforces at build time."""
    on_disk = load_state(tree)
    fresh = state_dict(MicroConfig(), seed=SEED)
    assert sorted(on_disk) == sorted(fresh)
    for name, tensor in fresh.items():
        assert torch.equal(tensor, on_disk[name]), name


# ---------------------------------------------------------------------------
# The endpoint declaration — sdxl's slot shape, which is pgw#969's shape
# ---------------------------------------------------------------------------


def test_the_slot_is_a_catalog_slot_with_no_code_default() -> None:
    spec = Generate.__gen_worker_endpoint__
    slot = spec.slots["pipeline"]
    assert slot.selected_by == "model"
    assert slot.default_checkpoint is None
    assert slot.pipeline_cls is MicroPipeline
    assert spec.resources.gpu is True


def test_both_fork_arms_have_a_served_method() -> None:
    """A declared arm no function reaches is an compiled graph nothing can ever call."""
    assert callable(Generate.generate)
    assert callable(Generate.generate_turbo)
    assert [s.value for s in Size] == ["256x256", "384x384"]
