"""pgw#1346 W1b-2 — a declared model becomes a SERVABLE instance.

This is the gap that blocked the `Slot` deletion: pgw#1332 landed the
declaration, the codegen and the backings, but nothing on a real pod ever
CONSTRUCTED an instance, so `@endpoint(models={...})` type-checked, published a
manifest, and handed the handler nothing. `set_instance_resolver`,
`bind_models` and `resolver_instances` had zero call sites in `src/`.

Everything below runs the real path: the real declaration, the real
declaration-time export, the real codegen, the real `Runner.component` lookup
over a real module tree, the real `Model.adopt`, and real typed runner calls
returning real tensors. The only thing not present is a GPU and a hub, and
neither is on this path.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest

from gen_worker.model.bind import Bind
from gen_worker.model.codegen import render_module
from gen_worker.model.errors import ModelError, ModelRefusal
from gen_worker.model.export import export_model
from gen_worker.model.residency import eager_modules, instance_for, instances_for
from gen_worker.model.snapshot import ModelExport

from harness.model_toys_pgw1332 import TOY_DIFFUSION, WIDTH, toy_loaded_tree

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def toy_export() -> ModelExport:
    return export_model(TOY_DIFFUSION)


@pytest.fixture(scope="module")
def toy_binding(toy_export: ModelExport, tmp_path_factory: pytest.TempPathFactory) -> Any:
    root = tmp_path_factory.mktemp("pgw1346_bindings")
    package = root / "pgw1346_generated"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "toy_diffusion.export.json").write_text(toy_export.dumps())
    (package / "toy_diffusion.py").write_text(
        render_module(
            toy_export,
            spec_module="harness.model_toys_pgw1332",
            spec_attr="TOY_DIFFUSION",
        )
    )
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("pgw1346_generated.toy_diffusion")
    finally:
        sys.path.remove(str(root))
    return module.ToyDiffusion


# ---------------------------------------------------------------------------
# Runner.component is the map, and it is the thing that was missing
# ---------------------------------------------------------------------------


def test_the_component_path_finds_every_declared_runner_in_a_loaded_tree() -> None:
    """The declaration says `transformer` and `vae.decoder`; the tree has them.

    A dotted path, because a component tree is a tree — the VAE's decoder is a
    submodule, and a flat name could not reach it.
    """

    modules = eager_modules(TOY_DIFFUSION, toy_loaded_tree())
    assert sorted(modules) == ["decoder", "denoiser"]


def test_a_runner_with_no_component_declared_simply_has_no_eager_module() -> None:
    """A legitimate state, not an error: a runner that only ever exists as a
    compiled graph class. It is absent from the eager backing, and calling it
    with no armed cell refuses BY NAME rather than running something else.
    """

    from dataclasses import replace

    graphs_only = replace(
        TOY_DIFFUSION,
        runners=tuple(replace(r, component="") for r in TOY_DIFFUSION.runners),
    )
    assert eager_modules(graphs_only, toy_loaded_tree()) == {}


def test_an_adopt_only_pod_has_no_declaration_and_that_is_not_a_failure() -> None:
    """`SPEC` is None where importing the declaration would acquire a model
    library the serve role forbids (pgw#1328). Such a pod serves compiled
    cells, so an empty eager map is the correct answer.
    """

    assert eager_modules(None, toy_loaded_tree()) == {}


# ---------------------------------------------------------------------------
# The instance is real, and its typed calls reach the loaded weights
# ---------------------------------------------------------------------------


def test_a_declared_model_becomes_an_instance_whose_typed_call_runs_the_loaded_module(
    toy_binding: Any,
) -> None:
    """The whole point of W1b-2, end to end.

    `inst.denoiser(...)` is the GENERATED typed callable; it reaches the
    `EagerBacking`, which reaches the module `Runner.component` found in the
    loaded tree, which is a real `nn.Linear`. The assertion is on the tensor
    that comes back, because a call that returned a shape-correct fake would
    pass a weaker test and is exactly what this must not be.
    """

    tree = toy_loaded_tree()
    inst = instance_for(toy_binding, ref="hub:toy/ckpt-a", tree=tree)

    assert inst.ref == "hub:toy/ckpt-a"
    tokens = 64 // 64
    hidden = torch.ones(1, tokens, WIDTH, dtype=torch.float32)
    out = inst.denoiser(
        resolution=64, hidden_states=hidden, timestep=torch.zeros((), dtype=torch.float32)
    )
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, tokens, WIDTH)
    # The eager module is the one from the tree — same object, so the same
    # weights. Proven by computing the expected value through it directly.
    expected = tree.transformer(hidden, torch.zeros((), dtype=torch.float32))
    assert torch.equal(out, expected)


def test_two_parameters_of_one_model_bind_two_checkpoints_independently(
    toy_binding: Any,
) -> None:
    """The axis split, on the real construction path: one model class, two
    parameters, two refs, two instances that do not share identity.
    """

    binds = {"left": Bind(toy_binding), "right": Bind(toy_binding)}
    tree = toy_loaded_tree()
    out = instances_for(
        binds,
        refs={"left": "hub:toy/a", "right": "hub:toy/b"},
        trees={"left": tree, "right": tree},
    )
    assert out["left"].ref == "hub:toy/a"
    assert out["right"].ref == "hub:toy/b"
    assert out["left"] is not out["right"]


def test_the_catalog_stamp_lands_on_the_instance_not_on_ctx(toy_binding: Any) -> None:
    """`inst.tuned` is the replacement for `ctx.defaults`, and it decodes the
    SAME wire field the retiring context surface reads
    (`ModelBinding.inference_defaults`), so the two cannot disagree.
    """

    inst = instance_for(
        toy_binding, ref="hub:toy/ckpt", tree=toy_loaded_tree(),
        stamped='{"steps": 7}',
    )
    assert inst.tuned.steps == 7


def test_a_malformed_catalog_stamp_refuses_rather_than_serving_neutral_values(
    toy_binding: Any,
) -> None:
    """tensorhub schema-validates at PUT time, so a decode failure here is real
    version skew — and serving a checkpoint with values nobody could parse is
    worse than refusing it.
    """

    with pytest.raises(ModelError) as caught:
        instance_for(
            toy_binding, ref="hub:toy/ckpt", tree=toy_loaded_tree(),
            stamped='{"steps": "not-a-number"}',
        )
    assert caught.value.reason is ModelRefusal.TUNED_INVALID


# ---------------------------------------------------------------------------
# The refusals, which are what stop a wrong render nobody can explain
# ---------------------------------------------------------------------------


def test_a_model_with_neither_an_eager_module_nor_an_armed_cell_refuses_by_name(
    toy_binding: Any,
) -> None:
    with pytest.raises(ModelError) as caught:
        instance_for(toy_binding, ref="hub:toy/ckpt", tree=None)
    assert caught.value.reason is ModelRefusal.BACKING_MISSING
    assert "toy_diffusion" in str(caught.value)
    assert "hub:toy/ckpt" in str(caught.value)


def test_a_parameter_the_request_names_no_checkpoint_for_is_refused(
    toy_binding: Any,
) -> None:
    """Never a default. A model bound to "whatever was resident" is the
    cross-request bleed the axis split exists to prevent.
    """

    with pytest.raises(ModelError) as caught:
        instances_for(
            {"toy": Bind(toy_binding)}, refs={}, trees={"toy": toy_loaded_tree()},
        )
    assert caught.value.reason is ModelRefusal.BACKING_MISSING
    assert "names no checkpoint" in str(caught.value)


def test_warm_up_skips_a_model_no_dispatch_has_named_yet(toy_binding: Any) -> None:
    """A warm pass runs before any dispatch names a checkpoint, so "no ref yet"
    is the expected state there. Skipping leaves the model un-warmed, which is
    honest; refusing would break the boot of every hub-resolved endpoint.
    """

    out = instances_for(
        {"toy": Bind(toy_binding)}, refs={}, trees={"toy": toy_loaded_tree()},
        skip_unresolved=True,
    )
    assert out == {}
