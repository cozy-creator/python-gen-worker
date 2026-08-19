"""pgw#1506: the provenance walk must reach a pipeline inside an ENGINE.

`_named_marked_modules` looked exactly one level deep -- `vars(instance)` plus
`.components` on each value -- so an endpoint whose model owns an engine that
owns the pipeline could not name its own compiled denoiser:

    error: derive: a module marked via ctx.compile()
    (MiniMaxH3Transformer3DModel) is not reachable as a model attribute or
    pipeline component; the release document cannot name its provenance

Measured on minimax-h3 by introspecting the live instance: the DiT is at
`model.engine.pipeline.components['transformer']`, depth 2, and the pipeline
is internally consistent (attribute and registry are one object) -- so this is
reachability FROM THE MODEL, not an attachment failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
import gen_worker._vendor.torchcg  # noqa: E402,F401
import torch  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "release_fixtures"

LOCK = (
    "version = 1\n"
    '\n[[package]]\nname = "torch"\nversion = "2.13.0"\n'
    '\n[[package]]\nname = "triton"\nversion = "3.7.1"\n'
    '\n[[package]]\nname = "nvidia-cublas"\nversion = "13.1.1.3"\n'
    '\n[[package]]\nname = "diffusers"\nversion = "0.39.0"\n'
)


@pytest.fixture(scope="module")
def config_only_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sys.path.insert(0, str(FIXTURES))
    try:
        import tiny_tree
    finally:
        sys.path.remove(str(FIXTURES))
    return tiny_tree.save_config_only(tmp_path_factory.mktemp("engine-config-only"))


def test_a_pipeline_inside_an_engine_names_its_denoiser_like_any_other(
    config_only_tree: Path, tmp_path: Path
) -> None:
    """End to end through the CLI: the depth-2 DiT is named `unet`.

    The bare component name, exactly as a depth-1 endpoint gets -- the wrapper
    changes where the module LIVES, never what the graph is called.
    """

    from gen_worker.cli import main

    out = tmp_path / "release.json"
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text(LOCK)
    assert main([
        "release", "derive",
        "--dir", str(FIXTURES),
        "--module", "engine_wrapper_endpoint",
        "--checkpoint", str(config_only_tree),
        "--lockfile", str(lockfile),
        "--out", str(out),
    ]) == 0

    (lane,) = json.loads(out.read_bytes())["graphs"]["lanes"]
    assert lane["unobserved_targets"] == []
    assert {record["target"] for record in lane["graphs"]} == {"unet"}


class _Pipe:
    def __init__(self, **components: Any) -> None:
        self._components = components
        for name, value in components.items():
            setattr(self, name, value)

    @property
    def components(self) -> dict[str, Any]:
        return dict(self._components)


class _Model:
    pass


def _named(instance: Any, marked: list[Any]) -> dict[str, Any]:
    from gen_worker.release.derive import _named_marked_modules

    return _named_marked_modules(instance, marked)


def test_a_cyclic_object_graph_terminates() -> None:
    """An engine that back-references its model is an ordinary shape."""

    from gen_worker.release.derive import DeriveError  # noqa: F401

    unet = torch.nn.Linear(2, 2)
    model = _Model()
    engine = _Model()
    engine.model = model  # type: ignore[attr-defined]
    engine.pipeline = _Pipe(unet=unet)  # type: ignore[attr-defined]
    model.engine = engine  # type: ignore[attr-defined]

    assert _named(model, [unet]) == {"unet": unet}


def test_a_module_nested_BELOW_the_bound_refuses_and_says_where_it_stopped() -> None:
    """Depth-bounded, and the bound is NAMED rather than read as absent."""

    from gen_worker.release.derive import PROVENANCE_MAX_DEPTH, DeriveError

    unet = torch.nn.Linear(2, 2)
    node: Any = _Pipe(unet=unet)
    for _ in range(PROVENANCE_MAX_DEPTH + 2):
        wrapper = _Model()
        wrapper.inner = node  # type: ignore[attr-defined]
        node = wrapper
    model = _Model()
    model.engine = node  # type: ignore[attr-defined]

    with pytest.raises(DeriveError) as refusal:
        _named(model, [unet])
    assert "stopped at depth" in str(refusal.value)
    assert "nested deeper than the derive will look" in str(refusal.value)


def test_the_same_bare_name_at_two_paths_takes_the_DOTTED_spelling() -> None:
    """The existing preference order, asked across the whole instance.

    Two different denoisers both called `unet` must not collapse onto one
    name -- each keeps its full path, so both are nameable and neither is
    picked over the other.
    """

    first = torch.nn.Linear(2, 2)
    second = torch.nn.Linear(2, 2)
    model = _Model()
    model.a = _Pipe(unet=first)  # type: ignore[attr-defined]
    engine = _Model()
    engine.pipeline = _Pipe(unet=second)  # type: ignore[attr-defined]
    model.b = engine  # type: ignore[attr-defined]

    named = _named(model, [first, second])
    assert named == {"a.unet": first, "b.pipeline.unet": second}


def test_two_marks_that_collapse_onto_ONE_name_refuse_by_name() -> None:
    """The ambiguity the walk cannot resolve is refused, never guessed.

    A module reachable BOTH as a bare attribute and as the unique component
    of a pipeline held under that same attribute name: the shortest-name
    preference gives them one spelling, and two graphs cannot share one
    provenance row.
    """

    from gen_worker.release.derive import DeriveError

    first = torch.nn.Linear(2, 2)
    second = torch.nn.Linear(2, 2)
    model = _Model()
    # Both marked modules are offered the SAME bare name: `second` as the
    # unique `unet` component, `first` as the attribute literally called
    # `unet`. Equal length, so neither displaces the other.
    model.unet = first  # type: ignore[attr-defined]
    model.pipe = _Pipe(unet=second)  # type: ignore[attr-defined]

    with pytest.raises(DeriveError) as refusal:
        _named(model, [first, second])
    assert "both resolve to provenance name" in str(refusal.value)
    assert "cannot pick between them" in str(refusal.value)


def test_a_module_that_is_nowhere_on_the_instance_still_refuses() -> None:
    """The original refusal survives -- the walk got wider, not permissive."""

    from gen_worker.release.derive import DeriveError

    model = _Model()
    model.pipe = _Pipe(unet=torch.nn.Linear(2, 2))  # type: ignore[attr-defined]

    with pytest.raises(DeriveError) as refusal:
        _named(model, [torch.nn.Linear(2, 2)])
    assert "cannot name its provenance" in str(refusal.value)
