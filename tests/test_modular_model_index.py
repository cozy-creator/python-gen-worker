"""pgw#1481 — a MODULAR ``model_index.json`` must load, and must refuse BY NAME.

The article is a real diffusers pipeline saved by ``streaming_fixture``, whose
``model_index.json`` is then rewritten into the 3-element modular form that
``tensorhub/minimax-h3`` actually ships:

    "vae": ["diffusers", "AutoencoderKL",
            {"type_hint": [...], "pretrained_model_name_or_path": "...",
             "subfolder": "vae", "variant": null, "revision": null}]

Before the fix ``skeleton.build`` filtered on ``len(spec) != 2`` and
``continue``d, so every component vanished and the walk died on
``declares no nn.Module component`` — a sentence that is false when the index
declares nine, and that names none of them. The red arm here is that exact
mutation: revert the walker to a 2-element filter and ``test_modular_index_loads``
fails.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from gen_worker.serving.streaming import skeleton as sk  # noqa: E402
from gen_worker.serving.streaming.skeleton import SkeletonError  # noqa: E402
from streaming_fixture import build_source  # noqa: E402


def _modularize(source: Path, **overrides: object) -> None:
    """Rewrite a classic index in place as a modular one."""
    index_path = source / "model_index.json"
    index = json.loads(index_path.read_text())
    for name, spec in list(index.items()):
        if name.startswith("_") or not isinstance(spec, list) or len(spec) != 2:
            continue
        library, class_name = spec
        meta = {
            "type_hint": [library, class_name],
            "pretrained_model_name_or_path": "Org/Model",
            "subfolder": name,
            "variant": None,
            "revision": None,
        }
        meta.update(overrides)
        index[name] = [library, class_name, meta]
    index_path.write_text(json.dumps(index))


@pytest.fixture(scope="module")
def modular_source(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, type, dict[str, object]]:
    target = tmp_path_factory.mktemp("pgw1481")
    pipeline_cls = build_source(target)
    classic = json.loads((target / "model_index.json").read_text())
    _modularize(target)
    return target, pipeline_cls, classic


def test_modular_index_loads(modular_source: tuple[Path, type, dict[str, object]]) -> None:
    """Every component a modular index declares must reach the skeleton.

    THE RED ARM: restore ``len(spec) != 2: continue`` in ``skeleton.build`` and
    this fails with ``declares no nn.Module component``.
    """
    source, pipeline_cls, classic = modular_source
    declared = {k for k in classic if not k.startswith("_")}

    built = sk.build(pipeline_cls, source)

    reached = set(built.modules) | set(built.passthrough)
    assert reached == declared, (
        f"modular index declared {sorted(declared)}; skeleton reached "
        f"{sorted(reached)} — a component the walker drops is streamed nowhere"
    )
    assert built.modules, "no nn.Module component survived the modular walk"
    # Identity, not truthiness — the pgw#1410 fence's property, restated here so
    # a modular index cannot reintroduce the orphan it was written to catch.
    for name, module in built.modules.items():
        assert getattr(built.pipeline, name, None) is module


def test_relocating_subfolder_is_refused_by_name(tmp_path: Path) -> None:
    """A modular entry may not send a component somewhere else, silently."""
    source = tmp_path / "relocated"
    pipeline_cls = build_source(source)
    _modularize(source, subfolder="elsewhere")

    with pytest.raises(SkeletonError) as excinfo:
        sk.build(pipeline_cls, source)
    message = str(excinfo.value)
    assert "subfolder" in message
    assert "elsewhere" in message


@pytest.mark.parametrize("field, value", [("variant", "fp16"), ("revision", "abc123")])
def test_pinned_variant_or_revision_is_refused_by_name(
    tmp_path: Path, field: str, value: str
) -> None:
    """The projected tree carries ONE cut; a pin it cannot honour is refused."""
    source = tmp_path / f"pinned-{field}"
    pipeline_cls = build_source(source)
    _modularize(source, **{field: value})

    with pytest.raises(SkeletonError) as excinfo:
        sk.build(pipeline_cls, source)
    message = str(excinfo.value)
    assert field in message
    assert value in message


@pytest.mark.parametrize(
    "bad",
    [
        ["diffusers"],
        "nonsense",
        ["diffusers", "AutoencoderKL", "not-a-dict", "extra"],
    ],
    ids=["one-element", "scalar", "no-metadata-object"],
)
def test_unreadable_entry_names_itself(tmp_path: Path, bad: object) -> None:
    """An entry the walker cannot read is refused, and the message names it.

    This is the half that outlives minimax-h3: before pgw#1481 an unreadable
    entry was skipped, so N specific failures became one count-of-zero at the
    end of the loop.
    """
    source = tmp_path / "unreadable"
    pipeline_cls = build_source(source)
    _modularize(source)
    index_path = source / "model_index.json"
    index = json.loads(index_path.read_text())
    index["vae"] = bad
    index_path.write_text(json.dumps(index))

    with pytest.raises(SkeletonError) as excinfo:
        sk.build(pipeline_cls, source)
    assert "vae" in str(excinfo.value)
