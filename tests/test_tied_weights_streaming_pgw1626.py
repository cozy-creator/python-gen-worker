from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from gen_worker._vendor.tensorfs import LocalCAS, project_snapshot  # noqa: E402
from cas_fixture import ingest_repository  # noqa: E402
from gen_worker.models.projection import REF_PREFIX, SNAPSHOTS_DIR  # noqa: E402
from gen_worker.serving.streaming import NameMismatch, engine_for  # noqa: E402
from gen_worker.serving.streaming import skeleton as skeleton_mod  # noqa: E402
from streaming_fixture import (  # noqa: E402
    Lane,
    assert_byte_equal,
    build_tied_source,
)

ALIAS = "encoder.embed_tokens.weight"
SOURCE = "shared.weight"


def _project(base: Path, source: Path, key: str) -> Path:
    cas = LocalCAS(base)
    manifest = ingest_repository(cas, source)
    cas.compare_and_swap_ref(
        REF_PREFIX + key, cas.store_manifest(manifest), expected=None
    )
    tree = base / SNAPSHOTS_DIR / key
    project_snapshot(cas, manifest, tree)
    return tree


@pytest.fixture(scope="module")
def article(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    base = tmp_path_factory.mktemp("pgw1626")
    source = base / "source-model"
    pipeline_cls = build_tied_source(source)
    tree = _project(base, source, key="e" * 64)
    return {"base": base, "source": source, "tree": tree,
            "pipeline_cls": pipeline_cls}


def _container(root: Path) -> Path:
    found = sorted((root / "text_encoder").glob("*.safetensors"))
    assert len(found) == 1, found
    return found[0]


def _keys(path: Path) -> Tuple[str, ...]:
    raw = path.read_bytes()
    (size,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + size])
    return tuple(sorted(key for key in header if key != "__metadata__"))


def test_the_article_really_is_a_tied_checkpoint(article: Dict[str, Any]) -> None:
    """The guard on the guard: a T5 container that happened to carry BOTH names would make every assertion below pass on a lie."""
    keys = _keys(_container(article["source"]))
    assert SOURCE in keys, keys
    assert ALIAS not in keys, (
        f"the fixture's T5 container carries {ALIAS}; a real T5 checkpoint "
        f"never does, so this article cannot witness the incident"
    )


def test_a_tied_t5_checkpoint_loads_clean(article: Dict[str, Any]) -> None:
    """The incident's exact shape, green: zero survivors, and the alias is the SAME TENSOR as its source — a tie, not a copy."""
    engine = engine_for(article["tree"], device="cpu")
    assert engine is not None
    pipeline = engine.build(
        article["pipeline_cls"], checkpoint_dir=article["tree"], lane=Lane()
    )

    text_encoder = pipeline.text_encoder
    assert skeleton_mod.meta_survivors(text_encoder) == ()
    assert text_encoder.encoder.embed_tokens.weight is text_encoder.shared.weight, (
        "the alias holds its own tensor — the weights were duplicated in "
        "memory rather than tied, which doubles the embedding's VRAM"
    )
    for component in ("unet", "vae"):
        assert skeleton_mod.meta_survivors(getattr(pipeline, component)) == ()
    assert assert_byte_equal(pipeline, article["source"]) > 50


def test_an_absent_tensor_under_a_tied_name_is_still_refused(
    article: Dict[str, Any], tmp_path: Path
) -> None:
    """Dropping `shared.weight` leaves BOTH tied names unfilled."""
    source = tmp_path / "source-model"
    shutil.copytree(article["source"], source)
    _drop_named_tensor(_container(source), SOURCE)
    tree = _project(tmp_path, source, key="f" * 64)

    engine = engine_for(tree, device="cpu")
    assert engine is not None
    with pytest.raises(NameMismatch) as caught:
        engine.build(article["pipeline_cls"], checkpoint_dir=tree, lane=Lane())

    message = str(caught.value)
    assert "still on meta" in message, message
    assert SOURCE in message and ALIAS in message, message
    assert "the checkpoint does not carry" not in message, message
    assert "T5EncoderModel" in message, message
    assert "TIES to another parameter" in message, message


def test_without_the_retie_the_incident_reproduces(
    article: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delete the fix and test 1 dies exactly as the pod did."""
    monkeypatch.setattr(skeleton_mod, "retie", lambda module: False)

    engine = engine_for(article["tree"], device="cpu")
    assert engine is not None
    with pytest.raises(NameMismatch) as caught:
        engine.build(
            article["pipeline_cls"], checkpoint_dir=article["tree"], lane=Lane()
        )
    message = str(caught.value)
    assert "text_encoder" in message and ALIAS in message, message


def test_retie_is_a_no_op_on_a_component_that_cannot_tie(
    article: Dict[str, Any]
) -> None:
    """A diffusers `ModelMixin` exposes no `tie_weights` and needs none; the hasattr guard must report that rather than raise."""
    built = skeleton_mod.build(article["pipeline_cls"], article["tree"])
    assert skeleton_mod.retie(built.modules["unet"]) is False
    assert skeleton_mod.retie(built.modules["text_encoder"]) is True
    assert skeleton_mod.tied_names(built.modules["text_encoder"]) == (ALIAS,)
    assert skeleton_mod.tied_names(built.modules["unet"]) == ()


def _drop_named_tensor(path: Path, victim: str) -> None:
    raw = path.read_bytes()
    (size,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + size])
    body = raw[8 + size :]
    assert victim in header, sorted(header)
    header.pop(victim)

    rebuilt = bytearray()
    for name in sorted(
        (key for key in header if key != "__metadata__"),
        key=lambda key: header[key]["data_offsets"][0],
    ):
        start, end = header[name]["data_offsets"]
        header[name]["data_offsets"] = [len(rebuilt), len(rebuilt) + (end - start)]
        rebuilt += body[start:end]
    blob = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(rebuilt))
