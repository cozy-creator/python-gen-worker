from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fake_hub import _FakeHub

from gen_worker.convert.produced import ProducedFlavor
from gen_worker.convert.publish import publish_flavors
from gen_worker.models import ladder

RELEASE = "2026.08"

EXPECTED: dict[str, dict[str, Any] | None] = {
    "": None,
    "base": None,
    "fp8": {"precision_class": "fp8"},
    "svdq-int4": {"precision_class": "svdq-int4"},
    "svdq-fp4": {"precision_class": "svdq-fp4"},
    "nvfp4": {"precision_class": "nvfp4"},
    "nvfp4-w4a4": {"precision_class": "nvfp4-w4a4"},
    "gguf": {"precision_class": "gguf"},
}

DELETED_ADMISSION_KEYS = ("sm_allowed", "sm_min", "engines")


class _Ctx:
    def __init__(self, base_url: str) -> None:
        self._file_api_base_url = base_url
        self._worker_capability_token = "cap-token"
        self.owner = "acme"

    def log(self, message: str, **fields: Any) -> None:
        pass


def _tree(tmp_path: Path, name: str) -> Path:
    out = tmp_path / name
    out.mkdir()
    (out / "diffusion.safetensors").write_bytes(b"\x11" * 2048)
    return out


def _publish(fake_hub: Any, tmp_path: Path, cls: str, **attrs: str) -> dict:
    ctx = _Ctx(f"http://127.0.0.1:{fake_hub.server_port}")
    if cls:
        attrs = {"precision_class": cls, **attrs}
    publish_flavors(
        ctx,
        [ProducedFlavor(
            path=_tree(tmp_path, "out"), attributes=dict(attrs),
        )],
        destination_repo="acme/qwen-image",
        release=RELEASE,
    )
    return dict(_FakeHub.state["publish_request"].get("metadata") or {})


@pytest.mark.parametrize("cls", sorted(EXPECTED))
def test_the_declared_block_is_precision_class_and_nothing_else(
    fake_hub: Any, tmp_path: Path, cls: str
) -> None:
    """Read off the declare request the fake hub actually receives: every laddered class carries its `precision_class`, and no class carries an admission key the hub deleted."""
    meta = _publish(fake_hub, tmp_path, cls)

    expected = EXPECTED[cls]
    if expected is None:
        assert "placement" not in meta, (
            f"{cls!r} is not a ladder rung; a block here restates the hub's "
            "own ClassBase fallback")
        return
    assert meta["placement"] == expected
    assert set(meta["placement"]) == {"precision_class"}, (
        "th#2055 deleted every reader but `precision_class`; a second key is "
        "unread JSON at best and a resurrected purchase veto at worst")


@pytest.mark.parametrize("cls", ["svdq-fp4", "svdq-int4", "nvfp4-w4a4", "fp8", "gguf"])
def test_no_admission_key_survives_at_the_wire(
    fake_hub: Any, tmp_path: Path, cls: str
) -> None:
    block = _publish(fake_hub, tmp_path, cls)["placement"]
    present = [k for k in DELETED_ADMISSION_KEYS if k in block]
    assert present == [], f"{cls!r} re-stamped deleted admission keys: {present}"


def test_the_dead_placement_override_attrs_are_dropped_not_published(
    fake_hub: Any, tmp_path: Path
) -> None:
    meta = _publish(
        fake_hub, tmp_path, "nvfp4-w4a4",
        placement_sm_allowed="89", placement_engines="nunchaku,triton",
    )

    assert meta["placement"] == {"precision_class": "nvfp4-w4a4"}
    assert not [k for k in meta if k.startswith("placement_")]


def test_every_exported_class_constant_is_in_the_vocabulary_and_back() -> None:
    """The self-consistency half, which no transcribed literal can state: the `CLASS_*` names `__all__` exports and the members of `PRECISION_CLASSES` are the SAME set."""
    exported_classes = {
        getattr(ladder, name) for name in ladder.__all__
        if name.startswith("CLASS_")
    }
    assert exported_classes == set(ladder.PRECISION_CLASSES)
    assert {n for n in vars(ladder) if n.startswith("CLASS_")} == {
        n for n in ladder.__all__ if n.startswith("CLASS_")
    }

