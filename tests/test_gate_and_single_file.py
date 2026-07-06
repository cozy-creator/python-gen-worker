"""VRAM-gate GiB rounding + single-file checkpoint loading (#347 roster sweep).

A "24GB" card reports ~23.99GiB via torch.cuda.mem_get_info (driver/ECC
reserve); the availability gate must not disable functions declaring
``Resources(vram_gb=24)`` on exactly the card they target. Single-file repos
(Illustrious-XL, civitai checkpoints) have no ``model_index.json`` and must
route through ``cls.from_single_file``.
"""

from __future__ import annotations

from pathlib import Path

import msgspec

from gen_worker.api.binding import HF
from gen_worker.api.decorators import Resources
from gen_worker.executor import Executor
from gen_worker.models.loading import _single_file_checkpoint, load_from_pretrained
from gen_worker.registry import EndpointSpec


class _In(msgspec.Struct):
    x: str


class _Out(msgspec.Struct):
    y: str


def _spec(name: str, vram_gb: float) -> EndpointSpec:
    class Endpoint:
        def setup(self, model: str) -> None:  # pragma: no cover
            pass

        def run(self, ctx, payload: _In) -> _Out:  # pragma: no cover
            return _Out(y=payload.x)

    return EndpointSpec(
        name=name, method=Endpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint,
        attr_name="run", models={"model": HF("acme/tiny")},
        resources=Resources(vram_gb=vram_gb),
    )


async def _noop_send(msg) -> None:  # pragma: no cover
    pass


RTX_4090_TOTAL_BYTES = 25_757_220_864  # 23.99 GiB — what mem_get_info reports


def test_gate_rounds_detected_gib_up_to_declared_24() -> None:
    ex = Executor([_spec("fits-24", 24.0)], _noop_send)
    ex.gate_functions({"gpu_count": 1, "gpu_total_mem": RTX_4090_TOTAL_BYTES, "gpu_sm": "89"})
    assert "fits-24" not in ex.unavailable


def test_gate_still_blocks_genuinely_bigger_requirements() -> None:
    ex = Executor([_spec("needs-32", 32.0)], _noop_send)
    ex.gate_functions({"gpu_count": 1, "gpu_total_mem": RTX_4090_TOTAL_BYTES, "gpu_sm": "89"})
    assert ex.unavailable["needs-32"][0] == "insufficient_vram"


def test_single_file_checkpoint_detection(tmp_path: Path) -> None:
    ckpt = tmp_path / "Illustrious-XL-v1.0.safetensors"
    ckpt.write_bytes(b"stub")
    assert _single_file_checkpoint(tmp_path) == ckpt
    assert _single_file_checkpoint(ckpt) == ckpt

    # A diffusers layout is NOT a single-file checkpoint.
    (tmp_path / "model_index.json").write_text("{}")
    assert _single_file_checkpoint(tmp_path) is None


def test_single_file_checkpoint_requires_exactly_one(tmp_path: Path) -> None:
    (tmp_path / "a.safetensors").write_bytes(b"a")
    (tmp_path / "b.safetensors").write_bytes(b"b")
    assert _single_file_checkpoint(tmp_path) is None
    assert _single_file_checkpoint(tmp_path / "missing") is None


def test_load_from_pretrained_routes_single_file(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"stub")
    calls: dict[str, object] = {}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):  # pragma: no cover
            raise AssertionError("single-file snapshot must not use from_pretrained")

        @classmethod
        def from_single_file(cls, path, **kwargs):
            calls["path"] = path
            calls["kwargs"] = kwargs
            return cls()

    out = load_from_pretrained(FakePipeline, tmp_path, dtype="fp16")
    assert isinstance(out, FakePipeline)
    assert calls["path"] == str(ckpt)
    assert "variant" not in calls["kwargs"]


def test_load_from_pretrained_still_uses_pretrained_for_layouts(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text("{}")
    calls: dict[str, object] = {}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["path"] = path
            return cls()

        @classmethod
        def from_single_file(cls, path, **kwargs):  # pragma: no cover
            raise AssertionError("layout snapshot must not use from_single_file")

    out = load_from_pretrained(FakePipeline, tmp_path, dtype="bf16")
    assert isinstance(out, FakePipeline)
    assert calls["path"] == str(tmp_path)
