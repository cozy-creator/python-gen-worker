"""gw#534 rung 2 — "fp8 download, bf16 resident".

W8A16 layerwise casting is never voluntary: a planned fp8 storage lane
(stored #fp8 flavor or resolved cast) is upgraded to plain bf16-resident
weights when the snapshot fits free VRAM with headroom. Hooks remain only
when bf16 does not fit. The traced weight lane keys the compile cache
(lane_drift): bf16-resident pipelines must never adopt hook-cast graphs.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from gen_worker.compile_cache import artifact_metadata, lane_drift
from gen_worker.models.loading import (
    bf16_resident_fits,
    load_from_pretrained,
    pipeline_weight_lane,
)


class _FakeDenoiser:
    def __init__(self) -> None:
        self.casting_calls: list = []

    def parameters(self):
        return iter(())

    def enable_layerwise_casting(self, *, storage_dtype: Any, compute_dtype: Any) -> None:
        self.casting_calls.append((storage_dtype, compute_dtype))


class _Pipe:
    calls: list = []

    def __init__(self) -> None:
        self.transformer = _FakeDenoiser()
        self.text_encoder = _FakeDenoiser()

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> Any:
        cls.calls.append(kwargs)
        return cls()


def _write_safetensors(path: Path, dtype: str, nbytes: int) -> None:
    header = json.dumps(
        {"w": {"dtype": dtype, "shape": [nbytes], "data_offsets": [0, nbytes]}}
    ).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)


def _snapshot(tmp_path: Path, dtype: str = "F8_E4M3", nbytes: int = 1024,
              te_dtype: str = "", te_nbytes: int = 0) -> Path:
    index: dict = {"_class_name": "Pipe", "transformer": ["diffusers", "X"]}
    if te_nbytes:
        index["text_encoder"] = ["transformers", "Y"]
    (tmp_path / "model_index.json").write_text(json.dumps(index))
    (tmp_path / "transformer").mkdir(exist_ok=True)
    _write_safetensors(
        tmp_path / "transformer" / "diffusion_pytorch_model.safetensors",
        dtype, nbytes)
    if te_nbytes:
        (tmp_path / "text_encoder").mkdir(exist_ok=True)
        _write_safetensors(
            tmp_path / "text_encoder" / "model.safetensors", te_dtype, te_nbytes)
    return tmp_path


@pytest.fixture
def vram(monkeypatch: pytest.MonkeyPatch):
    def _set(gb: float) -> None:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: gb > 0)
        from gen_worker.models import memory

        monkeypatch.setattr(memory, "get_available_vram_gb", lambda *a, **k: gb)

    return _set


def test_stored_fp8_upgrades_to_bf16_resident(vram, tmp_path: Path) -> None:
    """A stored #fp8 flavor on a roomy card serves bf16-RESIDENT: the small
    download stays, the cast hooks (the +44%/+73% per-forward tax) do not."""
    import torch

    vram(24.0)
    snap = _snapshot(tmp_path, "F8_E4M3", 2 << 30)
    _Pipe.calls = []
    pipe = load_from_pretrained(_Pipe, snap)
    (kwargs,) = _Pipe.calls
    assert kwargs["torch_dtype"] is torch.bfloat16
    assert pipe.transformer.casting_calls == []
    assert pipe._cozy_weight_lane == "bf16-resident"
    assert pipeline_weight_lane(pipe) == ""  # traces as plain bf16


def test_stored_fp8_keeps_hooks_when_bf16_cannot_fit(vram, tmp_path: Path) -> None:
    """8GB fp8 weights on a 10GB-free card: bf16-resident would be ~16GB —
    the involuntary W8A16 rung keeps fp8 bytes resident via hooks."""
    import torch

    vram(10.0)
    snap = _snapshot(tmp_path, "F8_E4M3", 8 << 30)
    _Pipe.calls = []
    pipe = load_from_pretrained(_Pipe, snap)
    ((storage, _compute),) = pipe.transformer.casting_calls
    assert storage is torch.float8_e4m3fn
    assert pipeline_weight_lane(pipe) == "fp8-hooks"


def test_requested_cast_upgrades_on_roomy_card(vram, tmp_path: Path) -> None:
    """An explicit storage_dtype="fp8" over a small bf16 snapshot is upgraded
    too — the cast is a fit lever, never a preference (Paul 2026-07-13)."""
    vram(24.0)
    snap = _snapshot(tmp_path, "BF16", 2 << 30)
    _Pipe.calls = []
    pipe = load_from_pretrained(_Pipe, snap, dtype="bf16", storage_dtype="fp8")
    assert pipe.transformer.casting_calls == []
    assert pipe._cozy_weight_lane == "bf16-resident"
    assert not getattr(pipe, "_cozy_fp8_storage_requested", False)


def test_no_cuda_keeps_todays_path(vram, tmp_path: Path) -> None:
    vram(0.0)
    snap = _snapshot(tmp_path, "F8_E4M3", 1 << 20)
    _Pipe.calls = []
    pipe = load_from_pretrained(_Pipe, snap)
    assert len(pipe.transformer.casting_calls) == 1
    assert pipeline_weight_lane(pipe) == "fp8-hooks"


def test_fp8_te_counts_text_encoders_in_the_estimate(vram, tmp_path: Path) -> None:
    """fp8+te: the upcast estimate covers the text encoders too — 4+2GB fp8
    doubles to 12GB resident, over a 13GB card's 9GB budget -> hooks stay."""
    vram(13.0)
    snap = _snapshot(tmp_path, "F8_E4M3", 4 << 30, te_dtype="F8_E4M3",
                     te_nbytes=2 << 30)
    _Pipe.calls = []
    pipe = load_from_pretrained(_Pipe, snap, storage_dtype="fp8+te")
    assert len(pipe.transformer.casting_calls) == 1
    assert len(pipe.text_encoder.casting_calls) == 1
    assert bf16_resident_fits(snap, text_encoders=True, free_gb=13.0) is False
    assert bf16_resident_fits(snap, text_encoders=True, free_gb=17.0) is True


def test_bf16_resident_fits_boundaries(tmp_path: Path) -> None:
    snap = _snapshot(tmp_path, "F8_E4M3", 2 << 30)  # 2GB fp8 -> 4GB resident
    assert bf16_resident_fits(snap, free_gb=0.0) is False
    assert bf16_resident_fits(snap, free_gb=7.9) is False  # 4 > 7.9 - 4
    assert bf16_resident_fits(snap, free_gb=8.1) is True
    (tmp_path / "b").mkdir()
    bf16 = _snapshot(tmp_path / "b", "BF16", 2 << 30)
    assert bf16_resident_fits(bf16, free_gb=6.5) is True  # no doubling


# --------------------------------------------------------------------------
# compile-cache lane parity (gw#534)
# --------------------------------------------------------------------------


def test_lane_drift_is_symmetric() -> None:
    hooked = _Pipe()
    hooked.transformer._cozy_fp8_storage_applied = True
    plain = _Pipe()
    bf16_meta = artifact_metadata(family="f")
    hook_meta = artifact_metadata(family="f", weight_lane="fp8-hooks")
    assert bf16_meta["weight_lane"] == ""
    assert lane_drift(bf16_meta, plain) == ""
    assert lane_drift(hook_meta, hooked) == ""
    assert "weight_lane" in lane_drift(bf16_meta, hooked)
    assert "weight_lane" in lane_drift(hook_meta, plain)


def test_lane_drift_bf16_resident_matches_plain_cells() -> None:
    pipe = _Pipe()
    pipe._cozy_weight_lane = "bf16-resident"
    assert lane_drift(artifact_metadata(family="f"), pipe) == ""
    assert "weight_lane" in lane_drift(
        artifact_metadata(family="f", weight_lane="fp8-hooks"), pipe)


def test_declared_envelope_gates_the_upgrade(tmp_path: Path) -> None:
    """ie#381: the upgrade must leave the DECLARED activation envelope
    intact — free must exceed declared_vram + the upcast's extra weight
    bytes, or the model that fit by design starts serving degraded."""
    snap = _snapshot(tmp_path, "F8_E4M3", 2 << 30)  # 2GB fp8 -> +2GB upcast
    # weights-margin alone passes at 79 GB free (4GB resident + 4 <= 79)…
    assert bf16_resident_fits(snap, free_gb=79.0) is True
    # …but a 78GB declared envelope needs 78 + 2 = 80 free: refuse.
    assert bf16_resident_fits(snap, free_gb=79.0, declared_vram_gb=78) is False
    # roomy card (B200-class): 140 + 2 <= 178 — upgrade stands.
    assert bf16_resident_fits(snap, free_gb=178.0, declared_vram_gb=140) is True
    # declared unknown (0): old rule only.
    assert bf16_resident_fits(snap, free_gb=79.0, declared_vram_gb=0) is True


def test_load_honors_declared_vram(vram, tmp_path: Path, monkeypatch) -> None:
    """load_from_pretrained(declared_vram_gb=…) reaches the fits check: an
    fp8+te load that would upgrade on a roomy card keeps hooks when the
    declared envelope forbids it."""
    snap = _snapshot(tmp_path, "F8_E4M3", 1 << 20, te_dtype="F8_E4M3",
                     te_nbytes=1 << 20)
    vram(79.0)
    _Pipe.calls.clear()
    pipe = load_from_pretrained(
        _Pipe, snap, storage_dtype="fp8+te", declared_vram_gb=200.0)
    assert pipeline_weight_lane(pipe) == "fp8-hooks"
    pipe2 = load_from_pretrained(
        _Pipe, snap, storage_dtype="fp8+te", declared_vram_gb=20.0)
    assert pipeline_weight_lane(pipe2) == ""  # upgraded: 20 + ~0 <= 79


def _write_mixed_safetensors(path: Path, fp8_bytes: int, bf16_tensors: int,
                             bf16_bytes_each: int) -> None:
    """One shard: a single big F8_E4M3 weight + many small BF16 scale/norm
    tensors (majority by COUNT is bf16, majority by BYTES is fp8 — the
    produced-flavor layout, ie#381)."""
    entries = {"w": {"dtype": "F8_E4M3", "shape": [fp8_bytes],
                     "data_offsets": [0, fp8_bytes]}}
    off = fp8_bytes
    for i in range(bf16_tensors):
        entries[f"s{i}"] = {"dtype": "BF16", "shape": [bf16_bytes_each],
                            "data_offsets": [off, off + bf16_bytes_each]}
        off += bf16_bytes_each
    header = json.dumps(entries).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)


def test_mixed_dtype_fp8_bytes_still_count(tmp_path: Path) -> None:
    """ie#381 fix 2: a produced fp8 flavor stores scales/norms in bf16 —
    majority-by-COUNT says "bf16" but the upcast doubles the fp8 WEIGHT
    bytes. The fits check must count them (the majority-dtype gate counted
    zero and upgraded LTX into its own activation budget)."""
    from gen_worker.models.loading import snapshot_component_fp8_bytes

    snap = tmp_path
    (snap / "model_index.json").write_text(json.dumps(
        {"_class_name": "Pipe", "transformer": ["diffusers", "X"]}))
    (snap / "transformer").mkdir()
    _write_mixed_safetensors(
        snap / "transformer" / "diffusion_pytorch_model.safetensors",
        fp8_bytes=3 << 30, bf16_tensors=200, bf16_bytes_each=1 << 20)
    # majority label is bf16 (200 tensors vs 1) yet fp8 bytes = 3GB
    fp8 = snapshot_component_fp8_bytes(snap)
    assert fp8.get("transformer", 0) == 3 << 30
    # total ~3.2GB + upcast 3GB = ~6.2GB: fits at 12GB free, not at 9.9GB
    assert bf16_resident_fits(snap, free_gb=12.0) is True
    assert bf16_resident_fits(snap, free_gb=9.9) is False
    # envelope term composes: declared 8 + upcast 3 = 11 > 10.5 free
    assert bf16_resident_fits(snap, free_gb=10.5, declared_vram_gb=8) is False
