from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gen_worker.convert.clone import (
    CloneDiskSpaceError,
    OutputSpec,
    _preflight_disk,
)
from gen_worker.convert.ingest import plan_huggingface

GIB = 1024 ** 3
TRANSFORMER_2_BYTES = 57_154_175_562
SHARDS = 12
POD_FREE_BYTES = int(199.9 * GIB)


_WIDTH = {"F32": 4, "BF16": 2, "F16": 2}


def _safetensors_bytes(*tensors: tuple[str, int]) -> bytes:
    header: dict[str, Any] = {}
    offset = 0
    for i, (dtype, params) in enumerate(tensors):
        end = offset + params * _WIDTH[dtype]
        header[f"blocks.{i}.weight"] = {
            "dtype": dtype, "shape": [params], "data_offsets": [offset, end]}
        offset = end
    blob = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(blob)) + blob + b"\x00" * offset


def _wan22_remote(tmp_path: Path, dtype: str = "F32") -> Path:
    remote = tmp_path / "remote"
    (remote / "transformer_2").mkdir(parents=True)
    (remote / "model_index.json").write_text(json.dumps({
        "_class_name": "WanPipeline",
        "transformer_2": ["diffusers", "WanTransformer3DModel"],
    }), encoding="utf-8")
    (remote / "transformer_2" / "config.json").write_text(
        json.dumps({"_class_name": "WanTransformer3DModel"}), encoding="utf-8")
    for i in range(1, SHARDS + 1):
        name = f"diffusion_pytorch_model-{i:05d}-of-{SHARDS:05d}.safetensors"
        (remote / "transformer_2" / name).write_bytes(
            _safetensors_bytes((dtype, 4)))
    return remote


def _fake_hf(remote: Path, *, headers_readable: bool = True) -> Any:

    def _files() -> list[Path]:
        return sorted(p for p in remote.rglob("*") if p.is_file())

    def _reported_size(rel: str, real: int) -> int:
        if rel.startswith("transformer_2/") and rel.endswith(".safetensors"):
            return TRANSFORMER_2_BYTES // SHARDS
        return real

    class _Api:
        def __init__(self, token: str | None = None) -> None:
            self.token = token

        def repo_info(self, repo_id: str, revision: str | None = None) -> Any:
            return SimpleNamespace(sha="5be7df96" + "0" * 32)

        def list_repo_tree(self, repo_id: str, revision: str | None = None,
                           recursive: bool = False) -> Any:
            for p in _files():
                rel = p.relative_to(remote).as_posix()
                yield SimpleNamespace(
                    path=rel, size=_reported_size(rel, p.stat().st_size),
                    lfs=SimpleNamespace(sha256=f"{abs(hash(rel)):064x}"[:64]),
                    blob_id="")

        def parse_safetensors_file_metadata(
            self, repo_id: str, filename: str, *, revision: str | None = None,
            repo_type: str | None = None, token: str | None = None,
        ) -> Any:
            """The real range-read call, served off the local header bytes."""
            if not headers_readable:
                raise OSError("header unreadable")
            raw = (remote / filename).read_bytes()
            (n,) = struct.unpack("<Q", raw[:8])
            header = json.loads(raw[8:8 + n])
            counts: dict[str, int] = {}
            for value in header.values():
                if not isinstance(value, dict) or "dtype" not in value:
                    continue
                params = 1
                for dim in value.get("shape") or []:
                    params *= int(dim)
                counts[str(value["dtype"])] = (
                    counts.get(str(value["dtype"]), 0) + params)
            return SimpleNamespace(parameter_count=counts, metadata={})

    def _download(repo_id: str, filename: str, revision: str | None = None,
                  token: str | None = None) -> str:
        return str(remote / filename)

    return SimpleNamespace(
        HfApi=_Api, hf_hub_download=_download,
        snapshot_download=lambda *a, **k: "")


def _plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kw: Any) -> Any:
    remote = _wan22_remote(tmp_path, dtype=kw.pop("dtype", "F32"))
    monkeypatch.setattr("gen_worker.convert.ingest.hf",
                        lambda: _fake_hf(remote, **kw))
    return plan_huggingface(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        source_include=["model_index.json", "transformer_2/*"],
    )


def _with_free(monkeypatch: pytest.MonkeyPatch, free: int) -> None:
    monkeypatch.setattr(
        "gen_worker.convert.clone.shutil.disk_usage",
        lambda _p: shutil._ntuple_diskusage(  # type: ignore[attr-defined]
            total=free * 2, used=free, free=free))


BF16 = OutputSpec(dtype="bf16", file_layout="multi-file", file_type="safetensors")


def test_untagged_fp32_source_resolves_its_dtype_at_plan_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No filename says fp32 — the safetensors HEADER does, and the plan reads it."""
    plan = _plan(tmp_path, monkeypatch)

    assert plan.classification.attrs["dtype"] == "fp32"


def test_bf16_cast_of_the_live_wan22_leg_is_not_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request 3190bc31: 53.2 GiB of untagged fp32 -> bf16 on a 200 GB pod."""
    plan = _plan(tmp_path, monkeypatch)
    _with_free(monkeypatch, POD_FREE_BYTES)

    _preflight_disk(tmp_path, plan, [BF16])


def test_the_estimate_is_source_plus_half_plus_margin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bracket the bound: a bf16 output of an fp32 source is HALF the source, so ~82 GiB is required and ~81 GiB is not enough."""
    plan = _plan(tmp_path, monkeypatch)
    source_bytes = sum(size for _, size, _ in plan.bank_files())
    required = source_bytes + -(-source_bytes // 2) + 2 * GIB
    assert 81 * GIB < required < 82 * GIB

    _with_free(monkeypatch, required)
    _preflight_disk(tmp_path, plan, [BF16])

    _with_free(monkeypatch, required - 1)
    with pytest.raises(CloneDiskSpaceError) as excinfo:
        _preflight_disk(tmp_path, plan, [BF16])
    assert "need ~81.8 GiB" in str(excinfo.value)


def test_an_unreadable_header_assumes_the_widest_dense_width_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no header can be read the preflight still has to guess, and the guess is 32 bits — the direction whose failure is a late, loud, retryable ENOSPC rather than an early, permanent refusal of a jo..."""
    plan = _plan(tmp_path, monkeypatch, headers_readable=False)
    assert not plan.classification.attrs.get("dtype")

    _with_free(monkeypatch, POD_FREE_BYTES)
    _preflight_disk(tmp_path, plan, [BF16])

    _with_free(monkeypatch, 40 * GIB)
    with pytest.raises(CloneDiskSpaceError) as excinfo:
        _preflight_disk(tmp_path, plan, [BF16])
    assert "source dtype unreadable, assumed 32-bit" in str(excinfo.value)


def test_a_mixed_dtype_tree_is_sized_by_bits_per_parameter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Half fp32, half fp16 by BYTES."""
    remote = _wan22_remote(tmp_path)
    for p in sorted((remote / "transformer_2").glob("*.safetensors")):
        p.write_bytes(_safetensors_bytes(("F32", 4), ("F16", 8)))
    monkeypatch.setattr("gen_worker.convert.ingest.hf", lambda: _fake_hf(remote))

    plan = plan_huggingface(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        source_include=["model_index.json", "transformer_2/*"],
    )
    assert plan.source_storage_bits == 21

    source_bytes = sum(size for _, size, _ in plan.bank_files())
    _with_free(monkeypatch, int(source_bytes * 1.77) + 2 * GIB)
    _preflight_disk(tmp_path, plan, [BF16])

    _with_free(monkeypatch, int(source_bytes * 1.7) + 2 * GIB)
    with pytest.raises(CloneDiskSpaceError):
        _preflight_disk(tmp_path, plan, [BF16])


def test_a_tagged_source_still_wins_over_the_header_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stamp only fills a GAP: a source the variant-tag heuristic already resolved is untouched, so no existing classification changes shape."""
    remote = _wan22_remote(tmp_path, dtype="BF16")
    for p in sorted((remote / "transformer_2").glob("*.safetensors")):
        p.rename(p.with_name(p.name.replace(".safetensors", ".fp16.safetensors")))
    monkeypatch.setattr("gen_worker.convert.ingest.hf", lambda: _fake_hf(remote))

    plan = plan_huggingface(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        source_include=["model_index.json", "transformer_2/*"],
    )

    assert plan.classification.attrs["dtype"] == "fp16"
