"""gw#462 clone scratch hygiene + disk preflight (real filesystem, real flocks).

The J24 qwen postmortem: a 20GB conversion pod ENOSPC-died mid-download with
no preflight, and failed-clone workdirs accumulated forever.
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import struct
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gen_worker.convert.clone import (
    CloneDiskSpaceError,
    _clone_workdir,
    _preflight_disk,
    _stage_oversize_safetensors,
    _sweep_stale_workdirs,
    normalize_outputs,
    run_clone,
)


class _Plan:
    def __init__(
        self,
        sizes: list[int],
        *,
        strategy: str = "",
        attrs: dict[str, str] | None = None,
        paths: list[str] | None = None,
        provider: str = "",
    ) -> None:
        self._sizes = sizes
        self._paths = paths or [f"f{i}.safetensors" for i in range(len(sizes))]
        self.classification = SimpleNamespace(strategy=strategy, attrs=attrs or {})
        self.provider = provider
        self.paths = self._paths

    def bank_files(self):
        return [
            (path, size, f"cid{i}") for i, (path, size) in enumerate(zip(self._paths, self._sizes))
        ]


def _specs(
    dtype: str = "bf16",
    layout: str = "diffusers",
    file_type: str = "safetensors",
):
    return normalize_outputs(
        [
            {
                "dtype": dtype,
                "file_layout": layout,
                "file_type": file_type,
            }
        ]
    )


def _z_image_plan() -> _Plan:
    files = [
        ("README.md", 6_015),
        ("model_index.json", 467),
        ("scheduler/scheduler_config.json", 173),
        ("text_encoder/config.json", 726),
        ("text_encoder/generation_config.json", 239),
        ("text_encoder/model-00001-of-00003.safetensors", 3_957_900_840),
        ("text_encoder/model-00002-of-00003.safetensors", 3_987_450_520),
        ("text_encoder/model-00003-of-00003.safetensors", 99_630_640),
        ("text_encoder/model.safetensors.index.json", 32_819),
        ("tokenizer/merges.txt", 1_671_853),
        ("tokenizer/tokenizer.json", 11_422_654),
        ("tokenizer/tokenizer_config.json", 9_732),
        ("tokenizer/vocab.json", 2_776_833),
        ("transformer/config.json", 500),
        ("transformer/diffusion_pytorch_model-00001-of-00002.safetensors", 9_973_727_144),
        ("transformer/diffusion_pytorch_model-00002-of-00002.safetensors", 2_336_146_728),
        ("transformer/diffusion_pytorch_model.safetensors.index.json", 48_969),
        ("vae/config.json", 820),
        ("vae/diffusion_pytorch_model.safetensors", 167_666_902),
    ]
    assert sum(size for _, size in files) == 20_538_494_574
    return _Plan(
        [size for _, size in files],
        strategy="diffusers",
        attrs={"dtype": "", "file_layout": "diffusers"},
        paths=[path for path, _ in files],
    )


# ---------------------------------------------------------------------------
# Disk preflight
# ---------------------------------------------------------------------------

def test_preflight_rejects_oversized_source_with_actionable_message(tmp_path: Path) -> None:
    # 10 PiB source cannot fit any real test filesystem.
    with pytest.raises(CloneDiskSpaceError, match=r"need ~.* GiB free .*have .* GiB"):
        _preflight_disk(tmp_path, _Plan([10 * 1024**5]), _specs())


def test_preflight_passes_tiny_source(tmp_path: Path) -> None:
    _preflight_disk(tmp_path, _Plan([1024]), _specs())  # must not raise


def test_preflight_skips_when_plan_unavailable(tmp_path: Path) -> None:
    _preflight_disk(tmp_path, None, _specs())  # fail-open: download surfaces its own error


def test_preflight_ltx2_singlefile_needs_only_source_plus_margin(
    tmp_path: Path, monkeypatch,
) -> None:
    """gw#592/gw#593 companion: run_clone routes strategy='aio_singlefile'
    LTX-2 sources through publish_as_is regardless of the requested output
    layout (no diffusers pipeline exists for the family) -- but without this
    fix, preflight had no ltx2_native concept and budgeted a full
    layout-repack + materialized-dtype tree for a clone that only ever needs
    the source bytes. Found live: e2e#185 run 7, CloneDiskSpaceError
    (need ~388.8 GiB) on a real 43GB LTX-2.3 dev-checkpoint clone that
    should only need ~45GB. Real filename (source_include-narrowed to the
    ONE checkpoint) + real size."""
    source_bytes = 46_149_344_974  # real ltx-2.3-22b-dev.safetensors size
    plan = _Plan(
        [source_bytes],
        strategy="aio_singlefile",
        attrs={"file_layout": "singlefile", "dtype": "bf16"},
        paths=["ltx-2.3-22b-dev.safetensors"],
        provider="huggingface",
    )
    # Requested {dtype bf16, layout diffusers} -- mismatched layout from the
    # singlefile source, which (absent the ltx2_native carve-out) would send
    # this down the materialized/repack disk-budget branch.
    specs = _specs(dtype="bf16", layout="diffusers")
    monkeypatch.setattr(
        "gen_worker.convert.clone.shutil.disk_usage",
        lambda _: SimpleNamespace(free=60 * 1024**3),  # fits source+margin, NOT the repack budget
    )
    _preflight_disk(tmp_path, plan, specs)  # must not raise


def test_preflight_non_ltx2_singlefile_still_budgets_the_repack(
    tmp_path: Path, monkeypatch,
) -> None:
    """Guardrail: the ltx2 carve-out must not swallow the general singlefile
    ->diffusers repack case (e.g. sdxl/flux/zimage) -- those genuinely need
    the wider budget."""
    plan = _Plan(
        [50 * 1024**3],
        strategy="aio_singlefile",
        attrs={"file_layout": "singlefile", "dtype": "bf16"},
        paths=["some-checkpoint-v1.safetensors"],
        provider="huggingface",
    )
    monkeypatch.setattr(
        "gen_worker.convert.clone.shutil.disk_usage",
        lambda _: SimpleNamespace(free=60 * 1024**3),  # fits source+margin, NOT the repack budget
    )
    with pytest.raises(CloneDiskSpaceError):
        _preflight_disk(tmp_path, plan, _specs(dtype="bf16", layout="diffusers"))


def test_preflight_has_no_environment_headroom_override(tmp_path: Path, monkeypatch) -> None:
    free = os.statvfs(tmp_path).f_bavail * os.statvfs(tmp_path).f_frsize
    source = max((free - 2 * 1024**3) // 2, 1)
    monkeypatch.setenv("COZY_CONVERT_DISK_HEADROOM", "1000000")
    _preflight_disk(
        tmp_path,
        _Plan([source], strategy="transformers"),
        _specs(dtype="source", layout="singlefile"),
    )
    monkeypatch.setenv("COZY_CONVERT_DISK_HEADROOM", "0.000001")
    with pytest.raises(CloneDiskSpaceError):
        _preflight_disk(
            tmp_path,
            _Plan([10 * 1024**5], strategy="transformers"),
            _specs(dtype="source", layout="singlefile"),
        )


def test_z_image_source_mirror_fits_real_runpod_cpu_disk(tmp_path: Path, monkeypatch) -> None:
    # Exact selected paths and sizes from immutable
    # Tongyi-MAI/Z-Image@04cc4abb...3021. Four existing HF shard members are
    # above Tensorhub's 2 GiB transfer limit and must be resharded together.
    plan = _z_image_plan()
    specs = _specs(dtype="source")
    monkeypatch.setattr(
        "gen_worker.convert.clone.shutil.disk_usage",
        lambda _: SimpleNamespace(free=40 * 1024**3),
    )
    _preflight_disk(tmp_path, plan, specs)

    monkeypatch.setattr(
        "gen_worker.convert.clone.shutil.disk_usage",
        lambda _: SimpleNamespace(free=30 * 1024**3),
    )
    with pytest.raises(
        CloneDiskSpaceError,
        match=r"hardlink passthrough.*oversize-safetensors reshard output",
    ):
        _preflight_disk(tmp_path, plan, specs)


def test_untyped_z_image_bf16_conversion_does_not_assume_passthrough(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _Plan(
        [20_538_494_574],
        strategy="diffusers",
        attrs={"dtype": "", "file_layout": "diffusers"},
    )
    monkeypatch.setattr(
        "gen_worker.convert.clone.shutil.disk_usage",
        lambda _: SimpleNamespace(free=40 * 1024**3),
    )
    with pytest.raises(CloneDiskSpaceError, match="materialized output tree"):
        _preflight_disk(tmp_path, plan, _specs(dtype="bf16"))


def test_civitai_plan_preserves_standard_cpu_clone_admission(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from gen_worker.convert.ingest import CivitaiSourcePlan

    source_bytes = 6_900_000_000
    plan = CivitaiSourcePlan(
        version_id=1,
        payload={"model": {"type": "Checkpoint"}},
        files=[{
            "name": "model.safetensors",
            "size_bytes": source_bytes,
            "sha256": "a" * 64,
        }],
        revision="sha256:" + "b" * 64,
    )
    monkeypatch.setattr(
        "gen_worker.convert.clone.shutil.disk_usage",
        lambda _: SimpleNamespace(free=40 * 1024**3),
    )
    _preflight_disk(tmp_path, plan, _specs(dtype="bf16", layout="singlefile"))


def test_source_mirrors_account_for_every_oversized_safetensors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gib = 1024**3
    plan = _Plan(
        [6 * gib, 6 * gib],
        strategy="diffusers",
        attrs={"dtype": "bf16", "file_layout": "diffusers"},
    )
    monkeypatch.setattr(
        "gen_worker.convert.clone.shutil.disk_usage",
        lambda _: SimpleNamespace(free=32 * gib),
    )
    with pytest.raises(CloneDiskSpaceError, match="oversize-safetensors reshard output"):
        _preflight_disk(
            tmp_path,
            plan,
            _specs(dtype="source") + _specs(dtype="bf16"),
        )


def test_layout_repackage_accounts_for_output_and_temporary_tree(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gib = 1024**3
    plan = _Plan(
        [20 * gib],
        strategy="diffusers",
        attrs={"dtype": "bf16", "file_layout": "diffusers"},
    )
    monkeypatch.setattr(
        "gen_worker.convert.clone.shutil.disk_usage",
        lambda _: SimpleNamespace(free=60 * gib),
    )
    with pytest.raises(CloneDiskSpaceError, match="layout-repack tree"):
        _preflight_disk(tmp_path, plan, _specs(dtype="bf16", layout="singlefile"))


def test_non_direct_gguf_accounts_for_f16_intermediate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gib = 1024**3
    plan = _Plan(
        [10 * gib],
        strategy="diffusers",
        attrs={"dtype": "bf16", "file_layout": "diffusers"},
    )
    monkeypatch.setattr(
        "gen_worker.convert.clone.shutil.disk_usage",
        lambda _: SimpleNamespace(free=31 * gib),
    )
    with pytest.raises(CloneDiskSpaceError, match="intermediate F16 GGUF"):
        _preflight_disk(
            tmp_path,
            plan,
            _specs(dtype="q4_k_m", layout="diffusers", file_type="gguf"),
        )


def test_multiple_gguf_outputs_share_one_intermediate_peak(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gib = 1024**3
    plan = _Plan(
        [10 * gib],
        strategy="diffusers",
        attrs={"dtype": "bf16", "file_layout": "diffusers"},
    )
    monkeypatch.setattr(
        "gen_worker.convert.clone.shutil.disk_usage",
        lambda _: SimpleNamespace(free=45 * gib),
    )
    _preflight_disk(
        tmp_path,
        plan,
        _specs(dtype="q4_k_m", layout="diffusers", file_type="gguf")
        + _specs(dtype="q5_k_m", layout="diffusers", file_type="gguf"),
    )


def _write_raw_safetensors(path: Path, tensors: dict[str, int]) -> None:
    offset = 0
    header = {}
    for name, size in tensors.items():
        header[name] = {
            "dtype": "F16",
            "shape": [size // 2],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    body = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(body)) + body + bytes(offset))


def test_existing_hf_shard_group_remains_loadable_after_reshard(tmp_path: Path) -> None:
    component = tmp_path / "transformer"
    component.mkdir()
    first = component / "model-00001-of-00002.safetensors"
    second = component / "model-00002-of-00002.safetensors"
    _write_raw_safetensors(first, {"a": 180, "b": 180})
    _write_raw_safetensors(second, {"c": 80})
    index = component / "model.safetensors.index.json"
    index.write_text(json.dumps({
        "metadata": {"total_size": 440},
        "weight_map": {
            "a": first.name,
            "b": first.name,
            "c": second.name,
        },
    }))

    _stage_oversize_safetensors(tmp_path, max_shard_bytes=200)

    payload = json.loads(index.read_text())
    assert set(payload["weight_map"]) == {"a", "b", "c"}
    assert not first.exists() and not second.exists()
    for tensor, shard_name in payload["weight_map"].items():
        shard = component / shard_name
        assert shard.is_file()
        raw = shard.read_bytes()
        header_len = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + header_len])
        assert tensor in header

    if importlib.util.find_spec("safetensors") and importlib.util.find_spec("numpy"):
        from safetensors import safe_open

        loaded = set()
        for shard_name in set(payload["weight_map"].values()):
            with safe_open(str(component / shard_name), framework="numpy") as handle:
                loaded.update(handle.keys())
                for name in handle.keys():
                    handle.get_tensor(name)
        assert loaded == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# Stale-scratch sweep
# ---------------------------------------------------------------------------

def _mkdir_aged(base: Path, name: str, age_s: float) -> Path:
    d = base / name
    d.mkdir(parents=True)
    (d / "junk.bin").write_bytes(b"x" * 8)
    stamp = time.time() - age_s
    os.utime(d, (stamp, stamp))
    return d


def test_sweep_removes_stale_unlocked_keeps_live_and_fresh(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("COZY_CONVERT_SCRATCH_TTL_S", "60")
    stale = _mkdir_aged(tmp_path, "clone-deadbeef00000001", age_s=3600)
    fresh = _mkdir_aged(tmp_path, "clone-deadbeef00000002", age_s=1)
    live = _mkdir_aged(tmp_path, "clone-deadbeef00000003", age_s=3600)
    mine = _mkdir_aged(tmp_path, "clone-deadbeef00000004", age_s=3600)

    # A concurrent clone holds `live`'s flock (real lock, same protocol).
    live_lock = os.open(tmp_path / f".{live.name}.lock", os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(live_lock, fcntl.LOCK_EX)
    try:
        _sweep_stale_workdirs(tmp_path, keep=mine)
    finally:
        os.close(live_lock)

    assert not stale.exists(), "stale unlocked scratch must be swept"
    assert fresh.exists(), "fresh scratch is inside the TTL"
    assert live.exists(), "flock-held scratch belongs to a live clone"
    assert mine.exists(), "the caller's own workdir is never swept"


def test_sweep_survives_missing_base(tmp_path: Path) -> None:
    _sweep_stale_workdirs(tmp_path / "does-not-exist")  # must not raise


# ---------------------------------------------------------------------------
# Workdir cleanup after EVERY job (success AND failure)
# ---------------------------------------------------------------------------

class _Ctx:
    _file_api_base_url = "http://127.0.0.1:1"
    _worker_capability_token = "tok"
    owner = "acme"


def test_failed_clone_removes_workdir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path))
    monkeypatch.delenv("COZY_CONVERT_RETAIN_WORKDIR", raising=False)
    with pytest.raises(ValueError, match="unsupported clone provider"):
        run_clone(_Ctx(), provider="bogus", destination_repo="acme/x")
    workdir = _clone_workdir("bogus", "", "acme/x")  # re-derives the keyed path
    # _clone_workdir recreates the dir; the failed run must have removed its
    # contents-bearing predecessor, so the fresh dir is empty.
    assert list(workdir.iterdir()) == []
    workdir.rmdir()
    assert [p.name for p in tmp_path.iterdir() if p.is_dir()] == []


def test_failed_clone_retains_workdir_when_opted_in(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path))
    monkeypatch.setenv("COZY_CONVERT_RETAIN_WORKDIR", "1")
    with pytest.raises(ValueError, match="unsupported clone provider"):
        run_clone(_Ctx(), provider="bogus", destination_repo="acme/x")
    dirs = [p for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("clone-")]
    assert len(dirs) == 1, "opt-in retention must keep the failed workdir"
