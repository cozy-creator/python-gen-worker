"""pgw#749: a cold-boot CANDIDATE key must equal the key a mint on the same
runtime PUBLISHES — otherwise boot-attach adoption can never fire and every
cold pod starts a mint it did not need (live: every sdxl 0.2.14 pod
demanded ck5-4c7e494b/ck5-ad5fdb4b while every mint published
ck5-41b367ab/ck5-e6e3be89; ADOPT-WITHOUT-MINT = 0 across a 13-worker
burst).

Mechanism (verified against the banked artifacts): the seal's loaded-lib
fact was a snapshot of /proc/self/maps "frozen at first computation" — but
the mapped set is a function of LOAD PHASE (torch preloads cublas/cudnn at
import, libtriton maps at first dynamo compile, libcuda at first CUDA
call), so a candidate computed at a cold phase and a mint sealed at a warm
phase digest DIFFERENT manifests and derive different keys forever. The
banked artifact manifests carry the 22-lib compile-warm set (libtriton,
libcudnn, libcuda...) while candidates stayed driver-cohort-stable — the
cold-set signature.

Red-verified: on the pre-fix (maps-frozen) tree the phase tests here fail
with exactly that shape."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import pytest

torch = pytest.importorskip("torch")

from gen_worker import cell_key as ck
from gen_worker import compile_cache as cc
from gen_worker import env_seal
from gen_worker.registry import CompileCell

_RT = {
    "sku": "l4", "sm": "sm_89", "cuda": "13.0", "cuda_driver": "580.126.16",
    "torch": "2.13.0+cu130", "triton": "3.7.1",
    "image_digest": "sha256:" + "1e" * 32,
}

_COLD_LIBS: Tuple[Tuple[str, bytes], ...] = (
    ("libtorch_cuda.so", b"torch-bytes"),
    ("libcudart.so.13", b"cudart-bytes"),
    ("libcublas.so.13", b"cublas-bytes"),
)
# What maps additionally to a compile-warm process: triton at first dynamo
# use, cudnn at first conv, the host driver at first CUDA call.
_WARM_EXTRA: Tuple[Tuple[str, bytes], ...] = (
    ("libtriton.so", b"triton-bytes"),
    ("libcudnn.so.9", b"cudnn-bytes"),
    ("libcuda.so.580.126.16", b"host-driver"),
)


@pytest.fixture(autouse=True)
def _pinned_runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(cc, "runtime_key", lambda: dict(_RT))
    monkeypatch.setattr(cc, "gen_worker_version", lambda: "0.76.3")
    monkeypatch.setattr(
        cc, "_lib_versions",
        lambda: {"diffusers": "0.39.0", "transformers": "5.13.1"})
    monkeypatch.setattr(
        cc, "toolchain_digest", lambda: (("torch", "1" * 16),))
    monkeypatch.setattr(
        cc, "static_code_closure",
        lambda roots=(): (("gen_worker/compile_cache.py", "3" * 16),))
    monkeypatch.setattr(cc, "content_keys", lambda: ())
    monkeypatch.setattr(env_seal, "_BOOT_SEAL", None)
    monkeypatch.setattr(env_seal, "_LIB_SNAPSHOT", None)
    yield


def _phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str,
    mapped: Tuple[Tuple[str, bytes], ...],
    disk: Tuple[Tuple[str, bytes], ...] = _COLD_LIBS + _WARM_EXTRA,
) -> None:
    """Enter one process phase: `mapped` is what /proc/self/maps shows,
    `disk` is what the python env ships. A fresh snapshot simulates a fresh
    process freezing at this phase."""
    root = tmp_path / name
    root.mkdir()
    lines = []
    for i, (base, content) in enumerate(mapped):
        target = root / base
        target.write_bytes(content)
        lines.append(
            f"7f{i:010x}000-7f{i:010x}fff r-xp 00000000 08:01 {i + 1} {target}")
    (root / "maps").write_text("\n".join(lines) + "\n")
    diskroot = root / "toolchain"
    diskroot.mkdir()
    for base, content in disk:
        (diskroot / base).write_bytes(content)
    monkeypatch.setattr(env_seal, "_MAPS_PATH", root / "maps")
    # raising=False: on the PRE-fix tree the override seam does not exist —
    # the seal reads the mapped set and these tests fail RED, not error.
    monkeypatch.setattr(
        env_seal, "_TOOLCHAIN_LIB_DIRS_OVERRIDE", (diskroot,), raising=False)
    monkeypatch.setattr(env_seal, "_LIB_SNAPSHOT", None)


def _cfg() -> CompileCell:
    return CompileCell(
        shapes=((1024, 1024), (832, 1216)), targets=("unet",), family="sdxl",
        regional=False, text_len=77, dynamic=(), lora_bucket=64,
        guidance_scales=(0.0, 5.0), text_lens=(77,),
    )


def test_seal_is_phase_independent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A cold-boot process and a compile-warm process must derive the SAME
    seal digest: identity comes from the toolchain ON DISK, never from
    what happens to be dlopened when the first key is computed."""
    _phase(monkeypatch, tmp_path, "cold", _COLD_LIBS)
    cold = env_seal.effective_seal()
    _phase(monkeypatch, tmp_path, "warm", _COLD_LIBS + _WARM_EXTRA)
    warm = env_seal.effective_seal()
    assert cold["loaded_libs"] == warm["loaded_libs"]
    assert env_seal.seal_digest(cold) == env_seal.seal_digest(warm)


def test_cold_candidate_key_equals_warm_published_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The pgw#749 contract end-to-end at the SDK level: the pre-load
    candidate key (cell_key.compute at a cold phase) must equal the key a
    compile-warm mint stamps into its artifact metadata
    (from_artifact_metadata over artifact_metadata). This is exactly the
    parity boot-attach adoption requires."""
    cfg = _cfg()

    # Cold phase — the executor's cell_lookups() candidate computation.
    _phase(monkeypatch, tmp_path, "cold", _COLD_LIBS)
    candidate = ck.compute(
        "sdxl", "w8a8", 64,
        contract=cfg.contract_digest(), regional=False,
    ).digest

    # Warm phase, fresh snapshot — the mint stamping its artifact.
    _phase(monkeypatch, tmp_path, "warm", _COLD_LIBS + _WARM_EXTRA)
    meta: Dict[str, Any] = cc.artifact_metadata(
        family="sdxl",
        shapes=cfg.shapes,
        targets=cfg.targets,
        guidance_scales=cfg.guidance_scales,
        weight_lane="w8a8-lora64",
        lora_bucket=64,
        shape_contract=cc.declared_contract_facts(cfg),
    )
    published = ck.from_artifact_metadata(meta).digest
    assert meta["cell_key"] == published  # the stamp agrees with its axes
    assert candidate == published, (
        "cold-boot candidate key != published key — boot-attach adoption "
        "is structurally dead (pgw#749)"
    )


def test_artifact_manifest_records_disk_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The metadata's per-library list is the DISK identity manifest (so a
    mismatch names the library), never the phase-dependent mapped set —
    and the host driver never appears in it (pgw#745)."""
    _phase(monkeypatch, tmp_path, "warm", _COLD_LIBS + _WARM_EXTRA)
    meta = cc.artifact_metadata(family="sdxl", shapes=((64, 64),),
                                targets=("unet",))
    libs = meta["loaded_libs"]
    assert "libtriton.so" in libs  # shipped on disk, phase-independent
    assert "libcudnn.so.9" in libs
    assert not any(base.startswith("libcuda.so") for base in libs), libs
