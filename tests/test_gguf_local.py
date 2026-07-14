"""Local-only GGUF selection, composition, loading, and reporting."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gen_worker.api.binding import Hub
from gen_worker.models.gguf_local import (
    GGUF_MARKER,
    compose_resolved,
    composed_digest,
    gguf_qtype,
    is_denoiser_weight_path,
    maybe_rebind_gguf,
    read_marker,
    select_gguf,
    write_marker,
)
from gen_worker.models.hub_client import (
    WorkerResolvedFlavor,
    WorkerResolvedRepo,
    WorkerResolvedRepoFile,
)

GB = 1_000_000_000


def _file(path: str, size: float, digest: str) -> WorkerResolvedRepoFile:
    return WorkerResolvedRepoFile(
        path=path, size_bytes=int(size * GB), blake3=digest, url="https://example.test",
    )


def _resolved(*, fp8: bool = False, large_te: bool = False) -> WorkerResolvedRepo:
    files = [
        _file("model_index.json", 0.000001, "aa"),
        _file("transformer/diffusion_pytorch_model-00001-of-00002.safetensors", 9, "bb"),
        _file("transformer/diffusion_pytorch_model-00002-of-00002.safetensors", 8, "cc"),
        _file("transformer/config.json", 0.000001, "dd"),
        _file("text_encoder/model.safetensors", 16.4 if large_te else 1, "ee"),
        _file("vae/diffusion_pytorch_model.safetensors", 0.2, "ff"),
    ]
    siblings = [
        WorkerResolvedFlavor(flavor="gguf-q8_0", size_bytes=int(9.8 * GB)),
        WorkerResolvedFlavor(flavor="gguf-q4_k_m", size_bytes=int(5.6 * GB)),
    ]
    if fp8:
        siblings.append(WorkerResolvedFlavor(flavor="fp8", size_bytes=9 * GB))
    return WorkerResolvedRepo(
        snapshot_digest="base-digest",
        files=files,
        size_bytes=sum(item.size_bytes for item in files),
        sibling_flavors=siblings,
    )


@pytest.mark.parametrize(
    ("token", "qtype"),
    [
        ("gguf-q4_k_m", "q4_k_m"),
        ("GGUF-Q8_0", "q8_0"),
        ("gguf-iq4_xs", ""),
        ("fp8", ""),
    ],
)
def test_gguf_qtype(token: str, qtype: str) -> None:
    assert gguf_qtype(token) == qtype


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("transformer/diffusion_pytorch_model.safetensors", True),
        ("unet/diffusion_pytorch_model.safetensors.index.json", True),
        ("transformer/config.json", False),
        ("text_encoder/model.safetensors", False),
        ("model_index.json", False),
    ],
)
def test_denoiser_path(path: str, expected: bool) -> None:
    assert is_denoiser_weight_path(path) is expected


def test_selects_best_fitting_gguf_after_native_rungs() -> None:
    pick = select_gguf(_resolved(), gpu_sm=89, free_vram_gb=7.4)
    assert pick is not None
    assert pick.flavor == "gguf-q4_k_m"
    assert pick.estimated_vram_gb == pytest.approx(6.3, abs=0.02)
    assert not pick.te_offload

    assert select_gguf(_resolved(fp8=True), gpu_sm=89, free_vram_gb=10.0) is None
    assert select_gguf(_resolved(), gpu_sm=120, free_vram_gb=7.4) is None


def test_selects_text_encoder_offload_bound() -> None:
    pick = select_gguf(_resolved(large_te=True), gpu_sm=89, free_vram_gb=7.0)
    assert pick is not None
    assert pick.flavor == "gguf-q4_k_m"
    assert pick.te_offload


def test_local_binding_rebinds_but_author_pins_do_not() -> None:
    binding = Hub("tensorhub/flux2-klein-9b")
    rebound = maybe_rebind_gguf(
        binding,
        resolved=_resolved(),
        gpu_sm=89,
        free_vram_gb=7.4,
        installed_libs=(),
    )
    assert rebound.flavor == "gguf-q4_k_m"

    pinned = Hub("tensorhub/flux2-klein-9b", flavor="fp8")
    assert maybe_rebind_gguf(
        pinned,
        resolved=_resolved(),
        gpu_sm=89,
        free_vram_gb=7.4,
        installed_libs=(),
    ) is pinned
    scoped = Hub("tensorhub/flux2-klein-9b", components=("vae",))
    assert maybe_rebind_gguf(
        scoped,
        resolved=_resolved(),
        gpu_sm=89,
        free_vram_gb=7.4,
        installed_libs=(),
    ) is scoped


def _flavor() -> WorkerResolvedRepo:
    return WorkerResolvedRepo(
        snapshot_digest="flavor-digest",
        files=[
            _file("flux2-klein-9b-Q4_K_M.gguf", 5.6, "99"),
            _file("README.md", 0.000001, "98"),
        ],
    )


def test_composition_drops_base_denoiser_weights() -> None:
    composed, gguf_path = compose_resolved(_resolved(), _flavor())
    paths = {item.path for item in composed.files}
    assert gguf_path == "flux2-klein-9b-Q4_K_M.gguf"
    assert "model_index.json" in paths
    assert "transformer/config.json" in paths
    assert gguf_path in paths
    assert not any(is_denoiser_weight_path(path) for path in paths)
    assert composed.snapshot_digest == composed_digest(
        "flavor-digest", "base-digest",
    )


def test_composition_rejects_invalid_manifests() -> None:
    flavor = _flavor()
    with pytest.raises(ValueError, match="exactly 1"):
        compose_resolved(
            _resolved(),
            WorkerResolvedRepo(
                snapshot_digest="bad",
                files=flavor.files + [_file("second.gguf", 1, "97")],
            ),
        )
    with pytest.raises(ValueError, match="model_index"):
        compose_resolved(
            WorkerResolvedRepo(
                snapshot_digest="bad-base",
                files=[item for item in _resolved().files if item.path != "model_index.json"],
            ),
            flavor,
        )


def _snapshot(tmp_path: Path, *, marker: bool = True) -> Path:
    snapshot = tmp_path / "snapshot"
    (snapshot / "transformer").mkdir(parents=True)
    (snapshot / "model_index.json").write_text(
        json.dumps({"transformer": [__name__, "_FakeDenoiser"]}),
        encoding="utf-8",
    )
    (snapshot / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model-Q4_K_M.gguf").write_bytes(b"GGUF")
    if marker:
        write_marker(
            snapshot, flavor="gguf-q4_k_m", gguf_relpath="model-Q4_K_M.gguf",
        )
    return snapshot


def test_marker_and_structural_detection(tmp_path: Path) -> None:
    from gen_worker.models.loading import detect_gguf_snapshot

    snapshot = _snapshot(tmp_path)
    assert read_marker(snapshot) == {
        "flavor": "gguf-q4_k_m",
        "qtype": "q4_k_m",
        "gguf_path": "model-Q4_K_M.gguf",
    }
    found = detect_gguf_snapshot(snapshot)
    assert found is not None and found[1] == "q4_k_m"

    (snapshot / GGUF_MARKER).unlink()
    found = detect_gguf_snapshot(snapshot)
    assert found is not None and found[0].name == "model-Q4_K_M.gguf"
    (snapshot / "model_index.json").unlink()
    assert detect_gguf_snapshot(snapshot) is None


class _FakeDenoiser:
    call: dict[str, Any] = {}

    @classmethod
    def from_single_file(cls, path: str, **kwargs: Any) -> object:
        cls.call = {"path": path, **kwargs}
        return object()


class _FakePipeline:
    call: dict[str, Any] = {}

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> SimpleNamespace:
        cls.call = {"path": path, **kwargs}
        return SimpleNamespace()


def test_loads_gguf_denoiser_into_base_tree(tmp_path: Path) -> None:
    pytest.importorskip("diffusers")
    from gen_worker.models.loading import load_gguf_pipeline

    snapshot = _snapshot(tmp_path)
    pipe = load_gguf_pipeline(
        _FakePipeline, snapshot, snapshot / "model-Q4_K_M.gguf",
    )
    assert isinstance(pipe, SimpleNamespace)
    assert _FakeDenoiser.call["config"] == str(snapshot / "transformer")
    assert _FakePipeline.call["transformer"] is not None


def test_listing_probe_matches_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    from gen_worker.cli.listing import _gguf_probe
    from gen_worker.models import hub_client

    monkeypatch.setattr(hub_client, "resolve_repo", lambda ref: _resolved())
    caps = SimpleNamespace(gpu_sm=89, installed_libs=[])
    got = _gguf_probe(Hub("tensorhub/flux2-klein-9b"), caps, 7.4)
    assert got is not None
    assert got["flavor"] == "gguf-q4_k_m"
    assert got["est_gb"] == pytest.approx(6.3, abs=0.02)

    monkeypatch.setattr(
        hub_client, "resolve_repo", lambda ref: (_ for _ in ()).throw(RuntimeError("down")),
    )
    assert _gguf_probe(Hub("tensorhub/flux2-klein-9b"), caps, 7.4) is None


def test_gguf_placement_keeps_a_measured_fit_resident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gen_worker.models import memory

    pipe = SimpleNamespace(_cozy_gguf_quant="q4_k_m")
    monkeypatch.setattr(memory, "estimate_pipeline_size_gb", lambda obj: 6.8)
    monkeypatch.setattr(memory, "estimate_cuda_resident_gb", lambda obj: 5.6)
    monkeypatch.setattr(memory, "get_available_vram_gb", lambda: 1.8)
    assert memory._gguf_resident_override(
        pipe, "model_offload", memory._LOG,
    ) == "vae_only"


def test_gguf_placement_clamps_unsupported_deep_offload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gen_worker.models import memory

    class Pipe:
        _cozy_gguf_quant = "q4_k_m"

        def enable_model_cpu_offload(self) -> None:
            self.model_offload = True

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    pipe = Pipe()
    placed = memory.apply_low_vram_config(pipe, mode="sequential")
    assert placed["mode"] == "model_offload"
    assert placed["model_offload"] is True
