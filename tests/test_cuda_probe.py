"""Boot-time CUDA probe (gw#529): a dead/busy GPU must fail startup before
hello, not terminal-fail a real job at model load. torch is monkeypatched
throughout — this suite never touches real CUDA regardless of the host."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import msgspec
import pytest
import torch

from gen_worker import entrypoint
from gen_worker.cuda_probe import (
    CUDA_PROBE_FAILED_MARKER,
    manifest_needs_cuda,
    probe_cuda,
    should_probe_cuda,
)


class _FakeTensor:
    def __add__(self, other: Any) -> "_FakeTensor":
        return self


def test_probe_cuda_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "ones", lambda *a, **k: _FakeTensor())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    result = probe_cuda()

    assert result.ok
    assert result.reason == ""


def test_probe_cuda_not_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = probe_cuda()

    assert not result.ok
    assert "is_available" in result.reason


def test_probe_cuda_alloc_raises_busy_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    busy = "CUDA error: CUDA-capable device(s) is/are busy or unavailable"

    def _raise(*a: Any, **k: Any) -> None:
        raise RuntimeError(busy)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "ones", _raise)

    result = probe_cuda()

    assert not result.ok
    assert busy in result.reason


def test_manifest_needs_cuda_true_when_any_function_needs_gpu() -> None:
    manifest = {
        "functions": [
            {"name": "a", "resources": {}},
            {"name": "b", "resources": {"gpu": True}},
        ]
    }
    assert manifest_needs_cuda(manifest) is True


def test_manifest_needs_cuda_false_for_accelerator_none() -> None:
    manifest = {"functions": [{"name": "a", "resources": {}}]}
    assert manifest_needs_cuda(manifest) is False


def test_manifest_needs_cuda_false_for_missing_manifest() -> None:
    assert manifest_needs_cuda(None) is False
    assert manifest_needs_cuda({}) is False


@pytest.mark.parametrize(
    ("gpu_flags", "cuda_build", "expected"),
    [
        ([False], False, False),
        ([False], True, False),
        ([True], False, True),
        ([True], True, True),
        ([False, True], False, False),
        ([False, True], True, True),
    ],
)
def test_should_probe_cuda_uses_torch_build_only_for_mixed_manifests(
    gpu_flags: list[bool], cuda_build: bool, expected: bool
) -> None:
    manifest = {
        "functions": [
            {"name": str(i), "resources": {"gpu": True} if gpu else {}}
            for i, gpu in enumerate(gpu_flags)
        ]
    }

    assert should_probe_cuda(manifest, cuda_build=cuda_build) is expected


def _write_manifest(path: Path, *, gpu: bool) -> None:
    manifest = {
        "functions": [
            {
                "name": "gen",
                "module": "fake_mod",
                "kind": "inference",
                "resources": {"gpu": True} if gpu else {},
            }
        ]
    }
    path.write_bytes(msgspec.toml.encode(manifest))


def _base_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, manifest_path: Path) -> None:
    monkeypatch.setenv("ORCHESTRATOR_PUBLIC_ADDR", "orchestrator.local:443")
    monkeypatch.setenv("TENSORHUB_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ENDPOINT_LOCK_PATH", str(manifest_path))


def test_entrypoint_exits_nonzero_on_cuda_probe_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    manifest_path = tmp_path / "endpoint.lock"
    _write_manifest(manifest_path, gpu=True)
    _base_env(monkeypatch, tmp_path, manifest_path)

    from gen_worker.cuda_probe import CudaProbeResult

    monkeypatch.setattr(
        entrypoint, "probe_cuda", lambda *a, **k: CudaProbeResult(ok=False, reason="busy or unavailable")
    )

    def _fail_worker(*a: Any, **k: Any) -> None:
        raise AssertionError("Worker must not be constructed when the CUDA probe fails")

    monkeypatch.setattr(entrypoint, "Worker", _fail_worker)

    with caplog.at_level(logging.ERROR):
        code = entrypoint._run_main()

    assert code == 1
    assert any(CUDA_PROBE_FAILED_MARKER in rec.message for rec in caplog.records)


def test_entrypoint_skips_probe_when_manifest_has_no_gpu_function(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "endpoint.lock"
    _write_manifest(manifest_path, gpu=False)
    _base_env(monkeypatch, tmp_path, manifest_path)

    def _unexpected_probe(*a: Any, **k: Any) -> None:
        raise AssertionError("probe_cuda must not run for an accelerator=none manifest")

    monkeypatch.setattr(entrypoint, "probe_cuda", _unexpected_probe)

    class _StubWorker:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def run(self) -> int:
            return 0

    monkeypatch.setattr(entrypoint, "Worker", _StubWorker)

    code = entrypoint._run_main()

    assert code == 0


def test_entrypoint_skips_probe_for_mixed_manifest_in_cpu_torch_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "endpoint.lock"
    manifest_path.write_bytes(
        msgspec.toml.encode(
            {
                "functions": [
                    {"name": "cpu", "module": "fake_mod", "resources": {}},
                    {"name": "gpu", "module": "fake_mod", "resources": {"gpu": True}},
                ]
            }
        )
    )
    _base_env(monkeypatch, tmp_path, manifest_path)
    monkeypatch.setattr(torch.version, "cuda", None)

    def _unexpected_probe(*a: Any, **k: Any) -> None:
        raise AssertionError("a mixed release's CPU image must not probe CUDA")

    monkeypatch.setattr(entrypoint, "probe_cuda", _unexpected_probe)

    class _StubWorker:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def run(self) -> int:
            return 0

    monkeypatch.setattr(entrypoint, "Worker", _StubWorker)

    assert entrypoint._run_main() == 0
