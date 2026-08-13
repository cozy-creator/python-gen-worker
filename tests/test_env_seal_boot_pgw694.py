"""pgw#694/#696 boot wiring: the worker entrypoint establishes the env seal.

The hardening set (chaos a73e6c8) shipped `env_seal.establish()` with the
executor-side boot wiring deliberately left to the executor owner. This
pins it: the entrypoint seals the process BEFORE the CUDA probe and before
any model/compile work, refuses to start on an unsealable environment
(naming the variable), and the sealed config is effective in-process.
RED on the unwired tree: `_establish_env_seal` does not exist and
`_run_main` never seals.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gen_worker import entrypoint, env_seal

import torch
from typing import Iterator


@pytest.fixture(autouse=True)
def _restore_global_matmul_flags() -> Iterator[None]:
    """The canonical imposition is deliberately process-global; the SUITE
    must not leak it across files (compiled graphs compiled under one TF32 state
    GlobalStateGuard-miss under another — the flux hit-counter tests)."""
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    benchmark = torch.backends.cudnn.benchmark
    yield
    torch.set_float32_matmul_precision(precision)
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_tf32
    torch.backends.cudnn.benchmark = benchmark



def test_entrypoint_establishes_effective_seal() -> None:
    import torch

    seal = entrypoint._establish_env_seal()
    # The canonical surface is EFFECTIVE, not merely recorded — and it IS
    # the pgw#654 serving posture (TF32 on), so mint==serve.
    assert torch.backends.cudnn.allow_tf32 is True
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.get_float32_matmul_precision() == "high"
    # Every canonical compiled graph is stated by the seal — pgw#1049: as the
    # DECLARATION (boot verified the read-back against it).
    from gen_worker import settings_authority as sa

    assert sa.DECLARED_TORCH.items() <= seal["config"].items()
    # The digest is the env_seal key axis and is deterministic.
    assert env_seal.seal_digest(seal) == env_seal.seal_digest(
        entrypoint._establish_env_seal())


class _ProbeReached(Exception):
    pass


def test_run_main_seals_after_settings_and_before_cuda_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The REAL _run_main control flow, observed at its seams: settings load
    first (so a seal refusal can dial the hub typed — the 0.70.3 silent
    pod_exited regression), then the seal, and only then the CUDA probe
    (which is before any model/compile work and the Worker start)."""
    order: list = []
    monkeypatch.setattr(entrypoint, "_install_stack_dump_handler", lambda: None)
    monkeypatch.setattr(
        entrypoint, "_bootstrap_configuration",
        lambda: order.append("settings") or SimpleNamespace(
            endpoint_lock_path=""))
    monkeypatch.setattr(
        entrypoint, "_establish_env_seal", lambda: order.append("seal") or {})
    monkeypatch.setattr(entrypoint, "load_manifest", lambda path: {})
    monkeypatch.setattr(entrypoint, "_preflight_cache_dirs", lambda: None)

    def _probe(manifest: object) -> bool:
        order.append("probe")
        raise _ProbeReached  # everything after (incl. Worker) is post-seal

    monkeypatch.setattr(entrypoint, "should_probe_cuda", _probe)
    with pytest.raises(_ProbeReached):
        entrypoint._run_main()
    assert order == ["settings", "seal", "probe"]


def test_seal_refusal_dials_the_hub_typed_with_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsealable environment exits typed AND the fatal report carries
    the loaded settings — the hub-dial precondition 0.70.3 broke."""
    fatal: list = []
    settings = SimpleNamespace(endpoint_lock_path="")
    monkeypatch.setattr(entrypoint, "_install_stack_dump_handler", lambda: None)
    monkeypatch.setattr(entrypoint, "_bootstrap_configuration", lambda: settings)

    def _refuse() -> dict:
        raise RuntimeError("unsealable: HOSTILE_VAR")

    monkeypatch.setattr(entrypoint, "_establish_env_seal", _refuse)
    monkeypatch.setattr(
        entrypoint, "_log_worker_fatal",
        lambda phase, exc, **kw: fatal.append((phase, str(exc), kw)))

    assert entrypoint._run_main() == 1
    (phase, message, kw), = fatal
    assert phase == "env_seal"
    assert "HOSTILE_VAR" in message
    assert kw.get("settings") is settings


def test_base_image_build_constants_never_kill_a_boot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fleet's pytorch/pytorch base image stamps PYTORCH_VERSION (and
    siblings). RED on 0.70.3: the allowlist gate refused them and every pod
    on the fleet base image died at boot before hello (the sdxl 0.2.12
    rollback). pgw#718 erase-and-impose makes the class impossible: the
    vars are ERASED and boot proceeds."""
    monkeypatch.setenv("PYTORCH_VERSION", "2.13.0")
    monkeypatch.setenv("PYTORCH_BUILD_VERSION", "2.13.0")
    monkeypatch.setenv("PYTORCH_BUILD_NUMBER", "1")
    entrypoint._establish_env_seal()  # must not raise
    import os

    assert "PYTORCH_VERSION" not in os.environ
