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

import inspect

import pytest

from gen_worker import entrypoint, env_seal


def test_entrypoint_refuses_unknown_torch_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCHDYNAMO_SUPPRESS_ERRORS", "1")
    with pytest.raises(env_seal.EnvSealError) as exc:
        entrypoint._establish_env_seal()
    assert "TORCHDYNAMO_SUPPRESS_ERRORS" in str(exc.value)


def test_entrypoint_establishes_effective_seal() -> None:
    import torch

    seal = entrypoint._establish_env_seal()
    # The canonical surface is EFFECTIVE, not merely recorded.
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "highest"
    # Every canonical entry is effective in the seal (the seal also records
    # non-canonicalized facts like CUBLAS_WORKSPACE_CONFIG).
    assert env_seal.CANONICAL_CONFIG.items() <= seal["config"].items()
    # The digest is the ck4 env_seal axis and is deterministic.
    assert env_seal.seal_digest(seal) == env_seal.seal_digest(
        entrypoint._establish_env_seal())


def test_run_main_seals_before_cuda_probe_and_worker() -> None:
    src = inspect.getsource(entrypoint._run_main)
    seal_at = src.index("_establish_env_seal")
    assert seal_at < src.index("should_probe_cuda"), (
        "the seal must be established before the CUDA probe")
    assert seal_at < src.index("Worker("), (
        "the seal must be established before the worker starts")
    # 0.70.3 regression: sealed BEFORE settings, so a refusal could not
    # dial the hub — every fleet pod died as a silent pod_exited.
    assert src.index("get_settings") < seal_at, (
        "the seal must run after settings so a refusal dials the hub typed")
    assert "settings=settings" in src[src.index("_establish_env_seal"):
                                      src.index("should_probe_cuda")]


def test_base_image_build_constants_are_allowlisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fleet's pytorch/pytorch base image stamps PYTORCH_VERSION (and
    siblings). RED on 0.70.3: the R7 prefix widening refused them and every
    pod on the fleet base image died at boot before hello (the sdxl 0.2.12
    rollback)."""
    monkeypatch.setenv("PYTORCH_VERSION", "2.13.0")
    monkeypatch.setenv("PYTORCH_BUILD_VERSION", "2.13.0")
    monkeypatch.setenv("PYTORCH_BUILD_NUMBER", "1")
    env_seal.check_torch_env()  # must not raise
    entrypoint._establish_env_seal()
