"""pgw#718/#719: erase-and-impose env contract plus seal composition."""

from __future__ import annotations

import os
import sys
from typing import Iterator

import pytest

torch = pytest.importorskip("torch")

from gen_worker import env_seal


@pytest.fixture(autouse=True)
def _restore_global_matmul_flags() -> Iterator[None]:
    """Do not leak process-global torch settings between tests."""
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    benchmark = torch.backends.cudnn.benchmark
    yield
    torch.set_float32_matmul_precision(precision)
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_tf32
    torch.backends.cudnn.benchmark = benchmark


def test_scrub_never_fails_and_names_what_it_erased(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile = "TORCHINDUCTOR_FORCE_DISABLE_CACHES"
    informational = "PYTORCH_VERSION"
    unknown = "TORCH_TOTALLY_UNKNOWN_TOGGLE_XYZ"
    owned = (
        hostile,
        informational,
        unknown,
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NVIDIA_TF32_OVERRIDE",
    )
    for name in owned:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/libs")
    erased = env_seal.scrub_env()
    for name in owned:
        assert name in erased and name not in os.environ, name
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert os.environ["LD_LIBRARY_PATH"] == "/opt/libs"


def test_establish_scrubs_then_imposes_and_is_pure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTORCH_VERSION", "2.13.0")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seal_a = env_seal.establish()
    assert "PYTORCH_VERSION" not in os.environ
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    assert "CUBLAS_WORKSPACE_CONFIG" not in seal_a["config"]
    seal_b = env_seal.establish()
    assert env_seal.seal_digest(seal_a) == env_seal.seal_digest(seal_b)


def test_typed_knob_is_sealed_and_unknown_knob_refuses() -> None:
    from gen_worker import settings_authority as sa

    try:
        default = sa.impose_torch()
        assert default["cudnn_benchmark"] == "False"
        base_digest = env_seal.seal_digest(env_seal.establish())
        knobbed_seal = env_seal.establish(
            overrides={"cudnn_benchmark": "True"})
        assert knobbed_seal["config"]["cudnn_benchmark"] == "True"
        assert torch.backends.cudnn.benchmark is True
        assert env_seal.seal_digest(knobbed_seal) != base_digest
        with pytest.raises(sa.SettingsImpositionError, match="not_a_real_knob"):
            sa.impose_torch(overrides={"not_a_real_knob": "1"})
    finally:
        env_seal.establish()


def test_hash_seed_facts_ride_the_seal() -> None:
    env_seal.establish()
    assert os.environ["PYTHONHASHSEED"] == "0"
    assert str(sys.flags.hash_randomization) in ("0", "1")
