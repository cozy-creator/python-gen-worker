"""Real GPU-capability detection — replaces the ~61-test import-surface mock
cluster (test_accel.py + test_acceleration_helpers.py + test_no_accelerator.py).

Those files mocked ``torch=None`` into sys.modules and asserted ``hasattr``
on the module — framework/import-surface, not behavior. This calls the REAL
``gpu_capability()`` against whatever hardware is actually present:

  * on CPU CI it must return the ``arch="none"`` report (never raise),
  * on a real GPU it reports a sane compute capability + dtype-support flags,
  * the FP8/NVFP4 dtype gates derive correctly from the SM-major mapping,
  * the optional-dependency shim raises a typed ImportError for removed
    capability helpers (the one genuine import-surface contract worth keeping).
"""

from __future__ import annotations

import pytest

from gen_worker import accel
from gen_worker.accel import GpuCapabilityReport, gpu_capability


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def test_gpu_capability_never_raises_and_returns_report() -> None:
    # The whole point of gpu_capability: tenant setup() branches on caps.arch
    # without try/except, so it must never raise on any host.
    report = gpu_capability(refresh=True)
    assert isinstance(report, GpuCapabilityReport)
    # gpu_count and vram are non-negative regardless of hardware.
    assert report.gpu_count >= 0
    assert report.vram_gb_total >= 0.0


@pytest.mark.skipif(_has_cuda(), reason="CUDA present — see the GPU-path test")
def test_cpu_host_reports_none_arch() -> None:
    report = gpu_capability(refresh=True)
    assert report.arch == "none"
    assert report.compute_capability == ""
    assert report.gpu_count == 0
    assert report.has_fp8 is False
    assert report.has_nvfp4 is False


@pytest.mark.skipif(not _has_cuda(), reason="no CUDA device visible")
def test_gpu_host_reports_sane_capability() -> None:
    report = gpu_capability(refresh=True)
    assert report.arch != "none"
    assert report.gpu_count >= 1
    # compute_capability is "major.minor".
    major = int(report.compute_capability.split(".")[0])
    # FP8 tensor cores from Hopper (SM 9+), NVFP4 from Blackwell (SM 10+).
    assert report.has_fp8 is (major >= 9)
    assert report.has_nvfp4 is (major >= 10)
    assert report.vram_gb_total > 0.0


def test_arch_classification_matches_sm_mapping() -> None:
    # _classify_arch is a pure mapping; verify the documented families
    # without needing the hardware present.
    classify = accel._classify_arch
    assert classify(10, 0) == "blackwell"
    assert classify(12, 0) == "blackwell"
    assert classify(9, 0) == "hopper"
    assert classify(8, 9) == "lovelace"
    assert classify(8, 0) == "ampere"
    assert classify(7, 5) == "turing"
    assert classify(6, 1) == "unknown"


def test_removed_capability_helpers_raise_typed_import_error() -> None:
    """The one genuine import-surface contract worth keeping: the 0.7.0
    removal of require_vram / require_compute_capability surfaces as an
    ImportError with migration guidance (not a bare AttributeError)."""
    import gen_worker.capability as cap

    for name in ("require_vram", "require_compute_capability", "require_cuda_library"):
        with pytest.raises(ImportError, match="0.7.0"):
            getattr(cap, name)
    # A genuinely-unknown attribute is still an AttributeError.
    with pytest.raises(AttributeError):
        cap.totally_made_up_symbol  # noqa: B018
