"""Is this host's NVML usable by torch?

A driver/library version mismatch (`nvidia-smi` -> "Driver/library version
mismatch") does not stop raw CUDA — a matmul and `nn.Module.to("cuda")` both
work — but torch's PeerToPeerAccess check calls `nvmlInit_v2_` and raises
`INTERNAL ASSERT FAILED` under some import orders, notably inside a diffusers
`pipeline.to("cuda")` and inside `Module.to` under pytest.

Rows that need a real device probe this rather than catching the exception:
catching it would swallow a genuine regression on a healthy box, while probing
names the HOST as the reason and lets the census classify it.
"""

from __future__ import annotations


def nvml_is_healthy() -> bool:
    try:
        import ctypes

        return ctypes.CDLL("libnvidia-ml.so.1").nvmlInit_v2() == 0
    except Exception:  # noqa: BLE001 - unreadable NVML is unhealthy NVML
        return False


__all__ = ["nvml_is_healthy"]
