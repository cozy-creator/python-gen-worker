"""Is this host's NVML usable by torch? A driver/library version mismatch (`nvidia-smi` -> "Driver/library version mismatch") does not stop raw CUDA — a matmul and `nn.Module.to("cuda")` both work — b..."""

from __future__ import annotations


def nvml_is_healthy() -> bool:
    try:
        import ctypes

        return ctypes.CDLL("libnvidia-ml.so.1").nvmlInit_v2() == 0
    except Exception:  # noqa: BLE001 - unreadable NVML is unhealthy NVML
        return False


__all__ = ["nvml_is_healthy"]
