from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class TensorRTCompatibility:
    cuda_version: str
    tensorrt_version: str
    sm: str  # compute capability, e.g. "89"


@runtime_checkable
class OrtRuntime(Protocol):
    """Worker-injected ONNX Runtime handle (tenant-visible contract)."""

    def run(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class TrtRuntime(Protocol):
    """Worker-injected TensorRT handle (tenant-visible contract)."""

    def run(self, *args: Any, **kwargs: Any) -> Any: ...

