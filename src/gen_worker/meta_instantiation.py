"""The META-INSTANTIATION GATE."""

from __future__ import annotations

import contextlib
import traceback
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Tuple

from .api.errors import WorkerError

VIRTUAL_DEVICES: Tuple[str, ...] = ("meta",)


class MetaMaterializationError(WorkerError):
    """A real-device tensor was materialized under the meta-instantiation gate."""

    def __init__(
        self, *, phase: str, op: str, device: str, shape: str, dtype: str,
        site: str,
    ) -> None:
        self.phase = str(phase or "")
        self.op = str(op or "")
        self.device = str(device or "")
        self.shape = str(shape or "")
        self.dtype = str(dtype or "")
        self.site = str(site or "")
        super().__init__(
            f"meta-instantiation gate: during {self.phase}, {self.op} "
            f"materialized a REAL tensor on {self.device!r} "
            f"(shape={self.shape}, dtype={self.dtype}) at {self.site or '?'}. "
            f"The zero-download forge instantiates structure on meta and never "
            f"holds checkpoint values, so a real allocation here means this "
            f"model needs weights to be exported at all. Fix it in the "
            f"ENDPOINT: derived tables that are pure functions of config "
            f"belong in `register_buffer` at __init__ with NO device pin "
            f"(ie#630's `rope_buffers` is the worked example); an explicit "
            f"`with torch.device(...)` inside model code is itself the "
            f"violation to remove."
        )


@dataclass
class Materialization:
    """One observed real-device allocation."""

    op: str
    device: str
    shape: str
    dtype: str
    site: str


@dataclass
class Census:
    """What the gate saw."""

    phase: str = ""
    events: List[Materialization] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.events


def is_virtual(tensor: Any, _depth: int = 0) -> bool:
    """Whether this tensor allocated nothing."""
    fake_cls: Optional[type] = None
    try:
        from torch._subclasses.fake_tensor import FakeTensor

        fake_cls = FakeTensor
    except Exception:  # noqa: BLE001 — old/absent torch: fall back to device
        fake_cls = None
    if fake_cls is not None and isinstance(tensor, fake_cls):
        return True
    inner = _wrapped_tensors(tensor) if _depth < _WRAPPER_DEPTH else ()
    if inner:
        return all(is_virtual(held, _depth + 1) for held in inner)
    device = getattr(tensor, "device", None)
    if device is None:
        return True
    return str(getattr(device, "type", device)) in VIRTUAL_DEVICES


_WRAPPER_DEPTH = 4


def _wrapped_tensors(tensor: Any) -> Tuple[Any, ...]:
    try:
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass

        if not is_traceable_wrapper_subclass(tensor):
            return ()
        names, _ctx = tensor.__tensor_flatten__()
    except Exception:  # noqa: BLE001 — an unwalkable object is not a subclass
        return ()
    held = tuple(
        value for value in (getattr(tensor, str(name), None) for name in names)
        if value is not None)
    return held


def _endpoint_site(skip: int = 0) -> str:
    for frame in reversed(traceback.extract_stack()[:-(1 + skip)]):
        path = frame.filename.replace("\\", "/")
        if "/gen_worker/" in path or "/torch/" in path:
            continue
        if path.startswith("<"):
            continue
        return f"{frame.filename}:{frame.lineno} in {frame.name}"
    return ""


@contextlib.contextmanager
def observe(phase: str, census: Optional[Census] = None) -> Iterator[Census]:
    """Record every real-device tensor produced inside the block."""
    import torch
    from torch.overrides import TorchFunctionMode

    out = census if census is not None else Census()
    out.phase = str(phase or out.phase)

    class _Watch(TorchFunctionMode):
        def __torch_function__(
            self, func: Any, types: Any, args: Any = (), kwargs: Any = None,
        ) -> Any:
            result = func(*args, **(kwargs or {}))
            if isinstance(result, torch.Tensor) and not is_virtual(result):
                out.events.append(Materialization(
                    op=getattr(func, "__name__", str(func)),
                    device=str(result.device),
                    shape=str(tuple(result.shape)),
                    dtype=str(result.dtype),
                    site=_endpoint_site(skip=1),
                ))
            return result

    with _Watch():
        yield out


@contextlib.contextmanager
def guard(phase: str, *, actionable_only: bool = False) -> Iterator[Census]:
    """Refuse the FIRST real-device materialization in this block, typed."""
    with observe(phase) as census:
        yield census
    events = [e for e in census.events if e.site] if actionable_only \
        else list(census.events)
    if events:
        first = events[0]
        raise MetaMaterializationError(
            phase=census.phase, op=first.op, device=first.device,
            shape=first.shape, dtype=first.dtype, site=first.site)


@contextlib.contextmanager
def meta_device() -> Iterator[None]:
    """Instantiate structure on meta."""
    import torch

    with torch.device("meta"):
        yield


__all__ = [
    "Census",
    "Materialization",
    "MetaMaterializationError",
    "VIRTUAL_DEVICES",
    "guard",
    "is_virtual",
    "meta_device",
    "observe",
]
