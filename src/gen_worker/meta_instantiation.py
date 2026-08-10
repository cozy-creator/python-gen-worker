"""pgw#1080 (pgw#1056 authoring corollary 2): the META-INSTANTIATION GATE.

The zero-download forge instantiates a model's STRUCTURE on the meta device and
exports it there — no checkpoint values ever enter the process that traces and
compiles. That property is invisible when it breaks: a module that does real
tensor work at construction, or (worse) lazily at CALL time under an explicit
device pin, silently materializes real tensors and the forge quietly starts
needing weights again. Nothing downstream notices, because the export still
succeeds.

**So the property is GATED, not hoped for.** Run the instantiation AND the
trace inside :func:`guard`, and any tensor that lands on a real device is a
typed refusal naming the site that allocated it.

WHY THE SCOPE IS "INSTANTIATION *OR* TRACE" (ie#628's widening)
---------------------------------------------------------------
An ``__init__``-inspecting gate is not enough, and z-image is the measured
counterexample that made the point: upstream's ``RopeEmbedder`` is not an
``nn.Module`` at all, holds ``self.freqs_cis = None``, and builds its tables on
FIRST CALL inside ``with torch.device("cpu")`` — a pin that overrides any
ambient meta context. Nothing happens at ``__init__``; the violation happens
mid-trace. (z-image's endpoint has since bound those tables as buffers with no
device pin — ie#630 — so it is the family this gate must stay SILENT on. The
gate exists for the NEXT one.)

WHY A ``TorchFunctionMode`` AND NOT A DEVICE CONTEXT
---------------------------------------------------
``with torch.device("meta")`` is a DEFAULT, and a default is exactly what model
code overrides — explicitly (``torch.device("cpu")``) or implicitly
(``device=`` / ``.to("cuda")`` / ``.cpu()``). A mode sits above the factory
functions themselves, so it observes the tensor that was actually produced
rather than the default that was requested, and an override is therefore
visible instead of authoritative.

WHAT IS DELIBERATELY *NOT* REFUSED
----------------------------------
Fake tensors. ``FakeTensorMode`` produces tensors whose ``.device`` reads as a
real device by design (that is what makes them faithful to trace against) while
allocating nothing. Refusing them would refuse the very mechanism this gate
protects, so :func:`is_virtual` treats a fake tensor as virtual regardless of
what its device says.
"""

from __future__ import annotations

import contextlib
import traceback
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Tuple

from .api.errors import WorkerError

#: Devices a virtual tensor may claim. Anything else is real allocation.
VIRTUAL_DEVICES: Tuple[str, ...] = ("meta",)


class MetaMaterializationError(WorkerError):
    """A real-device tensor was materialized under the meta-instantiation gate.

    Named-axis refusal (the gw#577 pattern): the message carries WHAT was
    allocated (op, shape, dtype, device), WHERE in the endpoint's own code it
    happened, and the authoring rule that was broken — because the fix is
    always an authoring change, never a knob.
    """

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
    """What the gate saw. Empty ``events`` is the property holding."""

    phase: str = ""
    events: List[Materialization] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.events


def is_virtual(tensor: Any) -> bool:
    """Whether this tensor allocated nothing.

    Meta tensors by device; fake tensors by TYPE, because a fake tensor
    deliberately reports a real device while backing it with no storage.
    """
    fake_cls: Optional[type] = None
    try:
        from torch._subclasses.fake_tensor import FakeTensor

        fake_cls = FakeTensor
    except Exception:  # noqa: BLE001 — old/absent torch: fall back to device
        fake_cls = None
    if fake_cls is not None and isinstance(tensor, fake_cls):
        return True
    device = getattr(tensor, "device", None)
    if device is None:
        return True
    return str(getattr(device, "type", device)) in VIRTUAL_DEVICES


def _endpoint_site(skip: int = 0) -> str:
    """The innermost stack frame OUTSIDE this SDK — i.e. the endpoint's own
    code, which is where the authoring fix has to be made.

    Reporting an SDK frame would name the observer instead of the cause; the
    whole value of this refusal is the `file:line` an author can go and edit.
    """
    for frame in reversed(traceback.extract_stack()[:-(1 + skip)]):
        path = frame.filename.replace("\\", "/")
        if "/gen_worker/" in path or "/torch/" in path:
            continue
        if path.startswith("<"):
            # `<frozen runpy>`, `<frozen importlib._bootstrap>`, `<string>`:
            # interpreter scaffolding, not a line anyone can edit. Treating
            # one as the cause is how a torch-internal allocation gets
            # reported as an authoring violation (measured on the micro rig:
            # a 380-byte `empty` inside `torch.export` attributed to
            # `<frozen runpy>:88`).
            continue
        return f"{frame.filename}:{frame.lineno} in {frame.name}"
    return ""


@contextlib.contextmanager
def observe(phase: str, census: Optional[Census] = None) -> Iterator[Census]:
    """Record every real-device tensor produced inside the block.

    Observation only — it never raises, so a caller can census a phase and
    decide separately. :func:`guard` is the refusing wrapper.
    """
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
    """Refuse the FIRST real-device materialization in this block, typed.

    First rather than all: the census is for reporting, but a mint that has
    already allocated real weights has lost the property this gate exists to
    keep, and continuing would only produce a longer list of the same fact.

    ``actionable_only`` refuses only an allocation this gate can NAME a
    file:line for — i.e. one made by the endpoint's own code. It exists for
    the TRACE window (pgw#1080), where ``torch.export`` legitimately allocates
    small real tensors of its own inside the fake mode: measured on the micro
    rig, a 380-byte ``empty`` with no endpoint frame anywhere on the stack.
    Refusing that would fail every mint for something no author can fix,
    while the class the gate exists for — a device pin inside model code —
    always has a frame. Unattributable events stay in the census, so they are
    reported rather than dropped.
    """
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
    """Instantiate structure on meta.

    A DEFAULT, deliberately paired with :func:`guard` rather than trusted on
    its own — see the module docstring on why a default is not a property.
    """
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
