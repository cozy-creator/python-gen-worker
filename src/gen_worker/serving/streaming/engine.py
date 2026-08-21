"""``ctx.load``'s engine: chunk store -> pinned staging -> device, no files."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING, Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple,
)

from . import keymap as _keymap
from . import skeleton as _skeleton
from .source import StreamedTensor, TensorStream, WeightStore, component_of
from .staging import DEFAULT_BUFFER_BYTES, DEFAULT_BUFFERS, StagingPool

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

logger = logging.getLogger(__name__)

_TORCH_DTYPE: Mapping[str, str] = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}


_WIDE_FLOATS = ("float64", "float32", "float16", "bfloat16")


def _is_wide_float(dtype: Any) -> bool:
    import torch

    return any(dtype is getattr(torch, name, None) for name in _WIDE_FLOATS)


class LoadError(RuntimeError):
    """A streamed load could not be completed as specified."""


class LaneDtypeUnmet(LoadError):
    """The loaded pipeline is not the dtype its lane declares."""


class NameMismatch(LoadError):
    """The checkpoint's tensor names and the skeleton's do not correspond."""


@dataclass(slots=True)
class LoadReport:
    """What the load actually did — the ``model_load`` span's attributes."""

    weights_streamed_bytes: int = 0
    weights_stream_gbps: float = 0.0
    source: str = "bridge"
    staging: str = "pageable"
    io: str = "buffered"
    containers: int = 0
    tensors: int = 0
    windows: int = 0
    seconds: float = 0.0
    dtypes: Tuple[str, ...] = ()
    cast_to_lane: int = 0

    def attributes(self) -> Dict[str, object]:
        return {
            "weights_streamed_bytes": self.weights_streamed_bytes,
            "weights_stream_gbps": round(self.weights_stream_gbps, 3),
            "weights_stream_source": self.source,
            "staging": self.staging,
            "io": self.io,
            "containers": self.containers,
            "tensors": self.tensors,
            "cast_to_lane": self.cast_to_lane,
        }


@dataclass(slots=True)
class _Placement:

    name: str
    offset: int
    nbytes: int
    flat: "torch.Tensor"

    @property
    def end(self) -> int:
        return self.offset + self.nbytes


@dataclass(slots=True)
class _Slot:

    owner: Any
    leaf: str
    parameter: bool


def _torch_dtype(spelling: str, name: str) -> "torch.dtype":
    import torch

    attribute = _TORCH_DTYPE.get(spelling.upper())
    if attribute is None:
        raise LoadError(
            f"{name}: container dtype {spelling!r} has no byte-addressable "
            f"torch dtype. Sub-byte and block-quantized containers are the "
            f"external-binary class (the #1303 ladder's tier 3), not the "
            f"serving pytorch path"
        )
    dtype = getattr(torch, attribute, None)
    if dtype is None:
        raise LoadError(
            f"{name}: this torch build has no {attribute}, so dtype "
            f"{spelling!r} cannot be loaded verbatim (and converting it "
            f"would be the store's job, not the loader's)"
        )
    return dtype  # type: ignore[no-any-return]


def _slots(module: "torch.nn.Module") -> Dict[str, _Slot]:
    found: Dict[str, _Slot] = {}
    for name, _ in module.named_parameters(remove_duplicate=False):
        owner, leaf = _owner_of(module, name)
        found[name] = _Slot(owner=owner, leaf=leaf, parameter=True)
    for name, _ in module.named_buffers(remove_duplicate=False):
        owner, leaf = _owner_of(module, name)
        found[name] = _Slot(owner=owner, leaf=leaf, parameter=False)
    return found


def _owner_of(root: "torch.nn.Module", qualified: str) -> Tuple[Any, str]:
    parent, _, leaf = qualified.rpartition(".")
    owner = root.get_submodule(parent) if parent else root
    return owner, leaf


def _install(slot: _Slot, tensor: "torch.Tensor") -> None:
    import torch

    if slot.parameter:
        held = slot.owner._parameters.get(slot.leaf)
        kind = type(held) if held is not None else torch.nn.Parameter
        slot.owner._parameters[slot.leaf] = kind(tensor, requires_grad=False)
    else:
        slot.owner._buffers[slot.leaf] = tensor


def _flat_bytes(tensor: "torch.Tensor") -> "torch.Tensor":
    import torch

    return tensor.view(-1).view(torch.uint8)


class StreamingLoader:

    def __init__(
        self,
        store: WeightStore,
        *,
        device: Any = "cuda",
        io: str = "buffered",
        buffer_bytes: int = DEFAULT_BUFFER_BYTES,
        buffers: int = DEFAULT_BUFFERS,
    ) -> None:
        if io not in ("buffered", "direct"):
            raise LoadError(f"io must be 'buffered' or 'direct', not {io!r}")
        self._store = store
        self._device = device
        self._io = io
        self._buffer_bytes = int(buffer_bytes)
        self._buffers = int(buffers)
        self._planned: List[Tuple[str, str]] = []
        self.last_report: Optional[LoadReport] = None

    def build(self, pipeline_cls: type, *, checkpoint_dir: Path, lane: Any) -> Any:
        """Meta skeleton, then weights streamed into it."""
        import torch

        started = time.perf_counter()
        device = torch.device(self._device)
        compute_dtype = _lane_compute_dtype(lane)
        built = _skeleton.build(
            pipeline_cls, Path(checkpoint_dir), compute_dtype=compute_dtype)
        report = LoadReport(
            io=self._io,
            source=str(getattr(self._store, "KIND", type(self._store).__name__)),
        )
        recasts: List[Tuple[_Slot, str, Any]] = []

        with StagingPool(
            device,
            buffer_bytes=self._buffer_bytes,
            buffers=self._buffers,
        ) as pool:
            report.staging = pool.staging
            for component, container in self._plan(built.modules):
                self._stream_container(
                    built.modules[component],
                    component,
                    container,
                    pool=pool,
                    device=device,
                    report=report,
                    compute_dtype=compute_dtype,
                    recasts=recasts,
                    lane_exempt=_lane_exempt(built, component),
                )
            report.containers = len(self._planned)
        self._cast_to_lane(recasts, compute_dtype, report)

        for component, module in built.modules.items():
            quantization = built.quantized.get(component)
            if quantization is not None:
                _skeleton.finish_quantized(module, quantization)
            _skeleton.retie(module)
            survivors = _skeleton.meta_survivors(module)
            if survivors:
                aliases = [
                    name for name in survivors
                    if name in set(_skeleton.tied_names(module))
                ]
                raise NameMismatch(
                    f"component {component!r} ({type(module).__name__}): "
                    f"{len(survivors)} tensor(s) were never filled and are "
                    f"still on meta after every container was streamed — "
                    f"{', '.join(survivors[:8])}"
                    + (" …" if len(survivors) > 8 else "")
                    + ". Weight tying was re-established first (pgw#1626), so "
                    "an untied alias is not the cause"
                    + (
                        f"; {', '.join(aliases[:4])} "
                        f"{'is a name' if len(aliases) == 1 else 'are names'} "
                        f"{type(module).__name__} TIES to another parameter, "
                        f"so the parameter it aliases went unfilled too"
                        if aliases else ""
                    )
                )

        report.seconds = time.perf_counter() - started
        if report.seconds > 0:
            report.weights_stream_gbps = (
                report.weights_streamed_bytes / report.seconds / 1e9
            )
        self.last_report = report
        self._place_uninstalled(built.pipeline, device)
        self._assert_lane_dtype(built.modules, compute_dtype, built)
        logger.info(
            "ctx.load: %s resident on %s — %.2f GiB streamed in %.2fs "
            "(%.2f GB/s, staging=%s io=%s, %d tensors over %d windows)",
            pipeline_cls.__name__,
            device,
            report.weights_streamed_bytes / (1 << 30),
            report.seconds,
            report.weights_stream_gbps,
            report.staging,
            report.io,
            report.tensors,
            report.windows,
        )
        return built.pipeline

    def _place_uninstalled(self, pipeline: Any, device: Any) -> None:
        import torch

        target = torch.device(device)
        if target.type == "meta":
            return
        moved = 0
        seen: set[int] = set()
        components = getattr(pipeline, "components", None) or {}
        roots = [m for m in components.values() if isinstance(m, torch.nn.Module)]
        if not roots and isinstance(pipeline, torch.nn.Module):
            roots = [pipeline]
        for root in roots:
            for module in root.modules():
                if id(module) in seen:
                    continue
                seen.add(id(module))
                for leaf, held in list(module._buffers.items()):
                    if (held is not None and held.device != target
                            and held.device.type != "meta"):
                        module._buffers[leaf] = held.to(target)
                        moved += 1
        if moved:
            logger.info(
                "ctx.load: placed %d tensor(s) the container did not carry "
                "onto %s (pgw#1454)", moved, target,
            )

    def _plan(self, modules: Mapping[str, Any]) -> List[Tuple[str, str]]:
        planned: List[Tuple[str, str]] = []
        for container in self._store.containers():
            component = component_of(container)
            if component in modules:
                planned.append((component, container))
        missing = sorted(set(modules) - {component for component, _ in planned})
        if missing:
            raise NameMismatch(
                f"no tensor container in the store belongs to component(s) "
                f"{', '.join(missing)}; their parameters would stay on meta"
            )
        self._planned = planned
        return planned

    def _stream_container(
        self,
        module: Any,
        component: str,
        container: str,
        *,
        pool: StagingPool,
        device: Any,
        report: LoadReport,
        compute_dtype: Any = None,
        recasts: Optional[List[Tuple[_Slot, str, Any]]] = None,
        lane_exempt: FrozenSet[str] = frozenset(),
    ) -> None:
        import torch

        stream: TensorStream = self._store.open(
            container, direct=self._io == "direct"
        )
        entries: Sequence[StreamedTensor] = stream.tensors
        if not entries:
            return

        slots = _slots(module)
        renames = _keymap.migration(module, (entry.name for entry in entries))
        placements: List[_Placement] = []
        unexpected: List[str] = []
        seen: set[str] = set()
        dtypes: set[str] = set(report.dtypes)

        for entry in entries:
            name = renames.get(entry.name, entry.name)
            slot = slots.get(name)
            if slot is None:
                unexpected.append(
                    entry.name if name == entry.name
                    else f"{entry.name} (migrated to {name})"
                )
                continue
            dtype = _torch_dtype(entry.dtype, f"{component}/{entry.name}")
            dtypes.add(entry.dtype.upper())
            destination = torch.empty(
                tuple(int(dim) for dim in entry.shape), dtype=dtype, device=device
            )
            flat = _flat_bytes(destination)
            if flat.numel() != entry.nbytes:
                raise LoadError(
                    f"{component}/{entry.name}: the container says "
                    f"{entry.nbytes} bytes, a {dtype} tensor of "
                    f"{tuple(entry.shape)} holds {flat.numel()}"
                )
            pool.track(destination)
            _install(slot, destination)
            # pgw#1638: a tensor a QUANTIZER owns is out of the lane cast's
            # scope, whatever its dtype. Property 3 above already says a
            # quantized container is the lane's own bytes — but it said it of
            # the WEIGHT only, which fp8 satisfied by not being a wide float.
            # An `hf.fp8-blockwise@1` scale grid is F32 beside a bf16 lane, so
            # the cast would have rounded 357 block scales to bf16 and
            # `_assert_lane_dtype` would then have passed: plausible numbers,
            # wrong ones, green. The rule's dtypes are the rule's.
            if (recasts is not None and compute_dtype is not None
                    and dtype is not compute_dtype and _is_wide_float(dtype)
                    and name not in lane_exempt):
                recasts.append((slot, f"{component}/{entry.name}", dtype))
            seen.add(entry.name)
            placements.append(
                _Placement(
                    name=entry.name,
                    offset=int(entry.offset),
                    nbytes=int(entry.nbytes),
                    flat=flat,
                )
            )

        if unexpected:
            raise NameMismatch(
                f"{container}: {len(unexpected)} tensor(s) name nothing in "
                f"component {component!r} — {', '.join(sorted(unexpected)[:8])}"
                + (" …" if len(unexpected) > 8 else "")
                + ". A name the skeleton cannot place is a checkpoint the "
                "code does not match; ctx.load never guesses. "
                + (
                    f"{type(module).__name__}'s own library migration was "
                    f"applied first and renamed {len(renames)} key(s), so this "
                    f"is a difference the library does not know about "
                    f"(pgw#1453)."
                    if renames else
                    f"{type(module).__name__}'s own library migration was "
                    f"applied first and renamed nothing (pgw#1453)."
                )
            )

        report.dtypes = tuple(sorted(dtypes))
        placements.sort(key=lambda placement: placement.offset)
        report.tensors += len(placements)
        report.windows += self._walk(stream, placements, pool=pool, report=report)

    def _walk(
        self,
        stream: TensorStream,
        placements: List[_Placement],
        *,
        pool: StagingPool,
        report: LoadReport,
    ) -> int:
        position = placements[0].offset
        finish = placements[-1].end
        window = pool.buffer_bytes
        first = 0
        windows = 0

        while position < finish:
            count = min(window, finish - position)
            slot = pool.acquire()
            stream.readinto(position, count, slot.view[:count])
            windows += 1
            report.weights_streamed_bytes += count

            index = first
            while index < len(placements) and placements[index].offset < position + count:
                placement = placements[index]
                if placement.end <= position:
                    index += 1
                    first = index
                    continue
                low = max(placement.offset, position)
                high = min(placement.end, position + count)
                pool.copy_out(
                    slot,
                    low - position,
                    placement.flat,
                    low - placement.offset,
                    high - low,
                )
                if placement.end <= position + count:
                    first = index + 1
                index += 1

            pool.release(slot)
            position += count

        return windows

    @staticmethod
    def _cast_to_lane(
        recasts: List[Tuple[_Slot, str, Any]],
        compute_dtype: Any,
        report: LoadReport,
    ) -> None:
        if not recasts:
            return
        counts: Dict[str, int] = {}
        while recasts:
            slot, name, stored = recasts.pop()
            held = (
                slot.owner._parameters.get(slot.leaf) if slot.parameter
                else slot.owner._buffers.get(slot.leaf)
            )
            if held is None:  # pragma: no cover — defensive
                continue
            _install(slot, held.to(compute_dtype))
            key = str(stored).rsplit(".", 1)[-1]
            counts[key] = counts.get(key, 0) + 1
            report.cast_to_lane += 1
        rendered = ", ".join(
            f"{count} {dtype}" for dtype, count in sorted(counts.items()))
        sentence = (
            f"{report.cast_to_lane} tensor(s) were stored off-lane "
            f"({rendered}) and were cast to the lane's "
            f"{str(compute_dtype).rsplit('.', 1)[-1]} at load; the tree was "
            f"not converted through the lane's layout contract before it "
            f"reached this pod"
        )
        logger.warning("ctx.load: %s", sentence)
        try:
            from ... import activity

            activity.emit_event(
                activity.KIND_SERVE_DEGRADE, sentence,
                phase="container_dtype_off_lane", step=report.cast_to_lane,
            )
        except Exception:  # noqa: BLE001 — a confession never fails a load
            logger.debug("ctx.load: could not emit the off-lane dtype event",
                         exc_info=True)

    @staticmethod
    def _assert_lane_dtype(
        modules: Mapping[str, Any], compute_dtype: Any,
        built: Optional["_skeleton.Skeleton"] = None,
    ) -> None:
        if compute_dtype is None:
            return
        offenders: List[str] = []
        for component, module in modules.items():
            exempt = _lane_exempt(built, component)
            held = list(module.named_parameters(remove_duplicate=False))
            held += list(module.named_buffers(remove_duplicate=False))
            for name, tensor in held:
                if tensor is None or not _is_wide_float(tensor.dtype):
                    continue
                if name in exempt:
                    continue
                if tensor.dtype is not compute_dtype:
                    offenders.append(f"{component}/{name}={tensor.dtype}")
                    if len(offenders) >= 8:
                        break
            if len(offenders) >= 8:
                break
        if offenders:
            raise LaneDtypeUnmet(
                f"the loaded pipeline is not the lane's {compute_dtype}: "
                f"{', '.join(offenders)}. A tensor reached the module by a "
                f"path this engine did not install and so was never cast — "
                f"serving it would mix dtypes inside one forward"
            )


def _lane_exempt(built: Any, component: str) -> FrozenSet[str]:
    """The tensor names in ``component`` a quant RULE owns, not the lane.

    Empty for every unquantized component, which is every component the
    fleet served before pgw#1638 — so this narrows nothing that was working.
    """
    quantized = getattr(built, "quantized", None) or {}
    quantization = quantized.get(component)
    return quantization.tensors if quantization is not None else frozenset()


def _lane_compute_dtype(lane: Any) -> Any:
    if lane is None:
        return None
    from ..context import _lane_torch_dtype

    dtype = _lane_torch_dtype(lane)
    return dtype if _is_wide_float(dtype) else None


__all__ = [
    "LaneDtypeUnmet",
    "LoadError",
    "LoadReport",
    "NameMismatch",
    "StreamingLoader",
]
