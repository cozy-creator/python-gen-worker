"""``ctx.load``'s engine: chunk store -> pinned staging -> device, no files.

Paul's 2026-08-19 ruling: filling tensor bytes into a real safetensors file so
``from_pretrained`` can mmap it is *"the completely wrong design"*. This is the
replacement. A serving pytorch endpoint reads its weights out of the CAS
objects it already holds and lands them on the device; nothing is written, and
``materialized_view`` is not on this path at all.

Four properties, each load-bearing:

1. **File order, never module order.** ``load_state_dict`` iterates the
   MODULE, which walks the checkpoint's bytes in whatever order the module
   tree happens to have — the scrambled access pattern the load-order ruling
   measured at ~10x worse. This engine walks the container's byte range
   forward, once, and assigns by NAME as it goes. Canonical packaging is
   load-order-optimal, so sequential streaming preserves that win by
   construction rather than by luck.
2. **Windows, not tensors.** A window is a staging buffer's worth of
   consecutive file bytes; the tensors that overlap it are scattered out of
   it. Small tensors coalesce into one read for free (SDXL has ~2.5k of
   them), and a tensor larger than the window simply spans several.
3. **dtype passthrough.** Bytes land verbatim in the container's own dtype —
   bf16 stays bf16, fp8 stays fp8 with its scale tensors beside it. Any
   conversion is the STORE's contract-negotiation job (the lane IS a tensorfs
   layout contract), never load time. A container that disagrees with the
   active lane is reported, not silently repaired.
4. **Nothing survives on meta.** Every parameter and buffer is accounted for
   by name. A missing name would serve uninitialized memory on the first
   request, so it is a typed refusal naming the names.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import keymap as _keymap
from . import skeleton as _skeleton
from .source import StreamedTensor, TensorStream, WeightStore, component_of
from .staging import DEFAULT_BUFFER_BYTES, DEFAULT_BUFFERS, StagingPool

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

logger = logging.getLogger(__name__)

#: safetensors' own dtype spellings -> torch dtypes. Passthrough only: this
#: table exists to ALLOCATE the destination, never to convert anything.
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


class LoadError(RuntimeError):
    """A streamed load could not be completed as specified."""


class NameMismatch(LoadError):
    """The checkpoint's tensor names and the skeleton's do not correspond."""


@dataclass(slots=True)
class LoadReport:
    """What the load actually did — the ``model_load`` span's attributes."""

    weights_streamed_bytes: int = 0
    weights_stream_gbps: float = 0.0
    #: Which byte source produced the number above — ``native`` or ``bridge``.
    #: Unlabelled, the two differ by ~10x and read as the same measurement.
    source: str = "bridge"
    staging: str = "pageable"
    io: str = "buffered"
    containers: int = 0
    tensors: int = 0
    windows: int = 0
    seconds: float = 0.0
    dtypes: Tuple[str, ...] = ()

    def attributes(self) -> Dict[str, object]:
        return {
            "weights_streamed_bytes": self.weights_streamed_bytes,
            "weights_stream_gbps": round(self.weights_stream_gbps, 3),
            "weights_stream_source": self.source,
            "staging": self.staging,
            "io": self.io,
            "containers": self.containers,
            "tensors": self.tensors,
        }


@dataclass(slots=True)
class _Placement:
    """One tensor's byte span in the file and the memory it lands in."""

    name: str
    offset: int
    nbytes: int
    flat: "torch.Tensor"

    @property
    def end(self) -> int:
        return self.offset + self.nbytes


@dataclass(slots=True)
class _Slot:
    """Where one name lives in the skeleton."""

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
    """name -> where it lives. Built once per component, off the MODULE — so
    assignment is a dict lookup during a forward byte walk, never a tree walk
    per tensor."""
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
    """Replace the meta placeholder with real memory.

    Assignment rather than ``copy_``: the placeholder has no storage to copy
    into, and this is the same move ``load_state_dict(assign=True)`` makes.
    ``requires_grad`` stays False — a serving pipeline never backpropagates,
    and an autograd-tracking parameter would silently hold a graph.
    """
    import torch

    if slot.parameter:
        held = slot.owner._parameters.get(slot.leaf)
        kind = type(held) if held is not None else torch.nn.Parameter
        slot.owner._parameters[slot.leaf] = kind(tensor, requires_grad=False)
    else:
        slot.owner._buffers[slot.leaf] = tensor


def _flat_bytes(tensor: "torch.Tensor") -> "torch.Tensor":
    """A flat uint8 view of a freshly allocated contiguous tensor."""
    import torch

    return tensor.view(-1).view(torch.uint8)


class StreamingLoader:
    """The engine behind ``ctx.load`` — the pgw#1382 ``LoaderEngine`` seam.

    Constructed by the WORKER, which supplies the byte source and the device:
    placement is a worker decision (pgw#1372's residency manager admits the
    exact weight byte size before a single allocation), and author code names
    no device anywhere. ``build`` is the whole surface.
    """

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

    # -- the LoaderEngine protocol -----------------------------------------

    def build(self, pipeline_cls: type, *, checkpoint_dir: Path, lane: Any) -> Any:
        """Meta skeleton, then weights streamed into it. Returns a genuine
        ``pipeline_cls`` with everything resident on the worker's device."""
        import torch

        started = time.perf_counter()
        device = torch.device(self._device)
        built = _skeleton.build(pipeline_cls, Path(checkpoint_dir))
        report = LoadReport(
            io=self._io,
            source=str(getattr(self._store, "KIND", type(self._store).__name__)),
        )

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
                )
            report.containers = len(self._planned)

        for component, module in built.modules.items():
            survivors = _skeleton.meta_survivors(module)
            if survivors:
                raise NameMismatch(
                    f"component {component!r}: {len(survivors)} tensor(s) are "
                    f"still on meta after the checkpoint's containers were "
                    f"streamed — the checkpoint does not carry "
                    f"{', '.join(survivors[:8])}"
                    + (" …" if len(survivors) > 8 else "")
                )

        report.seconds = time.perf_counter() - started
        if report.seconds > 0:
            report.weights_stream_gbps = (
                report.weights_streamed_bytes / report.seconds / 1e9
            )
        self.last_report = report
        self._place_uninstalled(built.pipeline, device)
        self._warn_on_lane(lane, report)
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
        """Place tensors the CONTAINER did not carry (pgw#1454).

        This engine installs each tensor the container names, straight onto the
        device. A tensor that is NOT in the container is never visited, so it
        materialises wherever `__init__` put it — the host — and nothing moved
        it. `from_pretrained` cannot have this bug: it builds on CPU and then
        `.to(device)` moves parameters and buffers together, whatever their
        provenance.

        Measured: CLIP's `embeddings.position_ids` is a NON-PERSISTENT buffer,
        so a modern `state_dict()` omits it by design (sd1.5's raw mirror
        carries 197 keys, the reconverted tree 196). The engine then served a
        text encoder whose weights were all on CUDA and whose position ids were
        on the CPU, and the first `nn.Embedding` forward died on the mismatch.
        A conversion step that is individually correct produced a tree this
        loader could not serve.

        Deliberately NOT `pipeline.to(device)`: that would walk and re-copy
        everything this engine just streamed, throwing away the whole point.
        Only what is still off-target moves, and only once.
        """
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
                    # `meta` is left ALONE on purpose: a tensor still on meta
                    # was never installed, and moving it would fabricate
                    # uninitialised memory and hide the very mismatch the
                    # survivors check above exists to raise.
                    if (held is not None and held.device != target
                            and held.device.type != "meta"):
                        module._buffers[leaf] = held.to(target)
                        moved += 1
        if moved:
            logger.info(
                "ctx.load: placed %d tensor(s) the container did not carry "
                "onto %s (pgw#1454)", moved, target,
            )

    # -- planning ----------------------------------------------------------

    def _plan(self, modules: Mapping[str, Any]) -> List[Tuple[str, str]]:
        """(component, container) in MANIFEST order.

        Manifest order is the packaging order, which for a canonical snapshot
        is contract memory order. Reading containers in it keeps the whole
        load sequential across component boundaries too, not just inside one.
        """
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

    # -- the file-order walk -----------------------------------------------

    def _stream_container(
        self,
        module: Any,
        component: str,
        container: str,
        *,
        pool: StagingPool,
        device: Any,
        report: LoadReport,
    ) -> None:
        import torch

        stream: TensorStream = self._store.open(
            container, direct=self._io == "direct"
        )
        entries: Sequence[StreamedTensor] = stream.tensors
        if not entries:
            return

        slots = _slots(module)
        # pgw#1453: the checkpoint's names may predate the installed library.
        # `from_pretrained` absorbs that difference; this engine installs by
        # EXACT name, so it asks the SAME library for the SAME migration rather
        # than matching nothing (sd1.5's text_encoder: 0 of 197). Empty for
        # every checkpoint the installed version already spells.
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
        # `tensors` is ascending file offset BY CONTRACT (tensorfs#115). The
        # sort is belt-and-braces against a source that forgets: without it a
        # scrambled walk would still be CORRECT and merely slow, which is the
        # regression that hides forever.
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
        """Read the container's data range forward once, scattering as we go.

        Everything between the first tensor's offset and the last tensor's end
        is read, including any alignment padding between tensors: a window is
        a byte range, not a tensor list, and skipping a few padding bytes
        would cost a seek to save a memcpy.
        """
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

    # -- the lane contract -------------------------------------------------

    @staticmethod
    def _warn_on_lane(lane: Any, report: LoadReport) -> None:
        """The lane IS the dtype (there is no ``torch_dtype=`` to pass).

        A container whose floating dtype is not the lane's means the tree was
        not converted through the active contract before it reached this pod.
        That is a STORE defect — reported here, never repaired here, because
        a load-time conversion is exactly the byte-rewriting this whole
        design deleted.
        """
        declared = getattr(lane, "dtype", None)
        if declared is None or not report.dtypes:
            return
        spelling = str(declared).rsplit(".", 1)[-1].upper()
        floating = {
            name for name in report.dtypes if name.startswith(("F", "BF"))
        }
        aliases = {spelling, {"BFLOAT16": "BF16", "FLOAT16": "F16",
                              "FLOAT32": "F32"}.get(spelling, spelling)}
        if floating and not (floating & aliases):
            logger.warning(
                "ctx.load: the active lane declares dtype %s but this "
                "checkpoint's containers carry %s — the tree was not "
                "converted through the lane's layout contract before it "
                "reached this pod; loading verbatim (conversion is the "
                "store's job, never the loader's)",
                declared,
                ", ".join(sorted(floating)),
            )


__all__ = [
    "LoadError",
    "LoadReport",
    "NameMismatch",
    "StreamingLoader",
]
