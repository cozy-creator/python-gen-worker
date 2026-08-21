"""The checkpoint juggle: a backing swap under fixed virtual addresses.

pgw#1607, implementing pgw#1602's multi-checkpoint scope addition. Within a
lane, every admitted checkpoint shares the canonical arena layout — derived
from the SERVING MODULE TREE by :func:`~gen_worker.models.arena_residency.
plan_layout`, never from any file — so a checkpoint switch is a backing swap:
the compiled artifact's pointers, the module's parameter identities and the
reservation's addresses are all invariant, and the cost is byte transfer only.

Design decisions this module encodes (the tracker section is the argument;
this is the executable half):

* **D1 admission**: a checkpoint joins a lane's juggle set only after a
  header-only shape-identity proof against the lane template's slot table.
  The tensor-layout contract pins keys/dtypes/rank, NOT concrete shapes or
  file offsets — so identity is proven per checkpoint, and file geometry is
  normalized away by keying byte ranges to canonical arena offsets.
* **D3 swap = in-place refill**: regions that are BACKED are overwritten
  under their fixed addresses (pure H2D; no unmap/map round trip whose chunks
  would be recycled into an identical mapping). Regions that are UNBACKED
  swap for free — their refill source repoints.
* **D5 warm tier**: one pinned host ARENA IMAGE per (lane x checkpoint),
  laid out exactly as the layout (same offsets), so a swap refill is one
  contiguous copy per region. The image doubles as the streamed tier's host
  mirror — no duplicate RAM. Host RAM sizes the warm set adaptively
  (``MemAvailable`` minus a floor, read at ingest); eviction is LRU with
  hysteresis (a pressure-evicted image is not rebuilt within the same
  pressure epoch — the checkpoint still serves, disk-direct).
* **D7 serving validity** (coordinator requirement, 2026-08-20): every
  region carries ``valid@X -> refilling(Y) -> valid@Y | INVALID``. A refill
  that dies partway leaves the region typed-INVALID and serving REFUSES —
  the franken-weights state (two checkpoints' interleaved bytes under one
  address) is unreachable, not detected after the fact. Recovery is a
  re-refill: the source is immutable, so the retry is idempotent.
* **D6 integrity split** (va#12 coordination): content-level correctness is
  THIS module's — per-region digests banked at ingest, compared on the test
  legs. varena's per-page signatures (when adopted) detect backing-level
  surprises only.

Boundaries, restated once: per-lane only (a cross-lane switch is a different
layout and a real switch); LoRA variants stay on the fold path (pgw#1571);
tensors outside the managed regions (non-leaf buffers such as
``position_ids`` — architecture-derived, not learned) are NOT swapped, and
the switch report says how many bytes that leaves untouched.

TWO TEMPLATES, AND WHICH ONE IS LIVE (pgw#1621)
-----------------------------------------------
:func:`admission_refusal` is the last-line byte gate, and it diffs an incoming
checkpoint's safetensors headers against a TEMPLATE. There are now two, and
they are at different stages, so this says plainly which is which rather than
letting a reader assume the newer one is the one running:

* :class:`~gen_worker.models.arena_residency.ArenaLayout` — the LIVE arm, and
  the only one anything constructs today. The template is derived from the
  SERVING MODULE TREE (``plan_layout``), which is why it can also state byte
  counts and DLPack dtypes: it is the arena this process actually allocated.
  :class:`CheckpointCatalog` and :class:`CheckpointJuggler` are built on it.
* :class:`~gen_worker._vendor.tensorfs.layout2.ExpectedHeader` — the v2 arm,
  AWAITING ITS PRODUCER. The template is the computed ``quant(topology)``
  header, and this arm is what tracker ``research/tensor-layout-v2/0-DESIGN-v2.md``
  row "juggle admission consumes v2" describes. Nothing in ``src/`` calls it
  yet — and nothing in ``src/`` calls the ArenaLayout arm either: this whole
  module has NO ``src/`` caller as of pgw#1621 (checked: only
  ``tests/test_checkpoint_juggle.py`` and
  ``benchmarks/checkpoint_juggle_pgw1607.py``). The serving-side wiring that
  builds a residency, admits a checkpoint and hands over a binding is pgw#1602's,
  not landed.

**THE WORKER ENFORCES; IT NEVER RE-DECIDES.** The v2 arm takes BOTH its inputs
— the expected header AND the admission verdict — from its caller. It computes
neither. ``quant(topology)`` is evaluated by the Go engine and by nothing else
(there is deliberately no pyo3 binding for the evaluator or the decision), and
the tri-state verdict is the hub bind gate's, arriving binding-carried on
``DeployBinding.lane_verdicts``. v1 shipped three copies of one matching rule
and the copies were free to disagree about an admit; a fourth one here, in the
shape of a header this module computed for itself or a verdict it inferred from
a clean diff, would undo that cut. So ``verdict=`` is a REQUIRED keyword on the
v2 arm — a missing verdict refuses, because the one direction that fails
silently is the permissive one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .._vendor.tensorfs.layout2 import ExpectedHeader, LayoutTensor
from .arena_residency import (
    ArenaLayout,
    ArenaResidency,
    RegionSpec,
    SlotSpec,
    dlpack_dtype,
)
from .safetensors_header import header_len_ok

logger = logging.getLogger(__name__)

_DL_UINT8 = (1, 8)

DEFAULT_HOST_FLOOR_BYTES = 4 << 30

_SAFETENSORS_DTYPES: Dict[str, Tuple[int, int]] = {
    "F64": (2, 64),
    "F32": (2, 32),
    "F16": (2, 16),
    "BF16": (4, 16),
    "I64": (0, 64),
    "I32": (0, 32),
    "I16": (0, 16),
    "I8": (0, 8),
    "U8": (1, 8),
    "BOOL": (6, 8),
}

_CASTABLE = {(2, 16), (2, 32), (2, 64), (4, 16)}


class JuggleRefusal(RuntimeError):
    """Typed refusal: the checkpoint cannot join this lane's juggle set."""


class RegionInvalid(RuntimeError):
    """Typed refusal: a serving region holds no single checkpoint's bytes."""


@dataclass(frozen=True)
class SlotSource:
    """One tensor's bytes as one checkpoint's files state them."""

    path: Path
    offset: int
    length: int
    dtype_code: int
    dtype_bits: int
    shape: Tuple[int, ...]
    #: The safetensors CONTAINER SPELLING, verbatim from the header (``"BF16"``,
    #: ``"F8_E4M3"``, ``"U8"``). pgw#1621: the v2 arm compares this against
    #: ``LayoutTensor.dtypes``, which is a list of container spellings, so the
    #: spelling is KEPT rather than reconstructed from ``(dtype_code,
    #: dtype_bits)`` — a reverse table would be a second producer of the fact
    #: the header already states, and the DLPack pair cannot express the
    #: spellings the arena has no placement for at all (fp8, packed fp4).
    spelling: str = ""


def read_manifest(
    directory: "str | Path", *, variant: Optional[str] = None
) -> Dict[str, SlotSource]:
    """``{tensor key: SlotSource}`` for one component directory."""
    root = Path(directory)
    tail = f".{variant}.safetensors" if variant else ".safetensors"
    files = sorted(p for p in root.glob("*.safetensors") if p.name.endswith(tail))
    if not variant:
        files = [p for p in files if p.name.count(".") == 1]
    if not files:
        raise FileNotFoundError(f"no *{tail} weight files under {root}")

    out: Dict[str, SlotSource] = {}
    for path in files:
        with path.open("rb") as fh:
            raw = fh.read(8)
            if len(raw) != 8:
                raise ValueError(f"{path}: truncated safetensors header length")
            (hlen,) = struct.unpack("<Q", raw)
            if not header_len_ok(hlen):
                raise ValueError(f"{path}: implausible safetensors header length {hlen}")
            header = json.loads(fh.read(hlen))
        base = 8 + hlen
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            spelling = str(meta["dtype"])
            code_bits = _SAFETENSORS_DTYPES.get(spelling)
            if code_bits is None:
                raise JuggleRefusal(
                    f"{path}: tensor {key!r} has dtype {spelling!r}, which this "
                    f"module has no placement for; refusing rather than guessing"
                )
            start, end = meta["data_offsets"]
            out[key] = SlotSource(
                path=path,
                offset=base + int(start),
                length=int(end) - int(start),
                dtype_code=code_bits[0],
                dtype_bits=code_bits[1],
                shape=tuple(int(d) for d in meta["shape"]),
                spelling=spelling,
            )
    return out


def merge_component_manifests(
    components: Dict[str, "str | Path"], *, variant: Optional[str] = None
) -> Dict[str, SlotSource]:
    """Manifests for a multi-component lane, keyed the way the facade keys."""
    if len(components) == 1:
        (directory,) = components.values()
        return read_manifest(directory, variant=variant)
    out: Dict[str, SlotSource] = {}
    for name, directory in components.items():
        for key, src in read_manifest(directory, variant=variant).items():
            out[f"{name}.{key}"] = src
    return out


def _slot_key(slot: SlotSpec) -> str:
    _root, _, rest = slot.leaf.partition(".")
    return f"{rest}.{slot.attr}" if rest else slot.attr


#: The hub bind gate's tri-state, exactly as it arrives on
#: ``DeployBinding.lane_verdicts``. Spelled here so the enforcement side and
#: the transport side share one vocabulary; this module never PRODUCES one.
LANE_VERDICT_SATISFIES = "satisfies"
LANE_VERDICT_DERIVABLE = "derivable"
LANE_VERDICT_INCOMPATIBLE = "incompatible"


def admission_refusal(
    template: "ArenaLayout | ExpectedHeader",
    manifest: Dict[str, SlotSource],
    *,
    verdict: Optional[str] = None,
    component: Optional[str] = None,
) -> Optional[str]:
    """None when the checkpoint may join this lane's juggle set; else the first
    refusal, naming the offending tensor.

    Two templates, and :func:`admission_refusal` is one gate over both so a
    future producer swaps the template without moving the call site. See the
    module docstring for which arm is live.

    ``ArenaLayout`` — the tree-derived template. ``verdict`` and ``component``
    do not apply and are refused if passed: the arena IS this process's own
    allocation, so there is no second party whose decision could be carried,
    and its slot table is already flat.

    ``ExpectedHeader`` — the computed ``quant(topology)`` template. ``verdict``
    is REQUIRED: the admission DECISION is the hub's and arrives
    binding-carried, so this arm enforces one rather than reaching a rival
    conclusion from a clean diff. ``component`` names which component of a
    multi-component layout the manifest holds; ``None`` compares the whole
    layout, keyed the way :func:`merge_component_manifests` keys it.
    """
    if isinstance(template, ExpectedHeader):
        return _expected_header_refusal(
            template, manifest, verdict=verdict, component=component
        )
    if verdict is not None or component is not None:
        raise TypeError(
            "admission_refusal(ArenaLayout, ...): verdict= and component= "
            "belong to the ExpectedHeader arm. An arena layout is this "
            "process's OWN allocation — there is no second party whose "
            "admission decision could be carried, and its slot table is flat."
        )
    for region in template.regions:
        for slot in region.slots:
            key = _slot_key(slot)
            src = manifest.get(key)
            if src is None:
                return f"tensor {key!r} is absent from the checkpoint"
            if src.shape != slot.shape:
                return (
                    f"tensor {key!r} is shaped {list(src.shape)} in the "
                    f"checkpoint and {list(slot.shape)} in the lane template — "
                    f"a different layout, i.e. a cross-lane switch"
                )
            same = (src.dtype_code, src.dtype_bits) == (slot.dtype_code, slot.dtype_bits)
            castable = (
                (src.dtype_code, src.dtype_bits) in _CASTABLE
                and (slot.dtype_code, slot.dtype_bits) in _CASTABLE
            )
            if not (same or castable):
                return (
                    f"tensor {key!r} is dtype ({src.dtype_code},{src.dtype_bits}) "
                    f"in the checkpoint and ({slot.dtype_code},{slot.dtype_bits}) "
                    f"in the lane; only float-to-float casts are representation "
                    f"choices — this is a different artifact"
                )
            if same and src.length != slot.nbytes:
                return (
                    f"tensor {key!r} states {src.length} bytes for shape "
                    f"{list(src.shape)}; the lane template holds {slot.nbytes}"
                )
    return None


def _expected_keys(
    expected: ExpectedHeader, component: Optional[str]
) -> Dict[str, LayoutTensor]:
    """The computed header's finite map, keyed the way a manifest is keyed.

    One component -> its own keys, unprefixed. Several -> ``<component>.<key>``.
    That is :func:`merge_component_manifests`'s rule verbatim, not a parallel
    one: two spellings of the same key set is how a gate starts refusing a
    checkpoint for being spelled differently.
    """
    if component is not None:
        return dict(expected.component(component))
    if len(expected.components) == 1:
        return dict(next(iter(expected.components.values())))
    return {
        f"{name}.{key}": entry
        for name, tensors in expected.components.items()
        for key, entry in tensors.items()
    }


def _expected_header_refusal(
    expected: ExpectedHeader,
    manifest: Dict[str, SlotSource],
    *,
    verdict: Optional[str],
    component: Optional[str],
) -> Optional[str]:
    """The v2 arm: enforce the hub's verdict, then diff against ``quant(topology)``.

    Key set, shape and dtype — the three facts the computed header states, and
    the three v1 could not state together (v1's schema was deliberately
    shapeless, which is how SDXL-inpainting's 9-channel ``conv_in`` legally
    admitted and served garbage).

    Two details that are load-bearing rather than incidental:

    * ``LayoutTensor.dtypes`` is a LIST. Several entries mean the rule is
      REFERENCE-TOLERANT — the reference packaging the topology was extracted
      from really did ship this key in a wider container (an ``I64``
      ``num_batches_tracked`` on a bf16 tree) — so acceptance is
      :meth:`LayoutTensor.accepts`, membership in the list, never equality with
      the first entry. Refusing the others would refuse the very checkpoint the
      topology came from.
    * ``LayoutTensor.optional`` marks a key a calibration MAY not have produced
      (``input_scale``, ``pre_quant_scale``). Absent is fine; PRESENT still has
      to match, because optional is about existence and never about shape.

    Extra tensors the header does not name are not refused: they are bytes this
    lane does not read, and the checkpoint's admissibility is a statement about
    what the lane needs. The hub's stamping already decided what the checkpoint
    IS.
    """
    if verdict is None:
        return (
            f"no admission verdict was carried for {expected.stamp}. The "
            f"verdict is the hub bind gate's and arrives binding-carried "
            f"(`DeployBinding.lane_verdicts`); the worker ENFORCES it and "
            f"never re-decides, so an absent verdict refuses rather than "
            f"being read as an admit"
        )
    if verdict != LANE_VERDICT_SATISFIES:
        if verdict == LANE_VERDICT_DERIVABLE:
            return (
                f"the hub bind gate answered {LANE_VERDICT_DERIVABLE!r} for "
                f"{expected.stamp}: these bytes reach the lane only through a "
                f"priced conversion job, and the endpoint serves the PRODUCED "
                f"artifact — never the source"
            )
        return (
            f"the hub bind gate answered {verdict!r} for {expected.stamp}; "
            f"only {LANE_VERDICT_SATISFIES!r} admits"
        )

    for key, entry in _expected_keys(expected, component).items():
        src = manifest.get(key)
        if src is None:
            if entry.optional:
                continue
            return (
                f"tensor {key!r} is absent from the checkpoint; "
                f"{expected.stamp} states it at shape {list(entry.shape)}"
            )
        if src.shape != entry.shape:
            return (
                f"tensor {key!r} is shaped {list(src.shape)} in the checkpoint "
                f"and {list(entry.shape)} in {expected.stamp} — a different "
                f"layout, i.e. a different lane"
            )
        if not src.spelling:
            return (
                f"tensor {key!r} carries no safetensors dtype spelling, so it "
                f"cannot be checked against {expected.stamp}'s "
                f"{list(entry.dtypes)}. Manifests come from `read_manifest`, "
                f"which keeps the header's own spelling"
            )
        if not entry.accepts(src.spelling):
            return (
                f"tensor {key!r} is dtype {src.spelling!r} in the checkpoint; "
                f"{expected.stamp} accepts {list(entry.dtypes)} — a different "
                f"quant rule, not a representation choice"
            )
    return None


def _mem_available_bytes() -> int:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:  # pragma: no cover - /proc always exists on this fleet
        pass
    return 0


class CheckpointImage:
    """One checkpoint's bytes, normalized into the canonical arena layout."""

    def __init__(
        self,
        checkpoint_id: str,
        layout: ArenaLayout,
        manifest: Dict[str, SlotSource],
        *,
        torch_mod: Any,
        varena_mod: Any = None,
        engine: Any = None,
        pin: bool = True,
    ) -> None:
        self.checkpoint_id = checkpoint_id
        self.layout = layout
        self.manifest = manifest
        self._torch = torch_mod
        self.casts = 0
        self.pinned = False
        self.region_digests: Dict[str, str] = {}

        n = int(layout.virtual_bytes)
        self._slab = None
        self._pool = None
        buffer = None
        if pin and varena_mod is not None:
            try:
                self._pool = varena_mod.SlabPool(n, pin="require")
                self._slab = self._pool.alloc(n)
                buffer = torch_mod.frombuffer(
                    self._slab.memoryview(), dtype=torch_mod.uint8
                )
                self.pinned = bool(self._slab.is_pinned)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "checkpoint juggle: pinned image for %s refused (%s: %s); "
                    "degrading to pageable RAM — H2D will be synchronous and "
                    "slower, counted in the switch report",
                    checkpoint_id, type(exc).__name__, exc,
                )
                buffer = None
                self._slab = None
                self._pool = None
        if buffer is None:
            buffer = torch_mod.empty(n, dtype=torch_mod.uint8)
        self.buffer = buffer
        self._build(engine)

    def _build(self, engine: Any) -> None:
        torch = self._torch
        straight: List[Tuple[str, int, int, int, int]] = []
        casts: List[Tuple[SlotSpec, SlotSource]] = []
        base_ptr = int(self.buffer.data_ptr())
        for region in self.layout.regions:
            for slot in region.slots:
                src = self.manifest[_slot_key(slot)]
                if (src.dtype_code, src.dtype_bits) == (slot.dtype_code, slot.dtype_bits):
                    straight.append(
                        (str(src.path), src.offset, src.length, base_ptr + slot.offset, 0)
                    )
                else:
                    casts.append((slot, src))
        if straight:
            if engine is not None:
                engine.submit(straight, 0).wait()
            else:
                self._pread(straight, base_ptr)
        for slot, src in casts:
            file_dtype = _torch_dtype(torch, src.dtype_code, src.dtype_bits)
            lane_dtype = _torch_dtype(torch, slot.dtype_code, slot.dtype_bits)
            raw = torch.empty(src.shape, dtype=file_dtype)
            with open(src.path, "rb") as fh:
                fh.seek(src.offset)
                view = raw.view(torch.uint8).view(-1)
                got = fh.readinto(memoryview(view.numpy()))
            if got != src.length:
                raise JuggleRefusal(
                    f"{self.checkpoint_id}: short read for {_slot_key(slot)!r} "
                    f"({got} of {src.length} bytes)"
                )
            self.slot_view(slot).copy_(raw.to(lane_dtype))
            self.casts += 1
        for region in self.layout.regions:
            self.region_digests[region.name] = self.region_digest(region)

    def _pread(
        self, requests: Sequence[Tuple[str, int, int, int, int]], base_ptr: int
    ) -> None:
        mv = memoryview(self.buffer.numpy())
        for path, offset, length, host_ptr, _dev in requests:
            start = host_ptr - base_ptr
            with open(path, "rb") as fh:
                fh.seek(offset)
                got = fh.readinto(mv[start : start + length])
            if got != length:
                raise JuggleRefusal(
                    f"{self.checkpoint_id}: short read from {path} "
                    f"({got} of {length} bytes)"
                )

    def region_bytes(self, region: RegionSpec) -> Any:
        return self.buffer[region.offset : region.offset + region.span]

    def slot_view(self, slot: SlotSpec) -> Any:
        torch = self._torch
        dtype = _torch_dtype(torch, slot.dtype_code, slot.dtype_bits)
        flat = self.buffer[slot.offset : slot.offset + slot.nbytes]
        return flat.view(dtype).view(slot.shape)

    def region_mirror(self, region: RegionSpec) -> List[Any]:
        """Per-slot views, in slot order — the facade's ``_host`` shape."""
        return [self.slot_view(slot) for slot in region.slots]

    def region_digest(self, region: RegionSpec) -> str:
        """Content digest of the region's WEIGHT bytes (padding excluded)."""
        h = hashlib.blake2b(digest_size=16)
        for slot in region.slots:
            h.update(memoryview(self.buffer.numpy())[slot.offset : slot.offset + slot.nbytes])
        return h.hexdigest()

    @property
    def nbytes(self) -> int:
        return int(self.buffer.numel())

    def close(self) -> None:
        self.buffer = None
        self._slab = None
        self._pool = None


def _torch_dtype(torch: Any, code: int, bits: int) -> Any:
    for dtype in (
        torch.float16, torch.bfloat16, torch.float32, torch.float64,
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool,
    ):
        if dlpack_dtype(torch, dtype) == (code, bits):
            return dtype
    raise TypeError(f"no torch dtype for DLPack ({code}, {bits})")


class CheckpointCatalog:
    """The lane's juggle set: admission, warm images, eviction."""

    def __init__(
        self,
        layout: ArenaLayout,
        *,
        torch_mod: Any,
        varena_mod: Any = None,
        engine_factory: Optional[Callable[[], Any]] = None,
        host_floor_bytes: int = DEFAULT_HOST_FLOOR_BYTES,
        mem_available: Callable[[], int] = _mem_available_bytes,
    ) -> None:
        self.layout = layout
        self._torch = torch_mod
        self._varena = varena_mod
        self._engine_factory = engine_factory
        self.host_floor_bytes = int(host_floor_bytes)
        self._mem_available = mem_available
        self.manifests: Dict[str, Dict[str, SlotSource]] = {}
        self.images: Dict[str, CheckpointImage] = {}
        self._lru: List[str] = []
        self.protected: Set[str] = set()
        self.pressure_epoch = 0
        self._evicted_epoch: Dict[str, int] = {}
        self.evictions = 0
        self.ingests = 0

    def admit(self, checkpoint_id: str, manifest: Dict[str, SlotSource]) -> None:
        """D1: the shape-identity proof."""
        refusal = admission_refusal(self.layout, manifest)
        if refusal is not None:
            raise JuggleRefusal(
                f"checkpoint {checkpoint_id!r} refused from the juggle set: {refusal}"
            )
        self.manifests[checkpoint_id] = manifest

    def is_admitted(self, checkpoint_id: str) -> bool:
        return checkpoint_id in self.manifests

    def warm(self, checkpoint_id: str) -> Optional[CheckpointImage]:
        image = self.images.get(checkpoint_id)
        if image is not None:
            self._touch(checkpoint_id)
        return image

    def ensure_warm(self, checkpoint_id: str) -> Optional[CheckpointImage]:
        """Build the image if RAM admits it; None means SERVE DISK-DIRECT."""
        manifest = self.manifests.get(checkpoint_id)
        if manifest is None:
            raise JuggleRefusal(
                f"checkpoint {checkpoint_id!r} was never admitted to this lane; "
                f"admit() it (with its manifest) before hinting or switching"
            )
        image = self.images.get(checkpoint_id)
        if image is not None:
            self._touch(checkpoint_id)
            return image
        if self._evicted_epoch.get(checkpoint_id) == self.pressure_epoch:
            logger.info(
                "checkpoint juggle: %s stays cold under hysteresis "
                "(pressure epoch %d); serving disk-direct",
                checkpoint_id, self.pressure_epoch,
            )
            return None
        need = int(self.layout.virtual_bytes)
        evictable = [n for n in self._lru if n not in self.protected]
        while self._mem_available() - need < self.host_floor_bytes and evictable:
            self._evict(evictable.pop(0), cause="pressure")
        if self._mem_available() - need < self.host_floor_bytes:
            self.pressure_epoch += 1
            self._evicted_epoch[checkpoint_id] = self.pressure_epoch
            logger.warning(
                "checkpoint juggle: host RAM cannot admit an image for %s "
                "(%.2f GiB needed, floor %.2f GiB); serving disk-direct "
                "(pressure epoch now %d)",
                checkpoint_id, need / (1 << 30),
                self.host_floor_bytes / (1 << 30), self.pressure_epoch,
            )
            return None
        engine = self._engine_factory() if self._engine_factory else None
        image = CheckpointImage(
            checkpoint_id, self.layout, manifest,
            torch_mod=self._torch, varena_mod=self._varena, engine=engine,
        )
        self.images[checkpoint_id] = image
        self._lru.append(checkpoint_id)
        self.ingests += 1
        logger.info(
            "checkpoint juggle: ingested %s — %.2f GiB image, pinned=%s, "
            "casts=%d, %d regions digested",
            checkpoint_id, image.nbytes / (1 << 30), image.pinned,
            image.casts, len(image.region_digests),
        )
        return image

    def _touch(self, checkpoint_id: str) -> None:
        try:
            self._lru.remove(checkpoint_id)
        except ValueError:
            pass
        self._lru.append(checkpoint_id)

    def _evict(self, checkpoint_id: str, *, cause: str) -> None:
        image = self.images.pop(checkpoint_id, None)
        if image is None:
            return
        try:
            self._lru.remove(checkpoint_id)
        except ValueError:
            pass
        if cause == "pressure":
            self.pressure_epoch += 1
            self._evicted_epoch[checkpoint_id] = self.pressure_epoch
        image.close()
        self.evictions += 1
        logger.info(
            "checkpoint juggle: evicted %s (%s); demote is free/unpin only — "
            "weights are immutable, refill comes from disk",
            checkpoint_id, cause,
        )

    def evict(self, checkpoint_id: str) -> None:
        self._evict(checkpoint_id, cause="explicit")

    def close(self) -> None:
        for checkpoint_id in list(self.images):
            self._evict(checkpoint_id, cause="teardown")


class RegionValidity(Enum):
    VALID = "valid"
    REFILLING = "refilling"
    INVALID = "invalid"


@dataclass
class _RegionState:
    validity: RegionValidity
    checkpoint_id: str


class ValidityLedger:
    """Per-region serving validity."""

    def __init__(self, layout: ArenaLayout, checkpoint_id: str) -> None:
        self._state: Dict[str, _RegionState] = {
            r.name: _RegionState(RegionValidity.VALID, checkpoint_id)
            for r in layout.regions
        }

    def begin(self, region: str, target: str) -> None:
        st = self._state[region]
        st.validity = RegionValidity.REFILLING
        st.checkpoint_id = target

    def complete(self, region: str) -> None:
        self._state[region].validity = RegionValidity.VALID

    def poison(self, region: str) -> None:
        self._state[region].validity = RegionValidity.INVALID

    def of(self, region: str) -> Tuple[RegionValidity, str]:
        st = self._state[region]
        return st.validity, st.checkpoint_id

    def assert_servable(self, expected_checkpoint: str) -> None:
        for name, st in self._state.items():
            if st.validity is not RegionValidity.VALID:
                raise RegionInvalid(
                    f"region {name!r} is {st.validity.value} "
                    f"(target {st.checkpoint_id!r}); refusing to serve — "
                    f"re-run the switch to recover (the refill is idempotent)"
                )
            if st.checkpoint_id != expected_checkpoint:
                raise RegionInvalid(
                    f"region {name!r} holds {st.checkpoint_id!r} while the lane "
                    f"claims to serve {expected_checkpoint!r}; refusing mixed "
                    f"checkpoints"
                )


@dataclass
class SwitchReport:
    """One switch, priced."""

    from_id: str
    to_id: str
    tier: str
    wall_s: float
    bytes_moved: int
    regions_refilled: int
    regions_unbacked: int
    transfer_gbps: float
    residue_bytes: int
    pinned_source: bool
    backing_verified: bool = False

    def loud(self) -> str:
        return (
            f"checkpoint switch {self.from_id!r} -> {self.to_id!r}: tier={self.tier} "
            f"wall={self.wall_s * 1000:.1f}ms bytes={self.bytes_moved} "
            f"({self.transfer_gbps:.2f} GB/s) refilled={self.regions_refilled} "
            f"free-swapped={self.regions_unbacked} residue_untouched={self.residue_bytes} "
            f"pinned_source={self.pinned_source} backing_verified={self.backing_verified}"
        )


class CheckpointJuggler:
    """The lane's switch engine over one :class:`ArenaResidency`."""

    def __init__(
        self,
        residency: ArenaResidency,
        serving_id: str,
        serving_manifest: Dict[str, SlotSource],
        *,
        catalog: Optional[CheckpointCatalog] = None,
    ) -> None:
        self.residency = residency
        self.layout = residency.layout
        self.catalog = catalog or CheckpointCatalog(
            self.layout,
            torch_mod=residency._torch,
            varena_mod=residency._varena,
            engine_factory=residency._engine_for,
        )
        if not residency.adopted:
            raise ValueError(
                "checkpoint juggle: the residency is not engaged; call "
                "engage() before constructing the juggler — the swap protocol "
                "assumes the arena already holds the serving checkpoint"
            )
        self.catalog.admit(serving_id, serving_manifest)
        self.catalog.protected.add(serving_id)
        self.serving_id = serving_id
        self.ledger = ValidityLedger(self.layout, serving_id)
        self.switches = 0
        self.rearms = 0
        self.reports: List[SwitchReport] = []
        self._residue_bytes = self._count_residue()

    def admit(self, checkpoint_id: str, manifest: Dict[str, SlotSource]) -> None:
        self.catalog.admit(checkpoint_id, manifest)

    def hint_next(self, checkpoint_id: str) -> bool:
        """Warm the image CPU-side while the current checkpoint serves."""
        return self.catalog.ensure_warm(checkpoint_id) is not None

    def assert_servable(self) -> None:
        self.ledger.assert_servable(self.serving_id)

    def switch_to(self, checkpoint_id: str) -> SwitchReport:
        """The backing swap."""
        torch = self.residency._torch
        if checkpoint_id == self.serving_id:
            report = SwitchReport(
                from_id=checkpoint_id, to_id=checkpoint_id, tier="serving",
                wall_s=0.0, bytes_moved=0, regions_refilled=0,
                regions_unbacked=0, transfer_gbps=0.0,
                residue_bytes=self._residue_bytes, pinned_source=True,
            )
            logger.info(report.loud())
            return report
        image = self.catalog.ensure_warm(checkpoint_id)
        manifest = self.catalog.manifests[checkpoint_id]
        tier = "host-warm" if image is not None else "disk-cold"
        if image is None:
            for region in self.layout.regions:
                for slot in region.slots:
                    src = manifest[_slot_key(slot)]
                    if (src.dtype_code, src.dtype_bits) != (
                        slot.dtype_code, slot.dtype_bits,
                    ):
                        raise JuggleRefusal(
                            f"disk-cold switch to {checkpoint_id!r} would need "
                            f"a cast for {_slot_key(slot)!r}; warm it first "
                            f"(casts happen at ingest, once) — still serving "
                            f"{self.serving_id!r}"
                        )

        started = time.perf_counter()
        bytes_moved = 0
        refilled = 0
        unbacked = 0
        from_id = self.serving_id

        device = self.residency.device
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.no_grad():
            self.residency.ring.drain()
            for region in self.layout.regions:
                if self.residency.is_resident(region.name):
                    self.ledger.begin(region.name, checkpoint_id)
                    try:
                        with torch.cuda.stream(stream):
                            bytes_moved += self._refill_backed(
                                region, image, manifest, stream
                            )
                    except Exception:
                        self.ledger.poison(region.name)
                        logger.error(
                            "checkpoint juggle: refill of region %r died "
                            "mid-transfer; region is INVALID and serving will "
                            "refuse until a switch completes",
                            region.name,
                        )
                        raise
                    if image is not None:
                        self.residency._host[region.name] = image.region_mirror(region)
                    self.ledger.complete(region.name)
                    refilled += 1
                else:
                    self.ledger.begin(region.name, checkpoint_id)
                    if image is not None:
                        self.residency._host[region.name] = image.region_mirror(region)
                    else:
                        self.residency._host.pop(region.name, None)
                    self.residency._rebind_off_device(region)
                    self.ledger.complete(region.name)
                    unbacked += 1
            self.residency._triples = {
                key: (src.path, src.offset, src.length)
                for key, src in manifest.items()
            }
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.cuda.synchronize(device)
            backing_verified = self._verify_backing(checkpoint_id)

        wall = time.perf_counter() - started
        self.serving_id = checkpoint_id
        self.catalog.protected = {checkpoint_id}
        if image is not None:
            self.catalog._touch(checkpoint_id)
        self.switches += 1
        report = SwitchReport(
            from_id=from_id, to_id=checkpoint_id, tier=tier, wall_s=wall,
            bytes_moved=bytes_moved, regions_refilled=refilled,
            regions_unbacked=unbacked,
            transfer_gbps=(bytes_moved / wall / 1e9) if wall > 0 else 0.0,
            residue_bytes=self._residue_bytes,
            pinned_source=bool(image.pinned) if image is not None else False,
            backing_verified=backing_verified,
        )
        self.reports.append(report)
        logger.info(report.loud())
        return report

    def _verify_backing(self, checkpoint_id: str) -> bool:
        res = self.residency.reservation
        if not hasattr(res, "page_signatures"):
            return False
        for region in self.layout.regions:
            if not self.residency.is_resident(region.name):
                continue
            sigs = res.page_signatures(region.offset, region.span, True)
            dead = sum(1 for s in sigs if int(s) == 0)
            if dead:
                self.ledger.poison(region.name)
                raise RegionInvalid(
                    f"region {region.name!r}: driver disagrees on {dead} of "
                    f"{len(sigs)} chunks after the swap to {checkpoint_id!r}; "
                    f"the region is INVALID — residency was lost outside this "
                    f"process (suspend/reset/external eviction)"
                )
        return True

    def _refill_backed(
        self,
        region: RegionSpec,
        image: Optional[CheckpointImage],
        manifest: Dict[str, SlotSource],
        stream: Any,
    ) -> int:
        if image is not None:
            dst = self._device_bytes(region)
            src = image.region_bytes(region)
            dst.copy_(src, non_blocking=image.pinned)
            return region.span
        base = int(self.residency.reservation.base_ptr)
        requests = []
        moved = 0
        for slot in region.slots:
            src = manifest[_slot_key(slot)]
            if (src.dtype_code, src.dtype_bits) != (slot.dtype_code, slot.dtype_bits):
                raise JuggleRefusal(
                    f"disk-cold switch cannot cast {_slot_key(slot)!r} in "
                    f"flight; warm this checkpoint first (the image is where "
                    f"casts happen, once)"
                )
            requests.append(
                (str(src.path), src.offset, src.length, 0, base + slot.offset)
            )
            moved += src.length
        handle = self.residency._engine_for().submit(
            requests, int(stream.cuda_stream) if stream is not None else 0
        )
        handle.wait()
        return moved

    def _device_bytes(self, region: RegionSpec) -> Any:
        torch = self.residency._torch
        return torch.from_dlpack(
            self.residency.reservation.tensor(
                region.offset, [region.span], *_DL_UINT8
            )
        )

    def _count_residue(self) -> int:
        from .stream_residency import own_tensors, tensor_bytes

        managed = {
            (slot.leaf, slot.attr)
            for region in self.layout.regions
            for slot in region.slots
        }
        total = 0
        for root_name, root in self.residency._roots:
            for name, module in root.named_modules():
                qualified = f"{root_name}.{name}" if name else root_name
                for attr, _is_param, tensor in own_tensors(module):
                    if (qualified, attr) not in managed and not tensor.is_meta:
                        total += tensor_bytes(tensor)
        return total


__all__ = [
    "LANE_VERDICT_DERIVABLE",
    "LANE_VERDICT_INCOMPATIBLE",
    "LANE_VERDICT_SATISFIES",
    "CheckpointCatalog",
    "CheckpointImage",
    "CheckpointJuggler",
    "JuggleRefusal",
    "RegionInvalid",
    "RegionValidity",
    "SlotSource",
    "SwitchReport",
    "ValidityLedger",
    "admission_refusal",
    "merge_component_manifests",
    "read_manifest",
]
