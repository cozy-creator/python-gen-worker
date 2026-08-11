"""pgw#1117 / th#1777: the pre-stage envelope precondition.

THE INCIDENT (ie#642, 2026-08-11). A release declaring ``vram_gb = 22`` with
``strict_vram`` was bound to a repo's bare ``prod`` head that the th#1754 wipe
had repointed at the fp32 archive clone. The worker printed

    load_phase :: pipeline:tensorhub/hidream-o1-image: load; staged 0.67 GiB of 32.81 GiB

— it KNEW the artifact weighed 32.81 GiB and it KNEW the declaration said 22 —
staged anyway, OOMed 192 MiB short of a 23.52 GiB card inside ``setup()``,
disabled every function in its lane, tripped the hub's
``model_load_failure_streak`` and darkened a live endpoint. Cost: a billed
4090 per attempt for a fact that was on disk before the first byte moved.

THE PRECONDITION. Before staging, weigh the artifact AS IT WILL LOAD and
compare it against the envelope the release declares. A clear breach is a
typed refusal naming both numbers and the artifact digest — not an OOM, not a
retry, and not evidence for the hub's breaker.

WHAT THE ESTIMATE IS, AND WHAT IT IS NOT. It is the header-declared tensor
bytes of the snapshot's safetensors files, re-scaled by the dtype the loader
will actually ask for (:func:`estimate_resident_bytes`). That is a LOWER BOUND
on resident bytes and deliberately nothing more:

* it counts WEIGHTS only — activations, attention workspace, allocator
  fragmentation and CUDA context all sit on top, unmeasured here;
* it counts the files the loader will open, after variant-twin de-duplication,
  but it cannot know which components a pipeline leaves on the host;
* it applies no upward "overhead factor". There is no measured one, and
  inventing a multiplier would convert an honest lower bound into a guess.

Because it is a lower bound, the comparison is sound in exactly one direction:
weights alone exceeding the declared envelope means the load cannot fit,
whatever the activations do. The margin below is therefore the ESTIMATOR's own
error budget (which twin the loader picks, buffers that never reach the card,
a component that stays on the host), not a headroom allowance for the model.

WHEN IT DOES NOT SPEAK. The estimator refuses to answer — and the load
proceeds untouched — for any snapshot whose resident size is not plainly
computable from headers: a runtime storage cast, an on-disk quantization
config, the svdq / w8a8 / w4a4 / gguf lanes, an unknown dtype token, or no
safetensors at all. A marginal artifact still tries. Only a clear breach is
refused; this is a precondition against a wrong BINDING, not a second
strict_vram.

RELATION TO ``strict_vram``. Complementary, not overlapping. ``strict_vram``
is the AUTHOR's declaration that CPU-resident weights are unacceptable, which
is what removes the offload rung that would otherwise carry an oversized
artifact slowly instead of fatally. This check catches the other half — the
right declaration bound to the WRONG ARTIFACT — and it only fires where the
offload escape is already closed, because without ``strict_vram`` an artifact
that does not fit degrades rather than dies ("degrade, don't OOM").
"""

from __future__ import annotations

import json
import logging
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .safetensors_header import header_len_ok

logger = logging.getLogger(__name__)

_GIB = float(1 << 30)

#: The estimator's own error budget, as a fraction of the declared envelope.
#: NOT headroom for the model: the estimate is a weights-only LOWER bound, so
#: any true excess is already a breach. This covers what the estimator can get
#: wrong — which variant twin the loader opens, non-parameter buffers, a
#: component that never reaches the card. 10% of a 22 GB envelope is 2.2 GB,
#: which is far wider than any of those and still leaves the ie#642 artifact
#: (32.81 vs 22) refused by a factor of 1.36.
ENVELOPE_ERROR_BAND = 0.10

#: Bytes per element for every safetensors dtype token we can weigh. A token
#: outside this table makes the estimate UNMEASURABLE rather than approximate.
_ELEMENT_BYTES: Dict[str, int] = {
    "BOOL": 1, "U8": 1, "I8": 1,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}

#: The float dtypes a ``torch_dtype`` cast actually re-widths. Integer and
#: boolean tensors keep their storage whatever the compute dtype is.
_CASTABLE = frozenset({"F8_E4M3", "F8_E5M2", "F8_E8M0", "F16", "BF16", "F32", "F64"})

#: Binding dtype string -> bytes per element. Mirrors ``loading._DTYPE_MAP``;
#: an unrecognised string makes the estimate unmeasurable (the loader would
#: raise on it anyway).
_CAST_ELEMENT_BYTES: Dict[str, int] = {
    "fp16": 2, "float16": 2,
    "bf16": 2, "bfloat16": 2,
    "fp32": 4, "float32": 4,
}

#: Variant infixes diffusers uses to publish twins of the same tensor set.
#: They appear both as a suffix (``diffusion_pytorch_model.fp16.safetensors``)
#: and INSIDE a shard name (``diffusion_pytorch_model.fp16-00001-of-00002``),
#: so the match is delimiter-anchored rather than a plain endswith — a missed
#: shard twin would double-count a set and inflate the estimate, which is the
#: one direction a refusal-grade number must never err in.
_VARIANT_RE = re.compile(r"\.(fp16|bf16|fp8|fp32)(?=$|[.\-_])")


class ArtifactEnvelopeExceeded(RuntimeError):
    """The bound artifact cannot fit the envelope the release declares.

    Terminal and deterministic: every pod of this release reaches the same
    verdict from the same bytes, so it is never retried and never re-probed.
    Blame is RELEASE/BINDING-scoped — the declaration and the artifact
    disagree — never the payload (no request participated) and never the host
    (no card was consulted). Deliberately NOT a
    :class:`~gen_worker.capability.HardwareUnmetError`: that family means
    "this MACHINE cannot satisfy a correct declaration" and feeds the hub's
    hardware-fact and buy-floor paths, which would send an operator shopping
    for a bigger card to serve an archive clone.
    """

    #: FnUnavailable reason vocabulary token (the closed set in
    #: worker_scheduler.proto).
    reason = "artifact_envelope_exceeded"

    def __init__(
        self,
        message: str,
        *,
        estimated_bytes: int = 0,
        declared_vram_gb: float = 0.0,
        artifact_digest: str = "",
        ref: str = "",
        on_disk_dtype: str = "",
        load_dtype: str = "",
    ) -> None:
        super().__init__(message)
        self.estimated_bytes = int(estimated_bytes)
        self.declared_vram_gb = float(declared_vram_gb)
        self.artifact_digest = artifact_digest
        self.ref = ref
        self.on_disk_dtype = on_disk_dtype
        self.load_dtype = load_dtype

    def axes(self) -> Dict[str, str]:
        """Axis map for ``WorkerFunctionUnavailableSignal``. Both numbers are
        axes, not prose: the hub's operator view reads them as columns."""
        out = {
            "estimated_vram_gb": f"{self.estimated_bytes / _GIB:.2f}",
            "declared_vram_gb": f"{self.declared_vram_gb:.2f}",
        }
        if self.artifact_digest:
            out["artifact_digest"] = self.artifact_digest
        if self.ref:
            out["ref"] = self.ref
        if self.on_disk_dtype:
            out["on_disk_dtype"] = self.on_disk_dtype
        if self.load_dtype:
            out["load_dtype"] = self.load_dtype
        return out


@dataclass(frozen=True)
class EnvelopeEstimate:
    """What the artifact weighs as it will load, plus the honesty about it."""

    #: Header-declared tensor bytes of the files the loader will open.
    disk_bytes: int
    #: The same tensors re-scaled by the dtype the loader will ask for. A
    #: LOWER bound on resident bytes: weights only.
    resident_bytes: int
    #: Majority on-disk dtype ("bf16"/"fp16"/"fp32"/"fp8"/"").
    on_disk_dtype: str
    #: The dtype the load will present ("" = as stored, no cast directive).
    load_dtype: str
    #: False when the resident size is not plainly computable from headers.
    #: An unmeasurable estimate never refuses anything.
    measurable: bool
    #: Why it is unmeasurable, or how it was computed. Carried into logs so a
    #: silent non-answer is still an answer someone can read.
    method: str

    @property
    def resident_gb(self) -> float:
        return self.resident_bytes / _GIB


def _tensor_bytes_by_dtype(path: Path) -> Optional[Dict[str, int]]:
    """Header-declared bytes per dtype token in one safetensors file, or None
    when the header cannot be read (a truncated or malformed file is not a
    zero — it is an unmeasurable one)."""
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return None
            (n,) = struct.unpack("<Q", raw)
            if not header_len_ok(n):
                return None
            header = json.loads(f.read(n))
    except (OSError, ValueError):
        return None
    out: Dict[str, int] = {}
    for key, value in header.items():
        if key == "__metadata__" or not isinstance(value, dict):
            continue
        offsets = value.get("data_offsets")
        dtype = value.get("dtype")
        if not offsets or dtype is None:
            continue
        try:
            start, end = int(offsets[0]), int(offsets[1])
        except (TypeError, ValueError, IndexError):
            return None
        out[str(dtype)] = out.get(str(dtype), 0) + max(0, end - start)
    return out


def _loaded_files(root: Path, variant: str) -> List[Path]:
    """The safetensors files the loader will actually open, with variant twins
    collapsed to one. Two twins of the same tensor set on disk are one set in
    memory; counting both is the one error the estimate must not make."""
    groups: Dict[Tuple[str, str], Dict[str, Path]] = {}
    for p in sorted(root.rglob("*.safetensors")):
        stem = p.name[: -len(".safetensors")]
        match = _VARIANT_RE.search(stem.lower())
        var = match.group(1) if match else ""
        if match:
            stem = stem[: match.start()] + stem[match.end():]
        groups.setdefault((str(p.parent), stem), {})[var] = p
    out: List[Path] = []
    for by_variant in groups.values():
        if len(by_variant) == 1:
            out.append(next(iter(by_variant.values())))
            continue
        if variant and variant in by_variant:
            out.append(by_variant[variant])
        elif "" in by_variant:
            out.append(by_variant[""])
        else:
            # Twins only, none of them the loader's declared variant: take the
            # SMALLEST, so an ambiguity can never inflate the estimate.
            out.append(min(by_variant.values(), key=lambda q: q.stat().st_size))
    return sorted(out)


def estimate_resident_bytes(
    trees: Sequence[Path | str],
    *,
    cast_dtype: str = "",
    variant: str = "",
) -> EnvelopeEstimate:
    """Weigh the snapshot AS IT WILL LOAD, from headers alone. No tensor is
    read, no torch is imported, nothing is downloaded.

    ``cast_dtype`` is the binding's ``dtype`` directive. Empty means the loader
    passes no ``torch_dtype`` for an fp32/unknown tree and honors the sniffed
    precision for a bf16/fp16 one — either way the tensors land at their stored
    width, which is exactly what happened in ie#642: an fp32 tree loaded fp32.
    """
    files: List[Path] = []
    for tree in trees:
        root = Path(tree)
        if root.is_file():
            if root.suffix == ".safetensors":
                files.append(root)
            continue
        try:
            files.extend(_loaded_files(root, variant))
        except OSError:
            return EnvelopeEstimate(
                0, 0, "", "", False, "snapshot tree is unreadable")

    if not files:
        return EnvelopeEstimate(
            0, 0, "", "", False, "no safetensors weights in the snapshot")

    by_dtype: Dict[str, int] = {}
    for p in files:
        per_file = _tensor_bytes_by_dtype(p)
        if per_file is None:
            return EnvelopeEstimate(
                0, 0, "", "", False,
                f"unreadable safetensors header in {p.name}")
        for token, nbytes in per_file.items():
            by_dtype[token] = by_dtype.get(token, 0) + nbytes

    unknown = sorted(t for t in by_dtype if t not in _ELEMENT_BYTES)
    if unknown:
        return EnvelopeEstimate(
            0, 0, "", "", False,
            f"unweighable safetensors dtype(s) {unknown}")

    disk_bytes = sum(by_dtype.values())
    # Majority by BYTES, not by tensor count (which is what
    # `loading.detect_on_disk_dtype` reports): this label describes what the
    # artifact WEIGHS, and a thousand tiny fp32 norm tensors do not make a
    # bf16 transformer an fp32 tree.
    top = max(by_dtype, key=lambda k: by_dtype[k]) if by_dtype else ""
    on_disk = {"BF16": "bf16", "F16": "fp16", "F32": "fp32",
               "F8_E4M3": "fp8"}.get(top, top.lower())

    cast = (cast_dtype or "").strip().lower()
    if not cast:
        return EnvelopeEstimate(
            disk_bytes, disk_bytes, on_disk, "", True,
            f"{len(files)} safetensors file(s), header-declared tensor bytes, "
            f"loaded as stored ({on_disk or 'mixed'}) — no cast directive")

    target = _CAST_ELEMENT_BYTES.get(cast)
    if target is None:
        return EnvelopeEstimate(
            disk_bytes, disk_bytes, on_disk, "", False,
            f"unrecognised binding dtype {cast_dtype!r}")

    resident = 0
    for token, nbytes in by_dtype.items():
        elem = _ELEMENT_BYTES[token]
        if token in _CASTABLE:
            resident += (nbytes // elem) * target
        else:
            resident += nbytes
    return EnvelopeEstimate(
        disk_bytes, resident, on_disk, cast, True,
        f"{len(files)} safetensors file(s), header-declared tensor bytes "
        f"re-scaled from {on_disk or 'mixed'} to the binding's {cast}")


def envelope_refusal(
    trees: Sequence[Path | str],
    *,
    declared_vram_gb: float,
    strict_vram: bool,
    cast_dtype: str = "",
    storage_dtype: str = "",
    variant: str = "",
    specialized_layout: str = "",
    slot: str = "",
    ref: str = "",
    artifact_digest: str = "",
) -> Optional[ArtifactEnvelopeExceeded]:
    """The precondition itself: the refusal to raise before staging, or None.

    None is the answer for everything the estimator cannot settle CLEANLY — no
    declaration, no ``strict_vram`` (the offload rung still carries it), a
    runtime storage cast, a specialized weight layout, an unmeasurable tree,
    or a merely marginal excess. A load that is not refused here is a load
    that proceeds exactly as it did before this check existed.
    """
    if declared_vram_gb <= 0:
        return None
    if not strict_vram:
        # Without strict_vram the fit ladder may legitimately place an
        # oversized artifact on the offload rung. Refusing would break the
        # "degrade, don't OOM" ruling; this precondition exists for the case
        # where that escape is already closed by the author.
        return None
    if storage_dtype:
        # A runtime fp8 storage cast restructures a subset of the modules
        # AFTER the load, and which subset is a property of the pipeline
        # class, not of the tree. Not plainly computable -> no verdict.
        return None
    if specialized_layout:
        return None

    est = estimate_resident_bytes(trees, cast_dtype=cast_dtype, variant=variant)
    if not est.measurable:
        logger.debug("envelope precondition abstained for %s: %s", ref or slot,
                     est.method)
        return None

    declared_bytes = declared_vram_gb * _GIB
    ceiling = declared_bytes * (1.0 + ENVELOPE_ERROR_BAND)
    if est.resident_bytes <= ceiling:
        return None

    digest = artifact_digest or "(digest unknown)"
    over = est.resident_bytes / declared_bytes
    message = (
        f"artifact does not fit the declared envelope: slot {slot or '?'!r} "
        f"ref {ref or '?'} artifact {digest} weighs "
        f"{est.resident_gb:.2f} GiB as it will load, and the release declares "
        f"vram_gb={declared_vram_gb:.2f} with strict_vram "
        f"({over:.2f}x the envelope). Estimate: {est.method}; WEIGHTS ONLY, so "
        f"it is a lower bound — activations and workspace are on top of it. "
        f"Refused BEFORE staging: no bytes were staged onto the card and no "
        f"GPU time was spent. The declaration and the bound artifact "
        f"disagree — rebind the slot to the serving flavor, or raise the "
        f"declaration to what this artifact actually needs."
    )
    return ArtifactEnvelopeExceeded(
        message,
        estimated_bytes=est.resident_bytes,
        declared_vram_gb=declared_vram_gb,
        artifact_digest=artifact_digest,
        ref=ref,
        on_disk_dtype=est.on_disk_dtype,
        load_dtype=est.load_dtype or est.on_disk_dtype,
    )


__all__ = [
    "ENVELOPE_ERROR_BAND",
    "ArtifactEnvelopeExceeded",
    "EnvelopeEstimate",
    "envelope_refusal",
    "estimate_resident_bytes",
]
