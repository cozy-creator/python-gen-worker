"""pgw#1089 x pgw#1090 (DESIGN-RULINGS §4.27 steps 1-3): the boot-time adopt.

ONE function, and it is the whole of §4.27's first three steps for a serving
pod:

1. **Derive** the ``ck1`` key from code alone — ``boot_key.derive``, process-
   parallel structure-only traces, no weight resident anywhere.
2. **Ask** the hub by that key — ``cell_resolve.resolve``, ONE artifact or a
   MISS, never a listing.
3. **Materialize** a hit through the EXISTING delivery path and hand the
   caller an admission expectation, so the arm runs the gates the Plan path
   runs and not one gate fewer.

A MISS returns ``None`` and that is a complete answer: the pod serves eager,
mints in the background (§4.28: trusted hardware publishes, untrusted keeps it
local) and the request path waits for none of it (pgw#1091).

Why this is a module and not three lines at the call site
---------------------------------------------------------
Because a refusal at any of the three steps must **degrade to the pre-existing
behaviour and say why**, and there are eight distinct ways to fail:
the family has no structure-only build, a trace child died, the hub refused
typed, the hub answered a different key, the transport was unusable, the bytes
failed their digest, the delivered cell would not state its metadata, and
(pgw#1031) the cell's recorded graph witness is not the graph this pod traced.
Every one of them means "boot as we booted yesterday", and every one of them is
a different sentence someone will need. Spreading that across an executor
branch is how the reason ends up being `logger.debug`.

**Never fatal.** A cold boot that cannot adopt is the boot every pod did before
this existed. The one thing this must never do is prevent a pod from serving.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from . import aot_identity, artifact_meta, boot_key, cell_resolve
from .mint_process import CompileCellSpec, MintSlot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BootAdoption:
    """A cell this boot resolved by its OWN derived key, ready to arm."""

    derived: boot_key.DerivedKey
    cell: cell_resolve.ResolvedCell
    artifact: Path

    @property
    def expected(self) -> aot_identity.ExpectedIdentity:
        return self.cell.expected_identity()

    @property
    def ref(self) -> str:
        return self.cell.cell_ref

    @property
    def snapshot_digest(self) -> str:
        return self.cell.transport.snapshot_digest


@dataclass(frozen=True)
class BootAdoptOutcome:
    """What the boot-adopt attempt did — always a complete answer.

    ``adoption`` set means arm it. ``adoption`` None with a ``reason`` means
    boot exactly as this pod booted before the seam existed, and the reason is
    the countable token for why.
    """

    adoption: Optional[BootAdoption] = None
    reason: str = ""
    detail: str = ""
    derived_key: str = ""
    derive_ms: int = 0

    @property
    def adopted(self) -> bool:
        return self.adoption is not None


def attempt(
    *,
    function: str,
    modules: Sequence[str],
    cfg: Any,
    slots: Mapping[str, MintSlot],
    declared_hint: int,
    envelope: Mapping[str, Any],
    work_root: Path,
    memo_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    base_url: str = "",
    bearer: str = "",
    device: int = -1,
) -> BootAdoptOutcome:
    """Derive, ask, materialize. Never raises — every failure is an outcome.

    ``declared_hint`` is ``len(aot_declaration.cell_plans(decl))`` — it sizes
    the trace pool and nothing else. ``envelope`` is
    ``fleet_cells.declared_envelope_block(cfg)``, i.e. the same extraction the
    publish path recomputes the envelope axis from.
    """
    family = str(getattr(cfg, "family", "") or "")
    spec = CompileCellSpec(
        shapes=tuple(
            tuple(int(v) for v in row)
            for row in (getattr(cfg, "shapes", ()) or ())),
        targets=tuple(str(t) for t in (getattr(cfg, "targets", ()) or ())),
        family=family,
        lora_bucket=int(getattr(cfg, "lora_bucket", 0) or 0),
        guidance_scales=tuple(
            float(v) for v in (getattr(cfg, "guidance_scales", ()) or ())),
        text_lens=tuple(int(v) for v in (getattr(cfg, "text_lens", ()) or ())),
    )

    try:
        derived = boot_key.derive(
            function=str(function),
            modules=tuple(str(m) for m in modules),
            family=family,
            cfg=spec,
            slots=dict(slots),
            declared_hint=int(declared_hint),
            envelope=dict(envelope),
            work_root=Path(work_root),
            memo_dir=Path(memo_dir) if memo_dir else None,
            device=int(device),
        )
    except boot_key.BootKeyUnavailable as exc:
        logger.info(
            "boot-adopt: no derived key for %s (%s) — this pod adopts nothing "
            "and mints the ordinary way: %s", family, exc.reason, exc.detail)
        return BootAdoptOutcome(reason=exc.reason, detail=exc.detail)
    except Exception as exc:  # noqa: BLE001 — never fatal, see the docstring
        logger.warning("boot-adopt: key derivation failed", exc_info=True)
        return BootAdoptOutcome(
            reason="derive_failed", detail=f"{type(exc).__name__}: {exc}")

    key = derived.digest
    logger.info(
        "boot-adopt: %s derived %s in %d ms (%d class(es), K=%d, memo=%s)",
        family, key, derived.wall_ms, derived.traced, derived.workers,
        derived.memo)

    try:
        cell = cell_resolve.resolve(
            family, key, base_url=base_url, bearer=bearer)
    except cell_resolve.CellResolveRefused as exc:
        # A typed refusal is NOT a miss — treating it as one sends the pod to
        # a full cold mint believing the hub holds nothing, which is false and
        # expensive. It still degrades to the old boot; it is the SENTENCE
        # that differs, and that sentence is what gets the hub fixed.
        logger.warning(
            "boot-adopt: the hub REFUSED to resolve %s: %s", key, exc)
        return BootAdoptOutcome(
            reason=exc.code, detail=exc.detail,
            derived_key=key, derive_ms=derived.wall_ms)
    except Exception as exc:  # noqa: BLE001
        logger.warning("boot-adopt: resolve call failed", exc_info=True)
        return BootAdoptOutcome(
            reason="resolve_unreachable", detail=f"{type(exc).__name__}: {exc}",
            derived_key=key, derive_ms=derived.wall_ms)

    if cell is None:
        return BootAdoptOutcome(
            reason="miss",
            detail=f"the hub holds no entitled cell for {key}",
            derived_key=key, derive_ms=derived.wall_ms)

    try:
        artifact = cell_resolve.materialize(
            cell, cache_dir=Path(cache_dir) if cache_dir else None,
            what=f"boot adopt of {key} (family {family})")
    except Exception as exc:  # noqa: BLE001
        logger.warning("boot-adopt: materialize failed", exc_info=True)
        return BootAdoptOutcome(
            reason="materialize_failed", detail=f"{type(exc).__name__}: {exc}",
            derived_key=key, derive_ms=derived.wall_ms)

    # pgw#1031, THE GRAPH-WITNESS FLOOR. Everything above this line agreed on
    # the KEY, and the key is an ingress identity: two endpoints whose declared
    # contracts match while their bodies differ mint under one `ck1`
    # (`micro-pad32` vs `micro-pad32-branchy`, measured 2026-08-10 — identical
    # keying block from 112- and 102-node graphs). Pull-by-key would hand this
    # pod the other endpoint's kernels, and only the arm-time numerics
    # tolerance would stand in the way. So the last thing checked before an
    # adoption is returned is that the cell's compiled graphs ARE the graphs
    # this pod traced.
    try:
        # THE one bounded reader (pgw#1098): a second scan of the same
        # member is a divergence waiting for the first caller who bounds
        # only one of them.
        meta = artifact_meta.read_metadata(Path(artifact))
    except Exception as exc:  # noqa: BLE001 — unreadable IS unproven
        logger.warning("boot-adopt: metadata unreadable", exc_info=True)
        return BootAdoptOutcome(
            reason="witness_unreadable",
            detail=(
                f"the materialized cell for {key} would not state its "
                f"metadata ({type(exc).__name__}: {exc}), so its graphs cannot "
                f"be shown to be this pod's graphs"),
            derived_key=key, derive_ms=derived.wall_ms)
    mismatch = aot_identity.verify_graph_witness(
        meta, derived.graph_witnesses)
    if mismatch:
        logger.error(
            "boot-adopt: REFUSING %s (%s) — the key matched and the graph did "
            "not: %s. This pod serves eager and mints its own.",
            key, cell.cell_ref, mismatch)
        return BootAdoptOutcome(
            reason="graph_witness_mismatch", detail=mismatch,
            derived_key=key, derive_ms=derived.wall_ms)

    logger.info(
        "boot-adopt: %s resolved to %s (%s tier) and materialized at %s — "
        "arming with ZERO dispatches having occurred",
        key, cell.cell_ref, cell.publisher_tier, artifact)
    return BootAdoptOutcome(
        adoption=BootAdoption(derived=derived, cell=cell, artifact=artifact),
        reason="hit", derived_key=key, derive_ms=derived.wall_ms)


__all__ = ["BootAdoptOutcome", "BootAdoption", "attempt"]
