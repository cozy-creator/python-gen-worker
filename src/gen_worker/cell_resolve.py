"""pgw#1090 (DESIGN-RULINGS §4.29): ask the hub for a cell BY DERIVED KEY.

``POST /v1/worker/cells/resolve`` — the worker half of th#1750 (hub side merged
``26275ff8``). The worker derives its key from code alone (§4.27 step 1,
``boot_key``) and asks the hub, which answers as the entitlement authority with
ONE artifact or a MISS. Never a listing: pgw#904 deleted worker-side
fetch-and-filter discovery deliberately, and a resolve that returned candidates
would re-open it through a smaller aperture.

The circularity this closes
---------------------------
The hub's OTHER cell resolver only VERIFIES a cell the worker already armed
(``hubVerifiedCompileTargetForCellLaneLocked``: *"the worker's advertisement is
what got us here"*). **A cold pod arms nothing, so it advertises nothing, so the
hub names nothing, so it never adopts.** Boot-time adopt is what SEEDS the
advertisement the dispatch-time resolver requires — which is why cross-pod
adoption has been structurally impossible since 0.97.0 deleted ``aot_cells.py``.

NEVER a second admission brain
------------------------------
This module builds an :class:`aot_identity.ExpectedIdentity` from the answer and
hands the transport to ``aot_delivery.materialize_named_artifact``. That is a
RE-SITING of ``expected_from_plan``'s source — the Plan named the artifact, now
the hub's resolve answer does — and nothing else. Every gate downstream is the
one the Plan path already runs: ``receipts.refuse_untrusted_publisher``,
``aot_serve.verify_contract``, pgw#903's pre-dlopen graph-contract fence. The
answer's field set is not arbitrary; it is exactly ``ExpectedIdentity``'s five
axes plus ``materialize_named_artifact``'s two arguments.

**The receipt rides the answer deliberately, and must not be re-fetched.**
``handleWorkerCellReceipt`` scopes by ENDPOINT while resolve scopes by ORG
(§4.26), so a second fetch for the same cell could 403 what resolve just offered
— th#1680 exactly.

A MISS is one shape
-------------------
``200 {"found": false}``, byte-identical whether the key is absent, scoped out
or quarantined. That indistinguishability is the point: a distinguishable
refusal is an existence oracle across tenants. Refusals (409/400/503) are TYPED
and are NOT misses — a pod that read ``cell_resolve_incomplete`` as "no cell"
would go pay for a full cold mint believing the hub holds nothing, which is
false and expensive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

from . import aot_identity, boot_phases, cell_key
from .procsplit import broker

logger = logging.getLogger(__name__)

RESOLVE_PATH = "/v1/worker/cells/resolve"

#: Mandatory (``scripts/lint_http_timeouts.py``). Generous relative to the
#: hub's own 15 s transport resolve, because the answer is on the boot path and
#: a retry costs a whole cold mint — but bounded, because a boot that hangs here
#: is a boot that never serves.
RESOLVE_TIMEOUT_S = 30.0

#: The hub's typed refusal codes. NOT misses — see the module docstring.
REFUSAL_CODES = (
    "cell_resolve_ambiguous",
    "cell_resolve_incomplete",
    "cell_resolve_transport_unavailable",
    "cell_resolve_client_supplied_field",
)


class CellResolveRefused(RuntimeError):
    """The hub refused to answer, typed. Distinct from a MISS by construction."""

    def __init__(self, code: str, detail: str, *, status: int = 0) -> None:
        self.code = str(code)
        self.detail = str(detail)
        self.status = int(status)
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class TransportFile:
    """One presigned file of the answer's transport.

    Structurally duck-compatible with the Plan grant's ``PresignedSnapshot``
    file rows, because ``aot_delivery.materialize_named_artifact`` reads them by
    attribute — which is what lets the resolve answer feed the EXISTING delivery
    path instead of a parallel one.
    """

    path: str
    size_bytes: int
    digest: str
    url: str
    chunks: Tuple[Any, ...] = ()
    chunk_size_bytes: int = 0


@dataclass(frozen=True)
class TransportChunk:
    sha256: str
    url: str
    len: int  # noqa: A003 — the wire name; `materialize` reads `.len`


@dataclass(frozen=True)
class Transport:
    snapshot_digest: str
    files: Tuple[TransportFile, ...]


@dataclass(frozen=True)
class ResolvedCell:
    """The hub's answer for one derived key — ONE artifact, fully stated."""

    family: str
    cell_key: str
    cell_ref: str
    checkpoint_id: str
    content_digest: str
    artifact_path: str
    size_bytes: int
    publisher_org: str
    publisher_tier: str
    graph_contract: str
    toolchain_digest: str
    env_seal_digest: str
    identity_axes: Mapping[str, str]
    sm: str
    sku: str
    lane: str
    receipt: str
    transport: Transport

    def expected_identity(self) -> aot_identity.ExpectedIdentity:
        """The admission expectation this answer states.

        The SAME type ``expected_from_plan`` builds, so the artifact reaching
        ``aot_identity.verify_declared_identity`` cannot tell whether it was
        named by a Plan or pulled by key — which is the property that keeps
        this from being a second admission brain.
        """
        return aot_identity.ExpectedIdentity(
            cell_key=self.cell_key,
            toolchain_digest=self.toolchain_digest,
            env_seal_digest=self.env_seal_digest,
            graph_contract_digest=self.graph_contract,
            publisher_org=self.publisher_org,
        )


def _transport_from(body: Mapping[str, Any]) -> Transport:
    raw = body.get("transport") or {}
    files: List[TransportFile] = []
    for row in (raw.get("files") or ()):
        chunks = tuple(
            TransportChunk(
                sha256=str(c.get("sha256") or ""),
                url=str(c.get("url") or ""),
                len=int(c.get("len") or 0),
            )
            for c in (row.get("chunks") or ())
        )
        files.append(TransportFile(
            path=str(row.get("path") or ""),
            size_bytes=int(row.get("size_bytes") or 0),
            digest=str(row.get("digest") or ""),
            url=str(row.get("url") or ""),
            chunks=chunks,
            chunk_size_bytes=int(row.get("chunk_size_bytes") or 0),
        ))
    return Transport(
        snapshot_digest=str(raw.get("snapshot_digest") or ""),
        files=tuple(files),
    )


def _cell_from(body: Mapping[str, Any]) -> ResolvedCell:
    axes = body.get("identity_axes") or {}
    return ResolvedCell(
        family=str(body.get("family") or ""),
        cell_key=str(body.get("cell_key") or ""),
        cell_ref=str(body.get("cell_ref") or ""),
        checkpoint_id=str(body.get("checkpoint_id") or ""),
        content_digest=str(body.get("content_digest") or ""),
        artifact_path=str(body.get("artifact_path") or ""),
        size_bytes=int(body.get("size_bytes") or 0),
        publisher_org=str(body.get("publisher_org") or ""),
        publisher_tier=str(body.get("publisher_tier") or ""),
        graph_contract=str(body.get("graph_contract") or ""),
        toolchain_digest=str(body.get("toolchain_digest") or ""),
        env_seal_digest=str(body.get("env_seal_digest") or ""),
        identity_axes={str(k): str(v) for k, v in dict(axes).items()},
        sm=str(body.get("sm") or ""),
        sku=str(body.get("sku") or ""),
        lane=str(body.get("lane") or ""),
        receipt=str(body.get("receipt") or ""),
        transport=_transport_from(body),
    )


def resolve(
    family: str,
    derived_key: str,
    *,
    base_url: str = "",
    bearer: str = "",
) -> Optional[ResolvedCell]:
    """Ask the hub for the cell named by ``derived_key``. ``None`` is a MISS.

    Raises :class:`CellResolveRefused` on a typed refusal — which is NOT a
    miss and must not be treated as one by the caller. The body carries
    ``family`` and ``cell_key`` and NOTHING else: every entitlement input is
    resolved from the live session hub-side, and a body naming one is a named
    400 (``cell_resolve_client_supplied_field``). Sending an extra field would
    not merely be ignored — it would refuse the whole boot.
    """
    key = str(derived_key or "").strip()
    if not cell_key.is_key(key):
        raise CellResolveRefused(
            "invalid_request",
            f"{key!r} is not a cell key; the resolve route addresses cells by "
            f"ck<scheme>-<56 hex> and by nothing else")
    fam = str(family or "").strip()
    if not fam:
        raise CellResolveRefused(
            "invalid_request", "resolve requires the cell's family namespace")

    with boot_phases.span(
        boot_phases.PHASE_CELL_HUB_RTT, function="cells.resolve",
        artifact_key=key, ref=fam,
    ) if boot_phases.in_boot() else _null() as span:
        resp = broker.request(
            "POST", RESOLVE_PATH,
            base_url=base_url, bearer=bearer,
            json={"family": fam, "cell_key": key},
            timeout=RESOLVE_TIMEOUT_S,
        )
        try:
            body = resp.json()
        except ValueError:
            body = {}
        if resp.status_code != 200:
            code = str((body or {}).get("code") or (body or {}).get("error") or "")
            detail = str((body or {}).get("message") or resp.text)[:400]
            if span is not None:
                span.refused(code or f"http_{resp.status_code}", detail)
            raise CellResolveRefused(
                code or f"http_{resp.status_code}", detail,
                status=resp.status_code)
        if not bool((body or {}).get("found")):
            if span is not None:
                span.classify("miss", f"key={key}")
            logger.info("cell-resolve: MISS for %s (family=%s)", key, fam)
            return None
        cell = _cell_from(body)
        if span is not None:
            span.classify(
                "hit",
                f"key={key} tier={cell.publisher_tier} "
                f"checkpoint={cell.checkpoint_id}")
    if cell.cell_key != key:
        # The hub answered a DIFFERENT key than the one asked for. Never a
        # miss and never adoptable: the admission gate would refuse it anyway,
        # but refusing here names the seam that lied rather than the artifact
        # that arrived.
        raise CellResolveRefused(
            "cell_resolve_key_mismatch",
            f"asked for {key}, the hub answered {cell.cell_key!r}")
    logger.info(
        "cell-resolve: HIT %s -> %s (%s tier, %d bytes)",
        key, cell.cell_ref, cell.publisher_tier, cell.size_bytes)
    return cell


def materialize(
    cell: ResolvedCell, *, cache_dir: Optional[Path], what: str = "",
) -> Path:
    """Fetch the resolved cell's bytes through the EXISTING delivery path.

    Deliberately a two-line function: everything it could have added —
    digest verification, chunk assembly, the cache, the ``cell_fetch`` span —
    ``materialize_named_artifact`` already does, and a second downloader is a
    second place for "verified" to mean something slightly different.
    """
    from . import aot_delivery

    return aot_delivery.materialize_named_artifact(
        cell.cell_ref,
        cell.content_digest,
        cell.transport,
        cache_dir=cache_dir,
        what=what or f"boot adopt of {cell.cell_key}",
    )


def _null() -> Any:
    import contextlib

    return contextlib.nullcontext()


__all__ = [
    "REFUSAL_CODES",
    "RESOLVE_PATH",
    "RESOLVE_TIMEOUT_S",
    "CellResolveRefused",
    "ResolvedCell",
    "Transport",
    "TransportChunk",
    "TransportFile",
    "materialize",
    "resolve",
]
