"""pgw#1090 (DESIGN-RULINGS §4.29): ask the hub for a cell BY DERIVED KEY.

``POST /v1/worker/compiled-graphs/resolve`` — the worker half of th#1750 (hub side merged
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
RE-SITING of the expectation's source — the hub's resolve answer names the
artifact — and nothing else. Every gate downstream is the one the deleted Plan
path ran: ``receipts.refuse_untrusted_publisher``,
``aot_serve.verify_contract``, pgw#903's pre-dlopen graph-contract fence. The
answer's field set is not arbitrary; it is exactly ``ExpectedIdentity``'s five
axes plus ``materialize_named_artifact``'s two arguments.

**The receipt rides the answer deliberately, and must not be re-fetched.**
``handleWorkerCellReceipt`` scopes by ENDPOINT while resolve scopes by ORG
(§4.26), so a second fetch for the same cell could 403 what resolve just offered
— th#1680 exactly.

A MISS is one shape
-------------------
``{"status": "miss", "found": false}``, byte-identical whether the key is
absent, scoped out or quarantined. That indistinguishability is the point: a
distinguishable refusal is an existence oracle across tenants. Refusals are
TYPED and are NOT misses — a pod that read ``incomplete`` as "no cell" would go
pay for a full cold mint believing the hub holds nothing, which is false and
expensive.

pgw#1224 — THE BATCH WIRE (th#1834 Phase 2, hub half th#1842 PR #1118)
----------------------------------------------------------------------
The request is ``{family, keys[<=256]}`` and the answer is one entry per key,
**in request order, each independently signed**. The single-key
``{family, compiled_graph_key}`` shape is GONE — hard cut, no alias, no dual-accept, no
negotiation: a client that could speak both would make the hub's own hard cut
unprovable, and there is no released consumer to protect.

Why batch at all: under the per-entry atom (pgw#1176) a boot derives ONE KEY
PER GRAPH CLASS — 18-36 for sdxl, 198 for minimax-h3 — so a per-key wire makes
a boot pay that many round trips for one answer set. The entitlement inputs
(the caller's endpoint, its viewer org) are properties of the CALLER, so the
hub resolves them once per batch instead of once per key.

Four properties this client ENFORCES, because each is a way the answer could
lie and none of them is checked anywhere downstream:

1. **ARITY.** ``len(answers) == len(keys)``, always. A short answer is
   indistinguishable from a batch of misses, and a pod answers a miss by paying
   for a full cold mint — so a dropped answer is not a smaller reply, it is a
   wrong one that costs money.
2. **ORDER.** ``answers[i]`` answers ``keys[i]``. The hub echoes each key, so
   the position AND the echo are checked; either alone would admit a batch
   whose answers were transposed by a middlebox that preserved the other.
3. **PER-ANSWER SIGNATURE, never per-batch.** Every hit carries ITS OWN receipt
   — the hub's signature over THAT compiled graph, minted at publish. There is
   deliberately no signature over the batch, so a batch-level signature field
   is REFUSED rather than ignored: it would make the COLLECTION the unit of
   trust, which is the exact mistake th#1834 is undoing one layer down. Two
   answers sharing one receipt is the same fault wearing a different hat.
4. **A PER-KEY FAULT DOES NOT SINK THE BATCH.** Ambiguity, incompleteness and
   transport-unavailability were whole-request refusals; they are per-answer
   STATUSES now, carrying the SAME typed codes they had as refusals so the
   caller's vocabulary is unchanged. One duplicated row costs one graph class,
   not a boot's whole adoption. The refusals that stay whole-batch are the ones
   that are properties of the CALLER — answering those per key would report a
   hub fault as 256 misses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from torch_compiled_graphs import is_compiled_graph_key

from . import aot_identity, boot_phases
from .procsplit import broker

logger = logging.getLogger(__name__)

RESOLVE_PATH = "/v1/worker/compiled-graphs/resolve"

#: Mandatory (``scripts/lint_http_timeouts.py``). Generous relative to the
#: hub's own 15 s transport resolve, because the answer is on the boot path and
#: a retry costs a whole cold mint — but bounded, because a boot that hangs here
#: is a boot that never serves.
RESOLVE_TIMEOUT_S = 30.0

#: The hub's bound on one batch (``maxCellResolveKeys``). Mirrored rather than
#: discovered: over it the hub answers a NAMED 400 for the WHOLE batch, so a
#: caller that split late would lose every key's answer to learn the number.
MAX_RESOLVE_KEYS = 256

#: One key's outcome. ``hit`` and ``miss`` are answers; the other three are
#: HUB-SIDE faults the pod must read and must NOT self-mint over.
STATUS_HIT = "hit"
STATUS_MISS = "miss"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_INCOMPLETE = "incomplete"
STATUS_TRANSPORT_UNAVAILABLE = "transport_unavailable"

#: A per-answer status -> the typed code the caller already speaks. These three
#: were WHOLE-REQUEST refusals before the batch wire; the fault is the same
#: fact, so it keeps the same name and only its blast radius shrank. Mapping
#: rather than renaming is what keeps ``boot_adopt.ASK_REASONS`` and every
#: dashboard grouping on those codes working across the cut.
_STATUS_REFUSAL_CODE = {
    STATUS_AMBIGUOUS: "compiled_graph_resolve_ambiguous",
    STATUS_INCOMPLETE: "compiled_graph_resolve_incomplete",
    STATUS_TRANSPORT_UNAVAILABLE: "compiled_graph_resolve_transport_unavailable",
}

#: The hub's typed refusal codes. NOT misses — see the module docstring. The
#: first three are per-ANSWER now (see :data:`_STATUS_REFUSAL_CODE`); the rest
#: refuse the whole batch, because each is a property of the CALLER or of the
#: REQUEST rather than of one key.
REFUSAL_CODES = (
    "compiled_graph_resolve_ambiguous",
    "compiled_graph_resolve_incomplete",
    "compiled_graph_resolve_transport_unavailable",
    "compiled_graph_resolve_client_supplied_field",
    "compiled_graph_resolve_too_many_keys",
    "compiled_graph_resolve_duplicate_key",
    # Answered, but not to the question that was asked. Local verdicts, and
    # they are refusals rather than misses for the usual reason: a pod that
    # read a mangled batch as misses would pay for a full cold mint per key.
    "compiled_graph_resolve_short_answer",
    "compiled_graph_resolve_answer_out_of_order",
    "compiled_graph_resolve_batch_signature",
    "compiled_graph_resolve_shared_receipt",
    "compiled_graph_resolve_unknown_status",
)

#: Every field an accepted answer must NAME, and why. An answer omitting one
#: states an expectation nothing downstream can check — and the gate that
#: would catch it sits AFTER the whole cell is downloaded.
#: ``compiled_graph_resolve_incomplete`` is the hub's own code for this fact: the pod
#: reaches the same verdict from the same evidence, so it reports it under the
#: same name rather than inventing a second word for one condition.
_REQUIRED: Tuple[Tuple[str, str], ...] = (
    ("compiled_graph_key", "the admission expectation's identity axis"),
    ("toolchain_digest", "the admission expectation's toolchain axis"),
    ("env_seal_digest", "the admission expectation's environment axis"),
    ("graph_contract", "the admission expectation's graph axis"),
    ("publisher_org", "an artifact whose producer is unnamed is trusted by no "
                      "rule, and 'unknown publisher' is not a tier"),
    ("cell_ref", "the address the bytes are fetched from"),
    ("content_digest", "what the fetched bytes are verified against"),
    # pgw#1224: the per-ANSWER signature. Under the batch wire this is the only
    # thing that makes an answer stand on its own — there is no signature over
    # the batch, deliberately — so an unsigned hit is refused HERE rather than
    # discovered by `receipts.verify` after the bytes have been paid for.
    ("receipt", "the hub's signature over THIS compiled graph, which is what "
                "lets one answer of a batch be trusted, cached or replayed "
                "without trusting the collection it arrived in"),
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
    compiled_graph_key: str
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

        Built by THE map (``ExpectedIdentity.named_by``), so what reaches
        ``aot_identity.verify_declared_identity`` is the same object shape any
        naming source produces — which is the property that keeps this from
        being a second admission brain.
        """
        return aot_identity.ExpectedIdentity.named_by(self, self.graph_contract)


@dataclass(frozen=True)
class ResolveAnswer:
    """ONE key's answer out of a batch — the unit the caller acts on.

    It is a distinct type from :class:`ResolvedCell` because a batch answers
    every key, and three of the five outcomes name no artifact at all. Folding
    them into ``Optional[ResolvedCell]`` is exactly the collapse this wire
    exists to prevent: ``None`` would make "the hub holds nothing" and "the hub
    holds something it could not deliver" the same observation, and the pod
    answers the first by minting and must not answer the second that way.
    """

    compiled_graph_key: str
    status: str
    #: The artifact, on ``hit`` and only on ``hit``.
    cell: Optional[ResolvedCell] = None
    #: The typed code for a per-answer HUB FAULT — the same string the fault
    #: carried when it was a whole-request refusal. Empty on hit and on miss.
    refusal_code: str = ""
    detail: str = ""

    @property
    def hit(self) -> bool:
        return self.cell is not None

    @property
    def miss(self) -> bool:
        return self.status == STATUS_MISS


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
        ))
    return Transport(
        snapshot_digest=str(raw.get("snapshot_digest") or ""),
        files=tuple(files),
    )


def _cell_from(body: Mapping[str, Any]) -> ResolvedCell:
    axes = body.get("identity_axes") or {}
    return ResolvedCell(
        family=str(body.get("family") or ""),
        compiled_graph_key=str(body.get("compiled_graph_key") or ""),
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


def _require_batch(family: str, keys: Sequence[str]) -> Tuple[str, Tuple[str, ...]]:
    """The request the hub would accept, or a typed refusal before the wire.

    Every rule here is one the hub enforces as a NAMED 400 over the WHOLE
    batch, so failing locally costs one boot's round trip instead of every
    key's answer — and it names the caller's bug rather than reporting the
    hub's verdict on it.
    """
    fam = str(family or "").strip()
    if not fam:
        raise CellResolveRefused(
            "invalid_request", "resolve requires the compiled graph's family "
                               "namespace")
    asked: List[str] = []
    seen: Set[str] = set()
    for i, raw in enumerate(keys):
        key = str(raw or "").strip()
        if not is_compiled_graph_key(key):
            raise CellResolveRefused(
                "invalid_request",
                f"keys[{i}] is {key!r}, which is not a compiled-graph key; "
                f"the resolve route addresses compiled graphs by "
                f"<scheme>-<lowercase hex digest> and by nothing else. The "
                f"grammar rules on SHAPE, never scheme (th#1183): a newer "
                f"fleet's key is addressable here and is ruled on by its axes")
        if key in seen:
            # Answers are POSITIONAL. The hub refuses a repeat for that reason
            # and so does this: collapsing it would shift every later answer
            # against its request, and answering it twice is work nobody asked
            # for.
            raise CellResolveRefused(
                "compiled_graph_resolve_duplicate_key",
                f"keys[{i}] repeats {key}; answers come back in REQUEST ORDER "
                f"and a collapsed duplicate would shift every later answer "
                f"against its request")
        seen.add(key)
        asked.append(key)
    if not asked:
        raise CellResolveRefused(
            "invalid_request", "resolve requires a non-empty key set")
    if len(asked) > MAX_RESOLVE_KEYS:
        raise CellResolveRefused(
            "compiled_graph_resolve_too_many_keys",
            f"this batch names {len(asked)} keys and the bound is "
            f"{MAX_RESOLVE_KEYS}; split it rather than receiving a short "
            f"answer that reads as misses")
    return fam, tuple(asked)


def _answer_from(
    raw: Mapping[str, Any], key: str, family: str,
) -> ResolveAnswer:
    """One wire answer -> one :class:`ResolveAnswer`. The key is the ASKED one.

    ``status`` is read strictly: an unrecognised status is a refusal, never a
    miss. A client that defaulted unknown to miss would silently self-mint over
    every future hub-side fault the day the hub learns to name one.
    """
    status = str(raw.get("status") or "").strip()
    if status == STATUS_MISS:
        return ResolveAnswer(compiled_graph_key=key, status=STATUS_MISS)
    if status in _STATUS_REFUSAL_CODE:
        return ResolveAnswer(
            compiled_graph_key=key, status=status,
            refusal_code=_STATUS_REFUSAL_CODE[status],
            detail=str(raw.get("detail") or ""))
    if status != STATUS_HIT or not bool(raw.get("found")):
        raise CellResolveRefused(
            "compiled_graph_resolve_unknown_status",
            f"the answer for {key} carries status {status!r} "
            f"(found={raw.get('found')!r}), which this worker cannot act on; "
            f"reading it as a miss would send this pod to a full cold mint "
            f"over a fact it did not understand")
    cell = _cell_from(raw)
    missing = [(f, why) for f, why in _REQUIRED if not getattr(cell, f)]
    if missing:
        # PER ANSWER, not per batch. Refused HERE rather than at the identity
        # gate, which runs after `materialize` has already paid for the whole
        # artifact — and refused for THIS key only, so its 35 siblings still
        # arm. The hub reaches the same verdict from the same evidence and
        # calls it the same thing.
        return ResolveAnswer(
            compiled_graph_key=key, status=STATUS_INCOMPLETE,
            refusal_code="compiled_graph_resolve_incomplete",
            detail="the answer names no " + "; no ".join(
                f"{f} ({why})" for f, why in missing))
    logger.info(
        "cell-resolve: HIT %s -> %s (%s tier, %d bytes)",
        key, cell.cell_ref, cell.publisher_tier, cell.size_bytes)
    return ResolveAnswer(
        compiled_graph_key=key, status=STATUS_HIT, cell=cell,
        detail=f"family={family} tier={cell.publisher_tier} "
               f"checkpoint={cell.checkpoint_id}")


#: Top-level names that would make the COLLECTION the unit of trust. The hub
#: sends none of them, deliberately (#1118: "there is no signature over the
#: batch"), so seeing one means either a hub that grew a batch signature
#: without this client agreeing to it, or something in between that minted one.
#: Both are refused rather than ignored — an ignored signature field is one a
#: later reader assumes was checked.
_BATCH_SIGNATURE_FIELDS = ("signature", "receipt", "jws", "answers_signature")


def resolve_batch(
    family: str,
    keys: Sequence[str],
    *,
    base_url: str = "",
    bearer: str = "",
) -> Tuple[ResolveAnswer, ...]:
    """Ask the hub for a whole KEY SET at once. One answer per key, in order.

    ``POST {family, keys[<=256]}`` -> one answer per requested key. The
    returned tuple is positionally aligned with ``keys`` and is always the same
    length; a caller may zip them without checking, because this function has
    already refused every batch where that would not hold.

    Raises :class:`CellResolveRefused` for a WHOLE-BATCH fault — a malformed
    request, a caller-scoped hub failure, or an answer set that does not
    answer the question asked. A PER-KEY fault is not a raise: it is an answer
    with a ``refusal_code``, so one bad row costs one graph class.
    """
    fam, asked = _require_batch(family, keys)

    with boot_phases.span(
        boot_phases.PHASE_CELL_HUB_RTT, function="compiled_graphs.resolve",
        artifact_key=asked[0], ref=fam,
    ) if boot_phases.in_boot() else _null() as span:
        resp = broker.request(
            "POST", RESOLVE_PATH,
            base_url=base_url, bearer=bearer,
            json={"family": fam, "keys": list(asked)},
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
        answers = _answers_of(body, asked)
        hits = sum(1 for a in answers if a.hit)
        if span is not None:
            span.classify(
                "hit" if hits else "miss",
                f"keys={len(asked)} hits={hits} family={fam}")
    return answers


def _answers_of(
    body: Any, asked: Tuple[str, ...],
) -> Tuple[ResolveAnswer, ...]:
    """Decode the answer list, enforcing arity, order and per-answer signing."""
    raw = (body or {}) if isinstance(body, Mapping) else {}
    for field in _BATCH_SIGNATURE_FIELDS:
        if raw.get(field):
            raise CellResolveRefused(
                "compiled_graph_resolve_batch_signature",
                f"the batch carries a top-level {field!r}; every answer is "
                f"signed on its own and there is deliberately no signature "
                f"over the collection, so a batch-level one would make the "
                f"COLLECTION the unit of trust")
    rows = raw.get("answers")
    if not isinstance(rows, list):
        raise CellResolveRefused(
            "compiled_graph_resolve_short_answer",
            f"asked {len(asked)} keys and the reply names no answers[] at all")
    if len(rows) != len(asked):
        # An omission is indistinguishable from a truncation, and the pod
        # answers a missing answer by paying for a full cold mint. So a short
        # batch is not a smaller reply — it is a wrong one.
        raise CellResolveRefused(
            "compiled_graph_resolve_short_answer",
            f"asked {len(asked)} keys, got {len(rows)} answers; answers are "
            f"positional and a batch that does not answer every key is "
            f"indistinguishable from one truncated in transit")
    out: List[ResolveAnswer] = []
    receipts: Dict[str, str] = {}
    for i, key in enumerate(asked):
        row = rows[i]
        if not isinstance(row, Mapping):
            raise CellResolveRefused(
                "compiled_graph_resolve_short_answer",
                f"answers[{i}] is {type(row).__name__}, not an answer")
        echoed = str(row.get("compiled_graph_key") or "").strip()
        if echoed != key:
            # POSITION and ECHO both, because either alone admits a batch
            # transposed by something that preserved the other.
            raise CellResolveRefused(
                "compiled_graph_resolve_answer_out_of_order",
                f"answers[{i}] answers {echoed!r} and keys[{i}] asked {key}; "
                f"answers are consumed positionally, so a transposed batch "
                f"would arm every class with a sibling's kernels")
        answer = _answer_from(row, key, str(raw.get("family") or ""))
        if answer.cell is not None:
            seen_for = receipts.get(answer.cell.receipt)
            if seen_for is not None:
                # One receipt covering two answers is a batch-level signature
                # wearing a per-answer field name: the receipt is the hub's
                # signature over ONE compiled graph, so two keys sharing one
                # means at least one answer is being vouched for by the other's
                # proof.
                raise CellResolveRefused(
                    "compiled_graph_resolve_shared_receipt",
                    f"{key} and {seen_for} were answered with the SAME "
                    f"receipt; a receipt signs one compiled graph, and reusing "
                    f"it makes the collection the unit of trust")
            receipts[answer.cell.receipt] = key
        out.append(answer)
    return tuple(out)


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
        what=what or f"boot adopt of {cell.compiled_graph_key}",
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
    "MAX_RESOLVE_KEYS",
    "ResolveAnswer",
    "STATUS_AMBIGUOUS",
    "STATUS_HIT",
    "STATUS_INCOMPLETE",
    "STATUS_MISS",
    "STATUS_TRANSPORT_UNAVAILABLE",
    "materialize",
    "resolve_batch",
]
