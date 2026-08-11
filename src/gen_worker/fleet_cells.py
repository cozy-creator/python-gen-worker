"""Fleet self-mint compile cells (gw#587).

The serving worker's boot warmup IS a perfect mint by construction: right
SKU (its own card), right image (its own digest), right weight lane (its
own loader decision), right pipeline class and call path (its own endpoint
code), right shapes (its own declaration). gw#586 proved the replicated-
producer alternative is a parity treadmill — every axis the producer must
replicate was discovered as a live production failure, one at a time.

Under self-mint the arming policy for a compile-declared function becomes:

  1. HIT (a published ``aot-inductor`` cell this runtime's axes select, or
     the hub-attached artifact): arm through the delivered-cell path.
  2. MISS on a family that DECLARES an export: delegate an AOT mint to a
     child process (pgw#784/#805) while this pipeline serves eager, then
     adopt the child's cell through the same gates a hub-delivered ``.pt2``
     passes. Publish only what adopted — the hub's attested gate (th#910)
     decides accept/refuse, and publish failure never affects serving.
  3. MISS on a family with NO export declaration: **JIT intake** — arm the
     declared targets cold-allowed and guarded, let this pod's own warmup
     compile them, and produce NOTHING. This is pgw#1010's cut, and it is
     the ratified reuse ruling made structural: reuse is AOT-only, JIT is
     intake with honest cold boots. A JIT cell had no possible consumer
     (only ``aot-inductor`` artifacts are ever adopted, by name), so every one
     minted was pod time and platform storage spent on an artifact nothing
     could ever adopt. There is no seal, no key, no publish and no
     ``cell_store`` row on this path; the ONLY artifact class this module
     produces is ``aot-inductor``.

The publish transport reuses the existing repo-commit machinery
(``convert.hub.HubClient``) with a capability token minted by
``POST /v1/worker/cells/publish-intent`` (worker JWT) — the hub corroborates
every claimed key axis against its own records and pins the token to
exactly this cell key; the endpoint-scoped ``cell_store`` row is stamped
hub-side from the token claim, never from anything this module sends.

cozy-local USES this module since pgw#1127, through ``local_serve``, with
``publisher=None`` — one arming brain, two sinks. What it never has is a
PUBLISHER: user-controlled hardware is untrusted tier by definition, so its
cells land in ``local_cell_store`` (``local_keep_reason`` -> ``no_publish_sink``)
and its obligation ends at :func:`keep_self_mint_local`. That absence is
pinned structurally by ``tests/test_local_serve_no_publisher_pgw1127.py``,
not by this paragraph: before pgw#1127 the local CLI armed JIT through
``local_cells`` and never reached the local store at all.

Mint failures keep the pre-self-mint miss policy: plain lanes serve eager,
quantized (w8a8/w4a4) lanes keep their typed fail-closed refusal.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple)


from . import activity as activity_mod
from . import aot_identity, aot_serve, artifact_meta, cell_key, env_seal
from . import boot_phases as boot_mod
from . import local_cell_store
from . import serve_posture
from . import compile_cache as cc
from .cell_adopt import AdoptOutcome, CellAdoption, EagerPhase
from .models.chunk_cas import sha256_file
from .procsplit import broker
# module import (not `from .loading import pipeline_weight_lane`): tests
# monkeypatch models.loading.pipeline_weight_lane; stay late-bound.
from .models import loading, provision
from .request_context._helpers import _decode_unverified_jwt_claims
from .convert.hub import HubPublishError
from .api.export_contract import (
    blocker_refusal, export_declaration, open_blockers)
from .models import lora_lifted

logger = logging.getLogger(__name__)

#: pgw#805 mint recipes, as re-cut by pgw#1010. ``aot`` (torch.export +
#: AOTInductor) is the ONLY recipe that produces an artifact — the only kind
#: a hub-named ``Arm.artifact`` can deliver (pgw#904). ``dynamo`` is no longer a mint at
#: all: it names the JIT INTAKE posture a family without an export declaration
#: serves under (compile in this process, serve compiled for this pod's life,
#: publish nothing). Kept as a distinct token because every decline that
#: chooses it is on the wire, and the hub counts them.
RECIPE_DYNAMO = "dynamo"
RECIPE_AOT = "aot"


@dataclass(frozen=True)
class SelfMint:
    """Identity of one successfully adopted, FINALIZED self-minted cell.

    Produced only by :func:`adopt_delegated_mint`, after the child's cell
    passes the same arm gates a hub-delivered one does. The serving-bootstrap
    half of gw#587/th#910: the minting worker ADVERTISES this identity — the
    key STAMPED on the bytes it serves — so ``active_compile_artifacts``
    accounting treats the mint exactly like a hub-delivered cell; the warmup
    proof, not the artifact source, gates serving.

    pgw#1032: the hub fence this feeds is the one that verifies the ADVERTISED
    active ref against the hub's own store. The old self-attested spelling
    (``ActiveCompileRef == KeyRef(family, requested_cell_key)``) compared a
    stamped key against a COMPUTED one — disjoint spaces since pgw#1010, so it
    could never match — and is retired with the requested key itself.
    """

    family: str
    cell_key: str
    ref: str  # "root/family-<f>#<key>" — compile_cache.system_repo + key
    snapshot_digest: str  # "sha256:<hex>" of the packed artifact (self-attested)
    artifact: Path


#: The mint-obligation identity prefix. DELIBERATELY not ``ck``-shaped: an
#: arm token must never pass ``cell_key.is_key`` / the hub's
#: ``compilecache.IsCellKey``, because it is NOT a cell key — see
#: :class:`ArmIdentity`.
#:
#: The digit is the token's FACT-SET SCHEMA, and it is the memo-invalidation
#: mechanism (pgw#1113): ``arm2`` states the compile SUBJECT, ``arm1`` did
#: not, so an ``arm1-`` memo answers a question no ``arm2`` reader is asking.
#: A stale-schema entry is therefore unaddressable by construction rather
#: than misreadable, and :func:`arm_from_local_store` sweeps the predecessor
#: files once per process instead of leaving them to accumulate silently.
ARM_SCHEME = "arm2"


@dataclass(frozen=True)
class ArmIdentity:
    """The PRE-TRACE identity of one owed AOT mint — NOT a cell key.

    pgw#1059: the ck1 key is four axes (graph x envelope x sm x toolchain)
    and its ``graph`` axis is the traced-graph digest, which does not exist
    until the export finishes — so an obligation opened BEFORE any trace
    structurally cannot state a cell key, and pretending otherwise is the
    attempt-28 phantom (one axis name, two derivations, read as "the child
    computed a different key"). This type is what an obligation CAN state:
    every pre-trace-knowable fact, each produced by the SAME derivation the
    child's recorded metadata uses, so ``arm_axis_divergence`` can compare
    them byte-for-byte across the process boundary and name the first fact
    that failed to survive it.

    ``token`` (``arm1-<56hex>``) is a process-local ledger key and event
    label only: the pending/finalized/quarantine ledgers key on it, and the
    ``self_mint_*`` events print it as ``arm_key=``. It never selects a
    cell, never reaches a store row, and never matches ``IsCellKey``.
    """

    facts: tuple  # sorted ((name, value), ...)

    def facts_dict(self) -> Dict[str, str]:
        return dict(self.facts)

    @property
    def token(self) -> str:
        canonical = json.dumps(
            self.facts_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True,
        )
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        return f"{ARM_SCHEME}-{digest[:56]}"


#: The ENVIRONMENT half of an :class:`ArmIdentity` — the facts a delegated
#: child re-derives in its own process and RECORDS on the cell it hands back.
#: ``envelope`` and ``toolchain`` use the exported key's own derivations
#: (``cell_key.envelope_digest`` / ``cell_key.facts_digest``), ``lane`` the
#: one lane label (``cc.execution_lane_label``) — :func:`arm_axis_divergence`
#: compares exactly these, so an axis that fails to survive the parent->child
#: boundary is refused BY NAME at the handback seam.
#: ``graph`` is deliberately absent: it exists only after the export, and
#: comparing a declared-facts stand-in against the traced fact is the
#: phantom divergence this type retires.
ARM_ENVIRONMENT_FACTS = ("family", "format", "lane", "sm", "envelope",
                         "env_seal", "toolchain")

#: The SUBJECT half (pgw#1113): WHAT this obligation compiles, as opposed to
#: what runtime it compiles on. ``subject`` is the resolved slot identity
#: (:func:`cell_key.subject_digest` — which slot, which checkpoint refs,
#: which snapshot digest); ``targets``/``dynamic``/``regional`` are the rest
#: of ``cc.declared_compile_facts`` the token could not previously see.
#:
#: These are NOT compared at the handback seam, and that is the asymmetry
#: this issue is about rather than an omission: a cell records no subject and
#: must not, because one cell legally serves every checkpoint whose graph it
#: is. The subject splits the OBLIGATION — which pipe owes which mint, which
#: pending they may share, which memo row answers them — and the cell's own
#: key, which is the traced computation, is what says the computation matches.
ARM_SUBJECT_FACTS = ("subject", "targets", "dynamic", "regional")

#: Every fact in the token, in report order.
ARM_FACTS = ARM_ENVIRONMENT_FACTS + ARM_SUBJECT_FACTS

#: The pipeline attribute carrying the resolved :class:`cell_key.SlotSubject`
#: set the executor built this object from (pgw#1113). Stamped beside the
#: execution lane, read here for the same reason the lane is read here rather
#: than threaded through six call sites: the pipe is the one handle every arm
#: site holds. A pipeline the worker did not resolve carries none, and
#: :func:`cell_key.subject_digest` answers "" for it — honestly narrower, not
#: silently equal to some other subject.
ARM_SUBJECT_ATTR = "_cozy_arm_subject"


def stamp_arm_subject(
    pipe: Any, slot: str, refs: Sequence[str], snapshot_digest: str = "",
) -> None:
    """Record that ``pipe`` was resolved from ``slot`` at ``refs``.

    Additive per slot: a shared-component pipeline serving two slots ends up
    stating both. Best effort — a pipeline object that refuses attributes
    leaves the subject unstated, which is the pre-pgw#1113 posture and never
    an exception on a serving path.
    """
    subject = cell_key.SlotSubject(
        slot=str(slot or ""),
        refs=tuple(str(ref) for ref in refs if str(ref or "")),
        snapshot_digest=str(snapshot_digest or ""),
    )
    known = {sub.slot: sub for sub in pipeline_arm_subject(pipe)}
    known[subject.slot] = subject
    try:
        setattr(pipe, ARM_SUBJECT_ATTR,
                tuple(sorted(known.values(), key=lambda s: s.slot)))
    except Exception:  # noqa: BLE001 — a stamp is never worth a failed boot
        logger.debug("fleet-cells: pipeline refused the arm subject stamp",
                     exc_info=True)


def pipeline_arm_subject(pipe: Any) -> Tuple[cell_key.SlotSubject, ...]:
    """The resolved subject stamped on ``pipe``, or ``()``."""
    stamped = getattr(pipe, ARM_SUBJECT_ATTR, None) or ()
    return tuple(
        sub for sub in stamped if isinstance(sub, cell_key.SlotSubject))


def declared_envelope_block(cfg: Any) -> Dict[str, Any]:
    """The DECLARED-envelope block for ``cfg`` — byte-for-byte the same
    extraction :func:`aot_export_spec` performs (``shapes`` /
    ``text_lens`` / ``guidance_scales``, no fallbacks: ``text_len`` was
    dropped from the child handoff in pgw#1034), so the parent's pre-mint
    envelope digest and the digest of the block the child RECORDS under
    ``cell_key.EXPORT_ENVELOPE_KEY`` agree by construction (canonical form:
    ``cell_key.envelope_facts``). GPU-gauntlet-proven: a fallback here that
    the spec extraction does not share reds every handback as
    ``envelope`` divergence."""
    return {
        "shapes": [
            [int(v) for v in row] for row in (getattr(cfg, "shapes", ()) or ())],
        "text_lens": [int(v) for v in (getattr(cfg, "text_lens", ()) or ())],
        "guidance": [
            float(v) for v in (getattr(cfg, "guidance_scales", ()) or ())],
    }


def arm_identity(
    family: str, weight_lane: str, lora_bucket: int, cfg: Any,
    subject: Iterable[cell_key.SlotSubject] = (),
) -> ArmIdentity:
    """This runtime's :class:`ArmIdentity` for one owed mint.

    ``subject`` is WHAT is being compiled (pgw#1113): the resolved slots of
    the pipeline this obligation is for. Without it the token named a
    (family, lane) pair and nothing else, so two `@endpoint` classes sharing
    one ``Compile``, or two slots of one class bound to different
    checkpoints, computed ONE token — one pending, one child, one local-store
    memo row — and the first of them to arm handed its cell to the others.

    Raises :class:`ValueError` when the runtime cannot state a fact (no
    CUDA => no ``sm``): a worker that cannot state its obligation identity
    has no obligation to open.
    """
    sm = str(cc.runtime_key().get("sm") or "")
    if not sm:
        raise ValueError(
            "cannot state the compute capability (sm) of this runtime; no "
            "mint obligation can be opened without it")
    # ONE derivation of the declaration's facts (``cc.declared_compile_facts``
    # is the canonical form the cozy-local store verdict and the JIT semantic
    # tag already read), so the obligation and the contract cannot disagree
    # about what was declared. ``targets`` keeps DECLARATION ORDER: the child
    # picks its compile target first-match (``mint_child.pick_compile_target``),
    # so the order is meaning, not presentation.
    declared = cc.declared_compile_facts(
        cfg, lora_bucket_override=int(lora_bucket or 0))
    facts = {
        "family": str(family or ""),
        "format": str(cc.ARTIFACT_FORMAT),
        "lane": cc.execution_lane_label(
            str(weight_lane or ""), int(lora_bucket or 0)),
        "sm": sm,
        "envelope": cell_key.envelope_digest(declared_envelope_block(cfg)),
        "env_seal": env_seal.seal_digest(env_seal.effective_seal()),
        "toolchain": cell_key.facts_digest(dict(cc.toolchain_digest())),
        "subject": cell_key.subject_digest(subject),
        "targets": ",".join(str(t) for t in declared["targets"]),
        "dynamic": json.dumps(
            declared["dynamic"], sort_keys=True, separators=(",", ":")),
        "regional": "1" if declared["regional"] else "0",
    }
    return ArmIdentity(facts=tuple(sorted(facts.items())))


@dataclass(frozen=True)
class PendingSelfMint:
    """An AOT mint this worker OWES, handed to a child process (pgw#784).

    ``enable_compiled`` returns this on a miss instead of an already-packed
    :class:`SelfMint`: the live pipeline keeps serving eager while the child
    exports, and ``adopt_delegated_mint`` swaps it through the ordinary
    delivered-cell path once the child's cell earns adoption.

    One instance may be SHARED by several pipelines of one record whose
    facts compute the same obligation identity — which since pgw#1113
    includes the SUBJECT (which slot, resolved to which checkpoint), so
    sharing means "provably the same thing to compile", not "the same family
    on the same card". ``_state`` memoizes the adopt outcome so sibling
    candidates converge on one publish.

    pgw#1010: a DYNAMO miss no longer builds one of these. The JIT recipe
    keeps its serving role (intake, ``compile_cache.arm_jit_intake``) and
    produces no artifact at all, so every pending in this module is an AOT
    mint and every pending is delegated.
    """

    family: str
    #: The :class:`ArmIdentity` token (``arm1-…``) — the obligation's
    #: process-local ledger key and event label, NEVER a cell key
    #: (pgw#1059): the cell's key exists only once the child's export
    #: finishes, and it is the STAMP on the returned artifact.
    arm_token: str
    ref: str
    cfg: Any
    target: Path
    mint_root: Path
    publisher: Optional["CellPublisher"]
    cache_dir: Optional[Path] = None
    #: pgw#1042/pgw#1059: the parent's computed pre-trace identity, facts
    #: and all. Every fact the parent could state (family, format, lane, sm,
    #: envelope, env_seal, toolchain) must be byte-identical across the
    #: process boundary — same derivations both sides — and
    #: ``adopt_delegated_mint`` refuses BY FACT NAME when one is not.
    #: ``graph`` is structurally absent pre-trace and is never compared.
    arm_key: Optional[ArmIdentity] = None
    #: pgw#784: this mint is built by a CHILD PROCESS, so the live pipeline
    #: was never armed and this process holds no capture. The live pipe stays
    #: plain eager until ``adopt_delegated_mint`` swaps it through the
    #: ordinary delivered-cell path. Always true since pgw#1010 — the
    #: in-process capture existed only to pack a dynamo cell.
    delegated: bool = True
    #: th#1355: when this mint was ARMED. arm -> finalize is the window the
    #: compile happens in, so the elapsed time is this cell's mint cost. It is
    #: reported to the hub at publish-intent and lands on the cell's own
    #: `cell_store` row, because "what did this cell cost to build" was
    #: previously answerable only from an activity event that carried no cell
    #: key. Monotonic: a wall-clock step must not turn a mint negative.
    armed_at: float = dataclasses.field(default_factory=time.monotonic)
    _state: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class ArmOutcome:
    """Result of the fleet arming policy for one pipeline.

    ``armed`` mirrors the old boolean; ``self_mint`` is set only when the
    arm was satisfied by this worker's OWN mint (never for a delivered
    cell) — either a :class:`PendingSelfMint` (fresh arm, not yet proven)
    or, from callers that already hold a finalized identity, a
    :class:`SelfMint` — letting the executor synthesize the artifact
    selection it records/advertises.

    ``selection_bug`` carries a caught :class:`CellSelectionBugError`
    (th#1031): the th#883 invariant is a self-requested, identity-verified
    cell that never OUGHT to fail, so it stays a loud, wire-visible event —
    but a seeded self-cell verdict rules on pre-trace facts only (the
    traced graph is not knowable before the arm), so two structurally
    different graphs can legitimately share a verdict and a caught
    instance no longer aborts arming. The caller reports it (ModelEvent /
    pod_events, unchanged) while this same call proceeds to self-mint —
    the ordinary MISS recovery — instead of leaving the pipeline stuck
    retrying the identical unusable cell on every subsequent request.

    ``eager_reason`` (pgw#824) is the CLASSIFIED token for why this pipeline is
    not serving from a cell — a :class:`~.cell_adopt.EagerPhase` member, which
    is where that vocabulary is spelled ONCE — the same token the decline's
    ``self_mint_skipped``/``self_mint_started`` event carries in ``phase``, so a
    request row's ``fallback_reason`` and the worker's activity events join on
    one string. "" only when ``armed`` is true. Without it every eager request
    on a compile-declaring release reported ``serving_mode=eager,
    fallback_reason=""``, and "why is this fleet eager right now" was a question
    no query could answer.
    """

    armed: bool
    self_mint: Optional[Any] = None
    selection_bug: Optional["cc.CellSelectionBugError"] = None
    eager_reason: str = ""
    #: pgw#923: every ADOPTION attempt this policy made, in order, measured.
    #: A self-mint is not an adoption and never appears here. The caller turns
    #: these into the `ModelEvent{ADOPTED}` / `{FAILED}` the hub already stores
    #: as `kind=compile_cache_adopt` — the one place the wire is owned.
    adoptions: Tuple[CellAdoption, ...] = ()

    def __bool__(self) -> bool:
        return self.armed

_INTENT_TIMEOUT_S = 30
_COMPLETE_TIMEOUT_S = 30

# Live self-mint obligations by ARM token. The inductor capture dir is
# process-global (one TORCHINDUCTOR_CACHE_DIR), so at most one obligation's
# capture may be live at a time; same-token sibling pipes join it.
_PENDING_LOCK = threading.Lock()
_PENDING: Dict[str, "PendingSelfMint"] = {}
# pgw#672: cells this process already finalized (packed + folded into the
# live cache root). A later same-obligation arm re-arms cache_ready from the
# folded entries instead of opening a SECOND capture — which, with the first
# mint's compiled code resident in dynamo's in-memory cache, would capture
# nothing and disprove itself at finalize (the L4 churn loop).
#
# pgw#1033/pgw#1059: keyed on the ARM TOKEN (`ArmIdentity.token`) — the
# pre-trace obligation identity the miss path names its pending with — and
# NOT on the cell's own stamped key, which does not exist until the export
# finishes. A ledger written under the stamped key could never be read by
# the only caller there is: an arm that has computed its obligation and not
# yet minted anything. The VALUE carries the stamped identity
# (`SelfMint.cell_key`/`.ref`) — this map IS the process's
# arm-token -> stamped-cell index, and the quarantine gate below reads it
# for exactly that.
_FINALIZED: Dict[str, "SelfMint"] = {}


def finalized_in_process(key: str) -> Optional["SelfMint"]:
    """The cell this process already minted and adopted for ARM token
    ``key`` (:class:`ArmIdentity`; pgw#1033); the returned
    :class:`SelfMint` carries the artifact's own STAMPED key."""
    with _PENDING_LOCK:
        return _FINALIZED.get(str(key or "").strip())


# pgw#712 fence marker (see publish()): presence in metadata refuses
# republication. Nothing in-tree stamps it under exact identity.
ADOPTION_MARK = "equivalence_adopted"


#: th#1423: the code a 401 gets when the credential we presented was ALREADY
#: past its own `exp`. "Expired" and "revoked/wrong-worker" are different
#: operator actions, and only the worker can tell them apart — the hub sees an
#: unusable token either way.
CREDENTIAL_EXPIRED_CODE = "worker_credential_expired"


class CellPublishRefused(Exception):
    """Typed hub refusal (attestation / trust tier / quota). Terminal for
    this publish attempt — never retried, never fatal to serving.

    Carries the hub's own ``status``/``code`` for the same reason
    :class:`convert.hub.HubPublishError` does: a refusal reason re-derived from
    ``str(exc)`` is prose that nothing can group by.
    """

    def __init__(self, message: str, *, status: int = 0, code: str = "") -> None:
        super().__init__(message)
        self.status = int(status or 0)
        self.code = str(code or "")


def _hub_error_code(body: Any) -> str:
    """The hub's ``error.code``, accepting both envelope shapes it has used —
    nested ``{"error": {"code": ...}}`` and the flat ``{"code": ...}`` the
    worker-JWT refusals were observed with (th#1423)."""
    if not isinstance(body, dict):
        return ""
    err = body.get("error")
    if isinstance(err, dict):
        return str(err.get("code") or "")
    flat = str(body.get("code") or "")
    # pgw#987: gin's `AbortWithStatusJSON` (the body-cap middleware) emits the
    # code as a bare string under `error` and carries no `code` at all, so a
    # `request_body_too_large` refusal named itself and the client dropped the
    # name.
    return flat or (err.strip() if isinstance(err, str) else "")


def _credential_lapse_s(token: str, *, now: float) -> float:
    """Seconds ``token`` is PAST its own ``exp``; 0.0 when live or unreadable.

    Not a timeout (gw#666): ``exp`` is an absolute instant the credential
    itself carries, not a duration this code picked.
    """

    try:
        exp = float(_decode_unverified_jwt_claims(token).get("exp") or 0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, now - exp) if exp > 0 else 0.0


# th#1645/pgw#987: the blocks a cell's envelope carries that are UNBOUNDED in
# the size of the model, and that the publish declare must therefore not
# forward.
#
# Every one of them already ships INSIDE the artifact (`metadata.json` in the
# .tar.gz) and is read from there — `guard_closure.load_manifest` opens the
# tarball, `aot_serve` reads `entries` off the unpacked package. The copy in
# the declare body had exactly one hub-side reader,
# `api/cell_receipts.go:242-248`, which hashes it into two `omitempty` audit
# claims that nothing verifies against anything (the worker parses
# `manifest_digest`/`fingerprint_digest` into a dataclass field and never
# reads them) — and the content is already bound by `snapshot_digest`, as
# that file's own header says.
#
# THE MEASUREMENT that makes this a bound and not a preference: on a REAL
# published sdxl cell (checkpoint sha256:926bc9f5…, artifact 69,045,459 B),
# `guard_manifest` alone is 13,092,487 bytes of the 13,377,167-byte metadata
# — 98%. Everything else together is 285 KB. The declare body therefore grew
# with the ARTIFACT, and at ~200 MB (a real AOT cell) it crossed the 32 MiB
# route cap and the hub answered 413 thirty-two times.
_UNBOUNDED_ENVELOPE_BLOCKS = frozenset({
    "guard_manifest",   # JIT lane: per-graph guard closures
    "entries",          # AOT lane: per-class contracts + constant manifests
    "composition",      # pgw#697 composition fingerprint rows
    "weight_contract",  # per-tensor weight rows
})

# pgw#904: the pgw#988 declare-contract assert died with `aot_cells`. The
# declare's worker-side CONSUMER (fetch-and-filter discovery) is deleted —
# the hub now resolves the exact artifact and names it in `Arm.artifact`, so
# the declare's reader is the hub, and its contract is enforced there.

# The stated ceiling on a cell's CONTROL-plane declare (§4.24). Measured
# basis: the same real cell's metadata minus the blocks above is 285 KB, and
# the AOT lane's phase table adds tens of KB — so 4 MiB is roughly an order
# of magnitude of headroom over anything observed, while staying far below
# the route's 32 MiB.
#
# THREAT: without it, the next unbounded block someone adds to the envelope
# re-creates th#1645 exactly — a publish that cannot succeed, discovered as a
# 413 that names no key, ten minutes into a paid pod. This bound fails on the
# POD, before the wire, and NAMES the block that broke it.
CELL_DECLARE_MAX_BYTES = 4 << 20

#: The groupable phase a declare-bound refusal reports. Its own token, not the
#: generic `refused`: this refusal means "someone added an unbounded block to
#: the envelope", which is a code defect with a named owner, and it must not
#: land in the same bucket as the hub's trust-tier and quota decisions.
CELL_DECLARE_OVERSIZE_CODE = "cell_declare_oversize"


def control_plane_metadata(meta: Mapping[str, Any]) -> Dict[str, Any]:
    """The bounded subset of a cell envelope the publish declare may carry.

    The seam carries CONTROL, not DATA (pgw#763/#783). A cell's bulk envelope
    is data: it rides the artifact, over presigned PUTs, digest-bound. What
    the hub actually selects a cell on — family, key, sm, sku, image digest,
    gen_worker version, lane — reaches it on the publish-INTENT call and is
    stored in `cell_store`'s own columns; none of it is read back out of this
    dict.

    Raises :class:`CellPublishRefused` when what survives still exceeds
    :data:`CELL_DECLARE_MAX_BYTES`, naming the largest surviving key — a new
    unbounded block must be a named refusal here, not a 413 later.
    """
    kept = {
        k: v for k, v in dict(meta).items()
        if v is not None and k not in _UNBOUNDED_ENVELOPE_BLOCKS
    }
    encoded = len(json.dumps(kept, sort_keys=True, default=str).encode())
    if encoded > CELL_DECLARE_MAX_BYTES:
        widest = max(
            kept,
            key=lambda k: len(json.dumps(kept[k], sort_keys=True, default=str)),
            default="")
        raise CellPublishRefused(
            f"cell declare is {encoded} bytes, over the "
            f"{CELL_DECLARE_MAX_BYTES}-byte control-plane bound (th#1645); "
            f"the largest block is {widest!r} — it belongs in the artifact, "
            "not in the declare",
            code=CELL_DECLARE_OVERSIZE_CODE)
    return kept


class CellPublisher:
    """The fleet publish sink: intent -> commit flow -> complete.

    ``base_url`` is the hub base the worker already uses for its other
    worker-JWT surfaces (HelloAck ``file_base_url``); the repo-commit API
    lives on the same combined-binary host. ``worker_jwt`` is a zero-arg
    provider so token rotation (#561) is picked up per call.
    """

    def __init__(
        self,
        *,
        base_url: str,
        worker_jwt: Callable[[], str],
        image_digest: str,
    ) -> None:
        self.base_url = str(base_url or "").strip().rstrip("/")
        self._worker_jwt = worker_jwt
        self.image_digest = str(image_digest or "").strip()

    def enabled(self) -> bool:
        # pgw#763 delta 1: "we hold a worker JWT" stopped being the test for
        # "we can make a worker-authenticated call". In the compute child the
        # credential is the PARENT's and the call is mediated, so the honest
        # question is whether either route exists.
        return bool(
            self.base_url
            and ((self._worker_jwt() or "").strip() or broker.active())
        )

    def worker_jwt(self) -> str:
        """Current worker JWT (rotation-aware, #561)."""
        return str(self._worker_jwt() or "")

    # -- wire ---------------------------------------------------------------

    def _post(self, path: str, payload: dict, *, timeout: float) -> dict:

        # pgw#763 delta 1: parent-mediated under the split — the compute child
        # holds no worker JWT, so the parent makes the attested-intent call and
        # returns the KEY-PINNED capability token, which is a short-TTL,
        # least-authority grant the child is explicitly allowed to hold. The
        # cell bytes still go child -> CAS directly under that token: the seam
        # carries control, not data.
        # Read the credential ONCE: the bearer we present and the bearer whose
        # expiry we blame must be the same string, or a rotation landing
        # mid-call makes the diagnosis describe a token we never sent.
        bearer = self._worker_jwt()
        # pgw#1087: the worker<->hub cell control-plane round trip, timed.
        # There is NO worker-side key LOOKUP to measure — pgw#904 moved cell
        # resolution to the hub (`Arm.artifact`) and pgw#1032 deleted
        # `StateDelta.cell_lookups` — so these publish legs are the whole of
        # the hub RTT a boot's cell path pays, and the issue's "hub key lookup
        # round-trip" line is closed by naming that rather than by inventing a
        # phase with no producer (the pgw#924 rule).
        with boot_mod.span(
            boot_mod.PHASE_CELL_HUB_RTT, function=path,
        ) if boot_mod.in_boot() else contextlib.nullcontext() as sp:
            resp = broker.request(
                "POST",
                path,
                base_url=self.base_url,
                bearer=bearer,
                json=payload,
                timeout=timeout,
            )
            if sp is not None:
                sp.classify(f"http_{resp.status_code}")
        body: dict = {}
        try:
            body = resp.json() if resp.text else {}
        except Exception:
            body = {}
        code = _hub_error_code(body)
        if resp.status_code in (403, 429):
            # Typed refusals (cell_publish_forged_axis, _untrusted_tier,
            # _quota_exceeded, _family_undeclared): terminal by design.
            raise CellPublishRefused(
                f"{path} refused ({resp.status_code}): {resp.text[:300]}",
                status=resp.status_code, code=code)
        if resp.status_code < 200 or resp.status_code >= 300:
            # th#1423: this raised a BARE RuntimeError, so `_publish_failure_phase`
            # had nothing to group by and three production 401s all landed under
            # the phase `RuntimeError`. The typed carrier already exists.
            lapse = (_credential_lapse_s(bearer, now=time.time())
                     if resp.status_code == 401 else 0.0)
            raise HubPublishError(
                f"{path} failed ({resp.status_code}): {resp.text[:300]}"
                + (f" — the credential presented was {lapse:.0f}s past its own exp"
                   if lapse else ""),
                status=resp.status_code,
                code=CREDENTIAL_EXPIRED_CODE if lapse else code,
            )
        return body if isinstance(body, dict) else {}

    # -- publish ------------------------------------------------------------

    def publish(self, family: str, artifact: Path, meta: dict,
                mint_duration_ms: int = 0) -> str:
        """Publish one self-minted cell. Returns the checkpoint id.

        Steps: attested intent (worker JWT; hub corroborates the axes and
        mints a key-pinned capability token) -> the CHUNKED SHA-256 publish
        (declare -> {have, need} -> PUT -> complete; mode=replace, no tags —
        the hub refuses any tag bind under the claim anyway) ->
        publish-complete bookkeeping. Raises on any failure; the caller
        treats every raise as non-fatal to serving.
        """

        # pgw#712 (kept under the exact-identity ruling as
        # defense-in-depth): a cell whose metadata carries a foreign
        # adoption provenance must never republish under this worker's
        # key. Nothing in-tree stamps the mark anymore (equivalence
        # adoption was deleted with exact identity); a marked cell can only be a
        # foreign/hand-copied artifact — refuse it.
        mark = meta.get(ADOPTION_MARK)
        if mark:
            raise CellPublishRefused(
                f"cell carries adoption provenance {mark!r}; republishing "
                "it is fenced (pgw#712)")
        key = str(meta.get("cell_key") or "").strip()
        if not key:
            key = _recomputed_key(meta).digest
        axes = {
            "sku": str(meta.get("sku") or ""),
            "image_digest": self.image_digest,
            "gen_worker": str(meta.get("gen_worker") or ""),
        }
        # th#1423: a mint outliving its credential is only visible AFTER the
        # compile, in a 401 nothing could group. The credential states its own
        # `exp`, so the lapse is a MEASURED fact at the one moment it decides
        # the outcome — on the wire before the intent, not inferred later.
        lapse = _credential_lapse_s(self.worker_jwt(), now=time.time())
        if lapse:
            _publish_leg(family, key, "credential_expired",
                         {"past_exp_s": int(lapse)})
        intent = self._post(
            "/v1/worker/cells/publish-intent",
            {
                "family": family, "cell_key": key, "axes": axes,
                # th#1355: the identity the hub could not otherwise learn.
                # `axes` above are the three the hub ATTESTS against its own
                # records; `identity_axes` are the ck axes that HASH INTO
                # `cell_key`. Before this, they reached the hub only on the
                # DEMAND side and were deleted the moment the demand was
                # satisfied — so a minted cell's row could not say which lane
                # or which card it was for. Diagnostic by contract: the hub
                # cannot recompute the digest and must never select on them.
                "identity_axes": _identity_axes(family, meta),
                # The compile wall for THIS cell. Sent at INTENT, not
                # complete, because the mint is already finished by the time
                # we ask to publish — so the cost commits in the same INSERT
                # as the row it describes.
                "mint_duration_ms": max(0, int(mint_duration_ms or 0)),
            },
            timeout=_INTENT_TIMEOUT_S,
        )
        token = str(intent.get("capability_token") or "").strip()
        repo = str(intent.get("repo") or "").strip()
        if not token or not repo:
            raise RuntimeError("publish-intent response missing token/repo")

        try:
            from .convert.hub import CommitFile, HubClient

            # th#1303/pgw#807 item 3 — THE FLIP, taken. Both gates that held
            # it are discharged: th#1340 gave the v2 route the cell-publish
            # claim (receipt + `cell_store` + `cell_receipts`, the same three
            # writes v1 does), and the receipt reader dispatches on the
            # receipt's OWN algorithm tag, so a sha256-bound cell resolves.
            # The v1 (blake3) route is FROZEN hub-side — it answers a cell
            # publish with 410 `unsupported_digest_algorithm` — so this is
            # the only transport a cell can ship over at all.
            #
            # What v2 buys a 40 MB cell beyond being accepted: the digest is
            # signed into each presigned PUT, so R2 itself refuses bytes that
            # do not hash to the key; and resume needs no client state — a
            # re-plan comes back with the landed objects already resident, so
            # a pod that dies mid-upload costs only the in-flight chunks.
            client = HubClient(base_url=self.base_url, token=token, owner="root")
            result = client.publish_v2(
                destination_repo=repo,
                files=[CommitFile(path=artifact.name, local_path=artifact)],
                mode="replace",
                flavor=key,
                # th#1645: CONTROL only. The bulk envelope is in the artifact.
                metadata=control_plane_metadata(meta),
                on_stage=lambda stage, facts: _publish_leg(
                    family, key, stage, facts),
            )
            checkpoint_id = result.checkpoint_id
        except Exception as exc:
            # Best-effort failure report so the hub's ledger/alarms see it.
            try:
                self._post(
                    "/v1/worker/cells/publish-complete",
                    {"family": family, "cell_key": key, "ok": False,
                     "error": str(exc)[:300]},
                    timeout=_COMPLETE_TIMEOUT_S,
                )
            except Exception:
                logger.debug("publish-complete failure report failed", exc_info=True)
            raise

        _publish_leg(family, key, "committed", {
            "checkpoint": checkpoint_id[:24], "publish_id": result.revision_id,
            "uploaded": result.uploaded, "resident": result.deduped,
            "bytes": result.total_bytes,
        })
        self._post(
            "/v1/worker/cells/publish-complete",
            {"family": family, "cell_key": key,
             "checkpoint_id": checkpoint_id, "ok": True},
            timeout=_COMPLETE_TIMEOUT_S,
        )
        logger.info(
            "fleet-cells: published %s#%s (checkpoint %s, %.1f MB, "
            "%d uploaded / %d resident)",
            family, key, checkpoint_id, artifact.stat().st_size / 1e6,
            result.uploaded, result.deduped)
        return checkpoint_id


def _publish_leg(family: str, key: str, stage: str, facts: Dict[str, Any]) -> None:
    """One typed `self_mint_publish` event per LEG of the publish protocol.

    The publish is a background thread on a pod that may not outlive it, and
    until now it emitted exactly two things: `started` and a terminus. "Ships
    40 MB" and "was refused before a byte moved" were therefore the same
    observation for as long as the thread lived — and the ONE run in program
    history that reached a cell publish spent a whole L4 mint to learn that
    distinction from a stack trace. Each leg is one wire event, so
    `worker_activity_events` can answer where a publish stopped.
    """
    detail = " ".join(f"{k}={v}" for k, v in sorted(facts.items()))
    activity_mod.emit_event(
        "self_mint_publish", f"family={family} key={key}: {detail}", phase=stage)


def _publish_failure_phase(exc: BaseException) -> str:
    """The stable, greppable token for WHY a publish stopped.

    The hub's own `error.code` (or a th#1301 projection's `failure.code`)
    when it named one — `unsupported_digest_algorithm`,
    `cell_publish_flavor_mismatch`, `cell_receipt_signer_unavailable` — else
    the status class, else the exception type. Never prose: a phase that
    varies with the message cannot be grouped, and grouping refusals by
    reason is the whole point of putting them on the wire.
    """
    code = str(getattr(exc, "code", "") or "").strip()
    if code:
        return code[:120]
    status = int(getattr(exc, "status", 0) or 0)
    if status:
        return f"http_{status}"
    return type(exc).__name__


#: pgw#1046: the published entry carrying the artifact's
#: ``combined_graph_hash`` — the value pgw#903's pre-dlopen fence compares
#: against ``Arm.graph_contract_digest``, and the key tensorhub's producer
#: reads (``runattempt.ArmFromVerifiedCell``, `axisGraphContract`). Since
#: pgw#1059 its value IS the ``graph`` key axis; the hub-consumed entry NAME
#: is deliberately unchanged (a wire rename needs a paired tensorhub change).
GRAPH_CONTRACT_AXIS = "graph_contract"

#: pgw#1059: the published NON-KEY entry carrying the artifact's env-seal
#: digest. The seal left the key (amendment 4 — its declaration folds into
#: the ``toolchain`` axis), but the hub's ``ArtifactFromCellRecord`` reads
#: this entry to build ``ArtifactIdentity.env_seal_digest``, which pgw#904's
#: consumer requires — so it rides the map as a wire fact, exactly like
#: ``graph_contract``.
ENV_SEAL_AXIS = "env_seal"


def _recomputed_key(meta: Mapping[str, Any]) -> cell_key.CellKey:
    """The key this cell's OWN recorded facts describe.

    One derivation for the whole publish path, so the key a cell is
    published UNDER and the axes it is published WITH can never come from
    two different derivations of its metadata. Only exported
    (``aot-inductor``) cells have identity (pgw#1010/pgw#1059) — any other
    kind is refused here by the derivation itself.
    """
    try:
        return cell_key.from_exported_artifact_metadata(meta)
    except cell_key.CellKeyError as exc:
        raise CellPublishRefused(
            f"cell states no computable identity ({exc}); publishing it under "
            "partial axes would produce a row the fleet cannot arm from "
            "(pgw#1046)") from exc


def _identity_axes(family: str, meta: dict) -> Dict[str, str]:
    """The identity map the hub records for this cell, RECOMPUTED from the
    artifact's own facts.

    This is not inventory. th#1457's producer builds the worker's
    ``ExecutionSpec`` out of exactly this map: ``ArtifactFromCellRecord`` reads
    ``toolchain`` and ``env_seal`` from it, ``ArmFromVerifiedCell`` reads
    ``graph_contract``, and pgw#904's landed consumer REFUSES an
    ``ArtifactIdentity`` missing any of them. A row published without them is a
    cell the fleet can never arm.

    So this FAILS CLOSED (pgw#1046): a mint that cannot name an axis raises
    :class:`CellPublishRefused` here, before a byte moves.

    Contents (pgw#1059): the four ck1 key axes (``graph``, ``envelope``,
    ``sm``, ``toolchain``) verbatim, plus the wire facts the hub requires by
    name (``graph_contract`` — same value as ``graph``; ``env_seal``) and
    the demoted store metadata (``family``, ``lane`` — discovery scoping and
    row self-description, never identity).
    """
    key = _recomputed_key(meta)
    stamped = str(meta.get("cell_key") or "").strip()
    if stamped and stamped != key.digest:
        raise CellPublishRefused(
            f"cell_key stamp {stamped} disagrees with the key its recorded "
            f"axes describe ({key.digest}); refusing to publish an identity "
            "the artifact does not corroborate")
    axes = {k: str(v) for k, v in key.axes_dict().items()}
    # Non-empty by construction: the key above refuses a cell whose
    # metadata records no `combined_graph_hash` at all.
    axes[GRAPH_CONTRACT_AXIS] = str(meta["combined_graph_hash"]).strip()
    seal = meta.get(env_seal.SEAL_KEY)
    if not isinstance(seal, dict) or not seal:
        # The seal left the KEY (pgw#1059 amendment 4) but not the wire: the
        # hub's ArtifactFromCellRecord requires the entry, and a row without
        # it is a cell the fleet can never arm — same fail-closed rule as
        # the axes above.
        raise CellPublishRefused(
            "cell records no env_seal block; the hub's ArtifactIdentity "
            "requires its digest (pgw#903/pgw#1046)")
    axes[ENV_SEAL_AXIS] = env_seal.seal_digest(seal)
    axes["family"] = str(meta.get("family") or family or "")
    axes["lane"] = cc.execution_lane_label(
        str(meta.get("weight_lane") or ""),
        int(meta.get("lora_bucket") or 0))
    return axes


#: pgw#815: publishes currently in flight, keyed by cell key. A self-mint
#: publish is a fire-and-forget daemon thread, so an interrupted upload used
#: to leave NO trace anywhere: no cell row, no receipt, no event, no error —
#: the pod paid a 24-minute compile and reported success. This registry makes
#: an in-flight publish an observable fact that :func:`publishes_in_flight`
#: (drain/shutdown) and the executor's terminus assertion can both see.
_IN_FLIGHT_LOCK = threading.Lock()
_IN_FLIGHT: Dict[str, Tuple[str, float]] = {}

#: th#1359: the SETTLED half of the same ledger — ``{cell_key: checkpoint_id}``
#: for publishes the hub acknowledged, and ``{cell_key: phase}`` for the ones
#: it refused. In-flight alone answers "is an upload running"; a forge pod also
#: has to answer "did this pod produce a cell for the fleet, or not", and that
#: fact otherwise exists only as a wire event this process cannot read back.
_PUBLISHED: Dict[str, str] = {}
_REFUSED: Dict[str, str] = {}

#: pgw#848 item 1 / th#1359: a monotonic counter of DURABLE publish progress.
#: It advances when a NEW cell key begins uploading and when one lands — never
#: on a retry, never on a failure, never on "a message arrived". That
#: restriction is the whole design: the mint's activity stays RUNNING until the
#: cell is durable, and a publish that fails and retries forever must therefore
#: read as NOT PROGRESSING, or the fix is worse than the bug it closes (a
#: never-reap loop on a paid card with no attendant, which is exactly why
#: `self_mint_publish` was refused as a podguard progress KIND).
_DURABLE_PROGRESS = 0
_DURABLE_SEEN: set = set()


def publish_durable_progress() -> int:
    """Monotonic count of durable publish transitions in this process."""
    with _IN_FLIGHT_LOCK:
        return _DURABLE_PROGRESS


def _note_durable(key: str, event: str) -> None:
    """Advance the durable counter for a transition that is NOT a retry."""
    global _DURABLE_PROGRESS
    token = f"{key}|{event}"
    with _IN_FLIGHT_LOCK:
        if token in _DURABLE_SEEN:
            return
        _DURABLE_SEEN.add(token)
        _DURABLE_PROGRESS += 1


def publishes_in_flight() -> Dict[str, Tuple[str, float]]:
    """``{cell_key: (family, started_monotonic)}`` for every publish whose
    thread has neither succeeded nor failed yet (pgw#815)."""
    with _IN_FLIGHT_LOCK:
        return dict(_IN_FLIGHT)


def published_cells() -> Dict[str, str]:
    """``{cell_key: checkpoint_id}`` the hub accepted from this process."""
    with _IN_FLIGHT_LOCK:
        return dict(_PUBLISHED)


def refused_publishes() -> Dict[str, str]:
    """``{cell_key: failure phase}`` for publishes this process could not land."""
    with _IN_FLIGHT_LOCK:
        return dict(_REFUSED)


def _publish_async(
    publisher: CellPublisher, family: str, artifact: Path, meta: dict,
    cell_key_digest: str = "", mint_duration_ms: int = 0,
    arm_token: str = "",
) -> threading.Thread:
    """Ship the cell in the background — readiness never waits on an upload.

    EVERY outcome is a typed event now (pgw#815), success included: this
    boundary used to emit only on failure, so "published" and "the thread was
    killed mid-upload when the pod retired" were the same observation —
    silence. The mint dir is cleaned once the publish attempt finishes (the
    adoption already staged its own copy under the cache dir).
    """
    key = cell_key_digest or str(meta.get("cell_key") or "")
    try:
        size_mb = artifact.stat().st_size / 1e6
    except OSError:
        size_mb = 0.0
    with _IN_FLIGHT_LOCK:
        _IN_FLIGHT[key] = (family, time.monotonic())
    # A NEW key beginning its upload is durable new work; a retry of the same
    # key is not, and `_note_durable` dedupes on (key, event) to enforce that.
    _note_durable(key, "started")
    activity_mod.emit_event(
        "self_mint_publish",
        f"family={family} key={key}: uploading {size_mb:.1f} MB to the fleet "
        f"store; this pod must survive the upload or the cell is lost",
        phase="started",
    )

    def run() -> None:
        t0 = time.monotonic()
        try:
            checkpoint_id = publisher.publish(
                family, artifact, meta, mint_duration_ms)
        except CellPublishRefused as exc:
            logger.warning("fleet-cells: publish refused (hub decision): %s", exc)
            # pgw#1096/§4.28: the hub has just ASSERTED this hardware's trust
            # class. Learn it (the worker never declares its own — the code is
            # the only authority), and KEEP the cell: the `finally` below is
            # about to rmtree the mint root, and this exact discard is what
            # th#1643 books as SUNK ("a sealed cell was produced and thrown
            # away"). An untrusted machine mints for ITSELF, so the bytes go
            # to its own store and every later boot of it arms from disk.
            if local_cell_store.note_refusal(
                    str(getattr(exc, "code", "") or ""), str(exc)
            ) and cell_key.is_key(key):
                local_cell_store.store(
                    artifact, key=key, family=family, arm_token=arm_token)
            with _IN_FLIGHT_LOCK:
                _REFUSED[key] = "refused"
            activity_mod.emit_event(
                "self_mint_publish_failed",
                f"family={family} key={key}: hub refused the publish: {exc}",
                # th#1645: the hub's own code when it named one (and the
                # declare-bound's own token when the refusal never reached the
                # hub at all) — `refused` alone puts a code defect, a trust
                # tier and a quota in one bucket.
                phase=_publish_failure_phase(exc) or "refused",
            )
        except Exception as exc:  # noqa: BLE001 — reported, never fatal
            logger.warning("fleet-cells: publish failed; the next worker on this key re-mints", exc_info=True)
            with _IN_FLIGHT_LOCK:
                _REFUSED[key] = _publish_failure_phase(exc)
            activity_mod.emit_event(
                "self_mint_publish_failed",
                f"family={family} key={key}: publish attempt failed: "
                f"{type(exc).__name__}: {exc}",
                # The hub's OWN code when it gave one, so a fleet-wide
                # refusal is one `phase=` group instead of N prose strings.
                phase=_publish_failure_phase(exc),
            )
        else:
            with _IN_FLIGHT_LOCK:
                _PUBLISHED[key] = str(checkpoint_id or "")
            _note_durable(key, "published")
            activity_mod.emit_event(
                "self_mint_publish",
                f"family={family} key={key} checkpoint={checkpoint_id}: "
                f"{size_mb:.1f} MB published to the fleet store",
                phase="published",
                duration_ms=int(round((time.monotonic() - t0) * 1000)),
            )
        finally:
            with _IN_FLIGHT_LOCK:
                _IN_FLIGHT.pop(key, None)
            shutil.rmtree(artifact.parent, ignore_errors=True)

    t = threading.Thread(target=run, name="cell-publish", daemon=True)
    t.start()
    return t


#: Typed PIPELINE-side refusals of out-of-process minting (pgw#813). The
#: operator-side half lives in ``mint_delegate.delegation_refusal``.
REFUSAL_NO_EAGER_TIER = "no_eager_tier"


def delegation_refusal(pipe: Any, cfg: Any) -> str:
    """"" when this PIPELINE's mint may be DELEGATED, else the typed reason.

    The premise of pgw#784 is that the live pipeline keeps serving while a
    child compiles: nothing is armed here, so a pipe with no eager tier has
    nothing to serve at all until the child finishes.

    pgw#813 CORRECTION. This used to refuse ``mandatory_serving(pipe)`` — i.e.
    it read "executes quantized activations" as "cannot serve eager". That is
    a category error and it was the operative cause of AOT being unmintable on
    every lane: the plain lane is held on dynamo by #730, and w8a8 — the lane
    Paul ruled AOT-first — was refused a delegated minter here, so the miss
    fell back to a dynamo cell that AOT discovery can never adopt. A w8a8
    pipeline serves eager perfectly well (``_Fp8ScaledLinear.forward`` is a
    complete ``torch._scaled_mm`` forward; the fleet's own cold-boot ladder
    measures it; pgw#672/#673 already made mandatory lanes DEGRADE to eager
    loudly rather than raise). ``compile_cache.eager_tier_available`` is the
    honest predicate and this is now its only caller-side use.

    `Compile.regional` (the dynamo/JIT per-block knob, ie#381) is NOT a
    refusal here: since pgw#846 the AOT mint is always whole-graph and
    ignores it. A family with no registered EXPORT declaration cannot mint an
    aot-inductor cell at all, and ``mint_recipe`` declines that by its own
    name.
    """
    try:
        if not cc.eager_tier_available(pipe):
            return REFUSAL_NO_EAGER_TIER
    except Exception:  # noqa: BLE001 — an unanswerable arm keeps the old path
        return REFUSAL_NO_EAGER_TIER
    return ""


def _arm_candidate(
    pipe: Any,
    cfg: Any,
    cache_dir: Optional[Path],
    artifact: Optional[Path],
    *,
    ref: str,
    snapshot_digest: str,
    artifact_kind: str,
) -> Tuple[AdoptOutcome, CellAdoption]:
    """Arm ONE candidate cell, measured, and record the boot phase for it.

    pgw#923/#924. The `cell_arm` boot span and the adoption's `duration_ms`
    bracket the same interval, taken in the one place that does the arming, so
    the boot ladder and the hub's adoption measurement cannot disagree about
    what an arm cost. `cell_arm` was a DECLARED boot phase with no producer at
    all until this call site existed.
    """
    # A CANDIDATE must exist for this to be a cell arm. `provision.
    # enable_compiled` with no artifact also covers the seeded and ALLOW_COLD
    # inductor lanes, and bracketing those would put a near-zero `cell_arm` row
    # on every compile-declaring boot — the same "a default that reads as a
    # fact" defect pgw#924 is closing for `warmup`.
    span = (
        boot_mod.open_span(
            boot_mod.PHASE_CELL_ARM,
            ref=ref,
            artifact_kind=artifact_kind,
            artifact_key=snapshot_digest,
        )
        if artifact is not None and boot_mod.in_boot()
        else None
    )
    started = time.monotonic()
    try:
        outcome = provision.enable_compiled(pipe, cfg, cache_dir, artifact)
    except BaseException as exc:
        if span is not None:
            span.close(exc)
        raise
    arm_ms = int(round(max(0.0, time.monotonic() - started) * 1000.0))
    if span is not None:
        if outcome.armed:
            if outcome.identity:
                span.note(outcome.identity)
        else:
            span.refused(outcome.reason or "no_cell", outcome.detail)
        span.close()
    if outcome.armed:
        # pgw#1087: the SECOND user-visible timestamp. Deliberately NOT gated
        # on `in_boot()` — a self-minted cell routinely arms twenty minutes
        # after the boot closed, and "how long did this pod serve eager before
        # its cell arrived" is the question the compiled-serving campaign is
        # about. Recorded once per process: the interval is measured from
        # process start, so a re-arm hours later is not a second answer to it.
        boot_mod.mark_once(
            boot_mod.PHASE_COMPILED_SWAP,
            ref=ref, artifact_kind=artifact_kind, artifact_key=snapshot_digest,
            detail=outcome.identity)
    return outcome, CellAdoption(
        ref=ref,
        snapshot_digest=snapshot_digest,
        artifact_kind=artifact_kind,
        arm_ms=arm_ms,
        armed=outcome.armed,
        reason=outcome.reason,
        detail=outcome.detail or outcome.identity,
        pipeline_id=id(pipe),
    )


def enable_compiled(
    pipe: Any,
    cfg: Any,
    cache_dir: Optional[Path] = None,
    artifact: Optional[Path] = None,
    publisher: Optional[CellPublisher] = None,
    delegate: Optional[bool] = None,
    delivered_ref: str = "",
    delivered_digest: str = "",
    boot_local_key: str = "",
) -> ArmOutcome:
    """Fleet arming policy, plus the adoption ledger every exit shares.

    pgw#923: the policy has a dozen exits and an adoption attempt can precede
    any of them, so the measured attempts are collected in ONE place rather
    than threaded through each ``return``. That is the shape that lets a
    refusal be reported: the old code could only announce an adoption from the
    frame that made it, which is why the successful ones were narrated and the
    measured ones never sent.
    """
    if serve_posture.eager_only():
        # pgw#1142 / §4.32 item 4. `arming_block` would refuse every arm below
        # anyway, but the policy would first resolve, download and materialize
        # a cell to hand to a gate that has already decided — and on a MISS it
        # would open a self-mint whose child re-derives the refusal minutes and
        # a full compile later. The order is answered where it costs nothing,
        # and it is answered with a token, not a bare False.
        logger.info("fleet-cells: %s", serve_posture.block())
        return ArmOutcome(
            armed=False, eager_reason=EagerPhase.OPERATOR_EAGER_ONLY)
    adoptions: List[CellAdoption] = []
    outcome = _arming_policy(
        pipe, cfg, cache_dir, artifact,
        publisher=publisher, delegate=delegate,
        delivered_ref=delivered_ref, delivered_digest=delivered_digest,
        adoptions=adoptions, boot_local_key=boot_local_key,
    )
    if not adoptions:
        return outcome
    return dataclasses.replace(outcome, adoptions=tuple(adoptions))


class OrderedArmError(cc.CompiledExecutionLaneUnavailableError):
    """An ordered (Plan-named) arm could not be satisfied AS NAMED.

    Typed and terminal for the attempt: the hub named one exact arm, so there
    is no sibling to scan, no self-mint to fall back to and no eager degrade —
    any of those would substitute an artifact the spec did not name (pgw#904).
    ``reason`` is the countable token; the message names expected/have.
    """

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}")


def arm_ordered(
    pipe: Any,
    cfg: Any,
    cache_dir: Optional[Path],
    *,
    backend: str,
    artifact: Optional[Path],
    delivered_ref: str,
    delivered_digest: str,
    expected: Optional["aot_identity.ExpectedIdentity"],
    publisher_org: str,
) -> ArmOutcome:
    """Obey one Plan's ``Arm`` (pgw#904) — the fleet POLICY does not run.

    ``aot_cell`` arms exactly the pre-materialized ``artifact`` (its bytes
    already verified against the spec's content digest by the head): the
    hub-signed receipt is verified STRICTLY (a refused receipt is a typed
    refusal, never a drop-to-eager), the receipt's ``publisher_org_id`` must
    equal the org the spec named (§4.26 — the trust answer is WHO produced
    it), and pgw#903's declared-identity verification runs inside
    ``stage_artifact`` via ``expected``. ``dynamo`` arms JIT intake.
    ``eager_only`` arms nothing, by order.
    """
    if serve_posture.eager_only():
        # pgw#1142 / §4.32 item 4: the operator's order outranks the hub's
        # Plan, and this is the only place in the ORDERED path where that is
        # true. It is not a refusal — `OrderedArmError` would fail the attempt
        # typed, and the operator asked for eager service, not for the function
        # to go down — so the order is obeyed the way `eager_only` below is
        # obeyed: arm nothing, serve eager, say which of the two eager orders
        # it was.
        logger.info("arm-ordered: %s", serve_posture.block())
        return ArmOutcome(
            armed=False, eager_reason=EagerPhase.OPERATOR_EAGER_ONLY)

    bucket = int(getattr(cfg, "lora_bucket", 0) or 0)
    if bucket and not cc.has_compile_target(pipe, cfg):
        bucket = 0  # bare component slot (gw#627): branchless-eager, not an error

    if backend == "eager_only":
        if artifact is not None:
            raise OrderedArmError(
                "artifact_on_eager_arm",
                "the spec ordered eager_only but an artifact was materialized "
                "— refusing to arm anything the spec did not name")
        return ArmOutcome(armed=False, eager_reason=EagerPhase.HUB_ORDERED_EAGER)

    if backend == "dynamo":
        if artifact is not None:
            raise OrderedArmError(
                "artifact_on_dynamo_arm",
                "the spec ordered a dynamo (JIT intake) arm but an artifact "
                "was materialized — since pgw#1010 dynamo has no artifact")
        if bucket:
            cc.apply_lora_execution_lane(pipe, bucket)
        try:
            cc.arm_jit_intake(pipe, cfg)
        except Exception as exc:  # noqa: BLE001 — typed refusal, never silent
            if bucket:
                cc.drop_lora_execution_lane(pipe)
            raise OrderedArmError(
                "jit_arm_failed", f"ordered JIT intake arm failed: {exc}"
            ) from exc
        return ArmOutcome(armed=True)

    if backend != "aot_cell":
        raise OrderedArmError(
            "unknown_ordered_backend", f"no arm for backend {backend!r}")
    if artifact is None or expected is None:
        raise OrderedArmError(
            "artifact_unmaterialized",
            "an aot_cell arm reached arming with no materialized artifact "
            "or no expected identity — the head must supply both")

    # Deferred: receipts pulls +151 modules onto the `import gen_worker` path.
    from . import receipts

    family = str(getattr(cfg, "family", "") or "")
    if not receipts.configured():
        raise OrderedArmError(
            "receipt_gate_unconfigured",
            "an ordered cell arm requires the hub receipt gate; this process "
            "has no hub wiring to verify the publisher against")
    try:
        receipt = receipts.verify_delivered_artifact(Path(artifact), family)
    except receipts.ReceiptError as exc:
        raise OrderedArmError(
            "artifact_receipt_refused", f"{exc.reason}: {exc}") from exc
    want_org = str(publisher_org or "").strip()
    have_org = str(receipt.publisher_org_id or "").strip()
    # Exact publisher (§4.26): the spec NAMES the producing org and the signed
    # receipt must name the same one. Fail-closed on silence — an artifact
    # whose receipt cannot name its publisher cannot be shown to match.
    if not want_org or have_org != want_org:
        raise OrderedArmError(
            "publisher_mismatch",
            f"publisher_org: expected {want_org or '<unnamed>'}, receipt "
            f"names {have_org or '<unnamed>'}")

    if bucket:
        cc.apply_lora_execution_lane(pipe, bucket)
    started = time.monotonic()
    try:
        outcome = provision.arm_aot(
            pipe, cfg, cache_dir, Path(artifact), bucket, expected=expected)
    except BaseException:
        if bucket:
            cc.drop_lora_execution_lane(pipe)
        raise
    arm_ms = int(round(max(0.0, time.monotonic() - started) * 1000.0))
    row = CellAdoption(
        ref=delivered_ref,
        snapshot_digest=delivered_digest,
        artifact_kind=aot_serve.ARTIFACT_KIND,
        arm_ms=arm_ms,
        armed=outcome.armed,
        reason=outcome.reason,
        detail=outcome.detail or outcome.identity,
        pipeline_id=id(pipe),
    )
    if not outcome.armed:
        if bucket:
            cc.drop_lora_execution_lane(pipe)
        raise OrderedArmError(
            outcome.reason or "ordered_arm_refused",
            f"the named cell did not arm: {outcome.detail or outcome.identity}")
    return ArmOutcome(armed=True, adoptions=(row,))


def _arming_policy(
    pipe: Any,
    cfg: Any,
    cache_dir: Optional[Path],
    artifact: Optional[Path],
    *,
    publisher: Optional[CellPublisher],
    delegate: Optional[bool],
    delivered_ref: str,
    delivered_digest: str,
    adoptions: List[CellAdoption],
    boot_local_key: str = "",
) -> ArmOutcome:
    """Fleet arming policy (gw#587): delivered cell first, self-mint on miss.

    ``delivered_ref``/``delivered_digest`` name the HUB-attached candidate.
    They are the identity the hub fences an adoption on, they are known only to
    the caller that resolved the delivery, and without them a boot-attached
    adoption has nothing to report itself as (pgw#923). Every arm attempt is
    appended to ``adoptions``, measured.

    Replaces the executor's bare ``provision.enable_compiled`` call. HIT
    keeps today's semantics for a genuine match. A ``CellSelectionBugError``
    (th#1031: a self-requested, identity-verified cell that STILL refuses to
    arm — the seeded verdict rules on pre-trace facts, so two structurally
    different graphs can legitimately share one) no longer aborts arming: it
    is captured onto the returned :class:`ArmOutcome` for the caller to
    report loudly (unchanged wire event), and this call falls through to
    self-mint exactly like an ordinary miss — a live worker must recover
    into a working compiled state instead of retrying the identical
    unusable cell forever. MISS self-mints and serves compiled; the ONLY
    remaining eager/fail-closed exits are genuine mint impossibilities (no
    CUDA, no C toolchain, the mint itself failing), where plain lanes serve
    eager and quantized lanes keep their typed refusal — exactly the
    cozy-local store policy.

    Returns :class:`ArmOutcome`; ``self_mint`` carries the minted cell's
    identity so the caller can record/advertise it (serving-bootstrap half
    of th#910 — the hub's self-attested fence needs the worker to claim its
    own key as the active compile ref).
    """
    family = str(getattr(cfg, "family", "") or "")
    selection_bug: Optional[cc.CellSelectionBugError] = None
    delegate_refusal = ""
    if delegate is None:
        # pgw#784: whether a miss mints out of process is a POLICY of the
        # arming brain, not an argument its callers thread through. Keeping it
        # here means the executor's call is unchanged, every existing arming
        # double keeps working, and there is exactly one place the decision
        # lives. The parameter stays for tests that need to force either shape.
        from . import mint_delegate

        delegate_refusal = mint_delegate.delegation_refusal()
        delegate = not delegate_refusal
    elif not delegate:
        delegate_refusal = "caller_forced_in_process"

    # pgw#904: catalog discovery (`aot_cells.discover` fetch-and-filter) is
    # DELETED. A connected worker never lists, ranks or chooses a published
    # cell — the hub resolves the exact artifact and names it in
    # `Arm.artifact`, and `arm_ordered` is that path. What remains below is
    # the hub-less policy: the delivered artifact, then self-mint/intake.
    try:
        delivered_out, delivered_row = _arm_candidate(
            pipe, cfg, cache_dir, artifact,
            ref=delivered_ref,
            snapshot_digest=delivered_digest,
            artifact_kind="",
        )
        if artifact is not None:
            # ONE rule, the same one `_arm_candidate` opens its boot span on: a
            # call with no delivered artifact is not an adoption ATTEMPT —
            # `compile_cache.enable` also covers the seeded and ALLOW_COLD
            # lanes — so it must not manufacture a row for a cell that was
            # never offered, in either direction.
            adoptions.append(delivered_row)
        if delivered_out.armed:
            return ArmOutcome(armed=True)
        # Plain-lane miss: no cell delivered / artifact unusable. Fall
        # through to the self-mint instead of the pre-gw#587 silent eager.
    except cc.CellSelectionBugError as exc:
        # Self-requested, identity-verified cell refused to arm — always
        # reported loudly (the caller sends the wire event), but no longer
        # fatal: fall through and self-mint a cell this runtime can prove.
        logger.warning(
            "fleet-cells: cell_selection_bug (%s); self-minting instead of "
            "retrying the same unusable cell", exc)
        selection_bug = exc
    except cc.CompiledExecutionLaneUnavailableError:
        # Mandatory (w8a8/w4a4) miss: production used to fail closed here.
        # The whole point of self-mint is that this worker can produce the
        # cell itself.
        logger.info("fleet-cells: no delivered cell for mandatory lane; self-minting")

    if not family:
        return _fail_closed(
            pipe, "Compile decl has no family", selection_bug,
            phase=EagerPhase.NO_FAMILY)
    if not _cuda_ready():
        return _fail_closed(
            pipe, "CUDA unavailable", selection_bug, phase=EagerPhase.NO_CUDA)
    if not cc.toolchain_present():
        return _fail_closed(
            pipe, "no C compiler for the self-mint", selection_bug,
            phase=EagerPhase.NO_TOOLCHAIN)
    # A slot object with no resolvable compile target (the LTX upsampler
    # shape) has nothing to arm and nothing to mint.
    if not cc.has_compile_target(pipe, cfg):
        return _fail_closed(
            pipe, "no compile target resolves on this pipeline", selection_bug,
            phase=EagerPhase.NO_COMPILE_TARGET)
    # gw#561: the eager-miss rollback in provision.enable_compiled dropped
    # the branch lane; the mint must key + trace the DECLARED graph family.
    bucket = int(getattr(cfg, "lora_bucket", 0) or 0)
    if bucket:
        cc.apply_lora_execution_lane(pipe, bucket)

    if delegate:
        pipe_refusal = delegation_refusal(pipe, cfg)
        if pipe_refusal:
            logger.info(
                "fleet-cells: %s cannot mint out of process (%s) — no AOT "
                "mint is possible on this pipe (pgw#784)", family, pipe_refusal)
            delegate = False
            delegate_refusal = pipe_refusal

    # pgw#805/#1010: WHICH recipe this miss runs, decided ONCE, after
    # `delegate` is final and on the branch-bearing lane the export would
    # trace. `aot` is the only recipe that produces an artifact; every other
    # answer is JIT INTAKE, and every one of them is named on the wire by
    # `mint_recipe` itself.
    recipe = mint_recipe(
        pipe, cfg, delegate=delegate, delegate_refusal=delegate_refusal)
    if recipe != RECIPE_AOT:
        # pgw#1010: a MANDATORY (w8a8/w4a4) lane is compiled-from-a-CELL by
        # contract — the dispatch fence pins every request to an active
        # compile incarnation (th#910), and an intake arm has no identity to
        # pin. Arming it would compile a whole boot's worth of graphs for a
        # pod that then refuses every request `required_compile_missing`. So
        # it fails closed here, typed, exactly as a mandatory lane did before
        # self-mint existed: to serve a mandatory lane a family must declare
        # an export, because an AOT cell is the only cell there is.
        if cc.mandatory_serving(pipe):
            if bucket:
                cc.drop_lora_execution_lane(pipe)
            return _fail_closed(
                pipe,
                "this lane serves only from a cell and this family declares "
                "no export, so no cell can be minted for it (pgw#1010)",
                selection_bug, phase=EagerPhase.MANDATORY_LANE_NEEDS_A_CELL)
        # INTAKE. Arm the declared targets and let this pod's own warmup
        # compile them — nothing is captured, keyed, packed, published or
        # owed. There is no capture dir, so gw#608's process-global cache
        # move, pgw#777's multi-group refusal and the one-capture-per-process
        # conflict are all gone with it: an intake arm is just an arm.
        try:
            cc.arm_jit_intake(pipe, cfg)
        except Exception as exc:  # noqa: BLE001 — arm failure => miss policy
            logger.warning("fleet-cells: JIT intake arm failed (%s)", exc)
            if bucket:
                cc.drop_lora_execution_lane(pipe)
            return _fail_closed(
                pipe, f"jit intake arm failed: {exc}", selection_bug,
                phase=EagerPhase.JIT_ARM_FAILED)
        logger.info(
            "fleet-cells: JIT intake armed for %s (lane=%s) — this pod "
            "compiles its own graphs and serves them for its own life; no "
            "cell is minted, keyed or published (pgw#1010)",
            family, loading.pipeline_weight_lane(pipe) or "plain")
        return ArmOutcome(armed=True, selection_bug=selection_bug)

    # --- AOT, and only AOT, from here down --------------------------------
    # pgw#1059: an obligation opened before any trace has NO cell key (the
    # `graph` axis is the traced-graph digest). What it has is an
    # ArmIdentity: every pre-trace-knowable fact, computed with the same
    # derivations the child's recorded metadata uses — so the ledgers, the
    # events and the handback divergence check name the obligation without
    # ever pretending to name a cell.

    try:
        arm_key = arm_identity(
            family, loading.pipeline_weight_lane(pipe), bucket, cfg,
            subject=pipeline_arm_subject(pipe))
        key = arm_key.token
    except Exception as exc:  # noqa: BLE001 — obligation facts must be statable
        logger.warning("fleet-cells: self-mint identity computation failed (%s)", exc)
        if bucket:
            cc.drop_lora_execution_lane(pipe)
        return _fail_closed(
            pipe, f"self-mint identity computation failed: {exc}", selection_bug,
            phase=EagerPhase.KEY_COMPUTATION_FAILED)

    # pgw#672: consult the process ledgers BEFORE opening a capture.
    #
    # pgw#1033: TWO identities can be quarantined for one arm, and the gate
    # has to ask about both, because the ledger holds whatever ref actually
    # failed:
    #   * the ARM identity — an owed mint that died before it produced
    #     anything has no cell of its own, so the pending's computed ref is
    #     what `executor._abandon_pending_mint`'s quarantine writes;
    #   * the CELL identity — a serve/finalize proof fails on an ARMED
    #     artifact, whose ref is the exported cell's STAMPED key. That is the
    #     ref the runtime-guard and boot-proof writers record, and the arm can
    #     only reach it through `_FINALIZED`, this process's arm-key -> cell
    #     index. A mint is deterministic in its axes, so re-minting an arm
    #     whose own cell was just disproven rebuilds the same disproven cell.
    key_ref = f"{cc.system_repo(family)}#{key}"
    finalized_prior = finalized_in_process(key)
    quarantined_ref = ""
    if cc.cell_quarantined_in_process(key_ref):
        quarantined_ref = key_ref
    elif (finalized_prior is not None
            and cc.cell_quarantined_in_process(finalized_prior.ref)):
        quarantined_ref = finalized_prior.ref
    if quarantined_ref:
        # This exact identity already failed its serve/finalize proof in
        # this process — re-minting it is the churn loop, not recovery.
        # DEGRADE (explicit eager, mandatory lanes included): a broken
        # optimization must never kill a serving worker.
        logger.error(
            "fleet-cells: declining self-mint for %s key=%s (quarantined "
            "ref=%s) — this identity was quarantined by a failed proof in "
            "this process; serving eager (pgw#672)",
            family, key, quarantined_ref)
        if bucket:
            cc.drop_lora_execution_lane(pipe)
        # pgw#824: the ONE eager exit in this function that never rode a typed
        # event — it returns before `_fail_closed` and only logged. A pod that
        # quarantined its own cell serves eager for the rest of its life, which
        # is precisely the state the hub most needs named.
        activity_mod.emit_event(
            "self_mint_skipped",
            f"family={family} key={key} quarantined_ref={quarantined_ref} "
            f"lane={loading.pipeline_weight_lane(pipe) or 'plain'}: this "
            f"identity was quarantined by a failed serve/finalize proof "
            f"earlier in this process; re-minting it is the churn loop, so "
            f"this worker serves eager for the rest of its life and "
            f"publishes nothing",
            phase=EagerPhase.CELL_QUARANTINED,
        )
        return ArmOutcome(
            armed=False, selection_bug=selection_bug,
            eager_reason=EagerPhase.CELL_QUARANTINED)
    if finalized_prior is not None:
        # This process already minted and ADOPTED this exact cell — re-arm
        # the same artifact through the same AOT gates instead of paying a
        # second export for bytes that are already on this disk.
        #
        # pgw#1113: through `_arm_exported_cell`, which is THE gate every cell
        # this machine produced for itself passes, and not the bare
        # `provision.arm_aot` that stood here. This was the one branch that
        # armed a pipe from another pipe's artifact while skipping
        # `arm_axis_divergence` entirely — and since the arm token is what
        # decides "this exact cell", a token that could not name its subject
        # made "another pipe" and "this pipe" the same sentence. The token now
        # names the subject, so the branch is reached far less often; when it
        # is reached it now earns the arm the same way every other
        # self-produced cell does.
        try:
            armed_ready, _meta, refusal = _arm_exported_cell(
                pipe, cfg, cache_dir, bucket,
                Path(finalized_prior.artifact), arm_key)
            if not armed_ready:
                logger.warning(
                    "fleet-cells: the in-process finalized cell for key=%s "
                    "did not re-arm (%s%s); falling through to a fresh mint",
                    key, refusal[0], f": {refusal[1]}" if refusal[1] else "")
        except Exception as exc:  # noqa: BLE001 — fall back to a fresh mint
            logger.warning(
                "fleet-cells: re-arm from the in-process finalized cell "
                "failed (%s); falling through to a fresh mint", exc)
            armed_ready = False
        if armed_ready:
            logger.info(
                "fleet-cells: re-armed %s from this process's finalized "
                "cell (key=%s) — no second mint (pgw#672)", family, key)
            return ArmOutcome(
                armed=True, self_mint=finalized_prior,
                selection_bug=selection_bug)

    # §4.28 / pgw#1096: THIS MACHINE's own store, before any mint is opened.
    # Sited after the in-process ledgers (a cell already resident in this
    # process costs no disk read) and after the quarantine gate (an identity
    # this process disproved must not be re-armed from any source), and
    # BEFORE the pending: compile-once-run-forever means the second boot never
    # reaches the code below. Costs one memo read + one digest on a hit and a
    # single missing-file stat on a miss, so the trusted-fleet path — which
    # never stores anything locally — pays a stat and nothing else.
    local_prior = arm_from_local_store(
        pipe, cfg, cache_dir, bucket, arm_key, family,
        boot_local_key=boot_local_key)
    if local_prior is not None:
        return ArmOutcome(
            armed=True, self_mint=local_prior, selection_bug=selection_bug)

    # Pipes whose obligation identity is the SAME token share one child mint:
    # same runtime, same declaration AND same subject — the same slot resolved
    # to the same checkpoint — so there is one computation to buy and one
    # pending to open. pgw#1113 deleted the premise this comment used to
    # state ("the qwen edit shape: two lanes, one family cell"): two lanes are
    # one cell only when the graph says so, and the graph does not exist yet
    # here. The obligation names its subject instead of assuming one.
    with _PENDING_LOCK:
        existing = _PENDING.get(key)
    if existing is not None:
        mint_root = existing.mint_root
        target = existing.target
    else:
        mint_root = Path(tempfile.mkdtemp(prefix="selfmint-"))
        label = cc.flavor_label(
            cc.runtime_key()["sku"], cc.runtime_key()["torch"],
            loading.pipeline_weight_lane(pipe))
        target = mint_root / f"{label}.tar.gz"

    # pgw#784: NOTHING is armed on the live pipeline. It keeps serving plain
    # eager — no guarded wrappers, no branch containers, no inductor state.
    # ``armed=False`` is the honest answer: this pipe serves eager right now.
    # The mint obligation rides the returned pending.
    pending = PendingSelfMint(
        family=family, arm_token=key,
        ref=f"{cc.system_repo(family)}#{key}",
        cfg=cfg, target=target,
        mint_root=mint_root, publisher=publisher, cache_dir=cache_dir,
        arm_key=arm_key,
    )
    with _PENDING_LOCK:
        _PENDING.setdefault(key, pending)
    logger.info(
        "fleet-cells: DELEGATED %s self-mint for %s (key=%s) — a child "
        "process builds the cell while this process serves eager "
        "(pgw#784/#805)", recipe, family, key)
    activity_mod.emit_event(
        "self_mint_started",
        # pgw#1042: labeled `arm_key` — the cell the child returns carries a
        # STAMPED key in a disjoint space (kind/contract differ by formula),
        # so an unlabeled `key=` here invited reading the two as one
        # diverging key.
        f"family={family} recipe={recipe} arm_key={key} "
        f"lane={loading.pipeline_weight_lane(pipe) or 'plain'}: a "
        f"compile-cell miss opened a delegated mint; this worker serves "
        f"eager throughout",
        phase=recipe,
    )
    return ArmOutcome(
        armed=False, self_mint=pending, selection_bug=selection_bug,
        # pgw#824: eager RIGHT NOW, but for a reason with an end — the
        # child is building the cell. A request row carrying
        # `mint_in_progress` is a fleet that is warming up; one carrying
        # `no_toolchain` is a fleet that never will. Reading them as the
        # same "" was the whole gap.
        eager_reason=EagerPhase.MINT_IN_PROGRESS)


def _packed_metadata(artifact: Path) -> Dict[str, Any]:
    """The stamped metadata inside a packed cell (metadata members only)."""
    return artifact_meta.read_metadata(artifact)


def arm_axis_divergence(
    arm_key: ArmIdentity, meta: Mapping[str, Any],
) -> str:
    """'' when the child's cell-metadata states the parent's obligation
    identity on every ENVIRONMENT fact (:data:`ARM_ENVIRONMENT_FACTS`), else
    the FIRST diverging fact with both values (pgw#1042).

    A delegated child re-derives every environment-shaped fact in its own
    process, so a fact that fails to survive the boundary (the measured
    case: torch's `aot_compile` mutating global inductor config between the
    child's establish and its metadata assembly) previously surfaced only as
    a downstream numerics or constants error on a $3 pod. Here it is refused
    BY NAME at the handback seam. ``graph`` is deliberately NOT compared —
    it exists only post-trace, and comparing a declared-facts stand-in
    against the traced fact was the attempt-28 phantom divergence
    (pgw#1059). Every compared fact uses the SAME derivation on both sides.

    :data:`ARM_SUBJECT_FACTS` are equally deliberately NOT compared
    (pgw#1113). A cell records no subject and must not: the key is the traced
    computation, so one cell legally serves every checkpoint whose graph it
    is, and demanding the cell restate the checkpoint it was minted from
    would refuse exactly the reuse the membership axiom exists to allow. The
    subject splits the obligation on THIS side of the boundary; what crosses
    it is compared here.
    """
    envelope_block = meta.get(cell_key.EXPORT_ENVELOPE_KEY)
    child: Dict[str, str] = {
        "family": str(meta.get("family") or ""),
        "format": str(meta.get("format") or ""),
        "lane": cc.execution_lane_label(
            str(meta.get("weight_lane") or ""),
            int(meta.get("lora_bucket") or 0)),
        "sm": str(meta.get("sm") or ""),
        "envelope": (
            cell_key.envelope_digest(envelope_block)
            if isinstance(envelope_block, dict) and envelope_block else ""),
        "env_seal": env_seal.seal_digest(
            dict(meta.get(env_seal.SEAL_KEY) or {})),
        "toolchain": cell_key.facts_digest(dict(meta.get("toolchain") or {})),
    }
    parent = arm_key.facts_dict()
    for fact in ARM_ENVIRONMENT_FACTS:
        if parent.get(fact, "") != child.get(fact, ""):
            return (f"{fact}: child cell states {child.get(fact, '')!r}, "
                    f"this runtime computed {parent.get(fact, '')!r}")
    return ""


#: pgw#1096: WHY a machine keeps the cell it just minted. Each is a fact about
#: the SINK, and none of them is the worker deciding its own trust class —
#: which stays what §4.28 makes it, the hub's call and only the hub's.
KEEP_HUB_ASSERTED_UNTRUSTED = "hub_asserted_untrusted"
KEEP_NO_PUBLISHER = "no_publish_sink"
KEEP_PUBLISH_DISARMED = "publish_disarmed"


def local_keep_reason(publisher: Optional[CellPublisher]) -> str:
    """Why this machine keeps its own cell, "" when it has no reason to.

    §4.28 says an untrusted machine mints for ITSELF. Three disjoint ways a
    machine learns it has nowhere to ship — and NONE of them is a worker-side
    trust self-declaration, which the ruling forbids:

    * :data:`KEEP_HUB_ASSERTED_UNTRUSTED` — the HUB refused a publish from this
      hardware (`cell_publish_untrusted_tier`; `cloudtier.PublishRefusal` on a
      community/marketplace/unknown tier), recorded by `local_cell_store`. The
      community-cloud case, and the only one that is about trust at all.
    * :data:`KEEP_NO_PUBLISHER` — this process constructed no publisher. That
      is **cozy-local**, which never has one (`fleet_cells` docstring) and never
      reaches a hub to be refused BY — so a design that waited for a hub verdict
      would leave the cozy-local half of §4.28 permanently unimplemented, which
      is the whole product promise. The fleet's `SELF_MINT_WITHOUT_PUBLISH_SINK`
      wiring alarm is unchanged and still fires: keeping the bytes does not make
      a mis-wired fleet pod quiet, it just stops it burning the compute twice.
    * :data:`KEEP_PUBLISH_DISARMED` — a pgw#980 probe, whose publish the control
      PARENT removes from its hub-call allowlist. Read from the predicate that
      owns that decision rather than sniffed off an exception.

    Keeping bytes you already paid an hour of compute for, and cannot ship, is
    not a claim about trust. It is the absence of a reason to delete them.
    """
    if local_cell_store.keeps_cells_locally():
        return KEEP_HUB_ASSERTED_UNTRUSTED
    if publisher is None or not publisher.enabled():
        return KEEP_NO_PUBLISHER
    from .procsplit import actions

    if actions.publish_disarmed():
        return KEEP_PUBLISH_DISARMED
    return ""


def _arm_exported_cell(
    pipe: Any, cfg: Any, cache_dir: Optional[Path], bucket: int,
    artifact: Path, arm_key: Optional[ArmIdentity],
    *, verify_numerics: bool = False,
) -> Tuple[bool, Optional[Dict[str, Any]], Tuple[str, str]]:
    """THE gate every cell this machine produced for itself must pass.

    ``verify_numerics`` (§4.32) is set by ONE caller —
    :func:`adopt_delegated_mint`, the pod that just minted these bytes and is
    about to publish them. The other three routes here are ADOPTIONS of bytes
    already proven at their own mint (an in-process finalized cell, the local
    store's, a fresh child mint being re-armed), and adoption runs no quality
    gate.

    pgw#1096 extracted this from :func:`adopt_delegated_mint` so the two
    self-produced sources — a child process's fresh mint, and this machine's
    OWN local store (§4.28) — pass through ONE gate rather than two that agree
    today. A cell out of the local store is neither newer nor more trusted
    than the child's: it is the same bytes the same machine minted, and it
    earns its arm the same way.

    Two checks, in this order:

    1. **key-axis divergence** (pgw#1042) — the cell's recorded metadata must
       state THIS runtime on every pre-trace environment axis
       (:data:`ARM_ENVIRONMENT_FACTS`), refused
       by fact name. This is what makes a toolchain/sm/envelope move an honest
       refusal instead of a wrong arm;
    2. **the AOT arm** (pgw#805) — ``provision.arm_aot``: lifted-binding
       install, ``aot_serve.enable``, rollback on failure. Deliberately NOT
       ``provision.enable_compiled``, whose pgw#709 receipts gate drops any
       artifact the hub has not countersigned — which by construction is every
       cell in this family.

    Returns ``(armed, meta, (reason, detail))``. ``reason`` is initialized to a
    NAMED unset rather than "" so a branch that forgets to classify shows up as
    a gap in this function instead of as an empty string that reads like "no
    reason exists" (pgw#999).
    """
    refusal: Tuple[str, str] = ("unclassified_arm_refusal", "")
    # pgw#1098: UNREADABLE IS NOT ABSENT, and this is check 0.
    #
    # `try_read_metadata` answers None for both "this cell has no envelope" and
    # "I refused to read the envelope it has", and every consumer here spent
    # that distinction on `meta is not None`. Measured on row 7: a 16 MiB bound
    # refused a 36-entry sdxl envelope, so check 1 below SILENTLY DID NOT RUN,
    # `arm_aot` was handed None and skipped the lifted-binding install, and the
    # refusal that reached the wire named a downstream contract gate
    # (`lifted_inputs_unbindable`) with no root. 36/36 entries, 92 minutes and
    # $1.584 discarded; the only trace of the cause was the word `unreadable`
    # in one event's `cell_key=` field.
    #
    # An envelope this runtime cannot READ is refused here, by name, before any
    # arm — the same class as a cell that does not describe us. It belongs in
    # THIS function rather than at either call site, because pgw#1096's whole
    # point is that the child's cell and the local store's earn their arm the
    # same way, and a store whose envelope cannot be read is exactly as
    # unarmable as a child's.
    try:
        meta: Optional[Dict[str, Any]] = artifact_meta.read_metadata(artifact)
    except artifact_meta.ArtifactMetadataError as exc:
        return False, None, ("cell_envelope_unreadable", (
            f"the cell's {artifact_meta.METADATA_NAME} could not be read, so "
            f"no gate that reads it could run: {exc}"))
    divergence = ""
    if meta is not None and arm_key is not None:
        divergence = arm_axis_divergence(arm_key, meta)
    if divergence:
        stamped = str(meta.get("cell_key") or "") if meta else "MISSING"
        return False, meta, ("key_axis_divergence", (
            f"the cell (stamped key {stamped}) does not describe this "
            f"runtime: {divergence}"))
    try:
        outcome = provision.arm_aot(
            pipe, cfg, cache_dir, artifact, bucket, meta,
            verify_numerics=verify_numerics)
        if outcome:
            return True, meta, ("", "")
        refusal = (outcome.reason or "unclassified_arm_refusal",
                   outcome.detail or outcome.identity)
    except cc.CellSelectionBugError as exc:
        # th#883: a cell whose axes describe exactly this runtime refused to
        # arm. Loud — a bug in the one selection brain, not a compat miss.
        logger.error(
            "fleet-cells: cell_selection_bug arming a self-produced cell "
            "(%s): %s", artifact, exc)
        refusal = ("cell_selection_bug", str(exc))
    except Exception as exc:  # noqa: BLE001 — adoption failure => eager
        logger.warning(
            "fleet-cells: self-produced cell %s did not adopt (%s)",
            artifact, exc)
        # An `AdoptError` already carries the token; anything else is named by
        # its type rather than flattened into one word nobody can count.
        refusal = (str(getattr(exc, "reason", "") or "") or type(exc).__name__,
                   str(exc))
    return False, meta, refusal


#: One sweep per process: the store is this machine's, the predecessor
#: entries are finite, and re-listing a directory on every arm buys nothing.
_MEMO_SWEPT = False


def _sweep_superseded_memos_once() -> None:
    """Discard memo rows written under a superseded arm-token schema.

    pgw#1113 states the cost up front: the subject facts change every token,
    so every machine with a local store pays ONE re-mint per family, once.
    That cost is the point — an entry keyed by a token that could not state
    what it was compiling is exactly the row that could hand this pipeline
    another checkpoint's cell — but it must be spent EXPLICITLY, in a counted
    line, not discovered later as a store full of unreadable files.
    """
    global _MEMO_SWEPT
    if _MEMO_SWEPT:
        return
    _MEMO_SWEPT = True
    try:
        local_cell_store.sweep_superseded_memos(ARM_SCHEME)
    except Exception:  # noqa: BLE001 — a cache sweep is never fatal
        logger.debug("fleet-cells: memo sweep failed", exc_info=True)


#: pgw#1127: how this machine ADDRESSED the cell it armed. Two routes into one
#: CAS, and the difference is what a refusal is allowed to do about it.
ROUTE_MEMO = "memo"
ROUTE_BOOT_KEY = "boot_key"


def arm_from_local_store(
    pipe: Any, cfg: Any, cache_dir: Optional[Path], bucket: int,
    arm_key: ArmIdentity, family: str, boot_local_key: str = "",
) -> Optional[SelfMint]:
    """§4.28 step 3: arm from THIS MACHINE's own store, before minting anything.

    Paul's compile-once-run-forever, at the one place a miss is decided. No
    network is touched on this path at all: a box with no hub reachable arms
    exactly as well as one online, which is what "fully offline-capable" means.

    TWO ROUTES, ONE CAS, ONE GATE (pgw#1127 S2). Both end at
    :func:`_arm_exported_cell`, the gate a child's fresh mint also passes, so a
    stored cell is never more trusted than a freshly minted one and never less.

    * :data:`ROUTE_MEMO` — the pre-trace ``ArmIdentity`` (milliseconds), through
      the memo this machine's own mint wrote for that exact token. The fast
      path, and the only one before pgw#1127.
    * :data:`ROUTE_BOOT_KEY` — the ``ck1`` key §4.27's boot derivation produced
      and ``local_cell_store`` answered on. This is what makes an arm-token
      SCHEME BUMP cost a trace instead of a mint: `sweep_superseded_memos`
      deletes the shortcut and leaves the CELLS under their own keys, so after
      the sweep the memo misses and the derived key still addresses the cell it
      used to name. On a fleet pod both routes are one stat on an empty store.

    A refusal on the memo route DROPS the entry — this machine wrote that memo
    for this exact identity, so a cell that will not arm under it is stale.
    A refusal on the boot-key route does NOT: that hit is an inference about
    which pipe owns the bytes, and destroying another pipe's cell to punish a
    wrong guess would turn one honest re-mint into two.

    ``None`` = no usable local cell; the caller mints, honestly.
    """
    _sweep_superseded_memos_once()
    route = ROUTE_MEMO
    try:
        local = local_cell_store.lookup_for_arm(arm_key.token)
        if local is None and boot_local_key:
            # The memo is a SHORTCUT, never an authority (§4.28) — so when it
            # has nothing to say, the address the boot derived is asked
            # directly. Same store, same key space, same gate below.
            route = ROUTE_BOOT_KEY
            local = local_cell_store.lookup(boot_local_key)
    except Exception as exc:  # noqa: BLE001 — a cache read must never be fatal
        logger.warning("fleet-cells: local cell store unreadable (%s)", exc)
        return None
    if local is None:
        return None
    armed, meta, (reason, detail) = _arm_exported_cell(
        pipe, cfg, cache_dir, bucket, local.artifact, arm_key)
    if not armed:
        dropped = route == ROUTE_MEMO
        logger.warning(
            "fleet-cells: the local store's cell for %s (key=%s, route=%s) did "
            "not arm (%s%s); %s and minting", family, local.key, route, reason,
            f": {detail}" if detail else "",
            "dropping it" if dropped else "leaving it in place")
        if dropped:
            local_cell_store.drop(local.key)
        activity_mod.emit_event(
            "local_cell_refused",
            f"family={family} arm_key={arm_key.token} cell_key={local.key} "
            f"route={route}: this machine's own stored cell did not arm "
            f"({reason}{': ' + detail if detail else ''}); it has been "
            + ("dropped from the local store" if dropped else
               "LEFT in the local store — a boot-key hit is an inference "
               "about which pipe owns the bytes, not a memo this machine "
               "wrote for this arm")
            + " and this boot mints a fresh one",
            phase=reason,
        )
        return None
    key = str((meta or {}).get("cell_key") or "").strip() or local.key
    if route == ROUTE_BOOT_KEY:
        # The shortcut, REPAIRED. The cell just proved it arms under this arm
        # token, so the memo the sweep deleted (or that a re-keyed graph left
        # pointing elsewhere) can be rewritten from evidence instead of from a
        # mint. This is the whole cost argument for the derived-key route: with
        # it an arm-scheme bump costs one TRACE per family per machine; without
        # it, one MINT.
        local_cell_store.note_memo(arm_key.token, local.key)
    aot_serve.note_aot_key(key)
    minted = SelfMint(
        family=family, cell_key=key,
        ref=f"{cc.system_repo(family)}#{key}",
        snapshot_digest=local.content_digest,
        artifact=local.artifact,
    )
    with _PENDING_LOCK:
        # Same in-process index a delegated adopt fills: a sibling pipe of the
        # same record must re-arm these bytes rather than pay a second lookup.
        _FINALIZED[arm_key.token] = minted
    logger.info(
        "fleet-cells: armed %s from THIS MACHINE's local cell store "
        "(key=%s, %.1f MB, route=%s) — no mint, no hub, no network (§4.28)",
        family, key, local.bytes / 1e6, route)
    activity_mod.emit_event(
        "local_cell_armed",
        f"family={family} arm_key={arm_key.token} cell_key={key} "
        f"route={route}: this machine minted this cell on an earlier boot and "
        f"stored it locally; it arms from disk with no mint and no publish "
        f"route"
        + (" — addressed by the key THIS BOOT derived, so the arm-token memo "
           "was not needed and has been rewritten from the proven arm"
           if route == ROUTE_BOOT_KEY else ""),
        phase="local_store_hit",
    )
    return minted


def adopt_delegated_mint(
    pipe: Any, pending: "PendingSelfMint", artifact: Path,
) -> Optional[SelfMint]:
    """pgw#784: adopt a cell a CHILD PROCESS just built, then publish it.

    There is nothing to pack here — the child already packed — so this is
    exactly the DELIVERED-cell adoption the cache-HIT path runs. A child-built
    cell EARNS adoption through the same gates a hub-delivered cell does, and
    when it cannot, the honest outcome is the one every miss already has:
    unwrap, serve eager, leave the cell absent, publish nothing. A parity gap
    degrades; it never poisons the store.

    ``None`` = not adoptable. The caller treats that exactly like a disproven
    candidate (the mint failed, the worker keeps serving).
    """
    state = pending._state
    if "minted" in state:
        return state["minted"]

    artifact = Path(artifact)
    if artifact != pending.target:
        try:
            pending.target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(artifact, pending.target)
        except OSError:
            shutil.copy2(artifact, pending.target)
    # pgw#1096: ONE gate for every self-produced cell — the child's, and the
    # local store's (§4.28). pgw#999's classification, pgw#1042's pre-arm axis
    # divergence and pgw#805's AOT-only arm all live in `_arm_exported_cell`;
    # the reasons this call site used to compute inline are unchanged, and the
    # $2.72 lesson (attempt 26: a 36/36 mint refused by three events that all
    # said "could not adopt") is kept there rather than repeated here.
    # §4.32 THE MINT-TIME GATE, and this is the only site that arms it: this
    # process compiled these bytes and is about to publish them to every pod
    # that will ever adopt this key. It runs the freshly compiled artifact and
    # the eager forward it was traced from on the same feed, and it is STRICT —
    # identical or refuse, no gray band — because an adopter runs no gate that
    # could re-check what ships. A refusal below unwraps, serves eager, emits
    # `self_mint_abort` to the hub and publishes nothing.
    armed, meta, refusal = _arm_exported_cell(
        pipe, pending.cfg, pending.cache_dir,
        int(getattr(pending.cfg, "lora_bucket", 0) or 0),
        pending.target, pending.arm_key, verify_numerics=True)
    if not armed:
        reason, detail = refusal
        # pgw#999: `phase` is the countable column, so it carries the CLASS —
        # the same convention `self_mint_skipped` already uses. The old
        # constant `delegated_adopt_failed` said only which call site fired,
        # which every reader already knew from the event kind.
        state["adopt_refusal"] = (reason, detail)
        # pgw#1042: BOTH identities, labeled. The parent's computed ARM key
        # and the child's stamped cell key live in disjoint spaces by
        # formula (pgw#1032/#1033); one unlabeled `key=` carrying the arm
        # key while the detail quoted the stamped one is what the pod lane
        # read as "the child computed a different key".
        stamped = str((meta or {}).get("cell_key") or "") or "unreadable"
        activity_mod.emit_event(
            "self_mint_abort",
            f"family={pending.family} arm_key={pending.arm_token} "
            f"cell_key={stamped}: the child process produced a cell this "
            f"runtime could not adopt "
            f"({reason}{': ' + detail if detail else ''}); serving stays "
            f"eager and nothing is published",
            phase=reason,
        )
        mark_terminus(pending, TERMINUS_ABORTED)
        state["minted"] = None
        _unregister(pending)
        shutil.rmtree(pending.mint_root, ignore_errors=True)
        return None

    meta = dict(meta) if meta is not None else _packed_metadata(pending.target)
    key = str(meta.get("cell_key") or "").strip()
    if not key:
        # pgw#1059: a produced cell without a stamped key has no identity to
        # advertise, publish or ledger — and the arm token is NOT a cell key,
        # so the old fallback to it would have advertised an arm1- ref the
        # hub can never corroborate. Refuse, typed, like any other
        # unadoptable candidate.
        state["adopt_refusal"] = (
            "cell_key_missing",
            "the child's cell carries no stamped cell_key")
        activity_mod.emit_event(
            "self_mint_abort",
            f"family={pending.family} arm_key={pending.arm_token}: the "
            f"child's cell carries no stamped cell_key; serving stays eager "
            f"and nothing is published",
            phase="cell_key_missing",
        )
        mark_terminus(pending, TERMINUS_ABORTED)
        state["minted"] = None
        _unregister(pending)
        shutil.rmtree(pending.mint_root, ignore_errors=True)
        return None
    minted = SelfMint(
        family=pending.family, cell_key=key,
        ref=f"{cc.system_repo(pending.family)}#{key}",
        snapshot_digest="sha256:" + sha256_file(pending.target),
        artifact=pending.target,
    )
    state["minted"] = minted
    state["meta"] = dict(meta)
    # pgw#1033: this process has now READ a stamped `aot-inductor` key off a
    # packed envelope, which is the one event that teaches a runtime that a
    # key-flavored ref names an EXPORTED cell (the delivered-arm path
    # registers its keys the same way; a SELF-MINTED cell was the
    # unregistered half, so the executor's #734/#735 kind dispatch scored this
    # pod's own `.pt2` by FX cache hits it can never produce).
    aot_serve.note_aot_key(minted.cell_key)
    # th#1355: the mint cost, banked at the moment the cell becomes real.
    state["mint_duration_ms"] = max(
        0, int((time.monotonic() - pending.armed_at) * 1000))
    _unregister(pending)
    with _PENDING_LOCK:
        # pgw#1033: under the ARM token (see `_FINALIZED`). The stamped key is
        # the VALUE's identity; keying the ledger by it made the memo
        # unreadable by the only lookup there is, so every same-key re-arm in
        # this process paid a second full export.
        _FINALIZED[pending.arm_token] = minted
    keep = local_keep_reason(pending.publisher)
    if keep:
        # §4.28: a machine with nowhere to ship keeps its own cell, where its
        # next boot can find it. Done HERE, not after the publish attempt,
        # because a pod that dies between adopting and being refused would
        # otherwise lose the whole mint — the th#1643 SUNK case, one boot later.
        stored = local_cell_store.store(
            pending.target, key=minted.cell_key, family=pending.family,
            arm_token=pending.arm_token)
        if stored is not None:
            activity_mod.emit_event(
                "local_cell_stored",
                f"family={pending.family} arm_key={pending.arm_token} "
                f"cell_key={minted.cell_key}: this machine cannot ship this "
                f"cell ({keep}), so it keeps it — every later boot of this "
                f"machine arms it from disk with no mint (§4.28)",
                phase=keep,
            )
    logger.info(
        "fleet-cells: DELEGATED mint adopted for %s (key=%s, %.1f MB) — the "
        "worker served eager throughout and now serves compiled",
        pending.family, key, pending.target.stat().st_size / 1e6)
    return minted


def adopt_refusal(pending: "PendingSelfMint") -> Tuple[str, str]:
    """Why :func:`adopt_delegated_mint` refused this pending's cell (pgw#999).

    ``("", "")`` when it did not refuse — the mint adopted, or never got as
    far as arming. The classification lives on the pending's own state rather
    than being re-derived by the caller, so the abort event, the delegated
    result and the executor's decline all quote ONE string that was produced
    at the one place that knows it.
    """
    reason, detail = pending._state.get("adopt_refusal") or ("", "")
    return str(reason), str(detail)


def publish_self_mint(pending: "PendingSelfMint") -> None:
    """Ship a FINALIZED mint to the fleet store, once (gw#612 restructure).

    The executor calls this AFTER the whole warmup-proof pass, when sibling
    coverage of the shared capture is known: a family cell is only published
    when every sharer's graphs are inside it. Serve first, publish behind
    (gw#587: publish failure never blocks the request that triggered the
    miss); the hub's attested gate decides accept/refuse."""
    state = pending._state
    if state.get("publish_resolved"):
        return
    if state.get("minted") is None:
        # pgw#815: this WAS a bare `return`. A caller that believed it had a
        # finalized cell (the executor's publish gate runs only for pendings
        # it packed) and a caller that has nothing produced the identical
        # observation — nothing. Name it.
        state["publish_resolved"] = True
        activity_mod.emit_event(
            "self_mint_publish_withheld",
            f"family={pending.family} key={pending.arm_token}: the publish "
            "gate ran with no finalized cell on this pending — nothing was "
            "packed, so nothing can ship",
            phase="nothing_to_publish",
        )
        mark_terminus(pending, TERMINUS_WITHHELD)
        return
    state["publish_resolved"] = True
    publisher = pending.publisher
    if publisher is not None and publisher.enabled():
        mark_terminus(pending, TERMINUS_PUBLISHING)
        _publish_async(
            publisher, pending.family, pending.target,
            dict(state.get("meta") or {}),
            cell_key_digest=str(
                getattr(state.get("minted"), "cell_key", "")
                or pending.arm_token),
            mint_duration_ms=int(state.get("mint_duration_ms") or 0),
            # pgw#1096: the PRE-TRACE identity, carried so a hub refusal that
            # teaches this machine it is untrusted can also write the memo
            # that lets its next boot find the salvaged cell without tracing.
            arm_token=pending.arm_token)
    else:
        # Runtime assertion (gw#587): every fleet cell miss must produce a
        # publish attempt. A fleet worker minting with no usable sink is a
        # wiring defect (file_base_url/worker JWT absent at arming time),
        # not a policy choice — loud, greppable, alarm-adjacent. cozy-local
        # DOES enter this module since pgw#1127, and it must never enter
        # HERE: it ends its obligation at :func:`keep_self_mint_local`, whose
        # whole point is that a machine with no sink by design is not a
        # machine that lost its sink by accident.
        logger.warning(
            "fleet-cells: SELF_MINT_WITHOUT_PUBLISH_SINK family=%s — cell "
            "stays local to this pod; the fleet store gains nothing",
            pending.family)
        activity_mod.emit_event(
            "self_mint_publish_withheld",
            f"family={pending.family} key={pending.arm_token}: no publish "
            "sink (file_base_url/worker JWT absent at arming time); the cell "
            "was kept in this machine's own store (§4.28/pgw#1096 — the "
            "sentence 'stays local to this pod' is now TRUE; before it was "
            "printed on the way to an rmtree) and this machine reuses it, but "
            "the fleet store still gains nothing",
            phase="no_sink",
        )
        mark_terminus(pending, TERMINUS_WITHHELD)
        shutil.rmtree(pending.mint_root, ignore_errors=True)


def keep_self_mint_local(pending: "PendingSelfMint") -> None:
    """§4.28 terminus for a machine that has NO sink by design (cozy-local).

    The cell is already in this machine's own store — ``adopt_delegated_mint``
    put it there under its stamped ``ck1`` key before any publish could be
    attempted — so all that is left is to END the obligation (pgw#815: a
    pending that reaches the end of a boot carrying no terminus VANISHED) and
    drop the capture directory.

    Deliberately NOT :func:`publish_self_mint`. That function's sinkless branch
    is a WIRING ALARM — a fleet pod whose ``file_base_url``/JWT went missing —
    and firing it on cozy-local would make the one machine §4.28 was written
    about permanently indistinguishable from a broken pod. It also takes a
    publisher, and this one cannot: the local serve entry's never-publish
    property is pinned structurally by
    ``tests/test_local_serve_no_publisher_pgw1127.py``, and a terminus that
    accepted a sink would be the hole that fence exists to close.
    """
    state = pending._state
    if state.get("publish_resolved"):
        return
    state["publish_resolved"] = True
    if state.get("minted") is None:
        activity_mod.emit_event(
            "self_mint_publish_withheld",
            f"family={pending.family} key={pending.arm_token}: this machine "
            f"has no publish sink and nothing was packed for this pending, so "
            f"there is no cell to keep either",
            phase="nothing_to_publish",
        )
        mark_terminus(pending, TERMINUS_ABANDONED)
        shutil.rmtree(pending.mint_root, ignore_errors=True)
        return
    logger.info(
        "fleet-cells: %s stays on THIS MACHINE (key=%s) — it has no publish "
        "sink by design, and its own store is where every later run finds it "
        "(§4.28)", pending.family, pending.arm_token)
    activity_mod.emit_event(
        "self_mint_publish_withheld",
        f"family={pending.family} key={pending.arm_token}: this machine mints "
        f"for ITSELF (§4.28) — the cell is in its own store, addressed by the "
        f"same ck1 key the hub store would use, and no publish was attempted "
        f"because no sink exists to attempt one against",
        phase=KEEP_NO_PUBLISHER,
    )
    mark_terminus(pending, TERMINUS_WITHHELD)
    shutil.rmtree(pending.mint_root, ignore_errors=True)


def withhold_self_mint_publish(pending: "PendingSelfMint", reason: str) -> None:
    """Refuse to publish a finalized-but-INCOMPLETE family cell (gw#612).

    A shared capture packs only the graphs the warmup actually compiled: a
    mandatory sibling lane the warmup never exercised contributes nothing,
    and publishing that partial pack as the family cell bricks every
    adopting boot at the gw#607 per-object proof (the gw#611 qwen shape:
    hits=1/misses=1 -> compile_cell_failed -> release broken). The mint
    keeps serving THIS process (compiled callables live, identity
    advertised self-attested); only the store publish is withheld, so the
    next boot re-mints instead of adopting a poisoned cell."""
    state = pending._state
    if state.get("publish_resolved"):
        return
    if state.get("minted") is None:
        # pgw#815: the twin of `publish_self_mint`'s bare return. A withhold
        # asked of a pending that packed nothing is not a no-op — it means the
        # mint produced no cell at all, and that end must be named too.
        state["publish_resolved"] = True
        activity_mod.emit_event(
            "self_mint_publish_withheld",
            f"family={pending.family} key={pending.arm_token}: {reason} — and "
            "nothing was packed for this pending, so there was no cell to "
            "withhold either",
            phase="nothing_to_publish",
        )
        mark_terminus(pending, TERMINUS_ABANDONED)
        shutil.rmtree(pending.mint_root, ignore_errors=True)
        return
    state["publish_resolved"] = True
    logger.error(
        "fleet-cells: SELF_MINT_PUBLISH_WITHHELD family=%s key=%s — %s; "
        "cell stays local to this pod",
        pending.family, pending.arm_token, reason)
    # pgw#677 reopen: the withhold decision is hub-relevant truth (the mint
    # obligation stays undischarged and every cold pod re-mints) — it must
    # never live only in unreachable pod logs.
    activity_mod.emit_event(
        "self_mint_publish_withheld",
        f"family={pending.family} key={pending.arm_token}: {reason}",
        phase="incomplete",
    )
    mark_terminus(pending, TERMINUS_WITHHELD)
    shutil.rmtree(pending.mint_root, ignore_errors=True)


#: pgw#815: every way a self-mint obligation can END. A pending that carries
#: none of these when its boot finishes was neither published, nor withheld,
#: nor aborted, nor abandoned — it VANISHED, which is exactly what a 24-minute
#: L4 mint did while its activity reported `finalize completed`.
TERMINUS_SEALED = "sealed"
TERMINUS_PUBLISHING = "publishing"
TERMINUS_WITHHELD = "withheld"
TERMINUS_ABORTED = "aborted"
TERMINUS_ABANDONED = "abandoned"


def mark_terminus(pending: "PendingSelfMint", name: str) -> None:
    """Record that this mint obligation reached ``name`` (pgw#815)."""
    pending._state["terminus"] = name


def terminus_of(pending: "PendingSelfMint") -> str:
    """The terminus this mint obligation reached, "" when it reached none."""
    return str(pending._state.get("terminus") or "")


def abandon_self_mint(pending: "PendingSelfMint") -> None:
    """Discard a self-mint capture the proof did not certify (disproven or
    genuinely unexercised with no proven sibling). Never packed, never
    published — only the temp capture dir is cleaned up. A no-op when a
    proven sibling already finalized the shared capture (the artifact and
    its publish must survive).

    pgw#848 item 5: this rmtree is why the crash-only resume bank is NOT sited
    under ``mint_root``. Abandonment is how a crashed mint ends, so a bank here
    would be destroyed on its way out of the one case it exists for. It lives
    in the worker-local resume area instead (``aot_resume.bank_root``), keyed
    by scope, and is dropped only when a cell actually ADOPTS. Keeping it past
    an abandonment is safe by construction rather than by policy: nothing is
    re-admitted without its identity being re-derived from a freshly exported
    program."""
    if pending._state.get("minted") is not None:
        return
    mark_terminus(pending, TERMINUS_ABANDONED)
    _unregister(pending)
    shutil.rmtree(pending.mint_root, ignore_errors=True)


def _unregister(pending: "PendingSelfMint") -> None:
    with _PENDING_LOCK:
        if _PENDING.get(pending.arm_token) is pending:
            del _PENDING[pending.arm_token]


#: pgw#813: the typed `self_mint_skipped` phase each delegation refusal
#: declines under. The old single `aot_requires_delegation` phase carried a
#: hand-written either/or sentence ("GEN_WORKER_MINT_IN_PROCESS or eager-first
#: off") that named two causes which were BOTH false on the measured pod while
#: the true cause — the pipeline-side mandatory-lane misclassification — was
#: not named at all. A refusal that cannot name its own cause is the defect.
#: pgw#995 dropped `eager_first_disabled`: eager-first is unconditional, so
#: that cause can no longer arise and a reason nobody can reach is dead prose.
#: pgw#1010 dropped `mint_in_process_forced` with the env that produced it —
#: there is no in-process mint shape to force.
_DELEGATION_DECLINE_PHASE = {
    "no_eager_tier": "aot_no_eager_tier",
    "caller_forced_in_process": "aot_mint_forced_in_process",
}
_DELEGATION_DECLINE_DETAIL = {
    "no_eager_tier":
        "an armed non-eager backend (AOTI cell or TRT engine) has replaced "
        "this pipeline's forward, so there is no eager tier to serve from",
    "caller_forced_in_process":
        "the caller passed delegate=False, and an AOTI export has no eager "
        "tier to serve from while it compiles",
}


def mint_recipe(
    pipe: Any, cfg: Any, *, delegate: bool, emit: bool = True,
    delegate_refusal: str = "",
) -> str:
    """WHICH mint a miss on this pipeline should run (pgw#805).

    The AOT lane was a pure CONSUMER: discovery filtered for
    ``kind == "aot-inductor"`` artifacts and a miss fell through to the dynamo
    self-mint, whose cell can never satisfy that filter. So a fleet missed,
    re-minted the wrong kind (or nothing), and missed identically on every
    subsequent pod, forever. pgw#1010 finished that: the dynamo answer no
    longer mints anything at all — it is the JIT INTAKE posture, and every
    return of it here is a decline that says so.

    Every decline here is NAMED on the wire. A silent decline is the defect
    class this issue exists to kill: five real L4 pods produced no mint and no
    refusal, which is indistinguishable from a crash.

    pgw#996 split the branches by WHEN they are knowable. What is left asks
    only questions a pod can answer and a build cannot: whether delegation is
    available right now, whether this family declares an export at all, and
    whether the declaration fits the pipeline this pod actually COMPOSED. The
    image's own properties — C++ toolchain, torch floor, whether the
    declaration module even imports — are refused at build time by
    ``aot_preconditions``; reaching a pod is proof they hold.
    """
    family = str(getattr(cfg, "family", "") or "")

    def _decline(reason: str, detail: str) -> str:
        logger.info("fleet-cells: AOT mint declined (%s): %s", reason, detail)
        if emit:
            activity_mod.emit_event(
                "self_mint_skipped",
                f"family={family}: this miss cannot "
                f"mint an aot-inductor cell — {detail}; this pod serves JIT "
                f"INTAKE instead (it compiles its own graphs and serves them "
                f"for its own life) and MINTS NOTHING — no cell, no key, no "
                f"publish, no obligation (pgw#1010)",
                phase=reason,
            )
        return RECIPE_DYNAMO

    if not delegate:
        # An AOTI export holds the GPU for the whole compile with no router to
        # yield through; in-process it would violate the eager-first serving
        # contract outright. Delegation is not an optimization here.
        #
        # pgw#813: the REASON is threaded in, never re-guessed. Each refusal
        # declines under its own phase so `self_mint_skipped` is groupable by
        # actual cause instead of by a sentence listing candidates.
        return _decline(
            _DELEGATION_DECLINE_PHASE.get(
                delegate_refusal, "aot_requires_delegation"),
            _DELEGATION_DECLINE_DETAIL.get(
                delegate_refusal,
                "out-of-process minting is unavailable and an AOTI export "
                "has no eager tier to serve from while it compiles"))

    # pgw#853 put the refusal HERE rather than at import, because a refusal to
    # MINT is not a refusal to IMPORT. pgw#1107 retired the thunk that carried
    # it: the accessor is now a registry read that cannot raise, and the
    # refusal arrives below as DATA (`open_blockers`) under its own phase. The
    # `try/except` that used to wrap this call went with it — a gate that can
    # no longer fire is a decorative one.
    decl = export_declaration(family)

    if decl is None:
        return _decline(
            "no_export_declaration",
            f"family {family!r} registered no export declaration (a "
            f"`compile=` block carrying graph classes, pgw#739/#758) — the "
            f"class set a multi-graph cell covers is undeclared")

    # pgw#1115: the refusal, as DATA. A `Compile` carrying unresolved
    # `blockers=` declines here under its own phase, naming the ids — the one
    # form the fold onto `@endpoint(compile=)` can carry, and since pgw#1107
    # the only one. Serving is untouched: this pod serves eager exactly as it
    # did.
    blocked = open_blockers(decl)
    if blocked:
        return _decline("declaration_blocked", blocker_refusal(family, blocked))

    # CYCLE: aot_mint imports CellPublisher from this module at module scope,
    # so this direction of the pair must stay deferred.
    from . import aot_mint

    spec = aot_export_spec(pipe, cfg)
    # pgw#850/#879: there is NO lane admission here. The lane this pod serves
    # was chosen by the hub's resolution tree and observed off the composed
    # pipeline; re-ranking it at mint time was a second opinion that composed
    # with tensorhub's compiled-only `fp8-w8a8-dynamic` into a total block —
    # the one lane the mint admitted was the one lane no AUTO pod could be on.
    # Every check below answers "can this compile physically run", never
    # "should this lane exist".
    # pgw#996: the C++ toolchain (pgw#823) and the lifted-LoRA torch floor
    # (pgw#723) USED to be asked here. Both are properties of the IMAGE — apt
    # installed g++, the pinned torch wheel — decided long before this pod was
    # rented, and answering them here could only ever downgrade the recipe and
    # bill the fleet for eager serving. They are now `aot_preconditions` rows
    # that the image build refuses on (`discovery.validate_endpoint_lock`), so
    # an image that reaches a pod has already proven them. What survives below
    # is the residue that a build genuinely cannot know: the COMPOSED pipeline.

    # pgw#822: the LAST thing checkable without renting anything. Every
    # declared graph class's input names against its target module's own
    # forward signature — per class, because the adapter fork's two halves
    # declare different contracts. A mismatch is a DECLARATION defect: no
    # child, no pod and no compile can resolve it, so spending one to
    # rediscover the sentence is pure waste. Declines only the mint; the
    # pipeline serves eager exactly as it did.
    decl_gaps = aot_mint.declaration_module_gaps(pipe, spec, decl)
    if decl_gaps:
        return _decline(
            "declaration_module_mismatch",
            f"family {family!r}'s export declaration does not fit the "
            f"composed pipeline: " + "; ".join(decl_gaps))

    # pgw#846: the AOT mint is always WHOLE-GRAPH. `Compile.regional` keeps
    # its dynamo/JIT meaning (ie#381) and is ignored here — regional export
    # is retired for production.
    return RECIPE_AOT


def aot_export_spec(pipe: Any, cfg: Any) -> "Any":
    """The :class:`aot_mint.ExportSpec` a LIVE serving pipeline describes.

    The operator CLI reads these facts from a hand-written mint-request JSON.
    A serving pod has something strictly better: the composed pipeline and the
    endpoint's own ``CompileCell``. Everything the request file carried is
    either declared (shapes/text_lens/guidance/bucket/family), observed on the
    pipeline (weight lane, precision), or owned by the export DECLARATION
    (the class rows, coordinates, dynamic contracts and input bindings) — so
    nothing here is a per-pod guess.
    """
    # CYCLE: aot_mint imports CellPublisher from this module at module scope,
    # so this direction of the pair must stay deferred.
    from . import aot_mint

    # pgw#1087: composing the declaration a mint will trace against. Expected
    # to be trivial and never proven so — and if it is not (an endpoint whose
    # `export_declaration()` does real work at compose time), that is exactly
    # the finding, because it sits on the critical path before the first trace.
    with boot_mod.span(
        boot_mod.PHASE_DECLARATION_COMPOSE,
        ref=str(getattr(cfg, "family", "") or ""),
    ) if boot_mod.in_boot() else contextlib.nullcontext():
        return _aot_export_spec(aot_mint, pipe, cfg)


def _aot_export_spec(aot_mint: Any, pipe: Any, cfg: Any) -> "Any":
    execution_lane = loading.pipeline_weight_lane(pipe)
    bucket = int(getattr(cfg, "lora_bucket", 0) or 0)
    return aot_mint.ExportSpec(
        family=str(getattr(cfg, "family", "") or ""),
        target="",
        weight_lane=execution_lane,
        # pgw#1076: the lane when the pipeline HAS one, and otherwise absent —
        # never "bf16". `aot_mint` derives the stamp from the modules it
        # traces when this is empty, so an unlabelled fp32 pipeline records
        # fp32 instead of a cast nobody performed.
        precision=execution_lane,
        lora_bucket=bucket,
        shapes=tuple(
            tuple(int(v) for v in row) for row in (getattr(cfg, "shapes", ()) or ())),
        text_lens=tuple(int(v) for v in (getattr(cfg, "text_lens", ()) or ())),
        guidance_scales=tuple(
            float(v) for v in (getattr(cfg, "guidance_scales", ()) or ())),
        # Input LIFTING is an SDK fact, not a family one: the bucket-bearing
        # lane promotes exactly these adapter tensors to graph inputs so one
        # artifact serves the whole bucket (pgw#725 option 2).
        lifted_inputs=(
            tuple(lora_lifted.LIFTED_INPUT_NAMES) if bucket else ()),
    )


def _fail_closed(
    pipe: Any, reason: str,
    selection_bug: Optional["cc.CellSelectionBugError"] = None,
    *, phase: EagerPhase = EagerPhase.MINT_UNAVAILABLE,
) -> ArmOutcome:
    """The quantized-lane policy at every exit that cannot produce a cell:
    plain lanes serve eager (never-raise miss policy), w8a8/w4a4 keep the
    typed refusal (same as the cozy-local store / pre-gw#587 production).
    ``selection_bug``, when set, is a genuine mint-impossibility exit that
    ALSO followed a caught cell_selection_bug (th#1031) — chained onto the
    raised refusal so the caller's report is never dropped."""

    execution_lane = loading.pipeline_weight_lane(pipe)
    # pgw#677 reopen: ONE serveability brain (cc.mandatory_serving) — the
    # hub-resolved execution lane outranks the weight-lane prefix, so an
    # eager-serveable mixed lane (sdxl #fp8-w8a8 storage on fp8-w8a16
    # execution) degrades to eager here instead of a typed refusal.
    if cc.mandatory_serving(pipe):
        refusal = cc.CompiledExecutionLaneUnavailableError(
            f"{execution_lane[:4].upper()} requires a compile cell and the self-mint "
            f"is unavailable ({reason})")
        if selection_bug is not None:
            raise refusal from selection_bug
        raise refusal
    # pgw#805: this exit used to be a bare `logger.info` — and a serve pod
    # exposes no logs (pgw#760), so "declared a compile target, minted
    # nothing, refused nothing" was the whole observable behaviour of five
    # real L4 pods. A plain lane DEGRADING to eager is a legitimate policy
    # outcome; being unable to say so is not.
    #
    # pgw#824: the phase is the CLASSIFIED cause, not the constant
    # ``mint_unavailable`` every one of the nine exits used to share. The cause
    # was only ever in the free-text detail, so counting "how much of this fleet
    # is eager because there is no C toolchain" meant substring-matching a
    # sentence. It is the same token the request row's ``fallback_reason``
    # carries, so the two join.
    logger.info("fleet-cells: serving eager (%s)", reason)
    activity_mod.emit_event(
        "self_mint_skipped",
        f"lane={execution_lane or 'plain'}: no cell and no mint — {reason}; this "
        f"worker serves eager and publishes nothing",
        phase=phase,
    )
    return ArmOutcome(
        armed=False, selection_bug=selection_bug, eager_reason=phase)


def _cuda_ready() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


__all__ = [
    "ArmOutcome",
    "CellPublishRefused",
    "CellPublisher",
    "PendingSelfMint",
    "SelfMint",
    "abandon_self_mint",
    "arm_axis_divergence",
    "delegation_refusal",
    "enable_compiled",
    "finalized_in_process",
    "mark_terminus",
    "mint_recipe",
    "publish_self_mint",
    "publishes_in_flight",
    "terminus_of",
    "withhold_self_mint_publish",
]
