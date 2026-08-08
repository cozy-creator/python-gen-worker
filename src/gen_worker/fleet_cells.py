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
     (``aot_cells`` adopts ``aot-inductor`` only, by name), so every one
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

cozy-local NEVER uses this module (its self-mint stays local-store-only in
the cozy-local cell store module — user-controlled hardware is untrusted tier by
definition); the local CLI path does not construct a publisher.

Mint failures keep the pre-self-mint miss policy: plain lanes serve eager,
quantized (w8a8/w4a4) lanes keep their typed fail-closed refusal.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple


from . import activity as activity_mod
from . import aot_cells, cell_key
from . import boot_phases as boot_mod
from . import compile_cache as cc
from .cell_adopt import AdoptOutcome, CellAdoption, EagerPhase
from .models.chunk_cas import sha256_file
from .procsplit import broker
# module import (not `from .loading import pipeline_weight_lane`): tests
# monkeypatch models.loading.pipeline_weight_lane; stay late-bound.
from .models import loading, provision
from .request_context._helpers import _decode_unverified_jwt_claims
from .convert.hub import HubPublishError
from .api.export_contract import export_declaration
from .models import lora_lifted

logger = logging.getLogger(__name__)

#: pgw#805 mint recipes, as re-cut by pgw#1010. ``aot`` (torch.export +
#: AOTInductor) is the ONLY recipe that produces an artifact — the only kind
#: ``aot_cells.discover`` will ever adopt. ``dynamo`` is no longer a mint at
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
    half of gw#587/th#910:
    the minting worker ADVERTISES this identity under its own key so the
    hub's self-attested dispatch fence
    (``ActiveCompileRef == KeyRef(family, requested_cell_key)``) and
    ``active_compile_artifacts`` accounting treat the mint exactly like a
    delivered cell — the warmup proof, not the artifact source, gates
    serving.
    """

    family: str
    cell_key: str
    ref: str  # "root/family-<f>#<key>" — compile_cache.system_repo + key
    snapshot_digest: str  # "sha256:<hex>" of the packed artifact (self-attested)
    artifact: Path


@dataclass(frozen=True)
class PendingSelfMint:
    """An AOT mint this worker OWES, handed to a child process (pgw#784).

    ``enable_compiled`` returns this on a miss instead of an already-packed
    :class:`SelfMint`: the live pipeline keeps serving eager while the child
    exports, and ``adopt_delegated_mint`` swaps it through the ordinary
    delivered-cell path once the child's cell earns adoption.

    One instance may be SHARED by several pipelines of one record whose axes
    compute the same key (the qwen edit shape: two lanes, one family cell).
    ``_state`` memoizes the adopt outcome so sibling candidates converge on
    one publish.

    pgw#1010: a DYNAMO miss no longer builds one of these. The JIT recipe
    keeps its serving role (intake, ``compile_cache.arm_jit_intake``) and
    produces no artifact at all, so every pending in this module is an AOT
    mint and every pending is delegated.
    """

    family: str
    cell_key: str
    ref: str
    cfg: Any
    target: Path
    mint_root: Path
    publisher: Optional["CellPublisher"]
    cache_dir: Optional[Path] = None
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
    but a live cell key can legitimately collide across two structurally
    different graphs (cell_key has no graph-shape axis), so a caught
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

# Live self-mint captures by cell key. The inductor capture dir is process-
# global (one TORCHINDUCTOR_CACHE_DIR), so at most one key's capture may be
# live at a time; same-key sibling pipes join the existing capture.
_PENDING_LOCK = threading.Lock()
_PENDING: Dict[str, "PendingSelfMint"] = {}
# pgw#672: cells this process already finalized (packed + folded into the
# live cache root). A later same-key arm re-arms cache_ready from the folded
# entries instead of opening a SECOND capture — which, with the first mint's
# compiled code resident in dynamo's in-memory cache, would capture nothing
# and disprove itself at finalize (the L4 churn loop).
_FINALIZED: Dict[str, "SelfMint"] = {}


def finalized_in_process(key: str) -> Optional["SelfMint"]:
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

# pgw#988 — the OTHER half of this decision, asserted rather than assumed.
#
# What the declare carries and what the consumer verifies off it are two halves
# of ONE contract, and until now they were two independent computations of it in
# two modules. th#1645 dropped `entries` here — right about SIZE — while
# `aot_cells._candidates` still refused any cell whose declare had no entries
# map, so 100% of AOT cells published from that commit on were undiscoverable on
# every pod. A pod that finds no cell mints its own, so the fleet paid a full
# compile per cold boot and the failure presented as COST, never as an error.
#
# Checked at import, over two frozensets, so the next block that moves fails
# every test run rather than a fleet.
def assert_declare_contract(
    dropped: "frozenset[str]" = _UNBOUNDED_ENVELOPE_BLOCKS,
    read: "frozenset[str]" = aot_cells.DECLARE_CONTRACT_KEYS,
) -> None:
    """Raise when the declare strips a key its consumer still verifies."""
    breach = sorted(dropped & read)
    if breach:
        raise RuntimeError(
            f"cell declare contract broken (pgw#988): {breach!r} is stripped "
            "from the publish declare but is still read off it by "
            "aot_cells._candidates — every cell published this way would be "
            "undiscoverable, and every pod would re-mint. Either keep the "
            "block in the declare or stop verifying against it before the "
            "download.")


assert_declare_contract()

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
        resp = broker.request(
            "POST",
            path,
            base_url=self.base_url,
            bearer=bearer,
            json=payload,
            timeout=timeout,
        )
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
            key = cell_key.from_artifact_metadata(meta).digest
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


def _identity_axes(family: str, meta: dict) -> Dict[str, str]:
    """The ck axes that hash into this cell's key, for the hub's inventory.

    Recomputed from the artifact's OWN recorded axes so a stamp can never
    disagree with what it summarizes — the same rule
    :func:`cell_key.from_artifact_metadata` enforces for the key itself.

    Two artifacts legitimately cannot state them: EXPORTED (``aot-inductor``)
    cells carry a STAMPED key whose axes are not an inductor cache's
    (pgw#735), and pre-key artifacts record no contract block at all. Both
    fall back to the axes the metadata does hold, so the hub still learns the
    family, the lane and the card rather than nothing. Never raises — an
    inventory detail must not fail a publish.
    """
    try:
        return {k: str(v) for k, v in
                cell_key.from_artifact_metadata(meta).axes_dict().items()}
    except Exception:  # noqa: BLE001 — diagnostic only, never fatal
        pass
    fallback: Dict[str, str] = {}
    for name, value in (
        ("family", family or meta.get("family")),
        ("kind", meta.get("kind")),
        ("format", meta.get("format")),
        ("sm", meta.get("sm")),
        ("mode", meta.get("compile_mode")),
    ):
        text = str(value or "").strip()
        if text:
            fallback[name] = text
    try:
        base, observed = cc.execution_lane_bucket(str(meta.get("weight_lane") or ""))
        bucket = observed or int(meta.get("lora_bucket") or 0)
        token = cc.execution_lane_token(base)
        execution_lane = f"{token}-lora{bucket}" if bucket and token else (
            f"lora{bucket}" if bucket else token)
        if execution_lane:
            fallback["lane"] = execution_lane
    except Exception:  # noqa: BLE001
        pass
    return fallback


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
) -> ArmOutcome:
    """Fleet arming policy, plus the adoption ledger every exit shares.

    pgw#923: the policy has a dozen exits and an adoption attempt can precede
    any of them, so the measured attempts are collected in ONE place rather
    than threaded through each ``return``. That is the shape that lets a
    refusal be reported: the old code could only announce an adoption from the
    frame that made it, which is why the successful ones were narrated and the
    measured ones never sent.
    """
    adoptions: List[CellAdoption] = []
    outcome = _arming_policy(
        pipe, cfg, cache_dir, artifact,
        publisher=publisher, delegate=delegate,
        delivered_ref=delivered_ref, delivered_digest=delivered_digest,
        adoptions=adoptions,
    )
    if not adoptions:
        return outcome
    return dataclasses.replace(outcome, adoptions=tuple(adoptions))


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
    arm — cell_key has no graph-shape axis, so two structurally different
    graphs can legitimately collide on one key) no longer aborts arming: it
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

    # PREFER a published aot-inductor cell over the delivered dynamo artifact.
    # Discovery is fetch-and-filter (the worker cannot compute a stamped AOT
    # key); the downloaded artifact rides the SAME choke point below, so the
    # pgw#709 receipt gate and the aot_serve arm gates run unchanged. Any
    # miss/failure falls through to today's policy with the originally
    # delivered artifact.
    #
    # pgw#990: UNCONDITIONAL. This was `aot_cells.prefer_aot()` — a default-OFF
    # pilot switch carried by a release-scoped env entry — and the release
    # stopped declaring the name on 2026-08-05, so every serving pod between
    # then and attempt twenty-four skipped discovery entirely and self-minted
    # over a published cell. Adoption is the path; a gate that un-arms it on a
    # rebuild protects nothing.
    if (
        family
        and publisher is not None
        and publisher.enabled()
        and cc.has_compile_target(pipe, cfg)
    ):
        adopted = aot_cells.discover(
            pipe, cfg,
            base_url=publisher.base_url,
            worker_jwt=publisher.worker_jwt,
            cache_dir=cache_dir,
        )
        if adopted is not None:
            from . import aot_serve

            try:
                aot_out, aot_row = _arm_candidate(
                    pipe, cfg, cache_dir, adopted.artifact,
                    ref=adopted.ref,
                    snapshot_digest=adopted.snapshot_digest,
                    artifact_kind=aot_serve.ARTIFACT_KIND,
                )
            except cc.CompiledExecutionLaneUnavailableError:
                # The AOT arm refused and the mandatory-lane no-artifact
                # fallthrough raised — the ordinary policy below (with the
                # delivered artifact) is still to run, so this is not
                # terminal here.
                logger.warning(
                    "fleet-cells: discovered AOT cell %s did not arm; "
                    "falling through to the ordinary arming policy",
                    adopted.ref)
                adoptions.append(CellAdoption(
                    ref=adopted.ref,
                    snapshot_digest=adopted.snapshot_digest,
                    artifact_kind=aot_serve.ARTIFACT_KIND,
                    arm_ms=0,
                    armed=False,
                    reason="lane_unavailable",
                    detail=(
                        f"family={family} key={adopted.cell_key}: arm refused "
                        "and the mandatory-lane fallthrough raised; ordinary "
                        "policy resumes with the delivered artifact"),
                    pipeline_id=id(pipe),
                ))
            else:
                if aot_out.armed and not aot_serve.is_armed(pipe):
                    # Armed, but not through the exported artifact (e.g. a
                    # seeded dynamo cell picked up on the fallthrough): honest
                    # plain HIT — never advertise the AOT identity for bytes
                    # this pipe does not serve, so the DISCOVERED cell's own
                    # adoption is a miss with its own token.
                    aot_row = dataclasses.replace(
                        aot_row, armed=False, reason="armed_other_path",
                        detail=(
                            f"family={family} key={adopted.cell_key}: armed "
                            "via the fallthrough, NOT the discovered "
                            "artifact; AOT identity not advertised"))
                    adoptions.append(aot_row)
                    return ArmOutcome(armed=True)
                adoptions.append(aot_row)
                if aot_out.armed:
                    logger.info(
                        "fleet-cells: armed discovered AOT cell %s (pgw#722)",
                        adopted.ref)
                    return ArmOutcome(armed=True, self_mint=adopted)

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
    # ``cell_key`` is computable from STATIC axes (sku/torch/image/weight
    # lane/declared shapes+targets/module structure) — never the traced FX
    # graph bytes — so the ref the hub's self-attested dispatch fence needs
    # is known BEFORE any compile has happened.

    try:
        key = cell_key.compute(
            family, loading.pipeline_weight_lane(pipe), bucket,
            contract=cell_key.contract_digest(
                cc.declared_contract_facts(cfg)),
            regional=bool(getattr(cfg, "regional", False)),
        ).digest
    except Exception as exc:  # noqa: BLE001 — key axes must be computable
        logger.warning("fleet-cells: self-mint key computation failed (%s)", exc)
        if bucket:
            cc.drop_lora_execution_lane(pipe)
        return _fail_closed(
            pipe, f"self-mint key computation failed: {exc}", selection_bug,
            phase=EagerPhase.KEY_COMPUTATION_FAILED)

    # pgw#672: consult the process ledgers BEFORE opening a capture.
    key_ref = f"{cc.system_repo(family)}#{key}"
    if cc.cell_quarantined_in_process(key_ref):
        # This exact identity already failed its serve/finalize proof in
        # this process — re-minting it is the churn loop, not recovery.
        # DEGRADE (explicit eager, mandatory lanes included): a broken
        # optimization must never kill a serving worker.
        logger.error(
            "fleet-cells: declining self-mint for %s key=%s — this identity "
            "was quarantined by a failed proof in this process; serving "
            "eager (pgw#672)", family, key)
        if bucket:
            cc.drop_lora_execution_lane(pipe)
        # pgw#824: the ONE eager exit in this function that never rode a typed
        # event — it returns before `_fail_closed` and only logged. A pod that
        # quarantined its own cell serves eager for the rest of its life, which
        # is precisely the state the hub most needs named.
        activity_mod.emit_event(
            "self_mint_skipped",
            f"family={family} key={key} lane="
            f"{loading.pipeline_weight_lane(pipe) or 'plain'}: this identity "
            f"was quarantined by a failed serve/finalize proof earlier in this "
            f"process; re-minting it is the churn loop, so this worker serves "
            f"eager for the rest of its life and publishes nothing",
            phase=EagerPhase.CELL_QUARANTINED,
        )
        return ArmOutcome(
            armed=False, selection_bug=selection_bug,
            eager_reason=EagerPhase.CELL_QUARANTINED)
    finalized_prior = finalized_in_process(key)
    if finalized_prior is not None:
        # This process already minted and ADOPTED this exact cell — re-arm
        # the same artifact through the same AOT gates instead of paying a
        # second export for bytes that are already on this disk.
        try:
            armed_ready = bool(provision.arm_aot(
                pipe, cfg, cache_dir, Path(finalized_prior.artifact), bucket))
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

    # Sibling pipes of one record whose axes compute the SAME key share one
    # child mint (the qwen edit shape: two lanes, one family cell) — the
    # first arm registers the pending and every sharer adopts its cell.
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
        family=family, cell_key=key,
        ref=f"{cc.system_repo(family)}#{key}",
        cfg=cfg, target=target,
        mint_root=mint_root, publisher=publisher, cache_dir=cache_dir,
    )
    with _PENDING_LOCK:
        _PENDING.setdefault(key, pending)
    logger.info(
        "fleet-cells: DELEGATED %s self-mint for %s (key=%s) — a child "
        "process builds the cell while this process serves eager "
        "(pgw#784/#805)", recipe, family, key)
    activity_mod.emit_event(
        "self_mint_started",
        f"family={family} recipe={recipe} key={key} "
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
    with tarfile.open(artifact, mode="r:*") as tar:
        member = tar.extractfile(cc.METADATA_NAME)
        if member is None:
            raise RuntimeError(f"{artifact} carries no {cc.METADATA_NAME}")
        return dict(json.loads(member.read().decode()))


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
    # pgw#999: (reason, detail) of the arm refusal, set by whichever branch
    # refuses. Initialized to a NAMED unset rather than "" so a branch that
    # forgets to classify is visible on the wire as a gap in this function
    # instead of as an empty string that reads like "no reason exists".
    refusal: Tuple[str, str] = ("unclassified_arm_refusal", "")
    try:
        # pgw#805: an exported cell arms through the AOT gates (lifted-binding
        # install -> aot_serve.enable -> rollback), not the inductor seed path.
        # Same gates a hub-delivered `.pt2` passes; `provision.enable_compiled`
        # itself is not reusable here because its pgw#709 receipts gate would
        # drop a cell this pod minted seconds ago and the hub has not
        # countersigned yet.
        # pgw#999: the outcome is KEPT. `arm_aot` returns a classified
        # `AdoptOutcome` (`contract_invalid`, `constants_unbound`,
        # `no_arm_for_mode`, `numerics_refused`, …) and this call site
        # used to spend it on `bool(...)`. That discard cost attempt 26
        # 2 h 45 m and $2.72: a 36/36 mint sealed, finalized, and was then
        # refused by three events that all said "could not adopt".
        # pgw#1010: there is no second branch — a delegated mint is an AOT
        # mint, and the inductor-seed adopt it used to fall back to armed an
        # artifact class nothing publishes any more.
        outcome = provision.arm_aot(
            pipe, pending.cfg, pending.cache_dir, pending.target,
            int(getattr(pending.cfg, "lora_bucket", 0) or 0))
        armed = bool(outcome)
        if not armed:
            refusal = (outcome.reason or "unclassified_arm_refusal",
                       outcome.detail or outcome.identity)
    except cc.CellSelectionBugError as exc:
        # th#883, delegated edition: the child's own cell, whose axes describe
        # exactly this runtime, refused to arm. Loud — it is a bug in the one
        # selection brain, not a compatibility miss.
        logger.error(
            "fleet-cells: cell_selection_bug adopting this pod's DELEGATED "
            "mint (family=%s key=%s): %s",
            pending.family, pending.cell_key, exc)
        armed = False
        refusal = ("cell_selection_bug", str(exc))
    except Exception as exc:  # noqa: BLE001 — adoption failure => eager
        logger.warning(
            "fleet-cells: delegated mint for %s did not adopt (%s)",
            pending.family, exc)
        armed = False
        # An `AdoptError` already carries the token; anything else is named by
        # its type rather than flattened into one word nobody can count.
        refusal = (str(getattr(exc, "reason", "") or "") or type(exc).__name__,
                   str(exc))
    if not armed:
        reason, detail = refusal
        # pgw#999: `phase` is the countable column, so it carries the CLASS —
        # the same convention `self_mint_skipped` already uses. The old
        # constant `delegated_adopt_failed` said only which call site fired,
        # which every reader already knew from the event kind.
        state["adopt_refusal"] = (reason, detail)
        activity_mod.emit_event(
            "self_mint_abort",
            f"family={pending.family} key={pending.cell_key}: the child "
            f"process produced a cell this runtime could not adopt "
            f"({reason}{': ' + detail if detail else ''}); serving stays "
            f"eager and nothing is published",
            phase=reason,
        )
        mark_terminus(pending, TERMINUS_ABORTED)
        state["minted"] = None
        _unregister(pending)
        shutil.rmtree(pending.mint_root, ignore_errors=True)
        return None

    meta = _packed_metadata(pending.target)
    key = str(meta.get("cell_key") or "").strip() or pending.cell_key
    minted = SelfMint(
        family=pending.family, cell_key=key,
        ref=f"{cc.system_repo(pending.family)}#{key}",
        snapshot_digest="sha256:" + sha256_file(pending.target),
        artifact=pending.target,
    )
    state["minted"] = minted
    state["meta"] = dict(meta)
    # th#1355: the mint cost, banked at the moment the cell becomes real.
    state["mint_duration_ms"] = max(
        0, int((time.monotonic() - pending.armed_at) * 1000))
    _unregister(pending)
    with _PENDING_LOCK:
        _FINALIZED[key] = minted
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
            f"family={pending.family} key={pending.cell_key}: the publish "
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
                or pending.cell_key),
            mint_duration_ms=int(state.get("mint_duration_ms") or 0))
    else:
        # Runtime assertion (gw#587): every fleet cell miss must produce a
        # publish attempt. A fleet worker minting with no usable sink is a
        # wiring defect (file_base_url/worker JWT absent at arming time),
        # not a policy choice — loud, greppable, alarm-adjacent. (cozy-local
        # legitimately has no publisher, but it never enters this module.)
        logger.warning(
            "fleet-cells: SELF_MINT_WITHOUT_PUBLISH_SINK family=%s — cell "
            "stays local to this pod; the fleet store gains nothing",
            pending.family)
        activity_mod.emit_event(
            "self_mint_publish_withheld",
            f"family={pending.family} key={pending.cell_key}: no publish "
            "sink (file_base_url/worker JWT absent at arming time); cell "
            "stays local to this pod",
            phase="no_sink",
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
            f"family={pending.family} key={pending.cell_key}: {reason} — and "
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
        pending.family, pending.cell_key, reason)
    # pgw#677 reopen: the withhold decision is hub-relevant truth (the mint
    # obligation stays undischarged and every cold pod re-mints) — it must
    # never live only in unreachable pod logs.
    activity_mod.emit_event(
        "self_mint_publish_withheld",
        f"family={pending.family} key={pending.cell_key}: {reason}",
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
        if _PENDING.get(pending.cell_key) is pending:
            del _PENDING[pending.cell_key]


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

    The AOT lane was a pure CONSUMER: ``aot_cells.discover`` filtered for
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

    # pgw#853: THIS is where a declaration is allowed to refuse. A family with
    # open mint blockers (ltx/qwen/z-image) registers a THUNK, so its refusal
    # arrives here — as a typed `self_mint_skipped` carrying every word of the
    # blocker text — instead of as an ImportError that takes the endpoint
    # down at boot. A refusal to MINT is not a refusal to IMPORT.
    #
    # pgw#996: the STATIC half of this refusal (unresolved MINT_BLOCKERS) is
    # already recorded by the build gate, so no pod discovers it for the first
    # time. The branch survives because the other half is genuinely a pod
    # fact: ltx-2.3's thunk reads `LTX_SERVING_TIER` off the environment, and
    # an unknown tier there is a MintRefused the image could not have known.
    try:
        decl = export_declaration(family)
    except Exception as exc:  # noqa: BLE001 — serving outranks compiling
        # Deliberately `Exception`, not `BaseException`: this runs INSIDE the
        # serving process, where swallowing a KeyboardInterrupt/cancellation
        # would be its own defect. Every refusal shape the declarations
        # actually raise (MintRefused, DeclarationError, ImportError) is an
        # Exception. The import boundary is the other way round — see
        # `import_export_declaration`, which runs at BOOT and catches
        # everything, because there nothing outranks the endpoint coming up.
        return _decline(
            "declaration_refused",
            f"family {family!r}'s export declaration refuses to mint "
            f"({type(exc).__name__}): {exc}")

    if decl is None:
        return _decline(
            "no_export_declaration",
            f"family {family!r} registered no export declaration (a "
            f"`compile=` block carrying graph classes, pgw#739/#758) — the "
            f"class set a multi-graph cell covers is undeclared")

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

    execution_lane = loading.pipeline_weight_lane(pipe)
    bucket = int(getattr(cfg, "lora_bucket", 0) or 0)
    return aot_mint.ExportSpec(
        family=str(getattr(cfg, "family", "") or ""),
        target="",
        weight_lane=execution_lane,
        precision=execution_lane or "bf16",
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
