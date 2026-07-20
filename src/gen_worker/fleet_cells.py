"""Fleet self-mint compile cells (gw#587).

The serving worker's boot warmup IS a perfect mint by construction: right
SKU (its own card), right image (its own digest), right weight lane (its
own loader decision), right pipeline class and call path (its own endpoint
code), right shapes (its own declaration). gw#586 proved the replicated-
producer alternative is a parity treadmill — every axis the producer must
replicate was discovered as a live production failure, one at a time.

Under self-mint the arming policy for a compile-declared function becomes:

  1. HIT (hub attached the cell for this runtime's self-computed key):
     arm through the delivered-cell path — today's behavior, unchanged.
  2. MISS — prove-produces-the-mint (gw#587 CORRECT FIX): arm the
     pipeline COLD into a fresh capture dir with NO synthetic warm call
     (``compile_cache.begin_fleet_mint``). The executor's real warmup
     proof — the endpoint's own serving code, the exact call shapes
     production requests make — performs the only compile the mint ever
     sees. The old design ran the producer-style ``_warm_call`` loop
     first and proved afterwards; its synthetic call traced different FX
     graphs than a conditioned/two-stage endpoint warmup (the gw#586
     defect class resurfacing inside self-mint), so the proof correctly
     refused its own artifact. Now the artifact is byte-derived from the
     same execution the proof observed — there is no second code path
     that re-creates serving's execution to drift from.
  3. FINALIZE + PUBLISH, only after the proof PASSES
     (``finalize_self_mint``): pack the proven capture, advertise its
     real digest, then ship it through the hub's attested publish gate
     (th#910) in the background so the next worker on this key is
     store-served. A failed proof abandons the capture — nothing
     unproven is ever packed or published (this also closes the old
     publish-before-proof window). Publish failure NEVER affects serving;
     a refusal (untrusted tier, attestation, quota) is the hub's call and
     is fully recorded hub-side.

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
import logging
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from . import compile_cache as cc
from .models import provision

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelfMint:
    """Identity of one successfully adopted, FINALIZED self-minted cell.

    Produced only by :func:`finalize_self_mint`, after the executor's
    warmup proof confirms the real serving call exercised the compiled
    targets it identifies. The serving-bootstrap half of gw#587/th#910:
    the minting worker ADVERTISES this identity under its own key so the
    hub's self-attested dispatch fence
    (``ActiveCompileRef == KeyRef(family, requested_cell_key)``) and
    ``active_compile_artifacts`` accounting treat the mint exactly like a
    delivered cell — the warmup proof, not the artifact source, gates
    serving.
    """

    family: str
    cell_key: str
    ref: str  # "_system/family-<f>#<key>" — compile_cache.system_repo + key
    snapshot_digest: str  # "blake3:<hex>" of the packed artifact (self-attested)
    artifact: Path


@dataclass(frozen=True)
class PendingSelfMint:
    """A self-mint ARMED for capture, not yet proven or packed (gw#587
    CORRECT FIX).

    ``enable_compiled`` returns this on a miss instead of an already-
    packed :class:`SelfMint`: the pipeline is armed cold, pointed at
    ``capture_dir``, with NO synthetic warm call run against it. Only the
    executor's real warmup proof — the endpoint's own serving code —
    performs the compile this mint will ever see. ``ref`` is computable
    immediately (STATIC axes: sku/torch/image/weight-lane/shapes/graph
    structure — never the traced FX graph bytes), so the worker can
    advertise its claimed key ref at arm time; ``finalize_self_mint``
    packs the artifact and computes the real digest only after the proof
    passes, and publishes only from that proven capture.

    One instance may be SHARED by several pipelines of one record whose
    axes compute the same key (the qwen edit shape: two lanes, one family
    cell) — they cold-compile into the one capture during the one warmup
    window, and the packed cell is their union. ``_state`` memoizes the
    finalize outcome so sibling candidates converge on one pack/publish.
    """

    family: str
    cell_key: str
    ref: str
    cfg: Any
    target: Path
    capture_dir: Path
    mint_root: Path
    publisher: Optional["CellPublisher"]
    cache_dir: Optional[Path] = None
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
    """

    armed: bool
    self_mint: Optional[Any] = None

    def __bool__(self) -> bool:
        return self.armed

_INTENT_TIMEOUT_S = 30
_COMPLETE_TIMEOUT_S = 30

# Live self-mint captures by cell key. The inductor capture dir is process-
# global (one TORCHINDUCTOR_CACHE_DIR), so at most one key's capture may be
# live at a time; same-key sibling pipes join the existing capture.
_PENDING_LOCK = threading.Lock()
_PENDING: Dict[str, "PendingSelfMint"] = {}


class CellPublishRefused(Exception):
    """Typed hub refusal (attestation / trust tier / quota). Terminal for
    this publish attempt — never retried, never fatal to serving."""


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
        return bool(self.base_url and (self._worker_jwt() or "").strip())

    # -- wire ---------------------------------------------------------------

    def _post(self, path: str, payload: dict, *, timeout: float) -> dict:
        import requests

        resp = requests.post(
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {self._worker_jwt()}"},
            json=payload,
            timeout=timeout,
        )
        body: dict = {}
        try:
            body = resp.json() if resp.text else {}
        except Exception:
            body = {}
        if resp.status_code in (403, 429):
            # Typed refusals (cell_publish_forged_axis, _untrusted_tier,
            # _quota_exceeded, _family_undeclared): terminal by design.
            raise CellPublishRefused(
                f"{path} refused ({resp.status_code}): {resp.text[:300]}")
        if resp.status_code < 200 or resp.status_code >= 300:
            raise RuntimeError(f"{path} failed ({resp.status_code}): {resp.text[:300]}")
        return body if isinstance(body, dict) else {}

    # -- publish ------------------------------------------------------------

    def publish(self, family: str, artifact: Path, meta: dict) -> str:
        """Publish one self-minted cell. Returns the checkpoint id.

        Steps: attested intent (worker JWT; hub corroborates the axes and
        mints a key-pinned capability token) -> the standard commit flow
        (create-commit -> presigned CAS upload -> finalize; mode=replace, no
        tags — the hub refuses any tag bind under the claim anyway) ->
        publish-complete bookkeeping. Raises on any failure; the caller
        treats every raise as non-fatal to serving.
        """
        from . import cell_key

        key = str(meta.get("cell_key") or "").strip()
        if not key:
            key = cell_key.from_artifact_metadata(meta).digest
        axes = {
            "sku": str(meta.get("sku") or ""),
            "image_digest": self.image_digest,
            "gen_worker": str(meta.get("gen_worker") or ""),
        }
        intent = self._post(
            "/v1/worker/cells/publish-intent",
            {"family": family, "cell_key": key, "axes": axes},
            timeout=_INTENT_TIMEOUT_S,
        )
        token = str(intent.get("capability_token") or "").strip()
        repo = str(intent.get("repo") or "").strip()
        if not token or not repo:
            raise RuntimeError("publish-intent response missing token/repo")

        try:
            from .convert.hub import CommitFile, HubClient

            client = HubClient(base_url=self.base_url, token=token, owner="_system")
            result = client.commit(
                destination_repo=repo,
                files=[CommitFile(path=artifact.name, local_path=artifact)],
                mode="replace",
                flavor=key,
                metadata={k: v for k, v in dict(meta).items() if v is not None},
                message=f"self-mint {family} {key}",
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

        self._post(
            "/v1/worker/cells/publish-complete",
            {"family": family, "cell_key": key, "checkpoint_id": checkpoint_id,
             "ok": True},
            timeout=_COMPLETE_TIMEOUT_S,
        )
        logger.info(
            "fleet-cells: published %s#%s (checkpoint %s, %.1f MB)",
            family, key, checkpoint_id, artifact.stat().st_size / 1e6)
        return checkpoint_id


def _publish_async(publisher: CellPublisher, family: str, artifact: Path, meta: dict) -> threading.Thread:
    """Ship the cell in the background — readiness never waits on an upload.
    Every outcome is logged; refusals are the hub's recorded decision. The
    mint dir is cleaned once the publish attempt finishes (the adoption
    already staged its own copy under the cache dir)."""

    def run() -> None:
        try:
            publisher.publish(family, artifact, meta)
        except CellPublishRefused as exc:
            logger.warning("fleet-cells: publish refused (hub decision): %s", exc)
        except Exception:
            logger.warning("fleet-cells: publish failed; the next worker on this key re-mints", exc_info=True)
        finally:
            shutil.rmtree(artifact.parent, ignore_errors=True)

    t = threading.Thread(target=run, name="cell-publish", daemon=True)
    t.start()
    return t


def enable_compiled(
    pipe: Any,
    cfg: Any,
    cache_dir: Optional[Path] = None,
    artifact: Optional[Path] = None,
    publisher: Optional[CellPublisher] = None,
) -> ArmOutcome:
    """Fleet arming policy (gw#587): delivered cell first, self-mint on miss.

    Replaces the executor's bare ``provision.enable_compiled`` call. HIT
    keeps today's semantics exactly (incl. ``CellSelectionBugError``
    propagation — the th#883 receipt invariant is untouched). MISS
    self-mints and serves compiled; the ONLY remaining eager/fail-closed
    exits are genuine mint impossibilities (no CUDA, no C toolchain, the
    mint itself failing), where plain lanes serve eager and quantized lanes
    keep their typed refusal — exactly the cozy-local store policy.

    Returns :class:`ArmOutcome`; ``self_mint`` carries the minted cell's
    identity so the caller can record/advertise it (serving-bootstrap half
    of th#910 — the hub's self-attested fence needs the worker to claim its
    own key as the active compile ref).
    """
    family = str(getattr(cfg, "family", "") or "")
    try:
        if provision.enable_compiled(pipe, cfg, cache_dir, artifact):
            return ArmOutcome(armed=True)
        # Plain-lane miss: no cell delivered / artifact unusable. Fall
        # through to the self-mint instead of the pre-gw#587 silent eager.
    except cc.CellSelectionBugError:
        # Self-requested, identity-verified cell refused to arm — the loud
        # invariant propagates verbatim; a re-mint would mask a selection
        # bug.
        raise
    except cc.CompiledLaneUnavailableError:
        # Mandatory (w8a8/w4a4) miss: production used to fail closed here.
        # The whole point of self-mint is that this worker can produce the
        # cell itself.
        logger.info("fleet-cells: no delivered cell for mandatory lane; self-minting")

    if not family:
        return _fail_closed(pipe, "Compile decl has no family")
    if not _cuda_ready():
        return _fail_closed(pipe, "CUDA unavailable")
    if not cc.toolchain_present():
        return _fail_closed(pipe, "no C compiler for the self-mint")

    from .models.loading import pipeline_weight_lane

    # gw#561: the eager-miss rollback in provision.enable_compiled dropped
    # the branch lane; the mint must key + trace the DECLARED graph family.
    bucket = int(getattr(cfg, "lora_bucket", 0) or 0)
    if bucket:
        cc.apply_lora_lane(pipe, bucket)

    # ``cell_key`` is computable from STATIC axes (sku/torch/image/weight
    # lane/declared shapes+targets/module structure) — never the traced FX
    # graph bytes — so the ref the hub's self-attested dispatch fence needs
    # is known BEFORE any compile has happened.
    from . import cell_key as cell_key_mod

    try:
        key = cell_key_mod.compute(
            family, pipeline_weight_lane(pipe), bucket,
            regional=bool(getattr(cfg, "regional", False)),
        ).digest
    except Exception as exc:  # noqa: BLE001 — key axes must be computable
        logger.warning("fleet-cells: self-mint key computation failed (%s)", exc)
        if bucket:
            cc.drop_lora_lane(pipe)
        return _fail_closed(pipe, f"self-mint key computation failed: {exc}")

    # gw#587 CORRECT FIX (the defect this replaces: the old design minted
    # via a separate producer-style ``mint_artifact``/``_warm_call`` BEFORE
    # the real serving warmup ran — a synthetic single-stage call that can
    # trace DIFFERENT FX graphs than a conditioned/two-stage endpoint's own
    # warmup (the gw#586 defect class, live-found resurfacing inside self-
    # mint). Arm cold instead: the caller's real warmup — run by the
    # executor immediately after this returns — is the ONLY compile this
    # mint will ever see, so the eventual capture is byte-derived from
    # exactly the execution the proof observes. Nothing is packed or
    # published here; ``finalize_self_mint`` does that, and only after the
    # proof passes.
    #
    # The inductor capture dir is process-global (one TORCHINDUCTOR_CACHE_DIR)
    # so at most ONE capture key can be live at a time: sibling pipes of the
    # same record computing the SAME key share the one capture (their union
    # is the family cell — the qwen edit shape); a DIFFERENT key while a
    # capture is pending declines loudly into the ordinary miss policy (a
    # second dir would corrupt the first capture's byte-derivation).
    with _PENDING_LOCK:
        existing = _PENDING.get(key)
        conflict = next((k for k in _PENDING if k != key), None)
    if conflict is not None and existing is None:
        logger.warning(
            "fleet-cells: self-mint declined for %s key=%s — capture already "
            "pending for key=%s (one inductor capture dir per process)",
            family, key, conflict)
        if bucket:
            cc.drop_lora_lane(pipe)
        return _fail_closed(
            pipe, f"another self-mint capture is pending (key {conflict})")

    if existing is not None:
        mint_root, capture_dir = existing.mint_root, existing.capture_dir
        target = existing.target
    else:
        mint_root = Path(tempfile.mkdtemp(prefix="selfmint-"))
        capture_dir = mint_root / "capture"
        label = cc.flavor_label(
            cc.runtime_key()["sku"], cc.runtime_key()["torch"],
            pipeline_weight_lane(pipe))
        target = mint_root / f"{label}.tar.gz"

    try:
        cc.begin_fleet_mint(pipe, cfg, capture_dir)
    except Exception as exc:  # noqa: BLE001 — arm failure => miss policy
        logger.warning("fleet-cells: self-mint arm failed (%s)", exc)
        if existing is None:
            shutil.rmtree(mint_root, ignore_errors=True)
        if bucket:
            cc.drop_lora_lane(pipe)
        return _fail_closed(pipe, f"self-mint arm failed: {exc}")

    if existing is not None:
        logger.info(
            "fleet-cells: joined pending self-mint capture for %s (key=%s)",
            family, key)
        return ArmOutcome(armed=True, self_mint=existing)

    pending = PendingSelfMint(
        family=family, cell_key=key, ref=f"{cc.system_repo(family)}#{key}",
        cfg=cfg, target=target, capture_dir=capture_dir, mint_root=mint_root,
        publisher=publisher, cache_dir=cache_dir,
    )
    with _PENDING_LOCK:
        _PENDING[key] = pending
    logger.info(
        "fleet-cells: armed self-mint capture for %s (key=%s) — the real "
        "warmup proof performs the only compile this mint will see",
        family, key)
    return ArmOutcome(armed=True, self_mint=pending)


def finalize_self_mint(pipe: Any, pending: "PendingSelfMint") -> Optional[SelfMint]:
    """Pack + publish a self-mint AFTER the executor's warmup proof passes.

    Called from the executor's warmup-proof loop, per proven candidate —
    never before the proof confirms a real, successful compiled call on
    ``pipe``. Memoized on the pending object: when several sibling pipes
    share one capture (same key), the first proven candidate packs and
    publishes; later siblings receive the same finalized identity without
    re-packing (the pack runs after the WHOLE warmup, so it already holds
    every sibling's graphs).

    Packing failure never un-serves the request (``pipe``'s compiled
    callables are already live in-process); it only means this boot cannot
    advertise/publish a cell, so the caller must treat a ``None`` return
    the same as a disproven candidate (unwrap, and fail closed for
    mandatory lanes — never advertise or publish an artifact nothing
    proved).
    """
    state = pending._state
    if "minted" in state:
        return state["minted"]  # sibling already finalized (or failed: None)

    try:
        meta = cc.finish_fleet_mint(
            pipe, pending.cfg, pending.family, pending.target,
            pending.capture_dir)
    except Exception as exc:  # noqa: BLE001 — pack failure => caller disproves
        logger.warning(
            "fleet-cells: self-mint pack failed after a passed proof (%s) — "
            "the compiled callables stay live for this process, but this "
            "boot cannot advertise or publish a cell", exc)
        state["minted"] = None
        _unregister(pending)
        shutil.rmtree(pending.mint_root, ignore_errors=True)
        return None

    from .convert.hub import blake3_file

    key = str(meta.get("cell_key") or "").strip() or pending.cell_key
    minted = SelfMint(
        family=pending.family, cell_key=key,
        ref=f"{cc.system_repo(pending.family)}#{key}",
        snapshot_digest="blake3:" + blake3_file(pending.target),
        artifact=pending.target,
    )
    state["minted"] = minted
    _unregister(pending)
    logger.info(
        "fleet-cells: self-mint proof passed for %s (key=%s, %.1f MB) — "
        "serving compiled, publishing",
        pending.family, key, pending.target.stat().st_size / 1e6)

    # Hygiene: fold the proven capture into the live compile-cache root and
    # re-point inductor there (the same end state the delivered-cell adoption
    # path leaves), so later boots/adoptions in this process are not aimed at
    # the soon-to-be-deleted temp capture dir. Best-effort — the in-process
    # compiled callables never depend on it.
    try:
        live_root = (
            Path(pending.cache_dir) if pending.cache_dir
            else Path.home() / ".cache" / "gen-worker") / "compile-cache"
        with cc._SEED_ARM_LOCK:
            cc._merge_staged_cache(pending.capture_dir, live_root)
            cc.seed_env(live_root)
    except Exception:
        logger.debug(
            "fleet-cells: live-cache fold of the proven capture failed",
            exc_info=True)

    # Serve first, publish behind (gw#587: publish failure never blocks the
    # request that triggered the miss). The hub's attested gate decides
    # accept/refuse; cozy-local never reaches here (no publisher).
    publisher = pending.publisher
    if publisher is not None and publisher.enabled():
        _publish_async(publisher, pending.family, pending.target, meta)
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
        shutil.rmtree(pending.mint_root, ignore_errors=True)
    return minted


def abandon_self_mint(pending: "PendingSelfMint") -> None:
    """Discard a self-mint capture the proof did not certify (disproven or
    genuinely unexercised with no proven sibling). Never packed, never
    published — only the temp capture dir is cleaned up. A no-op when a
    proven sibling already finalized the shared capture (the artifact and
    its publish must survive)."""
    if pending._state.get("minted") is not None:
        return
    _unregister(pending)
    shutil.rmtree(pending.mint_root, ignore_errors=True)


def _unregister(pending: "PendingSelfMint") -> None:
    with _PENDING_LOCK:
        if _PENDING.get(pending.cell_key) is pending:
            del _PENDING[pending.cell_key]


def _fail_closed(pipe: Any, reason: str) -> ArmOutcome:
    """The quantized-lane policy at every exit that cannot produce a cell:
    plain lanes serve eager (never-raise miss policy), w8a8/w4a4 keep the
    typed refusal (same as the cozy-local store / pre-gw#587 production)."""
    from .models.loading import pipeline_weight_lane

    lane = pipeline_weight_lane(pipe)
    if lane.startswith(("w8a8", "w4a4")):
        raise cc.CompiledLaneUnavailableError(
            f"{lane[:4].upper()} requires a compile cell and the self-mint "
            f"is unavailable ({reason})")
    logger.info("fleet-cells: serving eager (%s)", reason)
    return ArmOutcome(armed=False)


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
    "enable_compiled",
    "finalize_self_mint",
]
