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
  2. MISS: compile locally over the declared shape table (the mint brain
     shared with cozy-local, ``compile_cache.mint_artifact``), adopt the
     just-minted artifact, and serve COMPILED — never eager, never a
     fail-closed wait. The cold-compile tax is paid once fleet-wide
     instead of silently on every boot.
  3. PUBLISH (best-effort, in the background): ship the packed cell
     through the hub's attested publish gate (th#910) so the next worker
     on this key is store-served. Publish failure NEVER affects serving —
     the local adoption already happened; a refusal (untrusted tier,
     attestation, quota) is the hub's call and is fully recorded hub-side.

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

import logging
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from . import compile_cache as cc
from .models import provision

logger = logging.getLogger(__name__)

_INTENT_TIMEOUT_S = 30
_COMPLETE_TIMEOUT_S = 30


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
) -> bool:
    """Fleet arming policy (gw#587): delivered cell first, self-mint on miss.

    Replaces the executor's bare ``provision.enable_compiled`` call. HIT
    keeps today's semantics exactly (incl. ``CellSelectionBugError``
    propagation — the th#883 receipt invariant is untouched). MISS
    self-mints and serves compiled; the ONLY remaining eager/fail-closed
    exits are genuine mint impossibilities (no CUDA, no C toolchain, the
    mint itself failing), where plain lanes serve eager and quantized lanes
    keep their typed refusal — exactly the cozy-local store policy.
    """
    family = str(getattr(cfg, "family", "") or "")
    try:
        if provision.enable_compiled(pipe, cfg, cache_dir, artifact):
            return True
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

    mint_root = Path(tempfile.mkdtemp(prefix="selfmint-"))
    label = cc.flavor_label(
        cc.runtime_key()["sku"], cc.runtime_key()["torch"],
        pipeline_weight_lane(pipe))
    target = mint_root / f"{label}.tar.gz"
    started = time.monotonic()
    try:
        meta = cc.mint_artifact(pipe, cfg, family, target, mint_root / "capture")
    except Exception as exc:  # noqa: BLE001 — mint failure => miss policy
        logger.warning("fleet-cells: self-mint failed (%s)", exc)
        cc.unwrap(pipe)
        if bucket:
            cc.drop_lora_lane(pipe)
        return _fail_closed(pipe, f"self-mint failed: {exc}")
    logger.info(
        "fleet-cells: self-minted %s cell in %.0fs (%.1f MB) — adopting + publishing",
        family, time.monotonic() - started, target.stat().st_size / 1e6)

    # Adopt the just-minted cell through the delivered-cell path (drops the
    # unguarded mint wrappers; re-traces hit the captured FX cache). This
    # also proves the artifact round-trips before anyone else can pull it.
    cc.unwrap(pipe)
    try:
        armed = cc.enable(pipe, cfg, cache_dir, artifact=target)
    except cc.CellSelectionBugError:
        if bucket:
            cc.drop_lora_lane(pipe)
        raise
    except cc.CompiledLaneUnavailableError as exc:
        logger.warning("fleet-cells: minted cell failed re-adoption (%s)", exc)
        if bucket:
            cc.drop_lora_lane(pipe)
        raise
    if not armed:
        if bucket:
            cc.drop_lora_lane(pipe)
        return _fail_closed(pipe, "minted cell failed re-adoption")

    # Serve first, publish behind (gw#587: publish failure never blocks the
    # request that triggered the miss). The hub's attested gate decides
    # accept/refuse; cozy-local never reaches here (no publisher).
    if publisher is not None and publisher.enabled():
        _publish_async(publisher, family, target, meta)
    else:
        # Runtime assertion (gw#587): every fleet cell miss must produce a
        # publish attempt. A fleet worker minting with no usable sink is a
        # wiring defect (file_base_url/worker JWT absent at arming time),
        # not a policy choice — loud, greppable, alarm-adjacent. (cozy-local
        # legitimately has no publisher, but it never enters this module.)
        logger.warning(
            "fleet-cells: SELF_MINT_WITHOUT_PUBLISH_SINK family=%s — cell "
            "stays local to this pod; the fleet store gains nothing", family)
    return True


def _fail_closed(pipe: Any, reason: str) -> bool:
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
    return False


def _cuda_ready() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


__all__ = [
    "CellPublishRefused",
    "CellPublisher",
    "enable_compiled",
]
