"""Tensorhub publish client — the ONE publish path, CHUNKED SHA-256 (th#1303).

  POST /api/v1/repos/{org}/{name}/publishes
      {files: [{path, size_bytes, digest:"sha256:<hex>", chunks?}], mode,
       tags, flavor/dtype/file_layout/file_type, metadata, provenance, ...}
  → {publish_id, have: [...], need: [{digest, put_url, headers, ...}]}

Then PUT every `need` grant VERBATIM (the sha256 is signed into the presigned
URL, so the store itself refuses bytes that do not hash to the key), re-plan
`.../grants` to resume, and POST `.../complete`. One publish == one checkpoint
== one flavor.

THE v1 (blake3) `/commits` PROTOCOL IS GONE FROM THIS CLIENT (pgw#807). It was
frozen hub-side (th#1303 phase 3.5 — every new v1 commit answers 410
`unsupported_digest_algorithm`), and a retired protocol left resident in the
tree is a runtime failure on a rented pod instead of a compile failure in CI.
Deleted with it: `blake3_file`, `CommitFile.resolve`/`.blake3`, the
by-reference add, `BankedBlobGoneError`, and the th#592 download-skip bank
(`lookup_clone_manifests` / `record_clone_manifests`), whose adds were
by-reference and therefore un-migratable by construction.
"""

from __future__ import annotations

import json
import logging
import random
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import requests
import socket
from ..api.errors import AuthError
from ..request_context._helpers import _parse_owner_repo
from ..stall import SilenceWindow
from gen_worker.models.refs import flavor_token as _token
from .. import activity as _activity
from ..http_origin import is_definite_hub_answer
from ..models import chunk_upload as _cu
from ..models.chunk_upload import UploadGrant

logger = logging.getLogger(__name__)

_RETRY_BASE_DELAY_S = 1.0
_RETRY_MAX_DELAY_S = 30.0

# Connect timeout split from read (mirrors gen_worker._upload_transport and
# the gw#456 download-side floor): a dead host fails in seconds instead of
# consuming the whole read budget.
_CONNECT_TIMEOUT_S = 15.0

# Bounded RE-PLAN passes for a publish whose granted objects did not all land.
# Resume needs no client state: the need set comes back smaller because what
# landed is now resident, so a pass costs only the objects still missing.
_REUPLOAD_ATTEMPTS = 2

# tensorhub's /complete verifies the publish synchronously before answering.
# For a large tree that can outlast whatever timeout sits in front of the hub,
# and the client must not read its own impatience as a failure.
_COMPLETE_TIMEOUT_S = 600.0

# pgw#738/#743 (gw#666): how long the client tolerates hearing NOTHING
# DEFINITE from the hub before giving up — network errors, edge-masked 5xx,
# proxy-shaped 404s. Silence-bounded, never attempt-counted: a hub restart is
# seconds and a tunnel re-dial is minutes, and the old 5-attempt cap (~2 min)
# classified both FATAL and threw away paid GPU work at the finish line (two
# 58-minute clones, pgw#743). Six verify-lengths, because a container rebuild
# is tens of minutes and an hour parked on the CPU rig costs about what
# re-downloading costs — and unlike the re-download it cannot fail the same
# way twice. Waiting stays observable: the loop beats liveness every pass.
_COMPLETE_SILENCE_WINDOW_S = 6.0 * _COMPLETE_TIMEOUT_S

_SESSION: Optional[requests.Session] = None


def _http_session() -> requests.Session:
    """Shared session with TCP keepalives so NAT/conntrack middleboxes don't
    evict the idle flow while the server verifies (no response bytes for
    minutes)."""

    global _SESSION
    if _SESSION is not None:
        return _SESSION
    socket_options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    for name, value in (("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 30), ("TCP_KEEPCNT", 20)):
        if hasattr(socket, name):
            socket_options.append((socket.IPPROTO_TCP, getattr(socket, name), value))

    class _KeepaliveAdapter(requests.adapters.HTTPAdapter):
        def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
            kwargs["socket_options"] = socket_options
            super().init_poolmanager(*args, **kwargs)

    session = requests.Session()
    adapter = _KeepaliveAdapter()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    _SESSION = session
    return session


class HubPublishError(RuntimeError):
    """Terminal failure talking to tensorhub's commit API.

    ``status`` and ``code`` carry the hub's OWN classification when it gave
    one (the ``{"error": {"code": ...}}`` envelope, or a th#1301 projection's
    ``status.failure.code``). Callers that report a publish outcome to the
    fleet need a stable token to put on the wire; re-deriving one by matching
    substrings of ``str(exc)`` is how a refusal reason turns into prose that
    nothing can group by. "" / 0 honestly mean "the hub named nothing".
    """

    def __init__(self, message: str, *, status: int = 0, code: str = "",
                 retryable: Optional[bool] = None) -> None:
        super().__init__(message)
        self.status = int(status or 0)
        self.code = str(code or "")
        self.retryable = retryable


def _retry_after_s(resp: requests.Response) -> Optional[float]:
    try:
        value = float(str(resp.headers.get("Retry-After") or "").strip())
    except Exception:
        return None
    return min(value, _RETRY_MAX_DELAY_S) if value > 0 else None


def _error_code_of(resp: requests.Response) -> str:
    """Best-effort extraction of the structured `error.code` field
    (docs/api-conventions.md: `{"error": {"code": ..., ...}}`); "" if the
    body isn't that shape."""
    try:
        body = resp.json() if resp.text else {}
    except Exception:
        return ""
    if not isinstance(body, dict):
        return ""
    err = body.get("error")
    if isinstance(err, dict):
        return str(err.get("code") or "")
    # pgw#987: the publish envelope (`publishError.body()`) and gin's
    # `AbortWithStatusJSON` both emit the code as a STRING. Dropping it left
    # `_publish_failure_phase` with only `http_413` to group 32 identical
    # refusals by, when the hub had named the fault exactly.
    if isinstance(err, str) and err.strip():
        return err.strip()
    return ""


# pgw#738/#743: how long _send_with_retries tolerates hearing nothing
# DEFINITE from the hub (network errors, 5xx, 429, proxy-shaped 404s) before
# giving up. A hub restart is seconds and a tunnel re-dial is minutes; the old
# 5-attempt cap (~2 min) classified both FATAL and threw away paid GPU work
# at the finish line (two 58-min clones, #743). Silence-bounded per gw#666.
_SEND_SILENCE_WINDOW_S = 600.0


def _send_with_retries(
    what: str,
    send: Callable[[], requests.Response],
    *,
    silence_window_s: Optional[float] = None,
    definite: Optional[Callable[[requests.Response], bool]] = None,
) -> requests.Response:
    """Origin-discriminating, silence-bounded retry (pgw#715/#738/#743).

    Only a DEFINITE hub answer ends the loop: 2xx/3xx, or a 4xx that is
    really the hub speaking (not a proxy's offline page). Indefinite answers
    — network errors, 429, 5xx, and ANY proxy-shaped status (ngrok with no
    healthy backend while the hub restarts answers 404 AND 503 with the same
    HTML page; pgw#743) — retry with backoff until nothing definite has been
    heard for ``silence_window_s``, then surface the last state. Unknown
    exception types never leak raw: they become a typed ``HubPublishError``.
    Returns the last response for non-retryable statuses — callers keep their
    own status handling.
    """

    if silence_window_s is None:
        silence_window_s = _SEND_SILENCE_WINDOW_S
    contact = SilenceWindow(silence_window_s)
    delay = _RETRY_BASE_DELAY_S
    attempt = 0
    last_resp: Optional[requests.Response] = None
    last_exc: Optional[BaseException] = None
    while True:
        attempt += 1
        try:
            resp = send()
        except requests.RequestException as exc:
            last_resp, last_exc = None, exc
        except Exception as exc:  # noqa: BLE001 - typed terminal, never a raw leak
            raise HubPublishError(
                f"{what} failed ({type(exc).__name__}): {exc}") from exc
        else:
            code = int(resp.status_code)
            # `definite` lets a caller supply its OWN definiteness rule for a
            # route whose refusals are not envelope-shaped (th#1303's v2
            # completion answers with a projection carrying an explicit
            # `retryable` flag). The default stays the shape heuristic.
            is_definite = definite(resp) if definite is not None else is_definite_hub_answer(resp)
            if code != 429 and is_definite:
                return resp  # definite hub answer
            last_resp, last_exc = resp, None
            delay = _retry_after_s(resp) or delay
        # pgw#743: this loop can now legitimately run for the better part of
        # an hour riding out a hub rebuild, and a publisher that goes silent
        # for an hour is exactly the dead-job signature the watchdogs kill on
        # (pgw#738). Waiting IS work — say so on the liveness channel.
        _activity.note_progress()
        if contact.stalled():
            if last_resp is not None:
                return last_resp
            raise HubPublishError(
                f"{what} failed (network, no definite hub answer for "
                f"{contact.silent_for():.0f}s): {last_exc}") from last_exc
        logger.warning(
            "%s retrying (attempt %d, %s; no definite answer for %.0fs of %.0fs)",
            what, attempt,
            (f"status={last_resp.status_code}" if last_resp is not None
             else f"err={type(last_exc).__name__}"),
            contact.silent_for(), contact.window_s)
        time.sleep(delay + random.uniform(0, delay * 0.1))
        delay = min(delay * 2, _RETRY_MAX_DELAY_S)


@dataclass
class CommitFile:
    """One file to publish: repo path + the LOCAL BYTES.

    There is no by-reference form. v2's guarantee is that a digest is proven
    from bytes in hand — a caller who cannot supply them has nothing the
    protocol can attest, and the th#592 by-reference bank that used to rely on
    the opposite died with the v1 route it was built on."""

    path: str
    local_path: Optional[Path] = None
    size_bytes: int = 0


@dataclass
class CommitResult:
    revision_id: str
    uploaded: int
    deduped: int
    total_bytes: int
    # Content-addressed checkpoint id minted at finalize (tensorhub derives it
    # from the snapshot manifest); THE id for tree/lineage queries. The
    # revision_id above is the upload-session id, not queryable post-finalize.
    checkpoint_id: str = ""
    response: dict[str, Any] = field(default_factory=dict)


class HubClient:
    """Thin client over tensorhub's `/commits` API for one destination repo."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        owner: str,
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.token = str(token or "").strip()
        self.owner = str(owner or "").strip()
        self.timeout_s = timeout_s
        if not self.base_url or not self.token:
            raise HubPublishError("missing tensorhub base URL or capability token")

    @classmethod
    def from_ctx(cls, ctx: Any) -> "HubClient":
        """Build from a gen_worker RequestContext (cap-token identity)."""
        base = str(getattr(ctx, "_file_api_base_url", "") or "").strip()
        token = str(getattr(ctx, "_worker_capability_token", "") or "").strip()
        owner = str(getattr(ctx, "owner", "") or getattr(ctx, "_owner", "") or "").strip()
        return cls(base_url=base, token=token, owner=owner)

    # ---- internals ----

    def _headers(self) -> dict[str, str]:
        h = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        if self.owner:
            h["X-Cozy-Owner"] = self.owner
        return h

    def _repo_path(self, destination_repo: str) -> str:
        owner, _, name = str(destination_repo).partition("/")
        if not owner or not name:
            raise HubPublishError(f"destination_repo must be owner/repo, got {destination_repo!r}")
        return (
            f"/api/v1/repos/{urllib.parse.quote(owner, safe='')}/"
            f"{urllib.parse.quote(name, safe='')}"
        )

    def _post(self, path: str, payload: Optional[dict] = None, *, timeout: float | None = None) -> requests.Response:
        return _send_with_retries(f"POST {path}", lambda: _http_session().post(
            f"{self.base_url}{path}",
            headers=self._headers(),
            data=json.dumps(payload) if payload is not None else None,
            timeout=(_CONNECT_TIMEOUT_S, timeout or self.timeout_s),
        ))

    @staticmethod
    def _json(resp: requests.Response) -> dict[str, Any]:
        try:
            out = resp.json() if resp.text else {}
        except Exception:
            out = {}
        return out if isinstance(out, dict) else {}

    @staticmethod
    def _v2_failure(resp: requests.Response) -> Optional[dict[str, Any]]:
        """The v2 routes' TYPED failure, or None.

        A v2 completion does not answer with an error envelope: it answers with
        the th#1301 PROJECTION, and a refusal is
        ``{"status": {"stage": "repudiated", "terminal": true,
                      "failure": {"code", "retryable", "message"}}}``.
        That `retryable` bit is the hub's own classification — the whole point
        of th#1301's retryable-vs-terminal split — so it is read, never guessed
        at from the body's shape.
        """
        try:
            body = resp.json() if resp.text else {}
        except Exception:  # noqa: BLE001 - an unparseable body is not a verdict
            return None
        if not isinstance(body, dict):
            return None
        status = body.get("status")
        if not isinstance(status, dict):
            return None
        failure = status.get("failure")
        if not isinstance(failure, dict) or not failure.get("code"):
            return None
        return failure

    def _post_v2_complete(self, path: str) -> requests.Response:
        """POST /complete WITHOUT the envelope-guessing retry loop.

        `_send_with_retries` decides "did we hear from the hub?" from the body's
        SHAPE, and a v2 completion's refusal is a projection rather than an
        error envelope — so it read a real, typed, `retryable: false` refusal as
        a proxy non-answer and retried it. MEASURED LIVE (twice): the retry
        found the session already terminal and returned 409
        `publish_repudiated`, so the caller was told the publish had been
        repudiated instead of `invalid_manifest_for_kind:
        missing_diffusers_single_file_safetensors` — a consequence in place of a
        cause, and the actual reason was gone.

        A publish is not idempotent to re-complete, so a blind retry here can
        only ever destroy the diagnosis. Network-level failures still retry
        (nothing was heard); an HTTP answer of any status is the hub speaking
        and is returned as-is for the typed handling below.
        """
        return _send_with_retries(
            f"POST {path}",
            lambda: _http_session().post(
                f"{self.base_url}{path}", headers=self._headers(),
                timeout=(_CONNECT_TIMEOUT_S, _COMPLETE_TIMEOUT_S),
            ),
            definite=lambda resp: True,
        )

    def publish_v2(
        self,
        *,
        destination_repo: str,
        files: list[CommitFile],
        tags: list[str] | None = None,
        # th#1400: "replace" — a checkpoint is COMPLETE IN ITSELF. "merge"
        # unions this publish with the repo's prior :latest, so a caller that
        # never mentioned a sibling inherits its bytes; te#44 shipped an #fp8
        # checkpoint carrying 5.2 GB of fp16 base weights that way, and a
        # differently-sharded base splices into a quantization the same way.
        # te#44 was fixed at ONE call site (publish_flavors' own default)
        # instead of here, so every future caller re-acquired the bug — the
        # hub's normalizePublishMode("") had the same default until th#1400.
        # Both are "replace" now; pass mode="merge" explicitly and only for
        # what it is for: assembling ONE checkpoint across several commits
        # (clone.py's chunked full-clone, _stream.py's streamed output).
        mode: str = "replace",
        flavor: str = "",
        flavors: list[str] | None = None,
        default_flavor: str = "",
        dtype: str = "",
        file_layout: str = "",
        file_type: str = "",
        display_label: str = "",
        objective: str = "",
        distilled: Optional[bool] = None,
        required_paths: list[str] | None = None,
        declared_tensors: Mapping[str, Any] | None = None,
        deletions: list[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        repo_spec: Mapping[str, str] | None = None,
        progress: Any = None,
        part_progress: Optional[Callable[[int, int, int], None]] = None,
        on_stage: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> CommitResult:
        """Publish one checkpoint over the CHUNKED SHA-256 CAS (th#1303).

        Declare -> `{have, need}` -> PUT the needed objects -> complete. What
        makes this different from :meth:`commit` is not the transport but WHO
        GUARANTEES INTEGRITY: each object's sha256 is signed into its presigned
        PUT, so R2 refuses bytes that do not hash to the key (400, and the
        object does not exist afterwards) and refuses a substituted claim (403).
        A claimed digest stops being assertable, which is what kills th#1305's
        inherit/overwrite class structurally rather than by guard.

        Files > 64 MiB become an ordered list of enforced chunks, so a retry
        costs one chunk instead of a 2 GB shard and progress is monotone across
        retries — the pgw#786 lemon-host pathology, fixed by the unit of work
        rather than by a new watchdog.

        THERE IS NO PROTOCOL AUTO-SELECT AND NO ENV KNOB. The caller names the
        protocol, so flipping a producer class to sha256 is a code change and a
        deploy — which is exactly the controlled, blast-radius-ordered producer
        flip th#1303 phase 3 asks for. A hub without these routes answers 404,
        and this deliberately does NOT fall back: a silent downgrade to blake3
        would make "did this publish v2?" unanswerable from the outside (and a
        404 from a PROXY is not a 404 from the hub — th#715).

        ``provenance`` carries the WORKER-ADDABLE stamp subset only, exactly as
        :meth:`commit` does — the hub resolves the authoritative stamp from the
        capability token (th#606/th#1331), so lineage cannot be forged from here
        and is identical whichever protocol published it.

        ``on_stage(stage, facts)`` reports the protocol's own legs —
        ``declared`` (publish_id + the have/need split, i.e. what dedup
        actually saved), ``uploading`` (objects and bytes this pass will
        move), ``committing`` — so a caller whose publish is otherwise a
        silent background thread can put the LEG on the wire instead of only
        its terminus. Never load-bearing: a raising callback is the caller's
        bug and must not fail a publish that is transferring correctly.
        """

        # Read the constant off the module at CALL time rather than binding a
        # default: it must equal the hub's `storage.CASChunkSizeBytes`, and a
        # call-time read keeps that single source of truth substitutable in
        # tests without exposing a tuning knob on this method.
        chunk_size = _cu.CAS_CHUNK_SIZE_BYTES

        if not files:
            raise HubPublishError("publish_v2 requires at least one file")

        repo_path = self._repo_path(destination_repo)

        # ONE streaming pass per file yields the whole-file sha256 AND every
        # per-chunk sha256. A by-reference add has no local bytes and therefore
        # cannot be declared under a protocol whose whole point is that the
        # digest is proven from the bytes.
        decls = []
        sources: dict[str, tuple[Path, int, int]] = {}
        for f in files:
            if f.local_path is None:
                raise HubPublishError(
                    f"publish_v2: {f.path!r} is a by-reference add; v2 declares "
                    "digests computed from local bytes (use commit() or supply bytes)"
                )
            local = Path(f.local_path)
            decl = _cu.hash_file_and_chunks(
                local, chunk_size=chunk_size, rel_path=f.path)
            decls.append(decl)
            if decl.chunks:
                for c in decl.chunks:
                    sources["sha256:" + c.sha256] = (local, c.offset, c.length)
            else:
                sources[decl.digest] = (local, 0, decl.size_bytes)

        body: dict[str, Any] = {
            "mode": mode,
            "files": [d.to_wire() for d in decls],
        }
        # th#1411: an EMPTY list is an explicit "move no tags" and must reach
        # the wire (the classification gate refuses an OMITTED tags field on
        # repos that carry tag rows); only None omits the field.
        if tags is not None:
            df = _token(default_flavor)
            body["tags"] = [
                {"tag": t, **({"default_flavor": df} if df else {})} for t in tags
            ]
        for key, val in (
            ("flavor", _token(flavor)), ("default_flavor", _token(default_flavor)),
            ("dtype", _token(dtype)), ("file_layout", file_layout),
            ("file_type", file_type), ("display_label", display_label),
            ("objective", objective),
        ):
            if val:
                body[key] = val
        if flavors:
            body["flavors"] = [_token(f) for f in flavors]
        if distilled is not None:
            body["distilled"] = bool(distilled)
        if required_paths:
            body["required_paths"] = list(required_paths)
        if declared_tensors:
            body["declared_tensors"] = dict(declared_tensors)
        if deletions:
            body["deletions"] = list(deletions)
        if metadata:
            body["metadata"] = dict(metadata)
        if provenance:
            # th#1331: the v2 declare decodes this STRICTLY into the same
            # `commitProvenanceInput` v1 uses, so the accepted subset is exactly
            # `step_number | epoch_number | quantization_method |
            # quantization_library | upstream_revision | upstream_attestation`.
            # `parents` / `derivation_op` / `upstream_ref` / `job_id` are
            # orchestrator-derived (signed into the capability token) and are a
            # 400 that NAMES the field — deliberately, because a silent drop
            # would leave the worker believing it had stamped lineage it had
            # not. The merged stamp is captured at DECLARE, where the token is
            # presented, and replayed through `resolvePublishProvenance` at
            # completion, so v1 and v2 mirrors record identical lineage.
            body["provenance"] = {k: v for k, v in dict(provenance).items() if v}
        for key in ("kind", "library_name", "model_family", "class_name",
                    "adapter_for_family"):
            val = str((repo_spec or {}).get(key) or "").strip()
            if val:
                body[key] = val

        resp = self._post(f"{repo_path}/publishes", body)
        if resp.status_code < 200 or resp.status_code >= 300:
            raise HubPublishError(
                f"publish declare failed ({resp.status_code}): {resp.text[:800]}",
                status=resp.status_code, code=_error_code_of(resp))
        session = self._json(resp)
        publish_id = str(session.get("publish_id") or "").strip()
        if not publish_id:
            raise HubPublishError("publish response missing publish_id",
                                  code="publish_id_missing")

        distinct = int(session.get("distinct_objects") or 0)
        resident = int(session.get("resident_objects") or 0)
        uploaded_objects = 0
        total_bytes = sum(d.size_bytes for d in decls)

        def _stage(stage: str, **facts: Any) -> None:
            if on_stage is None:
                return
            try:
                on_stage(stage, dict(facts))
            except Exception:  # noqa: BLE001 — reporting never fails a publish
                logger.debug("publish_v2 on_stage(%s) raised", stage, exc_info=True)

        def _grants_of(payload: Mapping[str, Any]) -> list["UploadGrant"]:
            out = []
            for g in payload.get("need") or []:
                if not isinstance(g, dict):
                    continue
                out.append(UploadGrant(
                    digest=str(g.get("digest") or "").strip().lower(),
                    size_bytes=int(g.get("size_bytes") or 0),
                    put_url=str(g.get("put_url") or ""),
                    headers={str(k): str(v) for k, v in (g.get("headers") or {}).items()},
                    staging_key=str(g.get("staging_key") or ""),
                ))
            return out

        def _source_for(digest: str) -> tuple[Path, int, int]:
            span = sources.get(digest.strip().lower())
            if span is None:
                raise ValueError(
                    f"hub granted {digest[:20]}… which this publish never declared"
                )
            return span

        try:
            grants = _grants_of(session)
            _stage("declared", publish_id=publish_id, objects=distinct,
                   resident=resident, need=len(grants), bytes=total_bytes)
            for attempt in range(_REUPLOAD_ATTEMPTS + 1):
                if not grants:
                    break
                _stage("uploading", publish_id=publish_id, objects=len(grants),
                       bytes=sum(g.size_bytes for g in grants), attempt=attempt)
                report = _cu.upload_grants(
                    grants, _source_for,
                    on_bytes=(lambda n: part_progress(0, 0, n))
                    if callable(part_progress) else None,
                )
                uploaded_objects += report.uploaded
                if callable(progress):
                    progress(resident + uploaded_objects, distinct or len(grants))
                if report.ok:
                    break
                if attempt == _REUPLOAD_ATTEMPTS:
                    raise HubPublishError(
                        f"publish {publish_id}: {len(report.failures)} object(s) failed to "
                        f"upload after {_REUPLOAD_ATTEMPTS + 1} passes: "
                        + "; ".join(report.failures[:5])
                    )
                # RESUME NEEDS NO CLIENT STATE: re-plan and the need set comes
                # back smaller, because what landed is now resident. A kill
                # mid-upload costs the in-flight objects and nothing else.
                again = self._post(f"{repo_path}/publishes/"
                                   f"{urllib.parse.quote(publish_id, safe='')}/grants")
                if again.status_code < 200 or again.status_code >= 300:
                    raise HubPublishError(
                        f"publish re-plan failed ({again.status_code}): {again.text[:500]}",
                        status=again.status_code, code=_error_code_of(again))
                grants = _grants_of(self._json(again))

            _stage("committing", publish_id=publish_id, objects=distinct,
                   resident=resident, uploaded=uploaded_objects)
            done = self._post_v2_complete(
                f"{repo_path}/publishes/{urllib.parse.quote(publish_id, safe='')}/complete")
        except Exception:
            # Abort so the staging prefix is reclaimed and the session lands in
            # a TERMINAL state rather than looking forever in-flight.
            try:
                _http_session().delete(
                    f"{self.base_url}{repo_path}/publishes/"
                    f"{urllib.parse.quote(publish_id, safe='')}",
                    headers=self._headers(), timeout=30,
                )
            except Exception:
                pass
            raise

        if done.status_code < 200 or done.status_code >= 300:
            # Lead with the hub's OWN typed classification when it gave one. A
            # refusal that reports its `code`, its `retryable` bit and the stage
            # that produced it is actionable; a bare status code plus 800 bytes
            # of truncated projection is what sent this lane looking in the
            # wrong place twice.
            failure = self._v2_failure(done)
            if failure:
                stage = ""
                try:
                    for s in (done.json().get("status") or {}).get("stages") or []:
                        if s.get("status") == "failed":
                            stage = str(s.get("stage") or "")
                except Exception:  # noqa: BLE001 - the stage is a nicety
                    pass
                raise HubPublishError(
                    f"publish {publish_id} {'repudiated' if not failure.get('retryable') else 'failed'}"
                    f"{f' at {stage}' if stage else ''}: "
                    f"{failure.get('code')}: {failure.get('message')} "
                    f"(retryable={bool(failure.get('retryable'))})",
                    status=done.status_code, code=str(failure.get("code") or ""),
                    retryable=bool(failure.get("retryable")),
                )
            raise HubPublishError(
                f"publish complete failed ({done.status_code}): {done.text[:800]}",
                status=done.status_code, code=_error_code_of(done))
        final = self._json(done)
        ckpt = final.get("checkpoint") if isinstance(final.get("checkpoint"), dict) else {}
        checkpoint_id = str((ckpt or {}).get("checkpoint_id") or "").strip()
        if not checkpoint_id:
            raise HubPublishError(
                f"publish {publish_id} completed without a checkpoint id: "
                f"{json.dumps(final)[:500]}", code="checkpoint_id_missing")
        # Surfaced, not swallowed: the hub names which canonical th#1301 checks
        # it did NOT run. A list that promises 19 and silently runs 14 is worse
        # than one promising 14.
        skipped = final.get("checks_unavailable") or []
        if skipped:
            logger.info("publish %s checks_unavailable=%s", publish_id, skipped)
        logger.info(
            "publish_v2 done repo=%s publish=%s objects=%d resident=%d uploaded=%d "
            "checkpoint=%s",
            destination_repo, publish_id, distinct, resident, uploaded_objects,
            checkpoint_id[:24],
        )
        return CommitResult(
            revision_id=publish_id,
            uploaded=uploaded_objects,
            deduped=resident,
            total_bytes=total_bytes,
            checkpoint_id=checkpoint_id,
            response=final,
        )


_DATASET_PAGE_LIMIT = 200  # tensorhub's cap; a larger value is clamped there.
_DATASET_PAGE_CEILING = 1000  # pages, i.e. 200k datasets — a runaway guard.


def _find_dataset_id(
    base: str, headers: dict[str, str], owner: str, name: str,
) -> str:
    """dataset_id of ``owner/name``, or "" when it genuinely does not exist.

    Raises ``AuthError``/``RuntimeError`` rather than returning "" on any
    response it cannot READ — the caller creates on "", so a swallowed error
    is a duplicate dataset (pgw#656).
    """
    wanted = name.lower()
    cursor = ""
    seen_pages = 0
    while True:
        query = {"org": owner, "limit": str(_DATASET_PAGE_LIMIT)}
        if cursor:
            query["cursor"] = cursor
        resp = requests.get(
            f"{base}/api/v1/datasets?{urllib.parse.urlencode(query)}",
            headers=headers,
            timeout=30,
        )
        if resp.status_code in (401, 403):
            raise AuthError(f"dataset list unauthorized ({resp.status_code})")
        if resp.status_code < 200 or resp.status_code >= 300:
            raise RuntimeError(
                f"dataset list failed ({resp.status_code}): {resp.text[:256]}"
            )
        try:
            body = resp.json() if resp.text else {}
            items = list(body.get("items") or [])
        except (ValueError, AttributeError) as exc:
            raise RuntimeError(
                f"dataset list returned unreadable JSON: {exc}"
            ) from exc
        for it in items:
            if str(it.get("name") or "").lower() == wanted:
                return str(it.get("dataset_id") or "")
        next_cursor = str(body.get("next_cursor") or "").strip()
        seen_pages += 1
        # No cursor, no movement, or a server that ignores paging: stop. Never
        # spin — but never silently truncate the search either.
        if not next_cursor or next_cursor == cursor or not items:
            return ""
        if seen_pages >= _DATASET_PAGE_CEILING:
            raise RuntimeError(
                f"dataset list did not terminate after {seen_pages} pages "
                f"(org={owner}); refusing to create a possible duplicate"
            )
        cursor = next_cursor


def publish_dataset_revision(
    *,
    base_url: str,
    token: str,
    destination_dataset: str,
    features_json: dict[str, Any],
    row_artifacts_json: Optional[dict[str, Any]] = None,
    snapshot_manifest: Optional[list[dict[str, Any]]] = None,
    visibility: str = "private",
    kind: str = "",
    dataset_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Publish a dataset revision into ``tensorhub.datasets``.

    Hub-API plumbing for ``DatasetContext.publish_dataset_revision`` (which
    documents the tenant-facing contract). The flow:

    1. Resolve ``destination_dataset`` (owner/name) against tensorhub — the
       FULL cursor-paginated listing, and any unreadable response raises
       rather than reading as "no such dataset" (pgw#656: every silent miss
       here becomes a duplicate dataset in step 2).
    2. If the dataset row doesn't exist: ``POST /api/v1/datasets`` with
       ``{tenant, name, visibility, schema: features_json}``.
    3. Otherwise: ``PATCH /api/v1/datasets/:id`` to update the schema
       (+ row_artifacts / visibility).

    ``kind`` / ``dataset_info`` / ``snapshot_manifest`` ride inside
    ``features_json`` under reserved ``__cozy_*__`` keys until tensorhub
    grows dedicated columns (th#1162). Raises ``AuthError`` on 401/403 and
    ``RuntimeError`` on any other HTTP failure.
    """

    owner, name = _parse_owner_repo(destination_dataset)
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("publish_dataset_revision: no file_api_base_url")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Cozy-Owner": owner,
    }

    # Stash the kind (+ dataset_info if provided) inside features_json
    # using a reserved `__cozy_*__` key so it survives through the
    # server's features_json passthrough. Once tensorhub adds a
    # dedicated `kind` column, migrate these to top-level fields.
    raw_schema = dict(features_json or {})
    if isinstance(raw_schema.get("features"), dict):
        features_payload = dict(raw_schema)
    else:
        features_payload = {"features": raw_schema}
    if kind:
        features_payload["__cozy_kind__"] = kind
    if dataset_info:
        features_payload["__cozy_dataset_info__"] = dataset_info
    if snapshot_manifest:
        features_payload["__cozy_snapshot_manifest__"] = snapshot_manifest

    # Step 1: look up any existing dataset by (owner, name).
    #
    # pgw#656: every way this lookup can fail to SEE an existing row ends in
    # step 2a — CREATE — i.e. a duplicate dataset, silently. So it fails loud
    # instead of falling through, and it walks the whole listing:
    #   * a non-2xx / unparseable response now raises (it used to be
    #     indistinguishable from "no such dataset");
    #   * the listing is CURSOR-paginated (tensorhub caps `limit` at 200 and
    #     returns `next_cursor`), where it used to read one default page of 50
    #     and call that the org's whole dataset set;
    #   * the filter is `?org=`, which is the parameter the handler actually
    #     reads — `?tenant=` was silently ignored and the result only
    #     happened to be right because the token's own org was the default.
    existing_id = _find_dataset_id(base, headers, owner, name)

    if not existing_id:
        # Step 2a: create.
        create_body = {
            "tenant": owner,
            "name": name,
            "visibility": visibility,
            "schema": features_payload,
        }
        resp = requests.post(
            f"{base}/api/v1/datasets",
            headers=headers,
            data=json.dumps(create_body).encode("utf-8"),
            timeout=30,
        )
        if resp.status_code in (401, 403):
            raise AuthError(f"dataset create unauthorized ({resp.status_code})")
        if resp.status_code < 200 or resp.status_code >= 300:
            raise RuntimeError(
                f"dataset create failed ({resp.status_code}): {resp.text[:256]}"
            )
        data = resp.json() if resp.text else {}
        return {
            "ok": True,
            "dataset_id": str(data.get("dataset_id") or ""),
            "owner": owner,
            "name": name,
            "existed": False,
        }

    # Step 2b: update via PATCH.
    patch_url = f"{base}/api/v1/datasets/{urllib.parse.quote(existing_id, safe='')}"
    patch_body: dict[str, Any] = {"schema": features_payload}
    if row_artifacts_json is not None:
        patch_body["row_artifacts"] = row_artifacts_json
    if visibility in ("private", "public"):
        patch_body["visibility"] = visibility
    resp = requests.patch(
        patch_url,
        headers=headers,
        data=json.dumps(patch_body).encode("utf-8"),
        timeout=30,
    )
    if resp.status_code in (401, 403):
        raise AuthError(f"dataset patch unauthorized ({resp.status_code})")
    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(
            f"dataset patch failed ({resp.status_code}): {resp.text[:256]}"
        )
    return {
        "ok": True,
        "dataset_id": existing_id,
        "owner": owner,
        "name": name,
        "existed": True,
    }


def files_from_tree(tree: Path, *, prefix: str = "") -> list[CommitFile]:
    """Build CommitFile entries for every regular file under ``tree``.

    ``.cache/huggingface/**`` is skipped: huggingface_hub's local-dir download
    metadata is cache-layout junk, never repo content."""
    tree = Path(tree)
    out: list[CommitFile] = []
    for f in sorted(tree.rglob("*")):
        if not f.is_file():
            continue
        rel_parts = f.relative_to(tree).parts
        if rel_parts[:2] == (".cache", "huggingface"):
            continue
        rel = f.relative_to(tree).as_posix()
        if prefix:
            rel = f"{prefix.rstrip('/')}/{rel}"
        out.append(CommitFile(path=rel, local_path=f))
    return out


__all__ = [
    "HubClient",
    "HubPublishError",
    "CommitFile",
    "CommitResult",
    "files_from_tree",
    "publish_dataset_revision",
]
