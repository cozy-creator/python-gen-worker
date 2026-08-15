"""Tensorhub publish client — the ONE publish path, CHUNKED SHA-256.

  POST /api/v1/repos/{org}/{name}/publishes
      {files: [{path, size_bytes, digest:"sha256:<hex>", chunks?}], mode,
       release, dtype/file_layout/file_type, artifact_contract,
       metadata, provenance, ...}
  → {publish_id, have: [...], need: [{digest, put_url, headers, ...}]}

Then PUT every `need` grant VERBATIM (the sha256 is signed into the presigned
URL, so the store itself refuses bytes that do not hash to the key), re-plan
`.../grants` to resume, and POST `.../complete`. One publish == one
checkpoint; N artifacts are N publishes joining one RELEASE, told apart there
by their tensor-layout contracts. th#1987 deleted the tag axis and its head.

THE v1 (blake3) `/commits` PROTOCOL IS GONE FROM THIS CLIENT: it is frozen
hub-side (every new v1 commit answers 410 `unsupported_digest_algorithm`),
and a retired protocol left resident in the tree is a runtime failure on a
rented pod instead of a compile failure in CI. The by-reference add went with
it — a digest must be proven from bytes in hand.
"""

from __future__ import annotations

import json
import logging
import random
import socket
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, cast

import requests
from hashrepo import (
    CASRef,
    RepositoryManifest,
    TransferGrant,
    TransferJournal,
    TransferSession,
    upload,
)

from .. import activity as _activity
from ..http_origin import is_definite_hub_answer
from ..models.cache_paths import open_worker_cas
from ..stall import SilenceWindow
from .publish_state import JOURNAL_NAME, STATE_NAME, ProducerRecovery

logger = logging.getLogger(__name__)

# The retry loop's PACE, and deliberately not its end — the give-up is
# `SilenceWindow(_SEND_SILENCE_WINDOW_S)` over definite hub answers, so these
# two decide only how often a hub that is down gets asked. A publish riding out
# a hub rebuild for the better part of an hour must not become a retry flood
# against it, and must not stall an hour between attempts once it recovers. The
# ceiling also bounds a hostile or buggy `Retry-After`, which `_retry_after_s`
# clamps to it.
_RETRY_BASE_DELAY_S = 1.0
_RETRY_MAX_DELAY_S = 30.0

# Connect timeout split from read (mirrors gen_worker.hubio.transport and the
# download-side floor): a dead host fails in seconds instead of consuming the
# whole read budget.
_CONNECT_TIMEOUT_S = 15.0

# Bounded RE-PLAN passes for a publish whose granted objects did not all land.
# Resume needs no client state: the need set comes back smaller because what
# landed is now resident, so a pass costs only the objects still missing.
_REUPLOAD_ATTEMPTS = 2

# Re-plans spent purely on RE-MINTING expired presigns, charged to
# their own budget. An expired grant is not a failed transfer — nothing about
# the bytes is in question — so consuming a re-upload pass for one is how a
# 2 h TTL crossed mid-publish turns into a fatal job. Small, because the CAS
# grant TTL is 2 h and a publish that crosses it twice has a different problem.
_EXPIRY_REPLAN_ATTEMPTS = 3

# tensorhub's /complete verifies the publish synchronously before answering.
# For a large tree that can outlast whatever timeout sits in front of the hub,
# and the client must not read its own impatience as a failure.
_COMPLETE_TIMEOUT_S = 600.0

# How long the client tolerates hearing NOTHING DEFINITE from the hub before
# giving up — network errors, edge-masked 5xx, proxy-shaped 404s.
# Silence-bounded, never attempt-counted: a hub restart is seconds and a tunnel
# re-dial is minutes, and an attempt cap classifies both FATAL and throws away
# paid GPU work at the finish line. Six verify-lengths, because a container
# rebuild is tens of minutes and an hour parked on the CPU rig costs about what
# re-downloading costs — and unlike the re-download it cannot fail the same
# way twice. Waiting stays observable: the loop beats liveness every pass.
_COMPLETE_SILENCE_WINDOW_S = 6.0 * _COMPLETE_TIMEOUT_S

def _dtype_token(v: str) -> str:
    """Publish-body dtype hygiene: the internal dtype-axis colon
    forms (``gguf:q4_k_m``, ``int8:awq``) publish as ``-`` forms.

    `dtype` is the only field normalized here — `flavor`/`flavors`/
    `default_flavor` are not part of the publish body at all.
    """
    return str(v or "").replace(":", "-")


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
    one (the ``{"error": {"code": ...}}`` envelope, or a projection's
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


# The hub's typed refusal for a publish naming a release nobody cut (th#1980).
RELEASE_NOT_FOUND = "release_not_found"

#: The ONE legal "no release" on a v2 publish. A self-minted compiled graph is
#: selected by its endpoint-scoped `compiled_graph_store` row, never by
#: contract, so it joins no release and the hub answers `release_forbidden` to a
#: body that names one. It is a NAMED value rather than an empty string so a
#: caller that simply forgot is refused instead of silently taking the exemption.
COMPILED_GRAPH_NO_RELEASE = "compiled-graph:no-release"


class HubReleaseRequiredError(HubPublishError):
    """The publish named no release, and th#1987 made one mandatory.

    Raised HERE rather than after a multi-GB upload: the hub refuses at DECLARE
    with `release_required`, and a producer that cannot say which release its
    output belongs to has a caller-side defect (`destination.release` unset),
    not a transfer problem.
    """


class HubReleaseNotFoundError(HubPublishError):
    """The named release does not exist, and publishing does not cut one.

    Cutting a release is a deliberate act (``POST
    /repos/{org}/{name}/releases``); a publish ATTACHES an artifact to one that
    already exists. So the remedy is to cut the release and publish again — never
    to re-hash, re-cast or re-upload the bytes, which were never the problem.
    Typed because "the release is missing" and "the artifact is bad" are the two
    refusals a producer must not confuse: one is a control-plane act costing
    nothing, the other is gigabytes.
    """


def _publish_refusal(message: str, *, status: int = 0, code: str = "",
                     retryable: Optional[bool] = None) -> HubPublishError:
    """One hub refusal as the narrowest exception that fits its code."""
    if code == RELEASE_NOT_FOUND:
        return HubReleaseNotFoundError(
            f"{message} — cut the release first "
            "(POST /repos/{org}/{name}/releases), then publish into it; the "
            "bytes are not at fault, do not re-upload",
            status=status, code=code, retryable=False)
    return HubPublishError(message, status=status, code=code, retryable=retryable)


def _is_terminal_repudiation(exc: BaseException) -> bool:
    """Should this failure destroy the session's staged bytes?

    ONLY a refusal the hub itself classified terminal. `DELETE /publishes/:id`
    is not a tidy-up — hub-side it runs `cleanupCASPublishV2Staging` and
    deletes every staged chunk, so answering it is answering "these bytes can
    never be useful to anyone". That is true of a repudiation (audit findings,
    contract failure, possession refusal): the declaration itself is refused,
    so re-uploading against it cannot help. It is false of everything else — a
    transport blip, a timeout, a proxy outage, a KeyboardInterrupt, a local
    bug — where the staged objects are exactly what a retry wants.

    The default is therefore KEEP. An exception we cannot classify is not
    evidence that 37 GB is worthless.
    """
    if isinstance(exc, HubPublishError):
        return exc.retryable is False
    return False


def _retry_after_s(resp: requests.Response) -> Optional[float]:
    try:
        value = float(str(resp.headers.get("Retry-After") or "").strip())
    except Exception:
        return None
    return min(value, _RETRY_MAX_DELAY_S) if value > 0 else None


def _error_code_of(resp: requests.Response) -> str:
    """The hub's `error.code`, or "" when the body isn't an envelope.

    pgw#987: the publish envelope (`publishError.body()`) and gin's
    `AbortWithStatusJSON` both emit the code as a STRING alongside a sibling
    `message`. Dropping it left `_publish_failure_phase` with only `http_413`
    to group 32 identical refusals by, when the hub had named the fault
    exactly. pgw#1229 moved both shapes into the one parser.
    """
    from ..hub_error import hub_error_of

    return hub_error_of(resp).code


# How long _send_with_retries tolerates hearing nothing DEFINITE from the hub
# (network errors, 5xx, 429, proxy-shaped 404s) before giving up. A hub restart
# is seconds and a tunnel re-dial is minutes, so the bound is SILENCE, never an
# attempt count.
_SEND_SILENCE_WINDOW_S = 600.0


def _send_with_retries(
    what: str,
    send: Callable[[], requests.Response],
    *,
    silence_window_s: Optional[float] = None,
    definite: Optional[Callable[[requests.Response], bool]] = None,
) -> requests.Response:
    """Origin-discriminating, silence-bounded retry.

    Only a DEFINITE hub answer ends the loop: 2xx/3xx, or a 4xx that is
    really the hub speaking (not a proxy's offline page). Indefinite answers
    — network errors, 429, 5xx, and ANY proxy-shaped status (ngrok with no
    healthy backend while the hub restarts answers 404 AND 503 with the same
    HTML page) — retry with backoff until nothing definite has been
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
            # route whose refusals are not envelope-shaped (a v2 completion
            # answers with a projection carrying an explicit `retryable` flag).
            # The default stays the shape heuristic.
            is_definite = definite(resp) if definite is not None else is_definite_hub_answer(resp)
            if code != 429 and is_definite:
                return resp  # definite hub answer
            last_resp, last_exc = resp, None
            delay = _retry_after_s(resp) or delay
        # This loop can legitimately run for the better part of an hour riding
        # out a hub rebuild, and a publisher silent for an hour is the dead-job
        # signature the watchdogs kill on. Waiting IS work — say so on the
        # liveness channel.
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
    protocol can attest."""

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
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.token = str(token or "").strip()
        self.timeout_s = timeout_s
        if not self.base_url or not self.token:
            raise HubPublishError("missing tensorhub base URL or capability token")

    @classmethod
    def from_ctx(cls, ctx: Any) -> "HubClient":
        """Build from a gen_worker RequestContext (cap-token identity)."""
        base = str(getattr(ctx, "_file_api_base_url", "") or "").strip()
        token = str(getattr(ctx, "_worker_capability_token", "") or "").strip()
        return cls(base_url=base, token=token)

    # ---- internals ----

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

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

    def _abort_publish(self, repo_path: str, publish_id: str) -> None:
        """Land a REPUDIATED session in a terminal state. Destroys staging —
        call only from :func:`_is_terminal_repudiation`'s branch."""
        if not publish_id:
            return
        try:
            _http_session().delete(
                f"{self.base_url}{repo_path}/publishes/"
                f"{urllib.parse.quote(publish_id, safe='')}",
                headers=self._headers(), timeout=(_CONNECT_TIMEOUT_S, 30),
            )
        except Exception:  # noqa: BLE001 — best effort; the session TTL backstops it
            logger.debug("publish %s abort failed", publish_id, exc_info=True)

    def _post_v2_complete(self, path: str) -> requests.Response:
        """POST the hub's idempotent v2 completion.

        `_send_with_retries` decides "did we hear from the hub?" from the body's
        SHAPE, and a v2 completion's refusal is a projection rather than an
        error envelope — so it would read a real, typed, `retryable: false`
        refusal as a proxy non-answer and retry it. The retry then finds the
        session already terminal and returns 409 `publish_repudiated`: a
        consequence in place of the cause, which is gone.

        Tensorhub makes this route idempotent: completing an already-promoted
        session returns its terminal status and checkpoint id. Network-level
        failures therefore retry safely, including a lost success response;
        an HTTP answer of any status is the hub speaking and is returned as-is
        for the typed handling below.
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
        # The release this artifact ATTACHES to (th#1980), REQUIRED since
        # th#1987 — no default, so every call site states one. It must ALREADY
        # have been cut: publishing never cuts a release, exactly as uploading a
        # binary to GitHub never creates the release it lands in. An unknown
        # identifier is a typed, non-retryable `release_not_found`
        # (:class:`HubReleaseNotFoundError`), whose remedy is a control-plane
        # act, never a re-upload. `COMPILED_GRAPH_NO_RELEASE` is the ONE legal
        # empty value — see its docstring.
        release: str,
        # "replace" — a checkpoint is COMPLETE IN ITSELF. "merge" unions this
        # publish with the repo's prior manifest, so a caller that never
        # mentioned a sibling INHERITS its bytes (an fp8 checkpoint carrying
        # the fp16 base weights; a differently-sharded base spliced into a
        # quantization). Both this default and the hub's are "replace"; pass
        # mode="merge" explicitly and only for what it is for: assembling ONE
        # checkpoint across several commits (clone.py's chunked full-clone,
        # _stream.py's streamed output).
        mode: str = "replace",
        # `ns.name@N` — what the bytes ARE. PROVEN at tier 1 against
        # the safetensors header, so an unprovable claim refuses the publish
        # instead of being stored as a maybe.
        artifact_contract: str = "",
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
        journal_path: Optional[Path] = None,
        journal_state: Optional[Mapping[str, Any]] = None,
    ) -> CommitResult:
        """Publish one checkpoint over the CHUNKED SHA-256 CAS.

        Declare -> `{have, need}` -> PUT the needed objects -> complete. What
        makes this different from :meth:`commit` is not the transport but WHO
        GUARANTEES INTEGRITY: each object's sha256 is signed into its presigned
        PUT, so R2 refuses bytes that do not hash to the key (400, and the
        object does not exist afterwards) and refuses a substituted claim (403).
        A claimed digest stops being assertable, which kills the
        inherit/overwrite class structurally rather than by guard.

        Files may become an ordered list of enforced chunks at any size. Each
        manifest length is authoritative, so a retry costs one bounded object
        and progress is monotone across retries — a lemon host is bounded by
        the unit of work rather than by a new watchdog.

        THERE IS NO PROTOCOL AUTO-SELECT AND NO ENV KNOB. The caller names the
        protocol, so flipping a producer class to sha256 is a code change and a
        deploy. A hub without these routes answers 404, and this deliberately
        does NOT fall back: a silent downgrade to blake3 would make "did this
        publish v2?" unanswerable from the outside (and a 404 from a PROXY is
        not a 404 from the hub).

        ``provenance`` carries the WORKER-ADDABLE stamp subset only, exactly as
        :meth:`commit` does — the hub resolves the authoritative stamp from the
        capability token, so lineage cannot be forged from here
        and is identical whichever protocol published it.

        ``on_stage(stage, facts)`` reports the protocol's own legs —
        ``declared`` (publish_id + the have/need split, i.e. what dedup
        actually saved), ``resumed`` (a journalled session re-adopted),
        ``uploading`` (objects and bytes this pass will move), ``committing``
        — so a caller whose publish is otherwise a silent background thread can
        put the LEG on the wire instead of only its terminus. Never
        load-bearing: a raising callback is the caller's bug and must not fail
        a publish that is transferring correctly.

        ``journal_path`` turns "the upload died" into "re-upload"
        rather than "re-run the cast". The session id is recorded beside the
        produced bytes BEFORE the first PUT, so a retry on this pod re-adopts
        the same HashRepo session (same staging prefix) and re-plans it instead
        of declaring a fresh one.
        """

        if not files:
            raise HubPublishError("publish_v2 requires at least one file")
        attaches_to_no_release = release == COMPILED_GRAPH_NO_RELEASE
        release = "" if release == COMPILED_GRAPH_NO_RELEASE else release.strip()
        if not release and not attaches_to_no_release:
            raise HubReleaseRequiredError(
                f"publish into {destination_repo!r} names no release: th#1987 "
                "made `release` mandatory. Cut one "
                "(POST /repos/{org}/{name}/releases) and pass release=<id>; "
                "pass COMPILED_GRAPH_NO_RELEASE only for a self-mint "
                "compiled-graph publish, which attaches to no release.",
                code="release_required", retryable=False)

        repo_path = self._repo_path(destination_repo)

        # HashRepo owns the local objects and manifest. A by-reference add has
        # no bytes to ingest and therefore cannot enter this protocol.
        cas = open_worker_cas()
        entries = []
        for f in files:
            if f.local_path is None:
                raise HubPublishError(
                    f"publish_v2: {f.path!r} is a by-reference add; v2 declares "
                    "digests computed from local bytes"
                )
            entries.append(cas.ingest_file(Path(f.local_path), manifest_path=f.path))
        manifest = RepositoryManifest(tuple(entries))
        manifest_ref = manifest.digest()

        def _tensorhub_file(entry: Any) -> dict[str, object]:
            """Adapt HashRepo v1 to Tensorhub's ordered-sha256 request shape."""
            raw = cast(dict[str, object], entry.to_dict())
            chunks = raw.get("chunks")
            if isinstance(chunks, list):
                for chunk in chunks:
                    if isinstance(chunk, dict):
                        chunk["digest"] = CASRef.parse(str(chunk["digest"])).digest
            return raw

        body: dict[str, Any] = {
            "mode": mode,
            "files": [_tensorhub_file(entry) for entry in manifest.files],
        }
        for key, val in (
            ("release", release),
            ("dtype", _dtype_token(dtype)), ("file_layout", file_layout),
            ("file_type", file_type), ("display_label", display_label),
            ("objective", objective), ("artifact_contract", artifact_contract),
        ):
            if val:
                body[key] = val
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
            # The v2 declare decodes this STRICTLY into the same
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
            allowed = {
                "step_number",
                "epoch_number",
                "quantization_method",
                "quantization_library",
                "upstream_revision",
                "upstream_attestation",
            }
            body["provenance"] = {
                key: value
                for key, value in dict(provenance).items()
                if key in allowed and value
            }
        for key in ("kind", "library_name", "model_family", "class_name",
                    "adapter_for_family"):
            val = str((repo_spec or {}).get(key) or "").strip()
            if val:
                body[key] = val

        total_bytes = sum(entry.size_bytes for entry in manifest.files)
        declaration_ref = CASRef.digest_bytes(
            json.dumps(
                body,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        )
        session_name = (
            f"tensorhub:{destination_repo}:{mode}:"
            f"{manifest_ref.digest}:{declaration_ref.digest}"
        )
        journal = TransferJournal(journal_path) if journal_path else None
        recovery = ProducerRecovery(journal_path) if journal_path else None

        def _stage(stage: str, **facts: Any) -> None:
            if on_stage is None:
                return
            try:
                on_stage(stage, dict(facts))
            except Exception:  # noqa: BLE001 — reporting never fails a publish
                logger.debug("publish_v2 on_stage(%s) raised", stage, exc_info=True)

        def _grants_of(payload: Mapping[str, Any]) -> list[TransferGrant]:
            out = []
            for g in payload.get("need") or []:
                if not isinstance(g, dict):
                    continue
                expiry = str(g.get("expires_at") or "").strip() or None
                out.append(TransferGrant(
                    digest=CASRef.parse(str(g.get("digest") or "").strip().lower()),
                    size_bytes=int(g.get("size_bytes") or 0),
                    url=str(g.get("put_url") or ""),
                    headers={str(k): str(v) for k, v in (g.get("headers") or {}).items()},
                    staging_key=str(g.get("staging_key") or ""),
                    expires_at=expiry,
                ))
            return out

        def _replan(pid: str) -> Mapping[str, Any]:
            """Re-plan an EXISTING session. This is also the CAS path's presign
            re-mint: the hub answers with fresh grants for whatever the need
            set still names."""
            again = self._post(f"{repo_path}/publishes/"
                               f"{urllib.parse.quote(pid, safe='')}/grants")
            if again.status_code < 200 or again.status_code >= 300:
                raise HubPublishError(
                    f"publish re-plan failed ({again.status_code}): {again.text[:500]}",
                    status=again.status_code, code=_error_code_of(again))
            return self._json(again)

        # Adopt a journalled session before declaring a new one. Its
        # staging prefix is session-scoped, so reusing the id is the ONLY way
        # to reach bytes a predecessor already moved. A session the hub no
        # longer accepts simply falls through to a fresh declare — resuming is
        # an optimization and must never be a way to fail.
        #
        # The DECLARATION is not re-sent: the session already carries it, and
        # the journal key is the declared object set, so an adopted session is
        # about exactly these bytes and declaration semantics. The journal key
        # includes a canonical digest of tags, metadata and policy fields, so a
        # caller changing those facts declares a fresh session instead of
        # silently completing the predecessor's request.
        session: Optional[Mapping[str, Any]] = None
        publish_id = ""
        prior = journal.find(session_name, manifest_ref) if journal is not None else None
        if prior is not None:
            try:
                session = _replan(prior.session_id)
                publish_id = prior.session_id
                logger.info(
                    "publish_v2 resuming journalled session %s for %s (%d objects)",
                    publish_id, destination_repo, len(manifest.files))
                _stage("resumed", publish_id=publish_id,
                       need=len(session.get("need") or []), bytes=total_bytes)
            except HubPublishError as exc:
                logger.info(
                    "publish_v2 could not resume journalled session %s (%s); "
                    "declaring a fresh publish", prior.session_id, exc)
                journal.clear(session_name, session_id=prior.session_id)  # type: ignore[union-attr]
                if recovery is not None:
                    recovery.clear(prior.session_id)
                session = None

        if session is None:
            resp = self._post(f"{repo_path}/publishes", body)
            if resp.status_code < 200 or resp.status_code >= 300:
                raise _publish_refusal(
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

        # Recorded BEFORE the first PUT: a journal written after the transfer
        # is a journal that never survives the transfer failing.
        if journal is not None:
            journal.record(TransferSession(session_name, publish_id, manifest_ref))
            if recovery is not None:
                recovery.record(
                    publish_id,
                    paths=tuple(entry.path for entry in manifest.files),
                    producer_state=dict(journal_state or {}),
                )

        try:
            grants = _grants_of(session)
            _stage("declared", publish_id=publish_id, objects=distinct,
                   resident=resident, need=len(grants), bytes=total_bytes)
            attempt = 0
            expiry_replans = 0
            while grants:
                _stage("uploading", publish_id=publish_id, objects=len(grants),
                       bytes=sum(g.size_bytes for g in grants), attempt=attempt)
                report = upload(
                    grants,
                    cas,
                    progress=(lambda _digest, n: part_progress(0, 0, n))
                    if callable(part_progress) else None,
                )
                uploaded_objects += report.succeeded
                if callable(progress):
                    progress(resident + uploaded_objects, distinct or len(grants))
                if report.ok:
                    break
                if report.needs_replan:
                    # Nothing failed — presigns went stale. Re-mint them
                    # without charging the re-upload budget, which exists for
                    # objects that would not land.
                    expiry_replans += 1
                    if expiry_replans > _EXPIRY_REPLAN_ATTEMPTS:
                        raise HubPublishError(
                            f"publish {publish_id}: grants kept expiring after "
                            f"{_EXPIRY_REPLAN_ATTEMPTS} re-mints "
                            f"({len(report.expired)} object(s) still stale)",
                            code="grant_expiry_loop")
                    logger.info(
                        "publish %s re-minting %d expired grant(s) (re-mint %d/%d)",
                        publish_id, len(report.expired), expiry_replans,
                        _EXPIRY_REPLAN_ATTEMPTS)
                    grants = _grants_of(_replan(publish_id))
                    continue
                attempt += 1
                if attempt > _REUPLOAD_ATTEMPTS:
                    raise HubPublishError(
                        f"publish {publish_id}: {len(report.failures)} object(s) failed to "
                        f"upload after {_REUPLOAD_ATTEMPTS + 1} passes: "
                        + "; ".join(
                            f"{digest}: {detail}"
                            for digest, detail in report.failures[:5]
                        )
                    )
                # RESUME NEEDS NO CLIENT STATE FOR THIS PROCESS: re-plan and
                # the need set names what still has to move, and it comes back
                # SMALLER because what landed is now resident. The journal
                # above is what lets a LATER process re-enter this same loop.
                grants = _grants_of(_replan(publish_id))

            _stage("committing", publish_id=publish_id, objects=distinct,
                   resident=resident, uploaded=uploaded_objects)
            done = self._post_v2_complete(
                f"{repo_path}/publishes/{urllib.parse.quote(publish_id, safe='')}/complete")

            if done.status_code < 200 or done.status_code >= 300:
                # Lead with the hub's OWN typed classification when it gave one.
                # A refusal that reports its `code`, its `retryable` bit and the
                # stage that produced it is actionable; a bare status code plus
                # 800 bytes of truncated projection is not.
                failure = self._v2_failure(done)
                if failure:
                    stage = ""
                    try:
                        for s in (done.json().get("status") or {}).get("stages") or []:
                            if s.get("status") == "failed":
                                stage = str(s.get("stage") or "")
                    except Exception:  # noqa: BLE001 - the stage is a nicety
                        pass
                    raise _publish_refusal(
                        f"publish {publish_id} "
                        f"{'repudiated' if not failure.get('retryable') else 'failed'}"
                        f"{f' at {stage}' if stage else ''}: "
                        f"{failure.get('code')}: {failure.get('message')} "
                        f"(retryable={bool(failure.get('retryable'))})",
                        status=done.status_code, code=str(failure.get("code") or ""),
                        retryable=bool(failure.get("retryable")),
                    )
                raise _publish_refusal(
                    f"publish complete failed ({done.status_code}): {done.text[:800]}",
                    status=done.status_code, code=_error_code_of(done))
            final = self._json(done)
            ckpt = final.get("checkpoint") if isinstance(final.get("checkpoint"), dict) else {}
            status = final.get("status") if isinstance(final.get("status"), dict) else {}
            checkpoint_id = str(
                (ckpt or {}).get("checkpoint_id")
                or final.get("checkpoint_id")
                or (status or {}).get("checkpoint_id")
                or ""
            ).strip()
            if not checkpoint_id:
                raise HubPublishError(
                    f"publish {publish_id} completed without a checkpoint id: "
                    f"{json.dumps(final)[:500]}", code="checkpoint_id_missing")
        except BaseException as exc:
            # ABORT ONLY ON A TERMINAL REFUSAL. `DELETE /publishes/:id` runs
            # `cleanupCASPublishV2Staging` hub-side, which deletes every staged
            # chunk — aborting on a transport blip throws away every transferred
            # byte to keep a session row tidy. A transport failure, a timeout or
            # a KeyboardInterrupt leaves the session intact so a retry (this
            # pod, via the journal) can resume it. Sessions that genuinely leak
            # are the staging lifecycle's problem.
            if _is_terminal_repudiation(exc):
                self._abort_publish(repo_path, publish_id)
                if journal is not None:
                    journal.clear(session_name, session_id=publish_id)
                if recovery is not None:
                    recovery.clear(publish_id)
            else:
                logger.warning(
                    "publish %s failed with a non-terminal error (%s); leaving the "
                    "session and its staged objects intact for a retry",
                    publish_id, type(exc).__name__)
            raise

        # Promoted. The produced tree and this journal entry have no further
        # job: the bytes are in the CAS.
        if journal is not None:
            journal.clear(session_name, session_id=publish_id)
        if recovery is not None:
            recovery.clear(publish_id)
        # Surfaced, not swallowed: the hub names which canonical checks it did
        # NOT run. A list that promises 19 and silently runs 14 is worse than
        # one promising 14.
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


def files_from_tree(tree: Path, *, prefix: str = "") -> list[CommitFile]:
    """Build CommitFile entries for every regular file under ``tree``.

    ``.cache/huggingface/**`` is skipped: huggingface_hub's local-dir download
    metadata is cache-layout junk, never repo content. Publish recovery state
    and its locks are bookkeeping about the tree, not part of it.
    """
    tree = Path(tree)
    out: list[CommitFile] = []
    for f in sorted(tree.rglob("*")):
        if not f.is_file():
            continue
        rel_parts = f.relative_to(tree).parts
        if rel_parts[:2] == (".cache", "huggingface"):
            continue
        if f.name in (
            JOURNAL_NAME,
            STATE_NAME,
            f".{JOURNAL_NAME}.lock",
            f".{STATE_NAME}.lock",
        ):
            continue
        rel = f.relative_to(tree).as_posix()
        if prefix:
            rel = f"{prefix.rstrip('/')}/{rel}"
        out.append(CommitFile(path=rel, local_path=f))
    return out


__all__ = [
    "HubClient",
    "HubPublishError",
    "HubReleaseNotFoundError",
    "RELEASE_NOT_FOUND",
    "CommitFile",
    "CommitResult",
    "files_from_tree",
]
