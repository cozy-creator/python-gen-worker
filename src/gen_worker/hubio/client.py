"""Tensorhub publish client — the ONE publish path, CHUNKED SHA-256."""

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
from gen_worker._vendor.tensorfs import CASRef, RepositoryManifest
from gen_worker.cas.admission import ingest_file
from gen_worker.transfer.grants import TransferGrant, upload
from gen_worker.transfer.journal import TransferJournal, TransferSession

from .. import activity as _activity
from .. import scratchrepo
from ..http_origin import is_definite_hub_answer, response_is_from_hub
from ..models.cache_paths import open_worker_cas
from ..stall import SilenceWindow
from .publish_state import JOURNAL_NAME, STATE_NAME, ProducerRecovery

logger = logging.getLogger(__name__)

_RETRY_BASE_DELAY_S = 1.0
_RETRY_MAX_DELAY_S = 30.0

_CONNECT_TIMEOUT_S = 15.0

_REUPLOAD_ATTEMPTS = 2

_EXPIRY_REPLAN_ATTEMPTS = 3

_COMPLETE_TIMEOUT_S = 600.0

_COMPLETE_SILENCE_WINDOW_S = 240.0

def _dtype_token(v: str) -> str:
    return str(v or "").replace(":", "-")


_SESSION: Optional[requests.Session] = None


def _http_session() -> requests.Session:

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
    """Terminal failure talking to tensorhub's commit API."""

    def __init__(self, message: str, *, status: int = 0, code: str = "",
                 retryable: Optional[bool] = None) -> None:
        super().__init__(message)
        self.status = int(status or 0)
        self.code = str(code or "")
        self.retryable = retryable


RELEASE_NOT_FOUND = "release_not_found"

# The ONE legal "no release" on a v2 publish: a self-minted compiled graph is selected by its endpoint-scoped store row, never by contract, so it joins no release and the hub answers release_forbidden to a body that names one. A NAMED value rather than an empty string, so a caller that simply forgot is refused instead of silently taking the exemption.
COMPILED_GRAPH_NO_RELEASE = "compiled-graph:no-release"


class HubReleaseRequiredError(HubPublishError):
    """The publish named no release, and a release is mandatory. Raised at declare rather than after a multi-GB upload: a producer that cannot say which release its output belongs to has a caller-side defect (destination.release unset), not a transfer problem."""


class HubReleaseNotFoundError(HubPublishError):
    """The named release does not exist, and publishing does not cut one."""


def _publish_refusal(message: str, *, status: int = 0, code: str = "",
                     retryable: Optional[bool] = None) -> HubPublishError:
    if code == RELEASE_NOT_FOUND:
        return HubReleaseNotFoundError(
            f"{message} — cut the release first "
            "(POST /repos/{org}/{name}/releases), then publish into it; the "
            "bytes are not at fault, do not re-upload",
            status=status, code=code, retryable=False)
    return HubPublishError(message, status=status, code=code, retryable=retryable)


def _is_terminal_repudiation(exc: BaseException) -> bool:
    """Should this failure destroy the session's staged bytes? DELETE /publishes/:id runs cleanupCASPublishV2Staging hub-side and deletes every staged chunk, so answer yes ONLY for a refusal the hub itself classified terminal (repudiation). The default is KEEP: a transport blip's staged objects are exactly what a retry wants."""
    if isinstance(exc, HubPublishError):
        return exc.retryable is False
    return False


def _retry_after_s(resp: requests.Response) -> Optional[float]:
    try:
        value = float(str(resp.headers.get("Retry-After") or "").strip())
    except Exception:
        return None
    return min(value, _RETRY_MAX_DELAY_S) if value > 0 else None


FRONT_DOOR_UNAVAILABLE = "front_door_unavailable"


def _non_hub_origin(resp: requests.Response) -> str:
    try:
        ctype = str(resp.headers.get("Content-Type") or "").split(";")[0].strip()
    except Exception:  # noqa: BLE001 - header access must never mask the real error
        ctype = ""
    return f"HTTP {int(getattr(resp, 'status_code', 0))}, {ctype or 'no content-type'}"


def _error_code_of(resp: requests.Response) -> str:
    from ..hub_error import hub_error_of

    return hub_error_of(resp).code


_SEND_SILENCE_WINDOW_S = 600.0


def _send_with_retries(
    what: str,
    send: Callable[[], requests.Response],
    *,
    silence_window_s: Optional[float] = None,
    definite: Optional[Callable[[requests.Response], bool]] = None,
) -> requests.Response:

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
            is_definite = definite(resp) if definite is not None else is_definite_hub_answer(resp)
            if code != 429 and is_definite:
                return resp
            last_resp, last_exc = resp, None
            delay = _retry_after_s(resp) or delay
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
    """One file to publish: repo path + the LOCAL BYTES."""

    path: str
    local_path: Optional[Path] = None
    size_bytes: int = 0


@dataclass
class CommitResult:
    revision_id: str
    uploaded: int
    deduped: int
    total_bytes: int
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
        if not self.base_url and not self.token:
            raise HubPublishError(
                "the dispatch carried neither a tensorhub base URL nor a "
                "capability token", code="worker_credentials_missing",
                retryable=True)
        if not self.base_url:
            raise HubPublishError(
                "the dispatch carried a capability token but no tensorhub "
                "base URL", code="worker_hub_url_missing", retryable=True)
        if not self.token:
            raise HubPublishError(
                "the dispatch carried a tensorhub base URL but no capability "
                "token", code="worker_capability_token_missing", retryable=True)

    @classmethod
    def from_ctx(cls, ctx: Any) -> "HubClient":
        """Build from a gen_worker RequestContext (cap-token identity)."""
        base = str(getattr(ctx, "_file_api_base_url", "") or "").strip()
        token = str(getattr(ctx, "_worker_capability_token", "") or "").strip()
        return cls(base_url=base, token=token)

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

    @staticmethod
    def _v2_complete_is_definite(resp: requests.Response) -> bool:
        if HubClient._v2_failure(resp) is not None:
            return True
        return is_definite_hub_answer(resp)

    def _post_v2_complete(self, path: str) -> requests.Response:
        return _send_with_retries(
            f"POST {path}",
            lambda: _http_session().post(
                f"{self.base_url}{path}", headers=self._headers(),
                timeout=(_CONNECT_TIMEOUT_S, _COMPLETE_TIMEOUT_S),
            ),
            silence_window_s=_COMPLETE_SILENCE_WINDOW_S,
            definite=HubClient._v2_complete_is_definite,
        )

    def publish_v2(
        self,
        *,
        destination_repo: str,
        files: list[CommitFile],
        release: str,
        mode: str = "replace",
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
        """Publish one checkpoint over the CHUNKED SHA-256 CAS."""

        if not files:
            raise HubPublishError("publish_v2 requires at least one file")
        attaches_to_no_release = release == COMPILED_GRAPH_NO_RELEASE
        release = "" if release == COMPILED_GRAPH_NO_RELEASE else release.strip()
        derives = scratchrepo.derives_its_release(destination_repo)
        if not release and not attaches_to_no_release and not derives:
            raise HubReleaseRequiredError(
                f"publish into {destination_repo!r} names no release: th#1987 "
                "made `release` mandatory. Cut one "
                "(POST /repos/{org}/{name}/releases) and pass release=<id>; "
                "pass COMPILED_GRAPH_NO_RELEASE only for a self-mint "
                "compiled-graph publish, which attaches to no release.",
                code="release_required", retryable=False)

        repo_path = self._repo_path(destination_repo)

        cas = open_worker_cas()
        entries = []
        for f in files:
            if f.local_path is None:
                raise HubPublishError(
                    f"publish_v2: {f.path!r} is a by-reference add; v2 declares "
                    "digests computed from local bytes"
                )
            entries.append(ingest_file(cas, Path(f.local_path), manifest_path=f.path))
        manifest = RepositoryManifest(tuple(entries))
        manifest_ref = manifest.digest()

        def _tensorhub_file(entry: Any) -> dict[str, object]:
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
            again = self._post(f"{repo_path}/publishes/"
                               f"{urllib.parse.quote(pid, safe='')}/grants")
            if again.status_code < 200 or again.status_code >= 300:
                raise HubPublishError(
                    f"publish re-plan failed ({again.status_code}): {again.text[:500]}",
                    status=again.status_code, code=_error_code_of(again))
            return self._json(again)

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
                grants = _grants_of(_replan(publish_id))

            _stage("committing", publish_id=publish_id, objects=distinct,
                   resident=resident, uploaded=uploaded_objects)
            done = self._post_v2_complete(
                f"{repo_path}/publishes/{urllib.parse.quote(publish_id, safe='')}/complete")

            if done.status_code < 200 or done.status_code >= 300:
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
                if not response_is_from_hub(done):
                    raise _publish_refusal(
                        f"publish {publish_id}: the hub never answered complete — "
                        f"{FRONT_DOOR_UNAVAILABLE} ({_non_hub_origin(done)}). "
                        "Every declared object is staged and audited; the "
                        "session is live and /complete is idempotent, so a "
                        "retry resumes it without re-uploading a byte",
                        status=done.status_code, code=FRONT_DOOR_UNAVAILABLE,
                        retryable=True)
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

        if journal is not None:
            journal.clear(session_name, session_id=publish_id)
        if recovery is not None:
            recovery.clear(publish_id)
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
    """Build CommitFile entries for every regular file under ``tree``."""
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
