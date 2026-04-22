"""Module-level support code for Worker: helpers, dataclasses, and the
auth-interceptor / RealtimeSocket interfaces.

Pulled out of worker.py to keep that file focused on the Worker class. The
`_RealtimeSocketAdapter` lives in worker.py because it's tightly bound to
the protobuf WorkerSchedulerMessage types.
"""

from __future__ import annotations

import asyncio
import collections.abc as cabc
import json
import threading
import typing
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING

import grpc
import msgspec

from .api.decorators import ResourceRequirements
from .api.injection import InjectionSpec
from .request_context import RequestContext, _canonicalize_model_ref_string

if TYPE_CHECKING:
    pass  # forward refs only


def _workspace_scope_id(request_id: str, job_id: Optional[str]) -> str:
    jid = str(job_id or "").strip()
    if jid:
        return jid
    return str(request_id or "").strip()


def _extract_worker_capability_token(envelope: Any) -> str:
    return str(getattr(envelope, "worker_capability_token", "") or "").strip()


def _normalize_materialized_input_urls(envelope: Any) -> Dict[str, str]:
    """Collect materialized input-ref URLs from a scheduler envelope.

    Supports both `input_ref_urls` (protobuf map) and the legacy
    `input_ref_urls_json` (JSON-string) shape; merges both with the JSON
    shape overriding the map on collision. Every key is leading-slash-
    stripped; empty keys/values are dropped. Unparseable JSON is ignored
    silently.

    Used by both `_handle_job_request` and `_handle_realtime_open_cmd`
    — prior copy-paste.
    """
    out: Dict[str, str] = {}

    raw_urls_map = getattr(envelope, "input_ref_urls", None)
    if isinstance(raw_urls_map, cabc.Mapping):
        for k, v in raw_urls_map.items():
            ks = str(k or "").strip().lstrip("/")
            vs = str(v or "").strip()
            if ks and vs:
                out[ks] = vs

    raw_urls = getattr(envelope, "input_ref_urls_json", None)
    if isinstance(raw_urls, str) and raw_urls.strip():
        try:
            parsed = json.loads(raw_urls)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                ks = str(k or "").strip().lstrip("/")
                vs = str(v or "").strip()
                if ks and vs:
                    out[ks] = vs

    return out


def _extract_checkpoint_id_from_result(result: Any) -> str:
    """Best-effort: find the snapshot_digest of the produced checkpoint inside
    a ConversionOutput-like struct so the library can tag it. Returns "" when
    no digest is findable — caller logs a warning and skips tag apply.

    Recognizes:
      - result.weights.snapshot_digest (ConversionOutput with single Tensors)
      - result.weights[0].snapshot_digest (ConversionOutput with list[Tensors])
      - result.checkpoint_id (generic output carrying a digest string)
    """
    if result is None:
        return ""
    # Direct checkpoint_id attribute.
    cid = getattr(result, "checkpoint_id", None)
    if isinstance(cid, str) and cid.strip():
        return cid.strip()
    weights = getattr(result, "weights", None)
    if weights is not None:
        if isinstance(weights, (list, tuple)):
            for w in weights:
                d = getattr(w, "snapshot_digest", None)
                if isinstance(d, str) and d.strip():
                    return d.strip()
        else:
            d = getattr(weights, "snapshot_digest", None)
            if isinstance(d, str) and d.strip():
                return d.strip()
    return ""


@dataclass(frozen=True)
class _RequestSpec:
    name: str
    func: Callable[..., Any]
    resources: ResourceRequirements
    ctx_param: str
    payload_param: str
    payload_type: type[msgspec.Struct]
    output_mode: str  # "single" | "incremental"
    output_type: Optional[type[msgspec.Struct]] = None
    delta_type: Optional[type[msgspec.Struct]] = None
    injections: Tuple[InjectionSpec, ...] = ()
    input_schema_json: bytes = b""
    output_schema_json: bytes = b""
    delta_schema_json: Optional[bytes] = None
    injection_json: bytes = b""


@dataclass(frozen=True)
class _WebsocketSpec:
    name: str
    func: Callable[..., Any]
    resources: ResourceRequirements
    ctx_param: str
    socket_param: str
    injections: Tuple[InjectionSpec, ...] = ()


class RealtimeSocket:
    """
    Worker-owned socket interface for realtime handlers (no FastAPI dependency).
    """

    async def send_bytes(self, data: bytes) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def send_json(self, obj: Any) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def iter_bytes(self) -> typing.AsyncIterator[bytes]:  # pragma: no cover - interface
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class _RealtimeSessionState:
    session_id: str
    spec: _WebsocketSpec
    ctx: RequestContext
    loop: asyncio.AbstractEventLoop
    in_q: "asyncio.Queue[Optional[bytes]]"
    closed: threading.Event


def _parse_manifest_model_mapping(mapping: Dict[str, Any]) -> tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    ids: Dict[str, str] = {}
    specs: Dict[str, Dict[str, Any]] = {}
    for k, v in mapping.items():
        key = str(k).strip()
        if not key or not isinstance(v, dict):
            continue
        ref = _canonicalize_model_ref_string(str(v.get("ref") or "").strip())
        if not ref:
            continue
        dtypes = v.get("dtypes")
        ids[key] = ref
        specs[key] = {"ref": ref, "dtypes": [str(x) for x in dtypes if str(x).strip()] if isinstance(dtypes, list) else []}
    return ids, specs


class _AuthInterceptor(grpc.StreamStreamClientInterceptor):
    def __init__(self, token: str) -> None:
        self._token = token

    def intercept_stream_stream(self, continuation: Any, client_call_details: Any, request_iterator: Any) -> Any:
        metadata = list(client_call_details.metadata or [])
        metadata.append(('authorization', f'Bearer {self._token}'))
        new_details = client_call_details._replace(metadata=metadata)
        return continuation(new_details, request_iterator)
