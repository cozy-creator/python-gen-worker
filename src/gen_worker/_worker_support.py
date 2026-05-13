"""Module-level support code for Worker: helpers, dataclasses, and the
auth-interceptor interface.

Pulled out of worker.py to keep that file focused on the Worker class.
"""

from __future__ import annotations

import collections.abc as cabc
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING

import grpc
import msgspec

from .api.decorators import ResourceRequirements
from .api.injection import InjectionSpec
from .api.types import Compute
from .request_context import _canonicalize_model_ref_string

if TYPE_CHECKING:
    pass  # forward refs only


def _workspace_scope_id(request_id: str, job_id: Optional[str]) -> str:
    jid = str(job_id or "").strip()
    if jid:
        return jid
    return str(request_id or "").strip()


def _extract_worker_capability_token(envelope: Any) -> str:
    return str(getattr(envelope, "worker_capability_token", "") or "").strip()


def _extract_resolved_compute(envelope: Any) -> Optional[Compute]:
    """Pull the ResolvedCompute protobuf field off a JobExecutionRequest /
    BatchExecutionItem envelope and return the
    ``gen_worker.Compute`` Python dataclass the tenant sees on
    ``RequestContext.compute``.

    Returns None when the orchestrator didn't attach resolved_compute (proto3
    message presence). Caller passes None through to RequestContext; the ctx
    substitutes a sentinel default.
    """
    # #321: Use proto3 message-presence (HasField) instead of an
    # AND-of-eight-zeros guard. The old check mis-classified legitimately-
    # zero-axis requests (gpu_count=0, cpu_cores=1 looked set; everything-zero
    # looked unset — the boundary was bug bait).
    try:
        if not envelope.HasField("resolved_compute"):
            return None
    except (AttributeError, ValueError):
        # AttributeError: envelope without HasField (defensive; not expected).
        # ValueError: HasField called on a non-singular field.
        return None
    rc = envelope.resolved_compute
    tier = str(getattr(rc, "gpu_tier", "") or "") or None
    return Compute(
        accelerator=str(getattr(rc, "accelerator", "") or ""),
        cuda_compute_min=str(getattr(rc, "cuda_compute_min", "") or ""),
        vram_gb=int(getattr(rc, "vram_gb", 0) or 0),
        gpu_count=int(getattr(rc, "gpu_count", 0) or 0),
        gpu_tier=tier,
        memory_gb=int(getattr(rc, "memory_gb", 0) or 0),
        cpu_cores=int(getattr(rc, "cpu_cores", 0) or 0),
        disk_gb=int(getattr(rc, "disk_gb", 0) or 0),
    )


# #321: _normalize_materialized_input_urls removed — read fields
# (`input_ref_urls`, `input_ref_urls_json`) that don't exist in the proto.
# Always returned {}. Orchestrator passes URLs inside `input_payload` msgpack.


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
        # endpoint.lock shape: ``{ref}``. Flavor + pin-by-checkpoint_id are
        # encoded directly in the ref string (e.g. "owner/repo:tag#flavor"
        # or "owner/repo@blake3:<digest>"). No separate selector axes.
        ids[key] = ref
        specs[key] = {"ref": ref}
    return ids, specs


class _AuthInterceptor(grpc.StreamStreamClientInterceptor):
    def __init__(self, token: str) -> None:
        self._token = token

    def intercept_stream_stream(self, continuation: Any, client_call_details: Any, request_iterator: Any) -> Any:
        metadata = list(client_call_details.metadata or [])
        metadata.append(('authorization', f'Bearer {self._token}'))
        new_details = client_call_details._replace(metadata=metadata)
        return continuation(new_details, request_iterator)
