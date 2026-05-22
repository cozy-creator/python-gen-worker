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

from .api.decorators import Resources
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
        min_compute_capability=str(getattr(rc, "min_compute_capability", "") or ""),
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
    resources: Resources
    ctx_param: str
    payload_param: str
    payload_type: type[msgspec.Struct]
    output_mode: str  # "single" | "incremental"
    output_type: Optional[type[msgspec.Struct]] = None
    delta_type: Optional[type[msgspec.Struct]] = None
    # `injections` carries the per-parameter binding specs; type is left as
    # ``tuple[Any, ...]`` here to avoid a circular import on the worker-side
    # ``InjectionSpec`` (it's defined alongside the Worker class).
    injections: Tuple[Any, ...] = ()
    input_schema_json: bytes = b""
    output_schema_json: bytes = b""
    delta_schema_json: Optional[bytes] = None
    injection_json: bytes = b""


@dataclass(frozen=True)
class _BatchedWorkerSpec:
    """Dispatch entry for one ``@inference.function`` method on a
    ``BatchedWorker`` (continuous-batching) class (#273).

    Each entry binds the externally-addressable function name to:
      - ``instance`` — the singleton BatchedWorker class instance (one per
        worker process; created during startup).
      - ``method`` — the bound async / async-generator method on
        ``instance`` (``@inference.function``-decorated).
      - ``payload_type`` — msgspec.Struct type used to decode the wire
        ``input_payload`` bytes.
      - ``delta_type`` — msgspec.Struct type the method yields. Worker
        encodes each yielded item as JSON for the IncrementalTokenDelta
        wire envelope.
      - ``ctx_param`` / ``payload_param`` — keyword names the method
        expects (matches @inference.function signature
        ``async def fn(self, ctx, payload)``).

    Unlike ``_RequestSpec`` we do NOT carry ``injections`` — model
    download is owned by the SDK lifecycle (``setup(engine=...)``), not
    per-request injection. The engine instance + class instance both
    live on the Worker object alongside this spec.
    """

    name: str
    instance: Any
    method: Callable[..., Any]
    resources: Resources
    ctx_param: str
    payload_param: str
    payload_type: type[msgspec.Struct]
    delta_type: type[msgspec.Struct]
    runtime: str  # "sglang" | "vllm"
    input_schema_json: bytes = b""
    delta_schema_json: bytes = b""
    timeout_ms: Optional[int] = None


@dataclass(frozen=True)
class _SerialWorkerSpec:
    """Dispatch entry for one ``@inference.function`` method on a
    ``SerialWorker`` (sync) class (#322/#328).

    Mirrors ``_BatchedWorkerSpec`` but for the synchronous archetype:
    setup/warmup/shutdown + invocable methods are all plain sync callables.
    One request fully owns the GPU until done — no continuous-batching
    engine, no asyncio loop.

    Each entry binds the externally-addressable function name to:
      - ``instance`` — the singleton SerialWorker class instance (one per
        worker process; created during discovery, ``setup(self, **models)``
        runs lazily on first dispatch with model_path-resolved bindings).
      - ``method`` — the bound sync method on ``instance`` (or a sync
        generator yielding deltas when ``output_mode == "incremental"``).
      - ``payload_type`` — msgspec.Struct type used to decode the wire
        ``input_payload`` bytes.
      - ``output_type`` / ``delta_type`` — msgspec.Struct return / yield
        type. ``output_type`` is set when ``output_mode == "single"``;
        ``delta_type`` is set when ``output_mode == "incremental"``.
      - ``ctx_param`` / ``payload_param`` — keyword names the method
        expects (matches @inference.function signature
        ``def fn(self, ctx, payload)``).

    Unlike ``_RequestSpec`` we don't carry ``injections`` — models are
    loaded once via ``setup(**models)``, then re-used across requests.
    The class instance lives on the Worker object alongside this spec
    (one record per class in ``_serial_class_instances``).
    """

    name: str
    instance: Any
    method: Callable[..., Any]
    resources: Resources
    ctx_param: str
    payload_param: str
    payload_type: type[msgspec.Struct]
    output_mode: str  # "single" | "incremental"
    output_type: Optional[type[msgspec.Struct]] = None
    delta_type: Optional[type[msgspec.Struct]] = None
    input_schema_json: bytes = b""
    output_schema_json: bytes = b""
    delta_schema_json: Optional[bytes] = None
    timeout_ms: Optional[int] = None
    # #345 Improvement B: True when the bound method is an `async def` handler
    # (coroutine or async generator), derived from inspect at registration —
    # never tenant-declared. Async handlers run on the worker's shared asyncio
    # loop (`_batched_loop`) so I/O-bound endpoints scale to thousands of
    # concurrent coroutines instead of being bounded by the ThreadPoolExecutor.
    is_async: bool = False


@dataclass(frozen=True)
class _ConversionWorkerSpec:
    """Dispatch entry for one ``@conversion.function`` / ``@training.function``
    / ``@dataset.function`` method on a sync class endpoint (#332).

    Conversion-kind class endpoints are structurally SerialWorker
    (sync setup → generate → shutdown) but the result emission path
    differs: instead of an msgspec.Struct return decoded back to the
    caller, the method returns ``Iterator[ProducedFlavor]`` or
    ``list[ProducedFlavor]`` and the library uploads each flavor +
    publishes a revision via ``_finalize_produced_variants``.

    Fields mirror ``_SerialWorkerSpec`` (instance, method, payload_type,
    ctx/payload param names) plus conversion-specific metadata:

      - ``endpoint_kind`` — top-level decorator kind: ``"conversion"`` /
        ``"training"`` / ``"dataset"``. Picks the result-emission lane.
      - ``sub_kind`` — granular label declared on the class decorator
        (e.g. ``"format-conversion"`` / ``"quantization"`` /
        ``"dataset-generation"``). Forwarded to
        ``_finalize_produced_variants`` so the relationship_map +
        dataset-publish branch keep working.
      - ``ctx_class`` — the kind-specific RequestContext subclass to
        instantiate for this request (ConversionContext / DatasetContext).
      - ``calibration`` — per-scheme calibration policy carried over
        from the function-shape ``@conversion(calibration=...)``
        contract; enforced at dispatch time for unsupported schemes.

    Returns are always single-mode (one upload+publish per request);
    incremental streaming on conversion isn't supported.
    """

    name: str
    instance: Any
    method: Callable[..., Any]
    resources: Resources
    ctx_param: str
    payload_param: str
    payload_type: type[msgspec.Struct]
    endpoint_kind: str  # "conversion" | "training" | "dataset"
    sub_kind: str = ""
    ctx_class: Any = None  # type[RequestContext] subclass (resolved at register-time)
    calibration: Dict[str, str] = None  # type: ignore[assignment]
    input_schema_json: bytes = b""
    output_schema_json: bytes = b""
    timeout_ms: Optional[int] = None


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


def _binding_canonical_ref(entry: Dict[str, Any]) -> str:
    """Build the canonical bare ref string for a binding entry from
    endpoint.lock (post-issue #9 manifest shape).

    Tensorhub refs render bare; HF / civitai refs also render bare since the
    runtime wire format keys by the bare ref string. Provider is the side
    channel — see ``build_provider_index_from_manifest``.

    Entries are ``{provider, ref, flavor?, tag?, ...}``. Missing/empty
    tag → "latest" (tensorhub default); HF refs ignore tag entirely.
    """
    ref = str(entry.get("ref") or "").strip()
    if not ref:
        return ""
    provider = str(entry.get("provider") or "").strip() or "tensorhub"
    flavor = str(entry.get("flavor") or "").strip()
    tag = str(entry.get("tag") or "").strip()

    if provider == "tensorhub":
        # Match TensorhubRef.canonical() shape: owner/repo:tag[#flavor].
        if "@" in ref or ":" in ref:
            # Ref already encodes tag/digest — don't double-stamp.
            out = ref
        else:
            out = f"{ref}:{tag or 'latest'}"
        if flavor and "#" not in out:
            out = f"{out}#{flavor}"
        return out

    # HF / civitai keep the bare ref (optionally with flavor).
    out = ref
    if flavor and "#" not in out:
        out = f"{out}#{flavor}"
    return out


def _collect_binding_entries(bindings: Any) -> list[Dict[str, Any]]:
    """Flatten a function's ``bindings`` block into a list of leaf entries
    that each carry ``{provider, ref, ...}``.

    ``bindings`` is shaped ``{param_name: <entry>}`` where each entry is
    either ``kind: fixed`` (one ref) or ``kind: dispatch`` (a ``table`` of
    variant → entry). Other shapes are ignored defensively.
    """
    out: list[Dict[str, Any]] = []
    if not isinstance(bindings, dict):
        return out
    for _param_name, entry in bindings.items():
        if not isinstance(entry, dict):
            continue
        kind = str(entry.get("kind") or "").strip()
        if kind == "dispatch":
            table = entry.get("table")
            if isinstance(table, dict):
                for _variant, sub in table.items():
                    if isinstance(sub, dict):
                        out.append(sub)
            continue
        # kind == "fixed" or older shape that puts ref/provider at top level.
        if entry.get("ref"):
            out.append(entry)
    return out


def build_provider_index_from_manifest(manifest: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Walk the loaded endpoint.lock manifest and return a
    ``{bare_ref_string: provider}`` index.

    Issue #17: the runtime wire format passes refs as bare strings without
    a side-channel provider field. The downloader and other parse-time
    sites consult this index (set as a contextvar by the worker before
    invoking the model manager) to recover the provider for a bare ref.

    Refs not in the manifest (e.g. invoker-supplied overrides) default to
    ``"tensorhub"`` at the consuming site — match the wire-format contract.
    """
    index: Dict[str, str] = {}
    if not isinstance(manifest, dict):
        return index
    functions = manifest.get("functions")
    if not isinstance(functions, list):
        return index
    for fn in functions:
        if not isinstance(fn, dict):
            continue
        bindings = fn.get("bindings")
        for entry in _collect_binding_entries(bindings):
            ref_key = _binding_canonical_ref(entry)
            if not ref_key:
                continue
            provider = str(entry.get("provider") or "").strip() or "tensorhub"
            # First-write-wins: tenant-declared providers shouldn't conflict
            # across functions for the same ref, but if they do the first
            # entry (deterministic by function discovery order) takes priority.
            index.setdefault(ref_key, provider)
            # Also index the bare ref without tag — runtime payloads may
            # canonicalize "owner/repo:latest#bf16" or strip tags. The
            # provider is the same regardless of tag-form variations.
            ref_bare = str(entry.get("ref") or "").strip()
            flavor = str(entry.get("flavor") or "").strip()
            if ref_bare:
                if flavor:
                    alt = f"{ref_bare}#{flavor}"
                    index.setdefault(alt, provider)
                index.setdefault(ref_bare, provider)
    return index


class _AuthInterceptor(grpc.StreamStreamClientInterceptor):
    def __init__(self, token: str) -> None:
        self._token = token

    def intercept_stream_stream(self, continuation: Any, client_call_details: Any, request_iterator: Any) -> Any:
        metadata = list(client_call_details.metadata or [])
        metadata.append(('authorization', f'Bearer {self._token}'))
        new_details = client_call_details._replace(metadata=metadata)
        return continuation(new_details, request_iterator)
