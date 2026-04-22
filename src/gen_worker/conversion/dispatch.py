"""@conversion_function — unified decorator for transform-kind endpoints.

Signature-introspected dispatch: reserved parameter names bound to library-
injected types (ctx, source, datasets), everything else decoded by msgspec
from the wire payload by name. Supports ``Annotated[Source, ModelRef(
Src.PAYLOAD, '<wire_field>')]`` to declare secondary models the library
materializes alongside the primary source.

At decorator time:
  - Reject reserved-name + non-reserved-type combos (TypeError).
  - Require ``ctx`` + ``source``.
  - Build a per-function ``ref_registry`` for orchestrator token scoping:
    maps wire_field → parameter_name for every Annotated[Source, ModelRef(
    Src.PAYLOAD, ...)] param.

At dispatch time:
  - Build reserved-name injected values (ctx, source, datasets).
  - For each non-reserved param: decode from payload[name] using msgspec,
    or materialize via ModelRef-PAYLOAD if annotated.
  - Call fn(**kwargs) with only the names the tenant declared.
  - Upload each returned ProducedVariant; apply destination.tags on success.

See e2e progress.json issue #5 for the full contract.
"""

from __future__ import annotations

import inspect
import typing
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING, get_args, get_origin

import msgspec

from ..api.decorators import ResourceRequirements
from ..api.injection import ModelRef, ModelRefSource, parse_injection
from .context import ConversionContext
from .dataset import Dataset
from .produced import ProducedVariant
from .source import Source

if TYPE_CHECKING:
    from ..request_context import RequestContext


# Reserved parameter names and the exact types they must bind to.
# Reserved NAME with wrong TYPE → TypeError at decorator time.
RESERVED_TYPES: dict[str, Any] = {
    "ctx": ConversionContext,
    "source": Source,
    "datasets": list[Dataset],
}


# Recommended coarse labels for the training_jobs.kind column (issue #10).
# Free-form: tenants MAY use other values, and MAY use ``colon:sub-label`` to
# pick a sub-bucket. Recommended set is validated for typos with a warning.
RECOMMENDED_KINDS = frozenset({
    "quantization",
    "pruning",
    "distillation",
    "fine-tuning",
    "fusion",
    "format-conversion",
})

# Default when @conversion_function is used without kind=. Fine-tuning is
# the most common case for new tenants; quantization / format-conversion /
# fusion endpoints must declare explicitly.
DEFAULT_KIND = "fine-tuning"


class ConversionFunctionSpec:
    """Per-function metadata built at decorator time.

    ``ref_registry`` maps wire-field-name -> parameter-name for every
    ``Annotated[Source, ModelRef(Src.PAYLOAD, key)]`` parameter. Orchestrator
    reads this at publish to know which wire fields to include in the
    capability-token reads set.

    ``input_schema`` is the JSON schema for the full wire payload — library
    bakes it into endpoint.lock at publish; orchestrator validates incoming
    SubmitRequest payloads against it BEFORE minting capability tokens /
    dispatching to a worker. Catches ``specs[0].dtype='fp15'``-type enum
    violations at HTTP boundary instead of mid-job on the worker.

    ``kind`` is the tenant-declared job-kind label that populates
    ``orchestrator.training_jobs.kind`` on dispatch (issue #10). Free-form
    string; see ``RECOMMENDED_KINDS`` for the coarse label set.
    """

    __slots__ = ("fn", "signature", "ref_registry", "other_params", "input_schema", "kind")

    def __init__(
        self,
        *,
        fn: Callable[..., list[ProducedVariant]],
        signature: inspect.Signature,
        ref_registry: dict[str, str],
        other_params: dict[str, inspect.Parameter],
        input_schema: dict,
        kind: str,
    ) -> None:
        self.fn = fn
        self.signature = signature
        # wire_field -> param_name
        self.ref_registry = ref_registry
        # param_name -> Parameter (for non-reserved, non-ref params — decoded
        # from payload by msgspec)
        self.other_params = other_params
        # Full wire-payload JSON schema (dict). Includes source, destination,
        # datasets (if declared), every ModelRef-PAYLOAD field, and every
        # tenant-named param. Enum types come through via msgspec Literal
        # → JSON schema `enum` mapping.
        self.input_schema = input_schema
        self.kind = kind


def conversion_function(
    fn: Callable[..., list[ProducedVariant]] | None = None,
    *,
    kind: str = DEFAULT_KIND,
) -> Callable[..., list[ProducedVariant]]:
    """Mark a tenant function as a conversion/training endpoint.

    Usable as a plain decorator (``@conversion_function``) or with a kind
    argument (``@conversion_function(kind='quantization')``). The kind
    populates ``training_jobs.kind`` at dispatch (issue #10); see
    ``RECOMMENDED_KINDS`` for the coarse label set. Sub-labels via
    ``kind='quantization:gptq-w4'`` are permitted.

    At decorator time: validates the signature and attaches a
    ``ConversionFunctionSpec`` to the function as ``__conversion_spec__``.
    Raises TypeError for reserved-name/wrong-type violations or missing
    required params.

    The returned callable is a dispatch wrapper the library invokes with
    ``(request_context, payload)`` at runtime.
    """
    # Support both @conversion_function and @conversion_function(kind=...).
    if fn is None:
        def _apply(real_fn: Callable[..., list[ProducedVariant]]) -> Callable[..., list[ProducedVariant]]:
            return _build_dispatch(real_fn, kind=kind)
        return _apply  # type: ignore[return-value]
    return _build_dispatch(fn, kind=kind)


def _validate_kind(fn_name: str, kind: str) -> None:
    """Warn on typos against RECOMMENDED_KINDS. Free-form labels are allowed."""
    import warnings
    if not isinstance(kind, str) or not kind.strip():
        raise TypeError(f"{fn_name}: @conversion_function kind= must be a non-empty string")
    # Split off sub-label if present (e.g. 'quantization:gptq-w4').
    coarse = kind.split(":", 1)[0]
    if coarse not in RECOMMENDED_KINDS:
        warnings.warn(
            f"{fn_name}: @conversion_function kind={kind!r} is not in the "
            f"recommended set {sorted(RECOMMENDED_KINDS)}. "
            f"Free-form is allowed, but check for typos (did you mean 'fine-tuning'?).",
            stacklevel=3,
        )


def _build_dispatch(
    fn: Callable[..., list[ProducedVariant]],
    *,
    kind: str,
) -> Callable[..., list[ProducedVariant]]:
    """Inner: build the dispatch wrapper. Split from ``conversion_function`` so
    the same body serves both ``@conversion_function`` and
    ``@conversion_function(kind=...)`` entry paths."""
    _validate_kind(fn.__name__, kind)
    # Resolve string-form annotations (from __future__ import annotations) so
    # reserved-type comparisons work against actual class objects, not strings.
    sig = inspect.signature(fn)
    try:
        type_hints = typing.get_type_hints(fn, include_extras=True)
    except Exception:
        type_hints = {}
    new_params = []
    for name, p in sig.parameters.items():
        if name in type_hints:
            p = p.replace(annotation=type_hints[name])
        new_params.append(p)
    sig = sig.replace(parameters=new_params)
    ref_registry: dict[str, str] = {}
    other_params: dict[str, inspect.Parameter] = {}

    for name, p in sig.parameters.items():
        if name in RESERVED_TYPES:
            expected = RESERVED_TYPES[name]
            if p.annotation is inspect.Parameter.empty:
                raise TypeError(
                    f"{fn.__name__}: reserved param '{name}' must be typed as "
                    f"{_type_repr(expected)} (got no annotation)"
                )
            if not _annotation_matches(p.annotation, expected):
                raise TypeError(
                    f"{fn.__name__}: reserved param '{name}' must be typed as "
                    f"{_type_repr(expected)}; got {_type_repr(p.annotation)}"
                )
            continue
        # Non-reserved: is it an Annotated[..., ModelRef(Src.PAYLOAD, ...)] ?
        parsed = parse_injection(p.annotation)
        if parsed is not None:
            _base, ref = parsed
            if ref.source == ModelRefSource.PAYLOAD:
                ref_registry[ref.key] = name
                continue
        # Plain tenant-named param — library decodes from payload at dispatch
        other_params[name] = p

    for required in ("ctx", "source"):
        if required not in sig.parameters:
            raise TypeError(
                f"{fn.__name__}: must declare required parameter '{required}'"
            )

    input_schema = _build_wire_payload_schema(
        signature=sig,
        other_params=other_params,
        ref_registry=ref_registry,
    )

    spec = ConversionFunctionSpec(
        fn=fn,
        signature=sig,
        ref_registry=ref_registry,
        other_params=other_params,
        input_schema=input_schema,
        kind=kind,
    )

    def dispatch(request_context: "RequestContext", payload: Any) -> list[ProducedVariant]:
        return _run(spec, request_context, payload)

    # Attach metadata so discovery / publish-time validation can read it.
    # Setting _is_conversion_function + _worker_resources makes the existing
    # gen-worker discovery pick this up as a registered handler. Discovery
    # treats _is_conversion_function specially — it doesn't try to parse the
    # dispatch wrapper's (request_context, payload) signature as a regular
    # worker function; instead it reads ConversionFunctionSpec directly.
    dispatch.__conversion_spec__ = spec  # type: ignore[attr-defined]
    dispatch.__wrapped__ = fn  # type: ignore[attr-defined]
    dispatch.__name__ = fn.__name__
    dispatch.__doc__ = fn.__doc__
    dispatch._is_conversion_function = True  # type: ignore[attr-defined]
    dispatch._worker_resources = ResourceRequirements(kind="conversion")  # type: ignore[attr-defined]
    return dispatch


# ---------------------------------------------------------------------------
# dispatch-time runtime helpers
# ---------------------------------------------------------------------------


def _run(
    spec: ConversionFunctionSpec,
    request_context: "RequestContext",
    payload: Any,
) -> list[ProducedVariant]:
    """Build kwargs from payload + inject library helpers; call the tenant."""
    source = _build_source(request_context, payload)
    ctx = ConversionContext(request_context=request_context, source=source)
    datasets = _build_datasets(request_context, payload)

    injected: dict[str, Any] = {
        "ctx": ctx,
        "source": source,
        "datasets": datasets,
    }

    raw_fields = _payload_as_dict(payload)
    kwargs: dict[str, Any] = {}
    sig = spec.signature
    for name, p in sig.parameters.items():
        if name in injected:
            kwargs[name] = injected[name]
            continue
        # ModelRef-PAYLOAD secondary-model load
        parsed = parse_injection(p.annotation)
        if parsed is not None:
            base, ref = parsed
            if ref.source == ModelRefSource.PAYLOAD:
                kwargs[name] = _materialize_secondary_source(
                    request_context, raw_fields, ref.key, base,
                    default_is_optional=(p.default is not inspect.Parameter.empty),
                    default=p.default,
                )
                continue
        # Tenant-named param — decode from payload
        if name in raw_fields:
            kwargs[name] = msgspec.convert(raw_fields[name], type=p.annotation)
        elif p.default is not inspect.Parameter.empty:
            kwargs[name] = p.default
        else:
            raise ValueError(
                f"{spec.fn.__name__}: missing required wire field '{name}' in payload"
            )

    result = spec.fn(**kwargs)
    if not isinstance(result, list):
        raise TypeError(
            f"{spec.fn.__name__}: must return list[ProducedVariant]; got "
            f"{type(result).__name__}"
        )
    # Upload each ProducedVariant to the destination + apply tags on success.
    # Library appends exactly one provenance key (`produced_by_job_id`) — see
    # e2e progress.json #5 for the rationale (every other input-to-the-job
    # lives in the orchestrator job record; duplicating onto the variant
    # drifts).
    _finalize_produced_variants(request_context, result, kind=spec.kind)
    return result


def _finalize_produced_variants(
    request_context: "RequestContext",
    variants: list[ProducedVariant],
    *,
    kind: str,
) -> None:
    """Upload every variant, append library provenance, apply destination.tags.

    Uploads via ``ctx.save_checkpoint(attributes=...)`` when the variant path
    is a single file; walks the directory tree for dir-shaped variants
    (save_pretrained / diffusers component trees). For each upload:

      - Tenant-declared ``ProducedVariant.attributes`` passed through.
      - Library appends ``produced_by_job_id`` pointing at request_context.job_id
        (or request_id as fallback). One join-key is enough; the orchestrator
        job record carries source ref / datasets / specs / hyperparameters /
        timestamps — duplicating them onto the variant drifts.
      - Tenant ``@conversion_function(kind=...)`` label propagates as
        ``produced_by_kind`` for observability (issue #10).

    After all variants upload, applies each ``ctx.destination.tags`` entry to
    the produced checkpoint. Tag-move failures surface as job warnings
    (upload already succeeded); the per-destination-tag move endpoint on
    tensorhub is currently called via a non-exported attribute of the
    RequestContext — the host framework wires this in.
    """
    if not variants:
        return

    job_id = str(getattr(request_context, "job_id", "") or "") \
        or str(getattr(request_context, "request_id", "") or "")
    library_provenance: dict[str, str] = {"produced_by_job_id": job_id}
    if kind:
        library_provenance["produced_by_kind"] = kind

    for variant in variants:
        merged_attrs = {**library_provenance, **dict(variant.attributes or {})}
        path = variant.path
        if path.is_file():
            _upload_single_file_variant(
                request_context, path, attributes=merged_attrs,
            )
        elif path.is_dir():
            _upload_directory_variant(
                request_context, path, attributes=merged_attrs,
            )
        else:
            raise FileNotFoundError(f"ProducedVariant.path does not exist: {path}")

    # Tag application runs after every upload succeeds so partial-tag moves
    # can't happen. ctx.destination exposed as an attribute on request_context
    # by the host framework; fall back to no-op if absent (tests).
    destination = getattr(request_context, "destination", None)
    if destination is None:
        return
    tags = list(getattr(destination, "tags", None) or [])
    ref = str(getattr(destination, "ref", "") or "")
    apply_fn = getattr(request_context, "apply_destination_tag", None)
    if not (tags and ref and apply_fn is not None):
        return
    for tag in tags:
        try:
            apply_fn(ref, str(tag))
        except Exception as exc:  # noqa: BLE001 — best-effort; surface as job warning
            emit = getattr(request_context, "emit", None)
            if emit is not None:
                emit("transform.tag_apply_failed", {
                    "ref": ref, "tag": str(tag),
                    "error": f"{type(exc).__name__}: {exc}",
                })


def _upload_single_file_variant(
    request_context: "RequestContext",
    path: Path,
    *,
    attributes: dict[str, str],
) -> None:
    """Upload one safetensors/gguf/flashpack file, threading attributes through.

    Graceful when ``request_context.save_checkpoint`` is missing: tests that
    use a mock RequestContext without the upload machinery skip the upload
    silently. Production RequestContexts always have it.
    """
    save_fn = getattr(request_context, "save_checkpoint", None)
    if save_fn is None:
        return
    ref = _default_variant_ref(request_context, path)
    format_hint = path.suffix.lstrip(".") or "bin"
    try:
        save_fn(ref, str(path), format=format_hint, attributes=attributes)
    except TypeError:
        # Older RequestContext builds without the attributes kwarg — fall back.
        save_fn(ref, str(path), format=format_hint)


def _upload_directory_variant(
    request_context: "RequestContext",
    dir_path: Path,
    *,
    attributes: dict[str, str],
) -> None:
    """Upload every file under a directory variant (diffusers tree / save_pretrained output)."""
    save_fn = getattr(request_context, "save_checkpoint", None)
    if save_fn is None:
        return
    base_ref = _default_variant_ref(request_context, dir_path)
    for f in sorted(dir_path.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(dir_path).as_posix()
        ref = f"{base_ref}/{rel}"
        format_hint = f.suffix.lstrip(".") or "bin"
        try:
            save_fn(ref, str(f), format=format_hint, attributes=attributes)
        except TypeError:
            save_fn(ref, str(f), format=format_hint)


def _default_variant_ref(request_context: "RequestContext", path: Path) -> str:
    """Return a job-scoped ref for a produced variant."""
    job_id = str(getattr(request_context, "job_id", "") or "") \
        or str(getattr(request_context, "request_id", "r")) \
        or "r"
    return f"jobs/{job_id}/outputs/{path.name}"


def _build_source(request_context: "RequestContext", payload: Any) -> Source:
    """Construct a Source from the materialized snapshot + resolved attributes."""
    source_path = getattr(request_context, "source_path", None)
    if source_path is None:
        raise RuntimeError(
            "@conversion_function requires request_context.source_path to be set "
            "(endpoint_kind must be 'transform'/'conversion'/'training' and the "
            "gen-worker reserved-name dispatch must have materialized the source)"
        )
    ctx_source = getattr(request_context, "source", None)
    attrs: dict = {}
    ref = ""
    if ctx_source is not None:
        attrs = dict(getattr(ctx_source, "attributes", {}) or {})
        ref = str(getattr(ctx_source, "ref", "") or "")
    return Source(Path(source_path), attributes=attrs, ref=ref)


def _build_datasets(request_context: "RequestContext", payload: Any) -> list[Dataset]:
    """Materialize each DatasetRef in payload.datasets into a Dataset."""
    raw = _payload_as_dict(payload).get("datasets") or []
    out: list[Dataset] = []
    for entry in raw:
        if isinstance(entry, dict):
            ref = str(entry.get("ref", ""))
            split = str(entry.get("split", "train") or "train")
            attributes = dict(entry.get("attributes") or {})
        else:
            ref = str(getattr(entry, "ref", ""))
            split = str(getattr(entry, "split", "train"))
            attributes = dict(getattr(entry, "attributes", {}) or {})
        # TODO(dataset-materialization): once gen-worker's ref-downloader learns
        # dataset resolution, look up the materialized path here. For MVP, the
        # host framework injects dataset_paths by ref onto the request_context.
        dataset_paths = getattr(request_context, "dataset_paths", {}) or {}
        local_path = dataset_paths.get(ref)
        if local_path is None:
            raise RuntimeError(
                f"dataset {ref!r} not materialized: request_context.dataset_paths "
                f"missing this ref. Orchestrator must have resolved + materialized "
                f"each payload.datasets entry before dispatching the tenant function."
            )
        out.append(Dataset(
            ref=ref, split=split, path=Path(local_path), attributes=attributes,
        ))
    return out


def _materialize_secondary_source(
    request_context: "RequestContext",
    raw_fields: dict,
    wire_field: str,
    base_type: Any,
    *,
    default_is_optional: bool,
    default: Any,
) -> Any:
    """Materialize a secondary Source declared via Annotated[Source, ModelRef(Src.PAYLOAD, key)]."""
    if wire_field not in raw_fields:
        if default_is_optional:
            return default
        raise ValueError(
            f"secondary-model wire field '{wire_field}' missing from payload"
        )
    ref_value = raw_fields[wire_field]
    # Accept a bare string or a SourceRepo-shaped dict.
    if isinstance(ref_value, str):
        ref_str = ref_value
        attributes: dict = {}
    elif isinstance(ref_value, dict):
        ref_str = str(ref_value.get("ref", ""))
        attributes = dict(ref_value.get("attributes") or {})
    else:
        ref_str = str(getattr(ref_value, "ref", ""))
        attributes = dict(getattr(ref_value, "attributes", {}) or {})
    # TODO(secondary-model-materialization): delegate to the host's
    # capability-token-aware ref downloader. For MVP, the host framework
    # injects a dict on the request_context mapping ref -> local snapshot path.
    secondary_paths = getattr(request_context, "secondary_source_paths", {}) or {}
    local_path = secondary_paths.get(ref_str)
    if local_path is None:
        raise RuntimeError(
            f"secondary-model ref {ref_str!r} not materialized: "
            f"request_context.secondary_source_paths missing this ref. "
            f"Orchestrator must have pre-scoped the capability token and the "
            f"worker must have resolved+downloaded the snapshot before dispatching."
        )
    # For now all secondary models materialize as a Source (the base type in
    # Annotated[Source, ModelRef(...)]). Future richer types (Pipeline,
    # PreTrainedModel) can be dispatched on base_type here.
    return Source(Path(local_path), attributes=attributes, ref=ref_str)


def _payload_as_dict(payload: Any) -> dict:
    """Extract a dict view of the payload for by-name field access."""
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, msgspec.Struct):
        # msgspec structs expose attributes directly; build a dict by field name
        return {f: getattr(payload, f) for f in payload.__struct_fields__}
    # Fallback: any object with __dict__
    if hasattr(payload, "__dict__"):
        return dict(payload.__dict__)
    raise TypeError(f"unsupported payload type: {type(payload).__name__}")


def _annotation_matches(annotation: Any, expected: Any) -> bool:
    """Compare two type annotations, treating list[X] specially for Python 3.9+ compat."""
    if annotation is expected:
        return True
    origin_a = get_origin(annotation)
    origin_e = get_origin(expected)
    if origin_a is not None and origin_a is origin_e:
        return get_args(annotation) == get_args(expected)
    return False


def _type_repr(t: Any) -> str:
    mod = getattr(t, "__module__", None)
    name = getattr(t, "__qualname__", None) or getattr(t, "__name__", None)
    if mod and name and mod not in ("builtins", "typing"):
        return f"{mod}.{name}"
    return repr(t)


def _build_wire_payload_schema(
    *,
    signature: inspect.Signature,
    other_params: dict[str, inspect.Parameter],
    ref_registry: dict[str, str],
) -> dict:
    """Build the full wire-payload JSON schema for this function.

    Properties:
      - source / destination: always present; schemas come from SourceRepo /
        DestinationRepo (msgspec structs).
      - datasets: present iff the tenant declared `datasets: list[Dataset]`.
        Schema = array of DatasetRef.
      - Every Annotated[Source, ModelRef(Src.PAYLOAD, 'field')] param maps
        to a wire field named `field`. Schema = SourceRepo.
      - Every tenant-named param (entries in other_params) maps to a wire
        field of that name. Schema = msgspec.json.schema(param.annotation).

    Required-fields: source, destination; each tenant-named param without a
    default; each ModelRef-PAYLOAD param without a default.
    """
    from ..api.types import DatasetRef, DestinationRepo, SourceRepo

    # Collect all types we need schemas for so msgspec.json.schema_components
    # can emit one consolidated $defs block.
    types_to_schema: list[Any] = [SourceRepo, DestinationRepo]
    properties: dict[str, dict] = {}
    required: list[str] = ["source", "destination"]

    # datasets
    has_datasets_param = "datasets" in signature.parameters
    if has_datasets_param:
        types_to_schema.append(DatasetRef)

    # Tenant-named params (specs, prompts, reward_model_ref, ...)
    for name, p in other_params.items():
        if p.annotation is not inspect.Parameter.empty:
            types_to_schema.append(p.annotation)
        if p.default is inspect.Parameter.empty:
            required.append(name)

    # ModelRef(Src.PAYLOAD, ...) params — wire field is SourceRepo-shaped
    for wire_field, param_name in ref_registry.items():
        p = signature.parameters[param_name]
        if p.default is inspect.Parameter.empty:
            required.append(wire_field)

    # Build consolidated schemas
    try:
        schemas_list, components = msgspec.json.schema_components(
            types_to_schema, ref_template="#/$defs/{name}",
        )
    except Exception:
        # Fall back to per-type schemas if schema_components fails for any reason
        schemas_list = [msgspec.json.schema(t) for t in types_to_schema]
        components = {}

    type_to_schema: dict[int, dict] = {}
    for t, s in zip(types_to_schema, schemas_list):
        type_to_schema[id(t)] = s

    # Top-level properties
    properties["source"] = type_to_schema[id(SourceRepo)]
    properties["destination"] = type_to_schema[id(DestinationRepo)]
    if has_datasets_param:
        properties["datasets"] = {
            "type": "array",
            "items": type_to_schema[id(DatasetRef)],
            "default": [],
        }
    for name, p in other_params.items():
        if p.annotation is inspect.Parameter.empty:
            continue
        properties[name] = type_to_schema.get(id(p.annotation), {"type": "object"})
    for wire_field, param_name in ref_registry.items():
        # ModelRef-PAYLOAD wire fields are SourceRepo-shaped dicts
        properties[wire_field] = type_to_schema[id(SourceRepo)]

    schema: dict = {
        "type": "object",
        "properties": properties,
        "required": required,
    }
    if components:
        schema["$defs"] = components
    return schema


__all__ = [
    "ConversionFunctionSpec",
    "RESERVED_TYPES",
    "conversion_function",
]
