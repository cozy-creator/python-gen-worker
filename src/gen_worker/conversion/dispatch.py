"""@training_function — unified decorator for transform-kind endpoints.

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
  - Upload each returned ProducedFlavor; apply destination.tags on success.

See e2e progress.json issue #5 for the full contract.
"""

from __future__ import annotations

import inspect
import logging
import typing
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING, get_args, get_origin

import msgspec

_log = logging.getLogger(__name__)

from ..api.decorators import ResourceRequirements
from ..api.injection import ModelRef, ModelRefSource, parse_injection
from .calibration import CalibrationPolicy, lookup_policy, validate_policy_map
from .context import ConversionContext
from .dataset import Dataset
from .produced import ProducedFlavor
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
    # E2E issue #41 — dataset-generation endpoints (calibration dataset
    # generator, eval-set builder, etc.) emit a dataset artifact, not a
    # model checkpoint, but share the @training_function dispatch path.
    "dataset-generation",
})

# Default when @training_function is used without kind=. Fine-tuning is
# the most common case for new tenants; quantization / format-conversion /
# fusion endpoints must declare explicitly.
DEFAULT_KIND = "fine-tuning"


class TrainingFunctionSpec:
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

    __slots__ = (
        "fn", "signature", "ref_registry", "other_params", "input_schema",
        "kind", "calibration",
    )

    def __init__(
        self,
        *,
        fn: Callable[..., list[ProducedFlavor]],
        signature: inspect.Signature,
        ref_registry: dict[str, str],
        other_params: dict[str, inspect.Parameter],
        input_schema: dict,
        kind: str,
        calibration: dict[str, CalibrationPolicy] | None = None,
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
        # Per-scheme calibration policy (e2e #41/#42). See
        # gen_worker.conversion.calibration for the three policies
        # (required / beneficial / unsupported) and the resolver helper.
        # Empty dict for non-quantization functions; discovery emits this
        # into endpoint.lock so orchestrator / UI can surface "needs a
        # dataset" to callers up-front.
        self.calibration: dict[str, CalibrationPolicy] = dict(calibration or {})


def training_function(
    fn: Callable[..., list[ProducedFlavor]] | None = None,
    *,
    kind: str = DEFAULT_KIND,
    concurrency: str = "sequential",
    label: str | None = None,
    description: str | None = None,
    calibration: dict[str, CalibrationPolicy] | None = None,
) -> Callable[..., list[ProducedFlavor]]:
    """Mark a tenant function as a training endpoint (tensorhub #232).

    Usable as a plain decorator (``@training_function``) or with kwargs
    (``@training_function(kind='quantization')``).

    Args:
        kind: Populates ``training_jobs.kind`` at dispatch (e2e issue #10).
            See ``RECOMMENDED_KINDS`` for the coarse label set; sub-labels
            via ``kind='quantization:gptq-w4'`` permitted.
        concurrency: ``"sequential"`` (default — training workloads aren't
            typically reentrant) | ``"batched"`` | ``"concurrent"``. See
            tensorhub #232 for the scheduler dispatch model.
        label: Optional author-supplied UI / search label. Non-functional.
        description: Optional free-text description. Non-functional.
        calibration: Per-scheme calibration policy (e2e #41/#42), as a
            ``{scheme_name: "required"|"beneficial"|"unsupported"}`` dict.
            Tenants that quantize use this to declare, for each scheme
            their function supports, whether a calibration dataset is
            required (e.g. int4_awq), beneficial (e.g. modelopt fp8), or
            unsupported (torchao int4_wo). Discovery bakes this into
            endpoint.lock so the orchestrator / UI can tell callers up-
            front "supply a dataset." At runtime the tenant calls
            ``gen_worker.conversion.calibration.resolve_calibration_action``
            to enforce policy per spec.

    At decorator time: validates the signature and attaches a
    ``TrainingFunctionSpec`` to the function as ``__training_spec__``.
    Raises TypeError for reserved-name/wrong-type violations or missing
    required params.

    The returned callable is a dispatch wrapper the library invokes with
    ``(request_context, payload)`` at runtime. Reserved-name parameters
    (ctx / source / datasets) are library-injected; the rest of the
    signature decodes from the wire payload. See e2e issue #5 for the
    reserved-name contract.
    """
    # Support both @training_function and @training_function(kind=...).
    if fn is None:
        def _apply(real_fn: Callable[..., list[ProducedFlavor]]) -> Callable[..., list[ProducedFlavor]]:
            return _build_dispatch(
                real_fn, kind=kind, concurrency=concurrency,
                label=label, description=description,
                calibration=calibration,
            )
        return _apply  # type: ignore[return-value]
    return _build_dispatch(
        fn, kind=kind, concurrency=concurrency,
        label=label, description=description,
        calibration=calibration,
    )


def _validate_kind(fn_name: str, kind: str) -> None:
    """Warn on typos against RECOMMENDED_KINDS. Free-form labels are allowed."""
    import warnings
    if not isinstance(kind, str) or not kind.strip():
        raise TypeError(f"{fn_name}: @training_function kind= must be a non-empty string")
    # Split off sub-label if present (e.g. 'quantization:gptq-w4').
    coarse = kind.split(":", 1)[0]
    if coarse not in RECOMMENDED_KINDS:
        warnings.warn(
            f"{fn_name}: @training_function kind={kind!r} is not in the "
            f"recommended set {sorted(RECOMMENDED_KINDS)}. "
            f"Free-form is allowed, but check for typos (did you mean 'fine-tuning'?).",
            stacklevel=3,
        )


def _build_dispatch(
    fn: Callable[..., list[ProducedFlavor]],
    *,
    kind: str,
    concurrency: str = "sequential",
    label: str | None = None,
    description: str | None = None,
    calibration: dict[str, CalibrationPolicy] | None = None,
) -> Callable[..., list[ProducedFlavor]]:
    """Inner: build the dispatch wrapper. Split from ``training_function`` so
    the same body serves both ``@training_function`` and
    ``@training_function(kind=...)`` entry paths."""
    _validate_kind(fn.__name__, kind)
    validated_calibration: dict[str, CalibrationPolicy] = (
        validate_policy_map(fn.__name__, calibration) if calibration else {}
    )
    # Concurrency enum sanity check — matches CONCURRENCY_MODES in the
    # inference_function decorator. Tensorhub #232.
    if concurrency not in ("sequential", "batched", "concurrent"):
        raise ValueError(
            f"@training_function: concurrency={concurrency!r} must be one of "
            "('sequential', 'batched', 'concurrent')"
        )
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

    # ``ctx`` is always required. ``source`` is required for tenant
    # functions that operate on a model checkpoint (cast_dtype, quantization,
    # fine-tuning, ...). Dataset-generation tenants (e2e #45) don't need
    # a source — prompt corpora are model-agnostic — so source is optional
    # when kind starts with "dataset-generation".
    if "ctx" not in sig.parameters:
        raise TypeError(f"{fn.__name__}: must declare required parameter 'ctx'")
    if "source" not in sig.parameters and not kind.startswith("dataset-generation"):
        raise TypeError(
            f"{fn.__name__}: must declare required parameter 'source' "
            f"(only kind='dataset-generation' tenants may omit it)"
        )

    input_schema = _build_wire_payload_schema(
        signature=sig,
        other_params=other_params,
        ref_registry=ref_registry,
    )

    spec = TrainingFunctionSpec(
        fn=fn,
        signature=sig,
        ref_registry=ref_registry,
        other_params=other_params,
        input_schema=input_schema,
        kind=kind,
        calibration=validated_calibration,
    )

    def dispatch(request_context: "RequestContext", payload: Any) -> list[ProducedFlavor]:
        return _run(spec, request_context, payload)

    # Attach metadata so discovery / publish-time validation can read it.
    # Setting _is_training_function + _worker_resources makes the existing
    # gen-worker discovery pick this up as a registered handler. Discovery
    # treats _is_training_function specially — it doesn't try to parse the
    # dispatch wrapper's (request_context, payload) signature as a regular
    # worker function; instead it reads TrainingFunctionSpec directly.
    dispatch.__training_spec__ = spec  # type: ignore[attr-defined]
    dispatch.__wrapped__ = fn  # type: ignore[attr-defined]
    dispatch.__name__ = fn.__name__
    dispatch.__doc__ = fn.__doc__
    dispatch._is_training_function = True  # type: ignore[attr-defined]
    dispatch._worker_resources = ResourceRequirements(kind="training")  # type: ignore[attr-defined]
    # Tensorhub #232: scheduler-facing capability metadata. Discovery emits
    # these into endpoint.lock; orchestrator reads them onto FunctionMetadata
    # for per-function routing decisions.
    dispatch._concurrency_mode = concurrency  # type: ignore[attr-defined]
    dispatch._function_label = (label or "").strip() or None  # type: ignore[attr-defined]
    dispatch._function_description = (description or "").strip() or None  # type: ignore[attr-defined]
    # Per-scheme calibration policy (e2e #41/#42). Discovery / endpoint.lock
    # propagates this up so orchestrator + UI can flag calibrated schemes
    # without a dataset BEFORE dispatch (faster than waiting for the worker
    # to reject). Empty dict for functions that don't quantize.
    dispatch._calibration_policy = dict(validated_calibration)  # type: ignore[attr-defined]
    return dispatch


# ---------------------------------------------------------------------------
# dispatch-time runtime helpers
# ---------------------------------------------------------------------------


def _run(
    spec: TrainingFunctionSpec,
    request_context: "RequestContext",
    payload: Any,
) -> list[ProducedFlavor]:
    """Build kwargs from payload + inject library helpers; call the tenant."""
    # Only build the Source if the tenant declared it. Dataset-generation
    # tenants (e2e #45) skip source to keep corpus generation model-agnostic.
    source: Source | None = None
    if "source" in spec.signature.parameters:
        source = _build_source(request_context, payload)
    ctx = ConversionContext(request_context=request_context, source=source)
    datasets = _build_datasets(request_context, payload)

    # Auto-enforce calibration policy. When a scheme's policy is
    # ``unsupported``, reject any submitted datasets here so tenants don't
    # need to repeat ``resolve_calibration_action`` calls in their bodies.
    if datasets and spec.calibration:
        raw_fields = _payload_as_dict(payload)
        specs_field = raw_fields.get("specs") or []
        if not isinstance(specs_field, list):
            specs_field = []
        for s in specs_field:
            scheme = ""
            if isinstance(s, dict):
                scheme = str(s.get("scheme") or "")
            if not scheme:
                continue
            policy = lookup_policy(spec.calibration, scheme)
            if policy == "unsupported":
                raise ValueError(
                    f"{spec.fn.__name__}: scheme={scheme!r} does not accept a "
                    f"calibration dataset (calibration='unsupported'); drop "
                    f"--dataset and retry. For calibrated quantization use "
                    f"modelopt_quantization."
                )

    injected: dict[str, Any] = {
        "ctx": ctx,
    }
    # Only inject ``datasets`` when the tenant declares the parameter.
    # Tenants that don't accept calibration data (e.g. weight-only quant)
    # can omit the parameter entirely.
    if "datasets" in spec.signature.parameters:
        injected["datasets"] = datasets
    if source is not None:
        injected["source"] = source

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
            f"{spec.fn.__name__}: must return list[ProducedFlavor]; got "
            f"{type(result).__name__}"
        )
    # Upload each ProducedFlavor to the destination + apply tags on success.
    # Library appends exactly one provenance key (`produced_by_job_id`) — see
    # e2e progress.json #5 for the rationale (every other input-to-the-job
    # lives in the orchestrator job record; duplicating onto the variant
    # drifts).
    _finalize_produced_variants(request_context, result, kind=spec.kind)
    return result


def _finalize_produced_variants(
    request_context: "RequestContext",
    variants: list[ProducedFlavor],
    *,
    kind: str,
) -> None:
    """Upload each ProducedFlavor's files, build its manifest + snapshot_digest,
    and publish to the correct tensorhub subsystem based on ``kind``.

    Routes by kind (e2e #45):

    - ``kind.startswith("dataset-generation")`` → publish into
      ``tensorhub.datasets`` via ``publish_dataset_revision``. Used by
      ``generate_prompt_corpus`` / ``generate_eval_set``. File bytes land in
      CAS same as checkpoints; the difference is the publish target —
      dataset artifacts shouldn't pollute the model-checkpoint search space.
    - everything else → publish into ``tensorhub.repo_checkpoints`` via
      ``publish_repo_revision`` (the original path — model weights).

    Upload flow in both branches:
      1. Walks the variant's output dir (or single file) and uploads each file
         via ``ctx.save_checkpoint``, capturing the blake3-keyed Tensors handle
         the upload returns.
      2. Builds a snapshot_manifest entry list from those handles:
         ``[{path, digest, size_bytes}]``.
      3. Computes ``snapshot_digest = sha256(canonical_json(entries))`` —
         content-addresses the whole variant snapshot.
      4. Calls the appropriate publish method with the snapshot_manifest
         + attributes payload.

    ``destination.tags`` (e.g. ``:prod``) are forwarded to
    ``publish_repo_revision`` in the checkpoint path so the tag + checkpoint
    land atomically. Datasets use naming-based versioning (e2e #45 decision),
    not tags — the publish_dataset_revision path ignores destination.tags.
    """
    if not variants:
        return

    is_dataset_gen = (kind or "").startswith("dataset-generation")
    if is_dataset_gen:
        _finalize_dataset_variants(request_context, variants, kind=kind)
        return

    job_id = str(getattr(request_context, "job_id", "") or "") \
        or str(getattr(request_context, "request_id", "") or "")
    library_provenance: dict[str, str] = {"produced_by_job_id": job_id}
    if kind:
        library_provenance["produced_by_kind"] = kind

    # ctx.destination / ctx.source are dicts ({ref, tags, ...} / {ref, attributes, ...}).
    destination = getattr(request_context, "destination", None) or {}
    if not isinstance(destination, dict):
        destination = {}
    dest_ref = str(destination.get("ref") or "").strip()

    source_info = getattr(request_context, "source", None) or {}
    if not isinstance(source_info, dict):
        source_info = {}
    source_ref = str(source_info.get("ref") or "").strip()
    source_checkpoint_id = str(
        source_info.get("checkpoint_id")
        or ""
    ).strip()
    # Fallback: extract the snapshot_digest from the materialized source path.
    # _materialize_source_for_training caches under
    # ``<cache>/cas/snapshots/<algo>:<hex>``; the last path segment IS the
    # checkpoint_id. Covers attribute-selector resolves where the caller
    # didn't know the concrete checkpoint_id up front.
    if not source_checkpoint_id:
        src_path = getattr(request_context, "source_path", None)
        if src_path:
            tail = Path(str(src_path)).name
            if tail and (":" in tail or len(tail) == 64):
                source_checkpoint_id = tail

    relationship_map = {
        "quantization":      "quantization",
        "format-conversion": "format-conversion",
        "fine-tune":         "fine-tune",
        "fine-tuning":       "fine-tune",
        "distillation":      "distillation",
        "pruning":           "pruning",
    }
    rel_kind = relationship_map.get(str(kind or "").lower(), "manual-fork")

    publish_fn = getattr(request_context, "publish_repo_revision", None)
    destination_tags = list(destination.get("tags") or [])
    release_visibility = "private"
    dest_vis = str(destination.get("release_visibility") or destination.get("visibility") or "").strip().lower()
    if dest_vis in ("private", "public"):
        release_visibility = dest_vis

    publish_checkpoint_flavors: list[dict[str, Any]] = []
    aggregate_manifest_entries: list[dict[str, Any]] = []

    for variant in variants:
        merged_attrs = {**library_provenance, **dict(variant.attributes or {})}
        flavor = str(
            getattr(variant, "flavor", "")
            or merged_attrs.get("flavor")
            or merged_attrs.get("flavor")
            or ""
        ).strip()
        flavors: list[str] = []
        raw_flavors = getattr(variant, "flavors", []) or []
        if isinstance(raw_flavors, list):
            for raw_flavor in raw_flavors:
                item = str(raw_flavor or "").strip()
                if item and item not in flavors:
                    flavors.append(item)
        if flavor and flavor not in flavors:
            flavors.insert(0, flavor)
        if flavor:
            merged_attrs["flavor"] = flavor
        if flavors:
            merged_attrs["flavors"] = ",".join(flavors)
        path = variant.path
        _log.info("finalize flavor: path=%s flavor=%s is_file=%s is_dir=%s dest=%s", path, flavor, path.is_file() if path else False, path.is_dir() if path else False, dest_ref)

        # Upload every file under the variant and collect the returned Tensors.
        # Each Tensors handle carries the blake3 digest + size_bytes that
        # tensorhub's CAS upload flow committed — that's what we put in the
        # snapshot_manifest.
        uploaded: list[tuple[str, Any]] = []
        if path.is_file():
            t = _upload_single_file_flavor(
                request_context, path, attributes=merged_attrs,
            )
            if t is not None:
                uploaded.append((path.name, t))
        elif path.is_dir():
            uploaded.extend(_upload_directory_flavor(
                request_context, path, attributes=merged_attrs,
            ))
        else:
            raise FileNotFoundError(f"ProducedFlavor.path does not exist: {path}")
        _log.info("finalize uploaded %d files for variant at %s", len(uploaded), path)

        # Build manifest entries from the uploaded Tensors. Each entry is
        # {path, digest, size_bytes}. Entries without a digest (e.g. save_fn
        # stubbed out in tests) are dropped — they can't participate in the
        # content-addressed snapshot.
        manifest_entries: list[dict[str, Any]] = []
        for rel_path, t in uploaded:
            digest = _tensors_blake3_digest(t)
            if not digest:
                continue
            entry = {
                "path":       rel_path,
                "digest":     digest,
                "size_bytes": int(getattr(t, "size_bytes", 0) or 0),
            }
            manifest_entries.append(entry)

        # Hand the manifest to tensorhub — the server computes checkpoint_id
        # as sha256(canonical(entries)) and persists the checkpoint row. The
        # worker intentionally doesn't know the hashing rule; that lives in
        # one place server-side.
        _log.info("finalize: manifest_entries=%d publish_callable=%s dest_ref=%s", len(manifest_entries), callable(publish_fn), dest_ref)
        if callable(publish_fn) and dest_ref and manifest_entries:
            v_attrs = dict(variant.attributes or {})
            # Issue #22: server-authoritative metadata. The body
            # carries the manifest + flavor labels + size only; server
            # infers kind / library / dtype / file_layout / file_type
            # from the uploaded file contents and writes the canonical
            # values. file_type / file_layout / quantization / kind are
            # deliberately omitted — any caller-side value would be
            # ignored by the server anyway.
            # Derive a meaningful display label from the recipe attrs the
            # tenant supplied. Falls back to the output-dir basename only
            # when no recipe info is available — that fallback used to leak
            # `tmpXXXXXX` Python tempdir names into the catalog.
            label = str(flavor or "").strip()
            if not label:
                recipe = str(v_attrs.get("quant_recipe") or "").strip()
                if recipe:
                    label = recipe.replace(":", "-").replace("/", "-")
            if not label:
                label = variant.path.name
            publish_checkpoint_flavor = {
                # checkpoint_id left empty — server fills it in from
                # sha256(canonical(manifest.entries)).
                "snapshot_digest":   "",
                "flavor":            flavor or label,
                "flavors":           flavors or ([flavor or label] if flavor or label else []),
                "size_bytes":        sum(e.get("size_bytes", 0) for e in manifest_entries),
                # Per-variant manifest — publish_repo_revision forwards this
                # to tensorhub so each checkpoint_id is hashed from only
                # this flavor's files, not an aggregate across flavors.
                "snapshot_manifest": manifest_entries,
                # Issue #63: per-flavor tenant attributes (recipe ids, calib
                # set refs, library version stamps). Merged with library_provenance
                # on the publish_repo_revision body so they land on the catalog
                # row alongside flavor/flavors/display_label.
                "attributes":        dict(merged_attrs),
            }
            publish_checkpoint_flavors.append(publish_checkpoint_flavor)
            aggregate_manifest_entries.extend(manifest_entries)

    if callable(publish_fn) and dest_ref and publish_checkpoint_flavors:
        metadata = {
            "checkpoint_flavors": publish_checkpoint_flavors,
        }
        # Each checkpoint flavor carries its own per-flavor manifest above;
        # the aggregate manifest is kept only as request-level context.
        snapshot_manifest = {
            "version": 1,
            "entries": aggregate_manifest_entries,
        }
        try:
            publish_fn(
                destination_repo=dest_ref,
                metadata=metadata,
                source_repo=source_ref,
                source_version_id=source_checkpoint_id,
                snapshot_manifest=snapshot_manifest,
                relationship_kind=rel_kind,
                release_visibility=release_visibility,
                auto_create_external_parent=False,
                destination_repo_tags=destination_tags,
                merge_with_existing=True,
            )
        except Exception as exc:  # noqa: BLE001 — surface via event, don't fail the job
            emit = getattr(request_context, "emit", None)
            if emit is not None:
                emit("transform.publish_failed", {
                    "ref": dest_ref,
                    "error": f"{type(exc).__name__}: {exc}",
                })


def _finalize_dataset_variants(
    request_context: "RequestContext",
    variants: list[ProducedFlavor],
    *,
    kind: str,
) -> None:
    """Publish dataset-generation tenant outputs into ``tensorhub.datasets``.

    E2E #45. Called from ``_finalize_produced_variants`` when the tenant's
    kind starts with ``dataset-generation``. The flow:

      1. Upload every file under the variant via ``ctx.save_checkpoint``
         (reuses the same blob-CAS path as checkpoints; the bytes are
         identical, only the catalog metadata differs).
      2. Read ``dataset_info.json`` at the variant root — it's the
         authoritative source of truth for features_json / kind / num_rows.
      3. Call ``publish_dataset_revision`` to create (or update) the
         ``tensorhub.datasets`` row. Per e2e #45's design decision, the
         dataset table is mutable; versioning is by naming convention
         (``partiprompts-256-v1`` vs ``partiprompts-256-v2`` are separate
         dataset rows).

    Failures in publish are surfaced as ``dataset.publish_failed`` events
    rather than re-raised — the blob uploads already landed, so the
    content is recoverable even if the catalog update fails.
    """
    import json as _json

    destination = getattr(request_context, "destination", None) or {}
    if not isinstance(destination, dict):
        destination = {}
    dest_ref = str(destination.get("ref") or "").strip()
    if not dest_ref:
        _log.warning("dataset publish: no destination.ref; uploads will succeed but no dataset row created")
        return

    release_visibility = str(destination.get("release_visibility") or destination.get("visibility") or "").strip().lower()
    if release_visibility not in ("private", "public"):
        release_visibility = "private"

    job_id = str(getattr(request_context, "job_id", "") or "") \
        or str(getattr(request_context, "request_id", "") or "")
    library_provenance: dict[str, str] = {"produced_by_job_id": job_id, "produced_by_kind": kind}

    publish_fn = getattr(request_context, "publish_dataset_revision", None)

    for variant in variants:
        merged_attrs = {**library_provenance, **dict(variant.attributes or {})}
        path = variant.path
        _log.info("dataset finalize: path=%s dest=%s", path, dest_ref)

        # Upload files — same as checkpoint path, just lands in CAS.
        uploaded: list[tuple[str, Any]] = []
        if path.is_file():
            t = _upload_single_file_flavor(request_context, path, attributes=merged_attrs)
            if t is not None:
                uploaded.append((path.name, t))
        elif path.is_dir():
            uploaded.extend(_upload_directory_flavor(request_context, path, attributes=merged_attrs))
        else:
            raise FileNotFoundError(f"ProducedFlavor.path does not exist: {path}")

        # Build the snapshot_manifest from uploaded blob digests — records
        # content identity for the revision even though the dataset row
        # is mutable.
        manifest_entries: list[dict[str, Any]] = []
        for rel_path, t in uploaded:
            digest = _tensors_blake3_digest(t)
            if not digest:
                continue
            manifest_entries.append({
                "path": rel_path,
                "digest": digest,
                "size_bytes": int(getattr(t, "size_bytes", 0) or 0),
            })

        # Parse dataset_info.json for features_json + kind. Falls back to
        # a minimal schema if the file is missing (shouldn't happen — the
        # tenants write it — but don't let a malformed snapshot nuke
        # publish).
        info_path = path / "dataset_info.json" if path.is_dir() else None
        features_json: dict[str, Any] = {}
        dataset_kind: str = ""
        dataset_info: dict[str, Any] = {}
        if info_path is not None and info_path.exists():
            try:
                with open(info_path) as f:
                    dataset_info = _json.load(f)
                if isinstance(dataset_info, dict):
                    features_json = dict(dataset_info.get("features") or {})
                    dataset_kind = str(dataset_info.get("kind") or "")
            except Exception as exc:  # noqa: BLE001
                _log.warning("dataset finalize: failed to parse dataset_info.json (%s)", exc)

        if not callable(publish_fn):
            _log.warning(
                "dataset finalize: request_context.publish_dataset_revision missing; "
                "blobs uploaded to CAS but no tensorhub.datasets row created. "
                "gen-worker release needs bump."
            )
            continue

        try:
            result = publish_fn(
                destination_dataset=dest_ref,
                features_json=features_json,
                row_artifacts_json=None,
                snapshot_manifest=manifest_entries,
                visibility=release_visibility,
                kind=dataset_kind,
                dataset_info=dataset_info,
            )
            _log.info("dataset finalize: published %s → %s", dest_ref, result)
        except Exception as exc:  # noqa: BLE001
            emit = getattr(request_context, "emit", None)
            if emit is not None:
                emit("dataset.publish_failed", {
                    "ref": dest_ref,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            _log.warning("dataset finalize: publish failed for %s: %s", dest_ref, exc)


def _tensors_blake3_digest(t: Any) -> str:
    """Return the ``blake3:<hex>`` digest the upload flow bound for this
    Tensors handle. Tensorhub's CAS indexes blobs by blake3; the commit step
    verifies the uploaded bytes hashed to this digest, so we trust it here.
    Returns ``""`` when the upload path didn't populate a digest (e.g. test
    stubs), in which case the manifest entry is dropped.
    """
    bd = str(getattr(t, "blob_digest", "") or "").strip()
    if bd:
        return bd if ":" in bd else f"blake3:{bd}"
    b3 = str(getattr(t, "blake3", "") or "").strip()
    if b3:
        return b3 if ":" in b3 else f"blake3:{b3}"
    return ""


def _upload_single_file_flavor(
    request_context: "RequestContext",
    path: Path,
    *,
    attributes: dict[str, str],
) -> Any:
    """Upload one safetensors/gguf/flashpack file; return the Tensors handle.

    Returns ``None`` when ``save_checkpoint`` is missing (test stubs).
    """
    save_fn = getattr(request_context, "save_checkpoint", None)
    if save_fn is None:
        return None
    ref = _default_flavor_ref(request_context, path)
    format_hint = path.suffix.lstrip(".") or "bin"
    return save_fn(ref, str(path), format=format_hint, attributes=attributes)


def _upload_directory_flavor(
    request_context: "RequestContext",
    dir_path: Path,
    *,
    attributes: dict[str, str],
) -> list[tuple[str, Any]]:
    """Upload every file under a directory flavor; return ``[(rel_path, Tensors)]``."""
    save_fn = getattr(request_context, "save_checkpoint", None)
    if save_fn is None:
        return []
    base_ref = _default_flavor_ref(request_context, dir_path)
    uploaded: list[tuple[str, Any]] = []
    for f in sorted(dir_path.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(dir_path).as_posix()
        ref = f"{base_ref}/{rel}"
        format_hint = f.suffix.lstrip(".") or "bin"
        t = save_fn(ref, str(f), format=format_hint, attributes=attributes)
        uploaded.append((rel, t))
    return uploaded


def _default_flavor_ref(request_context: "RequestContext", path: Path) -> str:
    """Return a job-scoped ref for a produced flavor."""
    job_id = str(getattr(request_context, "job_id", "") or "") \
        or str(getattr(request_context, "request_id", "r")) \
        or "r"
    return f"jobs/{job_id}/outputs/{path.name}"


def _build_source(request_context: "RequestContext", payload: Any) -> Source:
    """Construct a Source from the materialized snapshot + resolved attributes."""
    source_path = getattr(request_context, "source_path", None)
    if source_path is None:
        raise RuntimeError(
            "@training_function requires request_context.source_path to be set "
            "(endpoint_kind must be 'training' and the gen-worker reserved-name "
            "dispatch must have materialized the source)"
        )
    ctx_source = getattr(request_context, "source", None)
    attrs: dict = {}
    ref = ""
    if ctx_source is not None:
        attrs = dict(getattr(ctx_source, "attributes", {}) or {})
        ref = str(getattr(ctx_source, "ref", "") or "")
    return Source(Path(source_path), attributes=attrs, ref=ref)


def _build_datasets(request_context: "RequestContext", payload: Any) -> list[Dataset]:
    """Materialize each DatasetRef in payload.datasets into a Dataset.

    Resolution order (e2e #45):

    1. ``request_context.dataset_paths[ref]`` — when the orchestrator
       pre-materialized the dataset, the local path is already in-hand.
       This is the fast path used in tests + when the orchestrator's
       dataset-resolver has wired the download before dispatch.
    2. ``request_context.resolve_dataset(ref)`` — when the
       gen-worker / orchestrator side has the resolver-helper method
       wired (the e2e #45 target state). Downloads the dataset's
       parquet + dataset_info.json into the worker's local cache and
       returns the path.
    3. Error out — the caller needs to wire one of the above.
    """
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

        local_path: str | None = None
        # Path 1: orchestrator-injected cache.
        dataset_paths = getattr(request_context, "dataset_paths", {}) or {}
        if ref in dataset_paths:
            local_path = dataset_paths[ref]

        # Path 2: gen-worker's own resolver (e2e #45 target path). When the
        # worker runtime has ``resolve_dataset`` wired, call it and cache
        # the returned path on the request_context so subsequent calls
        # don't re-download.
        if local_path is None:
            resolver = getattr(request_context, "resolve_dataset", None)
            if callable(resolver):
                try:
                    local_path = str(resolver(ref))
                    # Cache for future lookups in the same request.
                    dataset_paths = dict(dataset_paths)
                    dataset_paths[ref] = local_path
                    try:
                        request_context.dataset_paths = dataset_paths  # type: ignore[attr-defined]
                    except Exception:
                        pass  # read-only RC; first lookup path is sole cache.
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"dataset {ref!r}: resolver failed ({exc})"
                    ) from exc

        if local_path is None:
            raise RuntimeError(
                f"dataset {ref!r} not materialized: neither "
                f"request_context.dataset_paths[{ref!r}] nor "
                f"request_context.resolve_dataset({ref!r}) is wired. "
                f"Orchestrator must materialize datasets before dispatch, "
                f"OR the worker runtime must expose a resolver helper."
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
    # `source` is required only when the tenant signature declares a `source`
    # parameter. Dataset-generation tenants (e2e #45) deliberately omit it —
    # forcing `source` into required there would reject every valid request
    # at the orchestrator schema gate (e2e #51 hit this).
    has_source_param = "source" in signature.parameters
    required: list[str] = []
    if has_source_param:
        required.append("source")
    required.append("destination")

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

    # Top-level properties. Mirror `has_source_param` from above — emit the
    # `source` slot only when the tenant signature declares it. Dataset-
    # generation tenants (e2e #45) get a schema without `source`, so the
    # orchestrator gate accepts wire payloads that omit the field.
    if has_source_param:
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
    "TrainingFunctionSpec",
    "RESERVED_TYPES",
    "training_function",
]
