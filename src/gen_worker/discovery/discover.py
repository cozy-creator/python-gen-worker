"""
Function discovery module for Cozy workers.

This module auto-discovers all @inference decorated functions in the project
by scanning .py files and extracting metadata. Run as:

    python -m gen_worker.discovery

Outputs TOML endpoint lock to stdout.
"""

import hashlib
import json
import sys
import typing
import types as py_types
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import msgspec

from gen_worker.api.binding import Binding, Dispatch, HFRepo, Repo, Variant
from gen_worker.api.types import (
    Asset,
    AudioAsset,
    ExpectedOutput,
    ImageAsset,
    MediaAsset,
    PromptRole,
    Tensors,
    VideoAsset,
)

from gen_worker.discovery.toml_manifest import (
    EndpointToml,
    load_endpoint_toml,
)
from gen_worker.discovery.names import slugify_name
from gen_worker.discovery.walk import find_endpoint_classes


def _type_id(t: type) -> Dict[str, str]:
    """Get module and qualname for a type."""
    return {
        "module": getattr(t, "__module__", ""),
        "qualname": getattr(t, "__qualname__", getattr(t, "__name__", "")),
    }


def _type_qualname(t: type) -> str:
    """Get fully qualified name for a type."""
    mod = getattr(t, "__module__", "")
    qn = getattr(t, "__qualname__", getattr(t, "__name__", ""))
    if mod and qn:
        return f"{mod}.{qn}"
    return repr(t)


def _is_msgspec_struct(t: Any) -> bool:
    """Check if type is a msgspec.Struct subclass."""
    try:
        return isinstance(t, type) and issubclass(t, msgspec.Struct)
    except Exception:
        return False


_MEDIA_ASSET_TYPES = (MediaAsset, ImageAsset, VideoAsset, AudioAsset)


def _media_kind(t: type) -> str:
    if issubclass(t, ImageAsset):
        return "image"
    if issubclass(t, VideoAsset):
        return "video"
    if issubclass(t, AudioAsset):
        return "audio"
    return "media"


def _collect_payload_moderation_metadata(payload_type: type) -> Dict[str, Any]:
    out: Dict[str, list[Dict[str, str]]] = {"prompts": [], "media": []}
    seen_structs: set[type] = set()

    def walk(ann: Any, path: str) -> None:
        origin = typing.get_origin(ann)

        if origin is typing.Annotated:
            args = typing.get_args(ann)
            if not args:
                return
            base = args[0]
            roles = [m for m in args[1:] if isinstance(m, PromptRole)]
            if roles:
                if base is not str:
                    raise ValueError(
                        f"{path}: PromptRole markers must annotate str fields"
                    )
                out["prompts"].append({"field": path, "role": roles[-1].role})
                return
            walk(base, path)
            return

        if origin in (typing.Union, py_types.UnionType):
            for arg in typing.get_args(ann):
                if arg is not type(None):
                    walk(arg, path)
            return

        if origin in (list, tuple, set, frozenset):
            args = typing.get_args(ann)
            if args:
                walk(args[0], f"{path}[]")
            return

        if origin is dict:
            args = typing.get_args(ann)
            if len(args) == 2:
                walk(args[1], f"{path}.*")
            return

        if isinstance(ann, type):
            if issubclass(ann, _MEDIA_ASSET_TYPES):
                out["media"].append({"field": path, "kind": _media_kind(ann)})
                return
            if issubclass(ann, (Asset, Tensors)):
                return
            if _is_msgspec_struct(ann):
                if ann in seen_structs:
                    return
                seen_structs.add(ann)
                try:
                    hints = typing.get_type_hints(ann, include_extras=True)
                except Exception:
                    hints = getattr(ann, "__annotations__", {})
                for field in getattr(ann, "__struct_fields__", ()) or ():
                    if field in hints:
                        walk(hints[field], f"{path}.{field}" if path else field)
                seen_structs.discard(ann)

    walk(payload_type, "")
    return {k: v for k, v in out.items() if v}


def _unwrap_optional(ann: Any) -> Any:
    origin = typing.get_origin(ann)
    if origin in (typing.Union, py_types.UnionType):
        args = [arg for arg in typing.get_args(ann) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return ann


def _media_kind_for_annotation(ann: Any) -> str:
    ann = _unwrap_optional(ann)
    origin = typing.get_origin(ann)
    if origin in (list, tuple, set, frozenset):
        args = typing.get_args(ann)
        ann = _unwrap_optional(args[0]) if args else Any
    if isinstance(ann, type) and issubclass(ann, ImageAsset):
        return "image"
    if isinstance(ann, type) and issubclass(ann, VideoAsset):
        return "video"
    if isinstance(ann, type) and issubclass(ann, AudioAsset):
        return "audio"
    if isinstance(ann, type) and issubclass(ann, MediaAsset):
        return "file"
    return "other"


def _payload_has_field_path(payload_type: type, ref: str) -> bool:
    if not ref.startswith("input."):
        return True
    path = ref.removeprefix("input.")
    if not path:
        return False

    current: Any = payload_type
    for raw_part in path.replace("[]", "").split("."):
        part = raw_part.strip()
        if not part:
            return False
        current = _unwrap_optional(current)
        origin = typing.get_origin(current)
        if origin in (list, tuple, set, frozenset):
            args = typing.get_args(current)
            current = _unwrap_optional(args[0]) if args else Any
        if not _is_msgspec_struct(current):
            return False
        try:
            hints = typing.get_type_hints(current, include_extras=True)
        except Exception:
            hints = getattr(current, "__annotations__", {}) or {}
        if part not in hints:
            return False
        current = hints[part]
    return True


def _expected_output_expr(value: Any, *, payload_type: type, field: str, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{field}: ExpectedOutput.{key} must be positive")
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.startswith("input.") and not _payload_has_field_path(payload_type, raw):
            raise ValueError(f"{field}: ExpectedOutput.{key} references unknown payload field {raw!r}")
        return raw
    raise TypeError(f"{field}: ExpectedOutput.{key} must be int, str, or None")


def _collect_expected_output_metadata(payload_type: type, output_type: type) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    seen_structs: set[type] = set()

    def walk(ann: Any, path: str) -> None:
        origin = typing.get_origin(ann)

        if origin is typing.Annotated:
            args = typing.get_args(ann)
            if not args:
                return
            base = args[0]
            markers = [m for m in args[1:] if isinstance(m, ExpectedOutput)]
            if markers:
                marker = markers[-1]
                media_type = marker.media_type or _media_kind_for_annotation(base)
                item: Dict[str, Any] = {
                    "field": path,
                    "type": media_type,
                }
                count = _expected_output_expr(marker.count, payload_type=payload_type, field=path, key="count")
                if count is not None:
                    item["count"] = count
                width = _expected_output_expr(marker.width, payload_type=payload_type, field=path, key="width")
                if width is not None:
                    item["width"] = width
                height = _expected_output_expr(marker.height, payload_type=payload_type, field=path, key="height")
                if height is not None:
                    item["height"] = height
                aspect = _expected_output_expr(marker.aspect_ratio, payload_type=payload_type, field=path, key="aspect_ratio")
                if aspect is not None:
                    item["aspect_ratio"] = aspect
                mime = (marker.mime_type or "").strip()
                if mime:
                    item["mime_type"] = mime
                out.append(item)
                return
            walk(base, path)
            return

        if origin in (typing.Union, py_types.UnionType):
            for arg in typing.get_args(ann):
                if arg is not type(None):
                    walk(arg, path)
            return

        if origin in (list, tuple, set, frozenset):
            args = typing.get_args(ann)
            if args:
                walk(args[0], f"{path}[]")
            return

        if isinstance(ann, type) and _is_msgspec_struct(ann):
            if ann in seen_structs:
                return
            seen_structs.add(ann)
            try:
                hints = typing.get_type_hints(ann, include_extras=True)
            except Exception:
                hints = getattr(ann, "__annotations__", {})
            for field in getattr(ann, "__struct_fields__", ()) or ():
                if field in hints:
                    walk(hints[field], f"{path}.{field}" if path else field)
            seen_structs.discard(ann)

    walk(output_type, "")
    return out


def _binding_slot_name(repo: Repo, fallback: str) -> str:
    return str(getattr(repo, "slot_name", "") or fallback).strip()


def _binding_to_manifest(binding: Binding, param_annotation: Any, param_name: str = "") -> Dict[str, Any]:
    """Emit a `functions.bindings.<param>` block for the manifest.

    Wire shape (from `progress.json` issue #9):

    Fixed::

        [functions.bindings.pipeline]
        kind = "fixed"
        ref = "owner/repo"
        flavor = "bf16"
        tag = "prod"
        allow_override = false
        pipeline_classes = ["pkg.mod.PipelineClass"]

    Dispatch::

        [functions.bindings.pipeline]
        kind = "dispatch"
        field = "variant"
        allow_override = false
        pipeline_classes = ["pkg.mod.PipelineClass"]

        [functions.bindings.pipeline.table.nf4]
        ref = "owner/repo"
        flavor = "nf4"
        tag = "prod"
    """
    # `pipeline_classes` comes from `.allow_override(*classes)` (the explicit
    # tenant-declared allowlist). When the binding has no override declared,
    # we still emit the param-annotated class FQN as metadata so the
    # orchestrator / UI can surface it without parsing the Python signature.
    declared_classes: List[str] = list(getattr(binding, "_pipeline_classes", ()) or ())
    if not declared_classes:
        annotation_class = _annotation_to_fqn(param_annotation)
        if annotation_class:
            declared_classes = [annotation_class]

    if isinstance(binding, Repo):
        out: Dict[str, Any] = {
            "kind": "fixed",
            # `provider` distinguishes HFRepo / CivitaiRepo / Repo (tensorhub).
            # Without it the downstream catalog defaults to "tensorhub" and
            # silently rewrites the ref through tensorhub's slug normalizer,
            # so HFRepo("black-forest-labs/FLUX.2-klein-4B") landed as
            # tensorhub-provider `black-forest-labs/flux.2-klein-4b-base` in
            # the catalog. Always emit the actual provider.
            "provider": binding.provider,
            "slot_name": _binding_slot_name(binding, param_name),
            "ref": binding.ref,
            "flavor": binding._flavor,
            "tag": binding._tag,
            "allow_override": bool(binding._allow_override),
            "allow_lora": bool(getattr(binding, "_allow_lora", False)),
            "pipeline_classes": declared_classes,
        }
        # Issue #20 fix 2: HF bindings carry a `dtype` field (replaces the
        # old #flavor-suffix that leaked into model_id and got dropped by
        # the loader). Only emitted for HFRepo and only when non-empty.
        if isinstance(binding, HFRepo) and getattr(binding, "_dtype", ""):
            out["dtype"] = binding._dtype
        return out
    if isinstance(binding, Dispatch):
        table: Dict[str, Dict[str, str]] = {}
        dispatch_slot_name = ""
        for k, repo in binding.table.items():
            if not dispatch_slot_name:
                dispatch_slot_name = _binding_slot_name(repo, param_name)
            entry: Dict[str, str] = {
                "ref": repo.ref,
                "tag": repo._tag,
                # Per-entry provider — a Dispatch table can mix providers
                # across variants (e.g. fp8 from tensorhub, nf4 from HF).
                "provider": repo.provider,
                "slot_name": _binding_slot_name(repo, param_name),
            }
            if repo._flavor:
                entry["flavor"] = repo._flavor
            # Issue #20 fix 2: HF entries can carry dtype.
            if isinstance(repo, HFRepo) and getattr(repo, "_dtype", ""):
                entry["dtype"] = repo._dtype
            # #337: a SharedBase variant records its shared-component refs + its
            # per-variant swap-slot refs into the endpoint lock so the full
            # residency footprint of each selectable model is visible (shared
            # base loads once + pinned; only the variant slot swaps per model).
            if isinstance(repo, Variant):
                entry["variant_kind"] = "shared_base_variant"
                entry["pipeline_class"] = repo.pipeline_class_fqn
                entry["shared_components"] = {
                    name: {
                        "ref": comp.ref,
                        "provider": comp.provider,
                        "subfolder": str(getattr(comp, "_subfolder", "") or ""),
                        "dtype": str(getattr(comp, "_dtype", "") or ""),
                    }
                    for name, comp in dict(repo.shared_components).items()
                }
                entry["variant_slots"] = {
                    name: {
                        "ref": comp.ref,
                        "provider": comp.provider,
                        "subfolder": str(getattr(comp, "_subfolder", "") or ""),
                        "dtype": str(getattr(comp, "_dtype", "") or ""),
                    }
                    for name, comp in dict(repo.variant_slots).items()
                }
            table[k] = entry
        return {
            "kind": "dispatch",
            "field": binding.field,
            "slot_name": dispatch_slot_name or param_name,
            "table": table,
            "allow_override": bool(binding._allow_override),
            "allow_lora": bool(getattr(binding, "_allow_lora", False)),
            "pipeline_classes": declared_classes,
        }
    raise TypeError(f"unknown binding type: {type(binding).__name__}")


def _annotation_to_fqn(ann: Any) -> str:
    """Return a plain `module.qualname` for a parameter annotation, or ''."""
    if isinstance(ann, type):
        return f"{ann.__module__}.{ann.__qualname__}"
    return ""


def _schema_and_hash(t: type) -> Tuple[Dict[str, Any], str]:
    """Generate JSON schema and SHA256 hash for a msgspec type."""
    schema = msgspec.json.schema(t)
    try:
        from gen_worker.api.payload_constraints import apply_schema_constraints

        schema = apply_schema_constraints(schema, t)
    except Exception:
        pass
    raw = json.dumps(schema, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return schema, hashlib.sha256(raw).hexdigest()


def _find_endpoint_toml_path(root: Path) -> Path | None:
    p = root / "endpoint.toml"
    return p if p.exists() else None


def _load_endpoint_manifest_toml(root: Path) -> EndpointToml:
    p = _find_endpoint_toml_path(root)
    if p is None:
        raise ValueError("missing endpoint.toml (required for discovery)")
    return load_endpoint_toml(p)


def _assert_unique_function_names(functions: List[Dict[str, Any]]) -> None:
    """Fail the build if two functions share a routable name in one endpoint.

    Function names are the endpoint's external routing identifiers
    (``owner/endpoint/<name>``, the wire ``function_name``, the
    ``invoke <name>`` / ``serve --function <name>`` key), so they MUST be
    unique within an endpoint. A collision is an author error — e.g. two
    classes each exposing a generic ``name="generate"`` without an explicit
    override. The worker historically only logged ``Handler name conflict;
    skipping`` and silently dropped one route; fail loudly at
    discovery/endpoint.lock build time instead.
    """
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for fn in functions:
        nm = str(fn.get("name") or "").strip()
        if nm:
            by_name.setdefault(nm, []).append(fn)
    dupes = {nm: fns for nm, fns in by_name.items() if len(fns) > 1}
    if not dupes:
        return
    lines = []
    for nm, fns in sorted(dupes.items()):
        where = ", ".join(
            f"{f.get('class_name') or '<module-level>'} in "
            f"{f.get('module') or f.get('declared_module') or '?'}"
            for f in fns
        )
        lines.append(f"  {nm!r}: defined {len(fns)}x ({where})")
    raise ValueError(
        "duplicate function name(s) within the endpoint — function names are the "
        "external routing identifiers and must be unique. Give each method an "
        "explicit @invocable(name=...):\n" + "\n".join(lines)
    )


def discover_functions(root: Optional[Path] = None, *, main_module: str | None = None) -> List[Dict[str, Any]]:
    """
    Discover all @inference decorated functions in the project.

    Args:
        root: Project root directory. Defaults to current working directory.
        main_module: ``endpoint.toml``'s ``main`` pointer (e.g. ``"conversion.main"``).
            Required. The top-level package (``"conversion"``) is walked via
            :func:`gen_worker.discovery.walk.find_endpoint_classes` to find
            class-shape endpoints.

    Returns:
        List of function metadata dictionaries.
    """
    if root is None:
        root = Path.cwd()
    root = root.resolve()

    # Ensure root is in sys.path for imports
    root_str = str(root)
    src_str = str(root / "src")
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if (root / "src").exists() and src_str not in sys.path:
        sys.path.insert(0, src_str)

    functions: List[Dict[str, Any]] = []
    imported_modules: Set[str] = set()
    seen_functions: Set[Tuple[str, str, str]] = set()  # (declared_module, class_name, python_name)

    def _module_is_in_project(mod: Any) -> bool:
        """
        Limit discovery to modules that live under the project root.

        This avoids inspecting third-party modules (e.g. transformers LazyModule),
        which can trigger expensive/optional imports and break discovery.
        """
        mod_file = getattr(mod, "__file__", None)
        if not mod_file:
            return False
        try:
            p = Path(mod_file).resolve()
        except Exception:
            return False
        try:
            p.relative_to(root)
            return True
        except Exception:
            return False

    if main_module:
        # Build-time class-shape discovery now flows through the unified
        # walker shared with Worker._discover_and_register_functions. The
        # walker takes top-level package names; we derive the top-level
        # from main_module (``"conversion.main"`` -> ``"conversion"``) so
        # the walker's ``walk_packages`` finds every submodule whether or
        # not __init__.py re-exports it.
        top_level = main_module.split(".", 1)[0]
        try:
            found_classes = find_endpoint_classes([top_level])
        except Exception as e:
            raise ValueError(
                f"failed to walk endpoint package {top_level!r} (derived from "
                f"endpoint.toml main={main_module!r}): {e}"
            ) from e

        for found in found_classes:
            # Per-class entries are attributed to the walked top-level
            # package name to keep the lockfile ``module`` field stable
            # across submodule re-exports (matches pre-refactor behavior).
            module_name = found.walked_module
            try:
                class_entries = _extract_class_function_methods(
                    found.cls, module_name
                )
            except Exception as e:
                print(
                    f"warning: failed to extract class endpoint {found.qualname}: {e}",
                    file=sys.stderr,
                )
                raise
            for class_fn in class_entries:
                key = (
                    class_fn.get("declared_module", class_fn.get("module", "")),
                    class_fn.get("class_name", ""),
                    class_fn.get("python_name", ""),
                )
                if key in seen_functions:
                    continue
                seen_functions.add(key)
                functions.append(class_fn)

        _assert_unique_function_names(functions)
        return functions

    raise ValueError(
        "discover_functions requires main_module (endpoint.toml `main`); "
        "class-shape endpoints are discovered by walking that package"
    )


def _extract_class_function_methods(
    cls: type, module_name: str
) -> List[Dict[str, Any]]:
    """Manifest entries for a class-shape endpoint.

    Signature inspection (payload/output types, streaming shape, parametrize
    fan-out) is delegated to ``gen_worker.registry`` — the one walker shared
    with the worker runtime and the CLI. This function only adds the
    manifest-specific enrichment (schemas, moderation, bindings blocks).
    """
    from gen_worker.registry import extract_specs

    spec = getattr(cls, "__gen_worker_endpoint_spec__", None)
    if spec is None:
        return []
    # Parity with the historical manifest: only @<kind>.function/@invocable
    # methods are baked; @batched_inference methods stay runtime-only.
    plain_attrs = {
        attr for attr, _m, _s in getattr(cls, "__gen_worker_function_methods__", []) or []
    }
    if not plain_attrs:
        return []

    out: List[Dict[str, Any]] = []
    for es in extract_specs(cls, walked_module=module_name):
        if es.attr_name not in plain_attrs:
            continue
        res_dict: Dict[str, Any] = {}
        try:
            raw = msgspec.to_builtins(es.resources)
            if isinstance(raw, dict):
                res_dict.update(raw)
        except Exception:
            pass
        bindings_block = {
            key: _binding_to_manifest(binding, None, key)
            for key, binding in es.models.items()
        }

        input_schema, input_sha = _schema_and_hash(es.payload_type)
        moderation = _collect_payload_moderation_metadata(es.payload_type)
        output_type = es.output_type
        if output_type is None:
            raise ValueError(
                f"{cls.__name__}.{es.attr_name}: manifest requires a concrete "
                "msgspec.Struct output/delta type"
            )
        output_schema, output_sha = _schema_and_hash(output_type)
        expected_outputs = _collect_expected_output_metadata(es.payload_type, output_type)
        incremental = es.output_mode == "stream"
        delta_schema = None
        delta_sha = ""
        if incremental and es.delta_type is not None:
            delta_schema, delta_sha = _schema_and_hash(es.delta_type)

        function_name = slugify_name(es.name)
        if not function_name:
            raise ValueError(
                f"{cls.__name__}.{es.attr_name}: function name cannot be "
                f"normalized from {es.name!r}"
            )

        fn: Dict[str, Any] = {
            "name": function_name,
            "python_name": es.attr_name,
            "module": module_name,
            "declared_module": getattr(cls, "__module__", "") or module_name,
            "class_name": cls.__name__,
            "archetype": getattr(cls, "__gen_worker_archetype__", "SerialWorker"),
            "kind": es.kind,
            "sub_kind": es.sub_kind,
            "runtime": es.runtime,
            "resources": res_dict,
            "bindings": bindings_block,
            "payload_type": _type_id(es.payload_type),
            "payload_schema_sha256": input_sha,
            "input_schema": input_schema,
            "moderation": moderation,
            "expected_outputs": expected_outputs,
            "output_mode": "incremental" if incremental else "single",
            "output_type": _type_id(output_type),
            "output_schema_sha256": output_sha,
            "output_schema": output_schema,
            "incremental_output": incremental,
            "is_async": es.is_async,
            "decorator": f"@{es.kind}.function",
            "label": es.label,
            "description": es.description,
            "timeout_ms": es.timeout_ms,
            "allowed_shapes": [list(s) for s in es.allowed_shapes],
            "batch_window_ms": getattr(spec, "batch_window_ms", None),
            "max_batch": getattr(spec, "max_batch", None),
        }
        if incremental and es.delta_type is not None:
            fn["delta_type"] = _type_id(es.delta_type)
            fn["delta_schema_sha256"] = delta_sha
            fn["delta_output_schema"] = delta_schema
        out.append(fn)

    return out


def discover_manifest(root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Discover functions and load tensorhub manifest config to build complete manifest.

    Args:
        root: Project root directory. Defaults to current working directory.

    Returns: Complete manifest dict with functions + models/resources metadata.
    """
    if root is None:
        root = Path.cwd()
    root = root.resolve()

    tensorhub_manifest = _load_endpoint_manifest_toml(root)

    functions = discover_functions(root, main_module=tensorhub_manifest.main)

    seen_fn: Dict[str, str] = {}
    for fn in functions:
        fn_name = str(fn.get("name") or "").strip()
        py_name = str(fn.get("python_name") or "").strip()
        if not fn_name:
            raise ValueError("discovered function missing name")
        prior = seen_fn.get(fn_name)
        if prior and prior != py_name:
            raise ValueError(
                f"multiple functions normalize to the same function name '{fn_name}': {prior}, {py_name or '<unknown>'}"
            )
        seen_fn[fn_name] = py_name

    manifest: Dict[str, Any] = {
        "functions": functions,
    }
    return manifest


def _strip_none(obj: Any) -> Any:
    """Recursively remove None values from dicts/lists (TOML has no null type)."""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj if v is not None]
    return obj


def main() -> None:
    """Write the build-time endpoint manifest to stdout.

    #328: bake-time validation gate. After ``discover_manifest`` produces
    the ``functions`` list, ``validate_endpoint_lock`` confirms every entry
    is a class-shape declaration (post-#322). An old function-shape entry
    that slipped past the discovery refactor hard-fails the build with a
    pointer to the migration guide.
    """
    try:
        manifest = discover_manifest()
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    # #328 bake-time validation gate. Old function-shape entries trip the
    # missing-class_name check; same-class slug collisions trip the route
    # uniqueness check; non-class shapes (an entry without archetype/kind)
    # are caught by the required-field check. All errors flow out at once
    # so the build surfaces every problem rather than one-at-a-time.
    from .validation import validate_endpoint_lock

    val = validate_endpoint_lock(manifest)
    for w in val.warnings:
        print(f"warning: {w}", file=sys.stderr)
    if not val.ok:
        for err in val.errors:
            print(f"error: {err}", file=sys.stderr)
        sys.exit(1)

    if not manifest.get("functions"):
        print("warning: no @inference decorated functions found", file=sys.stderr)

    sys.stdout.write(msgspec.toml.encode(_strip_none(manifest)).decode("utf-8"))
    if not sys.stdout.isatty():
        sys.stdout.write("\n")
