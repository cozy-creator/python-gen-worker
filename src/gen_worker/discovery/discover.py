"""Build-time endpoint discovery: walk the ``[tool.gen_worker].main``
package, extract every ``@endpoint`` object, and emit the endpoint.lock
manifest as TOML on stdout. Run as ``python -m gen_worker.discovery``.
"""

import hashlib
import json
import sys
import typing
import types as py_types
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import msgspec

from gen_worker.api.binding import HF, Binding, Civitai, Hub, ModelScope
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
from gen_worker.discovery.names import slugify_name
from gen_worker.discovery.project import load_project_config
from gen_worker.discovery.walk import find_endpoints


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
                duration = _expected_output_expr(marker.duration_s, payload_type=payload_type, field=path, key="duration_s")
                if duration is not None:
                    item["duration_s"] = duration
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


def _binding_to_manifest(binding: Binding, param_name: str = "") -> Dict[str, Any]:
    """Emit a ``functions.bindings.<slot>`` block for the manifest.

    Every binding is a fixed pick; the slot name is the dict key. Keys stay
    compatible with ``models.download.build_provider_index_from_manifest``
    (``ref`` / ``provider`` / ``flavor``).
    """
    out: Dict[str, Any] = {
        "kind": "fixed",
        "provider": binding.provider,
        "slot_name": param_name,
        "ref": binding.ref,
    }
    if isinstance(binding, Hub):
        out["tag"] = binding.tag
        if binding.flavor:
            out["flavor"] = binding.flavor
        if binding.allow_lora:
            out["allow_lora"] = True
    elif isinstance(binding, HF):
        for k in ("revision", "dtype", "subfolder"):
            v = getattr(binding, k)
            if v:
                out[k] = v
        if binding.files:
            out["files"] = list(binding.files)
        if binding.allow_lora:
            out["allow_lora"] = True
    elif isinstance(binding, Civitai):
        if binding.version:
            out["version"] = binding.version
    elif isinstance(binding, ModelScope):
        if binding.revision:
            out["revision"] = binding.revision
        if binding.files:
            out["files"] = list(binding.files)
    return out


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
        "external routing identifiers and must be unique. Rename the handler "
        "method (or a variants= key):\n" + "\n".join(lines)
    )


def discover_functions(root: Optional[Path] = None, *, main_module: str | None = None) -> List[Dict[str, Any]]:
    """Discover every @endpoint object under ``main_module``'s top-level
    package and return the manifest ``functions`` entries."""
    if root is None:
        root = Path.cwd()
    root = root.resolve()
    if not main_module:
        raise ValueError(
            "discover_functions requires main_module ([tool.gen_worker].main)"
        )

    root_str = str(root)
    src_str = str(root / "src")
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if (root / "src").exists() and src_str not in sys.path:
        sys.path.insert(0, src_str)

    top_level = main_module.split(".", 1)[0]
    try:
        found = find_endpoints([top_level])
    except Exception as e:
        raise ValueError(
            f"failed to walk endpoint package {top_level!r} (derived from "
            f"[tool.gen_worker] main={main_module!r}): {e}"
        ) from e

    functions: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str, str]] = set()
    for f in found:
        for entry in _extract_entries(f.obj, f.walked_module):
            # name is part of the key: variants= share (module, class,
            # python_name) with their base function but stamp distinct names.
            key = (
                entry.get("declared_module", entry.get("module", "")),
                entry.get("class_name", ""),
                entry.get("python_name", ""),
                entry.get("name", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            functions.append(entry)

    _assert_unique_function_names(functions)
    return functions


def _extract_entries(obj: Any, module_name: str) -> List[Dict[str, Any]]:
    """Manifest entries for one @endpoint class or function.

    Signature inspection lives in ``gen_worker.registry`` — the one walker
    shared with the worker runtime and the CLI. This adds only the
    manifest-specific enrichment (schemas, moderation, bindings blocks).
    """
    from gen_worker.registry import extract_specs

    out: List[Dict[str, Any]] = []
    for es in extract_specs(obj, walked_module=module_name):
        res_dict: Dict[str, Any] = {}
        try:
            raw = msgspec.to_builtins(es.resources)
            if isinstance(raw, dict):
                res_dict.update(raw)
        except Exception:
            pass
        bindings_block = {
            key: _binding_to_manifest(binding, key)
            for key, binding in es.models.items()
        }
        # allow_lora bindings carry the endpoint's architecture family so the
        # hub's th#586 gate can police adapter targets (builder rejects
        # allow_lora without family).
        compile_family = es.compile.family if es.compile is not None else ""
        for block in bindings_block.values():
            if block.get("allow_lora"):
                if not compile_family:
                    raise ValueError(
                        f"{es.name}: allow_lora bindings require "
                        "Compile(family=...) on the endpoint"
                    )
                block["family"] = compile_family

        input_schema, input_sha = _schema_and_hash(es.payload_type)
        moderation = _collect_payload_moderation_metadata(es.payload_type)
        output_type = es.output_type
        if output_type is None:
            raise ValueError(
                f"{es.name}: manifest requires a concrete msgspec.Struct "
                "output/delta type"
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
                f"{es.name!r}: function name cannot be normalized"
            )

        fn: Dict[str, Any] = {
            "name": function_name,
            "python_name": es.attr_name or es.method.__name__,
            "module": module_name,
            "declared_module": es.module or module_name,
            "class_name": es.cls.__name__ if es.cls is not None else "",
            "kind": es.kind,
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
            "timeout_ms": es.timeout_ms,
        }
        if incremental and es.delta_type is not None:
            fn["delta_type"] = _type_id(es.delta_type)
            fn["delta_schema_sha256"] = delta_sha
            fn["delta_output_schema"] = delta_schema
        if es.compile is not None:
            # Hub keys family-cache lookups off this block (th#569).
            fn["compile"] = {
                "family": es.compile.family,
                "shapes": [[int(v) for v in s] for s in es.compile.shapes],
                "targets": list(es.compile.targets),
            }
            # ie#381: the primary binding's weight-storage lane (gw#389 fp8
            # layerwise casting) rides along so the hub's cell producer
            # builds from an identically-loaded pipeline — the cast hooks
            # are traced INTO the FX graphs; a bf16-built cell for an
            # fp8-served model misses on every request.
            primary = next(iter(es.models.values()), None)
            storage = str(getattr(primary, "storage_dtype", "") or "")
            if storage:
                fn["compile"]["storage_dtype"] = storage
            if getattr(es.compile, "regional", False):
                fn["compile"]["regional"] = True
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

    cfg = load_project_config(root)

    functions = discover_functions(root, main_module=cfg.main)

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
        print("warning: no @endpoint objects found", file=sys.stderr)

    sys.stdout.write(msgspec.toml.encode(_strip_none(manifest)).decode("utf-8"))
    if not sys.stdout.isatty():
        sys.stdout.write("\n")
