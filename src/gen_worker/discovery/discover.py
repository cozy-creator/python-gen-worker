"""
Function discovery module for Cozy workers.

This module auto-discovers all @inference decorated functions in the project
by scanning .py files and extracting metadata. Run as:

    python -m gen_worker.discovery

Outputs TOML endpoint lock to stdout.
"""

import ast
import collections.abc
import hashlib
import importlib
import importlib.util
import inspect
import json
import sys
import typing
import types as py_types
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import msgspec

from gen_worker import RequestContext
from gen_worker.api.binding import Binding, Dispatch, HFRepo, Repo
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


def _should_skip_path(path: Path, skip_patterns: Set[str]) -> bool:
    """Check if path should be skipped based on common patterns."""
    parts = path.parts
    for pattern in skip_patterns:
        if pattern in parts:
            return True
    # Skip hidden directories
    for part in parts:
        if part.startswith(".") and part not in (".", ".."):
            return True
    return False


def _find_python_files(root: Path, skip_patterns: Optional[Set[str]] = None) -> List[Path]:
    """Find all .py files in the project, respecting skip patterns."""
    if skip_patterns is None:
        skip_patterns = {
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".tox",
            "node_modules",
            "dist",
            "build",
            ".eggs",
            "*.egg-info",
        }

    py_files = []
    for path in root.rglob("*.py"):
        if not _should_skip_path(path.relative_to(root), skip_patterns):
            py_files.append(path)
    return py_files


def _file_uses_worker_decorator(filepath: Path) -> bool:
    """Quick AST check if file uses @inference or @conversion."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return False

    target_names = {"inference_function", "training_function"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                # @inference, @conversion (bare)
                if isinstance(decorator, ast.Name) and decorator.id in target_names:
                    return True
                # @inference(...), @conversion(kind=...)
                if isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Name) and decorator.func.id in target_names:
                        return True
                    if isinstance(decorator.func, ast.Attribute) and decorator.func.attr in target_names:
                        return True
    return False


def _compute_module_name(filepath: Path, root: Path) -> Optional[str]:
    """Compute Python module name from file path."""
    try:
        rel = filepath.relative_to(root)
    except ValueError:
        return None

    # Check if it's in a src/ directory
    parts = list(rel.parts)
    if parts and parts[0] == "src":
        parts = parts[1:]

    if not parts:
        return None

    # Remove .py extension
    parts[-1] = parts[-1].rsplit(".", 1)[0]

    # Handle __init__.py -> use parent module name
    if parts[-1] == "__init__":
        parts = parts[:-1]

    if not parts:
        return None

    return ".".join(parts)


def _extract_function_metadata(func: Any, module_name: str) -> Dict[str, Any]:
    """Extract metadata from a worker function."""
    resources = getattr(func, "_worker_resources", None)
    res_dict: Dict[str, Any] = {}
    if resources is not None:
        try:
            raw = msgspec.to_builtins(resources)
            if isinstance(raw, dict):
                res_dict.update(raw)
        except Exception:
            pass

    hints = typing.get_type_hints(func, globalns=func.__globals__, include_extras=True)
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if len(params) < 2:
        raise ValueError(
            f"{func.__name__}: must accept (ctx: RequestContext, payload: msgspec.Struct, ...)"
        )

    ctx_name = params[0].name
    if hints.get(ctx_name) is not RequestContext:
        raise ValueError(f"{func.__name__}: first param must be ctx: RequestContext")

    # Tensorhub #232: ``compute`` is a reserved payload key the orchestrator
    # strips before dispatch. Tenants who want the resolved hardware read
    # ``ctx.compute`` — not a reserved parameter. Reject the name to catch
    # the confusion early at publish.
    for p in params[1:]:
        if p.name == "compute":
            raise ValueError(
                f"{func.__name__}: `compute` is a reserved parameter name "
                "(tensorhub #232). Read the resolved hardware via ctx.compute "
                "instead of declaring it on the function signature."
            )

    bindings_map: Dict[str, Binding] = dict(getattr(func, "__gen_worker_bindings__", None) or {})

    payload_type = None
    payload_param = None
    bindings_block: Dict[str, Dict[str, Any]] = {}

    for p in params[1:]:
        ann = hints.get(p.name)
        if ann is None:
            raise ValueError(f"{func.__name__}: missing type annotation for param {p.name}")

        if p.name in bindings_map:
            bindings_block[p.name] = _binding_to_manifest(bindings_map[p.name], ann, p.name)
            continue

        if _is_msgspec_struct(ann):
            if payload_type is not None:
                raise ValueError(
                    f"{func.__name__}: must accept exactly one msgspec.Struct payload"
                )
            payload_type = ann
            payload_param = p.name
            continue

        raise ValueError(
            f"{func.__name__}: unsupported param type for {p.name}: {ann!r} "
            "(inject models via @inference(models={...}); payload must be msgspec.Struct)"
        )

    if payload_type is None or payload_param is None:
        raise ValueError(f"{func.__name__}: missing msgspec.Struct payload param")

    ret = hints.get("return")
    if ret is None:
        raise ValueError(f"{func.__name__}: missing return type annotation")

    output_mode = "single"
    incremental = False
    output_type = None
    delta_type = None

    if _is_msgspec_struct(ret):
        output_type = ret
    else:
        origin = typing.get_origin(ret)
        if origin in (
            typing.Iterator,
            typing.Iterable,
            collections.abc.Iterator,
            collections.abc.Iterable,
        ):
            args = typing.get_args(ret)
            if len(args) != 1 or not _is_msgspec_struct(args[0]):
                raise ValueError(
                    f"{func.__name__}: incremental output return must be Iterator[msgspec.Struct]"
                )
            incremental = True
            output_mode = "incremental"
            delta_type = args[0]
            output_type = args[0]
        else:
            raise ValueError(
                f"{func.__name__}: return type must be msgspec.Struct or Iterator[msgspec.Struct]"
            )

    input_schema, input_sha = _schema_and_hash(payload_type)
    moderation = _collect_payload_moderation_metadata(payload_type)
    output_schema, output_sha = _schema_and_hash(output_type)
    expected_outputs = _collect_expected_output_metadata(payload_type, output_type)
    delta_schema = None
    delta_sha = ""
    if delta_type is not None:
        delta_schema, delta_sha = _schema_and_hash(delta_type)

    function_name = slugify_name(func.__name__)
    if not function_name:
        raise ValueError(f"{func.__name__}: function name cannot be normalized")

    fn_label = getattr(func, "_function_label", None) or None
    fn_description = getattr(func, "_function_description", None) or None

    fn: Dict[str, Any] = {
        "name": function_name,
        "python_name": func.__name__,
        "module": module_name,
        "resources": res_dict,
        "bindings": bindings_block,
        "payload_type": _type_id(payload_type),
        "payload_schema_sha256": input_sha,
        "input_schema": input_schema,
        "moderation": moderation,
        "expected_outputs": expected_outputs,
        "output_mode": output_mode,
        "output_type": _type_id(output_type),
        "output_schema_sha256": output_sha,
        "output_schema": output_schema,
        "incremental_output": incremental,
        "decorator": "inference_function",
        "label": fn_label,
        "description": fn_description,
    }

    if delta_type is not None:
        fn["delta_type"] = _type_id(delta_type)
        fn["delta_schema_sha256"] = delta_sha
        fn["delta_output_schema"] = delta_schema

    return fn


def _extract_conversion_function_metadata(func: Any, module_name: str) -> Dict[str, Any]:
    """Extract endpoint.lock metadata from a @conversion-decorated tenant.

    Reads ``__training_spec__`` attached by the decorator — fetches the
    tenant-declared ``kind`` (issue #10), the full wire-payload JSON schema
    (issue #5), and the ref_registry for capability-token scoping. Returns a
    dict compatible with the endpoint.lock ``functions`` entry shape used by
    @inference discovery, so downstream consumers (orchestrator
    ``FunctionMetadata``, tensorhub publish-time validator) can read one
    unified format.
    """
    spec = getattr(func, "__training_spec__", None)
    if spec is None:
        raise ValueError(f"{func.__name__}: missing __training_spec__ (expected @conversion decoration)")

    resources = getattr(func, "_worker_resources", None)
    res_dict: Dict[str, Any] = {}
    if resources is not None:
        try:
            raw = msgspec.to_builtins(resources)
            if isinstance(raw, dict):
                res_dict.update(raw)
        except Exception:
            pass

    function_name = slugify_name(func.__name__)
    if not function_name:
        raise ValueError(f"{func.__name__}: function name cannot be normalized")

    input_schema = dict(spec.input_schema or {})
    input_schema_bytes = json.dumps(input_schema, separators=(",", ":"), sort_keys=True).encode("utf-8")
    input_sha = hashlib.sha256(input_schema_bytes).hexdigest()

    # Training functions: emit a `payload_refs` block listing the wire-field
    # → param mapping for each secondary Source materialization. Inference's
    # `bindings` block doesn't apply; training stays on the legacy wire-field
    # pattern for now.
    payload_refs: Dict[str, Dict[str, str]] = {}
    for wire_field, param_name in sorted(spec.ref_registry.items()):
        payload_refs[wire_field] = {"param": param_name, "kind": "source"}

    fn: Dict[str, Any] = {
        "name": function_name,
        "python_name": func.__name__,
        "module": module_name,
        "resources": res_dict,
        # Conversion functions have a single structured return (list[ProducedFlavor])
        # that the library uploads. Tenant input schema captures the full wire
        # payload.
        "payload_type": {"module": "", "qualname": "gen_worker.conversion.WirePayload"},
        "payload_schema_sha256": input_sha,
        "input_schema": input_schema,
        "output_mode": "single",
        "output_type": {"module": "gen_worker.conversion", "qualname": "list[ProducedFlavor]"},
        "output_schema_sha256": "",
        "output_schema": {},
        "incremental_output": False,
        "payload_refs": payload_refs,
        "kind": spec.kind,
        "decorator": "training_function",
        "label": getattr(func, "_function_label", None) or None,
        "description": getattr(func, "_function_description", None) or None,
    }
    return fn


def _find_endpoint_toml_path(root: Path) -> Path | None:
    p = root / "endpoint.toml"
    return p if p.exists() else None


def _load_endpoint_manifest_toml(root: Path) -> EndpointToml:
    p = _find_endpoint_toml_path(root)
    if p is None:
        raise ValueError("missing endpoint.toml (required for discovery)")
    return load_endpoint_toml(p)


def discover_functions(root: Optional[Path] = None, *, main_module: str | None = None) -> List[Dict[str, Any]]:
    """
    Discover all @inference decorated functions in the project.

    Args:
        root: Project root directory. Defaults to current working directory.
        main_module: ``endpoint.toml``'s ``main`` pointer (e.g. ``"conversion.main"``).
            When provided, the top-level package (``"conversion"``) is walked
            via :func:`gen_worker.discovery.walk.find_endpoint_classes` to find
            class-shape endpoints. When omitted, the fallback filesystem scan
            still picks up function-shape decorators.

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

        # Function-shape (@inference / @conversion at module level) still
        # ships through the per-module dict scan — only the class-discovery
        # path was duplicated between build-time and runtime, so that's
        # the path we unified. Function-shape decorators are legacy stubs
        # post-#322 but still scanned so a tenant who slipped one in gets
        # a loud build-time error rather than a silent miss.
        before = set(sys.modules.keys())
        try:
            importlib.import_module(main_module)
        except Exception as e:
            raise ValueError(
                f"failed to import endpoint.toml main module {main_module!r}: {e}"
            ) from e
        after = set(sys.modules.keys())
        candidates = set(sorted(after - before))
        candidates.add(main_module)
        for module_name in sorted(candidates):
            if module_name in imported_modules:
                continue
            imported_modules.add(module_name)
            mod = sys.modules.get(module_name)
            if mod is None:
                continue
            if not _module_is_in_project(mod):
                continue
            for obj in mod.__dict__.values():
                if not inspect.isfunction(obj):
                    continue
                is_worker = getattr(obj, "_is_inference_function", False)
                is_conversion = getattr(obj, "_is_training_function", False)
                if not (is_worker or is_conversion):
                    continue
                key = (
                    getattr(obj, "__module__", ""),
                    "",
                    getattr(obj, "__name__", ""),
                )
                if key in seen_functions:
                    continue
                seen_functions.add(key)
                if is_conversion:
                    fn_meta = _extract_conversion_function_metadata(obj, module_name)
                else:
                    fn_meta = _extract_function_metadata(obj, module_name)
                functions.append(fn_meta)
        return functions

    # Fallback: no main_module declared. Scan the filesystem for decorated
    # functions (legacy function-shape) AND walk every top-level package
    # under the project root for class-shape endpoints. The fallback is
    # rarely hit in practice — every published endpoint declares ``main``
    # in endpoint.toml — but exists so ad-hoc invocations of
    # ``discover_functions()`` against a project still work.
    py_files = _find_python_files(root)
    candidate_files = [f for f in py_files if _file_uses_worker_decorator(f)]

    # Class-shape: walk every top-level package the filesystem suggests.
    top_level_pkgs: Set[str] = set()
    for filepath in candidate_files:
        computed_module_name = _compute_module_name(filepath, root)
        if computed_module_name:
            top_level_pkgs.add(computed_module_name.split(".", 1)[0])
    if top_level_pkgs:
        try:
            found_classes = find_endpoint_classes(sorted(top_level_pkgs))
        except Exception as e:
            print(f"warning: walker failed: {e}", file=sys.stderr)
            found_classes = []
        for found in found_classes:
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

    # Function-shape: per-file import then module-dict scan.
    for filepath in candidate_files:
        computed_module_name = _compute_module_name(filepath, root)
        if computed_module_name is None or computed_module_name in imported_modules:
            continue
        imported_modules.add(computed_module_name)
        try:
            mod = importlib.import_module(computed_module_name)
        except Exception as e:
            print(f"warning: failed to import {computed_module_name}: {e}", file=sys.stderr)
            continue

        for name, obj in mod.__dict__.items():
            if not inspect.isfunction(obj):
                continue
            is_worker = getattr(obj, "_is_inference_function", False)
            is_conversion = getattr(obj, "_is_training_function", False)
            if not (is_worker or is_conversion):
                continue
            key = (
                getattr(obj, "__module__", ""),
                "",
                getattr(obj, "__name__", ""),
            )
            if key in seen_functions:
                continue
            seen_functions.add(key)
            try:
                if is_conversion:
                    fn_meta = _extract_conversion_function_metadata(obj, computed_module_name)
                else:
                    fn_meta = _extract_function_metadata(obj, computed_module_name)
                functions.append(fn_meta)
            except Exception as e:
                print(f"warning: failed to extract metadata from {name}: {e}", file=sys.stderr)
                raise

    return functions


def _extract_class_function_methods(
    cls: type, module_name: str
) -> List[Dict[str, Any]]:
    """Extract per-function manifest entries from a class-shape endpoint (#322).

    A class decorated with ``@inference`` / ``@training`` / ``@dataset`` /
    ``@conversion`` carries one or more ``@inference.function``-decorated
    methods. Each method becomes a separate entry in the manifest's
    ``functions`` list, sharing the class-level resources/bindings.
    """
    spec = getattr(cls, "__gen_worker_endpoint_spec__", None)
    if spec is None:
        return []

    function_methods = getattr(cls, "__gen_worker_function_methods__", None) or []
    if not function_methods:
        return []

    # Class-level resources serialize once; shared across all functions.
    res_dict: Dict[str, Any] = {}
    if spec.resources is not None:
        try:
            raw = msgspec.to_builtins(spec.resources)
            if isinstance(raw, dict):
                res_dict.update(raw)
        except Exception:
            pass

    # Class-level bindings — same shape as old function-decorator bindings,
    # keyed by setup() kwarg name. For each method, the bindings block is
    # the same (every method shares the loaded models).
    bindings_map: Dict[str, Binding] = dict(spec.models or {})
    bindings_block: Dict[str, Dict[str, Any]] = {}
    for key, binding in bindings_map.items():
        bindings_block[key] = _binding_to_manifest(binding, None, key)

    out: List[Dict[str, Any]] = []
    for attr_name, method, fn_spec in function_methods:
        # Method signature: (self, ctx, payload) — skip self.
        hints = typing.get_type_hints(method, include_extras=True)
        sig = inspect.signature(method)
        params = [p for p in sig.parameters.values() if p.name != "self"]

        if len(params) < 2:
            raise ValueError(
                f"{cls.__name__}.{attr_name}: must accept (self, ctx, payload). "
                f"Got params: {[p.name for p in params]}"
            )

        # First non-self param is ctx; second is payload struct.
        ctx_name = params[0].name
        payload_param = params[1]
        payload_ann = hints.get(payload_param.name)
        if payload_ann is None or not _is_msgspec_struct(payload_ann):
            raise ValueError(
                f"{cls.__name__}.{attr_name}: payload param {payload_param.name!r} "
                f"must be a msgspec.Struct (got {payload_ann!r})"
            )

        payload_type = payload_ann

        ret = hints.get("return")
        if ret is None:
            raise ValueError(
                f"{cls.__name__}.{attr_name}: missing return type annotation"
            )

        output_mode = "single"
        incremental = False
        output_type: Optional[type] = None
        delta_type: Optional[type] = None

        # BatchedWorker async generator returning AsyncIterator[X] OR
        # SerialWorker sync function returning Iterator[X] → incremental.
        origin = typing.get_origin(ret)
        if _is_msgspec_struct(ret):
            output_type = ret
        elif origin in (
            typing.Iterator,
            typing.Iterable,
            typing.AsyncIterator,
            typing.AsyncIterable,
            collections.abc.Iterator,
            collections.abc.Iterable,
            collections.abc.AsyncIterator,
            collections.abc.AsyncIterable,
        ):
            args = typing.get_args(ret)
            if len(args) != 1 or not _is_msgspec_struct(args[0]):
                raise ValueError(
                    f"{cls.__name__}.{attr_name}: incremental return must be "
                    f"Iterator[msgspec.Struct] or AsyncIterator[msgspec.Struct]"
                )
            incremental = True
            output_mode = "incremental"
            delta_type = args[0]
            output_type = args[0]
        else:
            raise ValueError(
                f"{cls.__name__}.{attr_name}: return type must be msgspec.Struct "
                f"or (Async)Iterator[msgspec.Struct], got {ret!r}"
            )

        input_schema, input_sha = _schema_and_hash(payload_type)
        moderation = _collect_payload_moderation_metadata(payload_type)
        output_schema, output_sha = _schema_and_hash(output_type)
        expected_outputs = _collect_expected_output_metadata(payload_type, output_type)
        delta_schema = None
        delta_sha = ""
        if delta_type is not None:
            delta_schema, delta_sha = _schema_and_hash(delta_type)

        function_name = slugify_name(fn_spec.name)
        if not function_name:
            raise ValueError(
                f"{cls.__name__}.{attr_name}: function name cannot be normalized "
                f"from {fn_spec.name!r}"
            )

        fn: Dict[str, Any] = {
            "name": function_name,
            "python_name": attr_name,
            "module": module_name,
            # Class's declaration module (vs the module being walked) — used
            # by discover_manifest to dedup re-exported classes.
            "declared_module": getattr(cls, "__module__", "") or module_name,
            "class_name": cls.__name__,
            "archetype": getattr(cls, "__gen_worker_archetype__", "SerialWorker"),
            "kind": spec.kind,
            "sub_kind": spec.sub_kind,
            "runtime": spec.runtime,
            "resources": res_dict,
            "bindings": bindings_block,
            "payload_type": _type_id(payload_type),
            "payload_schema_sha256": input_sha,
            "input_schema": input_schema,
            "moderation": moderation,
            "expected_outputs": expected_outputs,
            "output_mode": output_mode,
            "output_type": _type_id(output_type) if output_type else None,
            "output_schema_sha256": output_sha,
            "output_schema": output_schema,
            "incremental_output": incremental,
            # #345 Improvement B: surface whether the handler is an `async def`
            # (coroutine or async generator). Derived from inspect — never
            # tenant-declared. Lets the orchestrator/operators see which
            # endpoints run coroutine-bound (high-concurrency I/O) vs
            # ThreadPoolExecutor-bound (sync) on the SerialWorker archetype.
            "is_async": bool(
                inspect.iscoroutinefunction(method)
                or inspect.isasyncgenfunction(method)
            ),
            "decorator": f"@{spec.kind}.function",
            "label": fn_spec.label,
            "description": fn_spec.description,
            "timeout_ms": fn_spec.timeout_ms,
            "allowed_shapes": [list(s) for s in fn_spec.allowed_shapes]
                              or [list(s) for s in spec.allowed_shapes],
            # #324: SerialWorker cross-request micro-batching declaration.
            # Surfaced so the orchestrator can scorecard / route around this
            # at scheduling time. Both must be set for the worker to actually
            # enable batching; either-None means "no batching".
            "batch_window_ms": getattr(spec, "batch_window_ms", None),
            "max_batch": getattr(spec, "max_batch", None),
            # #324: Distilled-checkpoint awareness. When True the orchestrator
            # may auto-substitute a Lightning/Turbo/Schnell/Sprint variant of
            # the declared model Repo(...) at resolve time. SLA-aware
            # automatic selection is a follow-up depending on #320.
            "prefer_distilled": bool(getattr(spec, "prefer_distilled", False)),
        }

        if delta_type is not None:
            fn["delta_type"] = _type_id(delta_type)
            fn["delta_schema_sha256"] = delta_sha
            fn["delta_output_schema"] = delta_schema

        # Stage methods on the class — emit once per class as a flat list
        # under the first function entry (same for every function on the class,
        # but downstream consumers index by function so this is the natural place).
        stage_methods = getattr(cls, "__gen_worker_stage_methods__", None) or []
        if stage_methods:
            fn["stages"] = [
                {
                    "python_name": stage_attr,
                    "name": stage_spec.name,
                    "gpu_class": stage_spec.gpu_class,
                }
                for stage_attr, _stage_method, stage_spec in stage_methods
            ]

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
    for fn in functions:
        fn_name = str(fn.get("name") or "").strip()
        if not fn_name:
            continue
        batch_path = (tensorhub_manifest.function_batch_dimensions.get(fn_name) or "").strip()
        if batch_path:
            fn["batch_dimension"] = batch_path

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
