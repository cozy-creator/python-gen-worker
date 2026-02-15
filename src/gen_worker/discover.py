"""
Function discovery module for Cozy workers.

This module auto-discovers all @worker_function decorated functions in the project
by scanning .py files and extracting metadata. Run as:

    python -m gen_worker.discover

Outputs JSON manifest to stdout.
"""

import ast
import collections.abc
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import re
import sys
import typing
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import msgspec

from gen_worker import ActionContext
from gen_worker.injection import ModelRef

from gen_worker.cozy_toml import CozyModelSpec, CozyToml, load_cozy_toml
from gen_worker.names import slugify_endpoint_name, slugify_function_name

_ALLOWED_MODEL_DTYPES: frozenset[str] = frozenset({"fp16", "bf16", "fp8", "fp32", "int8", "int4"})


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


def _parse_annotated_model_ref(ann: Any) -> Optional[Tuple[type, ModelRef]]:
    """Extract ModelRef from Annotated type if present."""
    origin = typing.get_origin(ann)
    if origin is not typing.Annotated:
        return None
    args = typing.get_args(ann)
    if not args:
        return None
    base = args[0]
    for meta in args[1:]:
        if isinstance(meta, ModelRef):
            return base, meta
    return None


def _schema_and_hash(t: type) -> Tuple[Dict[str, Any], str]:
    """Generate JSON schema and SHA256 hash for a msgspec type."""
    schema = msgspec.json.schema(t)
    try:
        from .payload_constraints import apply_schema_constraints

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
    """Quick AST check if file uses @worker_function decorator."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                # Check for @worker_function or @worker_function(...)
                if isinstance(decorator, ast.Name) and decorator.id == "worker_function":
                    return True
                if isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Name) and decorator.func.id == "worker_function":
                        return True
                    if isinstance(decorator.func, ast.Attribute) and decorator.func.attr == "worker_function":
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
    max_concurrency = None
    if resources is not None:
        max_concurrency = getattr(resources, "max_concurrency", None)

    res_dict: Dict[str, Any] = {}
    if isinstance(max_concurrency, int):
        res_dict["max_concurrency"] = max_concurrency

    hints = typing.get_type_hints(func, globalns=func.__globals__, include_extras=True)
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if len(params) < 2:
        raise ValueError(
            f"{func.__name__}: must accept (ctx: ActionContext, payload: msgspec.Struct, ...)"
        )

    ctx_name = params[0].name
    if hints.get(ctx_name) is not ActionContext:
        raise ValueError(f"{func.__name__}: first param must be ctx: ActionContext")

    payload_type = None
    payload_param = None
    injections: List[Dict[str, Any]] = []

    for p in params[1:]:
        ann = hints.get(p.name)
        if ann is None:
            raise ValueError(f"{func.__name__}: missing type annotation for param {p.name}")

        inj = _parse_annotated_model_ref(ann)
        if inj is not None:
            base_t, mr = inj
            src = mr.source.value
            # Canonicalize older "release" terminology into "fixed" for manifests.
            if src == "release":
                src = "fixed"
            dtypes = []
            try:
                dtypes = [str(x).strip() for x in list(getattr(mr, "dtypes", ()) or ()) if str(x).strip()]
            except Exception:
                dtypes = []
            ref = ""
            try:
                ref = str(getattr(mr, "ref", "") or "").strip()
            except Exception:
                ref = ""
            injections.append({
                "param": p.name,
                "type": _type_qualname(base_t),
                "model_ref": {"source": src, "key": mr.key, "ref": ref, "dtypes": dtypes},
            })
            continue

        if _is_msgspec_struct(ann):
            if payload_type is not None:
                raise ValueError(
                    f"{func.__name__}: must accept exactly one msgspec.Struct payload"
                )
            payload_type = ann
            payload_param = p.name
            continue

        raise ValueError(f"{func.__name__}: unsupported param type for {p.name}: {ann!r}")

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
    output_schema, output_sha = _schema_and_hash(output_type)
    delta_schema = None
    delta_sha = ""
    if delta_type is not None:
        delta_schema, delta_sha = _schema_and_hash(delta_type)

    # Extract required_models: fixed-source model keys that must be available.
    # These are keys used by ModelRef(FIXED, "<key>", ...) injections.
    required_models = [
        inj["model_ref"]["key"]
        for inj in injections
        if inj.get("model_ref", {}).get("source") == "fixed"
    ]

    # Extract payload-based repo selectors so schedulers can compute required repos
    # at submit-time for cache-aware routing.
    payload_repo_selectors = []
    seen_fields = set()
    for inj in injections:
        mr = inj.get("model_ref", {}) or {}
        if mr.get("source") != "payload":
            continue
        field = str(mr.get("key") or "").strip()
        if not field or field in seen_fields:
            continue
        seen_fields.add(field)
        payload_repo_selectors.append({"field": field, "kind": "short_key"})

    function_name = slugify_function_name(func.__name__)
    if not function_name:
        raise ValueError(f"{func.__name__}: function name cannot be normalized")

    fn: Dict[str, Any] = {
        "name": function_name,
        "python_name": func.__name__,
        "module": module_name,
        "resources": res_dict,
        "payload_type": _type_id(payload_type),
        "payload_schema_sha256": input_sha,
        "input_schema": input_schema,
        "output_mode": output_mode,
        "output_type": _type_id(output_type),
        "output_schema_sha256": output_sha,
        "output_schema": output_schema,
        "incremental_output": incremental,
        "injection_json": injections,
        "required_models": required_models,  # release model keys needed by this function
        "payload_repo_selectors": payload_repo_selectors,
    }

    if delta_type is not None:
        fn["delta_type"] = _type_id(delta_type)
        fn["delta_schema_sha256"] = delta_sha
        fn["delta_output_schema"] = delta_schema

    return fn


def _find_cozy_toml_path(root: Path) -> Path | None:
    env_path = os.getenv("COZY_MANIFEST_PATH", "").strip()
    if env_path:
        p = Path(env_path)
        return p if p.exists() else None
    p = root / "cozy.toml"
    return p if p.exists() else None


def _load_cozy_manifest_toml(root: Path) -> CozyToml:
    p = _find_cozy_toml_path(root)
    if p is None:
        raise ValueError("missing cozy.toml (required for discovery)")
    return load_cozy_toml(p)


def _model_spec_to_json(spec: CozyModelSpec) -> Dict[str, Any]:
    return {"ref": spec.ref, "dtypes": list(spec.dtypes)}


def _models_by_key_to_json(models: Dict[str, CozyModelSpec]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, spec in models.items():
        out[str(k)] = _model_spec_to_json(spec)
    return out


def discover_functions(root: Optional[Path] = None, *, main_module: str | None = None) -> List[Dict[str, Any]]:
    """
    Discover all @worker_function decorated functions in the project.

    Args:
        root: Project root directory. Defaults to current working directory.

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
    seen_functions: Set[Tuple[str, str]] = set()  # (module, name)

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
        before = set(sys.modules.keys())
        try:
            importlib.import_module(main_module)
        except Exception as e:
            raise ValueError(f"failed to import cozy.toml main module {main_module!r}: {e}") from e

        after = set(sys.modules.keys())
        newly_loaded = sorted(after - before)
        # Also include the main module itself (it was present in before in some cases).
        candidates = set(newly_loaded)
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
            # Iterate module dict to avoid triggering module-level __getattr__.
            for obj in mod.__dict__.values():
                if not inspect.isfunction(obj) or not getattr(obj, "_is_worker_function", False):
                    continue
                key = (getattr(obj, "__module__", ""), getattr(obj, "__name__", ""))
                if key in seen_functions:
                    continue
                seen_functions.add(key)
                fn_meta = _extract_function_metadata(obj, module_name)
                functions.append(fn_meta)
        return functions

    # Fallback: scan the filesystem for decorated functions.
    py_files = _find_python_files(root)
    candidate_files = [f for f in py_files if _file_uses_worker_decorator(f)]
    for filepath in candidate_files:
        module_name = _compute_module_name(filepath, root)
        if module_name is None or module_name in imported_modules:
            continue
        imported_modules.add(module_name)
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            print(f"warning: failed to import {module_name}: {e}", file=sys.stderr)
            continue

        # Find decorated functions
        # Iterate module dict to avoid triggering module-level __getattr__.
        for name, obj in mod.__dict__.items():
            if not inspect.isfunction(obj) or not getattr(obj, "_is_worker_function", False):
                continue
            key = (getattr(obj, "__module__", ""), getattr(obj, "__name__", ""))
            if key in seen_functions:
                continue
            seen_functions.add(key)
            try:
                fn_meta = _extract_function_metadata(obj, module_name)
                functions.append(fn_meta)
            except Exception as e:
                print(f"warning: failed to extract metadata from {name}: {e}", file=sys.stderr)
                raise

    return functions


def discover_manifest(root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Discover functions and load Cozy manifest config to build complete manifest.

    Args:
        root: Project root directory. Defaults to current working directory.

    Returns: Complete manifest dict with functions + models/resources metadata.
    """
    if root is None:
        root = Path.cwd()
    root = root.resolve()

    cozy = _load_cozy_manifest_toml(root)

    functions = discover_functions(root, main_module=cozy.main)
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

    endpoint_name = slugify_endpoint_name(cozy.name)
    if not endpoint_name:
        raise ValueError("invalid cozy.toml name")

    manifest: Dict[str, Any] = {
        "endpoint_name": endpoint_name,
        "functions": functions,
    }

    if cozy.resources:
        manifest["resources"] = dict(cozy.resources)

    # Build per-function model keyspace.
    models_by_function: Dict[str, Any] = {}
    for fn in functions:
        fn_name = str(fn.get("name") or "").strip()
        if not fn_name:
            continue

        # Function-specific mapping wins; otherwise fall back to global [models].
        eff = cozy.function_models.get(fn_name) or cozy.models
        eff2: Dict[str, CozyModelSpec] = dict(eff or {})

        # If a fixed ModelRef has an explicit ref (and optional dtypes) in the signature,
        # auto-add it to the effective mapping so cozy.toml [models] isn't required for
        # fixed functions.
        required_keys = set(fn.get("required_models", []) or [])
        inj_list = list(fn.get("injection_json", []) or [])
        sig_specs: Dict[str, CozyModelSpec] = {}
        for inj in inj_list:
            mr = dict(inj.get("model_ref", {}) or {})
            if mr.get("source") != "fixed":
                continue
            key = str(mr.get("key") or "").strip()
            if not key:
                continue
            ref = str(mr.get("ref") or "").strip()
            if not ref:
                # Fallback: allow using a full repo ref directly as the key.
                # (Still treated as a FIXED keyspace entry; useful for quick examples.)
                if "/" in key:
                    ref = key
                else:
                    continue
            dtypes_raw = mr.get("dtypes")
            dtypes: tuple[str, ...] = ()
            if isinstance(dtypes_raw, list):
                cleaned = tuple(str(x).strip() for x in dtypes_raw if str(x).strip())
                bad = sorted({x for x in cleaned if x not in _ALLOWED_MODEL_DTYPES})
                if bad:
                    raise ValueError(f"function '{fn_name}' has invalid model dtypes for {key!r}: {bad}")
                dtypes = cleaned
            if dtypes:
                sig_specs[key] = CozyModelSpec(ref=ref, dtypes=dtypes)
            else:
                sig_specs[key] = CozyModelSpec(ref=ref)

        missing = []
        for k in sorted(required_keys):
            if k in eff2:
                continue
            if k in sig_specs:
                eff2[k] = sig_specs[k]
                continue
            missing.append(k)
        if missing:
            raise ValueError(
                f"function '{fn_name}' requires model keys not defined in cozy.toml [models] "
                f"and not provided via signature ModelRef(ref=...): {missing}"
            )
        if eff2:
            models_by_function[fn_name] = _models_by_key_to_json(eff2)

    if models_by_function:
        manifest["models_by_function"] = models_by_function

    return manifest


def main() -> None:
    """Main entry point for CLI usage."""
    # Check for legacy COZY_FUNCTION_MODULES env var
    legacy_modules = os.getenv("COZY_FUNCTION_MODULES", "").strip()
    if legacy_modules:
        print(
            "warning: COZY_FUNCTION_MODULES is deprecated; using auto-discovery instead",
            file=sys.stderr,
        )

    try:
        manifest = discover_manifest()
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    if not manifest.get("functions"):
        print("warning: no @worker_function decorated functions found", file=sys.stderr)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
