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

import tomllib  # Python 3.11+ built-in


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
    raw = json.dumps(schema, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return schema, hashlib.sha256(raw).hexdigest()


_NON_SLUG_CHARS = re.compile(r"[^a-z0-9.]+")
_DUP_SLUG_SEPARATORS = re.compile(r"-{2,}")


def _slugify_name(raw: str) -> str:
    raw = raw.strip().lower().replace("_", "-")
    if not raw:
        return ""
    raw = _NON_SLUG_CHARS.sub("-", raw)
    raw = _DUP_SLUG_SEPARATORS.sub("-", raw)
    raw = raw.strip("-.")
    if len(raw) > 128:
        raw = raw[:128].strip("-.")
    return raw


def _slugify_endpoint_name(raw: str) -> str:
    return _slugify_name(raw)


def _slugify_project_name(raw: str) -> str:
    return _slugify_name(raw)


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
            injections.append({
                "param": p.name,
                "type": _type_qualname(base_t),
                "model_ref": {"source": src, "key": mr.key},
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
    # These are short-keys declared in [models] (cozy.toml) that the function needs.
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

    endpoint_name = _slugify_endpoint_name(func.__name__)
    if not endpoint_name:
        raise ValueError(f"{func.__name__}: function name cannot be normalized to endpoint_name")

    fn: Dict[str, Any] = {
        "name": func.__name__,
        "endpoint_name": endpoint_name,
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


COZY_TOML_IMAGE_PATH = Path("/cozy/cozy.toml")
COZY_TOML_IMAGE_PATH_ALIAS = Path("/cozy/manifest.toml")


def _load_cozy_manifest_toml(root: Path) -> Dict[str, Any]:
    """Load Cozy manifest config from cozy.toml (flat schema)."""
    config: Dict[str, Any] = {}

    env_path = os.getenv("COZY_MANIFEST_PATH", "").strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(COZY_TOML_IMAGE_PATH)
    candidates.append(COZY_TOML_IMAGE_PATH_ALIAS)
    candidates.append(root / "cozy.toml")

    data: Dict[str, Any] | None = None
    for p in candidates:
        try:
            if not p.exists():
                continue
            data = tomllib.loads(p.read_text(encoding="utf-8"))
            break
        except Exception as e:
            print(f"warning: failed to parse cozy.toml at {p}: {e}", file=sys.stderr)
            return config

    if not isinstance(data, dict):
        return config

    schema_version = data.get("schema_version")
    if schema_version != 1:
        return config

    raw_name = data.get("name")
    if isinstance(raw_name, str):
        project_name = _slugify_project_name(raw_name)
        if project_name:
            config["project_name"] = project_name

    main = data.get("main")
    if isinstance(main, str) and main.strip():
        config["main"] = main.strip()

    gen_worker = data.get("gen_worker")
    if isinstance(gen_worker, str) and gen_worker.strip():
        config["gen_worker"] = gen_worker.strip()

    host = data.get("host")
    if isinstance(host, dict):
        req = host.get("requirements")
        if isinstance(req, dict):
            cuda = req.get("cuda")
            if isinstance(cuda, str) and cuda.strip():
                config["host_requirements"] = {"cuda": cuda.strip()}

    models = data.get("models")
    if isinstance(models, dict):
        config["models"] = {str(k): str(v) for k, v in models.items() if str(k).strip() and str(v).strip()}

    resources = data.get("resources")
    if isinstance(resources, dict):
        # Keep as-is but only allow the known keys.
        out: Dict[str, Any] = {}
        for k in ("vram_gb", "ram_gb", "cpu_cores", "disk_gb"):
            if k in resources:
                out[k] = resources[k]
        if out:
            config["resources"] = out

    return config


def discover_functions(root: Optional[Path] = None) -> List[Dict[str, Any]]:
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

    # Find Python files that might have worker functions
    py_files = _find_python_files(root)

    # Filter to files that use @worker_function (quick AST check)
    candidate_files = [f for f in py_files if _file_uses_worker_decorator(f)]

    # Compute module names and import
    functions: List[Dict[str, Any]] = []
    imported_modules: Set[str] = set()

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
        for name, obj in inspect.getmembers(mod):
            if inspect.isfunction(obj) and getattr(obj, "_is_worker_function", False):
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

    functions = discover_functions(root)
    endpoint_to_function: Dict[str, str] = {}
    for fn in functions:
        endpoint_name = str(fn.get("endpoint_name", "")).strip()
        function_name = str(fn.get("name", "")).strip()
        if not endpoint_name:
            raise ValueError(f"{function_name or '<unknown>'}: missing endpoint_name in discovered metadata")
        prior = endpoint_to_function.get(endpoint_name)
        if prior and prior != function_name:
            raise ValueError(
                f"multiple functions normalize to the same endpoint '{endpoint_name}': {prior}, {function_name}"
            )
        endpoint_to_function[endpoint_name] = function_name
    config = _load_cozy_manifest_toml(root)

    project_name = str(config.get("project_name", "")).strip()
    if not project_name:
        raise ValueError("missing cozy.toml name (flat schema: name=...)")

    manifest: Dict[str, Any] = {
        "project_name": project_name,
        "functions": functions,
    }

    if "resources" in config:
        manifest["resources"] = config["resources"]

    # Extract all required model keys from functions (static model source only)
    all_required_keys: Set[str] = set()
    for fn in functions:
        required = fn.get("required_models", [])
        all_required_keys.update(required)

    # Get models from [models] in cozy.toml
    config_models: Dict[str, str] = config.get("models", {})

    # Validate: all required model keys must be defined in config
    missing_keys = all_required_keys - set(config_models.keys())
    if missing_keys:
        print(
            f"warning: functions require model keys not defined in [models] (cozy.toml): {sorted(missing_keys)}",
            file=sys.stderr,
        )

    # Include models in manifest if we have any
    if config_models:
        manifest["models"] = config_models

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
