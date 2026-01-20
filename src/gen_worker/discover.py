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
            injections.append({
                "param": p.name,
                "type": _type_qualname(base_t),
                "model_ref": {"source": mr.source.value, "key": mr.key},
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

    # Extract required_models: deployment-source model keys that must be available
    # These are models declared in [tool.cozy.models] that the function needs
    required_models = [
        inj["model_ref"]["key"]
        for inj in injections
        if inj.get("model_ref", {}).get("source") == "deployment"
    ]

    fn: Dict[str, Any] = {
        "name": func.__name__,
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
        "required_models": required_models,  # deployment model keys needed by this function
    }

    if delta_type is not None:
        fn["delta_type"] = _type_id(delta_type)
        fn["delta_schema_sha256"] = delta_sha
        fn["delta_output_schema"] = delta_schema

    return fn


def _is_valid_deployment_id(deployment_id: str) -> bool:
    """
    Validate deployment ID format.

    Valid format: lowercase alphanumeric with hyphens, must start with letter,
    3-63 characters (DNS-like subdomain rules).
    """
    import re
    if not deployment_id:
        return False
    # Must be 3-63 chars, start with letter, only lowercase alphanumeric and hyphens
    # Cannot end with hyphen or have consecutive hyphens
    pattern = r'^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$'
    if not re.match(pattern, deployment_id):
        return False
    if len(deployment_id) < 3 or len(deployment_id) > 63:
        return False
    return True


def _load_cozy_config(root: Path) -> Dict[str, Any]:
    """
    Load [tool.cozy] config from pyproject.toml.

    Returns dict with:
        - deployment: default deployment ID
        - build: dict with gpu, cuda, torch, backend, base_image settings
        - models: dict mapping deployment keys to Cozy Hub model IDs
        - resources: dict with vram_gb, gpu_type, memory_gb, cpu_cores

    python-worker is the source of truth for all [tool.cozy.*] config parsing.
    gen-builder extracts this manifest and forwards it to the orchestrator.
    """
    config: Dict[str, Any] = {}
    pyproject_path = root / "pyproject.toml"

    if not pyproject_path.exists():
        return config

    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        print(f"warning: failed to parse pyproject.toml: {e}", file=sys.stderr)
        return config

    tool_cozy = data.get("tool", {}).get("cozy", {})

    # Parse [tool.cozy].deployment - default deployment ID
    deployment = tool_cozy.get("deployment")
    if deployment and isinstance(deployment, str):
        deployment = deployment.strip()
        if _is_valid_deployment_id(deployment):
            config["deployment"] = deployment
        else:
            print(
                f"warning: invalid deployment ID '{deployment}' - must be 3-63 chars, "
                "lowercase alphanumeric with hyphens, start with letter",
                file=sys.stderr,
            )
    if not tool_cozy:
        return config

    # Parse [tool.cozy.build] - build settings (gpu, cuda, torch, backend, base_image)
    build = tool_cozy.get("build", {})
    if build and isinstance(build, dict):
        bld: Dict[str, Any] = {}
        if "gpu" in build and isinstance(build["gpu"], bool):
            bld["gpu"] = build["gpu"]
        if "cuda" in build and isinstance(build["cuda"], str):
            bld["cuda"] = build["cuda"]
        if "torch" in build and isinstance(build["torch"], str):
            bld["torch"] = build["torch"]
        if "backend" in build and isinstance(build["backend"], str):
            bld["backend"] = build["backend"]
        if "base_image" in build and isinstance(build["base_image"], str):
            bld["base_image"] = build["base_image"]
        if bld:
            config["build"] = bld

    # Parse [tool.cozy.models] - deployment key -> Cozy Hub model ID
    models = tool_cozy.get("models", {})
    if models and isinstance(models, dict):
        config["models"] = {str(k): str(v) for k, v in models.items() if k and v}

    # Parse [tool.cozy.resources] - hardware requirements
    resources = tool_cozy.get("resources", {})
    if resources and isinstance(resources, dict):
        res: Dict[str, Any] = {}
        if "vram_gb" in resources and isinstance(resources["vram_gb"], int):
            res["vram_gb"] = resources["vram_gb"]
        if "gpu_type" in resources and isinstance(resources["gpu_type"], str):
            res["gpu_type"] = resources["gpu_type"]
        if "memory_gb" in resources and isinstance(resources["memory_gb"], int):
            res["memory_gb"] = resources["memory_gb"]
        if "cpu_cores" in resources and isinstance(resources["cpu_cores"], int):
            res["cpu_cores"] = resources["cpu_cores"]
        if res:
            config["resources"] = res

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
    Discover functions and load config to build complete manifest.

    Args:
        root: Project root directory. Defaults to current working directory.

    Returns:
        Complete manifest dict with functions, deployment, build, models, and resources.

    The manifest includes:
        - functions: list of discovered worker functions with required_models
        - deployment: default deployment ID from [tool.cozy].deployment
        - models: dict mapping deployment keys to Cozy Hub model IDs
        - build: build settings from [tool.cozy.build]
        - resources: hardware requirements from [tool.cozy.resources]
    """
    if root is None:
        root = Path.cwd()
    root = root.resolve()

    functions = discover_functions(root)
    config = _load_cozy_config(root)

    manifest: Dict[str, Any] = {"functions": functions}

    # Include config sections if present
    if "deployment" in config:
        manifest["deployment"] = config["deployment"]
    if "build" in config:
        manifest["build"] = config["build"]
    if "resources" in config:
        manifest["resources"] = config["resources"]

    # Extract all required model keys from functions (deployment source only)
    all_required_keys: Set[str] = set()
    for fn in functions:
        required = fn.get("required_models", [])
        all_required_keys.update(required)

    # Get models from [tool.cozy.models] config
    config_models: Dict[str, str] = config.get("models", {})

    # Validate: all required model keys must be defined in config
    missing_keys = all_required_keys - set(config_models.keys())
    if missing_keys:
        print(
            f"warning: functions require model keys not defined in [tool.cozy.models]: {sorted(missing_keys)}",
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
