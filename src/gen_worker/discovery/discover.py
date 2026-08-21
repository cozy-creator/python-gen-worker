"""Build-time endpoint discovery: walk the ``[tool.gen_worker].main`` package, extract every ``@entrypoint`` declaration, and emit the endpoint.lock manifest as TOML on stdout."""

from __future__ import annotations

import importlib.machinery
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

import msgspec

from gen_worker.discovery.decode_set import derive_decode_set
from gen_worker.discovery.decode_set import (
    manifest_block as decode_set_manifest_block,
)
from gen_worker.discovery.entrypoints_v2 import (
    EntrypointDiscoveryError,
    assert_manifest_advertises_something,
    discover_entrypoints,
    entrypoints_block,
    lift_engine_runtimes,
)
from gen_worker.discovery.execution_lanes import (
    derive_execution_lanes,
    execution_lanes_for_function,
    manifest_block,
)
from gen_worker.discovery.heavy_deps import stub_missing_heavy_deps
from gen_worker.discovery.project import load_project_config
from .validation import validate_endpoint_lock


class SourceOnlyModuleError(ValueError):
    """The bake imported a module the installed package won't have."""


def _audit_source_only_imports(
    *, root: Path, top_level: str, preloaded: FrozenSet[str] = frozenset(),
) -> None:

    root_str = str(root)
    src_str = str(root / "src")

    def _under_root(filename: str) -> bool:
        return filename.startswith(src_str + "/") or filename.startswith(root_str + "/")

    clean_path = [
        p for p in sys.path
        if p not in ("", ".", root_str, src_str)
    ]

    def _resolves_installed(name: str) -> bool:
        try:
            spec = importlib.machinery.PathFinder.find_spec(name, clean_path)
        except Exception:
            return False
        origin = getattr(spec, "origin", None) if spec is not None else None
        if spec is None:
            return False
        if isinstance(origin, str) and _under_root(origin):
            return False
        return True

    if not _resolves_installed(top_level):
        return

    offenders: List[str] = []
    for name, mod in list(sys.modules.items()):
        if name in preloaded:
            continue
        if "." in name:
            continue
        filename = getattr(mod, "__file__", None)
        if not isinstance(filename, str) or not _under_root(filename):
            continue
        if _resolves_installed(name):
            continue
        offenders.append(f"{name} (imported from {filename})")

    if offenders:
        raise SourceOnlyModuleError(
            "discovery imported module(s) that exist ONLY in the source tree, "
            "but the project is installed — the runtime worker will not have "
            "them and every pod of this release will die at boot with "
            "ModuleNotFoundError (pgw#833):\n  "
            + "\n  ".join(sorted(offenders))
            + "\nFix: DROP THE IMPORT if the module is not runtime code — a "
            "test, a conftest, a dev script and anything else a pod never "
            "imports must not be reachable from the endpoint's import graph, "
            "and must NEVER be packaged to satisfy this gate. Only if the "
            "module genuinely runs on the pod, add it to the built package "
            "(e.g. hatch [tool.hatch.build.targets.wheel] only-include / "
            "py-modules)."
        )




def _strip_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj if v is not None]
    return obj



BUILD_INPUT_FAILURE_MARKER = "TENSORHUB_BUILD_INPUT_FAILURE:discovery"


def _fail_build_input(*messages: str) -> None:
    for message in messages:
        print(message, file=sys.stderr)
    print(BUILD_INPUT_FAILURE_MARKER, file=sys.stderr)
    sys.exit(1)


def prime_sys_path(root: Path) -> None:
    """Put an endpoint tree on sys.path so its own imports resolve: <root> then <root>/src, each only when absent, front-inserted. The statement order is load-bearing: each insert goes to position 0, so effective precedence is src AHEAD of root — reversing the two statements silently reverses the precedence."""
    root = Path(root)
    root_text, src = str(root), root / "src"
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    src_text = str(src)
    if src.exists() and src_text not in sys.path:
        sys.path.insert(0, src_text)



def discover_manifest(root: Optional[Path] = None) -> Dict[str, Any]:
    """The complete endpoint.lock manifest for the project at ``root``."""
    if root is None:
        root = Path.cwd()
    root = root.resolve()

    cfg = load_project_config(root)

    prime_sys_path(root)
    top_level = cfg.main.split(".", 1)[0]
    preloaded = frozenset(sys.modules)

    with stub_missing_heavy_deps(cfg.discovery_heavy_deps):
        entrypoints = discover_entrypoints(cfg.main)
        decode_set = derive_decode_set()
        derived = derive_execution_lanes(decode_set=decode_set)

    _audit_source_only_imports(root=root, top_level=top_level, preloaded=preloaded)

    for row in entrypoints:
        row["execution_lanes"] = list(execution_lanes_for_function(derived))

    engine_runtimes = lift_engine_runtimes(entrypoints)

    manifest: Dict[str, Any] = {
        "entrypoints": entrypoints_block(entrypoints),
        "execution_lanes": manifest_block(derived),
        "decode_set": decode_set_manifest_block(decode_set),
        **({"engine_runtimes": engine_runtimes} if engine_runtimes else {}),
    }
    assert_manifest_advertises_something(manifest)
    return manifest


def main() -> None:
    """Write the build-time endpoint manifest to stdout."""
    try:
        manifest = discover_manifest()
    except Exception as e:
        cause: Optional[BaseException] = e
        while cause is not None and not isinstance(cause, EntrypointDiscoveryError):
            cause = cause.__cause__
        if cause is not None:
            traceback.print_exc(file=sys.stderr)
        _fail_build_input(f"error: {e}")

    val = validate_endpoint_lock(manifest)
    for w in val.warnings:
        print(f"warning: {w}", file=sys.stderr)
    if not val.ok:
        _fail_build_input(*(f"error: {err}" for err in val.errors))

    sys.stdout.write(msgspec.toml.encode(_strip_none(manifest)).decode("utf-8"))
    if not sys.stdout.isatty():
        sys.stdout.write("\n")
