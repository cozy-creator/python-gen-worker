"""Build-time endpoint discovery: walk the ``[tool.gen_worker].main``
package, extract every ``@entrypoint`` declaration, and emit the endpoint.lock
manifest as TOML on stdout. Run as ``python -m gen_worker.discovery``.

**ONE declaration block (pgw#1373).** The v1 ``@endpoint``/``@job`` walk and
the ``functions[]``/``jobs[]`` blocks it emitted are DELETED — Paul's SDK
hardcut, 2026-08-18. ``entrypoints[]`` is the only block this build writes,
and the hub decodes it at its one site (``builder.ParseManifest`` folds it
into ``Functions``; ``manifestFunctionRows`` reads it for the raw-document
rewrites). A manifest carrying BOTH keys was always a hub refusal, so the
successor spelling travels ALONE rather than beside an empty predecessor.
"""

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
    """The bake imported a module the installed package won't have.

    ``discover_functions`` injects ``root`` and ``root/src`` into ``sys.path``
    so discovery also works in an uninstalled source tree. In a BUILT image
    that injection is a trap: a module the wheel forgot to package still
    imports at bake time from the source tree, the gate passes, and the worker
    then dies at boot with ``ModuleNotFoundError`` on every pod the release
    staffs — untyped, pre-Hello, fleet-wide. So the gate runs the runtime's
    predicate: when the walked project is INSTALLED, everything the walk
    imported must resolve without the source tree.
    """


def _audit_source_only_imports(
    *, root: Path, top_level: str, preloaded: FrozenSet[str] = frozenset(),
) -> None:
    """Fail the bake when the walk leaned on source-tree-only modules.

    Applies only when the walked project is INSTALLED (its top-level package
    resolves with the injected ``root``/``root/src`` entries removed) — an
    uninstalled dev tree keeps working, since there the source tree IS the
    module set. In installed mode, every top-level module that was imported
    from under ``root`` must also resolve from the installed environment;
    ``cwd`` is deliberately not honoured (the worker's import set must not
    depend on the directory it happens to start in).

    ``preloaded`` is ``sys.modules`` as it stood BEFORE the walk, and the scan
    considers only what the walk ADDED — otherwise the audit reports on its
    CALLER's imports (a test runner's own ``conftest`` and test modules sit
    under ``root``, are absent from the wheel, and are imported by no pod).
    Excluding by already-loaded rather than by name keeps this free of any
    knowledge of the test runner.
    """

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
            # PathFinder directly: importlib.util.find_spec consults
            # sys.modules first, which holds the source-tree import we are
            # trying to look past.
            spec = importlib.machinery.PathFinder.find_spec(name, clean_path)
        except Exception:
            return False
        origin = getattr(spec, "origin", None) if spec is not None else None
        if spec is None:
            return False
        # A spec that itself points back into the source tree (e.g. via a
        # lingering .pth or cwd artifact) does not count as installed.
        if isinstance(origin, str) and _under_root(origin):
            return False
        return True

    if not _resolves_installed(top_level):
        return  # dev flow: project not installed; the source tree is the truth

    offenders: List[str] = []
    for name, mod in list(sys.modules.items()):
        if name in preloaded:
            continue  # the caller's import, not the walk's
        if "." in name:
            continue  # submodules resolve with their package
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
    """Recursively remove None values from dicts/lists (TOML has no null type)."""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj if v is not None]
    return obj



#: Tensorhub classifies a build failure carrying this marker as
#: ``ErrBuildInput`` → ``river.JobCancel``: the same source bytes are never
#: retried. Emitted here rather than by the build wrapper because discovery is
#: what knows the failure is about immutable endpoint source — so it says so on
#: both the synthesized-Dockerfile and bring-your-own-Dockerfile paths.
BUILD_INPUT_FAILURE_MARKER = "TENSORHUB_BUILD_INPUT_FAILURE:discovery"


def _fail_build_input(*messages: str) -> None:
    """Print the refusal, mark it non-retryable, and exit nonzero."""
    for message in messages:
        print(message, file=sys.stderr)
    print(BUILD_INPUT_FAILURE_MARKER, file=sys.stderr)
    sys.exit(1)


def prime_sys_path(root: Path) -> None:
    """Put an endpoint tree on ``sys.path`` so its own imports resolve.

    ``<root>`` then ``<root>/src``, each only when absent, front-inserted so a
    same-named installed package cannot shadow the tree being read. Discovery
    and the release derive both read an UNINSTALLED source tree, so both need
    this and both must do it identically — a derive that primed the path
    differently from the discovery that produced the manifest would derive
    against a different module than the one the image ships.

    pgw#1440: this is the one implementation. It was inlined here and copied
    into the v1 `cli/run.py` as `_ensure_sys_path`, whose own docstring said it
    existed to "match discover.py's sys.path priming" — a duplicate that named
    its own original. pgw#1373 (`cd46c957`) deleted the v1 SDK and with it that
    copy, leaving `cli/release.py` importing a symbol that no longer existed,
    so `gen-worker release derive` raised `ModuleNotFoundError` on the second
    line of its handler. Two copies of one rule is what let a deletion break a
    caller silently; there is now one, and it is exported.

    ROOT THEN SRC, in that order, because each insert goes to position 0 — so
    the resulting precedence is ``src`` ahead of ``root``, which is what a
    ``src/``-layout endpoint needs. Reversing the two statements silently
    reverses the precedence, so the order here is load-bearing, not stylistic.
    """
    root = Path(root)
    root_text, src = str(root), root / "src"
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    src_text = str(src)
    if src.exists() and src_text not in sys.path:
        sys.path.insert(0, src_text)



def discover_manifest(root: Optional[Path] = None) -> Dict[str, Any]:
    """The complete endpoint.lock manifest for the project at ``root``.

    ``entrypoints[]`` + the DERIVED censuses this image can prove about itself
    (``execution_lanes``, ``decode_set``, and ``engine_runtimes`` when
    anything hosts an external engine). No hand-maintained lists, and no
    second declaration vocabulary.
    """
    if root is None:
        root = Path.cwd()
    root = root.resolve()

    cfg = load_project_config(root)

    # Discovery also runs in an uninstalled source tree, so the tree is on the
    # path — and `_audit_source_only_imports` below is what stops that
    # injection from hiding a module the built wheel forgot to package.
    prime_sys_path(root)
    top_level = cfg.main.split(".", 1)[0]
    preloaded = frozenset(sys.modules)

    # ONE import walk, inside the heavy-dep stubbing, so a torch-less manifest
    # build derives exactly what an in-image build does. The decode-set is what
    # the image can READ; the lane block is what it can RUN; the entrypoints
    # are what it ADVERTISES.
    with stub_missing_heavy_deps(cfg.discovery_heavy_deps):
        entrypoints = discover_entrypoints(cfg.main)
        decode_set = derive_decode_set()
        derived = derive_execution_lanes(decode_set=decode_set)

    _audit_source_only_imports(root=root, top_level=top_level, preloaded=preloaded)

    for row in entrypoints:
        row["execution_lanes"] = list(execution_lanes_for_function(derived))

    # pgw#1421: the THIRD derived census — which external engine binaries
    # this image's models boot. Lifted off the slots (which is where the
    # static read lands them) so `entrypoints[]` keeps exactly the fields the
    # hub decodes. Omitted entirely when nothing hosts an engine, so a lock
    # carrying this block IS an engine-hosted endpoint.
    engine_runtimes = lift_engine_runtimes(entrypoints)

    manifest: Dict[str, Any] = {
        "entrypoints": entrypoints_block(entrypoints),
        "execution_lanes": manifest_block(derived),
        "decode_set": decode_set_manifest_block(decode_set),
        **({"engine_runtimes": engine_runtimes} if engine_runtimes else {}),
    }
    # A manifest that advertises nothing is a BUILD failure, never a warning
    # the bake outlives (pgw#1387).
    assert_manifest_advertises_something(manifest)
    return manifest


def main() -> None:
    """Write the build-time endpoint manifest to stdout.

    Every refusal below is about the endpoint SOURCE, which no retry can
    change, so each one carries ``BUILD_INPUT_FAILURE_MARKER``. A death that
    is NOT about the source — the process OOM-killed, the build host cut off —
    prints nothing and stays retryable, which is the correct answer for it.
    """
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
