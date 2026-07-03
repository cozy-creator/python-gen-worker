"""gen-worker prefetch — download an endpoint's model weights into the CAS
WITHOUT instantiating pipelines or touching the GPU.

It enumerates every endpoint class's model bindings (HFRepo / CivitaiRepo /
ModelScopeRepo / Repo) via the same discovery path as ``run``/``serve``, then
resolves + downloads each ref through the shared downloaders into the local
cache. With ``--all-variants`` it also walks Dispatch tables (every variant);
by default pure-Dispatch bindings are skipped (they need a payload to pick a
branch, which prefetch doesn't have). No ``setup()``, no torch device.

Honors ``--offline`` and ``TENSORHUB_CACHE_DIR`` (via the shared resolver).

Exit codes mirror the other subcommands:
  0  all refs resolved/cached
  1  endpoint import failure
  2  CLI usage / config error
  3  one or more refs failed to resolve (e.g. offline cache miss)
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from typing import Any, Dict, Tuple

from ..api.binding import Dispatch, Repo


def add_subparser(sub: "argparse._SubParsersAction[Any]") -> None:
    p = sub.add_parser(
        "prefetch",
        help="Download an endpoint's model weights into the cache (no GPU, no setup).",
        description=(
            "Resolve and download every model binding an endpoint's classes declare "
            "into the local cache, so the first invoke doesn't stall on a multi-GB "
            "download. Does not instantiate pipelines or load the GPU."
        ),
    )
    p.add_argument("--config", help="Path to endpoint.toml (default: ./endpoint.toml or cwd).")
    p.add_argument(
        "--all-variants", action="store_true",
        help="Also download every entry in Dispatch tables, not just static bindings.",
    )
    p.add_argument(
        "--offline", action="store_true",
        help="Use only the local cache; fail (exit 3) on a miss instead of downloading.",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Emit NDJSON progress events on stdout instead of human-readable lines on stderr.",
    )
    p.set_defaults(_handler=_handle_prefetch)


def _handle_prefetch(args: argparse.Namespace) -> int:
    # Reuse run's discovery + resolution/download machinery (same package).
    from .run import (
        _collect_class_methods,
        _ensure_sys_path,
        _load_endpoint_toml_main,
        _resolve_binding_to_ref,
        _resolve_local_path,
    )

    if args.json:
        def emit(ev: Dict[str, Any]) -> None:
            sys.stdout.write(json.dumps(ev) + "\n")
            sys.stdout.flush()
    else:
        def emit(ev: Dict[str, Any]) -> None:
            kind = ev.get("kind", "")
            if kind == "model_fetch.started":
                sys.stderr.write(f"  fetching {ev.get('ref')} ...\n")
            elif kind == "model_fetch.completed":
                sys.stderr.write(f"  cached {ev.get('ref')} -> {ev.get('local_dir')}\n")
            sys.stderr.flush()

    try:
        root, main_module = _load_endpoint_toml_main(args.config)
    except Exception as e:
        sys.stderr.write(f"prefetch: {e}\n")
        return 2
    _ensure_sys_path(root)
    try:
        mod = importlib.import_module(main_module)
    except Exception as e:
        sys.stderr.write(f"prefetch: failed to import endpoint module {main_module!r}: {e}\n")
        return 1

    candidates = _collect_class_methods(mod)

    # Collect unique (ref, provider) -> (ref, provider, allow_patterns) jobs so a
    # binding shared across functions/classes is downloaded once.
    jobs: Dict[Tuple[str, str], Tuple[str, str, tuple]] = {}
    skipped_dispatch = 0
    for c in candidates:
        for param_name, binding in c.bindings.items():
            try:
                if isinstance(binding, Repo):
                    ref, provider = _resolve_binding_to_ref(
                        param_name=param_name, binding=binding, payload=None, overrides={})
                    ap = tuple(getattr(binding, "_allow_patterns", ()) or ())
                    jobs[(ref, provider)] = (ref, provider, ap)
                elif isinstance(binding, Dispatch):
                    if not args.all_variants:
                        skipped_dispatch += 1
                        continue
                    for key, pick in binding.table.items():
                        payload = types.SimpleNamespace(**{binding.field: key})
                        ref, provider = _resolve_binding_to_ref(
                            param_name=param_name, binding=binding, payload=payload, overrides={})
                        ap = tuple(getattr(pick, "_allow_patterns", ()) or ())
                        jobs[(ref, provider)] = (ref, provider, ap)
            except Exception as e:
                sys.stderr.write(f"prefetch: skipping binding {param_name!r}: {e}\n")

    if not jobs:
        msg = "prefetch: no model bindings to fetch"
        if skipped_dispatch:
            msg += f" ({skipped_dispatch} dispatch binding(s) skipped — pass --all-variants to include them)"
        sys.stderr.write(msg + "\n")
        return 0

    emit({"kind": "prefetch.plan", "count": len(jobs)})
    if not args.json:
        sys.stderr.write(f"prefetch: {len(jobs)} model ref(s) to resolve\n")

    failures = 0
    for ref, provider, ap in jobs.values():
        try:
            path = _resolve_local_path(
                ref=ref, provider=provider, offline=args.offline, emit=emit, allow_patterns=ap)
            emit({"kind": "prefetch.ref.ready", "ref": ref, "provider": provider, "local": path})
        except Exception as e:  # _ModelResolutionError + any provider import/IO error
            failures += 1
            emit({"kind": "prefetch.ref.failed", "ref": ref, "provider": provider, "error": str(e)})
            if not args.json:
                sys.stderr.write(f"prefetch: FAILED {ref} ({provider}): {e}\n")

    if skipped_dispatch and not args.all_variants:
        sys.stderr.write(
            f"prefetch: skipped {skipped_dispatch} dispatch binding(s); pass --all-variants to fetch them too\n")

    emit({"kind": "prefetch.done", "fetched": len(jobs) - failures, "failed": failures})
    if failures:
        sys.stderr.write(f"prefetch: {failures} ref(s) failed to resolve\n")
        return 3
    if not args.json:
        sys.stderr.write(f"prefetch: done — {len(jobs)} ref(s) in cache\n")
    return 0
