"""``gen-worker describe`` — machine-readable endpoint introspection.

Emit a stable JSON document describing the endpoint WITHOUT loading any model:
its kind, classes, and every routable function with its input JSON Schema,
output type, generator-ness, and model bindings. Plus ``protocol_version`` +
``capabilities`` so a host (cozy-local) integrates against a versioned contract
instead of scraping ``--help`` or booting ``serve`` just to list functions.

``serve --list-functions --json`` is a thin alias of the ``functions`` array
built here (one shared builder, one shape).

The full design lives in ``progress.json`` issue #349.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

import msgspec

from . import run as run_mod
from .protocol import CAPABILITIES, PROTOCOL_VERSION, gen_worker_version


# --------------------------------------------------------------------------
# argparse wiring
# --------------------------------------------------------------------------

def add_subparser(sub: argparse._SubParsersAction[Any]) -> None:
    """Register the ``describe`` subcommand on the top-level parser."""
    p = sub.add_parser(
        "describe",
        help="Print a machine-readable description of the endpoint (no model load).",
        description=(
            "Introspect the endpoint and emit a stable JSON document: "
            "protocol_version, capabilities, kind, classes, and every routable "
            "function with its input JSON Schema, output type, generator flag, "
            "and model bindings. No model is loaded. This is the contract a host "
            "(cozy-local) integrates against."
        ),
    )
    p.add_argument(
        "--config", dest="config_path", default=None,
        help="Path to endpoint.toml (defaults to ./endpoint.toml).",
    )
    p.add_argument(
        "--module", dest="module", default=None,
        help="Python module path to import (overrides endpoint.toml `main`).",
    )
    p.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Emit JSON (the default and only format; accepted for explicitness).",
    )
    p.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print the JSON with newlines + 2-space indent.",
    )
    p.set_defaults(_handler=_handle_describe)


# --------------------------------------------------------------------------
# Reusable builders (shared with serve --list-functions --json)
# --------------------------------------------------------------------------

def describe_binding(binding: Any) -> Dict[str, Any]:
    """Introspect one model binding into a JSON-able dict (no model load).

    Handles the typed provider repos (Repo / HFRepo / CivitaiRepo) and
    Dispatch (a payload-field-keyed table of repos). Falls back to just the
    type name for anything unrecognized.
    """
    from ..api.binding import CivitaiRepo, Dispatch, HFRepo, Repo

    if isinstance(binding, Dispatch):
        table = getattr(binding, "table", {}) or {}
        return {
            "type": "Dispatch",
            "field": getattr(binding, "field", None),
            "table": {str(k): describe_binding(v) for k, v in table.items()},
        }
    if isinstance(binding, (HFRepo, CivitaiRepo, Repo)):
        out: Dict[str, Any] = {
            "type": type(binding).__name__,
            "provider": getattr(binding, "provider", None),
            "ref": getattr(binding, "ref", None),
            "tag": getattr(binding, "_tag", None),
            "flavor": getattr(binding, "_flavor", None) or None,
            "allow_override": bool(getattr(binding, "_allow_override", False)),
        }
        patterns = tuple(getattr(binding, "_allow_patterns", ()) or ())
        if patterns:
            out["allow_patterns"] = list(patterns)
        version_id = str(getattr(binding, "_version_id", "") or "")
        if version_id:
            out["version_id"] = version_id
        return out
    return {"type": type(binding).__name__}


def _function_input_schema(payload_type: Optional[type]) -> Dict[str, Any]:
    """JSON Schema for a function's payload Struct.

    msgspec emits a top-level ``{"$ref": "#/$defs/X", "$defs": {...}}``. We
    inline that top-level ref so ``input_schema["properties"]`` is directly
    available to a host (cozy builds field prompts off it), while keeping
    ``$defs`` for any NESTED struct references.
    """
    if payload_type is None:
        return {}
    try:
        s = msgspec.json.schema(payload_type)
    except Exception:
        return {}
    ref = s.get("$ref")
    defs = s.get("$defs")
    if isinstance(ref, str) and ref.startswith("#/$defs/") and isinstance(defs, dict):
        key = ref.split("/")[-1]
        target = defs.get(key)
        if isinstance(target, dict):
            merged = dict(target)
            rest = {k: v for k, v in defs.items() if k != key}
            if rest:
                merged["$defs"] = rest
            return merged
    return s


def function_entries(
    candidates: List["run_mod._SelectedFunction"],
) -> List[Dict[str, Any]]:
    """Build the stable ``functions`` array from discovered candidates.

    Shared by ``describe`` and ``serve --list-functions --json`` so both emit
    the identical per-function shape.
    """
    out: List[Dict[str, Any]] = []
    for c in sorted(candidates, key=lambda c: c.fn_name):
        output_type = getattr(c, "output_type", None)
        out.append({
            "name": c.fn_name,
            "class": getattr(c.cls, "__name__", "?"),
            "method": getattr(c, "attr_name", None),
            "kind": getattr(c, "kind", ""),
            "is_generator": bool(getattr(c, "is_generator", False)),
            "input_schema": _function_input_schema(getattr(c, "payload_type", None)),
            "output": getattr(output_type, "__name__", None) if output_type else None,
            "models": {
                param: describe_binding(binding)
                for param, binding in (getattr(c, "bindings", {}) or {}).items()
            },
        })
    return out


def build_description(
    *,
    main_module: str,
    candidates: List["run_mod._SelectedFunction"],
) -> Dict[str, Any]:
    """Assemble the full ``describe`` document (no model load)."""
    classes = sorted({getattr(c.cls, "__name__", "?") for c in candidates})
    kinds = sorted({str(getattr(c, "kind", "") or "") for c in candidates if getattr(c, "kind", "")})
    return {
        "protocol_version": PROTOCOL_VERSION,
        "gen_worker_version": gen_worker_version(),
        "capabilities": list(CAPABILITIES),
        "endpoint": {
            "main_module": main_module,
            "kind": kinds[0] if len(kinds) == 1 else kinds,
            "classes": classes,
        },
        "functions": function_entries(candidates),
    }


# --------------------------------------------------------------------------
# Handler
# --------------------------------------------------------------------------

def _handle_describe(args: argparse.Namespace) -> int:
    try:
        root, mod = run_mod.load_endpoint_module(
            config_path=args.config_path, module=args.module,
        )
        candidates = run_mod.discover_candidates(mod)
    except run_mod._UsageError as e:
        sys.stderr.write(f"gen-worker describe: {e}\n")
        return run_mod.EXIT_USAGE

    doc = build_description(
        main_module=getattr(mod, "__name__", args.module or "?"),
        candidates=candidates,
    )
    if args.pretty:
        sys.stdout.write(json.dumps(doc, indent=2, default=str) + "\n")
    else:
        sys.stdout.write(json.dumps(doc, separators=(",", ":"), default=str) + "\n")
    sys.stdout.flush()
    return run_mod.EXIT_OK
