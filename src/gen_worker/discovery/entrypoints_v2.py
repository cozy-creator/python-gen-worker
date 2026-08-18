"""Publish-time extraction for the pgw#1382 author surface.

The v1 walk looks for ``__gen_worker_endpoint__``; ``@entrypoint`` stamps
``__cozy_entrypoint__`` and ``Model[MT]`` stamps its class header. Nothing
read the new attributes, so a v2 endpoint discovered ZERO functions, the
builder warned and exited 0, and hub admission refused a manifest that
declared neither ``functions[]`` nor ``jobs[]`` nine minutes later (pgw#1387).

This module is that missing half. It emits the NEW manifest shape — an
``entrypoints[]`` block whose rows carry the payload/return schemas, the
ordered slots with their kinds, and each referenced model's LANES as tensorfs
contract stamps with their ie#740 placement floors. It deliberately does NOT
map onto the retired ``(execution_lane, artifact_contract, decoder,
key_topologies)`` quant vocabulary: that shape describes the pre-pgw#1382
world, and translating into it would make the new surface lie in the old
words. The hub reads the new rows (th#2140/th#2133 already read release
metadata).

WIRE FIELD NAMES ARE A CROSS-REPO CONTRACT. Everything below is what the hub
side must key on; changing a name here is a two-repo change.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, List, Set

from .schema import type_schema_and_hash

#: One manifest block, one version. Bumped only when a row's MEANING changes;
#: additive fields do not bump it (lenient decode, pgw#1376's evolution rule).
ENTRYPOINTS_BLOCK_VERSION = 1


class EntrypointDiscoveryError(ValueError):
    """A v2 endpoint package does not yield an extractable surface."""


def _entrypoint_specs(module: Any) -> List[Any]:
    from ..serving.entrypoints import ENTRYPOINT_ATTR

    out: List[Any] = []
    for name, value in vars(module).items():
        spec = getattr(value, ENTRYPOINT_ATTR, None)
        if spec is None:
            continue
        # Re-exports: only the DECLARING module owns the row, exactly as the
        # v1 walk skips re-exported @endpoint objects.
        if getattr(value, "__module__", None) != module.__name__:
            continue
        out.append(spec)
    return out


def _lane_rows(model_cls: type) -> List[Dict[str, Any]]:
    """One row per declared lane: the tensorfs contract stamp plus the
    ie#740 floor the class header declared for it, when it declared one."""
    from ..serving import lane_handle, model_lanes, model_requires

    requires = model_requires(model_cls)
    rows: List[Dict[str, Any]] = []
    for lane in model_lanes(model_cls):
        stamp = lane_handle(lane)
        row: Dict[str, Any] = {"contract": stamp}
        floor = requires.get(stamp)
        if floor is not None:
            row["requires"] = floor.render()
        rows.append(row)
    return rows


def _model_row(model_cls: type) -> Dict[str, Any]:
    from ..serving import model_type

    declared = model_type(model_cls)
    lanes = _lane_rows(model_cls)
    return {
        "class": model_cls.__name__,
        "module": model_cls.__module__,
        "model_type": getattr(declared, "name", None) or declared.__name__,
        "lanes": lanes,
        # `lanes=()` is the author STATING eager-permanent, and after Paul's
        # F3 narrowing (2026-08-20) that tier means external-binary runtimes
        # only. It is a declaration, never an inference about the model.
        "eager_permanent": not lanes,
    }


def _slot_row(slot: Any) -> Dict[str, Any]:
    from ..serving.context import DistillationAdapter

    row: Dict[str, Any] = {
        "name": slot.name,
        "kind": slot.kind,
        "required": bool(slot.required),
    }
    if slot.kind == "model":
        row["model_class"] = slot.annotation.__name__
    else:
        # The typed takeover guard is a WIRE fact: the hub refuses an envelope
        # pick whose adapter row is not distillation-marked for this slot.
        row["distillation"] = slot.annotation is DistillationAdapter
    return row


def _entrypoint_row(spec: Any) -> Dict[str, Any]:
    payload_schema, payload_hash = type_schema_and_hash(spec.payload_type)
    return_schema, return_hash = type_schema_and_hash(spec.return_type)
    return {
        "name": spec.name,
        "module": spec.fn.__module__,
        "python_name": spec.fn.__name__,
        "payload_schema": payload_schema,
        "payload_schema_hash": payload_hash,
        "return_schema": return_schema,
        "return_schema_hash": return_hash,
        "slots": [_slot_row(slot) for slot in spec.slots],
        "models": [_model_row(cls) for cls in spec.model_classes],
    }


def discover_entrypoints(main_module: str) -> List[Dict[str, Any]]:
    """Walk ``main_module``'s top-level package; return the manifest
    ``entrypoints`` rows. Empty list = this package has no v2 surface, which
    is a legal answer for a v1 endpoint and NOT an error here — the
    empty-manifest refusal belongs to the caller that knows about jobs and v1
    functions too (:func:`assert_manifest_advertises_something`)."""
    top_level = main_module.split(".", 1)[0]
    try:
        top = importlib.import_module(top_level)
    except Exception as exc:
        raise EntrypointDiscoveryError(
            f"could not import endpoint package {top_level!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    modules: List[Any] = [top]
    if hasattr(top, "__path__"):
        for sub in pkgutil.walk_packages(top.__path__, prefix=top.__name__ + "."):
            try:
                modules.append(importlib.import_module(sub.name))
            except Exception as exc:
                # Same rule as the v1 walk (pgw#689): a module that fails to
                # import silently drops entrypoints from the release.
                raise EntrypointDiscoveryError(
                    f"could not import endpoint submodule {sub.name!r} "
                    f"(walking {top_level!r}): {type(exc).__name__}: {exc}"
                ) from exc

    rows: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for module in modules:
        for spec in _entrypoint_specs(module):
            row = _entrypoint_row(spec)
            key = f"{row['module']}.{row['python_name']}"
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

    names: Dict[str, str] = {}
    for row in rows:
        prior = names.get(row["name"])
        if prior is not None:
            raise EntrypointDiscoveryError(
                f"two entrypoints both named {row['name']!r}: {prior} and "
                f"{row['module']}.{row['python_name']} — the request envelope "
                "routes by this name, so one would silently shadow the other"
            )
        names[row["name"]] = f"{row['module']}.{row['python_name']}"
    rows.sort(key=lambda r: r["name"])
    return rows


def entrypoints_block(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"v": ENTRYPOINTS_BLOCK_VERSION, "entrypoints": rows}


def assert_manifest_advertises_something(manifest: Dict[str, Any]) -> None:
    """A release that advertises NOTHING is a build failure, not a warning.

    pgw#1387, measured: the builder printed "no functions discovered", exited
    0, spent 9m20s baking, pushed, and only then did hub admission refuse with
    "manifest declares neither functions[] nor jobs[]". The refusal is
    correct; discovering it after the build is the defect. An endpoint that
    advertises nothing cannot serve anything, so there is no shape of release
    for which shipping it is the right answer.
    """
    functions = manifest.get("functions") or []
    jobs = manifest.get("jobs") or []
    entrypoints = (manifest.get("entrypoints") or {}).get("entrypoints") or []
    if functions or jobs or entrypoints:
        return
    raise EntrypointDiscoveryError(
        "this endpoint advertises NOTHING: discovery found no @entrypoint "
        "functions (pgw#1382), no @endpoint functions and no @job functions. "
        "A release with an empty manifest is refused at hub admission, so the "
        "build stops here rather than after the image bake. Check that the "
        "module named by [tool.gen_worker] main is the one carrying the "
        "declarations, and that its decorators are gen_worker's."
    )


__all__ = [
    "ENTRYPOINTS_BLOCK_VERSION",
    "EntrypointDiscoveryError",
    "assert_manifest_advertises_something",
    "discover_entrypoints",
    "entrypoints_block",
]
