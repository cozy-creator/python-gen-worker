"""Publish-time extraction for the pgw#1382 author surface.

The v1 walk looks for ``__gen_worker_endpoint__``; ``@entrypoint`` stamps
``__cozy_entrypoint__``. Nothing read the new attribute, so a v2 endpoint
discovered ZERO functions, the builder warned and exited 0, and hub admission
refused the empty manifest nine minutes later (pgw#1387).

``entrypoints[]`` IS ``functions[]``' SUCCESSOR SPELLING, NOT A SECOND
DOCUMENT SPACE (th#2146). The hub folds the key into ``Functions`` at its one
decode site (``builder.ParseManifest``) and nothing downstream learns a second
word — so the ITEM shape here must be the item shape ``_extract_entries``
already emits, key for key. Emitting a bespoke row shape (or wrapping the list
in a ``{"v": …}`` envelope) does not merely lose fields: ``entrypoints`` decodes
into ``[]manifestFunction``, so a JSON object where the hub expects an array
fails ``ParseManifest`` outright and the release does not admit.

A manifest carrying BOTH keys is refused by the hub rather than silently
merged, so this module emits ``entrypoints`` only for a package that HAS a v2
surface, and the v1 path is byte-identical when it does not.

**LANES ARE NOT HERE.** A v2 lane is a tensorfs layout contract, and it travels
as ``{stamp, document}`` on the release-derive document's ``lane_contracts``,
where the hub interns the document content-addressed and needs no prior
knowledge of the layout (th#2146's `docs/lane-vocabulary.md`). Making that
``document`` non-null is pgw#964's, not this module's. What DOES belong on a
slot is the ie#740 floor `requires=` declares, in the machinereq term shape
the hub's one parser reads — and because the hub refuses a requirement over a
handle the slot does not accept, the accepted set is emitted beside it.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import textwrap
from typing import Any, Dict, List, Set

from .schema import type_schema_and_hash

#: A v2 entrypoint is an inference handler. The vocabulary is the hub's
#: (`validation._KNOWN_KINDS`); the v2 surface declares no other kind today,
#: and inventing one here would be a value space no hub reader names.
ENTRYPOINT_KIND = "inference"

#: The component path a whole-model lane declaration lands under. The v1 slot
#: vocabulary is per-component (`pipeline`, `pipeline.vae`, …); a v2 `lanes=`
#: is a statement about the WHOLE pipeline, so it declares exactly the root and
#: never invents a component tree the author did not write.
PIPELINE_PATH = "pipeline"


class EntrypointDiscoveryError(ValueError):
    """A v2 endpoint package does not yield an extractable surface."""


def _entrypoint_specs(module: Any) -> List[Any]:
    from ..serving.entrypoints import ENTRYPOINT_ATTR

    out: List[Any] = []
    for value in vars(module).values():
        spec = getattr(value, ENTRYPOINT_ATTR, None)
        if spec is None:
            continue
        # Re-exports: only the DECLARING module owns the row, exactly as the
        # v1 walk skips re-exported @endpoint objects.
        if getattr(value, "__module__", None) != module.__name__:
            continue
        out.append(spec)
    return out


def _pipeline_class(model_cls: type) -> str:
    """The dotted pipeline class ``load()`` builds, read STATICALLY.

    The hub requires `pipeline_class` on a model slot. In v1 it was a `Slot`
    field; in v2 the author writes ``self.pipe = ctx.load(SomePipeline)`` inside
    ``load``, so it is recovered by parsing that call — no author code runs
    beyond the import that already happened, which is the same promise the
    class header makes. Unreadable (a dynamic class, no ``load``) returns "",
    and the caller decides what an absent one means; guessing a pipeline here
    would be a manifest that lies about what the worker will build.
    """
    load = getattr(model_cls, "load", None)
    if load is None:
        return ""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(load)))
    except (OSError, TypeError, SyntaxError):
        return ""
    module = inspect.getmodule(model_cls)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "load"):
            continue
        if not node.args:
            continue
        arg = node.args[0]
        if not isinstance(arg, ast.Name):
            continue
        target = getattr(module, arg.id, None)
        if isinstance(target, type):
            return f"{target.__module__}.{target.__qualname__}"
    return ""


def _lane_stamps(model_cls: type) -> List[str]:
    from ..serving import lane_handle, model_lanes

    return [lane_handle(lane) for lane in model_lanes(model_cls)]


def _model_slot(slot: Any) -> Dict[str, Any]:
    """A model slot in the hub's `functions[].slots[]` vocabulary."""
    from ..serving import model_requires, model_type

    model_cls = slot.annotation
    out: Dict[str, Any] = {
        "name": slot.name,
        # `kind` is omitted rather than spelled "model": empty IS model on the
        # hub side (th#2140 5c) and every pre-5c manifest means exactly that.
        "pipeline_class": _pipeline_class(model_cls),
    }
    declared = model_type(model_cls)
    family = getattr(declared, "name", "") or ""
    if family:
        out["family"] = family
    stamps = _lane_stamps(model_cls)
    if stamps:
        # The accepted layout set. `layout_requirements` below is refused by
        # the hub for a handle absent from here, which is the same rule the
        # class header already enforces on `requires=`.
        out["layouts"] = {PIPELINE_PATH: stamps}
    requirements = {
        stamp: row.manifest_row()
        for stamp, row in model_requires(model_cls).items()
        if row.declared()
    }
    if requirements:
        out["layout_requirements"] = requirements
    return out


def _adapter_slot(slot: Any) -> Dict[str, Any]:
    from ..serving.context import DistillationAdapter

    return {
        "name": slot.name,
        "kind": "adapter",
        # The typed takeover guard as a WIRE fact: the hub refuses an envelope
        # pick whose adapter row is not distillation-marked for this slot.
        "adapter_kind": (
            "distillation" if slot.annotation is DistillationAdapter else "general"
        ),
        "multiple": slot.kind == "adapters",
    }


def _entrypoint_row(spec: Any) -> Dict[str, Any]:
    input_schema, input_sha = type_schema_and_hash(spec.payload_type)
    output_schema, output_sha = type_schema_and_hash(spec.return_type)
    slots: List[Dict[str, Any]] = []
    for slot in spec.slots:
        slots.append(_model_slot(slot) if slot.kind == "model" else _adapter_slot(slot))
    return {
        "name": spec.name,
        "python_name": spec.fn.__name__,
        "module": spec.fn.__module__,
        "declared_module": spec.fn.__module__,
        "class_name": "",
        "kind": ENTRYPOINT_KIND,
        "input_schema": input_schema,
        "payload_schema_sha256": input_sha,
        "output_schema": output_schema,
        "output_schema_sha256": output_sha,
        # A v2 entrypoint returns one struct; streaming is not part of the
        # ratified surface, so the cardinality fact is stated, never omitted.
        "incremental_output": False,
        "slots": slots,
    }


def _pipeline_class_or_refuse(rows: List[Dict[str, Any]]) -> None:
    """The hub hard-fails a model slot with no `pipeline_class`, after the
    build and push. Refuse here instead, naming the author's own line."""
    for row in rows:
        for slot in row["slots"]:
            if slot.get("kind") == "adapter" or slot.get("pipeline_class"):
                continue
            raise EntrypointDiscoveryError(
                f"@entrypoint {row['name']!r} slot {slot['name']!r}: could not "
                "read the pipeline class from the model's load(). Publish "
                "needs it, so write the ONE spelling with a plain class name: "
                "`self.pipe = ctx.load(StableDiffusionXLPipeline)`"
            )


def discover_entrypoints(main_module: str) -> List[Dict[str, Any]]:
    """Walk ``main_module``'s top-level package; return the manifest
    ``entrypoints`` rows — a FLAT LIST in ``functions[]``' item shape.

    An empty list means the package has no v2 surface, which is a legal answer
    for a v1 endpoint and not an error here; the empty-manifest refusal belongs
    to the caller that knows about jobs and v1 functions too
    (:func:`assert_manifest_advertises_something`).
    """
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
    _pipeline_class_or_refuse(rows)
    rows.sort(key=lambda r: str(r["name"]))
    return rows


def assert_manifest_advertises_something(manifest: Dict[str, Any]) -> None:
    """A release that advertises NOTHING is a build failure, not a warning.

    pgw#1387, measured: the builder printed "no functions discovered", exited
    0, spent 9m20s baking, pushed, and only then did hub admission refuse. The
    refusal is correct; discovering it after the build is the defect.
    """
    if (
        (manifest.get("functions") or [])
        or (manifest.get("jobs") or [])
        or (manifest.get("entrypoints") or [])
    ):
        return
    raise EntrypointDiscoveryError(
        "this endpoint advertises NOTHING: discovery found no @entrypoint "
        "functions (pgw#1382), no @endpoint functions and no @job functions. "
        "A release with an empty manifest is refused at hub admission, so the "
        "build stops here rather than after the image bake. Check that the "
        "module named by [tool.gen_worker] main is the one carrying the "
        "declarations, and that its decorators are gen_worker's."
    )


def entrypoints_block(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The manifest value for ``entrypoints``. A FLAT LIST — the hub decodes
    this key into ``[]manifestFunction`` and folds it into ``Functions``."""
    return rows


__all__ = [
    "ENTRYPOINT_KIND",
    "EntrypointDiscoveryError",
    "assert_manifest_advertises_something",
    "discover_entrypoints",
    "entrypoints_block",
]
