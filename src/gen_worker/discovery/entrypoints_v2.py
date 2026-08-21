from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import textwrap
from typing import Any, Dict, List, Set

from .expected_outputs import expected_outputs
from .moderation import payload_moderation
from .schema import type_schema_and_hash

ENTRYPOINT_KIND = "inference"

class EntrypointDiscoveryError(ValueError):
    """A v2 endpoint package does not yield an extractable surface."""


def _entrypoint_specs(module: Any) -> List[Any]:
    from ..serving.entrypoints import ENTRYPOINT_ATTR

    out: List[Any] = []
    for value in vars(module).values():
        spec = getattr(value, ENTRYPOINT_ATTR, None)
        if spec is None:
            continue
        if getattr(value, "__module__", None) != module.__name__:
            continue
        out.append(spec)
    return out


def _import_sites(tree: ast.AST) -> Dict[str, str]:
    found: Dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if node.level:
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            found[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return found


def _pipeline_class(model_cls: type) -> str:
    load = getattr(model_cls, "load", None)
    if load is None:
        return ""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(load)))
    except (OSError, TypeError, SyntaxError):
        return ""
    module = inspect.getmodule(model_cls)
    static: Dict[str, str] = {}
    try:
        module_source = inspect.getsource(module) if module is not None else ""
    except (OSError, TypeError):
        module_source = ""
    if module_source:
        try:
            static.update(_import_sites(ast.parse(module_source)))
        except SyntaxError:
            pass
    static.update(_import_sites(tree))
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
        dotted = static.get(arg.id)
        if dotted:
            return dotted
    return ""


def _engine_runtime(model_cls: type) -> str:
    from ..serving.engine_runtime import ENGINE_SPEC_RUNTIMES, EngineSpec

    load = getattr(model_cls, "load", None)
    if load is None:
        return ""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(load)))
    except (OSError, TypeError, SyntaxError):
        return ""
    module = inspect.getmodule(model_cls)
    static: Dict[str, str] = {}
    try:
        module_source = inspect.getsource(module) if module is not None else ""
    except (OSError, TypeError):
        module_source = ""
    if module_source:
        try:
            static.update(_import_sites(ast.parse(module_source)))
        except SyntaxError:
            pass
    static.update(_import_sites(tree))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "engine"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Call):
            continue
        spec_fn = node.args[0].func
        if not isinstance(spec_fn, ast.Name):
            continue
        target = getattr(module, spec_fn.id, None)
        if isinstance(target, type) and issubclass(target, EngineSpec):
            runtime = str(getattr(target, "runtime", "") or "")
            if runtime:
                return runtime
        dotted = static.get(spec_fn.id, "")
        if dotted.startswith("gen_worker."):
            runtime = ENGINE_SPEC_RUNTIMES.get(dotted.rsplit(".", 1)[-1], "")
            if runtime:
                return runtime
    return ""


def lift_engine_runtimes(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The ``engine_runtimes`` census: every external engine binary this image will BOOT, and which declaration asked for it."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        for slot in row.get("slots", []):
            runtime = slot.pop("engine_runtime", "") or ""
            model_class = slot.pop("model_class", "") or ""
            if not runtime:
                continue
            out.append({
                "entrypoint": row["name"],
                "slot": slot["name"],
                "model_class": model_class,
                "runtime": runtime,
            })
    out.sort(key=lambda r: (str(r["entrypoint"]), str(r["slot"])))
    return out


def _model_slot(slot: Any) -> Dict[str, Any]:
    from ..serving import model_type

    from ..serving.model import SELF_LOADING_ATTR

    model_cls = slot.annotation
    out: Dict[str, Any] = {
        "name": slot.name,
    }
    self_loading = str(getattr(model_cls, SELF_LOADING_ATTR, "") or "").strip()
    readable = _pipeline_class(model_cls)
    if self_loading:
        if readable:
            raise EntrypointDiscoveryError(
                f"{model_cls.__qualname__}: declares self_loading= "
                f"({self_loading!r}) AND its load() calls "
                f"ctx.load({readable.rsplit('.', 1)[-1]}) — those contradict. "
                "self_loading= is for a pipeline ctx.load CANNOT drive; drop "
                "the marker if it can, or drop the ctx.load call if it cannot."
            )
        out["self_loading"] = self_loading
    else:
        out["pipeline_class"] = readable
    declared = model_type(model_cls)
    family = getattr(declared, "name", "") or ""
    if family:
        out["family"] = family
    runtime = _engine_runtime(model_cls)
    if runtime:
        out["engine_runtime"] = runtime
        out["model_class"] = f"{model_cls.__module__}.{model_cls.__qualname__}"
    return out


def _stricter(term: str, left: Any, right: Any) -> Any:
    from ..models.tensor_layout_contract import term_meets

    return left if term_meets(term, left, right) else right


def _merge_floors(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for row in rows:
        for term, value in row.items():
            if term == "recommended":
                continue
            merged[term] = _stricter(term, value, merged[term]) if term in merged else value
    recommended: Dict[str, Any] = {}
    for row in rows:
        for term, value in (row.get("recommended") or {}).items():
            recommended[term] = (
                _stricter(term, value, recommended[term]) if term in recommended else value
            )
    if recommended:
        merged["recommended"] = recommended
    return merged


def _resources(specs: List[Any]) -> Dict[str, Any] | None:
    from ..serving import model_requires

    floors: List[Dict[str, Any]] = []
    has_model_slot = False
    has_declaration = False
    declared: Dict[str, Any] = {}
    for spec in specs:
        resources = getattr(spec, "resources", None)
        if resources is not None:
            has_declaration = True
            block = resources.manifest_dict()
            row = block.pop("requires", None)
            if row:
                floors.append(row)
            declared.update(block)
        for slot in spec.slots:
            if slot.kind != "model":
                continue
            has_model_slot = True
            floors.extend(
                row.manifest_row()
                for row in model_requires(slot.annotation).values()
                if row.declared()
            )
    if not has_model_slot and not has_declaration:
        return None
    out: Dict[str, Any] = dict(declared)
    if has_model_slot:
        out["gpu"] = True
    merged = _merge_floors(floors)
    if merged:
        out["requires"] = merged
    return out


def _adapter_slot(slot: Any) -> Dict[str, Any]:
    from ..serving.context import DistillationAdapter

    return {
        "name": slot.name,
        "kind": "adapter",
        "adapter_kind": (
            "distillation" if slot.annotation is DistillationAdapter else "general"
        ),
        "multiple": slot.kind == "adapters",
    }


def _is_job_kind(kind: str) -> bool:
    return str(kind or "").strip().lower() not in ("", ENTRYPOINT_KIND)


def _entrypoint_row(spec: Any) -> Dict[str, Any]:
    resources = _resources([spec])
    row_kind = getattr(spec, "kind", "") or ENTRYPOINT_KIND
    input_schema, input_sha = type_schema_and_hash(spec.payload_type)
    output_schema, output_sha = type_schema_and_hash(spec.return_type)
    delta_type = getattr(spec, "delta_type", None)
    delta_schema, delta_sha = (
        type_schema_and_hash(delta_type) if delta_type is not None else (None, "")
    )
    moderation = payload_moderation(spec.payload_type)
    outputs = expected_outputs(spec.payload_type, spec.return_type)
    slots: List[Dict[str, Any]] = []
    for slot in spec.slots:
        slots.append(_model_slot(slot) if slot.kind == "model" else _adapter_slot(slot))
    return {
        "name": spec.name,
        "python_name": spec.fn.__name__,
        "module": spec.fn.__module__,
        "declared_module": spec.fn.__module__,
        "class_name": "",
        "kind": row_kind,
        "input_schema": input_schema,
        "payload_schema_sha256": input_sha,
        "output_schema": output_schema,
        "output_schema_sha256": output_sha,
        "incremental_output": bool(delta_schema is not None),
        **(
            {
                "delta_output_schema": delta_schema,
                "delta_output_schema_sha256": delta_sha,
            }
            if delta_schema is not None
            else {}
        ),
        **({"moderation": moderation} if moderation else {}),
        **({"expected_outputs": outputs} if outputs else {}),
        "slots": slots,
        **({"resources": resources} if resources is not None else {}),
        **({"publishes": True} if spec.publishes else {}),
        **({"env": list(spec.env)} if spec.env else {}),
        **({"child_calls": True} if spec.child_calls else {}),
        **({"handles": list(spec.handles)} if spec.handles else {}),
        **(
            {"emits_media": bool(spec.emits_media)}
            if spec.emits_media is not None and _is_job_kind(row_kind)
            else {}
        ),
    }


def _pipeline_class_or_refuse(rows: List[Dict[str, Any]]) -> None:
    for row in rows:
        for slot in row["slots"]:
            if slot.get("kind") == "adapter" or slot.get("pipeline_class"):
                continue
            if slot.get("self_loading"):
                continue
            runtime = slot.get("engine_runtime") or ""
            if runtime:
                raise EntrypointDiscoveryError(
                    f"@entrypoint {row['name']!r} slot {slot['name']!r} is "
                    f"ENGINE-HOSTED ({runtime}): its load() boots an external "
                    "engine, so there is no Python pipeline class to name. "
                    "Declare it — class YourModel(Model[X], lanes=(), "
                    f'self_loading="served by {runtime}; ctx.load drives no '
                    'part of it") — rather than inventing a class name to get '
                    "past this."
                )
            raise EntrypointDiscoveryError(
                f"@entrypoint {row['name']!r} slot {slot['name']!r}: could not "
                "read the pipeline class from the model's load(). Publish "
                "needs it, so write the ONE spelling with a plain class name: "
                "`self.pipe = ctx.load(StableDiffusionXLPipeline)`"
            )


def discover_entrypoints(main_module: str) -> List[Dict[str, Any]]:
    """Walk ``main_module``'s top-level package; return the manifest ``entrypoints`` rows — a FLAT LIST in ``functions[]``' item shape."""
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
    """A release that advertises NOTHING is a build failure, not a warning."""
    if manifest.get("entrypoints") or []:
        return
    raise EntrypointDiscoveryError(
        "this endpoint advertises NOTHING: discovery found no @entrypoint "
        "functions. A release with an empty manifest is refused at hub "
        "admission, so the build stops here rather than after the image "
        "bake. Check that the module named by [tool.gen_worker] main is the "
        "one carrying the declarations, and that its decorator is "
        "gen_worker's @entrypoint (pgw#1373: @endpoint/@job are deleted)."
    )


def entrypoints_block(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The manifest value for ``entrypoints``."""
    return rows


__all__ = [
    "ENTRYPOINT_KIND",
    "EntrypointDiscoveryError",
    "assert_manifest_advertises_something",
    "discover_entrypoints",
    "lift_engine_runtimes",
    "entrypoints_block",
]
