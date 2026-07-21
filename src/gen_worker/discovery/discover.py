"""Build-time endpoint discovery: walk the ``[tool.gen_worker].main``
package, extract every ``@endpoint`` object, and emit the endpoint.lock
manifest as TOML on stdout. Run as ``python -m gen_worker.discovery``.
"""

import hashlib
import json
import sys
import traceback
import typing
import types as py_types
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import msgspec

from gen_worker.api.binding import Binding
from gen_worker.api.slot import Slot
from gen_worker.api.types import (
    Asset,
    AudioAsset,
    ExpectedOutput,
    ImageAsset,
    MediaAsset,
    PromptRole,
    Tensors,
    VideoAsset,
)
from gen_worker.discovery.heavy_deps import stub_missing_heavy_deps
from gen_worker.discovery.names import slugify_name
from gen_worker.discovery.project import load_project_config
from gen_worker.discovery.walk import EndpointImportError, find_endpoints


def _type_id(t: type) -> Dict[str, str]:
    """Get module and qualname for a type."""
    return {
        "module": getattr(t, "__module__", ""),
        "qualname": getattr(t, "__qualname__", getattr(t, "__name__", "")),
    }


def _is_msgspec_struct(t: Any) -> bool:
    """Check if type is a msgspec.Struct subclass."""
    try:
        return isinstance(t, type) and issubclass(t, msgspec.Struct)
    except Exception:
        return False


def _media_kind(t: type) -> str:
    if issubclass(t, ImageAsset):
        return "image"
    if issubclass(t, VideoAsset):
        return "video"
    if issubclass(t, AudioAsset):
        return "audio"
    return "media"


def _annotation_carries_asset(ann: Any, _seen: Optional[Set[type]] = None) -> bool:
    """True when the annotation subtree can hold an input ``Asset``."""
    seen = _seen if _seen is not None else set()
    origin = typing.get_origin(ann)
    if origin is typing.Annotated:
        args = typing.get_args(ann)
        return bool(args) and _annotation_carries_asset(args[0], seen)
    if origin in (typing.Union, py_types.UnionType):
        return any(
            _annotation_carries_asset(arg, seen)
            for arg in typing.get_args(ann)
            if arg is not type(None)
        )
    if origin in (list, tuple, set, frozenset):
        args = typing.get_args(ann)
        return bool(args) and _annotation_carries_asset(args[0], seen)
    if origin is dict:
        args = typing.get_args(ann)
        return len(args) == 2 and _annotation_carries_asset(args[1], seen)
    if isinstance(ann, type):
        if issubclass(ann, Asset):
            return True
        if issubclass(ann, Tensors):
            return False
        if _is_msgspec_struct(ann):
            if ann in seen:
                return False
            seen.add(ann)
            try:
                hints = typing.get_type_hints(ann, include_extras=True)
            except Exception:
                hints = getattr(ann, "__annotations__", {})
            return any(
                _annotation_carries_asset(hints[field], seen)
                for field in getattr(ann, "__struct_fields__", ()) or ()
                if field in hints
            )
    return False


def _collect_payload_moderation_metadata(payload_type: type) -> Dict[str, Any]:
    out: Dict[str, list[Dict[str, str]]] = {"prompts": [], "media": []}
    seen_structs: set[type] = set()

    def walk(ann: Any, path: str) -> None:
        origin = typing.get_origin(ann)

        if origin is typing.Annotated:
            args = typing.get_args(ann)
            if not args:
                return
            base = args[0]
            roles = [m for m in args[1:] if isinstance(m, PromptRole)]
            if roles:
                if base is not str:
                    raise ValueError(
                        f"{path}: PromptRole markers must annotate str fields"
                    )
                out["prompts"].append({"field": path, "role": roles[-1].role})
                return
            walk(base, path)
            return

        if origin in (typing.Union, py_types.UnionType):
            for arg in typing.get_args(ann):
                if arg is not type(None):
                    walk(arg, path)
            return

        if origin in (set, frozenset):
            # th#886: input-asset manifests are ordered; unordered containers
            # have no stable occurrence order, so an Asset here is a build error.
            args = typing.get_args(ann)
            if args and _annotation_carries_asset(args[0]):
                raise ValueError(
                    f"{path}: Asset fields cannot ride unordered set/frozenset "
                    "containers; use list or tuple"
                )
            if args:
                walk(args[0], f"{path}[]")
            return

        if origin in (list, tuple):
            args = typing.get_args(ann)
            if args:
                walk(args[0], f"{path}[]")
            return

        if origin is dict:
            args = typing.get_args(ann)
            if len(args) == 2:
                if args[0] is not str and _annotation_carries_asset(args[1]):
                    raise ValueError(
                        f"{path}: Asset-bearing mappings require string keys"
                    )
                walk(args[1], f"{path}.*")
            return

        if isinstance(ann, type):
            if issubclass(ann, Tensors):
                return
            if issubclass(ann, Asset):
                # Base Asset/MediaAsset = kind "media"; typed subclasses carry
                # their exact kind (th#886 manifest vocabulary).
                out["media"].append({"field": path, "kind": _media_kind(ann)})
                return
            if _is_msgspec_struct(ann):
                if ann in seen_structs:
                    return
                seen_structs.add(ann)
                try:
                    hints = typing.get_type_hints(ann, include_extras=True)
                except Exception:
                    hints = getattr(ann, "__annotations__", {})
                for field in getattr(ann, "__struct_fields__", ()) or ():
                    if field in hints:
                        walk(hints[field], f"{path}.{field}" if path else field)
                seen_structs.discard(ann)

    walk(payload_type, "")
    return {k: v for k, v in out.items() if v}


def _unwrap_optional(ann: Any) -> Any:
    origin = typing.get_origin(ann)
    if origin in (typing.Union, py_types.UnionType):
        args = [arg for arg in typing.get_args(ann) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return ann


def _media_kind_for_annotation(ann: Any) -> str:
    ann = _unwrap_optional(ann)
    origin = typing.get_origin(ann)
    if origin in (list, tuple, set, frozenset):
        args = typing.get_args(ann)
        ann = _unwrap_optional(args[0]) if args else Any
    if isinstance(ann, type) and issubclass(ann, ImageAsset):
        return "image"
    if isinstance(ann, type) and issubclass(ann, VideoAsset):
        return "video"
    if isinstance(ann, type) and issubclass(ann, AudioAsset):
        return "audio"
    if isinstance(ann, type) and issubclass(ann, MediaAsset):
        return "file"
    return "other"


def _payload_has_field_path(payload_type: type, ref: str) -> bool:
    if not ref.startswith("input."):
        return True
    path = ref.removeprefix("input.")
    if not path:
        return False

    current: Any = payload_type
    for raw_part in path.replace("[]", "").split("."):
        part = raw_part.strip()
        if not part:
            return False
        current = _unwrap_optional(current)
        origin = typing.get_origin(current)
        if origin in (list, tuple, set, frozenset):
            args = typing.get_args(current)
            current = _unwrap_optional(args[0]) if args else Any
        if not _is_msgspec_struct(current):
            return False
        try:
            hints = typing.get_type_hints(current, include_extras=True)
        except Exception:
            hints = getattr(current, "__annotations__", {}) or {}
        if part not in hints:
            return False
        current = hints[part]
    return True


def _expected_output_expr(value: Any, *, payload_type: type, field: str, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{field}: ExpectedOutput.{key} must be positive")
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.startswith("input.") and not _payload_has_field_path(payload_type, raw):
            raise ValueError(f"{field}: ExpectedOutput.{key} references unknown payload field {raw!r}")
        return raw
    raise TypeError(f"{field}: ExpectedOutput.{key} must be int, str, or None")


def _collect_expected_output_metadata(payload_type: type, output_type: type) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    seen_structs: set[type] = set()

    def walk(ann: Any, path: str) -> None:
        origin = typing.get_origin(ann)

        if origin is typing.Annotated:
            args = typing.get_args(ann)
            if not args:
                return
            base = args[0]
            markers = [m for m in args[1:] if isinstance(m, ExpectedOutput)]
            if markers:
                marker = markers[-1]
                media_type = marker.media_type or _media_kind_for_annotation(base)
                item: Dict[str, Any] = {
                    "field": path,
                    "type": media_type,
                }
                count = _expected_output_expr(marker.count, payload_type=payload_type, field=path, key="count")
                if count is not None:
                    item["count"] = count
                width = _expected_output_expr(marker.width, payload_type=payload_type, field=path, key="width")
                if width is not None:
                    item["width"] = width
                height = _expected_output_expr(marker.height, payload_type=payload_type, field=path, key="height")
                if height is not None:
                    item["height"] = height
                aspect = _expected_output_expr(marker.aspect_ratio, payload_type=payload_type, field=path, key="aspect_ratio")
                if aspect is not None:
                    item["aspect_ratio"] = aspect
                duration = _expected_output_expr(marker.duration_s, payload_type=payload_type, field=path, key="duration_s")
                if duration is not None:
                    item["duration_s"] = duration
                mime = (marker.mime_type or "").strip()
                if mime:
                    item["mime_type"] = mime
                out.append(item)
                return
            walk(base, path)
            return

        if origin in (typing.Union, py_types.UnionType):
            for arg in typing.get_args(ann):
                if arg is not type(None):
                    walk(arg, path)
            return

        if origin in (list, tuple, set, frozenset):
            args = typing.get_args(ann)
            if args:
                walk(args[0], f"{path}[]")
            return

        if isinstance(ann, type) and _is_msgspec_struct(ann):
            if ann in seen_structs:
                return
            seen_structs.add(ann)
            try:
                hints = typing.get_type_hints(ann, include_extras=True)
            except Exception:
                hints = getattr(ann, "__annotations__", {})
            for field in getattr(ann, "__struct_fields__", ()) or ():
                if field in hints:
                    walk(hints[field], f"{path}.{field}" if path else field)
            seen_structs.discard(ann)

    walk(output_type, "")
    return out


def _binding_to_manifest(binding: Binding, param_name: str = "") -> Dict[str, Any]:
    """Emit a ``functions.bindings.<slot>`` block for the manifest.

    Every binding is a fixed pick; the slot name is the dict key. Keys stay
    compatible with ``models.download.build_provider_index_from_manifest``
    (``ref`` / ``provider`` / ``flavor``).
    """
    out: Dict[str, Any] = {
        "kind": "fixed",
        "provider": binding.source,
        "slot_name": param_name,
        "ref": binding.path,
    }
    if binding.source == "tensorhub":
        # Normal form (gw#492): the default tag ('latest') is elided at the
        # manifest boundary so hub-minted keep/routing refs stay byte-equal
        # to worker-minted wire refs (Go folds a non-empty tag verbatim).
        if binding.tag and binding.tag != "latest":
            out["tag"] = binding.tag
        if binding.flavor:
            out["flavor"] = binding.flavor
        if binding.components:
            # pgw#505: the hub's desired-snapshot scoping (platform-side,
            # not yet built) reads this to resolve only the named pipeline
            # component subfolders instead of the whole repo.
            out["components"] = list(binding.components)
    elif binding.source == "huggingface":
        for k in ("revision", "dtype", "subfolder"):
            v = getattr(binding, k)
            if v:
                out[k] = v
        if binding.files:
            out["files"] = list(binding.files)
        if binding.components:
            out["components"] = list(binding.components)
    elif binding.source == "civitai":
        if binding.version:
            out["version"] = binding.version
    elif binding.source == "modelscope":
        if binding.revision:
            out["revision"] = binding.revision
        if binding.files:
            out["files"] = list(binding.files)
    return out


def _stamp_family(binding_manifest: Dict[str, Any], family: str) -> None:
    """Stamp a binding manifest with the endpoint's architecture family
    (pgw#523: unconditional-when-known, not ``allow_lora``-triggered) so
    tensorhub's th#586 gate can family-police any LoRA overlay attached at
    this slot. Identity (the binding) and permission (whether a LoRA may
    attach here — the slot-policy ``loras`` axis, th#772) are separate
    concerns; this only carries the family fact through. Shared by
    top-level ``bindings`` blocks and ``model.choices[].binding`` rows
    (pgw#519) so both surfaces stamp identically. No-op when the family
    isn't known — nothing to police."""
    if not family:
        return
    binding_manifest["family"] = family


def _model_ref_to_manifest(ref: Any) -> Dict[str, Any]:
    """``default_checkpoint``/curated-choice ref shape shared by the slots
    block: ``{source, path, tag?, flavor?, revision?, version?, components?}``
    — a structured ModelRef (pgw#511; ``components`` added pgw#505)."""
    out: Dict[str, Any] = {"source": ref.source, "path": ref.path}
    if ref.tag and ref.tag != "latest":
        out["tag"] = ref.tag
    if ref.flavor:
        out["flavor"] = ref.flavor
    if ref.components:
        out["components"] = list(ref.components)
    if ref.source in ("huggingface", "modelscope") and ref.revision:
        out["revision"] = ref.revision
    if ref.source == "civitai" and ref.version:
        out["version"] = ref.version
    return out


def _slot_to_manifest(name: str, slot: Slot, *, compile_family: str) -> Dict[str, Any]:
    """One ``functions[].slots[]`` entry (pgw#520 / th#767): the hub-side
    mapping/resolution contract for a Slot-declared model slot — NOT a
    model choice list (``model.choices[]`` stays the ModelChoice-only
    surface; a Slot endpoint never emits it)."""
    out: Dict[str, Any] = {
        "name": name,
        "pipeline_class": f"{slot.pipeline_cls.__module__}.{slot.pipeline_cls.__qualname__}",
    }
    if slot.selected_by:
        out["selected_by"] = slot.selected_by
    if slot.default_checkpoint is not None:
        out["default_checkpoint"] = _model_ref_to_manifest(slot.default_checkpoint)
    # Compile(family=...) is the explicit, functionally-load-bearing
    # declaration (compile-cache keying) — it wins over the slot's own
    # default_config-preset registration when both are present, mirroring
    # _stamp_family's precedence for the bindings-block stamp below.
    family = compile_family or slot.family
    if family:
        out["family"] = family
    if slot.default_config is not None:
        out["default_config"] = msgspec.to_builtins(slot.default_config)
    return out


def _model_choice_in(ann: Any) -> Tuple[Optional[type], bool]:
    """Inspect a payload field annotation for a curated ``ModelChoice`` set.

    Returns ``(choice_enum, accepts_byom)``: the ``ModelChoice`` subclass the
    field is typed with (or ``None``), and whether the field ALSO accepts an
    arbitrary client-supplied :class:`ModelRef` (``ModelChoice | ModelRef`` =
    BYOM-open). ``Optional[...]`` (a ``None`` union member) does not imply
    BYOM."""
    from gen_worker.api.binding import ModelRef
    from gen_worker.api.model import is_model_choice

    origin = typing.get_origin(ann)
    if origin in (typing.Union, py_types.UnionType):
        args = typing.get_args(ann)
    else:
        args = (ann,)
    choice: Optional[type] = None
    byom = False
    for arg in args:
        if is_model_choice(arg):
            if choice is not None and choice is not arg:
                raise ValueError(
                    "a payload field may reference only one ModelChoice set"
                )
            choice = arg
        elif arg is ModelRef:
            byom = True
    return choice, byom


def _collect_model_placement_key(
    payload_type: type, models: Dict[str, Any], compile_family: str = "", name: str = ""
) -> Optional[Dict[str, Any]]:
    """The handler's checkpoint placement key (pgw#509), or ``None``.

    Scans the payload's top-level fields for one typed with a ``ModelChoice``
    subclass and emits the curated set — each choice's structured
    :class:`ModelRef` binding, typed per-model defaults, and optional
    ``hot``/``price`` hints — plus whether the field accepts BYOM and which
    ``models=`` slot the pick swaps into. This is the SDK->tensorhub contract
    (th#761): the scheduler warm-pools per ``choices[].binding`` ref; the
    catalog/UI renders ``choices[].defaults``.

    ``compile_family`` mirrors the endpoint's ``Compile(family=...)`` onto
    each choice binding via :func:`_stamp_family` — identically to how
    top-level ``bindings`` blocks are stamped (pgw#519), unconditionally
    when known (pgw#523)."""
    try:
        hints = typing.get_type_hints(payload_type)
    except Exception:
        hints = getattr(payload_type, "__annotations__", {}) or {}

    field_name: Optional[str] = None
    choice_enum: Optional[type] = None
    byom = False
    for name in getattr(payload_type, "__struct_fields__", ()) or ():
        if name not in hints:
            continue
        found, field_byom = _model_choice_in(hints[name])
        if found is None:
            continue
        if choice_enum is not None:
            raise ValueError(
                f"{payload_type.__name__}: multiple ModelChoice fields "
                f"({field_name!r}, {name!r}); a handler has one placement key"
            )
        field_name, choice_enum, byom = name, found, field_byom

    if choice_enum is None or field_name is None:
        return None

    # The pick swaps the primary (first-declared) model slot.
    slot = next(iter(models), "")
    choices: List[Dict[str, Any]] = []
    for row in choice_enum.rows():  # type: ignore[attr-defined]
        binding_manifest = _binding_to_manifest(row.ref, slot)
        _stamp_family(binding_manifest, compile_family)
        entry: Dict[str, Any] = {
            "id": row.id,
            "binding": binding_manifest,
            "defaults": msgspec.to_builtins(row.defaults),
        }
        if row.hot:
            entry["hot"] = True
        if row.price is not None:
            entry["price"] = row.price
        choices.append(entry)
    if not choices:
        raise ValueError(
            f"{payload_type.__name__}.{field_name}: ModelChoice "
            f"{choice_enum.__name__} declares no models"
        )
    block: Dict[str, Any] = {"field": field_name, "byom": byom, "choices": choices}
    if slot:
        block["slot"] = slot
    return block


def _schema_and_hash(t: type) -> Tuple[Dict[str, Any], str]:
    """Generate JSON schema and SHA256 hash for a msgspec type."""
    schema = msgspec.json.schema(t)
    try:
        from gen_worker.api.payload_constraints import apply_schema_constraints

        schema = apply_schema_constraints(schema, t)
    except Exception:
        pass
    raw = json.dumps(schema, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return schema, hashlib.sha256(raw).hexdigest()


def _assert_unique_function_names(functions: List[Dict[str, Any]]) -> None:
    """Fail the build if two functions share a routable name in one endpoint.

    Function names are the endpoint's external routing identifiers
    (``owner/endpoint/<name>``, the wire ``function_name``, the
    ``invoke <name>`` / ``serve --function <name>`` key), so they MUST be
    unique within an endpoint. A collision is an author error — e.g. two
    classes each exposing a generic ``name="generate"`` without an explicit
    override. The worker historically only logged ``Handler name conflict;
    skipping`` and silently dropped one route; fail loudly at
    discovery/endpoint.lock build time instead.
    """
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for fn in functions:
        nm = str(fn.get("name") or "").strip()
        if nm:
            by_name.setdefault(nm, []).append(fn)
    dupes = {nm: fns for nm, fns in by_name.items() if len(fns) > 1}
    if not dupes:
        return
    lines = []
    for nm, fns in sorted(dupes.items()):
        where = ", ".join(
            f"{f.get('class_name') or '<module-level>'} in "
            f"{f.get('module') or f.get('declared_module') or '?'}"
            for f in fns
        )
        lines.append(f"  {nm!r}: defined {len(fns)}x ({where})")
    raise ValueError(
        "duplicate function name(s) within the endpoint — function names are the "
        "external routing identifiers and must be unique. Rename the handler "
        "method:\n" + "\n".join(lines)
    )


def discover_functions(
    root: Optional[Path] = None,
    *,
    main_module: str | None = None,
    extra_heavy_deps: Tuple[str, ...] = (),
) -> List[Dict[str, Any]]:
    """Discover every @endpoint object under ``main_module``'s top-level
    package and return the manifest ``functions`` entries.

    Build-time discovery arms :func:`stub_missing_heavy_deps` around the walk:
    heavy roots (torch, ...) missing from the environment are stubbed so
    module-top ``import torch`` costs nothing, while any code that actually
    USES the dep at import time fails loudly. ``extra_heavy_deps`` extends the
    default allowlist (``[tool.gen_worker] discovery_heavy_deps``).
    """
    if root is None:
        root = Path.cwd()
    root = root.resolve()
    if not main_module:
        raise ValueError(
            "discover_functions requires main_module ([tool.gen_worker].main)"
        )

    root_str = str(root)
    src_str = str(root / "src")
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if (root / "src").exists() and src_str not in sys.path:
        sys.path.insert(0, src_str)

    top_level = main_module.split(".", 1)[0]
    with stub_missing_heavy_deps(extra_heavy_deps):
        try:
            found = find_endpoints([top_level])
        except Exception as e:
            raise ValueError(
                f"failed to walk endpoint package {top_level!r} (derived from "
                f"[tool.gen_worker] main={main_module!r}): {e}"
            ) from e

        functions: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str, str, str]] = set()
        for f in found:
            for entry in _extract_entries(f.obj, f.walked_module):
                # (module, class, python_name, name) dedups objects re-found under
                # multiple walked packages; name is one handler per method now.
                key = (
                    entry.get("declared_module", entry.get("module", "")),
                    entry.get("class_name", ""),
                    entry.get("python_name", ""),
                    entry.get("name", ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                functions.append(entry)

    _assert_unique_function_names(functions)
    return functions


def _extract_entries(obj: Any, module_name: str) -> List[Dict[str, Any]]:
    """Manifest entries for one @endpoint class or function.

    Signature inspection lives in ``gen_worker.registry`` — the one walker
    shared with the worker runtime and the CLI. This adds only the
    manifest-specific enrichment (schemas, moderation, bindings blocks).
    """
    from gen_worker.registry import extract_specs

    out: List[Dict[str, Any]] = []
    for es in extract_specs(obj, walked_module=module_name):
        res_dict: Dict[str, Any] = {}
        try:
            raw = msgspec.to_builtins(es.resources)
            if isinstance(raw, dict):
                res_dict.update(raw)
        except Exception:
            pass
        bindings_block = {
            key: _binding_to_manifest(binding, key)
            for key, binding in es.models.items()
        }
        # Every binding carries the endpoint's architecture family, when
        # known, so the hub's th#586 gate can family-police any LoRA
        # overlay attached at that slot (pgw#523: unconditional-when-known,
        # not allow_lora-triggered). pgw#519: the same stamp applies to
        # model.choices[].binding rows, not just top-level bindings. pgw#520:
        # a Slot-declared binding with no Compile(family=...) may still carry
        # a family via its own default_config preset's registration — that's what
        # es.slot_family reconciles (Compile(family=) wins when both exist).
        compile_family = es.compile.family if es.compile is not None else ""
        for key, block in bindings_block.items():
            slot_family = es.slot_family.get(key, "") if es.slot_family else ""
            _stamp_family(block, compile_family or slot_family)

        slots_block = [
            _slot_to_manifest(name, slot, compile_family=compile_family)
            for name, slot in es.slots.items()
        ]

        input_schema, input_sha = _schema_and_hash(es.payload_type)
        moderation = _collect_payload_moderation_metadata(es.payload_type)
        model_key = _collect_model_placement_key(
            es.payload_type, es.models, compile_family, es.name
        )
        output_type = es.output_type
        if output_type is None:
            raise ValueError(
                f"{es.name}: manifest requires a concrete msgspec.Struct "
                "output/delta type"
            )
        output_schema, output_sha = _schema_and_hash(output_type)
        expected_outputs = _collect_expected_output_metadata(es.payload_type, output_type)
        incremental = es.output_mode == "stream"
        delta_schema = None
        delta_sha = ""
        if incremental and es.delta_type is not None:
            delta_schema, delta_sha = _schema_and_hash(es.delta_type)

        function_name = slugify_name(es.name)
        if not function_name:
            raise ValueError(
                f"{es.name!r}: function name cannot be normalized"
            )

        fn: Dict[str, Any] = {
            "name": function_name,
            "python_name": es.attr_name or es.method.__name__,
            "module": module_name,
            "declared_module": es.module or module_name,
            "class_name": es.cls.__name__ if es.cls is not None else "",
            "kind": es.kind,
            "runtime": es.runtime,
            "resources": res_dict,
            "bindings": bindings_block,
            "payload_type": _type_id(es.payload_type),
            "payload_schema_sha256": input_sha,
            "input_schema": input_schema,
            "moderation": moderation,
            "expected_outputs": expected_outputs,
            "output_mode": "incremental" if incremental else "single",
            "output_type": _type_id(output_type),
            "output_schema_sha256": output_sha,
            "output_schema": output_schema,
            "incremental_output": incremental,
            "is_async": es.is_async,
            "timeout_ms": es.timeout_ms,
        }
        # th#826: the child-call declaration — the hub mints the invoke_child
        # capability grant only for declaring functions. Omitted when false.
        if es.child_calls:
            fn["child_calls"] = True
        if model_key is not None:
            fn["model"] = model_key
        if slots_block:
            fn["slots"] = slots_block
        if incremental and es.delta_type is not None:
            fn["delta_type"] = _type_id(es.delta_type)
            fn["delta_schema_sha256"] = delta_sha
            fn["delta_output_schema"] = delta_schema
        if es.compile is not None:
            # Hub keys family-cache lookups off this block (th#569).
            fn["compile"] = {
                "family": es.compile.family,
                "shapes": [[int(v) for v in s] for s in es.compile.shapes],
                "targets": list(es.compile.targets),
            }
            if es.compile.guidance_scales:
                fn["compile"]["guidance_scales"] = list(
                    es.compile.guidance_scales
                )
            # ie#381: the primary binding's weight-storage lane (gw#389 fp8
            # layerwise casting) rides along so the hub's cell producer
            # builds from an identically-loaded pipeline — the cast hooks
            # are traced INTO the FX graphs; a bf16-built cell for an
            # fp8-served model misses on every request.
            primary = next(iter(es.models.values()), None)
            storage = str(getattr(primary, "storage_dtype", "") or "")
            if storage:
                fn["compile"]["storage_dtype"] = storage
            if getattr(es.compile, "regional", False):
                fn["compile"]["regional"] = True
            # gw#561: dynamic-LoRA endpoints trace the branch-bearing graph
            # family; the hub's producer must build `-lora<bucket>` cells.
            if getattr(es.compile, "lora_bucket", 0):
                fn["compile"]["lora_bucket"] = int(es.compile.lora_bucket)
        out.append(fn)

    return out


def discover_manifest(root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Discover functions and load tensorhub manifest config to build complete manifest.

    Args:
        root: Project root directory. Defaults to current working directory.

    Returns: Complete manifest dict with functions + models/resources metadata.
    """
    if root is None:
        root = Path.cwd()
    root = root.resolve()

    cfg = load_project_config(root)

    functions = discover_functions(
        root, main_module=cfg.main, extra_heavy_deps=cfg.discovery_heavy_deps
    )

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

    manifest: Dict[str, Any] = {
        "functions": functions,
    }
    return manifest


def _strip_none(obj: Any) -> Any:
    """Recursively remove None values from dicts/lists (TOML has no null type)."""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj if v is not None]
    return obj


def main() -> None:
    """Write the build-time endpoint manifest to stdout.

    #328: bake-time validation gate. After ``discover_manifest`` produces
    the ``functions`` list, ``validate_endpoint_lock`` confirms every entry
    is a class-shape declaration (post-#322). An old function-shape entry
    that slipped past the discovery refactor hard-fails the build with a
    pointer to the migration guide.
    """
    try:
        manifest = discover_manifest()
    except Exception as e:
        # A broken endpoint module fails the BUILD, with the real import
        # traceback — never a log-and-continue that ships an endpoint.lock
        # silently missing functions.
        cause: Optional[BaseException] = e
        while cause is not None and not isinstance(cause, EndpointImportError):
            cause = cause.__cause__
        if cause is not None:
            traceback.print_exc(file=sys.stderr)
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    # #328 bake-time validation gate. Old function-shape entries trip the
    # missing-class_name check; same-class slug collisions trip the route
    # uniqueness check; non-class shapes (an entry without archetype/kind)
    # are caught by the required-field check. All errors flow out at once
    # so the build surfaces every problem rather than one-at-a-time.
    from .validation import validate_endpoint_lock

    val = validate_endpoint_lock(manifest)
    for w in val.warnings:
        print(f"warning: {w}", file=sys.stderr)
    if not val.ok:
        for err in val.errors:
            print(f"error: {err}", file=sys.stderr)
        sys.exit(1)

    if not manifest.get("functions"):
        print("warning: no @endpoint objects found", file=sys.stderr)

    sys.stdout.write(msgspec.toml.encode(_strip_none(manifest)).decode("utf-8"))
    if not sys.stdout.isatty():
        sys.stdout.write("\n")
