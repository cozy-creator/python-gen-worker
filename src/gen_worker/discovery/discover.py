"""Build-time endpoint discovery: walk the ``[tool.gen_worker].main``
package, extract every ``@endpoint`` object, and emit the endpoint.lock
manifest as TOML on stdout. Run as ``python -m gen_worker.discovery``.
"""

import hashlib
import inspect
import json
import sys
import traceback
import typing
import types as py_types
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import msgspec

from gen_worker.aot_preconditions import (
    adapter_backend_preconditions,
    declared_compile_families,
    static_mint_preconditions,
)
from gen_worker.api.binding import Binding, wire_ref
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
from gen_worker.discovery.decode_set import derive_decode_set
from gen_worker.discovery.decode_set import (
    manifest_block as decode_set_manifest_block,
)
from gen_worker.discovery.execution_lanes import (
    DerivedExecutionLanes,
    derive_execution_lanes,
    execution_lanes_for_function,
    manifest_block,
)
from gen_worker.discovery.heavy_deps import stub_missing_heavy_deps
from gen_worker.discovery.names import slugify_name
from gen_worker.discovery.project import load_project_config
from gen_worker.discovery.walk import EndpointImportError, find_endpoints
from gen_worker.registry import extract_specs
from .validation import validate_endpoint_lock
import importlib.machinery


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
            # Input-asset manifests are ordered; unordered containers have no
            # stable occurrence order, so an Asset here is a build error.
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
                # their exact kind.
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
    (``ref`` / ``provider``).
    """
    out: Dict[str, Any] = {
        "kind": "fixed",
        "provider": binding.source,
        "slot_name": param_name,
        "ref": binding.path,
    }
    if binding.source == "tensorhub":
        # th#1987: the release rides the REF, in normal form — there is no
        # side-channel key any more. The hub reads it with ParseCanonicalRef,
        # so worker-minted and hub-minted refs stay byte-equal.
        out["ref"] = str(wire_ref(binding))
        if binding.components:
            # The hub's desired-snapshot scoping reads this to resolve only the
            # named pipeline component subfolders instead of the whole repo.
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
    """Stamp a binding manifest with the endpoint's architecture family —
    unconditional-when-known, never gated on a declaration — so the hub can
    family-police any LoRA overlay attached at this slot. Identity (the
    binding) and permission (whether a LoRA may attach here — the slot-policy
    ``loras`` axis) are separate concerns; this only carries the family fact
    through. No-op when the family isn't known — nothing to police."""
    if not family:
        return
    binding_manifest["family"] = family


def _model_ref_to_manifest(ref: Any) -> Dict[str, Any]:
    """``default_checkpoint`` ref shape used by the slots block:
    ``{source, path, revision?, version?, components?}``. The tensorhub
    release rides ``path`` in normal form (th#1987)."""
    out: Dict[str, Any] = {"source": ref.source, "path": ref.path}
    if ref.source == "tensorhub":
        out["path"] = str(wire_ref(ref))
    if ref.components:
        out["components"] = list(ref.components)
    if ref.source in ("huggingface", "modelscope") and ref.revision:
        out["revision"] = ref.revision
    if ref.source == "civitai" and ref.version:
        out["version"] = ref.version
    return out


def _slot_to_manifest(
    name: str, slot: Slot[Any], *, family: str,
    components: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """One ``functions[].slots[]`` entry: the hub-side mapping/resolution
    contract for a Slot-declared model slot.

    Publishes the DERIVED component tree (``components``: ordered
    ``[{name, kind}]`` rows, kind ``weights`` | ``config``) — the path
    vocabulary the hub needs for per-path policy (``pipeline`` open /
    ``pipeline.vae`` curated / ``pipeline.unet`` fixed) and component-level
    routing. Pinned at publish so diffusers drift stays deterministic. There is
    deliberately no ``default_config``: recipe values are catalog data, and the
    schema derives from the handler's ``RequestContext[D]`` annotation."""
    out: Dict[str, Any] = {
        "name": name,
        "pipeline_class": f"{slot.pipeline_cls.__module__}.{slot.pipeline_cls.__qualname__}",
    }
    if slot.selected_by:
        out["selected_by"] = slot.selected_by
    if getattr(slot, "optional", False):
        # The deploy may leave this slot unbound (its setup param has a
        # default) — a release then serves only the lanes it bound.
        # Absent field = required, so existing manifests are unchanged.
        out["optional"] = True
    if slot.default_checkpoint is not None:
        out["default_checkpoint"] = _model_ref_to_manifest(slot.default_checkpoint)
    if family:
        out["family"] = family
    if components:
        out["components"] = [
            {"name": part, "kind": kind}
            for part, kind in sorted(components.items())
        ]
    # The per-component DEMAND. Handles only: the hub resolves each to its
    # descriptor DIGEST at ingest, against its own registry, which is the only
    # moment one wheel and one hub are both pinned. Absence is no longer a
    # state an image can ship — `validate_endpoint_lock` refuses it (A19).
    if slot.layouts:
        out["layouts"] = {
            path: list(handles) for path, handles in slot.layouts.items()
        }
    # The explicit third rung: this slot's bytes have no registered handle,
    # and the REASON travels rather than being lost to an absent key.
    if getattr(slot, "layouts_undeclarable", ""):
        out["layouts_undeclarable"] = slot.layouts_undeclarable
    # The REQUIREMENTS axis, keyed by the handle it guards. Only DECLARED
    # axes are emitted: an undeclared floor must not arrive at the hub as a
    # zero, which `contractspec.DecodeEntry.MinSM` reads as "no floor".
    requirements = {
        handle: row.manifest_row()
        for handle, row in (getattr(slot, "layout_requirements", None) or {}).items()
        if row.declared()
    }
    if requirements:
        out["layout_requirements"] = requirements
    return out


def _schema_and_hash(t: type) -> Tuple[Dict[str, Any], str]:
    """Generate JSON schema and SHA256 hash for a msgspec type."""
    schema = msgspec.json.schema(t)
    raw = json.dumps(schema, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return schema, hashlib.sha256(raw).hexdigest()


def _assert_unique_function_names(functions: List[Dict[str, Any]]) -> None:
    """Fail the build if two functions share a routable name in one endpoint.

    Function names are the endpoint's external routing identifiers
    (``owner/endpoint/<name>``, the wire ``function_name``, the
    ``invoke <name>`` / ``serve --function <name>`` key), so they MUST be
    unique within an endpoint. A collision is an author error — e.g. two
    classes each exposing a generic ``name="generate"`` without an explicit
    override. Fails loudly at discovery/endpoint.lock build time rather than
    silently dropping one route at runtime.
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
        # `declared_module` first, like the dedup key below: `module` is where
        # the WALK found the object, which for a re-exported handler is a
        # package `__init__` — the wrong file to send an author to.
        where = ", ".join(
            f"{f.get('class_name') or '<module-level>'} in "
            f"{f.get('declared_module') or f.get('module') or '?'}"
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
    # The audit below is about what THE WALK imported, so the set it compares
    # against has to be taken before the walk runs. Scanning all of
    # `sys.modules` instead attributes the CALLER's imports to discovery.
    preloaded = frozenset(sys.modules)
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
    _validate_variant_targets(functions)
    _audit_source_only_imports(
        root=root, top_level=top_level, preloaded=preloaded)
    return functions


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


def _validate_variant_targets(functions: List[Dict[str, Any]]) -> None:
    """Every ``@variant_of`` must target another discovered function on this
    endpoint, and the target must not itself be a variant (no chains).
    Build-time gate — a dangling pairing never ships."""
    by_name = {str(f.get("name") or ""): f for f in functions}
    for f in functions:
        target = str(f.get("variant_of") or "")
        if not target:
            continue
        name = str(f.get("name") or "")
        target_fn = by_name.get(target)
        if target_fn is None:
            raise ValueError(
                f"{name!r}: @variant_of targets unknown function {target!r} "
                f"(discovered: {sorted(by_name)})"
            )
        if target_fn.get("variant_of"):
            raise ValueError(
                f"{name!r}: @variant_of target {target!r} is itself a "
                "variant — chains are not allowed"
            )


def _extract_entries(obj: Any, module_name: str) -> List[Dict[str, Any]]:
    """Manifest entries for one @endpoint class or function.

    Signature inspection lives in ``gen_worker.registry`` — the one walker
    shared with the worker runtime and the CLI. This adds only the
    manifest-specific enrichment (schemas, moderation, bindings blocks).
    """

    out: List[Dict[str, Any]] = []
    for es in extract_specs(obj, walked_module=module_name):
        res_dict: Dict[str, Any] = {}
        try:
            # Resources owns its own manifest projection (the one
            # declaration -> wire-name mapping lives there, not here).
            project = getattr(es.resources, "manifest_dict", None)
            raw = project() if callable(project) else msgspec.to_builtins(es.resources)
            if isinstance(raw, dict):
                res_dict.update(raw)
        except Exception:
            pass
        bindings_block = {
            key: _binding_to_manifest(binding, key)
            for key, binding in es.models.items()
        }
        # Every binding carries the endpoint's architecture family, when known,
        # so the hub can family-police any LoRA overlay attached at that slot.
        # For Slot-declared bindings the map is AUTHORITATIVE: it already
        # reconciles the function family with the slot's explicit intent. Bare
        # bindings have no slot declaration and retain the function-level
        # compile family fallback.
        compile_family = es.compile.family if es.compile is not None else ""
        for key, block in bindings_block.items():
            family = (
                es.slot_family.get(key, "")
                if key in es.slots
                else compile_family
            )
            _stamp_family(block, family)

        # `es.slot_family` is AUTHORITATIVE per slot — it already folds in
        # Compile(family=...), and it deliberately holds "" for an auxiliary
        # bare-typed slot that never opted into the family vocabulary.
        # Re-defaulting to `compile_family` here would put the function's family
        # back on exactly those slots, making the hub's gate demand that a
        # family-agnostic artifact classify as that family and fail the whole
        # manifest closed.
        slots_block = [
            _slot_to_manifest(
                name, slot,
                family=es.slot_family.get(name, ""),
                components=es.slot_components.get(name),
            )
            for name, slot in es.slots.items()
        ]

        input_schema, input_sha = _schema_and_hash(es.payload_type)
        moderation = _collect_payload_moderation_metadata(es.payload_type)
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

        # pgw#1332: the families this function binds, so PLACEMENT can prefetch
        # their weights and verify the VRAM fit before a request lands — which
        # is the entire reason static declaration is the default and
        # `ModelSpec.instance(ref)` is the exception. `export_digest` pins the
        # declaration these bindings were generated against, so a pod holding a
        # different one is detectable without loading anything.
        families_block = [
            {
                "parameter": parameter,
                "family": str(getattr(row.model, "FAMILY", "")),
                "export_digest": str(getattr(row.model, "EXPORT_DIGEST", "")),
                # pgw#1346 K3: the endpoint-coupled axis, emitted so PLACEMENT
                # can see which payload field branches this parameter. Omitted
                # when unset, so a model bound without one is byte-identical to
                # what this key produced before `Bind` existed.
                **({"selected_by": row.selected_by} if row.selected_by else {}),
            }
            for parameter, row in sorted(es.families.items())
        ]

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
            "output_type": _type_id(output_type),
            "output_schema_sha256": output_sha,
            "output_schema": output_schema,
            # The output-cardinality fact, in the ONE spelling the hub decodes
            # (builder/manifest_contract.go). pgw#1320 deleted the sibling
            # `output_mode: "incremental"|"single"` key beside it — a third
            # value space for this bit that no hub reader has ever named.
            "incremental_output": incremental,
            "is_async": es.is_async,
        }
        # The child-call declaration — the hub mints the invoke_child
        # capability grant only for declaring functions. Omitted when false.
        if es.child_calls:
            fn["child_calls"] = True
        # Omitted when the function binds none, so an endpoint that declares no
        # family produces the byte-identical manifest it produced before this
        # key existed.
        if families_block:
            fn["families"] = families_block
        # The hub-write declaration (th#2049/pgw#1294), ALWAYS emitted on both
        # row shapes. Not omit-when-false like the flags above: the hub mints
        # a write grant off this, so "absent" must never be readable as
        # "declared false" — absent means a wheel too old to have the concept,
        # and those two need different answers.
        fn["publishes"] = bool(es.publishes)
        # Payload compile axes (equivalence classes) — catalog recipes validate
        # against the declared class names at publish time; the warm plan
        # derives from classes x buckets.
        if es.payload_axes:
            fn["compile_axes"] = [a.to_manifest() for a in es.payload_axes]
        if es.lora_bucket:
            fn["lora_bucket"] = int(es.lora_bucket)
        # The derived config schema (RequestContext[D]) — names the family
        # vocabulary the catalog's recipe values are validated against.
        if es.defaults_type is not None:
            fn["config_schema"] = es.defaults_type.__name__
            fam = str(getattr(es.defaults_type, "__gen_worker_family__", "") or "")
            if fam:
                fn["config_family"] = fam
        # The base<->variant pairing rides into the manifest so the hub's public
        # endpoint info can advertise it.
        if es.variant_of:
            fn["variant_of"] = es.variant_of
            fn["variant"] = es.variant_kind
        # Per-function objective contract: objectives omitted = unrestricted;
        # distilled omitted = either.
        if es.objectives is not None:
            fn["objectives"] = list(es.objectives)
        if es.distilled is not None:
            fn["distilled"] = bool(es.distilled)
        # Declared serving tasks — the axis the hub's lane-verdict store keys
        # on. Omitted = undeclared, and every quant lane then resolves
        # unmeasured (serves at base precision).
        if es.tasks is not None:
            fn["tasks"] = list(es.tasks)
        # The opt-in reference contract. Omitted = this function never sees the
        # concept; the hub refuses a ref_text sent to it.
        if es.accepts_references is not None:
            fn["accepts_references"] = es.accepts_references.to_manifest()
        # Opt-in declared lane bodies (behavioral divergence marker).
        if es.handles:
            fn["handles"] = list(es.handles)
        # Declared config parameters + env names — the hub persists these as the
        # release's declared surface and 422s config writes outside it.
        if es.config:
            fn["config_params"] = [p.to_manifest() for p in es.config]
        if es.env:
            fn["env"] = list(es.env)
        # Declared compute-time formula — the hub learns the constants per
        # physics compiled graph; the source string is the contract.
        if es.runtime_formula is not None:
            fn["runtime_formula"] = es.runtime_formula.source
        if slots_block:
            fn["slots"] = slots_block
        if incremental and es.delta_type is not None:
            fn["delta_type"] = _type_id(es.delta_type)
            fn["delta_schema_sha256"] = delta_sha
            fn["delta_output_schema"] = delta_schema
        ccontract = es.compile_contract()
        if es.compile is not None and ccontract is not None:
            # Hub keys family-cache lookups off this block.
            fn["compile"] = {
                "family": es.compile.family,
                "shapes": [[int(v) for v in s] for s in es.compile.shapes],
                "targets": list(es.compile.targets),
            }
            # Shape contract: the declared text axis and dynamic ranges ride to
            # the hub's compiled graph producer, and the contract digest is the ck2
            # compiled graph-key axis.
            if ccontract.text_len is not None:
                # THIS function's effective pin (a @worker_function
                # text_len= override wins over the class Compile's).
                fn["compile"]["text_len"] = int(ccontract.text_len)
            if ccontract.contract_text_lens():
                # The CLASS's per-lane pin union — what the shared compiled graph contract
                # digests (dual-pin classes describe both lanes).
                fn["compile"]["text_lens"] = [
                    int(v) for v in ccontract.contract_text_lens()
                ]
            if es.compile.dynamic:
                fn["compile"]["dynamic"] = [
                    {"dim": d.dim, "min": d.min, "max": d.max}
                    for d in es.compile.dynamic
                ]
            if ccontract.guidance_scales:
                # Warm representatives derived from the payload's CompileAxis
                # classes.
                fn["compile"]["guidance_scales"] = list(ccontract.guidance_scales)
            fn["compile"]["shape_contract_digest"] = ccontract.contract_digest()
            # The primary binding's weight-storage lane (fp8 layerwise casting)
            # rides along so the hub's compiled graph producer builds from an
            # identically-loaded pipeline — the cast hooks are traced INTO the
            # FX graphs; a bf16-built compiled graph for an fp8-served model misses on
            # every request.
            primary = next(iter(es.models.values()), None)
            storage = str(getattr(primary, "storage_dtype", "") or "")
            if storage:
                fn["compile"]["storage_dtype"] = storage
            if getattr(es.compile, "regional", False):
                fn["compile"]["regional"] = True
            # Dynamic-LoRA endpoints trace the branch-bearing graph family; the
            # hub's producer must build `-lora<bucket>` compiled graphs.
            if es.lora_bucket:
                fn["compile"]["lora_bucket"] = int(es.lora_bucket)
            # The AUTHOR's declared bar and refusals, so the hub's publish-time
            # validation session judges a release against a DECLARATION instead
            # of a number the platform picked. Absent when undeclared — the hub
            # reports `bar_undeclared` by name, and a default emitted here would
            # make the SDK the author of the bar the platform verifies.
            if es.compile.speed_metric:
                fn["compile"]["speed_metric"] = es.compile.speed_metric
            if es.compile.min_speedup is not None:
                fn["compile"]["min_speedup"] = float(es.compile.min_speedup)
            # OPEN blockers only: the hub reads a non-empty list as "the author
            # refuses to mint" and marks the mint check blocked-by-declaration,
            # so a RESOLVED id would park the family in that state forever. Ids,
            # not prose — open-vs-resolved is all the hub decides on.
            open_ids = [b.id for b in es.compile.open_blockers]
            if open_ids:
                fn["compile"]["blockers"] = open_ids
        out.append(fn)

    return out


def _job_source_file(spec: Any, root: Path) -> str:
    """The job's own ``.py`` file, relative to the release root.

    A POINTER, never a copy of the bytes. RECONCILED to th#2049's landed
    correction 6: the release tarball is the source of truth and already
    renders per-file views on read, so captured source text would be a second
    copy that can only drift. This lane emitted the text first; the hub lane
    landed the pointer, and the hub owns the contract.
    """
    path_str = inspect.getsourcefile(spec.method) or ""
    if not path_str:
        raise ValueError(
            f"@job {spec.name!r}: no source file could be located — the catalog "
            "serves a public job's code by this pointer, so it must live in a "
            "real module on disk"
        )
    path = Path(path_str).resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def _job_entry(spec: Any, root: Path) -> Dict[str, Any]:
    """One manifest ``jobs[]`` row.

    ``emits_media`` is jobs-only: the hub mints the ``upload_media`` grant off
    it (th#2069), where an endpoint gets media authority from being an
    endpoint.

    Deliberately the same shape as a function row where the two overlap
    (name/schemas/resources/env/publishes), because a job promoted to a
    serverless endpoint must not change identity on the way. What it does not
    carry is equally deliberate: no execution lanes, no compile block, no
    slots/bindings — a job that wants a hub-resolved model NAMES it in its
    payload.
    """
    res_dict: Dict[str, Any] = {}
    project = getattr(spec.resources, "manifest_dict", None)
    raw = project() if callable(project) else msgspec.to_builtins(spec.resources)
    if isinstance(raw, dict):
        res_dict.update(raw)
    input_schema, input_sha = _schema_and_hash(spec.payload_type)
    output_schema, output_sha = _schema_and_hash(spec.output_type)
    return {
        "name": spec.name,
        "python_name": spec.python_name,
        "module": spec.module,
        "resources": res_dict,
        "env": list(spec.env),
        "resumable": bool(spec.resumable),
        "visibility": spec.visibility,
        "publishes": bool(spec.publishes),
        "emits_media": bool(spec.emits_media),
        "payload_type": _type_id(spec.payload_type),
        "payload_schema_sha256": input_sha,
        "input_schema": input_schema,
        "output_type": _type_id(spec.output_type),
        "output_schema_sha256": output_sha,
        "output_schema": output_schema,
        "is_async": bool(spec.is_async),
        "source_file": _job_source_file(spec, root),
        **(
            {
                "families": [
                    {
                        "parameter": parameter,
                        "family": str(getattr(row.model, "FAMILY", "")),
                        "export_digest": str(getattr(row.model, "EXPORT_DIGEST", "")),
                        **({"selected_by": row.selected_by} if row.selected_by else {}),
                    }
                    for parameter, row in sorted(spec.families.items())
                ]
            }
            if spec.families
            else {}
        ),
    }


def discover_jobs(
    root: Path,
    *,
    main_module: str,
    extra_heavy_deps: Tuple[str, ...] = (),
) -> List[Dict[str, Any]]:
    """Every ``@job`` under ``main_module``'s top-level package, as manifest rows.

    Same heavy-dep stubbing as the endpoint walk, so a torch-less manifest
    build derives the same set an in-image build does. Sorted by name: the
    manifest is a published artifact and must be byte-stable across runs.
    """
    from ..registry import collect_jobs

    top_level = main_module.split(".", 1)[0]
    with stub_missing_heavy_deps(extra_heavy_deps):
        try:
            specs = collect_jobs([top_level])
        except Exception as e:
            raise ValueError(
                f"failed to walk job package {top_level!r} (derived from "
                f"[tool.gen_worker] main={main_module!r}): {e}"
            ) from e
    return [_job_entry(spec, root) for spec in sorted(specs, key=lambda s: s.name)]


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

    # The endpoint half of the lane intersection, DERIVED from the decoders this
    # image actually carries — never a hand-maintained list. Runs inside the
    # same heavy-dep stubbing the endpoint walk used, so a torch-less manifest
    # build derives the same set an in-image build does.
    with stub_missing_heavy_deps(cfg.discovery_heavy_deps) as stubbed:
        # ONE import walk: the decode-set is what the image can READ
        # (th#1938's third intersection) and the lane block is what it can
        # RUN. Two renders of one census, never two censuses.
        decode_set = derive_decode_set()
        derived = derive_execution_lanes(decode_set=decode_set)
        # The AOT lane's STATIC preconditions, decided in the image that will
        # run them. `validate_endpoint_lock` turns a refusal into a build error,
        # so an endpoint that declares an export it cannot compile never reaches
        # a pod to downgrade there.
        declared_families = declared_compile_families(functions)
        preconditions = static_mint_preconditions(
            declared_families, torch_available="torch" not in stubbed)
        # And the ADAPTER capability, which is not an AOT question — an endpoint
        # declaring `lora_bucket > 0` serves adapters whether or not it
        # compiles, and `peft` is honest under the heavy-dep stubbing (it is
        # deliberately not a stubbed root), so this decides here too.
        preconditions = preconditions + adapter_backend_preconditions(
            declared_families)
    for fn in functions:
        bucket = int((fn.get("compile") or {}).get("lora_bucket") or 0)
        lanes, exclusions = execution_lanes_for_function(
            derived, lora_bucket=bucket
        )
        fn["execution_lanes"] = list(lanes)
        if exclusions:
            fn["execution_lane_exclusions"] = [
                {"execution_lane": e.execution_lane, "reason": e.reason}
                for e in exclusions
            ]
        unbacked = _census_unbacked_layouts(fn, derived)
        if unbacked:
            fn["layouts_census_unbacked"] = list(unbacked)

    manifest: Dict[str, Any] = {
        "functions": functions,
        "execution_lanes": manifest_block(derived),
        "decode_set": decode_set_manifest_block(decode_set),
    }
    # The jobs block sits BESIDE functions, never inside it: one package may
    # carry both, publish once, submit as needed. A release with jobs and zero
    # functions is legal (th#2049).
    jobs = discover_jobs(
        root, main_module=cfg.main, extra_heavy_deps=cfg.discovery_heavy_deps)
    if jobs:
        manifest["jobs"] = jobs
    if preconditions:
        manifest["aot_preconditions"] = [
            row.manifest_row() for row in preconditions]
    return manifest


def _census_unbacked_layouts(
    fn: Dict[str, Any], derived: DerivedExecutionLanes,
) -> List[str]:
    """Declared handles no `@implements_contract` decoder in this image backs.

    **NOT a refusal**, deliberately: the census is a LOWER-bound sanity check,
    and a lower bound that refuses is an upper bound. A layout decoded natively
    by `transformers` via `quantization_config` carries zero cozy markers, so
    refusing here would make that legal case illegal. It lands on the manifest
    and in the build log so an author can see the gap and judge it.
    """
    backed = {c.contract for c in derived.contracts}
    unbacked: List[str] = []
    for slot in fn.get("slots") or []:
        for handles in (slot.get("layouts") or {}).values():
            for handle in handles:
                if handle not in backed and handle not in unbacked:
                    unbacked.append(handle)
    return sorted(unbacked)


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


def main() -> None:
    """Write the build-time endpoint manifest to stdout.

    Bake-time validation gate: after ``discover_manifest`` produces the
    ``functions`` list, ``validate_endpoint_lock`` confirms every entry is a
    class-shape declaration.

    Every refusal below is about the endpoint SOURCE, which no retry can
    change, so each one carries ``BUILD_INPUT_FAILURE_MARKER``. A death that
    is NOT about the source — the process OOM-killed, the build host cut off —
    prints nothing and stays retryable, which is the correct answer for it.
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
        _fail_build_input(f"error: {e}")

    # All errors flow out at once so the build surfaces every problem rather
    # than one-at-a-time.

    val = validate_endpoint_lock(manifest)
    for w in val.warnings:
        print(f"warning: {w}", file=sys.stderr)
    if not val.ok:
        _fail_build_input(*(f"error: {err}" for err in val.errors))

    if not manifest.get("functions") and not manifest.get("jobs"):
        print("warning: no @endpoint or @job objects found", file=sys.stderr)

    sys.stdout.write(msgspec.toml.encode(_strip_none(manifest)).decode("utf-8"))
    if not sys.stdout.isatty():
        sys.stdout.write("\n")
