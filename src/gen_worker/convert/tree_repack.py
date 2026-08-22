"""The tree-repack ENGINE and its registry — headers and names, never weights.

``apply_tree_repack`` rewrites a produced flat tree IN PLACE into the diffusers
component shape a declaration describes. It is a conversion-job leg (se#840's
ruling, pgw#1670): it runs on the pod, after the dtype cast, over the tree the
cast just wrote, so the 50 GB read is paid once and one producer emits the
published artifact.

Three properties are the whole point and each is covered by a case:

1. **No model is instantiated.** The engine reads safetensors HEADERS and
   copies tensor data as byte RANGES. There is no torch import in this module.
2. **Tensor bytes are unchanged.** A renamed tensor is the same bytes under a
   different name; a moved file is the same file. The engine never re-encodes,
   so a repack cannot change a dtype, a shape or a value.
3. **Members are PRESERVED, and the produced layout is STATED.** The repack
   neither shards nor de-shards: N source weight files become N members of the
   component they route to. What the tree ended up as is then read back off
   the tree (``observed_file_layout``) and returned, rather than echoed from
   the request — pgw#1669 is the record of what echoing the request costs.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from ..models.file_layout import observed_file_layout
from ..models.safetensors_header import header_len_ok
from .tree_repack_spec import (
    ComponentConfig,
    ConfigField,
    DeclarationError,
    FileRoute,
    RepackComponent,
    TreeRepack,
    TreeRepackError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "RepackReport",
    "apply_tree_repack",
    "register_tree_repack",
    "registered_tree_repacks",
    "require_tree_repack",
    "tree_repack",
]

_COPY_CHUNK = 8 * 1024 * 1024

_lock = RLock()
_registry: dict[str, TreeRepack] = {}
_builtins_loaded = False


# --------------------------------------------------------------------- registry


def _ensure_builtins() -> None:
    """Import the platform's own declarations once, lazily.

    They live in their own module so that reading "what families can be
    repacked" is reading a declaration table rather than tracing imports.
    """

    global _builtins_loaded
    with _lock:
        if _builtins_loaded:
            return
        _builtins_loaded = True
    from . import tree_repack_families  # noqa: F401  (registers on import)


def register_tree_repack(spec: TreeRepack, *, replace: bool = False) -> TreeRepack:
    """Register one family's tree-repack declaration."""

    with _lock:
        existing = _registry.get(spec.name)
        if existing is not None and existing != spec and not replace:
            raise DeclarationError(
                f"tree repack {spec.name!r} is already registered with a different "
                "declaration; pass replace=True only if you own both")
        _registry[spec.name] = spec
        return spec


def registered_tree_repacks() -> tuple[str, ...]:
    _ensure_builtins()
    with _lock:
        return tuple(sorted(_registry))


def tree_repack(name: str | None) -> TreeRepack | None:
    _ensure_builtins()
    with _lock:
        return _registry.get(str(name or "").strip().lower())


def require_tree_repack(name: str | None) -> TreeRepack:
    """The declaration, or a typed refusal that NAMES what is declared.

    An unknown family is refused rather than sniffed. A repack that guessed
    which family a tree belongs to would be the pgw#740 wrong-converter defect
    with a bigger blast radius: the wrong key map produces a tree that loads
    and serves noise.
    """

    spec = tree_repack(name)
    if spec is None:
        known = ", ".join(registered_tree_repacks()) or "<none>"
        raise TreeRepackError(
            f"no tree-repack declaration named {str(name or '').strip()!r}. "
            f"Declared: {known}. A repack is NAMED by the request, never detected — "
            "register one with gen_worker.convert.register_tree_repack()."
        )
    return spec


# ----------------------------------------------------------------- the report


@dataclass(frozen=True)
class RepackReport:
    """What the repack produced, read off the produced tree."""

    name: str
    file_layout: str
    members: Mapping[str, int]
    tensor_count: int
    moved_files: int
    rewritten_files: int

    def as_attrs(self) -> dict[str, str]:
        """Checkpoint attributes — the produced shape, stated not implied."""

        return {
            "tree_repack": self.name,
            "file_layout": self.file_layout,
            "repack_members": ",".join(
                f"{comp}:{n}" for comp, n in sorted(self.members.items())),
            "repack_tensor_count": str(self.tensor_count),
        }


# ------------------------------------------------------------- safetensors I/O


def _read_header(path: Path) -> tuple[dict[str, Any], int]:
    """``(header, data_start)`` from a safetensors file — 8 bytes plus a JSON blob."""

    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise TreeRepackError(f"{path.name} is not a safetensors file (truncated header)")
        (length,) = struct.unpack("<Q", raw)
        if not header_len_ok(length):
            raise TreeRepackError(
                f"{path.name} declares a {length}-byte safetensors header, which is past the "
                "sanctioned bound — a downloaded tree states this number and this process "
                "would allocate it")
        blob = f.read(length)
        if len(blob) != length:
            raise TreeRepackError(f"{path.name} is not a safetensors file (short header)")
    try:
        header = json.loads(blob)
    except Exception as exc:  # noqa: BLE001
        raise TreeRepackError(f"{path.name} has an unreadable safetensors header: {exc}") from exc
    if not isinstance(header, dict):
        raise TreeRepackError(f"{path.name} has a non-object safetensors header")
    return header, 8 + length


def _tensor_entries(header: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Header rows in ON-DISK order, ``__metadata__`` excluded."""

    rows = [
        (str(name), dict(value))
        for name, value in header.items()
        if name != "__metadata__" and isinstance(value, dict) and "data_offsets" in value
    ]
    rows.sort(key=lambda row: int(row[1]["data_offsets"][0]))
    return rows


def _write_subset(src: Path, dst: Path, mapping: Mapping[str, str]) -> int:
    """Write ``dst`` carrying ``mapping``'s tensors out of ``src``, byte for byte.

    Only the header is rebuilt. Each tensor's payload is copied as the byte
    range the source header points at, in source order, so a repacked tensor is
    bit-identical to the one it came from and no dtype or shape can move here.
    """

    header, data_start = _read_header(src)
    metadata = header.get("__metadata__")

    out_header: dict[str, Any] = {}
    plan: list[tuple[int, int]] = []
    offset = 0
    for name, info in _tensor_entries(header):
        if name not in mapping:
            continue
        start, end = int(info["data_offsets"][0]), int(info["data_offsets"][1])
        length = end - start
        out_header[mapping[name]] = {
            "dtype": info["dtype"],
            "shape": list(info["shape"]),
            "data_offsets": [offset, offset + length],
        }
        plan.append((data_start + start, length))
        offset += length
    if isinstance(metadata, dict):
        out_header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

    blob = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(struct.pack("<Q", len(blob)))
        fout.write(blob)
        for start, length in plan:
            fin.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fin.read(min(_COPY_CHUNK, remaining))
                if not chunk:
                    raise TreeRepackError(
                        f"{src.name} ended inside a declared tensor range — the file is "
                        "shorter than its own header says")
                fout.write(chunk)
                remaining -= len(chunk)
    return offset


def _root_weight_members(root: Path) -> list[Path]:
    """The root's safetensors files, in index order when an index names them."""

    ordered: list[Path] = []
    indexed: set[str] = set()
    for index_path in sorted(root.glob("*.safetensors.index.json")):
        try:
            weight_map = json.loads(index_path.read_text("utf-8")).get("weight_map") or {}
        except Exception:  # noqa: BLE001
            continue
        for name in dict.fromkeys(str(v) for v in weight_map.values()):
            indexed.add(name)
            member = root / name
            if member.is_file() and member not in ordered:
                ordered.append(member)
    for member in sorted(root.glob("*.safetensors")):
        if member.is_file() and member.name not in indexed and member not in ordered:
            ordered.append(member)
    return ordered


# --------------------------------------------------------------- JSON helpers


def _dotted_get(doc: Mapping[str, Any], path: str) -> Any:
    node: Any = doc
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _dotted_set(doc: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = doc
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _apply_fields(
    source_doc: Mapping[str, Any], fields: tuple[ConfigField, ...], *, where: str,
    into: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = dict(into or {})
    for f in fields:
        if not f.source:
            _dotted_set(out, f.target, f.value)
            continue
        value = _dotted_get(source_doc, f.source)
        if value is None:
            if f.required:
                raise TreeRepackError(
                    f"{where}: the source document carries no {f.source!r}, which "
                    f"{f.target!r} is declared to come from — a config field that "
                    "quietly defaulted would describe a checkpoint this is not")
            continue
        _dotted_set(out, f.target, value)
    return out


def _load_json(path: Path, *, where: str) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise TreeRepackError(f"{where}: {path.name} is not readable JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise TreeRepackError(f"{where}: {path.name} is not a JSON object")
    return doc


def _write_json(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- the engine


def apply_tree_repack(tree: Path | str, spec: TreeRepack) -> RepackReport:
    """Repack ``tree`` in place into ``spec``'s diffusers shape."""

    root = Path(tree)
    if not root.is_dir():
        raise TreeRepackError(f"repack {spec.name!r}: {root} is not a directory")
    if (root / "model_index.json").exists():
        raise TreeRepackError(
            f"repack {spec.name!r}: {root.name} already carries a model_index.json — it is "
            "already component-shaped, and repacking it again would route its keys a "
            "second time")
    already = sorted(
        d.name for d in root.iterdir() if d.is_dir() and any(d.glob("*.safetensors")))
    if already:
        raise TreeRepackError(
            f"repack {spec.name!r} expects a FLAT source tree; {root.name} already carries "
            f"weights under {already}")

    members = _root_weight_members(root)
    if not members:
        raise TreeRepackError(f"repack {spec.name!r}: no safetensors at the tree root")

    # ---- route every key, then refuse before a single byte moves
    per_file: list[tuple[Path, list[tuple[str, dict[str, Any]]]]] = [
        (member, _tensor_entries(_read_header(member)[0])) for member in members
    ]
    source_keys = [name for _, rows in per_file for name, _ in rows]
    missing = [p for p in spec.requires_key_prefixes
               if not any(k.startswith(p) for k in source_keys)]
    if missing:
        raise TreeRepackError(
            f"repack {spec.name!r}: this tree carries no key under {missing} — its "
            f"declaration requires {list(spec.requires_key_prefixes)}, so this is not a "
            f"{spec.name} tree and repacking it would produce a plausible wrong answer")

    routes: dict[str, dict[Path, dict[str, str]]] = {c.name: {} for c in spec.weight_components}
    produced_keys: dict[str, set[str]] = {c.name: set() for c in spec.weight_components}
    unrouted: list[str] = []
    tensor_count = 0
    for member, rows in per_file:
        for name, _info in rows:
            comp = spec.component_for(name)
            if comp is None:
                unrouted.append(name)
                continue
            new_key = comp.rename(name)
            if new_key in produced_keys[comp.name]:
                raise TreeRepackError(
                    f"repack {spec.name!r}: two source keys rename onto {new_key!r} in "
                    f"component {comp.name!r} — one of the tensors would be lost")
            produced_keys[comp.name].add(new_key)
            routes[comp.name].setdefault(member, {})[name] = new_key
            tensor_count += 1
    if unrouted:
        raise TreeRepackError(
            f"repack {spec.name!r}: {len(unrouted)} source key(s) match no component and "
            f"this declaration has no catch-all — first: {sorted(unrouted)[:4]}. A repack "
            "that dropped them would publish a tree that loads one tensor short")

    # ---- move or rewrite, per component, preserving members
    keys_in_file = {member: {name for name, _ in rows} for member, rows in per_file}
    written: dict[str, int] = {}
    moved = rewritten = 0
    for comp in spec.weight_components:
        contributions = routes[comp.name]
        files = [m for m in members if m in contributions]
        if not files:
            raise TreeRepackError(
                f"repack {spec.name!r}: component {comp.name!r} claims no key in this tree")
        comp_dir = root / comp.name
        comp_dir.mkdir(parents=True, exist_ok=True)
        total = len(files)
        written[comp.name] = total
        weight_map: dict[str, str] = {}
        for i, member in enumerate(files, start=1):
            mapping = contributions[member]
            name = (
                f"{comp.weight_stem}.safetensors" if total == 1
                else f"{comp.weight_stem}-{i:05d}-of-{total:05d}.safetensors"
            )
            dst = comp_dir / name
            whole = set(mapping) == keys_in_file[member]
            identity = all(src == dst_key for src, dst_key in mapping.items())
            if whole and identity:
                # The cheapest correct thing: the file IS the member. No bytes
                # are read, so a 35 GB member costs a rename.
                os.replace(member, dst)
                moved += 1
            else:
                _write_subset(member, dst, mapping)
                rewritten += 1
            for new_key in mapping.values():
                weight_map[new_key] = name
        if total > 1:
            _write_json(comp_dir / f"{comp.weight_stem}.safetensors.index.json", {
                "metadata": {"total_size": sum(
                    p.stat().st_size for p in comp_dir.glob(f"{comp.weight_stem}-*.safetensors"))},
                "weight_map": dict(sorted(weight_map.items())),
            })

    # A member that was MOVED is already gone; one that was rewritten (renamed
    # keys, or split across components) has had every one of its tensors copied
    # out by now — the loop above is the only reader — so the original dies here
    # rather than after each component, which would break the second reader.
    for member in members:
        if member.exists():
            member.unlink()

    # ---- component configs and routed files
    for comp in spec.components:
        comp_dir = root / comp.name
        comp_dir.mkdir(parents=True, exist_ok=True)
        if comp.config is not None:
            _write_component_config(root, comp_dir, comp, comp.config, spec)
        for route in comp.files:
            _move_routed_file(root, comp_dir, route, where=f"repack {spec.name!r}/{comp.name}")

    # ---- model_index.json, then the root is only what the declaration keeps
    index: dict[str, Any] = {
        "_class_name": spec.pipeline_class,
        "_diffusers_version": spec.diffusers_version,
    }
    for comp in spec.components:
        index[comp.name] = [comp.library, comp.class_name]
    _write_json(root / "model_index.json", index)

    keep = {"model_index.json", *spec.keep_root}
    for leftover in sorted(root.iterdir()):
        if leftover.is_dir():
            continue
        if leftover.name in keep:
            continue
        leftover.unlink()

    layout = observed_file_layout(root)
    report = RepackReport(
        name=spec.name, file_layout=layout, members=dict(written),
        tensor_count=tensor_count, moved_files=moved, rewritten_files=rewritten,
    )
    logger.info(
        "clone.repack name=%s components=%s members=%s tensors=%d moved=%d rewritten=%d "
        "file_layout=%s",
        spec.name, [c.name for c in spec.components], report.members, tensor_count,
        moved, rewritten, layout,
    )
    return report


def _write_component_config(
    root: Path, comp_dir: Path, comp: RepackComponent, config: ComponentConfig,
    spec: TreeRepack,
) -> None:
    where = f"repack {spec.name!r}/{comp.name}"
    doc: Mapping[str, Any] = {}
    if any(f.source for f in config.fields):
        source = root / config.source
        if not source.is_file():
            raise TreeRepackError(
                f"{where}: the source tree has no {config.source!r} to derive "
                f"{comp.name}/config.json from")
        doc = _load_json(source, where=where)
    out = _apply_fields(doc, config.fields, where=where, into={
        "_class_name": comp.class_name,
        "_diffusers_version": spec.diffusers_version,
    })
    _write_json(comp_dir / "config.json", out)


def _move_routed_file(root: Path, comp_dir: Path, route: FileRoute, *, where: str) -> None:
    src = root / route.source
    if not src.is_file():
        if route.required:
            raise TreeRepackError(
                f"{where}: the source tree has no {route.source!r}, which this declaration "
                "routes into the component — a component missing a declared file loads as a "
                "different component")
        return
    dst = comp_dir / route.name
    if route.json_overrides:
        doc = _load_json(src, where=where)
        _write_json(dst, _apply_fields({}, route.json_overrides, where=where, into=doc))
        src.unlink()
        return
    try:
        os.replace(src, dst)
    except OSError:  # pragma: no cover — cross-device moves inside one workdir
        shutil.move(str(src), str(dst))
