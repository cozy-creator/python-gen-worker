"""``endpoint.lock`` — the discovery manifest PLUS the derive document."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import msgspec

LOCK_FILENAME = "endpoint.lock"

DERIVE_BLOCK = "derive"

DERIVE_BLOCK_V = 1


class LockError(RuntimeError):
    """The lock on disk cannot be used as it stands."""


@dataclass(frozen=True)
class DeriveBlock:
    """``[derive]`` — the trace, saved."""

    v: int
    interface_v: int
    inputs_digest: str
    document_digest: str
    document: str
    trace_device: str
    endpoint: str

    def decoded(self) -> dict[str, Any]:
        """The document as a dict."""
        raw = self.document.encode("ascii")
        actual = hashlib.sha256(raw).hexdigest()
        if actual != self.document_digest:
            raise LockError(
                f"[{DERIVE_BLOCK}] document does not match its own digest "
                f"(stored {self.document_digest}, actual {actual}). The lock "
                f"was edited by hand or truncated in transit; re-run "
                f"`gen-worker lock --force`."
            )
        return dict(json.loads(raw))

    def as_table(self) -> dict[str, Any]:
        return {
            "v": self.v,
            "interface_v": self.interface_v,
            "inputs_digest": self.inputs_digest,
            "document_digest": self.document_digest,
            "trace_device": self.trace_device,
            "endpoint": self.endpoint,
            "document": self.document,
        }

    @classmethod
    def from_table(cls, table: Mapping[str, Any]) -> "DeriveBlock":
        missing = [
            k
            for k in ("v", "interface_v", "inputs_digest", "document_digest",
                      "document", "trace_device", "endpoint")
            if k not in table
        ]
        if missing:
            raise LockError(
                f"[{DERIVE_BLOCK}] is missing {missing}; it was written by a "
                f"different version of `gen-worker lock`. Re-run "
                f"`gen-worker lock --force`."
            )
        return cls(
            v=int(table["v"]),
            interface_v=int(table["interface_v"]),
            inputs_digest=str(table["inputs_digest"]),
            document_digest=str(table["document_digest"]),
            document=str(table["document"]),
            trace_device=str(table["trace_device"]),
            endpoint=str(table["endpoint"]),
        )


def torchcg_format_versions() -> tuple[int, int]:
    """``(GRAPH_INTERFACE_FORMAT, DOCUMENT_FORMAT)`` — IMPORTED, never spelled."""
    from .._vendor.torchcg import DOCUMENT_FORMAT, GRAPH_INTERFACE_FORMAT

    return int(GRAPH_INTERFACE_FORMAT), int(DOCUMENT_FORMAT)


def tracer_digest() -> str:
    """A fingerprint of the CODE THAT TRACES: torchcg plus gen_worker's derive."""
    h = hashlib.sha256()
    for module_name in ("torchcg", "gen_worker.release"):
        try:
            module = __import__(module_name, fromlist=["__file__"])
            origin = Path(getattr(module, "__file__", "") or "")
        except Exception:  # noqa: BLE001 - absence is an answer, not a failure
            continue
        if not origin.is_file():
            continue
        for path in sorted(origin.parent.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            h.update(path.name.encode("utf-8"))
            h.update(_file_digest(path).encode("ascii"))
    return h.hexdigest()


def graph_identities(document: Mapping[str, Any]) -> tuple[str, ...]:
    """Every graph IDENTITY the document names, deduped + sorted."""
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "graph" and isinstance(value, str) and value:
                    found.add(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(document.get("graphs", {}))
    return tuple(sorted(found))


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_files(root: Path) -> tuple[Path, ...]:
    """The author source the derive actually imports, sorted and stable."""
    seen: set[Path] = set()
    src = root / "src"
    if src.is_dir():
        seen.update(p for p in src.rglob("*.py"))
    seen.update(p for p in root.glob("*.py"))
    skip = ("__pycache__", ".venv", ".mypy_cache", ".pytest_cache")
    return tuple(
        sorted(
            p
            for p in seen
            if p.is_file() and not any(part in skip for part in p.parts)
        )
    )


def inputs_digest(
    *,
    root: Path,
    module_name: str,
    checkpoint_ref: str,
    trace_device: str,
    lockfile: Optional[Path] = None,
    extra: Iterable[str] = (),
) -> str:
    """Hash everything the derive READS."""
    h = hashlib.sha256()

    def field(label: str, value: str) -> None:
        h.update(f"{label}:{len(value)}:{value}\n".encode("utf-8"))

    interface_v, document_v = torchcg_format_versions()
    field("block_v", str(DERIVE_BLOCK_V))
    field("interface_v", str(interface_v))
    field("document_v", str(document_v))
    field("tracer", tracer_digest())
    field("module", module_name)
    field("checkpoint_ref", checkpoint_ref)
    field("trace_device", trace_device)
    field("lockfile", _file_digest(lockfile) if lockfile and lockfile.is_file() else "")
    for path in source_files(root):
        field(f"src:{path.relative_to(root).as_posix()}", _file_digest(path))
    for item in extra:
        field("extra", item)
    return h.hexdigest()


@dataclass(frozen=True)
class Reuse:
    """Whether a stored derive can be reused, and — always — WHY NOT."""

    ok: bool
    reason: str
    block: Optional[DeriveBlock] = None
    document: Optional[dict[str, Any]] = None


def derive_is_reusable(
    block: Optional[DeriveBlock],
    *,
    want_inputs_digest: str,
    want_trace_device: str,
    cas_has: Any = None,
) -> Reuse:
    """The skip check: can we serve this lock's trace instead of re-deriving? ``cas_has`` is a predicate over a GRAPH IDENTITY (typically ``LocalGraphStore.has_program``)."""
    if block is None:
        return Reuse(False, "no [derive] block — the trace was never saved")
    if block.v != DERIVE_BLOCK_V:
        return Reuse(
            False,
            f"[derive] block format v{block.v}, this gen-worker writes "
            f"v{DERIVE_BLOCK_V}",
        )
    if block.trace_device != want_trace_device:
        return Reuse(
            False,
            f"saved trace is {block.trace_device}-class, this host traces "
            f"{want_trace_device}-class — different specializations, not a "
            f"degraded one",
        )
    if block.inputs_digest != want_inputs_digest:
        return Reuse(
            False,
            "inputs changed (author source, uv.lock closure, checkpoint ref, "
            "trace device, a torchcg format version, or the tracer's own "
            "source)",
        )
    try:
        document = block.decoded()
    except LockError as exc:
        return Reuse(False, str(exc))
    if cas_has is not None:
        missing = [g for g in graph_identities(document) if not cas_has(g)]
        if missing:
            return Reuse(
                False,
                f"this box holds no serialized program for {len(missing)} of "
                f"the graphs the document names (first: {missing[0]}) — the "
                f"document survived a GC its programs did not, or it was "
                f"derived on another box",
            )
    return Reuse(True, "inputs unchanged and every program blob present",
                 block=block, document=document)


def read_lock(path: Path) -> dict[str, Any]:
    """The lock file as a dict, or a typed refusal naming the file."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise LockError(f"no {path} — run `gen-worker lock` first") from exc
    try:
        decoded = msgspec.toml.decode(raw)
    except Exception as exc:  # noqa: BLE001 - any decode failure is the answer
        raise LockError(f"{path} is not valid TOML: {exc}") from exc
    if not isinstance(decoded, dict):
        raise LockError(f"{path} must decode to a TOML table")
    return dict(decoded)


def read_derive_block(path: Path) -> Optional[DeriveBlock]:
    """``[derive]`` from the lock at ``path``, or None when it carries none."""
    table = read_lock(path).get(DERIVE_BLOCK)
    if table is None:
        return None
    if not isinstance(table, dict):
        raise LockError(f"[{DERIVE_BLOCK}] in {path} must be a table")
    return DeriveBlock.from_table(table)


def write_lock(
    path: Path, manifest: Mapping[str, Any], derive: Optional[DeriveBlock]
) -> None:
    """Write discovery blocks + ``[derive]`` to ``path``, atomically."""
    from ..discovery.discover import _strip_none

    document: dict[str, Any] = dict(_strip_none(dict(manifest)))
    if derive is not None:
        document[DERIVE_BLOCK] = derive.as_table()
    encoded = msgspec.toml.encode(document)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_bytes(encoded)
    tmp.replace(path)


__all__ = [
    "DERIVE_BLOCK",
    "DERIVE_BLOCK_V",
    "LOCK_FILENAME",
    "DeriveBlock",
    "LockError",
    "Reuse",
    "derive_is_reusable",
    "inputs_digest",
    "graph_identities",
    "read_derive_block",
    "read_lock",
    "source_files",
    "torchcg_format_versions",
    "write_lock",
]
