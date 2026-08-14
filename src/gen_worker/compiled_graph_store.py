"""Worker policy sidecars over the one TCG/HashRepo compiled-graph store.

Artifact bytes, exact-key refs, verification and quarantine belong exclusively
to :mod:`torch_compiled_graphs`.  This module records only worker facts that do
not belong in the closed artifact schema: family, mint obligation, admission
verdict and the remote-publish obligation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from torch_compiled_graphs import (
    CompiledGraphRunner,
    Engine,
    QuarantinedArtifact,
    StorageError,
    StoreOutcome,
    StoredCompiledGraph,
    is_compiled_graph_key,
)

from .models.cache_paths import open_worker_cas

logger = logging.getLogger(__name__)

ENV_STORE_DIR = "GEN_WORKER_LOCAL_COMPILED_GRAPHS_DIR"
SIDECARS_DIRNAME = "aot-compiled-graphs"
MEMO_DIRNAME = ".memo"
EXPORTS_DIRNAME = ".exports"
RECORD_NAME = "record.json"
TRUST_CLASS_NAME = "trust-class.json"

UNTRUSTED_REFUSAL_CODE = "compiled_graph_publish_untrusted_tier"
TRUST_UNTRUSTED = "untrusted"

VERDICT_UNVERIFIED = "unverified"
VERDICT_ADMITTED = "admitted"
VERDICT_QUARANTINED = "quarantined"

SINK_NONE = "none"
SINK_OWED = "owed"
SINK_DELIVERED = "delivered"
SINK_REFUSED = "refused"

_FORMAT = 1


def store_root() -> Path:
    configured = os.environ.get(ENV_STORE_DIR, "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "cozy" / "compiled-graphs"


def _root(root: Optional[Path]) -> Path:
    return Path(root) if root is not None else store_root()


def _engine(root: Optional[Path] = None) -> Engine:
    return Engine(open_worker_cas(root))


def sidecars_root(root: Optional[Path] = None) -> Path:
    return _root(root) / SIDECARS_DIRNAME


def sidecar_dir(key: str, root: Optional[Path] = None) -> Path:
    if not is_compiled_graph_key(key):
        raise ValueError(f"not a compiled-graph key: {key!r}")
    return sidecars_root(root) / key


def memo_path(arm_token: str, root: Optional[Path] = None) -> Path:
    token = str(arm_token or "").strip()
    if not token or "/" in token or token.startswith("."):
        raise ValueError(f"not an arm-identity token: {arm_token!r}")
    return sidecars_root(root) / MEMO_DIRNAME / f"{token}.json"


@dataclass(frozen=True)
class LocalCompiledGraph:
    compiled_graph_key: str
    artifact: Path
    content_digest: str
    family: str
    arm_token: str
    bytes: int
    stored_at: float
    manifest: str
    verdict: str = VERDICT_ADMITTED
    sink: str = SINK_NONE


@dataclass(frozen=True)
class LoadedCompiledGraph:
    """One admitted TCG graph and its not-yet-bound serving runner."""

    compiled_graph: StoredCompiledGraph
    runner: CompiledGraphRunner


_SIDECAR_FIELDS = frozenset({
    "format",
    "compiled_graph_key",
    "family",
    "arm_token",
    "bytes",
    "stored_at",
    "manifest",
    "verdict",
    "sink",
})


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    os.replace(temporary, path)


def _read_record(path: Path) -> Optional[dict[str, Any]]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if (
        not isinstance(value, dict)
        or frozenset(value) != _SIDECAR_FIELDS
        or value.get("format") != _FORMAT
        or not is_compiled_graph_key(value.get("compiled_graph_key"))
    ):
        return None
    return value


def _artifact_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 << 20):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _export(key: str, root: Optional[Path]) -> Path:
    destination = _root(root) / EXPORTS_DIRNAME / key / "compiled_graph.tar.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(_engine(root).export_artifact(key, destination))


def _local(record: dict[str, Any], root: Optional[Path]) -> LocalCompiledGraph:
    key = str(record["compiled_graph_key"])
    artifact = _export(key, root)
    return LocalCompiledGraph(
        compiled_graph_key=key,
        artifact=artifact,
        content_digest=_artifact_digest(artifact),
        family=str(record["family"]),
        arm_token=str(record["arm_token"]),
        bytes=int(record["bytes"]),
        stored_at=float(record["stored_at"]),
        manifest=str(record["manifest"]),
        verdict=str(record["verdict"]),
        sink=str(record["sink"]),
    )


def store(
    artifact: Path,
    *,
    key: str,
    family: str,
    arm_token: str = "",
    verdict: str = VERDICT_ADMITTED,
    sink: str = SINK_NONE,
    root: Optional[Path] = None,
) -> Optional[LocalCompiledGraph]:
    """Import one strict TCG artifact, then record its worker-only sidecar."""

    try:
        result = _engine(root).import_artifact(key, artifact)
        if result.outcome == StoreOutcome.DIVERGENT:
            logger.error(
                "compiled-graph-store: divergent bytes for %s were quarantined", key
            )
            return None
        now = time.time()
        record = {
            "format": _FORMAT,
            "compiled_graph_key": str(result.key),
            "family": str(family or ""),
            "arm_token": str(arm_token or ""),
            "bytes": int(Path(artifact).stat().st_size),
            "stored_at": now,
            "manifest": str(result.manifest),
            "verdict": str(verdict),
            "sink": str(sink),
        }
        _write_json_atomic(sidecar_dir(str(result.key), root) / RECORD_NAME, record)
        if arm_token:
            note_memo(arm_token, str(result.key), root)
        return _local(record, root)
    except (OSError, StorageError, ValueError) as exc:
        logger.warning("compiled-graph-store: could not import %s (%s)", key, exc)
        return None


def mark(
    key: str,
    *,
    verdict: Optional[str] = None,
    sink: Optional[str] = None,
    root: Optional[Path] = None,
) -> bool:
    try:
        path = sidecar_dir(key, root) / RECORD_NAME
    except ValueError:
        return False
    record = _read_record(path)
    if record is None:
        return False
    if verdict is not None:
        record["verdict"] = str(verdict)
    if sink is not None:
        record["sink"] = str(sink)
    try:
        _write_json_atomic(path, record)
    except OSError:
        return False
    return True


def lookup(key: str, root: Optional[Path] = None) -> Optional[LocalCompiledGraph]:
    try:
        record = _read_record(sidecar_dir(key, root) / RECORD_NAME)
    except ValueError:
        return None
    if (
        record is None
        or record["compiled_graph_key"] != key
        or record["verdict"] != VERDICT_ADMITTED
    ):
        return None
    destination = _root(root) / ".resolved" / key
    try:
        graph = _engine(root).resolve(key, destination)
        if graph is None:
            return None
        return _local(record, root)
    except (OSError, QuarantinedArtifact, StorageError):
        mark(key, verdict=VERDICT_QUARANTINED, root=root)
        return None


def load_runner(
    key: str,
    root: Optional[Path] = None,
) -> Optional[LoadedCompiledGraph]:
    """Resolve and load one exact TCG key without interpreting its artifact.

    TCG owns both verification and package loading.  The worker receives the
    closed metadata only so its ingress adapter can select the resident target
    and replay the declared call contract before invoking the runner.
    """

    if not is_compiled_graph_key(key):
        return None
    engine = _engine(root)
    destination = _root(root) / ".resolved" / key
    runner_destination = _root(root) / ".runners" / key
    try:
        graph = engine.resolve(key, destination)
        if graph is None:
            return None
        runner = engine.runner(key, runner_destination)
        if runner is None or runner.key != key or graph.key != key:
            return None
        return LoadedCompiledGraph(graph, runner)
    except (OSError, QuarantinedArtifact, StorageError):
        mark(key, verdict=VERDICT_QUARANTINED, root=root)
        return None


def note_memo(arm_token: str, key: str, root: Optional[Path] = None) -> bool:
    if not arm_token or not is_compiled_graph_key(key):
        return False
    try:
        _write_json_atomic(memo_path(arm_token, root), {
            "format": _FORMAT,
            "compiled_graph_key": key,
            "noted_at": time.time(),
        })
        return True
    except (OSError, ValueError):
        return False


def lookup_for_arm(
    arm_token: str, root: Optional[Path] = None,
) -> Optional[LocalCompiledGraph]:
    try:
        value = json.loads(memo_path(arm_token, root).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or set(value) != {
        "format", "compiled_graph_key", "noted_at",
    } or value.get("format") != _FORMAT:
        return None
    return lookup(str(value.get("compiled_graph_key") or ""), root)


def sweep_superseded_memos(scheme: str, root: Optional[Path] = None) -> int:
    current = str(scheme or "").strip() + "-"
    directory = sidecars_root(root) / MEMO_DIRNAME
    if len(current) < 2 or not directory.is_dir():
        return 0
    removed = 0
    for path in directory.glob("*.json"):
        if path.name.startswith(current):
            continue
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def drop(key: str, root: Optional[Path] = None) -> None:
    """Quarantine the worker sidecar; TCG retains immutable bytes for diagnosis."""

    mark(key, verdict=VERDICT_QUARANTINED, root=root)


def has_graphs(root: Optional[Path] = None) -> bool:
    """Return whether any well-formed sidecar exists without exporting bytes."""

    directory = sidecars_root(root)
    if not directory.is_dir():
        return False
    return any(
        path.is_dir()
        and not path.name.startswith(".")
        and _read_record(path / RECORD_NAME) is not None
        for path in directory.iterdir()
    )


def stored_graphs(root: Optional[Path] = None) -> list[LocalCompiledGraph]:
    rows: list[LocalCompiledGraph] = []
    directory = sidecars_root(root)
    if not directory.is_dir():
        return rows
    for path in sorted(directory.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        record = _read_record(path / RECORD_NAME)
        if record is None or record["compiled_graph_key"] != path.name:
            continue
        try:
            rows.append(_local(record, root))
        except (OSError, StorageError):
            continue
    return rows


def quarantined_graphs(root: Optional[Path] = None) -> list[LocalCompiledGraph]:
    return [row for row in stored_graphs(root) if row.verdict == VERDICT_QUARANTINED]


def graphs_owed_to_sink(root: Optional[Path] = None) -> list[LocalCompiledGraph]:
    return [
        row for row in stored_graphs(root)
        if row.verdict == VERDICT_ADMITTED and row.sink == SINK_OWED
    ]


def note_refusal(code: str, detail: str = "", root: Optional[Path] = None) -> bool:
    if str(code or "").strip() != UNTRUSTED_REFUSAL_CODE:
        return False
    try:
        _write_json_atomic(_root(root) / TRUST_CLASS_NAME, {
            "format": _FORMAT,
            "class": TRUST_UNTRUSTED,
            "code": UNTRUSTED_REFUSAL_CODE,
            "detail": str(detail or "")[:500],
            "learned_at": time.time(),
        })
        return True
    except OSError:
        return False


def trust_class(root: Optional[Path] = None) -> str:
    try:
        value = json.loads((_root(root) / TRUST_CLASS_NAME).read_text())
    except (OSError, ValueError):
        return ""
    if not isinstance(value, dict) or value.get("format") != _FORMAT:
        return ""
    return str(value.get("class") or "")


def keeps_graphs_locally(root: Optional[Path] = None) -> bool:
    return trust_class(root) == TRUST_UNTRUSTED


__all__ = [
    "ENV_STORE_DIR",
    "LoadedCompiledGraph",
    "LocalCompiledGraph",
    "MEMO_DIRNAME",
    "RECORD_NAME",
    "SIDECARS_DIRNAME",
    "SINK_DELIVERED",
    "SINK_NONE",
    "SINK_OWED",
    "SINK_REFUSED",
    "TRUST_CLASS_NAME",
    "TRUST_UNTRUSTED",
    "UNTRUSTED_REFUSAL_CODE",
    "VERDICT_ADMITTED",
    "VERDICT_QUARANTINED",
    "VERDICT_UNVERIFIED",
    "drop",
    "graphs_owed_to_sink",
    "has_graphs",
    "keeps_graphs_locally",
    "lookup",
    "lookup_for_arm",
    "load_runner",
    "mark",
    "memo_path",
    "note_memo",
    "note_refusal",
    "quarantined_graphs",
    "sidecar_dir",
    "sidecars_root",
    "store",
    "store_root",
    "stored_graphs",
    "sweep_superseded_memos",
    "trust_class",
]
