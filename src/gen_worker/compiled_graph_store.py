"""Worker policy sidecars over the one TCG/HashRepo compiled-graph store.

Artifact bytes, exact-key refs, verification and quarantine belong exclusively
to :mod:`torch_compiled_graphs`.  This module records only worker facts that do
not belong in the closed artifact schema: family, mint obligation, admission
verdict and the remote-publish obligation.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import math
import os
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

from hashrepo import CASRef, LocalCAS
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
# One PiB is far beyond a deployable compiled graph while still bounding hostile
# persisted integers before they reach filesystem or transport accounting.
_MAX_BYTES = 1 << 50
_VERDICTS = frozenset({
    VERDICT_UNVERIFIED,
    VERDICT_ADMITTED,
    VERDICT_QUARANTINED,
})
_SINKS = frozenset({SINK_NONE, SINK_OWED, SINK_DELIVERED, SINK_REFUSED})


def store_root() -> Path:
    """The canonical worker CAS namespace that owns graph bytes and sidecars."""

    return Path(open_worker_cas().root)


def _root(root: Optional[Path]) -> Path:
    return Path(root) if root is not None else store_root()


def _cas(root: Optional[Path] = None) -> LocalCAS:
    # Non-None is an explicit local-CLI/test namespace. Production never
    # derives another root: it opens the configured worker CAS unchanged.
    return open_worker_cas() if root is None else open_worker_cas(Path(root))


def _engine(root: Optional[Path] = None) -> Engine:
    return Engine(_cas(root))


def sidecars_root(root: Optional[Path] = None) -> Path:
    return _root(root) / SIDECARS_DIRNAME


def sidecar_dir(key: str, root: Optional[Path] = None) -> Path:
    if not is_compiled_graph_key(key):
        raise ValueError(f"not a compiled-graph key: {key!r}")
    return sidecars_root(root) / key


def memo_path(arm_token: str, root: Optional[Path] = None) -> Path:
    token = str(arm_token or "").strip()
    if not _valid_arm_token(token, allow_empty=False):
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
    "content_digest",
    "stored_at",
    "manifest",
    "verdict",
    "sink",
})


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prepare_parent(path: Path) -> None:
    missing: list[Path] = []
    cursor = path.parent
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    path.parent.mkdir(parents=True, exist_ok=True)
    for directory in missing:
        os.chmod(directory, 0o755)
    # Persist every newly created directory entry from the outside in.  The
    # later record-directory fsync cannot make an unfenced ancestor durable.
    for directory in reversed(missing):
        _fsync_directory(directory.parent)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _prepare_parent(path)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


@contextlib.contextmanager
def _record_lock(path: Path) -> Iterator[None]:
    """Serialize one fresh read-modify-write across threads and processes."""

    _prepare_parent(path)
    descriptor = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(), object_pairs_hook=_reject_duplicate_keys)


def _valid_text(value: object, *, maximum: int, allow_empty: bool) -> bool:
    if not isinstance(value, str) or value != value.strip():
        return False
    if not value:
        return allow_empty
    return (
        len(value) <= maximum
        and not value.startswith(".")
        and "/" not in value
        and "\\" not in value
        and all(character.isprintable() for character in value)
    )


def _valid_family(value: object) -> bool:
    return _valid_text(value, maximum=128, allow_empty=False)


def _valid_arm_token(value: object, *, allow_empty: bool = True) -> bool:
    return _valid_text(value, maximum=256, allow_empty=allow_empty)


def _canonical_ref(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return str(CASRef.parse(value)) == value
    except ValueError:
        return False


def _positive_finite_number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    numeric = float(cast(int | float, value))
    return math.isfinite(numeric) and numeric > 0.0


def _valid_record(value: object) -> bool:
    if not isinstance(value, dict) or frozenset(value) != _SIDECAR_FIELDS:
        return False
    stored_at = value.get("stored_at")
    return (
        type(value.get("format")) is int
        and value["format"] == _FORMAT
        and isinstance(value.get("compiled_graph_key"), str)
        and is_compiled_graph_key(value["compiled_graph_key"])
        and _valid_family(value.get("family"))
        and _valid_arm_token(value.get("arm_token"))
        and type(value.get("bytes")) is int
        and 0 < value["bytes"] <= _MAX_BYTES
        and _canonical_ref(value.get("content_digest"))
        and _positive_finite_number(stored_at)
        and _canonical_ref(value.get("manifest"))
        and value.get("verdict") in _VERDICTS
        and value.get("sink") in _SINKS
    )


def _allows_verdict(current: str, requested: str) -> bool:
    return (
        requested == current
        or current == VERDICT_UNVERIFIED
        or (current == VERDICT_ADMITTED and requested == VERDICT_QUARANTINED)
    )


def _allows_sink(current: str, requested: str) -> bool:
    return requested == current or (
        current == SINK_OWED and requested in {SINK_DELIVERED, SINK_REFUSED}
    )


def _read_record(path: Path) -> Optional[dict[str, Any]]:
    try:
        value = _read_json(path)
    except (OSError, ValueError):
        return None
    if not _valid_record(value):
        return None
    return cast(dict[str, Any], value)


def _export(key: str, root: Optional[Path]) -> Path:
    destination = (
        sidecars_root(root) / EXPORTS_DIRNAME / key / "compiled_graph.tar.gz"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(_engine(root).export_artifact(key, destination))


def _local(record: dict[str, Any], root: Optional[Path]) -> LocalCompiledGraph:
    key = str(record["compiled_graph_key"])
    artifact = _export(key, root)
    return LocalCompiledGraph(
        compiled_graph_key=key,
        artifact=artifact,
        content_digest=str(record["content_digest"]),
        family=str(record["family"]),
        arm_token=str(record["arm_token"]),
        bytes=int(record["bytes"]),
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

    if (
        not is_compiled_graph_key(key)
        or not _valid_family(family)
        or not _valid_arm_token(arm_token)
        or verdict not in {VERDICT_UNVERIFIED, VERDICT_ADMITTED}
        or sink not in {SINK_NONE, SINK_OWED}
    ):
        logger.warning("compiled-graph-store: invalid worker sidecar input")
        return None
    try:
        cas = _cas(root)
        result = Engine(cas).import_artifact(key, artifact)
        if result.outcome == StoreOutcome.DIVERGENT:
            logger.error(
                "compiled-graph-store: divergent bytes for %s were quarantined", key
            )
            return None
        manifest = cas.load_manifest(result.manifest)
        if len(manifest.files) != 1:
            raise StorageError("compiled graph CAS manifest is not one artifact")
        stored_file = manifest.files[0]
        now = time.time()
        record = {
            "format": _FORMAT,
            "compiled_graph_key": str(result.key),
            "family": family,
            "arm_token": arm_token,
            "bytes": int(stored_file.size_bytes),
            "content_digest": str(stored_file.digest),
            "stored_at": now,
            "manifest": str(result.manifest),
            "verdict": verdict,
            "sink": sink,
        }
        path = sidecar_dir(str(result.key), root) / RECORD_NAME
        with _record_lock(path):
            existing = _read_record(path)
            if existing is not None:
                immutable = (
                    "compiled_graph_key",
                    "family",
                    "bytes",
                    "content_digest",
                    "manifest",
                )
                if any(existing[field] != record[field] for field in immutable):
                    raise StorageError(
                        "worker sidecar disagrees with the selected compiled graph"
                    )
                record["stored_at"] = existing["stored_at"]
                record["arm_token"] = arm_token or existing["arm_token"]
                if not _allows_verdict(str(existing["verdict"]), verdict):
                    record["verdict"] = existing["verdict"]
                if not _allows_sink(str(existing["sink"]), sink):
                    record["sink"] = existing["sink"]
            if not _valid_record(record):
                raise StorageError("worker sidecar failed its strict schema")
            _write_json_atomic(path, record)
        if arm_token:
            note_memo(arm_token, str(result.key), root)
        return _local(record, root)
    except (OSError, QuarantinedArtifact, StorageError, ValueError) as exc:
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
    try:
        with _record_lock(path):
            record = _read_record(path)
            if record is None:
                return False
            current_verdict = str(record["verdict"])
            current_sink = str(record["sink"])
            next_verdict = current_verdict if verdict is None else verdict
            next_sink = current_sink if sink is None else sink
            if (
                not _allows_verdict(current_verdict, next_verdict)
                or not _allows_sink(current_sink, next_sink)
            ):
                return False
            record["verdict"] = next_verdict
            record["sink"] = next_sink
            if not _valid_record(record):
                return False
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
    destination = sidecars_root(root) / ".resolved" / key
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
    destination = sidecars_root(root) / ".resolved" / key
    runner_destination = sidecars_root(root) / ".runners" / key
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


def describe(
    key: str,
    root: Optional[Path] = None,
) -> Optional[StoredCompiledGraph]:
    """Resolve one exact graph for metadata without loading its package."""

    if not is_compiled_graph_key(key):
        return None
    try:
        return _engine(root).resolve(
            key,
            sidecars_root(root) / ".described" / key,
        )
    except (OSError, QuarantinedArtifact, StorageError):
        return None


def note_memo(arm_token: str, key: str, root: Optional[Path] = None) -> bool:
    if (
        not _valid_arm_token(arm_token, allow_empty=False)
        or not is_compiled_graph_key(key)
    ):
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
        value = _read_json(memo_path(arm_token, root))
    except (OSError, ValueError):
        return None
    noted_at = value.get("noted_at") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or set(value) != {"format", "compiled_graph_key", "noted_at"}
        or type(value.get("format")) is not int
        or value["format"] != _FORMAT
        or not isinstance(value.get("compiled_graph_key"), str)
        or not is_compiled_graph_key(value["compiled_graph_key"])
        or not _positive_finite_number(noted_at)
    ):
        return None
    return lookup(value["compiled_graph_key"], root)


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
        and (record := _read_record(path / RECORD_NAME)) is not None
        and record["compiled_graph_key"] == path.name
        for path in directory.iterdir()
    )


def graphs_owed_to_sink(root: Optional[Path] = None) -> list[LocalCompiledGraph]:
    directory = sidecars_root(root)
    if not directory.is_dir():
        return []
    rows: list[LocalCompiledGraph] = []
    for path in sorted(directory.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        record = _read_record(path / RECORD_NAME)
        if (
            record is None
            or record["compiled_graph_key"] != path.name
            or record["verdict"] != VERDICT_ADMITTED
            or record["sink"] != SINK_OWED
        ):
            continue
        try:
            rows.append(_local(record, root))
        except (OSError, QuarantinedArtifact, StorageError):
            continue
    return rows


def note_refusal(code: str, detail: str = "", root: Optional[Path] = None) -> bool:
    if str(code or "").strip() != UNTRUSTED_REFUSAL_CODE:
        return False
    try:
        _write_json_atomic(sidecars_root(root) / TRUST_CLASS_NAME, {
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
        value = _read_json(sidecars_root(root) / TRUST_CLASS_NAME)
    except (OSError, ValueError):
        return ""
    learned_at = value.get("learned_at") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or set(value) != {"format", "class", "code", "detail", "learned_at"}
        or type(value.get("format")) is not int
        or value["format"] != _FORMAT
        or value.get("class") != TRUST_UNTRUSTED
        or value.get("code") != UNTRUSTED_REFUSAL_CODE
        or not isinstance(value.get("detail"), str)
        or len(value["detail"]) > 500
        or not _positive_finite_number(learned_at)
    ):
        return ""
    return TRUST_UNTRUSTED


def keeps_graphs_locally(root: Optional[Path] = None) -> bool:
    return trust_class(root) == TRUST_UNTRUSTED
