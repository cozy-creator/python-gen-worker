"""Worker policy sidecars over the one TCG/HashRepo compiled-graph store.

Artifact bytes, exact-key refs, verification and integrity quarantine belong
exclusively to :mod:`torch_compiled_graphs`.  This module records only worker
facts that do not belong in the closed artifact schema: family, mint
obligation, admission verdict (including policy quarantine), and the
remote-publish obligation.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import math
import os
import stat
import tempfile
import time
from collections.abc import Callable, Iterator
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
_PENDING_RECORD_NAME = ".record.json.pending"
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
# A record has ten bounded scalar fields; memo and trust rows are smaller. 16
# KiB is over 8x their largest canonical encoding while making every persisted
# read independent of a hostile file's size.
_MAX_JSON_BYTES = 16 << 10
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
    metadata: Optional[dict[str, object]] = None


@dataclass(frozen=True)
class OwedCompiledGraph:
    """Publish obligation metadata; resolving/exporting is deliberately lazy."""

    compiled_graph_key: str
    content_digest: str
    family: str
    arm_token: str
    bytes: int


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
_MEMO_FIELDS = frozenset({"format", "compiled_graph_key", "noted_at"})
_TRUST_FIELDS = frozenset({"format", "class", "code", "detail", "learned_at"})
_IMMUTABLE_RECORD_FIELDS = (
    "compiled_graph_key",
    "family",
    "bytes",
    "content_digest",
    "manifest",
)


class _PersistedStateError(ValueError):
    """A named state file exists but is not one value this worker may replace."""


class _PersistedStateAbsent(FileNotFoundError):
    """The state name was absent at its initial metadata lookup."""


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
    if len(encoded) > _MAX_JSON_BYTES:
        raise ValueError(f"persisted JSON exceeds {_MAX_JSON_BYTES} bytes")
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
    try:
        discovered = path.lstat()
    except FileNotFoundError as exc:
        raise _PersistedStateAbsent(path) from exc
    if not stat.S_ISREG(discovered.st_mode):
        raise ValueError(f"persisted state is not a regular file: {path}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if not getattr(os, "O_PATH", 0):
        flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags | getattr(os, "O_PATH", 0))
    try:
        opened = os.fstat(descriptor)
        discovered_identity = (discovered.st_dev, discovered.st_ino)
        opened_identity = (opened.st_dev, opened.st_ino)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened_identity != discovered_identity
        ):
            raise ValueError(f"persisted state changed while opening: {path}")
        if getattr(os, "O_PATH", 0):
            reader = os.open(
                f"/proc/self/fd/{descriptor}",
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
            )
        else:
            reader = descriptor
            descriptor = -1
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    with os.fdopen(reader, "rb") as source:
        encoded = source.read(_MAX_JSON_BYTES + 1)
    if len(encoded) > _MAX_JSON_BYTES:
        raise ValueError(f"persisted JSON exceeds {_MAX_JSON_BYTES} bytes")
    return json.loads(encoded, object_pairs_hook=_reject_duplicate_keys)


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
        and isinstance(value.get("verdict"), str)
        and value["verdict"] in _VERDICTS
        and isinstance(value.get("sink"), str)
        and value["sink"] in _SINKS
    )


def _valid_memo(value: object) -> bool:
    return (
        isinstance(value, dict)
        and frozenset(value) == _MEMO_FIELDS
        and type(value.get("format")) is int
        and value["format"] == _FORMAT
        and isinstance(value.get("compiled_graph_key"), str)
        and is_compiled_graph_key(value["compiled_graph_key"])
        and _positive_finite_number(value.get("noted_at"))
    )


def _valid_trust(value: object) -> bool:
    return (
        isinstance(value, dict)
        and frozenset(value) == _TRUST_FIELDS
        and type(value.get("format")) is int
        and value["format"] == _FORMAT
        and value.get("class") == TRUST_UNTRUSTED
        and value.get("code") == UNTRUSTED_REFUSAL_CODE
        and isinstance(value.get("detail"), str)
        and len(value["detail"]) <= 500
        and _positive_finite_number(value.get("learned_at"))
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


def _read_state(
    path: Path,
    valid: Callable[[object], bool],
    label: str,
) -> Optional[dict[str, Any]]:
    try:
        value = _read_json(path)
    except _PersistedStateAbsent:
        return None
    except (OSError, ValueError) as exc:
        raise _PersistedStateError(f"invalid persisted {label} {path}: {exc}") from exc
    if not valid(value):
        raise _PersistedStateError(f"invalid persisted {label} {path}")
    return cast(dict[str, Any], value)


def _read_record(path: Path) -> Optional[dict[str, Any]]:
    return _read_state(path, _valid_record, "compiled-graph sidecar")


def _read_memo(path: Path) -> Optional[dict[str, Any]]:
    return _read_state(path, _valid_memo, "compiled-graph arm memo")


def _read_trust(path: Path) -> Optional[dict[str, Any]]:
    return _read_state(path, _valid_trust, "compiled-graph trust class")


def _export(key: str, engine: Engine, root: Optional[Path]) -> Path:
    destination = (
        sidecars_root(root) / EXPORTS_DIRNAME / key / "compiled_graph.tar.gz"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(engine.export_artifact(key, destination))


def _local(
    record: dict[str, Any],
    graph: StoredCompiledGraph,
    engine: Engine,
    root: Optional[Path],
) -> LocalCompiledGraph:
    key = str(record["compiled_graph_key"])
    return LocalCompiledGraph(
        compiled_graph_key=key,
        artifact=_export(key, engine, root),
        content_digest=str(record["content_digest"]),
        family=str(record["family"]),
        arm_token=str(record["arm_token"]),
        bytes=int(record["bytes"]),
        sink=str(record["sink"]),
        metadata=dict(graph.metadata),
    )


def _selected_record(
    key: str,
    allowed_verdicts: frozenset[str],
    root: Optional[Path],
) -> Optional[dict[str, Any]]:
    try:
        record = _read_record(sidecar_dir(key, root) / RECORD_NAME)
    except (OSError, ValueError):
        return None
    if (
        record is None
        or record["compiled_graph_key"] != key
        or record["verdict"] not in allowed_verdicts
    ):
        return None
    return record


def _resolve_selected(
    key: str,
    record: dict[str, Any],
    engine: Engine,
    cas: LocalCAS,
    destination: Path,
) -> Optional[StoredCompiledGraph]:
    graph = engine.resolve(key, destination)
    if graph is None or graph.key != key or str(graph.manifest) != record["manifest"]:
        return None
    manifest = cas.load_manifest(graph.manifest)
    if len(manifest.files) != 1:
        return None
    stored = manifest.files[0]
    if (
        str(stored.digest) != record["content_digest"]
        or stored.size_bytes != record["bytes"]
    ):
        return None
    return graph


def _merge_record(
    existing: Optional[dict[str, Any]],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if existing is None:
        return candidate
    if any(existing[field] != candidate[field] for field in _IMMUTABLE_RECORD_FIELDS):
        raise StorageError("worker sidecar disagrees with the selected compiled graph")
    candidate["stored_at"] = existing["stored_at"]
    candidate["arm_token"] = existing["arm_token"] or candidate["arm_token"]
    if not _allows_verdict(str(existing["verdict"]), str(candidate["verdict"])):
        candidate["verdict"] = existing["verdict"]
    if not _allows_sink(str(existing["sink"]), str(candidate["sink"])):
        candidate["sink"] = existing["sink"]
    return candidate


def _memo_preflight(arm_token: str, key: str, root: Path | None) -> None:
    if not arm_token:
        return
    memo = _read_memo(memo_path(arm_token, root))
    if memo is not None and memo["compiled_graph_key"] != key:
        raise StorageError("arm memo already names a different compiled graph")


def _install_staged_record(staged: Path, destination: Path) -> None:
    """Publish a durable staged row without replacing any named state."""

    os.link(staged, destination, follow_symlinks=False)
    try:
        _fsync_directory(destination.parent)
    except BaseException:
        with contextlib.suppress(OSError):
            destination.unlink()
            _fsync_directory(destination.parent)
        raise
    with contextlib.suppress(OSError):
        staged.unlink()
        _fsync_directory(destination.parent)


def _note_memo_locked(path: Path, key: str) -> bool:
    existing = _read_memo(path)
    if existing is not None:
        return str(existing["compiled_graph_key"]) == key
    _write_json_atomic(path, {
        "format": _FORMAT,
        "compiled_graph_key": key,
        "noted_at": time.time(),
    })
    return True


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
        path = sidecar_dir(key, root) / RECORD_NAME
        staged_path = path.with_name(_PENDING_RECORD_NAME)
        with _record_lock(path):
            # Present-invalid is an operator fact, never permission to replace
            # it. Read it before importing so corrupt policy state cannot
            # mutate TCG as a side effect of a refused worker operation.
            existing = _read_record(path)
            staged = _read_record(staged_path)
            if existing is not None and existing["compiled_graph_key"] != key:
                raise _PersistedStateError(
                    "compiled-graph sidecar is stored under a different key"
                )
            if staged is not None and staged["compiled_graph_key"] != key:
                raise _PersistedStateError(
                    "staged compiled-graph sidecar is stored under a different key"
                )
            if existing is not None and staged is not None:
                if staged != existing:
                    raise _PersistedStateError(
                        "live and staged compiled-graph sidecars disagree"
                    )
                staged_path.unlink()
                _fsync_directory(staged_path.parent)
                staged = None
            basis = existing if existing is not None else staged
            _memo_preflight(arm_token, key, root)
            if basis is not None:
                _memo_preflight(str(basis["arm_token"]), key, root)
            cas = _cas(root)
            engine = Engine(cas)
            result = engine.import_artifact(key, artifact)
            if result.outcome == StoreOutcome.DIVERGENT:
                logger.error(
                    "compiled-graph-store: divergent bytes for %s were quarantined",
                    key,
                )
                return None
            if str(result.key) != key:
                raise StorageError("TCG import returned a different exact key")
            manifest = cas.load_manifest(result.manifest)
            if len(manifest.files) != 1:
                raise StorageError("compiled graph CAS manifest is not one artifact")
            stored_file = manifest.files[0]
            record = _merge_record(basis, {
                "format": _FORMAT,
                "compiled_graph_key": str(result.key),
                "family": family,
                "arm_token": arm_token,
                "bytes": int(stored_file.size_bytes),
                "content_digest": str(stored_file.digest),
                "stored_at": time.time(),
                "manifest": str(result.manifest),
                "verdict": verdict,
                "sink": sink,
            })
            if not _valid_record(record):
                raise StorageError("worker sidecar failed its strict schema")
            if record["verdict"] == VERDICT_QUARANTINED:
                return None
            if existing is not None:
                # A new alias may safely precede the state update because it
                # already resolves to this visible exact-key record. If alias
                # persistence fails, the old record remains byte-for-byte.
                if arm_token:
                    alias_path = memo_path(arm_token, root)
                    with _record_lock(alias_path):
                        if not _note_memo_locked(alias_path, str(result.key)):
                            raise StorageError("arm memo was not durably persisted")
                        _write_json_atomic(path, record)
                else:
                    _write_json_atomic(path, record)
            else:
                # Stage the complete row under a non-selectable name. Alias
                # registration may fail or the process may die without ever
                # exposing it. The final hard-link is a no-replace commit: any
                # named record that raced us wins and is never overwritten.
                _write_json_atomic(staged_path, record)
                if record["arm_token"]:
                    alias_path = memo_path(str(record["arm_token"]), root)
                    with _record_lock(alias_path):
                        if not _note_memo_locked(alias_path, str(result.key)):
                            raise StorageError("arm memo was not durably persisted")
                        _install_staged_record(staged_path, path)
                else:
                    _install_staged_record(staged_path, path)
        if record["verdict"] == VERDICT_QUARANTINED:
            return None
        graph = _resolve_selected(
            key,
            record,
            engine,
            cas,
            sidecars_root(root) / ".resolved" / key,
        )
        return None if graph is None else _local(record, graph, engine, root)
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
    except (OSError, ValueError):
        return False
    return True


def lookup(key: str, root: Optional[Path] = None) -> Optional[LocalCompiledGraph]:
    record = _selected_record(key, frozenset({VERDICT_ADMITTED}), root)
    if record is None:
        return None
    try:
        cas = _cas(root)
        engine = Engine(cas)
        graph = _resolve_selected(
            key,
            record,
            engine,
            cas,
            sidecars_root(root) / ".resolved" / key,
        )
        return None if graph is None else _local(record, graph, engine, root)
    except (OSError, QuarantinedArtifact, StorageError, ValueError):
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

    record = _selected_record(
        key,
        frozenset({VERDICT_UNVERIFIED, VERDICT_ADMITTED}),
        root,
    )
    if record is None:
        return None
    try:
        cas = _cas(root)
        engine = Engine(cas)
        graph = _resolve_selected(
            key,
            record,
            engine,
            cas,
            sidecars_root(root) / ".resolved" / key,
        )
        if graph is None:
            return None
        runner_destination = sidecars_root(root) / ".runners" / key
        runner = engine.runner(key, runner_destination)
        if runner is None or runner.key != key or graph.key != key:
            return None
        return LoadedCompiledGraph(graph, runner)
    except (OSError, QuarantinedArtifact, StorageError, ValueError):
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
        directory = sidecar_dir(key, root)
        record = _read_record(directory / RECORD_NAME)
        if record is None:
            record = _read_record(directory / _PENDING_RECORD_NAME)
        if (
            record is None
            or record["compiled_graph_key"] != key
            or record["verdict"] == VERDICT_QUARANTINED
        ):
            return False
        path = memo_path(arm_token, root)
        with _record_lock(path):
            return _note_memo_locked(path, key)
    except (OSError, ValueError):
        return False


def lookup_for_arm(
    arm_token: str, root: Optional[Path] = None,
) -> Optional[LocalCompiledGraph]:
    try:
        value = _read_memo(memo_path(arm_token, root))
    except (OSError, ValueError):
        return None
    if value is None:
        return None
    return lookup(value["compiled_graph_key"], root)


def sweep_superseded_memos(scheme: str, root: Optional[Path] = None) -> int:
    current = str(scheme or "").strip() + "-"
    directory = sidecars_root(root) / MEMO_DIRNAME
    if len(current) < 2 or not directory.is_dir():
        return 0
    removed = 0
    with _record_lock(directory / ".sweep"):
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
    for path in directory.iterdir():
        if not path.is_dir() or path.name.startswith("."):
            continue
        try:
            record = _read_record(path / RECORD_NAME)
        except (OSError, ValueError):
            continue
        if record is not None and record["compiled_graph_key"] == path.name:
            return True
    return False


def graphs_owed_to_sink(root: Optional[Path] = None) -> list[OwedCompiledGraph]:
    directory = sidecars_root(root)
    if not directory.is_dir():
        return []
    rows: list[OwedCompiledGraph] = []
    for path in sorted(directory.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        try:
            record = _read_record(path / RECORD_NAME)
        except (OSError, ValueError):
            continue
        if (
            record is None
            or record["compiled_graph_key"] != path.name
            or record["verdict"] != VERDICT_ADMITTED
            or record["sink"] != SINK_OWED
        ):
            continue
        rows.append(OwedCompiledGraph(
            compiled_graph_key=str(record["compiled_graph_key"]),
            content_digest=str(record["content_digest"]),
            family=str(record["family"]),
            arm_token=str(record["arm_token"]),
            bytes=int(record["bytes"]),
        ))
    return rows


def note_refusal(code: str, detail: str = "", root: Optional[Path] = None) -> bool:
    if str(code or "").strip() != UNTRUSTED_REFUSAL_CODE:
        return False
    try:
        path = sidecars_root(root) / TRUST_CLASS_NAME
        with _record_lock(path):
            if _read_trust(path) is not None:
                return True
            record = {
                "format": _FORMAT,
                "class": TRUST_UNTRUSTED,
                "code": UNTRUSTED_REFUSAL_CODE,
                "detail": str(detail or "")[:500],
                "learned_at": time.time(),
            }
            if not _valid_trust(record):
                return False
            _write_json_atomic(path, record)
        return True
    except (OSError, ValueError):
        return False


def trust_class(root: Optional[Path] = None) -> str:
    try:
        value = _read_trust(sidecars_root(root) / TRUST_CLASS_NAME)
    except (OSError, ValueError):
        return ""
    return TRUST_UNTRUSTED if value is not None else ""


def keeps_graphs_locally(root: Optional[Path] = None) -> bool:
    return trust_class(root) == TRUST_UNTRUSTED
