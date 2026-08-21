"""Tensorhub policy for retaining an expensive produced tree."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

JOURNAL_NAME = "publish-journal.json"
STATE_NAME = "publish-state.json"
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProducedTree:
    session_id: str
    paths: tuple[str, ...]
    producer_state: dict[str, Any]

    def declares(self, paths: Sequence[str]) -> bool:
        return tuple(sorted(map(str, paths))) == tuple(sorted(self.paths))


class ProducerRecovery:
    """Durable producer metadata keyed by tensorfs transfer-session id."""

    def __init__(self, journal_path: str | Path) -> None:
        path = Path(journal_path)
        self.path = path.with_name(STATE_NAME)
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")

    @contextmanager
    def _lock(self, *, exclusive: bool) -> Iterator[BinaryIO]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(
                lock.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            )
            yield lock

    def _read_unlocked(self) -> dict[str, ProducedTree]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_bytes())
        except (OSError, ValueError):
            return {}
        if not isinstance(raw, dict) or raw.get("format") != 1:
            return {}
        result: dict[str, ProducedTree] = {}
        for item in raw.get("entries") or ():
            if not isinstance(item, dict):
                continue
            session_id = str(item.get("session_id") or "").strip()
            paths = item.get("paths")
            state = item.get("producer_state")
            if not session_id or not isinstance(paths, list) or not isinstance(state, dict):
                continue
            result[session_id] = ProducedTree(
                session_id,
                tuple(str(path) for path in paths),
                dict(state),
            )
        return result

    def _write_unlocked(self, entries: Mapping[str, ProducedTree]) -> None:
        payload = json.dumps(
            {
                "format": 1,
                "entries": [
                    {
                        "session_id": entry.session_id,
                        "paths": list(entry.paths),
                        "producer_state": entry.producer_state,
                    }
                    for entry in sorted(entries.values(), key=lambda value: value.session_id)
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, raw_path = tempfile.mkstemp(
                prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
            )
            temporary = Path(raw_path)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
                directory = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        except OSError:
            logger.warning("could not write producer recovery %s", self.path)

    def record(
        self,
        session_id: str,
        *,
        paths: Sequence[str],
        producer_state: Mapping[str, Any],
    ) -> None:
        with self._lock(exclusive=True):
            entries = self._read_unlocked()
            entries[session_id] = ProducedTree(
                session_id,
                tuple(map(str, paths)),
                dict(producer_state),
            )
            self._write_unlocked(entries)

    def clear(self, session_id: str) -> None:
        with self._lock(exclusive=True):
            entries = self._read_unlocked()
            if entries.pop(session_id, None) is not None:
                self._write_unlocked(entries)

    def find(self, **producer_state: Any) -> ProducedTree | None:
        with self._lock(exclusive=False):
            for entry in self._read_unlocked().values():
                if all(
                    entry.producer_state.get(key) == value
                    for key, value in producer_state.items()
                ):
                    return entry
        return None

    def count(self) -> int:
        with self._lock(exclusive=False):
            return len(self._read_unlocked())


__all__ = ["JOURNAL_NAME", "ProducerRecovery", "ProducedTree"]
