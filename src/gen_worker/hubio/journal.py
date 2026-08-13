"""The publish journal — "never redo the cast to redo the upload".

A multi-hour fp8 cast on an H100 costs GPU dollars plus the wall time. If the
upload of its tens of GB of output fails, the honest cost is "re-upload" — which
needs two things the naive shape lacks: a ``publish_id`` that outlives
``publish_v2``'s stack frame, so a successor can name the session whose staged
bytes it should resume, and a produced tree that is not rmtree'd on the failure
path.

**What this is.** A tiny append-and-rewrite JSON file living NEXT TO the
produced tree, recording for each in-flight publish: the ``publish_id``, the
destination repo and mode, the identity of the artifact being published
(``artifact_key`` — the sha256 of the sorted declared object digests), and the
declared object digests themselves. Written durably (temp + fsync + rename),
cleared on success, and LEFT BEHIND on failure.

**What it buys, and only in combination with the hub.** The staging prefix is
session-scoped (``staging/sha256/<publish_id>/``), so resuming means reusing
the id — which is exactly what the journal restores. A successor re-plans that
same session through ``POST .../publishes/:id/grants`` and uploads only what
the need set still names. Note the dependency, honestly: while the hub's
``Plan`` decides residency off ``blobs/`` alone, the need set does not shrink
for staged-but-unpromoted objects, so re-planning re-uploads the tree. The
client is built to exploit the shrink the moment it exists, and costs nothing
extra before then — one extra ``/grants`` round trip.

**Scope, stated rather than implied.** This is a SAME-POD durability record.
The retained tree and the journal live on pod-local disk; if the pod dies the
disk dies with it and neither survives. It covers a transport blip, a hub
restart, a completion timeout, an orchestrator requeue onto the same worker —
and deliberately NOT cross-pod resume, which needs the id to travel through the
hub. What it must NOT do is let a session's staged bytes be destroyed by a
failure a retry could have survived; that half is enforced in ``hub.py``, which
aborts only on a terminal repudiation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..models.cozy_cas import fsync_dir, fsync_file

logger = logging.getLogger(__name__)

JOURNAL_NAME = "publish-journal.json"

__all__ = [
    "JOURNAL_NAME",
    "JournalEntry",
    "PublishJournal",
    "artifact_key",
]


def artifact_key(digests: Sequence[str]) -> str:
    """Stable identity of an artifact: the sha256 of its sorted object set.

    A re-run that produced the same bytes declares the same objects and
    therefore matches its predecessor's session. A re-run that produced
    DIFFERENT bytes does not, and gets a fresh declare — the journal can never
    splice one artifact's staging into another's publish.
    """
    joined = "\n".join(sorted({str(d).strip().lower() for d in digests if d}))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class JournalEntry:
    publish_id: str
    destination_repo: str
    mode: str
    artifact_key: str
    objects: int = 0
    bytes_declared: int = 0
    source_root: str = ""
    #: The repo-relative paths this publish declared. A retained tree that no
    #: longer yields exactly this set is not the artifact the session names.
    paths: tuple = ()
    #: Opaque to this module: whatever the PRODUCER needs to recognize its own
    #: output without recomputing it. The journal owns durability; the producer
    #: owns meaning (clone.py stores the spec label, the tree path and the
    #: flavor attrs so it can skip a re-cast).
    producer_state: Dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> Dict[str, Any]:
        return {
            "publish_id": self.publish_id,
            "destination_repo": self.destination_repo,
            "mode": self.mode,
            "artifact_key": self.artifact_key,
            "objects": self.objects,
            "bytes_declared": self.bytes_declared,
            "source_root": self.source_root,
            "paths": list(self.paths),
            "producer_state": dict(self.producer_state),
        }

    @classmethod
    def from_wire(cls, raw: Any) -> "Optional[JournalEntry]":
        if not isinstance(raw, dict):
            return None
        pid = str(raw.get("publish_id") or "").strip()
        key = str(raw.get("artifact_key") or "").strip()
        if not pid or not key:
            return None
        state = raw.get("producer_state")
        return cls(
            publish_id=pid,
            destination_repo=str(raw.get("destination_repo") or ""),
            mode=str(raw.get("mode") or ""),
            artifact_key=key,
            objects=int(raw.get("objects") or 0),
            bytes_declared=int(raw.get("bytes_declared") or 0),
            source_root=str(raw.get("source_root") or ""),
            paths=tuple(str(p) for p in (raw.get("paths") or ())),
            producer_state=dict(state) if isinstance(state, dict) else {},
        )

    def declares(self, paths: Sequence[str]) -> bool:
        """Does ``paths`` name exactly the file set this entry declared?

        The cheap half of "is the retained tree still the artifact this session
        is about". The expensive half is free: ``publish_v2`` re-hashes every
        file at declare and adopts the session only when the resulting
        ``artifact_key`` matches, so a same-shape-but-corrupted tree gets a
        fresh declare rather than a wrong resume.
        """
        return tuple(sorted(str(p) for p in paths)) == tuple(sorted(self.paths))


@dataclass
class PublishJournal:
    """Durable record of the publishes in flight beside one produced tree.

    Every mutation is a full durable rewrite: the file holds at most a handful
    of entries (one per flavor of one conversion job), and a torn journal is
    worse than a slightly slower one. Nothing here may raise into a publish —
    a journal that cannot be written costs a resume, never the job.
    """

    path: Path
    entries: List[JournalEntry] = field(default_factory=list)

    @classmethod
    def open(cls, path: Path | str) -> "PublishJournal":
        p = Path(path)
        journal = cls(path=p)
        try:
            raw = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except (OSError, ValueError):
            logger.warning("publish journal at %s is unreadable; starting fresh", p)
            return journal
        for item in (raw.get("entries") if isinstance(raw, dict) else None) or []:
            entry = JournalEntry.from_wire(item)
            if entry is not None:
                journal.entries.append(entry)
        return journal

    def find(self, *, destination_repo: str, mode: str, key: str) -> Optional[JournalEntry]:
        """The session a re-run of THIS publish should try to resume."""
        for entry in self.entries:
            if (entry.artifact_key == key
                    and entry.destination_repo == destination_repo
                    and entry.mode == mode):
                return entry
        return None

    def for_producer(self, **match: Any) -> Optional[JournalEntry]:
        """The in-flight entry whose ``producer_state`` carries these facts."""
        for entry in self.entries:
            if all(entry.producer_state.get(k) == v for k, v in match.items()):
                return entry
        return None

    def record(self, entry: JournalEntry) -> None:
        self.entries = [
            e for e in self.entries
            if not (e.destination_repo == entry.destination_repo
                    and e.mode == entry.mode
                    and e.artifact_key == entry.artifact_key)
        ]
        self.entries.append(entry)
        self._flush()

    def clear(self, publish_id: str) -> None:
        """Drop one session — called when it PROMOTED. What remains in the file
        is exactly the set a successor should try to resume."""
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.publish_id != publish_id]
        if len(self.entries) != before:
            self._flush()

    def _flush(self) -> None:
        payload = json.dumps(
            {"version": 1, "entries": [e.to_wire() for e in self.entries]},
            separators=(",", ":"),
        ).encode("utf-8")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
            tmp = Path(tmp_name)
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(payload)
                fsync_file(tmp)
                os.replace(tmp, self.path)
                fsync_dir(self.path.parent)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        except OSError:
            # A journal is a recovery affordance, never a precondition.
            logger.warning("could not write publish journal %s", self.path, exc_info=True)
