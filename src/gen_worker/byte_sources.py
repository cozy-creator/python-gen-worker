"""pgw#1632(b): every byte counter declares HOW it was measured.

`load:staged_bytes` sampled `/proc/self/io` **read_bytes** plus anonymous-RSS
growth — a READ meter wearing a WRITE verb. th#2246's first mechanism lane read
the name, went looking for a per-child write that does not exist, and spent a
pass on it. The number was never wrong; the name was, and a name is the only
thing a reader has.

So a byte counter is not just a name and a unit. It is a name, a unit, and the
INSTRUMENT it was read off. Once the instrument is declared, the admissible
verbs follow from it mechanically, and `tests/test_counter_source_lint_pgw1632.py`
enforces both halves: the verb must suit the source, and a counter that names no
source cannot be created at all.

The design review's ``NET_SOCKET`` is split by direction here. A socket meter is
a read instrument on the way in and a write instrument on the way out, so an
undirected socket source would make the lint vacuous for the network path —
which is the largest byte mover in the repo and the one most worth naming
correctly.

``CAS_ACCOUNTED`` is the honest source for the aggregate positions: they credit
bytes that are PRESENT for a manifest regardless of whether they arrived over a
socket, came off a volume, or were already banked. It admits no directional verb
at all, which is the correct outcome — `weight_position`'s own docstring already
warns that its position is "bytes accounted for the snapshot, not bytes off the
wire", and no read verb or write verb can say that truthfully.
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping


class Direction(Enum):
    """What class of verb a source can honestly wear."""

    READ = "read"
    WRITE = "write"
    #: Free/used space on a filesystem — neither a read nor a write.
    SPACE = "space"
    #: Bytes credited against a plan, from any of several instruments.
    ACCOUNTED = "accounted"


class Source(Enum):
    """The instrument a byte counter is sampled from."""

    #: `/proc/<pid>/io` read_bytes — bytes this process pulled off block devices.
    PROC_READ_IO = (Direction.READ,)
    #: `/proc/<pid>/io` write_bytes — bytes this process pushed to block devices.
    PROC_WRITE_IO = (Direction.WRITE,)
    #: Bytes arriving over a socket.
    NET_SOCKET_RECV = (Direction.READ,)
    #: Bytes leaving over a socket.
    NET_SOCKET_SEND = (Direction.WRITE,)
    #: A `statvfs` delta on a mount — space freed or consumed.
    STATVFS_DELTA = (Direction.SPACE,)
    #: Anonymous resident-set growth — an inference about staging, from memory.
    RSS_GROWTH = (Direction.READ,)
    #: Bytes present for a manifest, whatever instrument proved they are there.
    CAS_ACCOUNTED = (Direction.ACCOUNTED,)

    @property
    def direction(self) -> Direction:
        return self.value[0]


#: Verb -> the direction a counter wearing it must have been measured with.
#: Extend it freely; the cost of a new verb is one line, and the cost of an
#: unclassified one is th#2246's wasted pass.
VERB_DIRECTIONS: Mapping[str, Direction] = {
    "staged": Direction.WRITE,
    "written": Direction.WRITE,
    "flushed": Direction.WRITE,
    "uploaded": Direction.WRITE,
    "read": Direction.READ,
    "ingested": Direction.READ,
    "pulled": Direction.READ,
    "freed": Direction.SPACE,
    "consumed": Direction.SPACE,
}


#: THE REGISTRY. Every byte counter and byte position in `src/gen_worker`,
#: keyed by its literal name or — for counters keyed per subject — by the
#: literal PREFIX before the interpolation. The lint refuses any byte counter
#: whose name is not resolvable here, so a new one cannot be added unclassified.
BYTE_COUNTERS: Mapping[str, Source] = {
    # `/proc/self/io` read_bytes, with RSS growth as the alternate when the
    # load is page-cache warm. Both read-side; the name says `ingested` for
    # exactly that reason (pgw#1632's seed fix, was `staged`).
    "load:ingested_bytes": Source.PROC_READ_IO,
    # `_progress(done, total)` where `done` = resident + volume-filled +
    # fetched. Bytes accounted for the snapshot, from three instruments.
    "download:": Source.CAS_ACCOUNTED,
    # Bytes pushed through the presigned PUT.
    "upload:bytes": Source.NET_SOCKET_SEND,
}


#: Byte POSITIONS — the durable per-object aggregates that ride the event
#: stream rather than the beat. Keyed by qualified class name; each class also
#: carries the same value as a ``SOURCE`` attribute, and the lint asserts the
#: two agree so a site cannot drift from the registry.
BYTE_POSITIONS: Mapping[str, Source] = {
    # Same quantity as `download:` — bytes accounted for the snapshot, from
    # residency, volume fill and the wire together.
    "gen_worker.weight_position.FetchPosition": Source.CAS_ACCOUNTED,
}


def verbs_in(name: str) -> set[str]:
    """The classified verbs a counter name wears."""

    tokens = {
        token
        for token in "".join(c if c.isalnum() else " " for c in name).split()
    }
    return tokens & set(VERB_DIRECTIONS)


def source_of(name: str) -> Source | None:
    """The declared source for a counter name, resolving prefix keys."""

    if name in BYTE_COUNTERS:
        return BYTE_COUNTERS[name]
    for key, source in BYTE_COUNTERS.items():
        if key.endswith(":") and name.startswith(key):
            return source
    return None


def misclassified(name: str, source: Source) -> str:
    """Why ``name`` may not wear its verbs given ``source``, or ``""``."""

    for verb in sorted(verbs_in(name)):
        want = VERB_DIRECTIONS[verb]
        if source.direction is not want:
            return (
                f"{name!r} carries the {want.value}-side verb {verb!r} but is "
                f"measured with {source.name} ({source.direction.value}-side)"
            )
    return ""


__all__ = [
    "BYTE_COUNTERS",
    "BYTE_POSITIONS",
    "Direction",
    "Source",
    "VERB_DIRECTIONS",
    "misclassified",
    "source_of",
    "verbs_in",
]
