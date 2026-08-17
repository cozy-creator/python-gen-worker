"""pgw#1096 / §4.28 — this machine's OWN cells: TCG holds the bytes, this holds the POLICY.

DESIGN-RULINGS §4.28 (Paul, 2026-08-10): *"Untrusted hardware (community
cloud, cozy-local) mints for ITSELF: local cell, local repo-CAS, reused
across its own boots — never uploaded, never requested."* And the UX
elaboration: *"download model + code ONCE, compile ONCE, and every
subsequent run of that code reuses the same compiled cell — same ck1 key
derivation, same memo shortcut, fully offline-capable."*

pgw#1283 — THE SPLIT. This module used to keep a SECOND copy of bytes TCG had
already admitted to its CAS: ``aot_compile_child`` packs with
``Engine.export_artifact``, and this module then hashed that file, copied it to
``<root>/aot-cells/<key>/cell.tar.gz``, and re-hashed it on every lookup. Three
digest passes over a multi-hundred-megabyte artifact, all of them re-deriving
what the CAS derived when it ingested the bytes. That duplication is deleted.
The two concerns that were tangled in it are now stated separately:

**EXISTENCE is TCG's.** :func:`store` hands the artifact to
``Engine.import_artifact``, which unpacks it, checks that its metadata restates
the key, admits its host ISA, and files it under the exact-key ref.
:func:`lookup` materializes it back through ``Engine.export_artifact``, which
verifies the CAS record and the envelope before it hands over a path. This
module contains no ``sha256`` and copies no artifact.

**PERMISSION is this module's.** A cell's ``verdict`` and its ``sink``
obligation are worker policy — facts about a gate this worker ran and an upload
this worker owes — and TCG has no concept of either. They live in a small
sidecar record beside the export, and nothing else here is load-bearing.

The rule that falls out, and the one to keep: **bytes are the authority on
whether a cell EXISTS; the sidecar is the authority on whether it may be
SERVED.** So a record with no bytes is a clean miss, bytes with no record are a
clean miss, and both heal on the next :func:`store` of the same mint rather
than needing a repair pass.

TWO QUARANTINES, DELIBERATELY NOT ONE (pgw#1283 criterion 4). TCG quarantines
a key because its CAS record failed verification; this module quarantines a
cell because a parity/arm gate refused it. Collapsing them was the reviewed
NO-SHIP defect in both directions: a TCG CAS repair would leave worker state
stuck at :data:`VERDICT_QUARANTINED` forever, and — the sharper half, created
by this very cutover — a worker-quarantined cell's bytes now live in the CAS
that ``aot_serve.arm_compiled_graph`` loads runners from, so the load path
would happily arm a cell this worker refused. :func:`is_quarantined` is the
seam that stops it, and ``aot_serve`` asks it before it resolves.

TRUST BOUNDARY — inherited verbatim from the JIT-era store this replaces
(``local_cells``). A compile cell is user-generated EXECUTABLE code
(compiled kernels + generated C++/Triton sources); accepting one from a user
machine into shared storage would let any user ship arbitrary code into other
people's GPU workers. Enforcement is STRUCTURAL, not a flag: this module has
no publish path — it imports no upload/transport machinery and writes only
under the store root and the worker's one CAS. ``tests/
test_aot_local_mint_pgw1096.py`` pins that structurally, the way
``test_local_cells`` pins it for the module this succeeds.

WHO DECIDES a machine is untrusted: **the hub, and only the hub**. There is no
worker-side self-declaration and no env flag — §1.18/§4.28. The class is
LEARNED from the hub's own typed publish refusal (``compiled_graph_publish_untrusted_tier``,
403, ``tensorhub internal/orchestrator/http/worker_cell_publish.go``) and
persisted here so the next boot does not have to pay a mint to rediscover it.

pgw#1183 / §1.5 — THE DURABILITY ORDERING IS UNCHANGED by the split: **pack ->
here -> verify -> arm -> publish**, on every tier. Two facts ride the sidecar
record and carry it:

* :data:`VERDICT_UNVERIFIED` -> :data:`VERDICT_ADMITTED` / :data:`VERDICT_QUARANTINED`
  — a cell is durable BEFORE it is proven, so only an ADMITTED one may be
  armed by a later boot. A crash between the two costs one re-compile; a
  quarantined one is kept for forensics and never served.
* :data:`SINK_OWED` -> :data:`SINK_DELIVERED` / :data:`SINK_REFUSED`
  — the upload obligation, recorded beside the bytes rather than held in a
  thread, so it survives process death and pod retirement
  (:func:`cells_owed_to_sink` is what the next boot reads).

LAYOUT (one directory per cell, so a cell and the facts about it move as a
unit)::

    <root>/aot-cells/<ck1-…>/record.json   the POLICY sidecar {verdict, sink, …}
    <root>/aot-cells/<ck1-…>/cell.tar.gz   TCG's export — a MATERIALIZATION, not custody
    <root>/aot-cells/.memo/<arm1-…>.json   the MEMO: pre-trace identity -> ck1 key
    <root>/trust-class.json                the learned trust class

The layout is preserved on purpose: cozy-local's ``cozy cells`` CLI and
tensorhub's runtime-env contract both name it, and the cutover changes byte
CUSTODY, not the cross-repo path contract.

THE MEMO, and why a content-addressed store needs one. The ck1 key's ``graph``
axis is the traced-graph digest, which does not exist until an export
finishes — so a boot that has not traced cannot address the CAS directly.
What it CAN state in milliseconds is ``fleet_cells.ArmIdentity``: every
pre-trace-knowable fact (family, format, lane, sm, envelope, env_seal,
toolchain). The memo maps that token to the ck1 key the last mint of it
produced. It is a SHORTCUT, never an authority: the artifact it points at is
verified by TCG and then passes the identical arm gate a child-minted cell
passes, so a wrong memo can only cost one re-mint.

EVICTION: there is none, deliberately (pre-launch). What accumulates is one
cell per (graph x sm x toolchain). :func:`stored_cells` enumerates the sidecar
records with no digest recomputation and no materialization, so a future sweep
has a fact base that costs a listdir.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from gen_worker._vendor.torchcg import ARTIFACT_KIND, is_compiled_graph_key
from gen_worker._vendor.torchcg.identity import KEY_SCHEME

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gen_worker._vendor.torchcg import Engine

logger = logging.getLogger(__name__)

#: A path relocation, never a behavior knob (§1.18, and the same rule
#: ``TENSORHUB_CACHE_DIR`` follows). cozy-local exports it from its own
#: ``workerEnv`` (``cozy-local/internal/paths/paths.go``) and the ``cozy
#: cells`` CLI reads the same root, so the NAME and the DEFAULT are a
#: cross-repo contract: changing either goes dark on that CLI.
ENV_STORE_DIR = "GEN_WORKER_LOCAL_CELLS_DIR"

CELLS_DIRNAME = "aot-cells"
MEMO_DIRNAME = ".memo"
ARTIFACT_NAME = "cell.tar.gz"
RECORD_NAME = "record.json"
TRUST_CLASS_NAME = "trust-class.json"

#: The hub's typed 403 that ASSERTS this machine may not publish. One string,
#: from ``tensorhub``'s ``worker_cell_publish.go``; a refusal the worker does
#: not recognize is left unlearned rather than guessed at, because guessing
#: "untrusted" from an unrelated failure would make a transient hub error
#: permanently change how this pod behaves.
UNTRUSTED_REFUSAL_CODE = "compiled_graph_publish_untrusted_tier"

TRUST_UNTRUSTED = "untrusted"

#: pgw#1183: whether this cell has passed the gate that lets it SERVE. The
#: store is written before the gate runs (§1.5), so "durable" and "provable"
#: are two facts and the sidecar records both.
VERDICT_UNVERIFIED = "unverified"
VERDICT_ADMITTED = "admitted"
VERDICT_QUARANTINED = "quarantined"

#: pgw#1183: this cell's standing with the fleet SINK, durable beside the
#: bytes. A transfer that dies mid-flight leaves :data:`SINK_OWED`, which the
#: next boot re-attempts from these bytes — the property that makes a failed
#: transfer cost a retry rather than the mint. :data:`SINK_NONE` is the honest
#: state for a machine with no sink by design (cozy-local, §4.28): nothing is
#: owed, so nothing accumulates.
#:
#: The vocabulary is deliberately the SINK's and not the transfer's. This
#: module's trust boundary is enforced by the ABSENCE of upload machinery, and
#: ``test_the_local_store_has_no_publish_path`` pins that by refusing the word
#: in any identifier here — correctly: the store records an obligation, it
#: does not know how one is discharged.
SINK_NONE = "none"
SINK_OWED = "owed"
SINK_DELIVERED = "delivered"
SINK_REFUSED = "refused"


def store_root() -> Path:
    """The policy sidecar root: the stated env path, else the cozy cache dir."""
    env = os.environ.get(ENV_STORE_DIR, "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "cozy" / "compile-cells"


def cells_root(root: Optional[Path] = None) -> Path:
    return (root or store_root()) / CELLS_DIRNAME


def cell_dir(key: str, root: Optional[Path] = None) -> Path:
    """The directory holding the sidecar for the cell keyed ``key``.

    Refuses anything that is not a ``ck1`` key rather than sanitizing it: a
    store addressed by a key-shaped string it did not verify is a store whose
    layout depends on what a caller happened to pass.
    """
    if not is_compiled_graph_key(key):
        raise ValueError(
            f"the local cell store addresses {KEY_SCHEME} keys only; "
            f"{key!r} is not one")
    return cells_root(root) / key


def memo_path(arm_token: str, root: Optional[Path] = None) -> Path:
    token = str(arm_token or "").strip()
    if not token or "/" in token or token.startswith("."):
        raise ValueError(f"not an arm-identity token: {arm_token!r}")
    return cells_root(root) / MEMO_DIRNAME / f"{token}.json"


def _engine(cas_root: Optional[Path] = None) -> "Engine":
    """TCG on the worker's ONE CAS.

    The import is function-local and goes through ``cache_paths``, which is the
    single sanctioned ``LocalCAS`` construction seam (pgw#1277): a second one
    would be a second compiled-graph store, which is the thing this cutover
    exists to remove.
    """
    from .models.cache_paths import open_worker_engine

    return open_worker_engine(cas_root)


@dataclass(frozen=True)
class CellRecord:
    """The POLICY facts this worker recorded about one cell. No bytes.

    What :func:`stored_cells` yields: a listdir and a small JSON read per
    entry, with nothing materialized and nothing hashed. pgw#1283 criterion 5
    — an owed-upload scan must cost a scan.
    """

    key: str
    family: str
    arm_token: str
    bytes: int
    stored_at: float
    #: One of the ``VERDICT_*`` constants.
    verdict: str = VERDICT_ADMITTED
    #: One of the ``SINK_*`` constants.
    sink: str = SINK_NONE


@dataclass(frozen=True)
class LocalCell:
    """A recorded cell whose bytes TCG has verified and materialized."""

    key: str
    artifact: Path
    family: str
    arm_token: str
    bytes: int
    stored_at: float
    verdict: str = VERDICT_ADMITTED
    sink: str = SINK_NONE


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Replace ``path`` with ``payload``, atomically and collision-safely.

    pgw#1283 criterion 6. The temp name used to be keyed on the pid alone,
    which is not a distinct name for two THREADS of one process — and the memo
    is written from the mint path and from the arm path, which do run
    concurrently. Two writers then shared one temp file and could publish a
    torn interleaving of both payloads under a name that says it is atomic.
    The random suffix makes the name distinct per WRITE, not per process, and
    the ``os.replace`` makes the publication atomic; the fsyncs make it
    survive the crash the ordering is designed around.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}")
    try:
        with open(tmp, "w") as handle:
            handle.write(json.dumps(payload, sort_keys=True, indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


class RecordUnreadable(Exception):
    """A store file EXISTS but its bytes are not a usable record.

    pgw#1283. Absence and corruption used to collapse into one ``None`` here,
    and every caller then read that ``None`` as "nothing was ever stored". They
    are not the same fact and they do not have the same cost: absence is the
    normal empty-store case, while corruption means a file this store wrote is
    no longer the file it wrote. Each caller now decides for itself, and every
    one of them says so in the log — a store that silently re-mints is a store
    that quietly bills a GPU pod for a defect nobody can see.

    Raising is safe precisely because every call site catches it: the module's
    rule that a cache miss must never be fatal is unchanged.
    """


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """The record at ``path``, ``None`` if it is genuinely ABSENT.

    Raises :class:`RecordUnreadable` when something IS there and cannot be
    used — unparseable bytes, a non-object payload, or an ``OSError`` that is
    not "no such file" (a directory in the record's place, or a permission
    failure: the file exists, we simply cannot read it).
    """
    try:
        text = path.read_text()
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as exc:
        raise RecordUnreadable(f"{path} exists but is unreadable: {exc}") from exc
    try:
        loaded = json.loads(text)
    except ValueError as exc:
        raise RecordUnreadable(f"{path} is not JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise RecordUnreadable(f"{path} holds {type(loaded).__name__}, not an object")
    return loaded


def _record_of(key: str, raw: Dict[str, Any]) -> CellRecord:
    return CellRecord(
        key=key,
        family=str(raw.get("family") or ""),
        arm_token=str(raw.get("arm_token") or ""),
        bytes=int(raw.get("bytes") or 0),
        stored_at=float(raw.get("stored_at") or 0.0),
        verdict=str(raw.get("verdict") or VERDICT_ADMITTED),
        sink=str(raw.get("sink") or SINK_NONE),
    )


def _record_payload(record: CellRecord) -> Dict[str, Any]:
    return {
        "compiled_graph_key": record.key,
        "family": record.family,
        "arm_token": record.arm_token,
        "bytes": record.bytes,
        "stored_at": record.stored_at,
        # TCG owns this string; pgw#1010/pgw#1059: an exported cell is the only
        # kind with an identity, so it is the only kind this store can hold.
        "kind": ARTIFACT_KIND,
        "verdict": record.verdict,
        "sink": record.sink,
    }


def read_record(key: str, root: Optional[Path] = None) -> Optional[CellRecord]:
    """This worker's recorded policy for ``key``, without touching the bytes.

    ``None`` when nothing was ever recorded. Raises :class:`RecordUnreadable`
    when a record file exists and cannot be used, so each caller decides.
    """
    try:
        target = cell_dir(key, root)
    except ValueError:
        return None
    raw = _read_json(target / RECORD_NAME)
    return None if raw is None else _record_of(key, raw)


def verdict_of(key: str, root: Optional[Path] = None) -> str:
    """``key``'s recorded verdict, or ``""`` when this worker recorded none.

    ``""`` is NOT a verdict: it is "this worker has no policy for these bytes",
    which is the honest answer for a cell that arrived by any route other than
    this machine's own mint. Callers that gate on it must gate on the
    QUARANTINED value they refuse, never on the absence of an admission.
    """
    try:
        record = read_record(key, root)
    except RecordUnreadable as exc:
        # An unreadable record cannot ASSERT a quarantine, and inventing one
        # would strand a cell no gate ever refused. It is loud instead.
        logger.error(
            "local-cell-store: %s has an unreadable policy record (%s); this "
            "worker states no verdict for it", key, exc)
        return ""
    return "" if record is None else record.verdict


def is_quarantined(key: str, root: Optional[Path] = None) -> bool:
    """True when THIS WORKER refused ``key`` at a parity/arm gate (§1.3.4).

    pgw#1283 criterion 4, and the guard the cutover made necessary. Before it,
    a quarantined cell's bytes lived only here, so nothing else could reach
    them; now they are in the CAS ``aot_serve.arm_compiled_graph`` loads
    runners from, and a load path that does not ask this question arms a cell
    this worker already refused.

    Deliberately NOT true for a TCG storage quarantine. That is a statement
    about a CAS record, it is repairable by re-storing the bytes, and reading
    it as a worker refusal is exactly how a repair would leave a cell stranded
    forever.
    """
    return verdict_of(key, root) == VERDICT_QUARANTINED


def store(
    artifact: Path, *, key: str, family: str, arm_token: str = "",
    verdict: str = VERDICT_ADMITTED, sink: str = SINK_NONE,
    root: Optional[Path] = None, cas_root: Optional[Path] = None,
) -> Optional[CellRecord]:
    """Admit ``artifact`` to TCG under its STAMPED ``key`` and record the policy.

    TCG does the byte work: ``import_artifact`` unpacks the envelope, checks
    that its metadata restates ``key``, admits its host ISA, and files it under
    the exact-key ref. This module then writes the sidecar — the verdict and
    the sink obligation, which TCG has no concept of.

    ``verdict`` defaults to :data:`VERDICT_ADMITTED` — "the caller vouches for
    these bytes", which is what every route that finds an already-proven cell
    means. The MINT route (pgw#1183) writes :data:`VERDICT_UNVERIFIED` and
    promotes with :func:`mark` once its own gate passes, because §1.5 makes the
    store write happen BEFORE the proof.

    Returns the recorded cell, or ``None`` when the store failed: a local store
    is a cache, and failing to fill it must never take down a worker that is
    already serving compiled.

    pgw#1283 criterion 7 — **IDEMPOTENT, so a crash-retry loses no alias.**
    Every step runs on every call: a TCG ``PRESENT`` (the bytes were already
    there, from an earlier attempt that died before the sidecar) still rewrites
    the record and still rewrites the memo. Skipping them because "it is
    already stored" is the crash-retry alias loss — the bytes survive, the
    arm-token alias does not, and every later boot re-mints a cell it is
    holding. Which of TCG's three success outcomes came back is a fact for the
    log, never a branch that skips work.
    """
    from gen_worker._vendor.torchcg.storage import StoreOutcome

    try:
        target_dir = cell_dir(key, root)
        outcome = _engine(cas_root).import_artifact(key, artifact).outcome
        if outcome == StoreOutcome.DIVERGENT:
            # TCG already binds this exact key to DIFFERENT admitted bytes, and
            # it quarantined the newcomer rather than choosing between them.
            # Recording a sidecar now would file worker policy against bytes
            # nothing will resolve.
            logger.error(
                "local-cell-store: REFUSING %s — this machine's CAS already "
                "binds that key to different admitted bytes; the artifact just "
                "offered is quarantined by TCG and nothing is recorded here",
                key)
            return None
        record = CellRecord(
            key=key, family=str(family or ""), arm_token=str(arm_token or ""),
            bytes=Path(artifact).stat().st_size, stored_at=time.time(),
            verdict=str(verdict), sink=str(sink),
        )
        _write_json_atomic(target_dir / RECORD_NAME, _record_payload(record))
    except Exception as exc:  # noqa: BLE001 — a cache miss must never be fatal
        logger.warning(
            "local-cell-store: could not store %s (%s); this boot serves "
            "compiled anyway and the next one re-mints", key, exc)
        return None
    # pgw#1283 — EVERYTHING BELOW THIS LINE IS BEST-EFFORT. The bytes are in
    # the CAS and the record is durable, which means the cell IS stored and
    # `lookup` will find it; anything that fails from here must not be able to
    # turn that into a reported failure. The memo write used to sit inside the
    # `try` above, so a failure to write a shortcut returned ``None`` while the
    # cell sat on disk — `fleet_cells._stage_durable` then emitted
    # `local_compiled_graph_store_failed` for a cell that was, in fact, stored.
    if arm_token:
        note_memo(arm_token, key, root)
    logger.info(
        "local-cell-store: stored %s (%s, %.1f MB, tcg=%s) — this machine "
        "reuses it on every later boot with the same key, offline",
        key, record.family, record.bytes / 1e6, outcome.value)
    return record


def mark(
    key: str, *, verdict: Optional[str] = None, sink: Optional[str] = None,
    root: Optional[Path] = None,
) -> bool:
    """Move a stored cell's ``verdict`` and/or ``sink`` state (pgw#1183).

    The two transitions the mint flow makes after the bytes are already
    durable: unverified -> admitted/quarantined once the parity gate and the
    arm answer, and owed -> delivered/refused once the upload does. Never
    fatal: a record that cannot be rewritten leaves the previous state, which
    is the conservative one in both directions (an unpromoted cell is
    re-minted; an unresolved upload is re-attempted).
    """
    try:
        target_dir = cell_dir(key, root)
    except ValueError:
        return False
    try:
        raw = _read_json(target_dir / RECORD_NAME)
    except RecordUnreadable as exc:
        # pgw#1283: the transition is LOST, and that costs money in both
        # directions — a dropped ADMITTED re-mints, a dropped DELIVERED
        # re-uploads. It used to be indistinguishable from "no such cell".
        logger.error(
            "local-cell-store: LOSING a state transition for %s — verdict=%r "
            "sink=%r were not applied because its record is unreadable (%s). "
            "The previous state stands, which is conservative but not free",
            key, verdict, sink, exc)
        return False
    if raw is None:
        return False
    if verdict is not None:
        raw["verdict"] = str(verdict)
    if sink is not None:
        raw["sink"] = str(sink)
    try:
        _write_json_atomic(target_dir / RECORD_NAME, raw)
    except OSError as exc:
        logger.warning(
            "local-cell-store: could not re-record %s (%s)", key, exc)
        return False
    return True


def materialize(
    key: str, root: Optional[Path] = None, cas_root: Optional[Path] = None,
) -> Optional[Path]:
    """TCG's verified bytes for ``key``, at this store's stable path.

    ``Engine.export_artifact`` re-checks the CAS record and unpacks the
    envelope before it links the bytes into place, so the path this returns is
    one TCG vouched for on THIS call — the property the deleted
    ``content_digest`` re-hash was reaching for, done once by the layer that
    owns the bytes instead of a second time by the layer that does not.

    ``None`` on any miss, including a TCG storage quarantine: that is a
    statement about a CAS record and is repairable by re-storing the artifact,
    so it must NOT be written into this worker's verdict (pgw#1283
    criterion 4).
    """
    from gen_worker._vendor.torchcg.storage import QuarantinedArtifact, StorageError

    try:
        destination = cell_dir(key, root) / ARTIFACT_NAME
    except ValueError:
        return None
    engine = _engine(cas_root)
    # Two attempts, and the second one only ever runs after the first has
    # DELETED the thing that refused it — progress, not a retry budget.
    for discarded_stale_copy in (False, True):
        try:
            return engine.export_artifact(key, destination)
        except FileExistsError as exc:
            # TCG refuses to clobber an occupied destination that is not the
            # bytes it selected — correctly, because it cannot know whose file
            # that is. It is OURS: this path is a materialization of the CAS
            # record, not custody of it, so a rotted or half-written export is
            # discarded and re-materialized from bytes that are still intact.
            # Without this the refusal is TERMINAL — every later lookup
            # re-refuses the same stale file — and a cell the CAS is holding
            # perfectly would cost a re-mint on every boot.
            if discarded_stale_copy:
                logger.error(
                    "local-cell-store: %s cannot be materialized even after "
                    "its stale copy was discarded (%s)", key, exc)
                return None
            logger.warning(
                "local-cell-store: %s's materialized copy is not the artifact "
                "TCG selected (%s); discarding it and re-exporting from the "
                "CAS, which still holds the bytes", key, exc)
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                return None
        except QuarantinedArtifact as exc:
            logger.error(
                "local-cell-store: TCG holds %s QUARANTINED in its CAS (%s) — "
                "that is a fact about the stored record, not this worker's "
                "verdict, so the verdict is left exactly as it stands and "
                "re-storing the bytes repairs it", key, exc)
            return None
        except StorageError as exc:
            logger.debug("local-cell-store: TCG cannot serve %s (%s)", key, exc)
            return None
        except OSError as exc:
            logger.warning(
                "local-cell-store: could not materialize %s (%s)", key, exc)
            return None
    return None


def lookup(
    key: str, root: Optional[Path] = None, cas_root: Optional[Path] = None,
) -> Optional[LocalCell]:
    """The resident, ADMITTED cell for ``key``, with its bytes on disk.

    pgw#1183: a cell whose verdict is not :data:`VERDICT_ADMITTED` is absent
    HERE and enumerable elsewhere (:func:`quarantined_cells`,
    :func:`stored_cells`). Durability made the bytes exist before the proof
    did, so this is the seam that keeps "we kept it" from meaning "we serve
    it".

    The permission is checked BEFORE the bytes are materialized: an unverified
    or quarantined cell must not even be unpacked by the serving path, and
    exporting first would make the refusal cost an export.
    """
    try:
        record = read_record(key, root)
    except RecordUnreadable as exc:
        # pgw#1283. The bytes may well be intact in the CAS, but the permission
        # to serve them lived in this record, and a cell is user-generated
        # EXECUTABLE code (module docstring). So the answer is "absent" — what
        # changes is that it is now SAID, and that the unusable sidecar goes so
        # the next store can rewrite it.
        logger.error(
            "local-cell-store: DROPPING the policy record for %s — it is "
            "unreadable (%s); the bytes stay in the CAS, but nothing vouches "
            "for arming them, so this costs one honest re-mint rather than a "
            "silent one", key, exc)
        drop(key, root)
        return None
    if record is None or record.verdict != VERDICT_ADMITTED:
        return None
    artifact = materialize(key, root, cas_root)
    if artifact is None:
        return None
    # NOT a place to rebuild the memo, and a test caught the attempt (pgw#1283,
    # `test_an_arm_scheme_bump_costs_a_TRACE_and_not_a_MINT`). The record names
    # the token this cell was STORED for, which may belong to a superseded
    # arm-token scheme that :func:`sweep_superseded_memos` has deliberately
    # invalidated — rewriting it here resurrects dead vocabulary and silently
    # undoes the sweep. The alias is rebuilt by the route that just PROVED it,
    # in the CURRENT scheme's token: `fleet_cells.arm_from_local_store` calls
    # :func:`note_memo` on the boot-key route, which is the whole cost argument
    # of pgw#1127 §5. Crash-retry alias loss is closed in :func:`store`
    # instead, by making every step run on every call.
    return LocalCell(
        key=record.key, artifact=artifact, family=record.family,
        arm_token=record.arm_token, bytes=artifact.stat().st_size,
        stored_at=record.stored_at, verdict=record.verdict, sink=record.sink)


def note_memo(
    arm_token: str, key: str, root: Optional[Path] = None,
) -> bool:
    """Bind a pre-trace ``arm_token`` to a ``ck1`` key that PROVED it arms.

    pgw#1127. :func:`store` writes this at mint time; this writes it when the
    cell was found by another route and then passed the arm gate — which is the
    only other moment the binding is known to be true. It is what makes an
    arm-token scheme bump (:func:`sweep_superseded_memos`) cost a trace instead
    of a mint.

    NEVER LOAD-BEARING, and pgw#1283 keeps it that way deliberately. The write
    is now collision-safe and atomic (:func:`_write_json_atomic`), because a
    torn memo is a wrong answer rather than a missing one; but a memo that
    cannot be written costs a lookup and nothing else, so this reports the
    failure LOUDLY and lets the caller succeed. A store whose success depended
    on a shortcut would be a store that fails when a cache directory is full.
    """
    if not arm_token or not key:
        return False
    try:
        _write_json_atomic(
            memo_path(arm_token, root),
            {"compiled_graph_key": key, "noted_at": time.time()})
        return True
    except Exception as exc:  # noqa: BLE001 — a shortcut is never load-bearing
        logger.warning(
            "local-cell-store: could not memo %s -> %s (%s); the cell is "
            "stored and addressable by key, and the next boot pays a derive "
            "instead of a lookup", arm_token, key, exc)
        return False


def sweep_superseded_memos(
    scheme: str, root: Optional[Path] = None,
) -> int:
    """Delete every memo entry written under a SUPERSEDED token scheme.

    pgw#1113: the arm token's scheme digit is its fact-set schema, so when
    the fact set gains an axis (``arm1`` -> ``arm2``, which added the compile
    SUBJECT) every stored ``arm1-`` row becomes an answer to a question
    nothing asks any more. That is already safe — no reader can address them
    — but "safe because unreachable" is how a store accumulates files whose
    meaning nobody can reconstruct, so the invalidation is EXPLICIT and
    counted rather than implicit and silent.

    Only the memo is swept. The CELLS keep their ``ck1`` keys, which did not
    move: an arm-token change re-derives the shortcut, never the identity.
    Returns the number of entries removed.
    """
    prefix = str(scheme or "").strip() + "-"
    memo_dir = cells_root(root) / MEMO_DIRNAME
    if len(prefix) < 2 or not memo_dir.is_dir():
        return 0
    removed = 0
    for entry in sorted(memo_dir.glob("*.json")):
        if entry.name.startswith(prefix):
            continue
        try:
            entry.unlink()
            removed += 1
        except OSError:
            logger.debug("local-cell-store: could not drop %s", entry,
                         exc_info=True)
    if removed:
        logger.info(
            "local-cell-store: dropped %d memo entry/entries written under a "
            "superseded arm-token schema (current %s) — this machine re-mints "
            "each affected family ONCE and memoizes the answer again",
            removed, prefix.rstrip("-"))
    return removed


def lookup_for_arm(
    arm_token: str, root: Optional[Path] = None,
    cas_root: Optional[Path] = None,
) -> Optional[LocalCell]:
    """The cell this machine last minted for pre-trace identity ``arm_token``.

    The memo is a shortcut, never an authority (module docstring): the answer
    it names is verified by TCG in :func:`lookup` and then passes the same arm
    gate a freshly child-minted cell passes. A stale memo costs one re-mint.
    """
    try:
        memo = _read_json(memo_path(arm_token, root))
    except ValueError:
        return None
    except RecordUnreadable as exc:
        # pgw#1283. A memo is a shortcut, never an authority, so an unreadable
        # one is genuinely equivalent to no memo — but leaving the file there
        # means paying that read on every boot, and it is only rewritten once a
        # cell arms. Delete it so the shortcut is rebuilt from evidence rather
        # than shadowed by garbage.
        logger.warning(
            "local-cell-store: discarding an unreadable memo for %s (%s); the "
            "next cell that arms rewrites it", arm_token, exc)
        try:
            memo_path(arm_token, root).unlink()
        except OSError:
            pass
        return None
    if memo is None:
        return None
    key = str(memo.get("compiled_graph_key") or "")
    if not key:
        return None
    return lookup(key, root, cas_root)


def drop(key: str, root: Optional[Path] = None) -> None:
    """Forget this worker's POLICY for one cell. The CAS keeps its bytes.

    pgw#1283 — a deliberate narrowing. This used to delete the only copy of an
    artifact; the bytes now live in the content-addressed store, where they may
    be the same bytes another key or another route resolves, so a worker
    decision about ONE arm identity must not delete them. What this removes is
    the sidecar that grants permission to serve, and the materialized export
    beside it, which is exactly what "this cell is stale for us" means. A later
    :func:`store` of the same artifact re-records it against bytes TCG will
    report ``PRESENT``, at no mint cost.
    """
    try:
        target = cell_dir(key, root)
    except ValueError:
        return
    shutil.rmtree(target, ignore_errors=True)


def stored_cells(root: Optional[Path] = None) -> List[CellRecord]:
    """Every cell this worker recorded, cheaply — NO bytes, NO digests.

    The accounting surface for what accumulates (module docstring: one cell
    per graph x sm x toolchain, so every toolchain upgrade leaves its
    predecessor behind), and the scan :func:`cells_owed_to_sink` filters.
    pgw#1283 criterion 5: it reads sidecars and nothing else — it does not ask
    TCG to export, verify or hash anything, so listing costs a listdir even
    when every resident cell is a gigabyte.
    """
    out: List[CellRecord] = []
    base = cells_root(root)
    if not base.is_dir():
        return out
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or entry.name == MEMO_DIRNAME:
            continue
        try:
            raw = _read_json(entry / RECORD_NAME)
        except RecordUnreadable as exc:
            # pgw#1283: skip the entry, never the LISTING. This is what a
            # `cozy cells`-style CLI reads, and one bad file must not blank the
            # inventory — but a cell missing from the accounting surface is
            # exactly the kind of thing nobody notices, so it is logged.
            logger.warning(
                "local-cell-store: %s has an unreadable record (%s); it is "
                "absent from this listing", entry.name, exc)
            continue
        if raw is None:
            continue
        key = str(raw.get("compiled_graph_key") or "").strip()
        if not key:
            # A record REFUSES rather than borrowing the directory it sits in.
            # `_record_payload` always writes the key, so a sidecar without one
            # is not something this store wrote — and a fabricated identity is
            # worse than an absent one here, because `cells_owed_to_sink` reads
            # this listing and would hand the invented key to an upload.
            logger.warning(
                "local-cell-store: %s holds a record with no "
                "compiled_graph_key; it is absent from this listing",
                entry.name)
            continue
        out.append(_record_of(key, raw))
    return out


def quarantined_cells(root: Optional[Path] = None) -> List[CellRecord]:
    """Every cell kept for FORENSICS: it existed, and it failed OUR gate.

    §1.3.4 — *"do not arm, do not publish, keep the artifact quarantined-local
    for forensics"*. These are worker verdicts, never TCG's storage
    quarantine: a cell whose CAS record failed verification is not a cell a
    gate refused, and listing the two together is how a repair would look like
    a refusal (pgw#1283 criterion 4).
    """
    return [c for c in stored_cells(root) if c.verdict == VERDICT_QUARANTINED]


def cells_owed_to_sink(root: Optional[Path] = None) -> List[CellRecord]:
    """Every ADMITTED cell whose upload is still owed — the cross-boot retry.

    This is what makes publish a retryable background transfer from durable
    bytes rather than a daemon thread the pod must outlive. Only admitted
    cells are listed: an unverified or quarantined artifact is not something
    any other pod may adopt, so it is not something to upload. The caller
    materializes the ones it actually ships (:func:`materialize`), so an owed
    scan never pays for bytes it does not send.
    """
    return [c for c in stored_cells(root)
            if c.verdict == VERDICT_ADMITTED and c.sink == SINK_OWED]


# ---------------------------------------------------------------------------
# The learned trust class — hub-asserted, never self-declared
# ---------------------------------------------------------------------------


def note_refusal(code: str, detail: str = "", root: Optional[Path] = None) -> bool:
    """Learn this machine's trust class from the HUB's typed refusal.

    ``True`` when the refusal was the untrusted-tier one and the class is
    now recorded. Every other code — a forged axis, a quota, an unknown
    family, a transport failure — leaves the class untouched: those say
    something about the CELL or the moment, not about the hardware, and a
    worker that concluded "untrusted" from a 429 would permanently mis-file
    itself off one bad minute.
    """
    if str(code or "").strip() != UNTRUSTED_REFUSAL_CODE:
        return False
    try:
        _write_json_atomic((root or store_root()) / TRUST_CLASS_NAME, {
            "class": TRUST_UNTRUSTED,
            "code": UNTRUSTED_REFUSAL_CODE,
            "detail": str(detail or "")[:500],
            "learned_at": time.time(),
        })
    except OSError as exc:
        logger.warning(
            "local-cell-store: could not record the hub's untrusted-tier "
            "verdict (%s); this machine will re-learn it next boot", exc)
        return False
    logger.warning(
        "local-cell-store: the hub asserts this hardware may not publish "
        "(%s) — cells minted here are kept LOCALLY and reused on every later "
        "boot of this machine; nothing is uploaded and nothing is requested "
        "(§4.28)", UNTRUSTED_REFUSAL_CODE)
    return True


def trust_class(root: Optional[Path] = None) -> str:
    """``"untrusted"`` once the hub has said so, else ``""`` (not yet known).

    "Not yet known" is not "trusted": a machine that has never tried to
    publish has learned nothing, and the honest consequence is that its first
    mint attempts the publish and learns from the answer.
    """
    try:
        recorded = _read_json((root or store_root()) / TRUST_CLASS_NAME)
    except RecordUnreadable as exc:
        # pgw#1283: "not yet known" is the self-healing answer here — the next
        # mint attempts the publish and re-learns the class from the hub's own
        # refusal. That is one wasted attempt, not a wrong trust decision, so
        # it is a warning rather than a refusal.
        logger.warning(
            "local-cell-store: the trust class is unreadable (%s); treating "
            "this machine as not-yet-known, which re-learns from the hub",
            exc)
        return ""
    if recorded is None:
        return ""
    return str(recorded.get("class") or "")


def keeps_cells_locally(root: Optional[Path] = None) -> bool:
    """Whether a cell this machine mints should be kept in the local store.

    True exactly when the hub has ASSERTED this hardware may not publish. The
    cozy-local CLI serve path does not consult this — it has no publisher and
    never will, so it calls :func:`store` directly.
    """
    return trust_class(root) == TRUST_UNTRUSTED


__all__ = [
    "ARTIFACT_NAME",
    "CELLS_DIRNAME",
    "CellRecord",
    "ENV_STORE_DIR",
    "LocalCell",
    "MEMO_DIRNAME",
    "RECORD_NAME",
    "RecordUnreadable",
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
    "cell_dir",
    "cells_owed_to_sink",
    "cells_root",
    "drop",
    "is_quarantined",
    "keeps_cells_locally",
    "lookup",
    "lookup_for_arm",
    "mark",
    "materialize",
    "memo_path",
    "note_memo",
    "note_refusal",
    "quarantined_cells",
    "read_record",
    "store",
    "store_root",
    "stored_cells",
    "sweep_superseded_memos",
    "trust_class",
    "verdict_of",
]
