"""pgw#1096 / §4.28 — the untrusted machine's OWN cell store: AOT cells, ck1-keyed.

DESIGN-RULINGS §4.28 (Paul, 2026-08-10): *"Untrusted hardware (community
cloud, cozy-local) mints for ITSELF: local cell, local repo-CAS, reused
across its own boots — never uploaded, never requested."* And the UX
elaboration: *"download model + code ONCE, compile ONCE, and every
subsequent run of that code reuses the same compiled cell — same ck1 key
derivation, same memo shortcut, fully offline-capable."*

ONE IDENTITY, TWO STORES. A cell stored here is addressed by exactly the
key the hub store addresses it by — ``cell_key.from_exported_artifact_metadata``
stamped on the bytes (pgw#1059's four axes: graph x envelope x sm x
toolchain). The hub store and this one differ in their SINK, never in their
addressing, so a cell that later becomes publishable needs no re-keying and a
local hit is the same artifact a hub hit would have delivered.

TRUST BOUNDARY — inherited verbatim from the JIT-era store this replaces
(``local_cells``). A compile cell is user-generated EXECUTABLE code
(compiled kernels + generated C++/Triton sources); accepting one from a user
machine into shared storage would let any user ship arbitrary code into other
people's GPU workers. Enforcement is STRUCTURAL, not a flag: this module has
no publish path — it is not a CAS client, imports no upload/transport
machinery, and writes only under the store root. ``tests/
test_aot_local_mint_pgw1096.py`` pins that structurally, the way
``test_local_cells`` pins it for the module this succeeds.

WHO DECIDES a machine is untrusted: **the hub, and only the hub**. There is no
worker-side self-declaration and no env flag — §1.18/§4.28. The class is
LEARNED from the hub's own typed publish refusal (``cell_publish_untrusted_tier``,
403, ``tensorhub internal/orchestrator/http/worker_cell_publish.go``) and
persisted here so the next boot does not have to pay a mint to rediscover it.
Before this module existed that refusal was terminal in the worst way: th#1643
books it as SUNK — *"a sealed cell was produced and thrown away"* — and
``fleet_cells._publish_async``'s ``finally`` then rmtree'd the bytes.

LAYOUT (one directory per cell, so a cell and the facts about it move as a
unit and a partial write leaves nothing admissible)::

    <root>/aot-cells/<ck1-…>/cell.tar.gz   the packed artifact
    <root>/aot-cells/<ck1-…>/record.json   {cell_key, content_digest, family, …}
    <root>/aot-cells/.memo/<arm1-…>.json   the MEMO: pre-trace identity -> ck1 key
    <root>/trust-class.json                the learned trust class
    <root>/.mint-resume/…                  aot_resume's crash-only bank (pgw#848)

THE MEMO, and why a content-addressed store needs one. The ck1 key's ``graph``
axis is the traced-graph digest, which does not exist until an export
finishes — so a boot that has not traced cannot address the CAS directly.
What it CAN state in milliseconds is ``fleet_cells.ArmIdentity``: every
pre-trace-knowable fact (family, format, lane, sm, envelope, env_seal,
toolchain). The memo maps that token to the ck1 key the last mint of it
produced. It is a SHORTCUT, never an authority: the artifact it points at is
verified against its recorded digest and then passes the identical arm gate a
child-minted cell passes, so a wrong memo can only cost one re-mint. When
pgw#1089 lands boot-side trace-for-key the derived key addresses the CAS
directly and the memo stays exactly what §4.28 calls it — the shortcut.

EVICTION: there is none, deliberately (pre-launch). What accumulates is one
cell per (graph x envelope x sm x toolchain), so every torch upgrade, every
gen-worker upgrade and every settings change leaves the previous cell resident
forever. :func:`stored_cells` enumerates them with sizes so a future sweep has
a fact base; an age-based sweep would be wrong (a cell nobody booted for six
months is still exactly the cell the next boot wants).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import cell_key

logger = logging.getLogger(__name__)

#: Read size for the content digest. The store deliberately hashes with the
#: stdlib rather than importing the CAS client's helper: this module must not
#: reach into transport machinery even for a pure function, because "it only
#: imports it for the hash" is how a boundary enforced by ABSENCE stops being
#: enforced by absence.
_DIGEST_CHUNK_BYTES = 4 * 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(_DIGEST_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

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

#: The only artifact class this store holds. pgw#1010/pgw#1059: an exported
#: cell is the only kind with an identity, so it is the only kind that can be
#: addressed — a JIT capture has no key and nothing could ever look it up.
KIND = "aot-inductor"

#: The hub's typed 403 that ASSERTS this machine may not publish. One string,
#: from ``tensorhub``'s ``worker_cell_publish.go``; a refusal the worker does
#: not recognize is left unlearned rather than guessed at, because guessing
#: "untrusted" from an unrelated failure would make a transient hub error
#: permanently change how this pod behaves.
UNTRUSTED_REFUSAL_CODE = "cell_publish_untrusted_tier"

TRUST_UNTRUSTED = "untrusted"


def store_root() -> Path:
    """The local store root: the stated env path, else the cozy cache dir.

    Moved here from ``local_cells`` by pgw#1096 with its value UNCHANGED —
    same env name, same default — so ``aot_resume``'s crash-only bank and
    cozy-local's ``cozy cells`` CLI keep resolving to the identical
    directory. That move is what makes ``local_cells.py`` deletable by
    pgw#1086 wave 1 (pgw#1092 §4 landmine 1: the production resume bank was
    routed through a module the JIT demolition deletes).
    """
    env = os.environ.get(ENV_STORE_DIR, "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "cozy" / "compile-cells"


def cells_root(root: Optional[Path] = None) -> Path:
    return (root or store_root()) / CELLS_DIRNAME


def cell_dir(key: str, root: Optional[Path] = None) -> Path:
    """The directory holding the cell keyed ``key``.

    Refuses anything that is not a ``ck1`` key rather than sanitizing it: a
    store addressed by a key-shaped string it did not verify is a store whose
    layout depends on what a caller happened to pass.
    """
    if not cell_key.is_key(key):
        raise ValueError(
            f"the local cell store addresses {cell_key.KEY_SCHEME} keys only; "
            f"{key!r} is not one")
    return cells_root(root) / key


def memo_path(arm_token: str, root: Optional[Path] = None) -> Path:
    token = str(arm_token or "").strip()
    if not token or "/" in token or token.startswith("."):
        raise ValueError(f"not an arm-identity token: {arm_token!r}")
    return cells_root(root) / MEMO_DIRNAME / f"{token}.json"


@dataclass(frozen=True)
class LocalCell:
    """One cell resident in the local store, with the facts recorded beside it."""

    key: str
    artifact: Path
    content_digest: str  # "sha256:<hex>" of the packed artifact
    family: str
    arm_token: str
    bytes: int
    stored_at: float


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2))
    os.replace(tmp, path)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        loaded = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def store(
    artifact: Path, *, key: str, family: str, arm_token: str = "",
    root: Optional[Path] = None,
) -> Optional[LocalCell]:
    """Copy ``artifact`` into the store under its STAMPED ``key``.

    The artifact is written to a temp sibling and ``os.replace``d, and the
    record — which carries the digest every later lookup is checked against —
    is written LAST, so a crash mid-store leaves a directory with no record
    and the next lookup treats it as absent rather than as a short cell. Same
    ordering rule ``aot_resume`` banks entries under, for the same reason.

    Returns the stored cell, or ``None`` when the store failed: a local store
    is a cache, and failing to fill it must never take down a worker that is
    already serving compiled.
    """
    try:
        target_dir = cell_dir(key, root)
        target_dir.mkdir(parents=True, exist_ok=True)
        digest = "sha256:" + _sha256_file(Path(artifact))
        final = target_dir / ARTIFACT_NAME
        tmp = target_dir / f".{ARTIFACT_NAME}.tmp-{os.getpid()}"
        shutil.copy2(str(artifact), str(tmp))
        os.replace(tmp, final)
        record = LocalCell(
            key=key, artifact=final, content_digest=digest,
            family=str(family or ""), arm_token=str(arm_token or ""),
            bytes=final.stat().st_size, stored_at=time.time(),
        )
        _write_json_atomic(target_dir / RECORD_NAME, {
            "cell_key": record.key,
            "content_digest": record.content_digest,
            "family": record.family,
            "arm_token": record.arm_token,
            "bytes": record.bytes,
            "stored_at": record.stored_at,
            "kind": KIND,
        })
        if arm_token:
            _write_json_atomic(
                memo_path(arm_token, root),
                {"cell_key": key, "noted_at": record.stored_at})
        logger.info(
            "local-cell-store: stored %s (%s, %.1f MB) — this machine reuses "
            "it on every later boot with the same key, offline",
            key, record.family, record.bytes / 1e6)
        return record
    except Exception as exc:  # noqa: BLE001 — a cache miss must never be fatal
        logger.warning(
            "local-cell-store: could not store %s (%s); this boot serves "
            "compiled anyway and the next one re-mints", key, exc)
        return None


def lookup(key: str, root: Optional[Path] = None) -> Optional[LocalCell]:
    """The resident cell for ``key``, or ``None``.

    The recorded ``content_digest`` is RECOMPUTED over the bytes on disk, so a
    truncated, half-written or edited artifact refuses here instead of being
    handed to the arm — the ``aot_resume`` rule (*"a bank cannot vouch for
    itself"*) applied to the store. A refusing entry is dropped, which turns a
    corrupted cell into exactly one honest re-mint.
    """
    try:
        target_dir = cell_dir(key, root)
    except ValueError:
        return None
    record = _read_json(target_dir / RECORD_NAME)
    artifact = target_dir / ARTIFACT_NAME
    if record is None or not artifact.is_file():
        return None
    want = str(record.get("content_digest") or "")
    try:
        have = "sha256:" + _sha256_file(artifact)
    except OSError as exc:
        logger.warning("local-cell-store: %s is unreadable (%s)", key, exc)
        return None
    if not want or have != want:
        logger.error(
            "local-cell-store: DROPPING %s — the stored bytes digest %s, the "
            "record states %s; a cell that cannot vouch for its own content "
            "is never armed", key, have, want or "nothing")
        drop(key, root)
        return None
    return LocalCell(
        key=key, artifact=artifact, content_digest=have,
        family=str(record.get("family") or ""),
        arm_token=str(record.get("arm_token") or ""),
        bytes=int(record.get("bytes") or artifact.stat().st_size),
        stored_at=float(record.get("stored_at") or 0.0),
    )


def lookup_for_arm(
    arm_token: str, root: Optional[Path] = None,
) -> Optional[LocalCell]:
    """The cell this machine last minted for pre-trace identity ``arm_token``.

    The memo is a shortcut, never an authority (module docstring): the answer
    it names is digest-verified by :func:`lookup` and then passes the same arm
    gate a freshly child-minted cell passes. A stale memo costs one re-mint.
    """
    try:
        memo = _read_json(memo_path(arm_token, root))
    except ValueError:
        return None
    if memo is None:
        return None
    key = str(memo.get("cell_key") or "")
    if not key:
        return None
    return lookup(key, root)


def drop(key: str, root: Optional[Path] = None) -> None:
    """Remove one cell and everything recorded about it."""
    try:
        shutil.rmtree(cell_dir(key, root), ignore_errors=True)
    except ValueError:
        return


def stored_cells(root: Optional[Path] = None) -> List[LocalCell]:
    """Every resident cell, cheaply — NO digest recomputation.

    The accounting surface for what accumulates (module docstring: one cell
    per graph x envelope x sm x toolchain, so every toolchain upgrade leaves
    its predecessor behind). Deliberately not a verification pass: this is the
    listing a ``cozy cells``-style CLI and any future sweep read, and making it
    hash every resident artifact would make listing cost as much as arming.
    """
    out: List[LocalCell] = []
    base = cells_root(root)
    if not base.is_dir():
        return out
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or entry.name == MEMO_DIRNAME:
            continue
        record = _read_json(entry / RECORD_NAME)
        artifact = entry / ARTIFACT_NAME
        if record is None or not artifact.is_file():
            continue
        out.append(LocalCell(
            key=str(record.get("cell_key") or entry.name),
            artifact=artifact,
            content_digest=str(record.get("content_digest") or ""),
            family=str(record.get("family") or ""),
            arm_token=str(record.get("arm_token") or ""),
            bytes=int(record.get("bytes") or artifact.stat().st_size),
            stored_at=float(record.get("stored_at") or 0.0),
        ))
    return out


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
    recorded = _read_json((root or store_root()) / TRUST_CLASS_NAME)
    if recorded is None:
        return ""
    return str(recorded.get("class") or "")


def keeps_cells_locally(root: Optional[Path] = None) -> bool:
    """Whether a cell this machine mints should be kept in the local store.

    True exactly when the hub has ASSERTED this hardware may not publish. The
    cozy-local CLI serve path does not consult this — it has no publisher and
    never will, so it calls :func:`store` directly (pgw#1086 wave 1 re-points
    it there when it deletes the JIT store).
    """
    return trust_class(root) == TRUST_UNTRUSTED


__all__ = [
    "ARTIFACT_NAME",
    "CELLS_DIRNAME",
    "ENV_STORE_DIR",
    "KIND",
    "LocalCell",
    "MEMO_DIRNAME",
    "RECORD_NAME",
    "TRUST_CLASS_NAME",
    "TRUST_UNTRUSTED",
    "UNTRUSTED_REFUSAL_CODE",
    "cell_dir",
    "cells_root",
    "drop",
    "keeps_cells_locally",
    "lookup",
    "lookup_for_arm",
    "memo_path",
    "note_refusal",
    "store",
    "store_root",
    "stored_cells",
    "trust_class",
]
