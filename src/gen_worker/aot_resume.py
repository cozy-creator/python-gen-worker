"""pgw#848 item 5: crash-only mint — a finished entry survives the attempt.

A mint is ~74 min of serial export (MEASURED: 2.06 and 2.07 min/row across two
independent pods, 36 sdxl entries) followed by ~626 s of AOTI compile PER
ENTRY. Today a crash at entry 30 of 36 discards all of it: ``build_cell``
gives every attempt a fresh ``child-<n>`` workdir, the pool's inductor cache
lives inside it, and ``abandon_self_mint`` rmtree's the mint root. Attempt 2
shares nothing with attempt 1 — not the compiled files, not even a cache hit.

This module is the store that survives. It banks one entry's compiled files
under a root that is STABLE across attempts, and it re-admits them on a later
attempt ONLY when the entry's identity is RE-DERIVED and matches.

The safety property, which everything else here is subordinate to
-----------------------------------------------------------------
**A finished entry is admissible only because its identity re-derives, never
because a file exists at a path.** Under pgw#846 a path-trusted resume is a
way to pack a stale artifact into a cell that then verifies, arms, and is
wrong — the worst failure this program has available. So:

* the ``graph_hash`` compared at admission is recomputed from the ExportedProgram
  THIS attempt holds in memory, never read back from the banked artifact. A
  bank cannot vouch for itself;
* every banked file's sha256 is recomputed and compared, so a truncated,
  half-written or edited artifact refuses;
* the identity axes that key the cell (``sm``, toolchain, env seal, the
  parent/child code digest, the inductor configs the compile ran under) must
  all match, and an axis this runtime cannot ASK about refuses rather than
  admits — "unknown" and "none" are recorded as different values, because
  collapsing them is how a check that reads as fail-closed quietly admits;
* the ledger is written LAST and atomically, after the files are copied, so a
  crash mid-bank leaves nothing admissible rather than a short entry.

What this does NOT cover, stated rather than papered over
---------------------------------------------------------
**The EXPORT is not resumable under this rule, and is deliberately excluded.**
An entry's identity is its traced graph; re-deriving it requires the traced
graph, and producing the traced graph IS the export. A resume that skipped the
export could only compare a hash the artifact records about itself, which is
exactly the path-trust this rule forbids. So an attempt re-exports (~2.06
min/row) and that is what BUYS the strong check: the freshly exported program
is the independent re-derivation. The recovered half is the compile — ~5.2 h of
the ~6.4 h a 36-entry sdxl mint spends at K=1.

Threat model: staleness, corruption and skew (a stale bank from an earlier
declaration, a half-written file, a toolchain that moved under us). NOT a local
adversary with write access to the mint root — anyone who can rewrite both the
ledger and the artifact consistently can equally replace the finished cell.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from . import local_cells
from . import graph_hash as graph_hash_mod

logger = logging.getLogger(__name__)

#: Where the bank lives, when the process was not told in code. Set by
#: ``mint_child`` from ``MintRequest.resume`` — a LOCATION, never a recipe.
ENV_RESUME_DIR = "GEN_WORKER_MINT_RESUME_DIR"

#: Ledger format. A bank written by another version is not admissible; it is
#: not read for "the parts we understand".
BANK_V = 1

LEDGER_NAME = "entry.json"
FILES_DIR = "files"
ENTRIES_DIR = "entries"
CACHE_DIR = "inductor-cache"

#: Typed refusal reasons. Every one of these means "recompile this entry",
#: never "fail the mint" — a bank is an optimization and must never be able to
#: cost a cell. They are counted onto the pool ledger so a bank that never
#: admits anything is VISIBLE rather than merely slow.
MISS = "cold"                      # nothing banked for this entry
REFUSE_FORMAT = "bank_format"      # ledger from another BANK_V / unreadable
REFUSE_ENTRY = "entry_mismatch"    # the slot holds a different entry name
REFUSE_GRAPH = "graph_hash"        # THE one that matters: a different graph
REFUSE_CONTEXT = "context"         # sm / toolchain / seal / code / configs
REFUSE_UNSTATED = "context_unstated"   # an axis this runtime cannot state
REFUSE_FILE_MISSING = "file_missing"
REFUSE_FILE_CONTENT = "file_content"   # size or sha256 moved under us

#: Where banks live: a WORKER-LOCAL, persistent directory, deliberately NOT
#: under the mint's own root. `fleet_cells.abandon_self_mint` rmtree's
#: `mint_root`, and abandonment is exactly what a crashed mint ends in — a bank
#: under that root would be deleted on its way out of the one case it exists
#: for. Sited beside `local_cells`' own mint staging (`.mint`) because that is
#: already the pod's compile-scratch neighbourhood, and NOT in the CAS
#: (`cache_dir`), which `disk_gc` owns and sweeps by reference index.
RESUME_DIRNAME = ".mint-resume"

#: A capacity bound on the whole resume area, enforced oldest-first at open.
#: A BYTE cap rather than a count: an entry's compiled files are code-only
#: (pgw#704 B1) and small, but "small" is a property of the graph, so a count
#: would bound the wrong thing. Whole key-scopes are dropped, never parts of
#: one — half a bank is a bank whose remaining entries still admit, which is
#: fine, but a partial drop wastes the copy for no recovery.
ENV_MAX_BYTES = "GEN_WORKER_MINT_RESUME_MAX_BYTES"
DEFAULT_MAX_BYTES = 4 * 1024**3

_ROOT: Optional[str] = None


def bank_root(scope: str) -> Path:
    """The bank directory for one mint SCOPE, outside the mint's own root.

    ``scope`` is the pending's ``cell_key``. For an AOT pending that is a
    CAPTURE HANDLE, not the published cell's key (a real AOT key folds the
    combined graph hash and is unknowable until the export finishes) — which is
    exactly right here: this is a SCOPE, and identity is the per-entry check
    that runs inside it. A colliding scope cannot admit anything wrong; it can
    only produce a graph-hash refusal.
    """

    safe = "".join(
        c if (c.isalnum() or c in "._-") else "_" for c in str(scope))[:48]
    digest = hashlib.sha256(str(scope).encode()).hexdigest()[:12]
    return local_cells.store_root() / RESUME_DIRNAME / f"{safe or 'scope'}-{digest}"


def _dir_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def sweep(keep: Path, *, max_bytes: int = 0) -> int:
    """Hold the resume area under its capacity bound, oldest scope first.

    ``keep`` is never dropped. Everything else is ordered by mtime, newest
    first, and whole scopes are removed from the tail until the total fits. An
    actively-running mint's bank is the newest thing in the area (every banked
    entry touches it), so the loser is an abandoned mint nobody came back for
    — and the cost of getting that wrong is one recompile, never a wrong cell.
    """
    cap = int(max_bytes or os.environ.get(ENV_MAX_BYTES, "") or DEFAULT_MAX_BYTES)
    area = keep.parent
    scopes: List[Tuple[float, Path, int]] = []
    try:
        children = [p for p in area.iterdir() if p.is_dir()]
    except OSError:
        return 0
    for path in children:
        try:
            scopes.append((path.stat().st_mtime, path, _dir_bytes(path)))
        except OSError:
            continue
    total = sum(size for _, _, size in scopes)
    dropped = 0
    for _, path, size in sorted(scopes, reverse=True):
        if total <= cap:
            break
        if path == keep:
            continue
        shutil.rmtree(path, ignore_errors=True)
        total -= size
        dropped += 1
        logger.info(
            "aot-resume: swept the banked scope %s (%.1f MB) — the resume area "
            "is capped at %.1f GB", path.name, size / 1e6, cap / 1024**3)
    return dropped


def discard(scope: str) -> None:
    """Drop one scope's bank. Called when the cell it was recovering has been
    ADOPTED — at that terminus the bank has no further job, and holding it is
    the only way this area grows without bound on a healthy pod."""
    try:
        shutil.rmtree(bank_root(scope), ignore_errors=True)
    except Exception:  # noqa: BLE001 — hygiene never fails a mint
        logger.debug("aot-resume: discard failed", exc_info=True)


def set_root(path: Any) -> None:
    """Install the process-wide resume root.

    Process-global on purpose and NOT an env var: the bank lives entirely in
    the pool's own process, and an inherited env var would follow every entry
    child into a directory it has no business writing.
    """
    global _ROOT
    _ROOT = str(path or "") or None


def resume_enabled() -> bool:
    """Whether this process may resume a banked mint at all.

    pgw#929: the enable used to be HIDDEN INSIDE AN EMPTY STRING — `root()`
    returned `""` and every caller re-derived "so do not resume" from the
    falsiness of a path. That is a decision carried by the absence of a value,
    which is the least visible carrier there is: nothing logs it, nothing
    reports it, and a typo'd directory is indistinguishable from a deliberate
    refusal. The path is the VALUE; this is the DECISION, and it is stated.
    """
    return bool(root())


def root() -> str:
    """The resume root, or "" when this process has none configured."""
    if _ROOT:
        return _ROOT
    return os.environ.get(ENV_RESUME_DIR, "").strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def facts_digest(facts: Mapping[str, Any]) -> str:
    payload = json.dumps(facts, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:32]


#: An axis this runtime could not ASK about. Distinct from a legitimately
#: empty answer (a box with no GPU truthfully has no ``sm``): "unknown" and
#: "none" are different facts, and collapsing them is how a check that reads
#: as fail-closed quietly admits. Never equal to anything, including itself.
UNSTATED = "?"


def context_facts(inductor_configs: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    """The non-graph identity axes an admitted entry must match.

    Deliberately the SAME axes ``aot_mint.cell_identity`` keys a cell on
    (``sm``, ``toolchain``, ``env_seal``) plus the two that are specific to who
    produced the loose files: the pool's parent/child contract digest (pgw#840
    — a child from another tree compiled the files the parent then packed) and
    the inductor configs the compile ran under.

    The configs are compared WHOLE, including keys pgw#757 measured as
    non-identity (``compile_threads``). That is deliberate: refusing too often
    costs one recompile, admitting too often costs a cell. It is also free in
    practice — ``MINT_COMPILE_THREADS`` is a constant, so a re-sized K
    (pgw#842) does not move this axis and a narrower retry keeps its bank.
    """
    facts: Dict[str, str] = {
        "configs": facts_digest(dict(inductor_configs or {})),
    }
    try:
        from . import aot_compile_pool

        # "" is legitimate here (zipimport: no source to compare) and the pool
        # itself proceeds on it — so it is an ANSWER, not a failure.
        facts["code"] = str(aot_compile_pool.CODE_DIGEST or "")
    except Exception:  # noqa: BLE001
        facts["code"] = UNSTATED
    try:
        from . import compile_cache as cc

        # "" is legitimate: a box with no CUDA device truthfully has no `sm`,
        # and a cell banked on a GPU pod carries one — so the two never match
        # by accident. `cell_identity` separately REFUSES to key a cell with no
        # sm, which is where that requirement belongs.
        facts["sm"] = str(cc.runtime_key().get("sm") or "")
    except Exception:  # noqa: BLE001
        facts["sm"] = UNSTATED
    try:
        from . import compile_cache as cc

        facts["toolchain"] = facts_digest(dict(cc.toolchain_digest()))
    except Exception:  # noqa: BLE001
        facts["toolchain"] = UNSTATED
    try:
        from . import env_seal

        facts["env_seal"] = str(env_seal.seal_digest(env_seal.effective_seal()))
    except Exception:  # noqa: BLE001
        facts["env_seal"] = UNSTATED
    return facts


@dataclass(frozen=True)
class Admission:
    """One admission decision, with the reason it was reached."""

    ok: bool
    reason: str = ""
    detail: str = ""
    files: Tuple[str, ...] = ()

    @property
    def refused(self) -> bool:
        """A REFUSAL is a bank that held something and would not hand it over;
        a cold miss is not a refusal and must not be reported as one."""
        return not self.ok and self.reason not in ("", MISS)


@dataclass
class EntryBank:
    """Compiled entries banked under a root that outlives one mint attempt."""

    root: Path
    context: Dict[str, str] = field(default_factory=dict)
    #: entry -> reason, for the pool ledger. Cold misses included, so "the bank
    #: was empty" and "the bank refused 36 entries" are distinguishable.
    outcomes: Dict[str, str] = field(default_factory=dict)
    resumed: List[str] = field(default_factory=list)
    #: entry -> the graph hash RE-DERIVED from this attempt's own program at
    #: admission. The only value ``put`` will bank under, so a file can never
    #: be banked under a hash that came from a file.
    graphs: Dict[str, str] = field(default_factory=dict)
    admit_s: float = 0.0
    banked_bytes: int = 0

    @property
    def cache_dir(self) -> Path:
        """The inductor cache root, under the SAME stable root.

        ``MintRequest.capture`` has documented itself as the
        "TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR root" since pgw#784 and the
        AOT pool has always overridden it with a per-attempt directory it then
        threw away — which is why attempt 2 got not even a cache hit. Inductor's
        cache is keyed by the GRAPH ("the cache key is the graph, not the
        process" — this pool's own measured note), so widening its scope from
        one attempt to one mint changes nothing about what is produced.
        """
        return self.root / CACHE_DIR

    def _slot(self, entry: str) -> Path:
        # Entry names carry '/' and '=' (``unet/adapter=true/dim=3``), so the
        # directory is a digest and the NAME is recorded in the ledger and
        # checked — a slot that holds someone else's entry refuses.
        return self.root / ENTRIES_DIR / hashlib.sha256(
            entry.encode()).hexdigest()[:32]

    # -- admission --------------------------------------------------------

    def admit(self, entry: str, program: Any) -> Admission:
        """Re-admit a banked entry, or say why not.

        ``program`` is THIS attempt's freshly exported program. Its graph hash
        is computed here, from the object in memory — that is the independent
        re-derivation the whole design rests on.
        """
        import time

        t0 = time.monotonic()
        try:
            self.graphs[entry] = graph = graph_hash_mod.graph_hash(program)
            return self._admit(entry, graph)
        except Exception as exc:  # noqa: BLE001 — a bank never fails a mint
            logger.warning(
                "aot-resume: admission for %r failed (%s: %s) — compiling it",
                entry, type(exc).__name__, exc)
            return self._record(entry, Admission(False, REFUSE_FORMAT, str(exc)))
        finally:
            self.admit_s = round(self.admit_s + (time.monotonic() - t0), 3)

    def _admit(self, entry: str, graph: str) -> Admission:
        slot = self._slot(entry)
        ledger_path = slot / LEDGER_NAME
        if not ledger_path.exists():
            return self._record(entry, Admission(False, MISS))
        try:
            ledger = json.loads(ledger_path.read_text())
        except (OSError, ValueError) as exc:
            return self._record(
                entry, Admission(False, REFUSE_FORMAT, str(exc)))
        if int(ledger.get("v") or 0) != BANK_V:
            return self._record(entry, Admission(
                False, REFUSE_FORMAT,
                f"ledger v={ledger.get('v')!r}, this build banks v={BANK_V}"))
        if str(ledger.get("entry") or "") != entry:
            return self._record(entry, Admission(
                False, REFUSE_ENTRY,
                f"slot holds {ledger.get('entry')!r}"))

        # THE check. Recomputed from the live program above; the banked value is
        # only ever the thing being compared AGAINST.
        if str(ledger.get("graph_hash") or "") != graph:
            return self._record(entry, Admission(
                False, REFUSE_GRAPH,
                f"banked graph {ledger.get('graph_hash')!r} != this attempt's "
                f"{graph!r} — the banked artifact is not this graph's"))

        banked_context = dict(ledger.get("context") or {})
        for axis, value in sorted(self.context.items()):
            if value == UNSTATED or banked_context.get(axis) == UNSTATED:
                return self._record(entry, Admission(
                    False, REFUSE_UNSTATED,
                    f"{axis!r} could not be stated ({'now' if value == UNSTATED else 'when it was banked'}); "
                    f"an artifact whose runtime we cannot name is not "
                    f"admissible"))
            if str(banked_context.get(axis, "")) != value:
                return self._record(entry, Admission(
                    False, REFUSE_CONTEXT,
                    f"{axis}: banked {banked_context.get(axis)!r} != "
                    f"{value!r}"))

        files: List[str] = []
        for row in ledger.get("files") or ():
            path = slot / FILES_DIR / str(row.get("name") or "")
            if not path.is_file():
                return self._record(entry, Admission(
                    False, REFUSE_FILE_MISSING, str(path)))
            if path.stat().st_size != int(row.get("size") or -1):
                return self._record(entry, Admission(
                    False, REFUSE_FILE_CONTENT,
                    f"{path.name}: size {path.stat().st_size} != banked "
                    f"{row.get('size')}"))
            if _sha256_file(path) != str(row.get("sha256") or ""):
                return self._record(entry, Admission(
                    False, REFUSE_FILE_CONTENT,
                    f"{path.name}: sha256 moved since it was banked"))
            files.append(str(path))
        if not files:
            return self._record(entry, Admission(
                False, REFUSE_FORMAT, "ledger records no files"))
        self.resumed.append(entry)
        return self._record(entry, Admission(True, files=tuple(files)))

    def _record(self, entry: str, admission: Admission) -> Admission:
        self.outcomes[entry] = "resumed" if admission.ok else admission.reason
        if admission.refused:
            # A cold bank is expected. A bank that HELD this entry and would
            # not hand it over is evidence — of a stale bank, a moved
            # toolchain, or a corrupted artifact — and it must not be silent.
            logger.warning(
                "aot-resume: REFUSED the banked entry %r (%s): %s — "
                "recompiling it", entry, admission.reason, admission.detail)
        elif admission.ok:
            logger.info(
                "aot-resume: entry %r re-admitted from the bank (%d file(s)) "
                "— identity re-derived and matched", entry, len(admission.files))
        return admission

    # -- banking ----------------------------------------------------------

    def put(self, entry: str, graph: str, files: Sequence[str]) -> None:
        """Bank one finished entry. Never raises — a bank that cannot write is
        a lost optimization, not a lost mint."""
        try:
            self._put(entry, graph, files)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "aot-resume: could not bank entry %r (%s: %s) — the mint is "
                "unaffected, a later attempt will recompile it",
                entry, type(exc).__name__, exc)

    def _put(self, entry: str, graph: str, files: Sequence[str]) -> None:
        if not graph:
            raise ValueError(
                "refusing to bank an entry with no graph hash — it could only "
                "ever be re-admitted on path trust")
        unstated = sorted(a for a, v in self.context.items() if v == UNSTATED)
        if unstated:
            raise ValueError(
                f"refusing to bank under a runtime this process cannot state "
                f"({unstated!r}) — it could never be re-admitted, so the copy "
                f"would be cost with no recovery")
        slot = self._slot(entry)
        files_dir = slot / FILES_DIR
        if slot.exists():
            shutil.rmtree(slot, ignore_errors=True)
        files_dir.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, Any]] = []
        total = 0
        for src in files:
            source = Path(src)
            dest = files_dir / source.name
            if dest.exists():
                raise ValueError(
                    f"two compiled files share the basename {source.name!r}; "
                    f"the bank stores by basename and would lose one")
            shutil.copy2(source, dest)
            size = dest.stat().st_size
            total += size
            rows.append({
                "name": source.name,
                "size": int(size),
                "sha256": _sha256_file(dest),
            })
        if not rows:
            raise ValueError("nothing to bank: the entry reported no files")
        payload = {
            "v": BANK_V,
            "entry": entry,
            "graph_hash": graph,
            "context": dict(self.context),
            "files": rows,
        }
        # LAST, and atomic. A crash between the copies and this leaves a slot
        # with no ledger, which is a cold miss — never a short entry that
        # admits.
        handle, tmp = tempfile.mkstemp(dir=str(slot), suffix=".tmp")
        with os.fdopen(handle, "w") as out:
            json.dump(payload, out, sort_keys=True)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, slot / LEDGER_NAME)
        self.banked_bytes += total
        logger.info(
            "aot-resume: banked entry %r (%d file(s), %.1f MB) under %s",
            entry, len(rows), total / 1e6, slot)

    # -- telemetry --------------------------------------------------------

    def facts(self) -> Dict[str, Any]:
        """The bank's contribution to the pool's ``phase=pool`` row.

        Refusals are reported BY REASON and by count: "the bank was cold" and
        "the bank refused every entry on a moved toolchain" are the same wall
        clock and completely different facts.
        """
        refused: Dict[str, int] = {}
        for reason in self.outcomes.values():
            if reason in ("resumed", MISS):
                continue
            refused[reason] = refused.get(reason, 0) + 1
        facts: Dict[str, Any] = {
            "resume_root": str(self.root),
            "resumed": len(self.resumed),
            "resume_cold": sum(
                1 for r in self.outcomes.values() if r == MISS),
            "resume_admit_s": round(self.admit_s, 2),
        }
        if refused:
            facts["resume_refused"] = refused
        if self.banked_bytes:
            facts["resume_banked_bytes"] = int(self.banked_bytes)
        return facts


def open_bank(
    resume_root: str = "",
    *,
    inductor_configs: Optional[Mapping[str, Any]] = None,
) -> Optional[EntryBank]:
    """The bank for this mint, or ``None`` when this process must not resume.

    ``None`` is the untouched pre-pgw#848 behaviour, exactly: no admission
    pass, no hashing, no copies.
    """
    where = str(resume_root or root() or "")
    if not where:
        return None
    try:
        path = Path(where)
        (path / ENTRIES_DIR).mkdir(parents=True, exist_ok=True)
        sweep(path)
        return EntryBank(
            root=path, context=context_facts(inductor_configs))
    except Exception as exc:  # noqa: BLE001 — never fail a mint for a cache
        logger.warning(
            "aot-resume: cannot open the resume bank at %s (%s: %s) — this "
            "mint runs without one", where, type(exc).__name__, exc)
        return None


__all__ = [
    "BANK_V",
    "DEFAULT_MAX_BYTES",
    "ENV_MAX_BYTES",
    "ENV_RESUME_DIR",
    "RESUME_DIRNAME",
    "MISS",
    "REFUSE_CONTEXT",
    "REFUSE_ENTRY",
    "REFUSE_FILE_CONTENT",
    "REFUSE_FILE_MISSING",
    "REFUSE_FORMAT",
    "REFUSE_GRAPH",
    "REFUSE_UNSTATED",
    "UNSTATED",
    "Admission",
    "EntryBank",
    "bank_root",
    "context_facts",
    "discard",
    "open_bank",
    "root",
    "set_root",
    "sweep",
]
