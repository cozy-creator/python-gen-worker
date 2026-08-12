"""pgw#1089 (DESIGN-RULINGS §4.27 step 1): derive this worker's ``ck1`` cell
key AT BOOT, from CODE ALONE, before a single weight byte is resident.

Paul, §4.27 step 1, verbatim: *"**Immediately** on boot: derive the cell key on
CPU from code alone — fake/meta-tensor traces, **as parallel as possible** ('to
get the trace time down as low as possible'). No weights required, ever, for
identity."* Target: **< 60 s, every time**, ~99 % of it the traces.

Why this can exist at all
-------------------------
Two things had to be true, and only one of them was:

1. **The compile target must be constructible without a checkpoint.**
   pgw#1080 increment 2 built that (``models.structure_only``) — for the MINT
   CHILD only. Widening it to the boot path is the gate pgw#1080's own owed
   list names, and it is discharged here by the simplest honest route: the boot
   derivation runs the mint child's own load, in its own processes.
2. **An entry's ``class_hash`` must be statable from the exported program.**
   It was NOT. ``aot_mint.entry_graph_block`` v2 read ``constant_fqns`` and
   ``fused_constants`` off the COMPILED PACKAGE, so identity could not be
   stated until after the compile it was supposed to let us skip. v3 makes the
   block program-only — see that function for why those two facts carry zero
   bits the key does not already hold, and for the live defect it closes (a
   weightless mint and a real-weight mint of the identical graph keyed
   DIFFERENTLY, because pgw#1080's mandatory runtime constant folding moves
   both package-side sets).

Processes, not threads — and the reason is measured
---------------------------------------------------
``aot_compile_pool``'s docstring records four concurrent ``aot_compile`` calls
in ONE process producing one usable result and three distinct internal failures
(``CURRENT_PATCHER is None``, ``KeyError: 'custom'`` in
``fx.traceback.annotate``, a fake-tensor propagation crash), because inductor
keeps process-global mutable state: *"a thread pool here is not slower, it is
WRONG"*. A boot trace is ``torch.export`` under a fake mode — the SAME dynamo
patcher and the SAME ``fx.traceback`` stack — so the unit of parallelism is an
OS process here too. The parent never imports the endpoint, never builds a fake
mode and never installs a dynamo patcher: the serving process's torch state is
untouched by construction, which is the second reason for the split.

Why the MINT can only export serially and this can be K-wide
------------------------------------------------------------
The mint exports *"by construction — one pipeline, one card, and the branch arm
is toggled once for the whole branchless group"*. A weight-free trace has no
card and no shared pipeline: each child composes its OWN structure-only
pipeline, toggles its OWN branch arm, and its rows are ordered adapter-first
inside that child exactly as the mint orders them inside its one process.

Parallelism is not an identity axis
-----------------------------------
Blocks are assembled by ENTRY NAME, never by completion, and the fold is over
``aot_serve.artifact_metadata`` — the mint's own stamping code, called with the
mint's own blocks. K-wide and 1-wide therefore produce the identical key by
construction rather than by care, and ``test_boot_key_pgw1089`` pins it.

The memo, and what it may hold
------------------------------
``closure digest -> the per-class KEYING BLOCKS`` — i.e. the GRAPH half of the
identity, and **never the folded key.** The other three axes (envelope, sm,
toolchain) re-derive in milliseconds every boot and MUST: an sm that changed, a
toolchain that changed or an envelope the author widened has to move the key on
the very next boot, and a memoized key would answer with the previous pod's.
Memoizing the graph half is sound because the traced graph is a pure function
of the code closure the digest names (pgw#990 demoted the closure to exactly
this: *"a memo, never identity"*).

**A memo hit SKIPS THE TRACES.** That is the point of having one — the memo
path is milliseconds, and pgw#1089 says so. It stores the blocks rather than
the finished class hashes for one reason: the hashes are stamped by
``aot_serve.artifact_metadata``, and a memo that stored them would make this
module recompute ``combined_graph_hash`` itself, which is the second derivation
the whole design forbids. Stored blocks re-fold through the mint's own code.

Honesty is enforced, not trusted: when this pod goes on to MINT, the freshly
traced per-class hashes are compared against whatever the memo answered
(:func:`assert_memo_honest`), and a mismatch invalidates the memo entry and
re-traces on the next boot. A wrong key is never produced — at worst a memo is
thrown away, and the pod that threw it away is the pod that proved it wrong.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import msgspec

from . import aot_serve, boot_phases, cell_key, compile_cache as cc, env_seal
from .mint_process import CompileCellSpec, MintSlot, slot_subjects
from .postmortem import cpu_quota_cores, effective_cpu_count

logger = logging.getLogger(__name__)

#: The child entrypoint. ``python -m gen_worker.boot_trace_child <job.json>``.
TRACE_CHILD_MODULE = "gen_worker.boot_trace_child"

#: The directory that must lead the child's ``PYTHONPATH`` so that
#: ``-m gen_worker.boot_trace_child`` means THIS gen_worker (pgw#840's lesson,
#: applied at the second child seam rather than rediscovered on a pod).
PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)

#: The modules that define this parent/child contract. A child running
#: different source than the parent must be caught, not believed — the same
#: backstop ``aot_compile_pool`` keeps.
_CONTRACT_MODULES = ("boot_key.py", "boot_trace_child.py")

#: Serving headroom, in whole cores, held back from the trace pool. A boot
#: trace races the weights download and the pipeline load, and starving those
#: to finish the key sooner trades the number Paul asked for against the
#: number he asked for it FOR (§4.27 step 4 / pgw#1091: nothing on the request
#: path, and the traces must OVERLAP the fetch rather than displace it).
SERVING_HEADROOM_CORES = 1

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_BAD_JOB = 4


def _code_digest() -> str:
    """Digest of the parent/child contract source, taken AT IMPORT."""
    here = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _CONTRACT_MODULES:
        try:
            digest.update(hashlib.sha256((here / name).read_bytes()).digest())
        except OSError:  # zipimport / frozen: no source to compare
            return ""
    return digest.hexdigest()[:16]


CODE_DIGEST = _code_digest()


class BootKeyUnavailable(RuntimeError):
    """This boot cannot state a cell key from code alone, and says why.

    Never a silent fallback: a pod that cannot derive its key adopts nothing
    and mints the old way, and the REASON is what tells us which family's
    structure-only build is missing. ``reason`` is a stable token; ``detail``
    is the sentence.
    """

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = str(reason)
        self.detail = str(detail)
        super().__init__(f"{reason}: {detail}")


# ---------------------------------------------------------------------------
# The parent/child wire
# ---------------------------------------------------------------------------


class TraceJob(msgspec.Struct, frozen=True, kw_only=True):
    """One child's share of the boot trace.

    Everything the child needs and nothing live — the same discipline
    ``MintRequest`` keeps, and it reuses that request's two resolved types so
    a slot resolved for a mint and a slot resolved for a boot trace cannot be
    two different notions of "resolved".
    """

    function: str
    modules: Tuple[str, ...]
    family: str
    cfg: CompileCellSpec
    slots: Dict[str, MintSlot] = {}
    #: This child's share of the declaration's row order, ``rows[i::K]``.
    #: Sharded by INDEX and not by name because the adapter fork is decided by
    #: the COMPOSED pipeline, so the parent cannot enumerate the names to hand
    #: out — see ``aot_mint.declared_class_rows``.
    share_index: int = 0
    share_count: int = 1
    device: int = -1
    report: str = ""
    code_digest: str = ""


class TraceReport(msgspec.Struct, frozen=True, kw_only=True):
    """What one child measured and derived.

    ``blocks`` carries each entry's keying block as CANONICAL JSON rather than
    a decoded object: the parent hands it straight to
    ``aot_serve.artifact_metadata``, and a re-encode on either side is a place
    for two canonicalizations to disagree about the thing being hashed.
    """

    ok: bool = False
    reason: str = ""
    detail: str = ""
    blocks: Dict[str, str] = {}
    nodes: Dict[str, int] = {}
    trace_ms: Dict[str, int] = {}
    setup_ms: int = 0
    #: How many classes the WHOLE declaration produced on this child's
    #: pipeline. Every child reports it, all must agree, and the union of the
    #: shares must be exactly that many — which proves the class set is whole
    #: without the parent ever enumerating it.
    declared_classes: int = 0
    structure_only: Tuple[str, ...] = ()
    weight_lane: str = ""
    precision: str = ""
    strict: bool = True
    lora_bucket: int = 0
    code_digest: str = ""
    #: pgw#847's economy, measured where the graphs are: ``FakeTensorProp``
    #: seconds over this child's first program, beside that program's export
    #: seconds. Never read by a decision here — see ``prop_economy``.
    prop_probe_ms: int = 0
    export_probe_ms: int = 0
    #: pgw#1165: this child's WHOLE device footprint (`total - free` on its own
    #: card), context and kernel images included. 0 = unmeasured, which the
    #: parent's budget reads as "no evidence", never as "free".
    device_peak_bytes: int = 0


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DerivedKey:
    """The key this boot derived, and the measurements that produced it."""

    key: cell_key.CellKey
    #: entry name -> that class's 16-hex ``class_hash``. THE memoizable half.
    class_hashes: Mapping[str, str]
    combined: str
    workers: int
    width_reason: str
    traced: int
    memo: str  # hit | miss | invalidated | disabled
    wall_ms: int
    trace_ms: Mapping[str, int] = field(default_factory=dict)
    nodes: Mapping[str, int] = field(default_factory=dict)
    #: pgw#1031: entry name -> the NODE-level digest of the graph this boot
    #: traced (``aot_mint.keying_block``'s ``graph_witness``). Since option a it
    #: is FOLDED into ``class_hash`` (so the key now separates two bodies behind
    #: one declaration) AND kept here for the adopt backstop, which compares it
    #: against the cell's own record (defense-in-depth). Rides the keying
    #: blocks, so a memo hit carries it exactly as a cold trace does.
    graph_witnesses: Mapping[str, str] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        return self.key.digest

    def axes(self) -> Dict[str, str]:
        return self.key.axes_dict()


@dataclass(frozen=True)
class PoolWidth:
    """How wide this pod may trace, and the reading that decided it."""

    workers: int
    reason: str
    #: pgw#1165: WHICH bound decided it — `classes`, `cpu`, `vram` or `cap`.
    #: Named rather than inferred from the prose so a regression that collapses
    #: the pool is assertable (`c9fb5d4a`'s lesson: the mint pool ran at K=1
    #: fleet-wide for weeks because nothing asserted the achieved width).
    binding: str = "classes"


def trace_workers(classes: int, *, limit: int = 0) -> PoolWidth:
    """How many trace children this pod may run at once.

    DERIVED from the pod's own measured CPU — never a constant and never an
    env (§1.17: an env may carry a value, never a decision). ``cpu_quota_cores``
    is the cgroup's ``cpu.max``: ``os.cpu_count()`` reports the HOST's cores,
    32 on a pod that owns 4, and sizing a pool by it is how a 4-vCPU pod ends
    up thrashing 32 ways.

    A boot trace holds no card and allocates no weights — its parameters are
    fake — so unlike the compile pool there is no VRAM bound and no host-RAM
    bound worth modelling: a structure-only export's measured device high-water
    is 9.8 MiB (pgw#1080) and its RSS is an import closure. CPU is the only
    real bound, and one core is reserved for the serving parent that is
    concurrently fetching and loading weights.

    pgw#1165 CORRECTS the paragraph above, which was measured wrong and said so
    confidently. A structure-only export's *allocator* high-water really is
    ~9.8 MiB — but the CHILD is a process, and a process that touches CUDA pays
    for a context plus the cuBLAS/cuDNN kernel images the export loads. Measured
    on a live pod (RTX 3090, sdxl, uncontended, one request): **7 children at
    1.25-1.29 GiB each**, card at 3.12 MiB free of 23.56 GiB, every one of the
    18 declared classes refused with `torch.export … OUT OF DEVICE MEMORY`. The
    same derivation on a 48 GB A40 succeeded twice. So there IS a device bound,
    it is per-PROCESS rather than per-tensor, and sizing this pool by CPU alone
    is what made a 96-vCPU pod spawn 18 contexts onto a 24 GB card.

    The device bound does NOT live here. This function decides the SHARDING —
    how the declared classes are divided (``rows[i::K]``) — which is a
    correctness question; how many of those children hold the card at once is a
    resource question, and it is answered from a MEASUREMENT by
    :func:`concurrency_budget`, which ``_run_children`` reaches on every boot.
    Splitting them is what lets the budget ship without moving any key, and it
    keeps this function free of a device parameter no caller would pass.
    """
    classes = max(1, int(classes))
    quota = cpu_quota_cores()
    if quota is not None:
        cores = max(1, int(quota))
        basis = f"cgroup cpu.max={quota:g}"
    else:
        cores = effective_cpu_count()
        basis = f"effective_cpu_count={cores} (uncapped cgroup)"
    usable = max(1, cores - SERVING_HEADROOM_CORES)
    workers = min(classes, usable)
    binding = "classes" if workers == classes else "cpu"
    reason = (
        f"K={workers} of {classes} class(es): {basis}, "
        f"minus {SERVING_HEADROOM_CORES} serving-headroom core(s)")
    if limit and limit < workers:
        workers = max(1, int(limit))
        binding = "cap"
        reason = f"K={workers} — caller cap {int(limit)} narrows: {reason}"
    return PoolWidth(workers=workers, reason=reason, binding=binding)


# ---------------------------------------------------------------------------
# The memo — closure digest -> per-class GRAPH HASHES, never the folded key
# ---------------------------------------------------------------------------


#: Memo file schema. Bumped when the MEANING of a stored hash changes; a
#: reader that finds an older version treats the whole file as absent, which
#: is a miss and a re-trace, never a wrong key.
#: v3 (pgw#1031): the stored blocks now carry ``graph_witness``, and a v2 entry
#: has none — which the adopt-side floor reads as "this pod cannot state its
#: own graph" and refuses. Bumped so a stale memo is a re-trace rather than a
#: refusal nobody can explain.
#: v4 (pgw#1113): the digest a row is filed under now names the RESOLVED SLOTS
#: as well as the code and the declaration. A v3 row was filed under a
#: checkpoint-blind digest, so it answers a strictly different question and
#: must not be read as an answer to this one. The version rides the digest
#: input as well as the file header, so a stale row is unaddressable AND the
#: file it lives in is discarded whole — two independent reasons it cannot be
#: misread, which is what "typed invalidation" has to mean for a cache whose
#: wrong answer is a wrong cell.
MEMO_VERSION = 4
MEMO_FILENAME = "boot-key-graphs.json"


def closure_digest(
    family: str, cfg: CompileCellSpec, *, function: str = "",
    slots: Optional[Mapping[str, MintSlot]] = None,
) -> str:
    """The memo's key: what a per-class graph hash is a pure function OF.

    ``cc.content_keys()`` is the SDK+endpoint code content (pgw#990 demoted it
    from a key axis to exactly this: *"a memo, never identity"*), and the
    declaration facts are the other half — two boots of the same code that
    declare different shape ladders trace different graphs.

    ``slots`` is the third half, and it is the one this memo went without for
    two issues (pgw#1113). The traced graph is a function of the CHECKPOINT's
    own config: ``zero_cond_t`` exists on ``Qwen-Image-Edit-2511`` and not on
    ``Qwen-Image``, and block counts, head counts and quantization ops are the
    general case. Without it, a redeploy that rebinds a slot to a different
    checkpoint answered from the PREVIOUS checkpoint's row — and because a
    memo hit skips the traces and returns the memo's own witnesses, the
    pgw#1031 witness floor that ``boot_adopt`` runs against those witnesses
    could only ever agree with itself. Folding the slots in is what makes that
    check capable of failing. Cost: one memo miss and one trace on the first
    boot after a rebinding, which is what correctness costs here.

    Deliberately NOT included: sm, toolchain, env seal. They are key AXES and
    they re-derive every boot in milliseconds; folding them in here would make
    the memo miss on facts whose whole point is that they are cheap to restate.
    Nor the slots' local PATHS — see ``mint_process.slot_subjects``: where the
    bytes landed on this disk is not what was traced.
    """
    facts = {
        "v": MEMO_VERSION,
        "family": str(family or ""),
        "function": str(function or ""),
        "content_keys": dict(cc.content_keys()),
        "declaration": {
            "targets": sorted(str(t) for t in cfg.targets),
            "shapes": sorted([int(v) for v in row] for row in cfg.shapes),
            "text_lens": sorted({int(v) for v in cfg.text_lens}),
            "guidance": sorted(float(v) for v in cfg.guidance_scales),
            "lora_bucket": int(cfg.lora_bucket or 0),
        },
        "subject": cell_key.subject_facts(slot_subjects(dict(slots or {}))),
    }
    blob = json.dumps(facts, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:32]


def _memo_path(memo_dir: Path) -> Path:
    return Path(memo_dir) / MEMO_FILENAME


def read_memo(
    memo_dir: Optional[Path], digest: str,
) -> Dict[str, Dict[str, Any]]:
    """The per-class KEYING BLOCKS memoized under ``digest``, or ``{}``."""
    if not memo_dir or not digest:
        return {}
    path = _memo_path(Path(memo_dir))
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(doc, dict) or int(doc.get("v") or 0) != MEMO_VERSION:
        return {}
    row = (doc.get("closures") or {}).get(str(digest))
    if not isinstance(row, dict):
        return {}
    blocks = row.get("blocks")
    if not isinstance(blocks, dict) or not blocks:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for name, canon in blocks.items():
        try:
            parsed = json.loads(canon)
        except (TypeError, ValueError):
            # One unreadable block invalidates the WHOLE entry: a partial class
            # set is not a narrower key, it is a wrong one (pgw#716).
            return {}
        if not isinstance(parsed, dict):
            return {}
        out[str(name)] = parsed
    return out


def write_memo(
    memo_dir: Optional[Path], digest: str,
    blocks: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Memoize this closure's per-class keying blocks. Best effort."""
    if not memo_dir or not digest or not blocks:
        return False
    path = _memo_path(Path(memo_dir))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            doc = {}
        if not isinstance(doc, dict) or int(doc.get("v") or 0) != MEMO_VERSION:
            doc = {"v": MEMO_VERSION, "closures": {}}
        doc.setdefault("closures", {})[str(digest)] = {
            "blocks": {
                str(k): json.dumps(
                    dict(v), sort_keys=True, separators=(",", ":"))
                for k, v in blocks.items()},
            "written_unix": int(time.time()),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")))
        tmp.replace(path)
        return True
    except OSError:
        logger.debug("boot-key: memo write failed", exc_info=True)
        return False


def invalidate_memo(memo_dir: Optional[Path], digest: str) -> bool:
    """Drop one closure's memoized hashes (a proven-dishonest entry)."""
    if not memo_dir or not digest:
        return False
    path = _memo_path(Path(memo_dir))
    try:
        doc = json.loads(path.read_text())
        if not isinstance(doc, dict):
            return False
        if str(digest) not in (doc.get("closures") or {}):
            return False
        doc["closures"].pop(str(digest), None)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")))
        tmp.replace(path)
        return True
    except (OSError, ValueError):
        return False


def assert_memo_honest(
    memo_dir: Optional[Path],
    digest: str,
    minted_entries: Mapping[str, Mapping[str, Any]],
) -> str:
    """THE honesty gate: what the memo answered must equal what the mint traced.

    Called from the publish path with the freshly minted artifact's own entry
    blocks. Returns ``''`` when the memo agreed (or held nothing for this
    closure), otherwise the reason — and the offending entry is invalidated so
    the next boot re-traces rather than re-reading a hash that has been proven
    wrong.

    This is what makes the memo safe to have at all. A memo that is merely
    *believed* is a key generator with no error path; a memo that is CHECKED
    against the traced truth every time this pod produces one can only ever
    cost a re-trace.
    """
    memoized = read_memo(memo_dir, digest)
    if not memoized:
        return ""
    # The memo holds BLOCKS, so the comparison stamps them through the same
    # `artifact_metadata` the mint stamped its own with — two class hashes
    # compared here were computed by one function, never by two.
    try:
        had_hashes = class_hashes_of(memoized)
    except Exception as exc:  # noqa: BLE001 — an unstampable memo IS dishonest
        invalidate_memo(memo_dir, digest)
        return (
            f"boot-key memo for closure {digest} could not be stamped "
            f"({type(exc).__name__}: {exc}) and has been invalidated")
    had_witnesses = graph_witnesses_of(memoized)
    disagreements: List[str] = []
    for name, block in sorted(minted_entries.items()):
        want = str((block or {}).get("class_hash") or "")
        had = had_hashes.get(str(name))
        if had and want and had != want:
            disagreements.append(f"{name}: memo {had} != traced {want}")
        # pgw#1031: the memo now also answers the ADOPT-side witness, so a
        # memo whose class hash is right and whose witness is stale would
        # admit a colliding cell on the very axis the witness exists to
        # separate. Checked here for the same reason the hash is.
        want_w = str((block or {}).get("graph_witness") or "")
        had_w = had_witnesses.get(str(name)) or ""
        if want_w and had_w != want_w:
            disagreements.append(
                f"{name}: memo graph_witness {had_w or '<absent>'} != traced "
                f"{want_w}")
    extra = sorted(set(had_hashes) - set(minted_entries))
    missing = sorted(set(minted_entries) - set(had_hashes))
    if extra or missing:
        disagreements.append(
            f"class set differs (memo-only {extra[:3]!r}, "
            f"traced-only {missing[:3]!r})")
    if not disagreements:
        return ""
    invalidate_memo(memo_dir, digest)
    return (
        f"boot-key memo for closure {digest} was DISHONEST and has been "
        f"invalidated: " + "; ".join(disagreements[:4]))


# ---------------------------------------------------------------------------
# The fold — the mint's OWN stamping code, called with the mint's own blocks
# ---------------------------------------------------------------------------


def graph_witnesses_of(
    blocks: Mapping[str, Mapping[str, Any]],
) -> Dict[str, str]:
    """``{entry: graph_witness}`` for one set of keying blocks (pgw#1031).

    Read off the blocks rather than recomputed: the witness is stamped where
    the program is, by ``aot_mint.keying_block``, and a second derivation here
    would be the same two-implementations hazard :func:`fold` refuses for the
    key itself. A block that carries none yields ``""`` — and an empty witness
    is what ``aot_identity.verify_graph_witness`` refuses on, never skips.
    """
    return {
        str(name): str((block or {}).get("graph_witness") or "")
        for name, block in blocks.items()
    }


def class_hashes_of(
    blocks: Mapping[str, Mapping[str, Any]],
) -> Dict[str, str]:
    """``{entry: class_hash}`` for one set of keying blocks.

    Stamped by ``aot_serve.artifact_metadata`` — the mint's own function — so a
    hash computed here and a hash the mint stamped are the same computation.
    The envelope/precision/strict arguments do not reach ``class_hash`` (it
    folds target/fork/class_dims/range_digest/graph/graph_witness/strict/
    lora_bucket), which is why this can answer without them; ``strict``/
    ``lora_bucket`` DO, so they are read off the blocks' own
    ``graph.specialization``.
    """
    head = next(iter(blocks.values()), {}) if blocks else {}
    spec = dict((head.get("graph") or {}).get("specialization") or {})
    meta = aot_serve.artifact_metadata(
        family="", precision="", cell_key="",
        entries={str(k): dict(v) for k, v in blocks.items()},
        strict_export=bool(spec.get("strict", True)),
        lora_bucket=int(spec.get("lora_bucket", 0) or 0))
    return {
        str(name): str(row.get("class_hash") or "")
        for name, row in (meta.get("entries") or {}).items()
    }


def fold(
    blocks: Mapping[str, Mapping[str, Any]],
    *,
    family: str,
    precision: str,
    strict: bool,
    lora_bucket: int,
    envelope: Mapping[str, Any],
) -> Tuple[cell_key.CellKey, Dict[str, str], str]:
    """``(key, {entry: class_hash}, combined_graph_hash)`` for one class set.

    The fold is ``aot_serve.artifact_metadata`` followed by
    ``cell_key.from_exported_artifact_metadata`` — i.e. the mint's own stamp
    and the publish path's own recomputation, reached through the identical
    functions. There is deliberately no ``class_hash``/``combined_graph_hash``
    arithmetic in this module: a second implementation of the fold is exactly
    the attempt-28 phantom (a declared-facts key beside a traced-facts key
    under one axis name) that pgw#1059 exists to have retired.

    The three non-graph axes are restated FRESH here on every call — envelope
    from the declaration, ``sm`` from ``aot_serve.artifact_metadata``'s own
    ``runtime_key()`` probe, toolchain from ``cc.toolchain_digest()`` — which
    is why the memo may hold graph hashes and must never hold the key.
    """
    meta = aot_serve.artifact_metadata(
        family=str(family or ""),
        precision=str(precision or ""),
        cell_key="",
        entries={str(k): dict(v) for k, v in blocks.items()},
        strict_export=bool(strict),
        lora_bucket=int(lora_bucket or 0),
    )
    meta[cell_key.EXPORT_ENVELOPE_KEY] = dict(envelope)
    meta["toolchain"] = dict(cc.toolchain_digest())
    meta[env_seal.SEAL_KEY] = env_seal.effective_seal()
    with boot_phases.span(
        boot_phases.PHASE_KEY_FOLD, function=str(family or ""),
    ) if boot_phases.in_boot() else _null() as span:
        key = cell_key.from_exported_artifact_metadata(meta)
        if span is not None:
            span.note(f"classes={len(blocks)}")
    class_hashes = {
        str(name): str(block.get("class_hash") or "")
        for name, block in (meta.get("entries") or {}).items()
    }
    return key, class_hashes, str(meta.get("combined_graph_hash") or "")


def _null() -> Any:
    import contextlib

    return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------


def child_argv(job_path: Path, *, python: str = "") -> List[str]:
    import sys

    return [python or sys.executable, "-m", TRACE_CHILD_MODULE, str(job_path)]


def child_env(base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The child's environment: THIS gen_worker, and no cwd shadowing it."""
    env = dict(os.environ if base is None else base)
    env["PYTHONSAFEPATH"] = "1"
    existing = [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
                if p and p != PACKAGE_ROOT]
    env["PYTHONPATH"] = os.pathsep.join([PACKAGE_ROOT] + existing)
    return env


def shares(classes: int, workers: int) -> List[Tuple[int, int]]:
    """``[(share_index, share_count)]`` — one entry per child that has work.

    Round-robin (``rows[i::K]``), never contiguous chunks: the declaration
    lists a family's rows grouped by TARGET, and a family's denoiser rows cost
    an order of magnitude more than its VAE rows. Contiguous chunks hand one
    child the whole denoiser group and the pool's wall becomes that child's
    wall.
    """
    count = max(1, int(workers))
    return [(i, count) for i in range(count) if i < max(1, int(classes))]


def free_device_bytes() -> int:
    """Bytes actually free on this process's card, or 0 when there is no card.

    pgw#1165: read at the moment of the fan-out, so the parent's OWN residents
    (it is serving throughout a §4.28 boot) are already excluded — the children
    compete for what is left, not for the nameplate capacity.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        free, _total = torch.cuda.mem_get_info()
        return max(0, int(free))
    except Exception:  # noqa: BLE001 — a probe never changes an outcome
        return 0


def concurrency_budget(
    pending: int, *, free_bytes: int, per_child_bytes: int,
) -> Tuple[int, str]:
    """How many trace children may hold this card AT ONCE.

    Separate from :func:`trace_workers`, which decides the SHARDING: how the
    declared classes are divided is a correctness question (`rows[i::K]`, and
    the union must be the whole class set), while how many of those children
    are resident simultaneously is purely a resource question. Bounding the
    second leaves the first — and therefore the derived key — byte-identical,
    which is the property that lets this ship without re-keying anything.

    ``per_child_bytes`` is the child's WHOLE PROCESS footprint on the card and
    must stay that. The th#1825 lane bounded the same question for finalize's
    adopt and ruled out per-entry literals as the dominant term by three
    independent measurements (literals live inside the artifact — 4.19 MB); the
    real cost is the loaded AOTI packages, plus device code, per-runner
    workspace and load-time buffers, **none of which appear in the artifact**.
    So sizing this budget against an artifact size, a literal size or an
    allocator high-water would under-count badly and in the direction that
    OOMs. `total - free` is the only input that sees all of it.

    Two conventions, matching `mint_workers`' surviving RSS bank: the number
    is MONOTONE (it only ever tightens — see `_run_children`),
    and the reason STATES ITS BASIS, so a measured bound and an absent one can
    never be read as the same claim.
    """
    pending = max(1, int(pending))
    if per_child_bytes <= 0 or free_bytes <= 0:
        return pending, (
            "basis=unmeasured — every pending child at once "
            "(pre-pgw#1165 behaviour)")
    affordable = max(1, int(free_bytes // int(per_child_bytes)))
    width = min(pending, affordable)
    return width, (
        f"W={width} of {pending} pending: basis=measured "
        f"{free_bytes / 1024**3:.2f} GiB free / "
        f"{int(per_child_bytes) / 1024**3:.2f} GiB per child "
        f"= {affordable} affordable")


def _spawn_wave(
    jobs: Sequence[Tuple[TraceJob, Path]], *, python: str = "",
) -> List[TraceReport]:
    """Spawn this wave's children at once and reap them all."""
    procs: List[Tuple[TraceJob, Path, subprocess.Popen, Any]] = []
    for job, job_path in jobs:
        stderr_path = job_path.parent / f"{job_path.stem}.stderr.log"
        handle = stderr_path.open("wb")
        try:
            proc = subprocess.Popen(
                child_argv(job_path, python=python),
                stdout=subprocess.DEVNULL,
                stderr=handle,
                env=child_env(),
                start_new_session=True,
            )
        finally:
            handle.close()
        procs.append((job, stderr_path, proc, None))
        logger.info(
            "boot-key: trace child pid %s -> share %d/%d",
            proc.pid, job.share_index, job.share_count)

    reports: List[TraceReport] = []
    for job, stderr_path, proc, _ in procs:
        code = proc.wait()
        report_path = Path(job.report)
        raw = b""
        try:
            raw = report_path.read_bytes()
        except OSError:
            pass
        if raw:
            try:
                reports.append(msgspec.json.decode(raw, type=TraceReport))
                continue
            except msgspec.ValidationError as exc:
                reports.append(TraceReport(
                    ok=False, reason="bad_report", detail=str(exc)))
                continue
        tail = ""
        try:
            tail = stderr_path.read_text(errors="replace")[-1200:]
        except OSError:
            pass
        reports.append(TraceReport(
            ok=False,
            reason="child_died" if code != EXIT_REFUSED else "refused",
            detail=(
                f"trace child exited {code} without a report; stderr tail: "
                f"{tail}"),
        ))
    return reports


def _run_children(
    jobs: Sequence[Tuple[TraceJob, Path]], *, python: str = "",
) -> List[TraceReport]:
    """Run every child, holding the card to what it can actually carry.

    pgw#1165. This used to be "spawn every child at once", and K came from CPU
    alone — so a 96-vCPU pod put 18 CUDA contexts on one card. On a 24 GB card
    that is 3.12 MiB free and 18 of 18 classes refusing with OUT OF DEVICE
    MEMORY; on a 48 GB card it fits, which is why the failure looked like "small
    cards cannot derive keys" for a day.

    The FIRST child is a probe: it runs alone and reports what one child costs
    the card (`TraceReport.device_peak_bytes`, the whole process footprint
    including its CUDA context). Every remaining child is admitted against that
    measurement and the card's real free bytes. Nothing is estimated, nothing is
    configured, and a card that reports nothing keeps exactly the old width —
    an absent measurement must not be able to throttle a pod.

    Deliberately NOT a serialization. `c9fb5d4a` is the cautionary case: the
    mint's entry pool ran at K=1 fleet-wide for weeks, cost every mint 2.4x, and
    survived because nothing asserted the achieved width. So the width reached
    here is reported (`wave_widths`) and asserted in tests, and the probe costs
    exactly one child's latency — on a card with room, wave 2 carries the whole
    remainder.
    """
    jobs = list(jobs)
    if len(jobs) <= 1:
        return _spawn_wave(jobs, python=python)

    reports: List[TraceReport] = list(_spawn_wave(jobs[:1], python=python))
    measured = max((int(r.device_peak_bytes or 0) for r in reports), default=0)
    pending = jobs[1:]
    widths: List[int] = [1]
    while pending:
        width, why = concurrency_budget(
            len(pending),
            free_bytes=free_device_bytes(),
            per_child_bytes=measured,
        )
        logger.info(
            "boot-key: pgw#1165 trace wave %s (%d child(ren) still pending)",
            why, len(pending))
        wave, pending = pending[:width], pending[width:]
        widths.append(len(wave))
        got = _spawn_wave(wave, python=python)
        reports.extend(got)
        # A later child may cost more than the probe did; keep the high-water
        # so the budget tightens on evidence and never loosens on hope.
        measured = max([measured] + [int(r.device_peak_bytes or 0) for r in got])
    logger.info(
        "boot-key: pgw#1165 traced %d child(ren) in %d wave(s) %s, "
        "per-child device high-water %.2f GiB",
        len(jobs), len(widths), widths, measured / 1024**3 if measured else 0.0)
    return reports


def derive(
    *,
    function: str,
    modules: Sequence[str],
    family: str,
    cfg: CompileCellSpec,
    slots: Mapping[str, MintSlot],
    declared_hint: int,
    envelope: Mapping[str, Any],
    work_root: Path,
    memo_dir: Optional[Path] = None,
    device: int = -1,
    workers: int = 0,
    python: str = "",
    trust_memo: bool = True,
    precision: str = "",
    strict: bool = True,
) -> DerivedKey:
    """Derive this boot's ``ck1`` key from code alone. §4.27 step 1.

    ``trust_memo=False`` forces the traces even when a memo is present and then
    RULES on what the memo held — the verify posture, never the boot default.

    ``declared_hint`` is the parent's read of the declaration's class-row count
    — ``len(aot_declaration.cell_plans(decl))``, which needs no pipeline. It
    sizes K and NOTHING else: the adapter fork can double it, and the true
    class set is whatever the children enumerate off their own composed
    pipelines. A hint that sizes a pool cannot move a key.

    Raises :class:`BootKeyUnavailable` naming the reason. A boot that cannot
    derive a key is a boot that adopts nothing and mints the ordinary way; it
    is never a boot that guesses one.
    """
    t0 = time.monotonic()
    if int(declared_hint) <= 0:
        raise BootKeyUnavailable(
            "no_classes",
            f"family {family!r} declares no graph classes; a cell with no "
            f"class set has no identity (pgw#716/#758)")

    digest = closure_digest(family, cfg, function=function, slots=slots)
    memoized = read_memo(memo_dir, digest) if trust_memo else {}
    memo_state = "miss"

    # THE MEMO PATH — milliseconds, and no trace at all. The graph half of the
    # identity is a pure function of the code closure this digest names, so a
    # hit re-folds the stored blocks; the other three axes are restated FRESH
    # inside `fold`, which is what makes it safe to skip the expensive half and
    # still re-key on a toolchain upgrade, a different card or a widened
    # envelope. Honesty is not assumed here — it is enforced at the next MINT
    # by `assert_memo_honest`, which is the only moment this pod holds a traced
    # truth to compare against.
    if memoized:
        key, class_hashes, combined = fold(
            memoized, family=family,
            precision=str(precision or ""), strict=strict,
            lora_bucket=int(cfg.lora_bucket or 0), envelope=envelope)
        wall_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "boot-key: %s from MEMO in %d ms — %d class(es), no trace "
            "(closure %s)", key.digest, wall_ms, len(class_hashes), digest)
        return DerivedKey(
            key=key, class_hashes=class_hashes, combined=combined,
            workers=0,
            width_reason="memo hit — no trace child was spawned",
            traced=0, memo="hit", wall_ms=wall_ms,
            graph_witnesses=graph_witnesses_of(memoized))

    work = Path(work_root)
    work.mkdir(parents=True, exist_ok=True)

    width = trace_workers(int(declared_hint), limit=workers)
    jobs: List[Tuple[TraceJob, Path]] = []
    for share_index, share_count in shares(int(declared_hint), width.workers):
        child_dir = work / f"trace-{share_index}"
        child_dir.mkdir(parents=True, exist_ok=True)
        job = TraceJob(
            function=str(function),
            modules=tuple(str(m) for m in modules),
            family=str(family),
            cfg=cfg,
            slots=dict(slots),
            share_index=share_index,
            share_count=share_count,
            device=int(device),
            report=str(child_dir / "report.json"),
            code_digest=CODE_DIGEST,
        )
        job_path = child_dir / "job.json"
        job_path.write_bytes(msgspec.json.encode(job))
        jobs.append((job, job_path))

    reports = _run_children(jobs, python=python)
    failed = [r for r in reports if not r.ok]
    if failed:
        first = failed[0]
        raise BootKeyUnavailable(
            first.reason or "trace_failed",
            f"{len(failed)} of {len(reports)} boot-trace child(ren) produced "
            f"no class hashes: {first.detail[:600]}")

    drifted = sorted({
        r.code_digest for r in reports
        if CODE_DIGEST and r.code_digest and r.code_digest != CODE_DIGEST})
    if drifted:
        raise BootKeyUnavailable(
            "code_drift",
            f"trace child ran contract source {drifted!r} while this parent "
            f"runs {CODE_DIGEST!r} — the key would name graphs this process's "
            f"code did not describe (pgw#840's failure, at the boot seam)")

    blocks: Dict[str, Dict[str, Any]] = {}
    nodes: Dict[str, int] = {}
    trace_ms: Dict[str, int] = {}
    for report in reports:
        for name, canon in report.blocks.items():
            blocks[str(name)] = json.loads(canon)
        nodes.update({str(k): int(v) for k, v in report.nodes.items()})
        trace_ms.update({str(k): int(v) for k, v in report.trace_ms.items()})

    # THE completeness proof, and it never consults a parent-side guess: every
    # child reports how many classes the WHOLE declaration produced on its own
    # composed pipeline, all of them must agree, and the union of the shares
    # must be exactly that many. A key that cannot name every class is a key a
    # mismatch cannot name (pgw#716).
    declared = sorted({int(r.declared_classes) for r in reports})
    if len(declared) != 1 or declared[0] <= 0:
        raise BootKeyUnavailable(
            "class_set_disagreement",
            f"the trace children do not agree on how many classes this "
            f"declaration produces ({declared!r}) — they composed different "
            f"pipelines, so their graphs are not one cell's graphs")
    total = declared[0]
    duplicated = sum(len(r.blocks) for r in reports) - len(blocks)
    if len(blocks) != total or duplicated:
        raise BootKeyUnavailable(
            "class_set_gap",
            f"the trace children returned {len(blocks)} distinct class(es) "
            f"({duplicated} duplicated) of the {total} this declaration "
            f"produces — the shares do not reconstruct the class set")

    head = reports[0]
    key, class_hashes, combined = fold(
        blocks,
        family=family,
        precision=head.precision,
        strict=head.strict,
        lora_bucket=head.lora_bucket,
        envelope=envelope,
    )

    # `trust_memo=False` is the VERIFY posture: trace anyway and rule on
    # whatever the memo held. It is what the rig and the mint-adjacent honesty
    # gate use, and it is the only way a stale memo is caught before a mint —
    # so it exists, and it is never the boot default (a boot that re-traced to
    # check its own memo would have no memo path at all).
    if not trust_memo:
        held = read_memo(memo_dir, digest)
        if held:
            try:
                had = class_hashes_of(held)
            except Exception:  # noqa: BLE001 — unstampable IS disagreement
                had = {}
            if had != class_hashes:
                invalidate_memo(memo_dir, digest)
                memo_state = "invalidated"
                logger.warning(
                    "boot-key: memo for closure %s disagreed with the fresh "
                    "trace — invalidated, and the FRESH hashes are what this "
                    "key names", digest)
            else:
                memo_state = "verified"
    write_memo(memo_dir, digest, blocks)

    wall_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "boot-key: %s in %d ms — %s, %d class(es), memo=%s, key=%s",
        key.digest, wall_ms, width.reason, len(class_hashes), memo_state,
        key.digest)
    return DerivedKey(
        key=key,
        class_hashes=class_hashes,
        combined=combined,
        workers=width.workers,
        width_reason=width.reason,
        traced=len(class_hashes),
        memo=memo_state,
        wall_ms=wall_ms,
        trace_ms=trace_ms,
        nodes=nodes,
        graph_witnesses=graph_witnesses_of(blocks),
    )


def prop_economy(reports: Sequence[TraceReport]) -> Dict[str, Any]:
    """pgw#847's number, aggregated off whatever the children measured.

    ``export_s - prop_s`` is the saving one-export-N-props would buy. This
    function REPORTS it and decides nothing (pgw#830: instrument first,
    optimise never in the same change) — the collapse itself is a separate
    increment, and it may only be built once this ratio is measured on a real
    family's graphs rather than bounded by an off-pod probe.
    """
    exports = [int(r.export_probe_ms) for r in reports if r.export_probe_ms > 0]
    props = [int(r.prop_probe_ms) for r in reports if r.prop_probe_ms > 0]
    if not exports or not props:
        return {"measured": False}
    export_ms = sum(exports) / len(exports)
    prop_ms = sum(props) / len(props)
    return {
        "measured": True,
        "export_ms": round(export_ms, 1),
        "prop_ms": round(prop_ms, 1),
        "ratio": round(prop_ms / export_ms, 4) if export_ms else 0.0,
        "samples": len(props),
    }


__all__ = [
    "BootKeyUnavailable",
    "CODE_DIGEST",
    "DerivedKey",
    "MEMO_FILENAME",
    "MEMO_VERSION",
    "PACKAGE_ROOT",
    "PoolWidth",
    "SERVING_HEADROOM_CORES",
    "TRACE_CHILD_MODULE",
    "TraceJob",
    "TraceReport",
    "assert_memo_honest",
    "child_argv",
    "child_env",
    "class_hashes_of",
    "closure_digest",
    "derive",
    "graph_witnesses_of",
    "shares",
    "fold",
    "invalidate_memo",
    "prop_economy",
    "read_memo",
    "trace_workers",
    "write_memo",
]
