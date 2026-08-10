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
``closure digest -> per-class GRAPH HASHES``. **Never the folded key.** The
other three axes (envelope, sm, toolchain) re-derive in milliseconds every boot
and MUST: an sm that changed, a toolchain that changed or an envelope the
author widened has to move the key on the very next boot, and a memoized key
would answer with the previous pod's. Memoizing the graph hashes alone is
sound because the graph is a pure function of the code closure the digest
names.

Honesty is enforced, not trusted: when this pod goes on to MINT, the freshly
traced per-class hashes are compared against whatever the memo answered
(:func:`assert_memo_honest`), and a mismatch invalidates the memo entry and
re-traces. A wrong key is never produced — at worst a memo is thrown away.
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
from .mint_process import CompileCellSpec, MintSlot
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
    reason = (
        f"K={workers} of {classes} class(es): {basis}, "
        f"minus {SERVING_HEADROOM_CORES} serving-headroom core(s)")
    if limit and limit < workers:
        workers = max(1, int(limit))
        reason = f"K={workers} — caller cap {int(limit)} narrows: {reason}"
    return PoolWidth(workers=workers, reason=reason)


# ---------------------------------------------------------------------------
# The memo — closure digest -> per-class GRAPH HASHES, never the folded key
# ---------------------------------------------------------------------------


#: Memo file schema. Bumped when the MEANING of a stored hash changes; a
#: reader that finds an older version treats the whole file as absent, which
#: is a miss and a re-trace, never a wrong key.
MEMO_VERSION = 1
MEMO_FILENAME = "boot-key-graphs.json"


def closure_digest(family: str, cfg: CompileCellSpec, *, function: str = "") -> str:
    """The memo's key: what a per-class graph hash is a pure function OF.

    ``cc.content_keys()`` is the SDK+endpoint code content (pgw#990 demoted it
    from a key axis to exactly this: *"a memo, never identity"*), and the
    declaration facts are the other half — two boots of the same code that
    declare different shape ladders trace different graphs.

    Deliberately NOT included: sm, toolchain, env seal. They are key AXES and
    they re-derive every boot in milliseconds; folding them in here would make
    the memo miss on facts whose whole point is that they are cheap to restate.
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
    }
    blob = json.dumps(facts, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:32]


def _memo_path(memo_dir: Path) -> Path:
    return Path(memo_dir) / MEMO_FILENAME


def read_memo(memo_dir: Optional[Path], digest: str) -> Dict[str, str]:
    """The per-class graph hashes memoized under ``digest``, or ``{}``."""
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
    hashes = row.get("class_hashes")
    if not isinstance(hashes, dict) or not hashes:
        return {}
    return {str(k): str(v) for k, v in hashes.items() if str(v)}


def write_memo(
    memo_dir: Optional[Path], digest: str, class_hashes: Mapping[str, str],
) -> bool:
    """Memoize this closure's per-class graph hashes. Best effort."""
    if not memo_dir or not digest or not class_hashes:
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
            "class_hashes": {str(k): str(v) for k, v in class_hashes.items()},
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
    disagreements: List[str] = []
    for name, block in sorted(minted_entries.items()):
        want = str((block or {}).get("class_hash") or "")
        had = memoized.get(str(name))
        if had and want and had != want:
            disagreements.append(f"{name}: memo {had} != traced {want}")
    extra = sorted(set(memoized) - set(minted_entries))
    missing = sorted(set(minted_entries) - set(memoized))
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


def _run_children(
    jobs: Sequence[Tuple[TraceJob, Path]], *, python: str = "",
) -> List[TraceReport]:
    """Spawn every child at once and reap them all. K is already the width."""
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
) -> DerivedKey:
    """Derive this boot's ``ck1`` key from code alone. §4.27 step 1.

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

    digest = closure_digest(family, cfg, function=function)
    memoized = read_memo(memo_dir, digest)
    memo_state = "miss"

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

    if memoized:
        stale = {
            name for name, value in class_hashes.items()
            if memoized.get(name) and memoized[name] != value}
        if stale or set(memoized) != set(class_hashes):
            invalidate_memo(memo_dir, digest)
            memo_state = "invalidated"
            logger.warning(
                "boot-key: memo for closure %s disagreed with the fresh trace "
                "on %d class(es) — invalidated, and the FRESH hashes are what "
                "this key names", digest, len(stale) or len(class_hashes))
        else:
            memo_state = "hit"
    write_memo(memo_dir, digest, class_hashes)

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
    "closure_digest",
    "derive",
    "shares",
    "fold",
    "invalidate_memo",
    "prop_economy",
    "read_memo",
    "trace_workers",
    "write_memo",
]
