"""MINT-LANE TOOLING (pgw#1327): derive a family's ``cg-key-v1`` entry key SET
from CODE ALONE, by structure-only ``torch.export`` traces in child processes.

**This module is not on the serve-boot path** (pgw#1327). Running the traces
below at boot makes torch, diffusers and the endpoint's model code a hard
dependency of a pod that will never compile anything. They compute a pure
function of code the mint lane already ran, so their OUTPUT is emitted as a
``cg-keyset-v1`` document (:mod:`gen_worker.keyset`) and a serve pod reads it as
data. What remains here is the DERIVATION, which the mint lane owns:

* an ordinary Python serving pod that misses and mints (§4.28) runs this once and
  records the closure in its own cache;
* ``scripts/emit_cg_keyset.py`` / ``scripts/boot_key_rig.py`` run it deliberately
  to produce the document that gets baked beside ``endpoint.lock``.

``boot_adopt`` does not import this module. It takes a deriver as an argument and
the executor supplies one only on a mint-capable pod, so pgw#1328's adopt-only
serve role can pass ``None`` and the import guard has nothing to catch.

The result is a KEY SET — one ``cg-key-v1`` per declared graph class — so a
PARTIAL resolve helps: a pod that resolves 30
of 36 keys arms 30 classes and compiles 6, where a single compiled graph key made that
same outcome a total miss and a full re-mint.

Processes, not threads
----------------------
Concurrent ``aot_compile`` calls in ONE process corrupt each other — inductor
keeps process-global mutable state (``CURRENT_PATCHER is None``, ``KeyError:
'custom'`` in ``fx.traceback.annotate``, fake-tensor propagation crashes). A
boot trace is ``torch.export`` under a
fake mode — the SAME dynamo patcher and the SAME ``fx.traceback`` stack — so
the unit of parallelism is an OS process here too. The parent never imports the
endpoint, never builds a fake mode and never installs a dynamo patcher: the
serving process's torch state is untouched by construction, which is the second
reason for the split.

Why the MINT can only export serially and this can be K-wide
------------------------------------------------------------
The mint exports serially by construction — one pipeline, one card, and the
branch arm is toggled once for the whole branchless group. A weight-free trace
has no card and no shared pipeline: each child composes its OWN structure-only
pipeline, toggles its OWN branch arm, and its rows are ordered adapter-first
inside that child exactly as the mint orders them inside its one process.

Parallelism is not an identity axis
-----------------------------------
TCG declarations are assembled by ENTRY NAME, never by completion. The boot
child and compile child share one worker-to-TCG translation, and only TCG's
validated declaration output crosses the process boundary. K-wide and 1-wide
therefore produce the identical key by construction rather than by care.

What this writes, and what it may hold
--------------------------------------
``closure digest -> the per-class TCG class hashes`` plus per-class metadata —
i.e. the GRAPH half of the identity, and **never the folded keys.** The other two
axes (sm, toolchain) re-derive in milliseconds every boot and MUST: an sm or
toolchain that changed has to move every key on the very next boot, and a stored
key would answer with the minting pod's. Storing the graph half is sound because
the traced graph is a pure function of the code closure the digest names — which
is also exactly why it can be SHIPPED to a pod that never traces.

That document is :mod:`gen_worker.keyset`; the grammar, the schema, the address
and the fold all live there, and this module is one of its two producers.

Honesty is enforced, not trusted: when this pod goes on to MINT, the freshly
traced per-class TCG class hashes are compared against whatever the cache
answered (``keyset.store.assert_honest``), and a mismatch invalidates the entry
and re-derives on the next boot. ``trust_memo=False`` is the explicit
verification posture. A wrong key is never produced — at worst a cached row is
thrown away, by the pod that proved it wrong.

THE CALLER, named because this check once had none in ``src/`` while its
description read exactly like enforcement: ``mint_supervisor.rule_on_boot_memo``,
at the seal/publish seam of every supervised mint. If you move the mint's
publish seam, move the call with it — a paragraph is not a caller.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple)

import msgspec

from . import hostfacts, keyset
from .child_contract import CompileSpec, MintSlot
from .keyset import closure as keyset_closure, emit as keyset_emit, store as keyset_store

if TYPE_CHECKING:
    from gen_worker._vendor.torchcg import GraphClassDeclaration

logger = logging.getLogger(__name__)

#: The child entrypoint. ``python -m gen_worker.boot_trace_child <job.json>``.
TRACE_CHILD_MODULE = "gen_worker.boot_trace_child"

#: The directory that must lead the child's ``PYTHONPATH`` so that
#: ``-m gen_worker.boot_trace_child`` means THIS gen_worker.
PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)

#: Serving headroom, in whole cores, held back from the trace pool. A boot
#: trace races the weights download and the pipeline load: the traces must
#: OVERLAP the fetch rather than displace it, and nothing may land on the
#: request path.
SERVING_HEADROOM_CORES = 1

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_BAD_JOB = 4


def _code_digest() -> str:
    """Digest of the parent/child contract source, taken AT IMPORT.

    ONE derivation, in ``keyset.closure`` — the same bytes address the shipped
    key set and fence the child wire, and two implementations of "which tracer
    contract is this" is exactly the drift pgw#840 cost us. Absent source
    (zipimport / frozen) is "" HERE, because a drift check with nothing to
    compare must not refuse a boot; the key-set address raises instead, because
    an unstatable closure must not silently read somebody else's document.
    """
    try:
        return keyset_closure.contract_code_digest()
    except keyset.KeySetError:
        return ""


CODE_DIGEST = _code_digest()


class BootKeyUnavailable(RuntimeError):
    """This boot cannot state its compiled-graph key set, and says why.

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
    cfg: CompileSpec
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

    ``declarations`` carries each entry's TCG declaration as canonical JSON.
    The parent reconstructs the public declaration type, which validates every
    fact and re-derives ``class_hash`` before any runtime axis is folded.
    """

    ok: bool = False
    reason: str = ""
    detail: str = ""
    declarations: Dict[str, str] = {}
    nodes: Dict[str, int] = {}
    trace_ms: Dict[str, int] = {}
    setup_ms: int = 0
    #: How many classes the WHOLE declaration produced on this child's
    #: pipeline. Every child reports it, all must agree, and the union of the
    #: shares must be exactly that many — which proves the class set is whole
    #: without the parent ever enumerating it.
    declared_classes: int = 0
    structure_only: Tuple[str, ...] = ()
    code_digest: str = ""
    #: ``FakeTensorProp`` seconds over this child's first program, beside that
    #: program's export seconds. Never read by a decision here — see
    #: ``prop_economy``.
    prop_probe_ms: int = 0
    export_probe_ms: int = 0
    #: This child's WHOLE device footprint (`total - free` on its own
    #: card), context and kernel images included. 0 = unmeasured, which the
    #: parent's budget reads as "no evidence", never as "free".
    device_peak_bytes: int = 0


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolWidth:
    """How wide this pod may trace, and the reading that decided it."""

    workers: int
    reason: str
    #: WHICH bound decided it — `classes`, `cpu`, `vram` or `cap`. Named rather
    #: than inferred from the prose so a regression that collapses the pool is
    #: assertable.
    binding: str = "classes"


def trace_workers(classes: int, *, limit: int = 0) -> PoolWidth:
    """How many trace children this pod may run at once.

    DERIVED from the pod's own measured CPU — never a constant and never an
    env (an env may carry a value, never a decision). ``cpu_quota_cores``
    is the cgroup's ``cpu.max``: ``os.cpu_count()`` reports the HOST's cores,
    32 on a pod that owns 4, and sizing a pool by it is how a 4-vCPU pod ends
    up thrashing 32 ways. One core is reserved for the serving parent that is
    concurrently fetching and loading weights.

    There IS also a device bound, but it is per-PROCESS rather than per-tensor:
    a structure-only export's *allocator* high-water is ~9.8 MiB, while the
    CHILD is a process that pays for a CUDA context plus the cuBLAS/cuDNN kernel
    images the export loads — measured at 1.25-1.29 GiB per child.

    That device bound does NOT live here. This function decides the SHARDING —
    how the declared classes are divided (``rows[i::K]``) — which is a
    correctness question; how many of those children hold the card at once is a
    resource question, and it is answered from a MEASUREMENT by
    :func:`concurrency_budget`, which ``_run_children`` reaches on every boot.
    Splitting them is what lets the budget ship without moving any key, and it
    keeps this function free of a device parameter no caller would pass.
    """
    classes = max(1, int(classes))
    # ONE reduction (hostfacts): quota AND affinity AND host count, floored.
    # The old branch took floor(quota) when a quota existed, ignoring the
    # affinity mask — a fourth answer to "how many cores may I use".
    allowance = hostfacts.cpu_allowance()
    cores = allowance.whole_cores
    basis = f"{allowance.basis}={allowance.cores:g} cores"
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
# The trace output -> the shipped document
# ---------------------------------------------------------------------------


def serialize_declaration(declaration: "GraphClassDeclaration") -> str:
    """Canonical wire form of one already-validated TCG declaration."""
    payload = {
        "graph_class": declaration.graph_class,  # tcg-vocab: declaration field
        "target": declaration.target,
        "graph": dict(declaration.graph),
        "graph_witness": declaration.graph_witness,
        "range_digest": declaration.range_digest,
        "fork": [[name, value] for name, value in declaration.fork],
        "class_dims": [[name, value] for name, value in declaration.class_dims],
        "strict": declaration.strict,
        "lora_bucket": declaration.lora_bucket,
        "literal_values": declaration.literal_values,
        "placement": list(declaration.placement),
        "class_hash": declaration.class_hash,
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


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

    Read at the moment of the fan-out, so the parent's OWN residents (it is
    serving throughout a boot) are already excluded — the children compete for
    what is left, not for the nameplate capacity.
    """
    free = hostfacts.free_vram_bytes()
    return max(0, int(free)) if free is not None else 0


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
    must stay that. The dominant cost is the loaded AOTI packages plus device
    code, per-runner workspace and load-time buffers, **none of which appear in
    the artifact** — so sizing this budget against an artifact size, a literal
    size or an allocator high-water would under-count badly and in the direction
    that OOMs. `total - free` is the only input that sees all of it.

    Two conventions, matching `mint_workers`' RSS bank: the number is MONOTONE
    (it only ever tightens — see `_run_children`), and the reason STATES ITS
    BASIS, so a measured bound and an absent one can never be read as the same
    claim.
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

    The FIRST child is a probe: it runs alone and reports what one child costs
    the card (`TraceReport.device_peak_bytes`, the whole process footprint
    including its CUDA context). Every remaining child is admitted against that
    measurement and the card's real free bytes. Nothing is estimated, nothing is
    configured, and a card that reports nothing keeps the unbounded width — an
    absent measurement must not be able to throttle a pod.

    Deliberately NOT a serialization: the width reached here is reported and
    asserted in tests, and the probe costs exactly one child's latency — on a
    card with room, wave 2 carries the whole remainder.
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
    cfg: CompileSpec,
    slots: Mapping[str, MintSlot],
    declared_hint: int,
    work_root: Path,
    memo_dir: Optional[Path] = None,
    device: int = -1,
    workers: int = 0,
    python: str = "",
    trust_memo: bool = True,
    emitted_by: str = "",
) -> keyset.DerivedKeySet:
    """Derive a family's ``cg-key-v1`` key set from code alone, and RECORD it.

    The mint lane's producer for :mod:`gen_worker.keyset`. Every successful
    derivation writes the closure into ``memo_dir``'s ``cg-keyset-v1.json``, so
    the pod that traced is also the pod that emits the document a fresh pod will
    read (§4.28: minting is a side effect of serving, and so is this).

    ``trust_memo=False`` forces the traces even when the cache already answers
    and then RULES on what it held — the verify posture, never the default.

    ``declared_hint`` is the parent's read of the declaration's class-row count
    — ``len(aot_declaration.compiled_graph_plans(decl))``, which needs no pipeline. It
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
            f"family {family!r} declares no graph classes; a compiled graph with no "
            f"class set has no identity (pgw#716/#758)")

    try:
        digest = keyset_closure.closure_digest(
            family,
            cfg,
            function=function,
            slots=slots,
            modules=modules,
        )
    except keyset.KeySetError as exc:
        raise BootKeyUnavailable("closure_unavailable", exc.detail) from exc

    # THE CACHE PATH — milliseconds, and no trace at all. Identical in substance
    # to what a SHIPPED key set buys a fresh pod (`keyset.boot`); the only
    # difference is who derived the hashes. `boot_adopt` reaches the shipped and
    # cached routes on its own and only calls this deriver when neither
    # answered, so `trust_memo=True` here now means "a caller asked for a
    # derivation and would still accept this machine's own cached one".
    if trust_memo:
        cached = keyset_store.class_hashes(digest, cache_dir=memo_dir)
        if cached:
            entry_keys = keyset.fold_entry_keys(cached, family=family)
            wall_ms = int((time.monotonic() - t0) * 1000)
            logger.info(
                "boot-key: %d key(s) from this machine's cache in %d ms — no "
                "trace (closure %s)", len(entry_keys), wall_ms, digest)
            return keyset.DerivedKeySet(
                entry_keys=entry_keys,
                source=keyset.KeySource.MEMO,
                closure=digest,
                wall_ms=wall_ms,
                memo=keyset.MemoVerdict.HIT,
                width_reason="cache hit — no trace child was spawned")

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
            f"no TCG declarations: {first.detail[:600]}")

    drifted = sorted({
        r.code_digest for r in reports
        if CODE_DIGEST and r.code_digest and r.code_digest != CODE_DIGEST})
    if drifted:
        raise BootKeyUnavailable(
            "code_drift",
            f"trace child ran contract source {drifted!r} while this parent "
            f"runs {CODE_DIGEST!r} — the key would name graphs this process's "
            f"code did not describe (pgw#840's failure, at the boot seam)")

    declarations: Dict[str, str] = {}
    nodes: Dict[str, int] = {}
    trace_ms: Dict[str, int] = {}
    for report in reports:
        declarations.update({str(k): str(v) for k, v in report.declarations.items()})
        nodes.update({str(k): int(v) for k, v in report.nodes.items()})
        trace_ms.update({str(k): int(v) for k, v in report.trace_ms.items()})

    # THE completeness proof, and it never consults a parent-side guess: every
    # child reports how many classes the WHOLE declaration produced on its own
    # composed pipeline, all of them must agree, and the union of the shares
    # must be exactly that many. A key that cannot name every class is a key a
    # mismatch cannot name.
    declared = sorted({int(r.declared_classes) for r in reports})
    if len(declared) != 1 or declared[0] <= 0:
        raise BootKeyUnavailable(
            "class_set_disagreement",
            f"the trace children do not agree on how many classes this "
            f"declaration produces ({declared!r}) — they composed different "
            f"pipelines, so their graphs are not one compiled graph's graphs")
    total = declared[0]
    duplicated = sum(len(r.declarations) for r in reports) - len(declarations)
    if len(declarations) != total or duplicated:
        raise BootKeyUnavailable(
            "class_set_gap",
            f"the trace children returned {len(declarations)} distinct class(es) "
            f"({duplicated} duplicated) of the {total} this declaration "
            f"produces — the shares do not reconstruct the class set")

    try:
        rows = keyset_emit.rows_from_declarations(declarations)
    except keyset.KeySetError as exc:
        raise BootKeyUnavailable("invalid_declaration", exc.detail) from exc
    class_hashes = keyset_emit.class_hashes_of(rows)

    # `trust_memo=False` is the VERIFY posture: trace anyway and rule on
    # whatever the cache held. It is the only way a stale row is caught before a
    # mint, and it is never the default (a caller that re-traced to check its own
    # cache would have no cache path at all).
    verdict = keyset.MemoVerdict.ABSENT
    if not trust_memo:
        held = keyset_store.class_hashes(digest, cache_dir=memo_dir)
        if held:
            if held != class_hashes:
                keyset_store.invalidate(memo_dir, digest)
                verdict = keyset.MemoVerdict.INVALIDATED
                logger.warning(
                    "boot-key: the cached key set for closure %s disagreed with "
                    "the fresh trace — invalidated, and the FRESH hashes are "
                    "what this key names", digest)
            else:
                verdict = keyset.MemoVerdict.VERIFIED

    # THE EMISSION (pgw#1327). A traced closure is recorded as a `cg-keyset-v1`
    # document — this machine's own cache AND, since pgw#1353, the platform's
    # durable root, which is what carries it to the next pod of this endpoint.
    #
    # BEFORE THE FOLD, and that ordering is load-bearing (pgw#1353). The document
    # is machine-INDEPENDENT by construction — TCG class hashes and nothing
    # folded — while `fold_entry_keys` restates this process's `sm` and refuses
    # `no_runtime_sm` when it cannot read one. With the fold first, a producer
    # that traced all 36 classes correctly on a box with no usable card threw
    # every one of them away at the last step: the mint-lane emitter
    # (`scripts/emit_cg_keyset.py --derive`) could never write a document at all,
    # which is exactly how it was found. Recording first means a failed fold
    # costs this BOOT its adoption and costs the fleet nothing.
    keyset_emit.record_closure(
        memo_dir, digest, rows,
        family=family,
        function=function,
        tcg_version=keyset_closure.tcg_version(),
        emitted_by=emitted_by or "gen_worker.boot_key.derive",
    )

    entry_keys = keyset.fold_entry_keys(class_hashes, family=family)
    wall_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "boot-key: %d key(s) in %d ms — %s, cache=%s",
        len(entry_keys), wall_ms, width.reason, verdict.value)
    return keyset.DerivedKeySet(
        entry_keys=entry_keys,
        source=keyset.KeySource.TRACED,
        closure=digest,
        wall_ms=wall_ms,
        memo=verdict,
        workers=width.workers,
        width_reason=width.reason,
        traced=len(class_hashes),
        trace_ms={
            keyset.parse_graph_class_name(k): v for k, v in trace_ms.items()},
        nodes={keyset.parse_graph_class_name(k): v for k, v in nodes.items()},
    )


def prop_economy(reports: Sequence[TraceReport]) -> Dict[str, Any]:
    """The prop/export ratio, aggregated off whatever the children measured.

    ``export_s - prop_s`` is the saving one-export-N-props would buy. This
    function REPORTS it and decides nothing — the collapse itself may only be
    built once this ratio is measured on a real family's graphs rather than
    bounded by an off-pod probe.
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
    "PACKAGE_ROOT",
    "PoolWidth",
    "SERVING_HEADROOM_CORES",
    "TRACE_CHILD_MODULE",
    "TraceJob",
    "TraceReport",
    "child_argv",
    "child_env",
    "concurrency_budget",
    "derive",
    "free_device_bytes",
    "prop_economy",
    "serialize_declaration",
    "shares",
    "trace_workers",
]
