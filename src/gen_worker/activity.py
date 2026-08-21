"""Generic worker-activity progress (gw#601 / th#929).

The worker reports whatever internal job it is doing — self-mint compile,
warmup — as ActivityUpdate envelopes on the existing worker->hub stream.
Liveness contract: while an activity runs, seq keeps advancing (phase
transitions, step advances, or watchdog heartbeats around long silent calls);
the hub enforces ONE generic stall rule on silence. A silent death is a bug
by contract: terminal FAILED carries the exception.

Kind/phase strings are wire-shared with tensorhub
(internal/orchestrator/grpc/worker_activity.go) — keep them identical.

Without a bound transport sink (cozy-local, tests) reports land on the
logger, which IS the local progress UI.

ie#522: process CPU seconds alone are NOT liveness evidence. Two real
activities are honestly alive while burning near-zero CPU on the reporting
process: (1) an I/O-bound network model fill (tens of GB over a real link —
CPU-light by design); (2) inductor compile phases that fork subprocess compile
workers (torch's async_compile), whose burn never reaches this process's own
`time.process_time()`. Both read as "stalled" to the hub's th#965 layer-3 rule
after 10 minutes, mid genuine progress. So `_process_cpu_evidence` sums live
(not just reaped) child-process CPU via psutil, and
`note_progress()` lets an I/O callback (model-download byte ticks) heartbeat
the current activity directly — proof-of-life from genuine external
progress, independent of the CPU-sampling watchdog thread entirely.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from types import TracebackType
from typing import Awaitable, Callable, Optional

import psutil

from . import progress as progress_mod
from . import warm_spans
from .pb import worker_scheduler_pb2 as pb

logger = logging.getLogger(__name__)

KIND_SELF_MINT_COMPILE = "self_mint_compile"
#: The RUNNING setup+warmup activity — opened by :func:`begin`, ended when the
#: load window closes. An eager release's boot load runs under this kind
#: through the same executor call site (th#1021).
KIND_WARMUP = "warmup"
# pgw#1067 (worker half of th#1723): the boot-forward roll-up is a countable
# EVENT about a span that already ended, and it needs its OWN kind. The hub
# keys `info.Activities` on kind alone, and the `warmup` slot feeds
# `SelfMintActivityRunning` (stall monitor, cap turnover, unservable reaper) and
# `InFlightMintKinds` — so emitting the terminal roll-up as `warmup` made a
# still-loading worker read as finished to all of them. No hub change: every
# kind is stored generically and served at
# GET /v1/admin/worker-activity-events?kind=.
KIND_WARMUP_SUMMARY = "warmup_summary"
# pgw#680: one tenant request hit fail-on-recompile on a compiled lane and
# was served eager; phase carries the guard-reason class token, detail the
# full confession. The hub accepts any kind and logs completions verbatim,
# so top-N reasons are countable per (worker -> release, SKU) hub-side.
KIND_GUARD_MISS = "guard_miss"
# pgw#756: the guard-closure classifier flagged guards it cannot tie to the
# declared contract. ADVISORY — the mint completes and publishes; the rows
# ride the compiled graph's guard manifest. Countable so a real leak class surfaces as
# a trend hub-side instead of as a fleet-wide mint refusal (pgw#691/#733).
KIND_GUARD_LEAK = "guard_leak"
# pgw#916: the AOT arm's counterpart of `guard_miss` — one tenant request
# arrived at a graph specialization the armed compiled graph does not cover, was NAMED at ingress
# and served eager. `guard_miss` is a dynamo concept (a torch guard fired), so
# an AOT-armed pod produced no shape-gap fact at all and the hub could not
# count AOT coverage holes the way it counts dynamo ones. `phase` carries the
# ingress reason token (`no_entry_admits` / `entry_ambiguous`), `detail` names
# the family, the target, the missing DECLARED SPECIALIZATION and the compiled graph.
KIND_SHAPE_GAP = "shape_gap"
# pgw#760 (error-visibility doctrine): a fail-soft outcome that changes what
# or how this worker SERVES — or that a hub decision depends on — must ride
# a typed event, never only a log line (hub-spawned workers expose no
# stdout). One shape everywhere: ``emit_event(kind, detail, phase=reason)``
# with ``phase`` a short stable reason token (countable hub-side) and
# ``detail`` naming the identifiers (ref/function/label/request) + the
# exception. Fail-soft BEHAVIOR is unchanged — these are confessions.
KIND_SERVE_DEGRADE = "serve_degrade"
# pgw#1142 / §4.32 item 4: the OPERATOR's eager-only order changed state on this
# worker. Two phases, both transitions and neither a degradation:
# `eager_only_engaged` (compiled serving suppressed — armed compiled graphs stay armed and
# are not called) and `eager_only_released`. Deliberately its own kind rather
# than a `serve_degrade` phase: every other member of that kind is something
# that went wrong, and an operator exercising a supported control must not land
# in a defect population the fleet reads as breakage.
KIND_SERVE_POSTURE = "serve_posture"
KIND_LORA_HYGIENE = "lora_hygiene"
# pgw#794: the serve-side adapter-fidelity gate. `phase=refused` is a
# fail-CLOSED decision (the request also gets a typed error); `phase=degraded`
# is the gray band — served, but the delta measurably lost direction to the
# target grid. `detail` carries the whole-adapter cosine, the grid judged, and
# the worst per-module rows, so a release shipping an adapter its own serving
# dtype destroys is countable hub-side without the pod's stdout.
KIND_LORA_FIDELITY = "lora_fidelity"
# pgw#817: the ADOPTION numerics gate — a compiled graph about to arm is run against the
# eager forward it replaces and judged on the shared verdict ladder. Same two
# phases and the same fail-closed shape as `lora_fidelity`, one population
# down: `phase=refused` means the compiled graph did NOT arm and this pod serves eager;
# `phase=degraded` means it armed inside the gray band. pgw#814 measured a
# real DEGRADED artifact (flux2 w8a8 whole-graph, cos 0.931-0.973 against
# eager) that nothing in the worker would have noticed, which is why the
# verdict has to reach the hub whether or not it refuses.
KIND_COMPILED_GRAPH_NUMERICS = "compiled_graph_numerics"
# pgw#1036: modular-pipeline hydration provenance — one event per slot load,
# detail lists component<-local_source pairs (the hydration guard's proof that
# every weight came from OUR tree, bankable hub-side; phase=hydrated).
KIND_MODULAR_HYDRATION = "modular_hydration"
# pgw#1048: a non-modular composition named a component that is in neither the
# materialized tree nor the dispatched injection. `phase=refused` — the load
# did NOT happen and will not be retried for this dispatched identity, because
# a refetch cannot widen a manifest the hub narrowed (th#1711/th#1715). The
# detail names missing/expected/injected: without it the hub sees only
# `OSError: Error no file named config.json found in directory <root>`, which
# names neither the component nor the cause, and which pgw#1047 watched a pod
# retry for 9 minutes.
KIND_COMPONENT_MISS = "component_miss"
# pgw#1094: the serve-path output-integrity floor's verdict on one output.
# `phase` is the verdict token — `noise` / `blank` / `nonfinite` are fail-CLOSED
# (the request also gets a typed `OutputIntegrityError` and nothing is
# uploaded), `unmeasured` is the confession that the screen could not run and
# the output served anyway. PASS emits nothing: one row per served render buys
# no decision. `detail` carries the measured adjacent_frame_corr and
# frame_std_min, so ie#634's "corr 0.29 was uploaded and billed" is countable
# hub-side instead of invisible.
KIND_OUTPUT_INTEGRITY = "output_integrity"
# th#1867 deleted KIND_ENVELOPE_REFUSAL with the pre-stage envelope
# precondition that emitted it (pgw#1117/th#1777): it compared the bound
# artifact against a declared `vram_gb`, and that declaration is gone. The
# question it was really asking — "is this slot bound to the wrong artifact?" —
# is th#1913's, and it is answered against the CATALOG, not a card size.
KIND_ROTATION_PRELOAD = "rotation_preload"
KIND_CAPABILITY_RENEWAL = "capability_renewal"
KIND_RESIDENCY_FAULT = "residency_fault"
# pgw#1104: a serve-time recipe reported the lane it APPLIED to the weights
# (`gen_worker.report_applied_lane`), and `metrics.lane` now follows it instead
# of the binding. `phase` is the component, `detail` carries the applied lane
# body, the module counts and the BOUND lane it diverged from — so "which pods
# serve a lane their checkpoint does not name" is one query, not an inference
# from allocated-bytes rows.
KIND_APPLIED_LANE = "applied_lane"
# pgw#1043 §PRODUCTIZATION: the ATTENTION path setup() actually installed
# (`gen_worker.report_applied_attention`). A third axis beside `lane` and
# `serving_mode`, for the same reason those two are separate: the lane string
# is a numerics descriptor and cannot say which key blocks were read, nor carry
# the MEASURED density the wall is a function of. `phase` is the component;
# `detail` carries mode, k, block size, density, selector and the bound index
# artifact — so "which pods served sparse, at what density, off which head" is
# one query.
KIND_APPLIED_ATTENTION = "applied_attention"
# pgw#1116: the boot-time adopt decision (§4.27 steps 1-3). EVERY boot of a
# compiled family emits exactly one of these — hit, miss, and each named
# refusal — because the alternative is what pgw#1108's pod proof measured: a
# merged fix and a published wheel both sailed past a boot-adopt that never
# called the hub, since all of `BootAdoptOutcome`'s eight refusals presented
# identically as "a pod that quietly self-minted". `phase` is the gate token
# (`boot_adopt.REASONS`, countable hub-side); `detail` names family, function,
# derived key and the sentence; `duration_ms` is the derivation's own wall.
KIND_BOOT_ADOPT = "boot_adopt"
# pgw#1441: the PER-BOOT roll-up, and it needs its own kind for exactly the
# reason `warmup_summary` above does. `boot_adopt`'s `phase` is a PER-KEY gate
# token (`hit`/`miss`/`no_export_declaration`, one row per graph); pgw#1371's
# adopt outcome is a PER-BOOT verdict over all of them. Emitting both under one
# kind gives that kind two vocabularies, and the hub keys `info.Activities` on
# kind alone — so `count(*) where kind='boot_adopt' and phase='reused'` reads 0
# on every pre-#1371 pod, which is indistinguishable from "nothing reused".
# Same trap as pgw#1067's, caught before it shipped rather than after.
#
# `phase` is `reused` (this boot adopted everything and mints nothing) /
# `minting` (holes remain) / `empty_lane` (nothing was marked: an eager endpoint
# saying so) / `marks_unmatched` (pgw#1534 — a `ctx.compile`-ed module fits NO
# record in its own lane, so the compiled path is off and NO mint can turn it
# on) / `all_ambiguous` (pgw#1534 — every record was disarmed as a literal-twin:
# the artifacts are present and VALID, and the dispatcher refuses to guess
# between two graphs with one tensor structure, so it disarms both).
# THE LAST THREE ALL CARRY `step=0, total_steps=0` and are three different facts.
# That is exactly why they are three phases: while they all read `empty_lane`,
# no query could find a pod whose compiled path had silently switched off, and
# the two zeros were measured on a box with 14 valid artifacts in its store.
# THE COUNTS ARE NUMERIC, not prose:
# `step` = graphs adopted, `total_steps` = graphs claimed, so the reuse ratio
# is a query rather than a regex over `detail`. `detail` keeps the sentence.
KIND_BOOT_ADOPT_SUMMARY = "boot_adopt_summary"
# pgw#1328: the ADOPT-ONLY role's answer where an eager-capable pod would have
# served eager and minted (§4.28). It is the only thing that role produces on a
# miss, so it must be readable off-pod or the role is unobservable. `phase` is
# the `serve.refusal.MissKind` token (countable, closed, and each one carries a
# ROUTE/REFUSE disposition decided once in `serve.refusal.DISPOSITIONS`);
# `detail` names the function, family, derived key, the tcg#37 selection
# outcome and the ranked candidate classes that missed. Deliberately NOT a
# `serve_degrade` phase: nothing degraded — the pod answered completely and the
# fleet is expected to act on it.
KIND_ADOPT_REFUSED = "adopt_refused"
# pgw#1271: the boot-key MEMO honesty verdict, ruled at the one moment a pod
# holds both halves — what the memo answered at boot, and what this mint just
# traced. `boot_key`'s memo path SKIPS THE TRACES and produces the `graph` axis
# of every published `cg-key-v1`, and the module header's promise ("Honesty is
# enforced, not trusted... A wrong key is never produced") was carried by
# `assert_memo_honest`, which had zero `src/` callers: the check existed, read
# as enforcement, and never ran on a fleet pod. A typed event rather than a log
# line because a hub-spawned pod's stdout goes nowhere (pgw#760/#1116) and a
# dishonest memo is a KEY-SPACE fault — the fleet needs it countable.
# `phase` is `memo_dishonest` / `memo_unrulable`; `detail` carries
# `assert_memo_honest`'s own sentence, which names every disagreeing class.
KIND_BOOT_MEMO = "boot_memo_honesty"
# th#1322: JIT (dynamo/inductor) compile duration, as a typed numeric event —
# never a log line, because a hub-spawned pod has no reachable stdout (pgw#760)
# and this is the one number an AOT-vs-JIT comparison needs. Same shape as
# `aot_mint_phases` deliberately:
# `phase=minted` carries the mint's roll-up in `duration_ms`, `phase=shape:<WxH>`
# / `phase=child:<phase>` carry the spans inside it, so comparing the two mint
# routes is ONE grouped query over worker_activity_events rather than a regex
# over free text on one side and a log grep on the other.
KIND_JIT_COMPILE = "jit_compile"
# pgw#805: the AOT (torch.export + AOTInductor) mint's duration, same shape.
# The hub has read this kind since th#1322 (`worker_activity_events`,
# /v1/admin/worker-activity-events?kind=aot_mint_phases) and the table stayed
# EMPTY on both stacks because no producer existed: `aot_mint` was reachable
# only from `python -m gen_worker.aot_mint`, and no serving-pod code path
# imported it. `phase=minted` carries the roll-up, `phase=entry:<name>` one
# declared graph specialization, `phase=stage:<name>` a mint-wide span (package/seal).
KIND_AOT_MINT = "aot_mint_phases"
# pgw#1134: a MEASURE-ONLY run (`gen_worker.measure_child`) — an operator
# exporting and compiling a declared class set to find out what it costs, so a
# declared mint blocker whose exit criterion is a number can be closed with
# one. Deliberately NOT `aot_mint_phases`: nothing was minted, nothing was
# published, and a measurement counted as a mint would be the first row of a
# lie. `phase` is the outcome (`measured` / `measured_export` / a
# `measure_child.REASONS` token), `duration_ms` the run's wall, `detail` the
# peaks in the mint's own vocabulary.
KIND_MEASURE_ONLY = "measure_only"
# pgw#1309: ONE compile child, at spawn and at collect. Same shape as
# `aot_mint_phases`' child rows (`phase=child:start` / `child:finish`), its own
# KIND because the two questions the pgw#1232 legs ask of it are PRESENCE
# questions — "a compile ran, in a child, under this serving pid" on leg A and
# "no compile ran at all" on B/C — and an absence assertion over a kind that
# also carries width, ledger and per-class rows answers neither. `detail`
# carries `child_pid` / `child_ppid` / `serving_pid` / `role` plus the packed
# `compiled_graph_keys`, because no ActivityUpdate field holds a pid and the
# th#1839 route serves `detail` verbatim (no hub change).
KIND_COMPILE_CHILD = "compile_child"
# pgw#1309: this process's own pid and role, stated once per sink bind. The
# serving pid used to exist only in `entrypoint._startup_payload` and
# `postmortem.write_boot_record`, both pod-local, so "no compile under the
# serving PID" was unassertable off-pod even with the child rows above.
KIND_PROCESS_ROLE = "process_role"
# pgw#1351: ONE snapshot pull, rolled up at completion — what the pod actually
# TRANSFERRED, against what the snapshot weighs. e2e#1892 tried to measure
# warm-boot dedup across three pods and could not: nothing on this channel
# reported wire bytes, and `tree on disk` is identical on a pod that fetched
# everything and a pod that fetched nothing. Its own KIND because the question
# is a RATIO over one pull — `fetched_bytes` against `tree_bytes`, with
# `resident_objects` as the evidence — and folding it into a boot or download
# activity would put the numerator and the denominator in different rows.
# `detail` carries the counts as `k=v` (see `snapshot_pull.py`); no
# ActivityUpdate field holds them and the th#1839 route serves `detail`
# verbatim, so this needs no hub change.
KIND_SNAPSHOT_PULL = "snapshot_pull"
# pgw#1541: the boot SNAPSHOT CENSUS — which trees this pod holds and whether
# each is PINNED. Its own kind because it answers a question no other row can:
# a projected tree without its manifest pin is unreadable by the streaming
# engine (the only reader of pointer stubs), and the pod then serves the eager
# bridge a stubbed tree and reports `header too large` about an intact
# checkpoint. pgw#1536 added that census as LOG LINES ONLY, which made it
# invisible to every DB-reading harness and — worse — DELETED AT TEARDOWN: a
# rental that ends by tearing down silently loses the exact fact the census
# exists to preserve, and its "free on any run" value held only for a runner
# who knew to pull `cozy logs` while the pod was alive.
#
# One row, emitted ONLY when at least one tree is unservable, because that is
# the incident fingerprint and it must outlive the pod. `step`/`total_steps`
# carry unservable/total so the state is a NUMERIC query and not a string
# search; per-tree detail stays in the logs, where volume is free.
KIND_SNAPSHOT_CENSUS = "snapshot_census"
# pgw#1355: the COLD BOOT's own decomposition — per-stage spans carrying
# start/end offsets from OS process start, plus one terminal roll-up. Its own
# KIND because it is the only row that answers "where did this pod's cold start
# go" without a join: e2e#1892 assembled an 877 s cold serve out of FOUR event
# kinds by hand (request-row timestamps, `worker_boot_phases` load spans,
# `boot_adopt.duration_ms`, `compute_ms`/`finalize_ms`), and the largest stage
# of all — pgw#1353's 805 s key-set derive — had no span on any channel.
# `phase=stage:<name>` is one span, `phase=ready` the roll-up carrying the
# packed table; `duration_ms` is that stage's wall, and the roll-up's is
# wall-to-ready. The stage vocabulary is CLOSED (`boot_stages.Stage`) because a
# renderer in another repo binds to these tokens. `detail` is the same `k=v`
# grammar `snapshot_pull` uses, served verbatim by the th#1839 route, so this
# needs no hub schema change. See `boot_stages.py` for why the total is a UNION
# and never a sum.
KIND_BOOT_STAGES = "boot_stages"
# pgw#1421: the ENGINE-HOSTED tier's boot — an external engine binary
# (`llama-server`, `vllm serve`) started, health-waited and stopped by the
# worker. Its own KIND because nothing else on the pod can answer "why is this
# endpoint not serving yet" for a tier whose work happens in another PROCESS:
# `boot_stages` decomposes THIS process's cold start and an engine subprocess
# is invisible to every one of its spans, exactly as it is invisible to torch's
# allocator. The phase vocabulary is CLOSED (`serving.engine_runtime`
# ENGINE_PHASES) so a reader can group on (kind, phase) rather than regex a
# sentence; `duration_ms` on the `engine_healthy` row is the measured boot wall.
# A DEGRADED rung is deliberately NOT reported here — it rides
# KIND_SERVE_DEGRADE, which is the countable quality-confession channel
# (th#2075's ingest), so the two questions stay two rows.
KIND_ENGINE_BOOT = "engine_boot"
# pgw#1455: the serving-path weight download's BYTE POSITION, in integral MiB
# on the typed `step`/`total_steps` columns. Its own KIND because it is the only
# fact that separates "downloading" from "downloading is PROGRESSING": rental #6
# parked a request on `model_download_pending` for 49 minutes with 0 rows in
# `worker_activity_events`, and the scheduler's `attempts` counter — its own 1 Hz
# placement poll — carries no download information at all. `snapshot_pull` is a
# terminal roll-up of a pull that ALREADY FINISHED, so it answers nothing about
# one in flight; `warmup`'s byte counter rides whichever activity happens to be
# open, which for a steady-state materialization is none. The hub keys
# `info.Activities` on kind alone and reads THIS kind's `step` into the
# `model_download_pending` decline explain (th#2191). Phases are a closed
# vocabulary (`weight_position.PHASE_*`).
KIND_WEIGHT_FETCH = "weight_fetch"
# pgw#1613: the boot materialization SPAN — one open activity for the whole
# fetch loop, from the first ref to the last byte. It is not telemetry: the
# compute-child watchdog (`procsplit/parent.py` `_hang_verdict`) arms on a
# silent event loop and then decides by what is OPEN, so a fill that declares
# nothing takes the `loop_wedged_no_activity` branch and is SIGKILLed while it
# is provably burning CPU. Measured twice on minimax-h3 (~105 GB) at two
# different pins: killed at 356 s and 687 s, `cause=watchdog_hang`, oom_kill=0.
# Its OWN kind, deliberately NOT `weight_fetch`: that kind's rows are the
# closed-vocabulary BYTE POSITIONS (`weight_position._emit`, fire-and-forget
# `emit_event` — which is exactly why nothing was ever open), and a span with
# an empty phase mixed into them would corrupt the one signal th#2191 reads.
# Nothing hub-side keys on this kind; like `warmup_summary` before it, a new
# kind is stored and served generically and needs no hub change.
KIND_BOOT_MATERIALIZE = "boot_materialize"
# th#1322: the roll-up phase both mint routes report their TOTAL under. A
# reader groups on (kind, phase) and must never sum a roll-up together with
# its own children.
PHASE_MINTED = "minted"

PHASE_LOAD = "load"
PHASE_TRACE_GRAPH = "trace_graph"
PHASE_INDUCTOR_COMPILE = "inductor_compile"
# pgw#989. Re-exported (defined in `warm_spans`, which the mint child can
# import without protobuf) so the vocabulary is enumerable from one module.
PHASE_ROUTER_DRAIN = warm_spans.PHASE_ROUTER_DRAIN
PHASE_WARMUP_FORWARD = "warmup_forward"
PHASE_SEAL_PUBLISH = "seal_publish"
# gw#612: post-proof tail — sibling-lane resolution, publish decision,
# residency/target bookkeeping through readiness. Phases are logged
# verbatim hub-side (only kinds are enumerated in worker_activity.go).
PHASE_FINALIZE = "finalize"

# Default watchdog cadence; the hub's stall rule (~10 min) tolerates many
# missed beats.
HEARTBEAT_INTERVAL_S = 60.0
# Minimum evidence advance (process CPU seconds) per interval for a heartbeat:
# a hung (blocked/deadlocked) call stops accruing CPU and the beat stops with
# it, which is exactly the silence the hub enforces on.
EVIDENCE_EPS = 0.05

_lock = threading.Lock()
_seq = 0
_sink: Optional[Callable[[pb.ActivityUpdate], None]] = None
_current: Optional["Activity"] = None


def bind_sink(
    emit: Callable[["pb.WorkerMessage"], Awaitable[None]],
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Route reports onto the worker->hub stream: emit is the async
    WorkerMessage sender, loop the transport loop. Thread-safe emission."""
    def sink(update: pb.ActivityUpdate) -> None:
        async def _ship() -> None:
            await emit(pb.WorkerMessage(activity_update=update))
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            loop.create_task(_ship())
        elif not loop.is_closed():
            asyncio.run_coroutine_threadsafe(_ship(), loop)
    global _sink
    with _lock:
        _sink = sink


def reset_for_tests() -> None:
    """Drop the bound sink and any in-flight activity. Test-only.

    The sink is process-lifetime in production (one worker, one transport), so
    nothing unbinds it. In a suite that is a leak: a bound sink outlives the
    ``Worker`` whose queue it feeds, and under ``--dist loadfile`` the next FILE
    in the same process reports into a queue nothing drains (pgw#1024).
    """
    global _sink, _current
    with _lock:
        _sink = None
        _current = None


def _next_seq() -> int:
    global _seq
    with _lock:
        _seq += 1
        return _seq


def _emit(update: pb.ActivityUpdate) -> None:
    with _lock:
        sink = _sink
    try:
        if sink is not None:
            sink(update)
        else:
            state = pb.ActivityState.Name(update.state)
            logger.info(
                "[activity] %s %s %s/%s %s%s %s", update.kind, update.phase,
                update.step, update.total_steps, state,
                # th#1322: the measured span belongs in the local UI too — the
                # sinkless path IS the cozy-local progress display.
                f" {update.duration_ms / 1000:.1f}s" if update.duration_ms else "",
                update.error or update.detail,
            )
    except Exception:  # reporting must never break the work it reports on
        logger.debug("activity report dropped", exc_info=True)


class Activity:
    """One running activity. Use begin() / the context manager `running()`."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        #: This activity's own scope (pgw#894). Process-local and stable for
        #: the activity's life: it keys the progress registry so a counter fed
        #: by unrelated work can never answer THIS activity's stall question.
        #: Not on the wire — the hub still identifies an activity by
        #: (worker, kind, phase) — so it costs no protocol change; what it
        #: buys is that the number the beat CARRIES describes this activity.
        self.id = f"{kind}:{_next_seq()}"
        self._phase = ""
        self._step = 0
        self._total = 0
        self._done = False
        # name -> (phase that registered it, counter). PHASE-scoped: see counter().
        self._counters: dict[str, tuple[str, progress_mod.Counter]] = {}

    def counter(
        self, name: str, unit: str, total: float = 0.0,
    ) -> progress_mod.Counter:
        """Register-or-get a progress counter owned by this activity's CURRENT
        PHASE — finished when the phase changes, and again when the activity
        ends (gw#621).

        pgw#962: activity-scoped lifetime was too long. `self_mint_compile`
        spans load -> warmup_forward -> load for every function on the pod, so
        `infer:steps` stayed open and frozen after its warmup finished; the
        next phase has no counter producer, `progress.self_diagnosis()` is a
        registry-wide min-age query, and the beat confessed `self_stalled`
        about work that had already succeeded. The hub kills on that confession
        without waiting out its own window. Producers re-acquire per tick, so a
        counter that is still being fed is simply re-registered under the new
        phase."""
        c = progress_mod.counter(name, unit, total, owner=self.id)
        self._counters[name] = (self._phase, c)
        return c

    def _finish_counters(self, keep_phase: Optional[str] = None) -> None:
        """Close counters the given phase does not own (all of them when None)."""
        for name, (phase, c) in list(self._counters.items()):
            if keep_phase is not None and phase == keep_phase:
                continue
            c.finish()
            del self._counters[name]

    def _report(self, state: "pb.ActivityState", error: str = "", detail: str = "") -> None:
        _emit(pb.ActivityUpdate(
            kind=self.kind, phase=self._phase, step=self._step,
            total_steps=self._total, seq=_next_seq(), state=state,
            error=error, detail=detail,
            updated_at_unix_ms=int(time.time() * 1000),
        ))

    @property
    def phase_name(self) -> str:
        """The phase this activity is reporting right now (pgw#1118). Read by
        the setup boundary to name WHICH pass failed, so a warm/compile fault
        can be typed as the release's instead of the caller's."""
        return self._phase

    def phase(self, phase: str, step: int = 0, total: int = 0) -> None:
        self._phase, self._step, self._total = phase, step, total
        self._finish_counters(keep_phase=phase)
        self._report(pb.ActivityState.ACTIVITY_STATE_RUNNING)

    def heartbeat(self) -> None:
        """Re-report the current phase with a fresh seq (liveness proof)."""
        self._report(pb.ActivityState.ACTIVITY_STATE_RUNNING)

    def progress_beat(
        self, snap: "progress_mod.Snapshot", self_stalled: bool = False,
    ) -> None:
        """One counter-carrying RUNNING update (gw#621), emitted from the
        10s app beat. The hub judges liveness by counter advancement; a
        self_stalled=True beat is the typed confession it kills on."""
        if self._done:  # racing completion: never re-open a terminal activity
            return
        _emit(pb.ActivityUpdate(
            kind=self.kind, phase=self._phase, step=self._step,
            total_steps=self._total, seq=_next_seq(),
            state=pb.ActivityState.ACTIVITY_STATE_RUNNING,
            counter=snap.name, counter_unit=snap.unit,
            counter_done=snap.done, counter_total=snap.total,
            rate_per_s=snap.rate_per_s, self_stalled=self_stalled,
            stalled_for_ms=int(snap.age_s * 1000) if self_stalled else 0,
            updated_at_unix_ms=int(time.time() * 1000),
        ))

    def note(self, detail: str) -> None:
        """One RUNNING report carrying a typed detail (pgw#672: the
        degrade-to-eager confession rides the activity stream so the hub
        sees WHY a compiled lane is now serving eager — loud, never a
        terminal state)."""
        if self._done:
            return
        self._report(
            pb.ActivityState.ACTIVITY_STATE_RUNNING, detail=detail[:2000])

    def completed(self) -> None:
        if not self._done:
            self._done = True
            self._report(pb.ActivityState.ACTIVITY_STATE_COMPLETED)
        _end(self)

    def failed(self, exc: BaseException) -> None:
        """The typed activity_failed terminal — a silent death is a bug."""
        if not self._done:
            self._done = True
            self._report(
                pb.ActivityState.ACTIVITY_STATE_FAILED,
                error=f"{type(exc).__name__}: {exc}"[:2000],
            )
        _end(self)

    def retrying(self, exc: BaseException, attempt: int, max_attempts: int) -> None:
        """gw#661: this attempt lost, but the work is contractually
        re-attempted — so the activity is still RUNNING, not FAILED.

        FAILED is the hub's terminal rung: ``lastWorkerProgressLocked``
        (th#1160) excludes failed activities from progress evidence, so a
        will-retry attempt that reports FAILED erases exactly the evidence
        that keeps a working pod alive. Measured 2026-07-25: 4 condemnations,
        4 self-mint compiles that then COMPLETED. Only exhaustion is failure.
        """
        if self._done:
            _end(self)
            return
        self._done = True
        self._report(
            pb.ActivityState.ACTIVITY_STATE_RUNNING,
            detail=(
                f"retrying (attempt {attempt}/{max_attempts}): "
                f"{type(exc).__name__}: {exc}"
            )[:2000],
        )
        _end(self)


#: pgw#1176 / th#1839: compiled-graph identity travels as TYPED WIRE FIELDS.
#:
#: ``ActivityUpdate.family`` / ``.compiled_graph_key`` / ``.graph_specialization`` (proto fields
#: 18-20) land in the hub's existing ``worker_activity_events`` columns. The
#: prose form they replace was measured: ``detail LIKE 'ref=%#'||compiled_graph_key||'
#: %'`` matched one of three rows correctly, because four emitters spelled
#: identity four ways. A fact that has to be regex-scraped out of a sentence
#: cannot be indexed, cannot be grouped, and returns NULL the first time the
#: formatting changes.
#:
#: ``proto/worker_scheduler.proto`` is a byte-for-byte vendored copy of
#: tensorhub's canonical one, gated on ``PROTO_DIGEST``, so field 19 cannot be
#: renamed here unilaterally: pgw#1278 renames the vocabulary AROUND it and
#: leaves the proto spelling alone. The single translation lives in
#: :func:`emit_event` below — one line, at the seam, never a second name in
#: the worker's own vocabulary. Field 19 becomes ``compiled_graph_key`` in the
#: proto lane (th#1947), and no caller here moves when it does.


def emit_event(
    kind: str, detail: str, phase: str = "", duration_ms: int = 0,
    *, family: str = "", compiled_graph_key: str = "", graph_specialization: str = "",
    step: int = 0, total_steps: int = 0,
) -> None:
    """One self-contained COMPLETED ActivityUpdate — a countable typed
    EVENT (pgw#680 ``guard_miss``), not a running activity.

    ``duration_ms`` (th#1322) is the MEASURED span of the work this event
    reports, taken from this process's monotonic clock. It lands in a numeric
    ``worker_activity_events.duration_ms`` column, so a duration reported here
    can be grouped and percentiled — which a number interpolated into
    ``detail`` cannot. Pass it whenever the emitting site already timed the
    work; leave it 0 when the event is not about a span (a refusal, a
    classification). Never pass a duration the caller had to guess: 0 honestly
    means "not measured here", and readers filter on ``duration_ms > 0``.

    Deliberately bypasses :func:`begin`: begin() swaps the module-global
    ``_current``, and ending an event would strand a concurrently open
    long-running activity (background mint) without its progress beat.
    ``step``/``total_steps`` (pgw#1441) carry a COUNT this event is about, in
    the typed numeric columns rather than interpolated into ``detail``. An
    event whose whole point is a ratio -- "N of M graphs adopted" -- is
    otherwise only readable by regexing a sentence, which is how a reader ends
    up building a metric on a column that renders 0 because nothing writes it.
    Leave both 0 when the event is not about a count.

    Thread-safe; without a bound sink it lands on the logger like every
    other report."""
    _emit(pb.ActivityUpdate(
        kind=kind, phase=phase[:300], seq=_next_seq(),
        step=max(0, int(step)), total_steps=max(0, int(total_steps)),
        state=pb.ActivityState.ACTIVITY_STATE_COMPLETED,
        detail=detail[:2000],
        family=str(family or "")[:200],
        # proto field 19 is still spelled `compiled_graph_key` — see the note above.
        compiled_graph_key=str(compiled_graph_key or "")[:200],
        graph_specialization=str(graph_specialization or "")[:300],
        duration_ms=max(0, int(duration_ms)),
        updated_at_unix_ms=int(time.time() * 1000),
    ))


def begin(kind: str, phase: str = "") -> Activity:
    global _current
    act = Activity(kind)
    with _lock:
        _current = act
    act.phase(phase) if phase else act.heartbeat()
    return act


def current_phase(phase: str, step: int = 0, total: int = 0) -> None:
    """Report a phase on the current activity; no-op when none is running.
    Setups serialize under the executor load lock, so one current is enough."""
    with _lock:
        act = _current
    if act is not None and not act._done:
        act.phase(phase, step, total)


def current_note(detail: str) -> None:
    """Attach a typed detail to the current activity (pgw#672 degrade
    visibility); no-op when none is running."""
    with _lock:
        act = _current
    if act is not None and not act._done:
        act.note(detail)


def _end(act: Activity) -> None:
    global _current
    with _lock:
        if _current is act:
            _current = None
    act._finish_counters()


def current() -> Optional[Activity]:
    with _lock:
        act = _current
    return act if act is not None and not act._done else None


def scoped_counter(
    name: str, unit: str, total: float = 0.0,
) -> "progress_mod.Counter":
    """The counter for ``name`` owned by the CURRENT activity, or an unowned
    process counter when none is open (pgw#894).

    For producers that are not themselves inside an activity method: a model
    download or a weight load belongs to whatever phase asked for it, and the
    fallback keeps library / CLI use — where no activity exists — working
    exactly as before.
    """
    act = current()
    if act is not None:
        return act.counter(name, unit, total)
    return progress_mod.counter(name, unit, total)


def on_beat() -> None:
    """Ride the 10s app heartbeat (lifecycle._heartbeat_loop, gw#621): while
    an activity is open and the progress registry has counters, emit one
    counter-carrying update per beat — frozen counters included (the hub's
    stall clock runs on non-advancement) — plus the typed self-diagnosis
    when even the freshest counter is stale past its per-phase window.
    Never raises; without a sink reports land on the logger."""
    try:
        act = current()
        if act is None:
            return
        # pgw#894: THIS activity's counters, not the registry's. The beat used
        # to attach `progress.freshest()` — the min-age counter anywhere in
        # the process — to whatever activity happened to be current, and the
        # hub advances an activity's `UpdatedAt` from a counter-name change or
        # value increase (`worker_activity.go:323-338`), which is the
        # timestamp its stall/condemnation path reads. So a serving request
        # deferred a background mint's stall verdict: 28 lines on the standing
        # chaos hub reported `infer:steps` under `self_mint_compile`, and line
        # 4542 declined a condemnation because that mint activity was "0s ago".
        snap = progress_mod.freshest(act.id)
        if snap is None:
            return
        act.progress_beat(
            snap,
            self_stalled=progress_mod.self_diagnosis(act.id) is not None)
    except Exception:
        logger.debug("progress beat dropped", exc_info=True)


# Floor between note_progress()-driven heartbeats: an active download can
# tick every few seconds (executor.py's _PROGRESS_EVENT_MIN_INTERVAL_S=5s);
# no need to re-report that often, just comfortably inside the hub's 10min
# stall window.
_PROGRESS_HEARTBEAT_MIN_INTERVAL_S = 5.0
_last_progress_heartbeat = 0.0


def note_progress() -> None:
    """Proof-of-life for the CURRENT activity from an external progress
    signal (model-download byte ticks, etc.) — an I/O-bound fill is
    CPU-light by design, so the watchdog's CPU-time evidence alone would
    read a healthy, slow-but-progressing network download as a stalled
    activity. Safe to call unconditionally from generic download code: a
    no-op when no activity is running, rate-limited otherwise."""
    global _last_progress_heartbeat
    now = time.monotonic()
    with _lock:
        if now - _last_progress_heartbeat < _PROGRESS_HEARTBEAT_MIN_INTERVAL_S:
            return
        _last_progress_heartbeat = now
        act = _current
    if act is not None and not act._done:
        act.heartbeat()


class running:
    """Context manager: begin() on enter; COMPLETED on clean exit, FAILED
    (carrying the exception) on raise."""

    def __init__(self, kind: str, phase: str = "") -> None:
        self._kind, self._phase = kind, phase
        self.activity: Optional[Activity] = None

    def __enter__(self) -> Activity:
        self.activity = begin(self._kind, self._phase)
        return self.activity

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        assert self.activity is not None
        if exc is not None:
            self.activity.failed(exc)
        else:
            self.activity.completed()
        _end(self.activity)


_this_process = psutil.Process()


def _process_cpu_evidence() -> float:
    """Process CPU seconds, counting live AND already-reaped descendants.

    torch's async_compile forks subprocess compile workers for inductor
    phases, so ``resource.getrusage(RUSAGE_CHILDREN)`` alone is useless for an
    in-flight compile burning real CPU in a still-running child; psutil reads
    /proc directly, so a live child's usage counts immediately.

    pgw#964: it took BOTH. Counting only the live tree makes this a sawtooth —
    a child's CPU leaves its own ``utime/stime`` and enters its parent's
    ``cutime/cstime`` the instant it is reaped, so a finishing compile worker
    SUBTRACTS its whole lifetime from the total. Every consumer here compares
    against a high-water mark, so that reads as a pod that stopped working. A
    process's CPU lives in exactly one of the two places and never both, so
    adding the reaped counters is a clean sum and makes this monotonic.

    Best-effort: a child that can't be inspected (raced exit, permissions)
    just doesn't contribute — never fatal to the heartbeat."""
    try:
        times = _this_process.cpu_times()
        total = float(times.user) + float(times.system)
        total += float(getattr(times, "children_user", 0.0) or 0.0)
        total += float(getattr(times, "children_system", 0.0) or 0.0)
    except psutil.Error:
        # Self is always readable in practice; if it somehow is not, fall back
        # to the interpreter's own accounting rather than to zero.
        total = time.process_time()
    try:
        children = _this_process.children(recursive=True)
    except psutil.Error:
        children = []
    for child in children:
        try:
            ct = child.cpu_times()
        except psutil.Error:
            continue
        total += float(ct.user) + float(ct.system)
        total += float(getattr(ct, "children_user", 0.0) or 0.0)
        total += float(getattr(ct, "children_system", 0.0) or 0.0)
    return total


def _process_io_evidence() -> float:
    """Process disk I/O bytes (read+write), MB granularity. Covers the
    on-disk model-LOAD phase (safetensors mmap/read + tensor
    materialization) — reproduced live: a worker was killed at
    `phase=load` after its network download had already finished, i.e.
    during the local-disk-to-GPU step, which is diffusers/torch/safetensors
    code gen-worker has no progress hook into. Process-level I/O accounting
    is a universal, non-invasive signal instead: real bytes move through
    the kernel's counters for this process regardless of which library
    triggered the read, so it needs no app-level instrumentation of
    third-party loading code at all. Unsupported platforms (no
    io_counters(), e.g. macOS) contribute 0 — never fatal."""
    try:
        io = _this_process.io_counters()
    except (psutil.Error, AttributeError, NotImplementedError):
        return 0.0
    return (io.read_bytes + io.write_bytes) / (1 << 20)


def default_evidence() -> float:
    """Combined default watchdog evidence: process+live-children CPU
    seconds PLUS process disk I/O megabytes. Either source alone advancing
    proves genuine life: a network download followed by on-disk model load
    is CPU-light throughout but moves real bytes the whole way (covers
    both halves of "load"); an inductor compile subprocess burns real
    (child) CPU while GPU sits idle and I/O sits flat; a true hang
    advances neither."""
    return _process_cpu_evidence() + _process_io_evidence()


class watchdog:
    """Bracket for a long call that may stay wire-silent (inductor compile,
    large fuse): a background thread samples an evidence counter every
    interval and heartbeats the activity ONLY while evidence advances. A hung
    call stops accruing evidence, the beat stops within one interval, and the
    hub's stall rule takes it from there.

    Default evidence is process+children CPU seconds plus process disk I/O
    (see default_evidence); pass a monotonic callable (e.g.
    compile-wall-seconds) for calls with a better signal."""

    def __init__(
        self,
        act: Activity,
        *,
        interval_s: float = HEARTBEAT_INTERVAL_S,
        evidence: Optional[Callable[[], float]] = None,
    ) -> None:
        self._act = act
        self._interval = interval_s
        self._evidence = evidence or default_evidence
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="activity-watchdog", daemon=True,
        )

    def _run(self) -> None:
        try:
            base = last = self._evidence()
        except Exception:
            base = last = 0.0
        while not self._stop.wait(self._interval):
            try:
                now = self._evidence()
            except Exception:
                continue
            if now - last >= EVIDENCE_EPS:
                last = now
                # gw#621: evidence advance is ALSO a registry counter, so the
                # 10s beat reports it and the hub sees a moving number
                # instead of inferring health.
                self._counter.set_done(now - base)
                self._act.heartbeat()

    def __enter__(self) -> "watchdog":
        self._counter = progress_mod.counter(
            f"evidence:{self._act.kind}", progress_mod.UNIT_EVIDENCE)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._counter.finish()
