"""The background mint yields the GPU to tenant work.

Doctrine under test — tenant requests ALWAYS win the GPU immediately;
mint work yields:

  1. NO STARVATION: during a background mint, tenant requests complete at
     serving latency. Asserted in absolute terms against the harness's own
     configured quantities; mint seed units inline-compiling while holding
     the per-instance run gate is the shape that must stay unreachable.
  2. RACE EXCLUSION (the SIGSEGV class): the shape-warm thread's
     compile can never execute the shared modules concurrently with a
     tenant forward. The compile owns a background turn (single-flight,
     instance turn_mutex, tenant-quiet admission); the tenant that arrives
     mid-compile waits — bounded by ONE compile — and its wait is
     attributed to `instance_gate_wait`, never to runtime_ms.
  3. MINIMUM PROGRESS: under a sustained tenant stream the mint still
     finishes — the steal rule grants one bounded background unit per
     debt window; stolen units are not preemptible.
  4. SEED UNITS NEVER COMPILE INLINE: inside the mint seed window a novel
     signature routes EAGER + background enqueue even on a degraded
     router — every real compile lands on the shape-warm thread.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple

import msgspec
import pytest

import gen_worker
import gen_worker.executor as executor_mod
from gen_worker import (
    AxisClass,
    Compile,
    CompileAxis,
    RequestContext,
    Resources,
    endpoint,
    worker_function,
)
from gen_worker import compile_cache as cc
from gen_worker import fleet_cells, hot_swap
from gen_worker.api.binding import Hub, wire_ref
from gen_worker.executor import Executor
from gen_worker.models.store import ModelStore
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs
from gen_worker.models import store as store_mod

from harness.progress_wait import Cadence, await_progress_async

FAMILY = "sdxl"
_ASPECT_AXIS = CompileAxis(classes=(
    AxisClass("sq", match=lambda v: v == "1:1", warm="1:1"),
    AxisClass("wide", match=lambda v: v == "16:9", warm="16:9"),
))


class _In(msgspec.Struct):
    prompt: str = ""
    aspect_ratio: Annotated[str, _ASPECT_AXIS] = "1:1"


class _Out(msgspec.Struct):
    ok: bool = True


class _Denoiser:
    def forward(self, aspect: str) -> str:
        return aspect


class _Pipe:
    def __init__(self) -> None:
        self.transformer = _Denoiser()


@pytest.fixture(autouse=True)
def _clean_process_registries() -> Any:
    with cc._PROVEN_CELLS_LOCK:
        cc._PROVEN_CELLS.clear()
    with fleet_cells._PENDING_LOCK:
        fleet_cells._PENDING.clear()
    for pipe in list(cc._armed_pipelines()):
        cc._armed_pipelines().discard(pipe)
    yield
    with cc._PROVEN_CELLS_LOCK:
        cc._PROVEN_CELLS.clear()
    with fleet_cells._PENDING_LOCK:
        fleet_cells._PENDING.clear()
    for pipe in list(cc._armed_pipelines()):
        cc._armed_pipelines().discard(pipe)


@pytest.fixture(autouse=True)
def _fast_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compress the gate's wall-clock constants for test speed; individual
    tapes override where the specific rule is under test."""
    monkeypatch.setattr(executor_mod, "_BG_COMPILE_QUIESCENCE_S", 0.05)
    monkeypatch.setattr(executor_mod, "_MINT_POLL_INTERVAL_S", 0.05)


class _Harness:
    """Eager-first boot over the real executor codepath, plus a REAL
    handle_run_job tenant lane and an instrumented compile leaf that
    records execution intervals + the executing thread."""

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        compile_delay_s: float = 0.0,
        seed_forward_s: float = 0.0,
        tenant_forward_s: float = 0.02,
    ) -> None:
        self.tmp_path = tmp_path
        self.compile_delay_s = compile_delay_s
        self.seed_forward_s = seed_forward_s
        self.tenant_forward_s = tenant_forward_s
        # (label, thread_name, start, end) per novel-signature compile.
        self.compiles: List[Tuple[str, str, float, float]] = []
        self.in_compile = threading.Event()
        # True whenever a tenant forward observed a live compile (the
        # pgw#676 race shape). Must stay empty post-fix.
        self.overlaps: List[str] = []
        self.tenant_windows: List[Tuple[float, float]] = []
        self.seed_runs: List[str] = []
        self.sent: List[pb.WorkerMessage] = []
        self.pipes: List[_Pipe] = []
        harness = self

        @endpoint(
            model=Hub("acme/sdxl-base"),
            resources=Resources(gpu=True),
            compile=Compile(family=FAMILY, shapes=((768, 768),), text_len=0),
        )
        class MintEndpoint:
            def setup(self, model: str) -> None:
                self.pipe = _Pipe()
                harness.pipes.append(self.pipe)
                gen_worker.arm_compile(self.pipe)

            @worker_function()
            def generate(self, ctx: RequestContext, payload: _In) -> _Out:
                if ctx.boot_warmup:
                    harness.seed_runs.append(payload.aspect_ratio)
                    deadline = time.monotonic() + harness.seed_forward_s
                    while time.monotonic() < deadline:
                        time.sleep(0.005)
                        ctx.raise_if_cancelled("seed preempted")
                    self.pipe.transformer.forward(payload.aspect_ratio)
                    return _Out()
                t0 = time.monotonic()
                if harness.in_compile.is_set():
                    harness.overlaps.append("tenant-entered-during-compile")
                self.pipe.transformer.forward(payload.aspect_ratio)
                deadline = time.monotonic() + harness.tenant_forward_s
                while time.monotonic() < deadline:
                    if harness.in_compile.is_set():
                        harness.overlaps.append("compile-during-tenant")
                    time.sleep(0.002)
                harness.tenant_windows.append((t0, time.monotonic()))
                return _Out()

        self.cls = MintEndpoint
        self.specs = extract_specs(MintEndpoint)
        (self.spec,) = [s for s in self.specs if s.name == "generate"]

        async def _send(msg: pb.WorkerMessage) -> None:
            self.sent.append(msg)

        store = ModelStore(
            _send, cache_dir=tmp_path / "cas", vram_budget_bytes=4 << 30)

        async def _fake_ensure_local(ref: str, **kwargs: Any) -> Path:
            p = tmp_path / "snap"
            p.mkdir(parents=True, exist_ok=True)
            return p

        monkeypatch.setattr(store_mod, "ensure_local", _fake_ensure_local)
        monkeypatch.setattr(
            fleet_cells, "enable_compiled", self._fake_enable_compiled)
        # The pgw#681 mint gate this simmed is deleted.
        # `guard_closure.closure_manifest` classified every compiled graph at
        # the MINT and wrote the result into the cell's metadata; it went with
        # the `torch-inductor-cache` format that carried it, so a rig whose
        # compiles never touch dynamo has no gate left to satisfy.
        self.ex = Executor(self.specs, _send, store=store)

    # -- the compile-arm leaf ------------------------------------------------

    def _fake_enable_compiled(
        self, pipe: Any, cfg: Any, cache_dir: Any = None,
        artifact: Any = None, publisher: Any = None,
    ) -> fleet_cells.ArmOutcome:
        mint_root = self.tmp_path / f"mint-{id(pipe)}"
        capture = mint_root / "capture"
        (capture / "inductor" / "fxgraph").mkdir(parents=True, exist_ok=True)
        pending = fleet_cells.PendingSelfMint(
            family=FAMILY, arm_token="cg-key-v1-" + "a" * 56,
            ref=f"{cc.system_repo(FAMILY)}#cg-key-v1-{'a' * 56}",
            cfg=cfg, target=mint_root / "cell.tar.gz", mint_root=mint_root,
            publisher=None, cache_dir=cache_dir,
        )
        original = pipe.transformer.forward
        harness = self

        seen_sigs: set = set()
        seen_lock = threading.Lock()

        def compiled(*args: Any, **kwargs: Any) -> Any:
            sig = hot_swap.signature(args, kwargs)
            with seen_lock:
                novel = sig not in seen_sigs
                seen_sigs.add(sig)
            if novel:
                start = time.monotonic()
                harness.in_compile.set()
                try:
                    time.sleep(harness.compile_delay_s)
                    (capture / "inductor" / "fxgraph"
                     / f"g{len(harness.compiles)}.bin").write_bytes(b"graph")
                finally:
                    harness.in_compile.clear()
                harness.compiles.append((
                    str(sig[0]), threading.current_thread().name,
                    start, time.monotonic()))
            return original(*args, **kwargs)

        signal: Dict[str, Any] = {
            "callback": None,
            "lock": threading.Lock(),
            "successful_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "router": hot_swap.Router(),
        }
        setattr(pipe, cc._MARKER_ATTR, {
            "targets": ["transformer"],
            "shapes": tuple(cfg.shapes),
            "cache": True,
            "originals": [(pipe.transformer, "forward", original)],
            "regional_mods": [],
            "failure_signal": signal,
        })
        pipe.transformer.forward = cc._guarded(
            original, compiled, "transformer", failure_signal=signal)
        cc._armed_pipelines().add(pipe)
        return fleet_cells.ArmOutcome(armed=True, self_mint=pending)

    # -- drive ---------------------------------------------------------------

    async def boot(self) -> Any:
        return await self.ex.ensure_setup(self.spec, {
            wire_ref(self.spec.models["model"]): pb.Snapshot(
                digest="blake3:" + "a" * 64),
        })

    @property
    def rec(self) -> Any:
        return self.ex._classes[self.spec.instance_key]

    async def wait_mint(self, timeout: float = 60.0) -> None:
        bg = self.rec.background_mint
        if bg is None or bg.task is None:
            return
        await asyncio.wait_for(asyncio.shield(bg.task), timeout)

    def _run_job(self, request_id: str, aspect: str = "1:1") -> pb.RunJob:
        model_ref = wire_ref(self.spec.models["model"])
        return pb.RunJob(
            request_id=request_id, attempt=1,
            function_name=self.spec.name,
            input_payload=msgspec.msgpack.encode(
                _In(prompt="a cat", aspect_ratio=aspect)),
            models=[pb.ModelBinding(slot="model", ref=model_ref)],
            snapshots={model_ref: pb.Snapshot(digest="blake3:" + "a" * 64)},
        )

    async def dispatch(
        self, request_id: str, aspect: str = "1:1",
    ) -> Tuple[pb.JobResult, float]:
        """One REAL tenant request through handle_run_job; returns the
        terminal result and the wall time from admission to completion."""
        run = self._run_job(request_id, aspect)
        t0 = time.monotonic()
        await self.ex.handle_run_job(run)
        job = self.ex.jobs[(run.request_id, run.attempt)]
        assert job.task is not None
        await job.task
        wall = time.monotonic() - t0
        results = [m.job_result for m in self.sent
                   if m.WhichOneof("msg") == "job_result"
                   and m.job_result.request_id == request_id]
        assert results, f"no job_result for {request_id}"
        return results[-1], wall


def _stage_ms(res: pb.JobResult, name: str) -> int:
    return int(res.metrics.stage_ms.get(name, 0))


#: pgw#1249. Slack for comparing two clocks that measure the SAME run — the
#: worker's stage map against the tape's own wall — not patience for a slow
#: machine. It bounds rounding and the few instructions between `t0` and the
#: first stage opening, so it does NOT grow with load: both sides of every
#: comparison below stretch together. That is the whole difference between
#: this and the `>= 100` floor it replaced, which measured the weather.
_CLOCK_SLOP_MS = 25.0


# ---------------------------------------------------------------------------
# 1 + 4 — the starvation shape, red-verified via the kill switch
# ---------------------------------------------------------------------------


# `test_tenant_serves_at_serving_latency_during_mint_and_red_verifies`
# stood here. Its mechanism is the mint SEED WINDOW — preemptible in-process
# seed units holding the instance gate while a tenant request arrives — and the
# in-process mint is deleted (it only ever built a dynamo cell). A delegated
# mint compiles in a CHILD PROCESS, so there is no seed unit in the serving
# interpreter for a tenant to preempt; what bounds the child against the tenant
# is the pgw#737 co-residency budget, which has its own coverage. The doctrine
# rows that do NOT depend on the seed window — no compile/tenant overlap, and
# no inline compile on a degraded router — are below and unchanged.


# ---------------------------------------------------------------------------
# 2 — the pgw#676 race shape cannot happen; the bounded wait is attributed
# ---------------------------------------------------------------------------


def test_compile_and_tenant_forward_never_overlap_and_red_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-fix: a tenant arriving while the shape-warm thread compiles
    WAITS (bounded by that one compile) — the concurrent execution that
    segfaulted sm_86 (pgw#676: _forward_with_branch racing compile_wrapper)
    is structurally impossible — and the wait is attributed to the
    instance_gate_wait stage, not billed as runtime. RED: with the kill
    switch the overlap is observed (pgw#995: that arm is gone — see below)."""

    # The RED arm drove `GEN_WORKER_BG_YIELD=0` and asserted that the
    # pre-fix tree DOES exhibit the pgw#676 overlap. The switch and that tree
    # are deleted, so the overlap is now structurally impossible rather than
    # merely not-selected — which is the stronger statement the docstring
    # already made.

    h_on = _Harness(
        tmp_path / "on", monkeypatch,
        compile_delay_s=0.8, seed_forward_s=0.0, tenant_forward_s=0.05)

    async def _on() -> None:
        await h_on.boot()
        # pgw#1349: this was `deadline = time.monotonic() + 10.0`, and it red
        # master as `assert 416.463803 < 416.463401` — a clock reporting that
        # a loaded runner had not yet scheduled the shape-warm thread, dressed
        # up as "no background compile ran". Nothing here is a claim about how
        # fast the machine is; what the tape needs is that a compile STARTS.
        # So the give-up keys on evidence: it is immediate and definitive when
        # the background mint has finished without ever compiling (nothing can
        # start after that), and otherwise it only bounds SILENCE.
        def _mint_finished() -> Optional[str]:
            # `_background_mint`'s finally clears `rec.background_mint`, so
            # ABSENCE is the completion signal here (boot has already set it —
            # the sibling tape asserts exactly that). Once the mint is over and
            # nothing compiled, no compile can start, so this is definitive and
            # needs no clock at all.
            bg = h_on.rec.background_mint
            if h_on.compiles or h_on.in_compile.is_set():
                return None
            if bg is None:
                return "the background mint finished without compiling anything"
            task = bg.task
            if task is not None and task.done():
                return ("the background mint task ended without compiling "
                        f"anything: {task.exception()!r}")
            return None

        await await_progress_async(
            lambda: (h_on.in_compile.is_set(), len(h_on.compiles)),
            lambda seen: seen[0],
            what="the shape-warm thread to enter a compile",
            cadence=Cadence(),
            gone=_mint_finished,
        )
        t_admit = time.monotonic()
        # The moment the in-flight compile RELEASES, watched from beside the
        # dispatch. `in_compile` is the same signal the loop above already
        # trusts to know a compile is holding the instance, so the ordering
        # assertions rest on nothing new.
        released: Dict[str, float] = {}

        async def _watch_release() -> None:
            while h_on.in_compile.is_set():
                await asyncio.sleep(0.001)
            released["at"] = time.monotonic()

        watcher = asyncio.ensure_future(_watch_release())
        res, wall = await h_on.dispatch("r-safe", aspect="1:1")
        t_done = time.monotonic()
        await watcher
        assert res.status == pb.JOB_STATUS_OK, res.safe_message
        # The wait is ATTRIBUTED: present, and not zero.
        #
        # pgw#1249: this was `wall >= 0.1` and `instance_gate_wait >= 100`.
        # Both are floors on how long a MACHINE took, and how long a machine
        # takes is not the property under test — it is the weather. Neither
        # had a defensible value either: the tenant enters somewhere INSIDE
        # an 800 ms sleep, so the true wait is anywhere in (0, 800]. On a
        # loaded runner it came in at 90 ms and turned a correct tree red on
        # the release path. What the test MEANS is causal — it waited for the
        # instance — and that is asserted as ordering below.
        gate_wait_ms = _stage_ms(res, "instance_gate_wait")
        assert gate_wait_ms > 0, dict(res.metrics.stage_ms)
        # ...and runtime EXCLUDES it. Measured against this run's OWN
        # end-to-end wall rather than against `compile_delay_s`: a slow runner
        # stretches both sides together, so the relation holds exactly where a
        # constant cannot, and a runtime that had absorbed the gate wait would
        # overshoot the wall it was measured inside.
        assert res.metrics.runtime_ms + gate_wait_ms <= wall * 1000 + _CLOCK_SLOP_MS, (
            res.metrics.runtime_ms, gate_wait_ms, wall)
        # th#1111 invariant survives the new pre-handler stage: the map
        # still closes against runtime_ms with a large gate wait present.
        from gen_worker.stage_timing import reconciliation

        attributed, total = reconciliation(dict(res.metrics.stage_ms))
        assert total == res.metrics.runtime_ms
        assert abs(attributed - total) <= 5, dict(res.metrics.stage_ms)
        await h_on.wait_mint()

        # ORDERING — the causal claim, and the reason none of this needs a
        # threshold. Timestamps THIS run produced, compared only to each
        # other: the tenant was admitted while the compile still held the
        # instance, and it did not complete until after that compile let go.
        # A slow machine moves all of them together, so there is nothing here
        # for load to falsify.
        assert "at" in released, "the in-flight compile never released"
        t_release = released["at"]
        assert t_admit < t_release, (
            "inconclusive tape: the tenant was admitted after the compile had "
            "already released the instance, so nothing here was a wait")
        assert t_release <= t_done + _CLOCK_SLOP_MS / 1000.0, (
            "the tenant completed before the compile released the instance — "
            "it was not gated at all", t_release - t_done)

    asyncio.run(_on())
    assert not h_on.overlaps, h_on.overlaps


# ---------------------------------------------------------------------------
# 3 — minimum progress: the mint finishes under a sustained tenant stream
# ---------------------------------------------------------------------------


# `test_mint_completes_under_sustained_tenant_load` stood here — the
# minimum-progress half of the same seed-window doctrine (a sustained tenant
# stream must not starve the mint of background turns). With the compile in a
# child process the mint's progress no longer competes for in-process turns at
# all, so the property it asserted is not merely untested but gone. The child's
# own progress is covered by mint_process's liveness rule
# (`test_mint_liveness_pgw784.py`).


# ---------------------------------------------------------------------------
# 4 — the seed-window routing contract, pinned at the Router
# ---------------------------------------------------------------------------


def test_mint_seed_window_forces_eager_enqueue_on_degraded_routers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inside the mint seed window a novel signature NEVER compiles inline:
    EAGER + background enqueue even when the router is non-concurrent or
    short on headroom. Outside the window the sequential verdict stands.

    pgw#1215 step 4: the row that used to close this test — an UNGATED router
    keeping "the pre-fix degrade (kill-switch parity)" — is gone with the mode
    it described. There is no kill switch to be at parity with
    (``test_no_env_restores_the_pre_pgw677_tree``, ten lines below, asserts
    ``GEN_WORKER_BG_YIELD`` cannot exist), and an ungated router now refuses
    typed rather than degrading. What replaces it asserts the contract that
    is actually live: a TURN-GATED router does not degrade on tight headroom,
    and an ungated one cannot route concurrently at all.
    """
    router = hot_swap.Router()
    router.set_turn_gate(lambda kind: contextlib.nullcontext())

    def compiled(*args: Any, **kwargs: Any) -> None:
        return None

    # Non-concurrent router (never enabled): ordinary calls keep the
    # sequential inline compile...
    verdict, _sig = router.route("t", compiled, ("a",), {})
    assert verdict == hot_swap.COMPILED
    # ...a mint seed forces eager + enqueue.
    with hot_swap.mint_seed_window():
        verdict, sig = router.route("t", compiled, ("b",), {})
    assert verdict == hot_swap.EAGER
    with router.lock:
        assert sig in router.pending

    # Turn-gated router with tight headroom: the ordinary call no longer
    # degrades to an inline compile either — the warm thread owns headroom
    # inside its exclusive turn.
    monkeypatch.setattr(hot_swap, "_headroom_ok", lambda device: False)
    gated = hot_swap.Router()
    gated.set_turn_gate(lambda kind: contextlib.nullcontext())
    gated.enable()
    verdict, sig = gated.route("t", compiled, ("c",), {})
    assert verdict == hot_swap.EAGER
    # An UNGATED router cannot reach that decision at all: concurrency is
    # refused typed, so the degrade branch has no state left to fire from.
    # RED before pgw#1215 step 4: `legacy.enable()` succeeded and the route
    # returned COMPILED.
    ungated = hot_swap.Router()
    with pytest.raises(hot_swap.RouterNotGated):
        ungated.enable()


# ---------------------------------------------------------------------------
# The deleted arm is UNREACHABLE, not merely unused
# ---------------------------------------------------------------------------


def test_no_env_restores_the_pre_pgw677_tree() -> None:
    """The two RED arms above were driven by an env. Deleting a switch and
    deleting its tests looks identical to deleting a switch and leaving a
    second reader behind — which is how `GEN_WORKER_PREFER_AOT` kept a live
    gate after one was believed removed. So assert on the SOURCE.
    """
    from pathlib import Path as _P
    import gen_worker.executor as _ex

    src = _P(_ex.__file__).read_text()
    assert 'environ.get("GEN_WORKER_BG_YIELD"' not in src, (
        "GEN_WORKER_BG_YIELD is read again — env carries config and secrets, "
        "never a branch selection")
    assert "_bg_yield_enabled" not in src, (
        "the bg-yield predicate is back; pgw#677's shape is unconditional")
