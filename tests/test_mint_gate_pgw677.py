"""pgw#677: the background mint yields the GPU to tenant work.

Doctrine under test — tenant requests ALWAYS win the GPU immediately;
mint work yields:

  1. STARVATION SHAPE (the live incident): during a background mint,
     tenant requests complete at serving latency. RED-verified via the
     GEN_WORKER_BG_YIELD=0 kill switch, which restores the pre-fix tree:
     mint seed units inline-compile while holding the per-instance run
     gate (the router's headroom degrade), so a tenant request queues for
     the length of the unit — the measured "completions land exactly as
     mint units free the gate / 0 renders in 19 min" shape.
  2. RACE EXCLUSION (the pgw#676 SIGSEGV class): the shape-warm thread's
     compile can never execute the shared modules concurrently with a
     tenant forward. Post-fix the compile owns a background turn
     (single-flight, instance turn_mutex, tenant-quiet admission); the
     tenant that arrives mid-compile waits — bounded by ONE compile — and
     its wait is attributed to `instance_gate_wait`, never to runtime_ms.
     RED-verified: with the kill switch the overlap is observed.
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
from typing import Annotated, Any, Dict, List, Tuple

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
from gen_worker import fleet_cells, guard_closure, hot_swap
from gen_worker.api.binding import Hub, wire_ref
from gen_worker.executor import Executor, ModelStore
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs

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

        monkeypatch.setattr(executor_mod, "ensure_local", _fake_ensure_local)
        monkeypatch.setattr(
            fleet_cells, "enable_compiled", self._fake_enable_compiled)
        monkeypatch.setattr(
            guard_closure, "assert_closure",
            lambda pipe, cfg, label="": {
                "v": 1, "graphs": [{"target": "transformer", "code": "sim",
                                    "entry": 0, "guards": []}],
                "verdicts": {}, "leaks": []})
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
            family=FAMILY, cell_key="ck4-" + "a" * 56,
            ref=f"{cc.system_repo(FAMILY)}#ck4-{'a' * 56}",
            cfg=cfg, target=mint_root / "cell.tar.gz",
            capture_dir=capture, mint_root=mint_root,
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


# ---------------------------------------------------------------------------
# 1 + 4 — the starvation shape, red-verified via the kill switch
# ---------------------------------------------------------------------------


def test_tenant_serves_at_serving_latency_during_mint_and_red_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-fix: tenant requests dispatched DURING the background mint
    complete at serving latency (preempting the in-flight seed), no seed
    ever inline-compiles, and every real compile lands on the shape-warm
    thread. RED: the kill switch restores the pre-fix tree and the same
    tenant request queues behind a minutes-scale (here: seconds-scale)
    inline-compiling mint unit — the live 19s-for-2s collapse shape."""
    # Degraded router everywhere: pre-fix this forces COMPILED (inline)
    # verdicts; post-fix seeds force EAGER and the warm thread ensures
    # headroom inside its exclusive turn instead.
    monkeypatch.setattr(hot_swap, "_headroom_ok", lambda device: False)

    # --- RED: pre-fix tree via the kill switch ---------------------------
    monkeypatch.setenv("GEN_WORKER_BG_YIELD", "0")
    h_off = _Harness(
        tmp_path / "off", monkeypatch,
        compile_delay_s=0.15, seed_forward_s=0.6, tenant_forward_s=0.02)

    async def _off() -> Tuple[float, pb.JobResult]:
        await h_off.boot()
        assert h_off.rec.background_mint is not None
        # Wait until a BACKGROUND mint seed unit is in flight, holding the
        # run gate (the boot's foreground eager pass contributed run #1).
        deadline = time.monotonic() + 10.0
        while len(h_off.seed_runs) < 2 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert len(h_off.seed_runs) >= 2, "mint never started seeding"
        res, wall = await h_off.dispatch("r-red", aspect="16:9")
        assert res.status == pb.JOB_STATUS_OK, res.safe_message
        return wall, res

    wall_off, _res_off = asyncio.run(_off())
    # The collapse shape: the request queued behind the seed unit's
    # inline compile (and then paid its own inline compile on the
    # degraded router) — an order of magnitude over serving latency.
    assert wall_off >= 0.5, (
        f"expected the pre-fix starvation shape, got {wall_off:.3f}s")
    # Pre-fix, inline compiles ran on handler (to_thread) workers.
    assert any(
        "shape-warm" not in thread for _, thread, _, _ in h_off.compiles), (
        "pre-fix tree should compile inline off the warm thread")

    # --- POST-FIX ---------------------------------------------------------
    monkeypatch.setenv("GEN_WORKER_BG_YIELD", "1")
    h_on = _Harness(
        tmp_path / "on", monkeypatch,
        compile_delay_s=0.15, seed_forward_s=0.6, tenant_forward_s=0.02)

    async def _on() -> None:
        await h_on.boot()
        bg = h_on.rec.background_mint
        assert bg is not None
        # Wait until a PREEMPTIBLE background seed unit holds the gate.
        deadline = time.monotonic() + 10.0
        while bg.seed_ctx is None and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        assert bg.seed_ctx is not None, "no preemptible seed in flight"
        # A stream of tenant requests lands mid-mint; each must complete
        # at serving latency — a fraction of one pre-fix unit — by
        # preempting idle-granted seed units cooperatively (the only
        # bounded residual is one short warm-thread compile).
        walls: List[float] = []
        for i in range(4):
            res, wall = await h_on.dispatch(f"r-{i}", aspect="16:9")
            assert res.status == pb.JOB_STATUS_OK, res.safe_message
            walls.append(wall)
        assert max(walls) < 0.45, (
            f"tenant latencies not at serving levels during mint: {walls}")
        # runtime_ms never absorbs gate contention: the handler runs
        # ~tenant_forward_s, so reported runtime stays near it.
        results = [m.job_result for m in h_on.sent
                   if m.WhichOneof("msg") == "job_result"]
        assert results
        assert all(r.metrics.runtime_ms < 300 for r in results)
        # The mint still finishes once the stream drains.
        await h_on.wait_mint()
        assert h_on.ex.serving_tiers() == {"generate": "compiled"}
        # pgw#677 seed contract: every real compile ran on the shape-warm
        # thread inside a granted turn — no seed unit compiled inline.
        assert h_on.compiles
        assert all(
            "shape-warm" in thread for _, thread, _, _ in h_on.compiles), (
            h_on.compiles)

    asyncio.run(_on())


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
    switch the overlap is observed."""

    # --- RED: pre-fix tree — overlap happens -----------------------------
    monkeypatch.setenv("GEN_WORKER_BG_YIELD", "0")
    h_off = _Harness(
        tmp_path / "off", monkeypatch,
        compile_delay_s=0.8, seed_forward_s=0.0, tenant_forward_s=0.05)

    async def _off() -> None:
        await h_off.boot()
        deadline = time.monotonic() + 10.0
        while not h_off.in_compile.is_set():
            assert time.monotonic() < deadline, "no background compile ran"
            await asyncio.sleep(0.005)
        res, _wall = await h_off.dispatch("r-race", aspect="1:1")
        assert res.status == pb.JOB_STATUS_OK, res.safe_message

    asyncio.run(_off())
    assert h_off.overlaps, (
        "pre-fix tree must exhibit the pgw#676 overlap (tenant forward "
        "concurrent with the warm thread's compile)")

    # --- POST-FIX: exclusion holds, wait is attributed -------------------
    monkeypatch.setenv("GEN_WORKER_BG_YIELD", "1")
    h_on = _Harness(
        tmp_path / "on", monkeypatch,
        compile_delay_s=0.8, seed_forward_s=0.0, tenant_forward_s=0.05)

    async def _on() -> None:
        await h_on.boot()
        deadline = time.monotonic() + 10.0
        while not h_on.in_compile.is_set():
            assert time.monotonic() < deadline, "no background compile ran"
            await asyncio.sleep(0.005)
        res, wall = await h_on.dispatch("r-safe", aspect="1:1")
        assert res.status == pb.JOB_STATUS_OK, res.safe_message
        # It waited for the in-flight compile (bounded by ONE unit)...
        assert wall >= 0.1
        # ...the wait is attributed, and runtime_ms excludes it.
        assert _stage_ms(res, "instance_gate_wait") >= 100
        assert res.metrics.runtime_ms < 400
        # th#1111 invariant survives the new pre-handler stage: the map
        # still closes against runtime_ms with a large gate wait present.
        from gen_worker.stage_timing import reconciliation

        attributed, total = reconciliation(dict(res.metrics.stage_ms))
        assert total == res.metrics.runtime_ms
        assert abs(attributed - total) <= 5, dict(res.metrics.stage_ms)
        await h_on.wait_mint()

    asyncio.run(_on())
    assert not h_on.overlaps, h_on.overlaps


# ---------------------------------------------------------------------------
# 3 — minimum progress: the mint finishes under a sustained tenant stream
# ---------------------------------------------------------------------------


def test_mint_completes_under_sustained_tenant_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tenant stream that never drains would starve a purely idle-gated
    mint forever. The steal rule grants one bounded background unit per
    debt window: the mint completes while the stream keeps serving."""
    monkeypatch.setenv("GEN_WORKER_BG_YIELD", "1")
    monkeypatch.setattr(executor_mod, "_BG_STEAL_FLOOR_S", 0.1)
    monkeypatch.setattr(executor_mod, "_BG_STEAL_DEBT_FACTOR", 1.0)
    monkeypatch.setattr(executor_mod, "_BG_COMPILE_QUIESCENCE_S", 0.0)
    h = _Harness(
        tmp_path, monkeypatch,
        compile_delay_s=0.05, seed_forward_s=0.05, tenant_forward_s=0.02)

    async def _run() -> None:
        await h.boot()
        bg = h.rec.background_mint
        assert bg is not None and bg.task is not None
        stop = asyncio.Event()
        served = 0

        async def _stream() -> None:
            nonlocal served
            i = 0
            while not stop.is_set():
                res, _wall = await h.dispatch(f"r-load-{i}", aspect="16:9")
                assert res.status == pb.JOB_STATUS_OK, res.safe_message
                served += 1
                i += 1

        stream = asyncio.create_task(_stream())
        try:
            await asyncio.wait_for(asyncio.shield(bg.task), timeout=60.0)
        finally:
            stop.set()
            await stream
        assert h.ex.serving_tiers() == {"generate": "compiled"}
        # The stream really was sustained while the mint progressed.
        assert served >= 5

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 4 — the seed-window routing contract, pinned at the Router
# ---------------------------------------------------------------------------


def test_mint_seed_window_forces_eager_enqueue_on_degraded_routers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inside the mint seed window a novel signature NEVER compiles inline:
    EAGER + background enqueue even when the router is non-concurrent or
    (turn-gated) short on headroom. Outside the window the legacy verdicts
    stand."""
    router = hot_swap.Router()

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
    gated.enable()
    gated.set_turn_gate(lambda kind: contextlib.nullcontext())
    verdict, sig = gated.route("t", compiled, ("c",), {})
    assert verdict == hot_swap.EAGER
    # Ungated legacy router with tight headroom keeps the pre-fix degrade
    # (kill-switch parity).
    legacy = hot_swap.Router()
    legacy.enable()
    verdict, _sig = legacy.route("t", compiled, ("d",), {})
    assert verdict == hot_swap.COMPILED
