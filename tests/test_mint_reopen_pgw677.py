"""pgw#677 REOPEN: the live break the 0.70.0 tapes could not see.

The ie#546 final cycle measured, on gen-worker 0.70.0, a cold L4 pod
holding one tenant `generate` for 26m25s while an 18-unit mint ran
4-7-minute units back to back, finalizing at unit 8/18 and publishing
nothing — with the turn gate armed. Root causes, each pinned here:

  1. ELIGIBILITY MISCLASSIFICATION (the starvation): sdxl's mixed
     ``#fp8-w8a8``-storage checkpoint stamps ``_cozy_weight_lane =
     "w8a8-lora64"`` while the hub serves it ``fp8-w8a16+eager``. The
     eager-first eligibility (and the router's ``fail_closed``) read the
     weight-lane prefix, classified the boot mandatory-quantized, and
     silently fell back to the FOREGROUND compile-then-serve mint: the
     tenant sat inside ensure_setup for the whole inline-compile plan,
     and the entire pgw#677 gate/preemption machinery never ran.
     Fix: ONE serveability brain (``compile_cache.mandatory_serving``) —
     the hub-resolved execution lane outranks the weight-lane stamp.
  2. STOLEN-COMPILE SIZING: a stolen turn is not preemptible and a real
     inductor compile is 4-7 unabortable minutes — ~100x the advertised
     30-90s residual. Compile turns now steal only against MINUTES of
     continuous demand (``_BG_COMPILE_STEAL_FLOOR_S``) and announce the
     steal on the wire.
  3. THE PUBLISH BREAK: every mint-abort / publish-withhold /
     closure-refusal exit died in unreachable pod logs. All of them now
     ride the wire as typed events (``self_mint_abort``,
     ``self_mint_publish_withheld``, ``self_mint_publish_failed``), and
     an OOM-truncated warm plan can no longer converge silently into a
     partial finalize.

Every tape here is RED on the 0.70.0 tree (this file is self-contained
so it can be dropped onto that tree to prove it).
"""

from __future__ import annotations

import asyncio
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
    def __init__(self, weight_lane: str = "") -> None:
        self.transformer = _Denoiser()
        if weight_lane:
            self._cozy_weight_lane = weight_lane


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
    monkeypatch.setattr(executor_mod, "_BG_COMPILE_QUIESCENCE_S", 0.02)
    monkeypatch.setattr(executor_mod, "_MINT_POLL_INTERVAL_S", 0.05)


class _Harness:
    """Real-executor rig: ensure_setup boot, REAL handle_run_job tenant
    lane, instrumented compile leaf. ``weight_lane`` stamps the sim pipe
    (the live sdxl shape is ``"w8a8-lora64"``); ``hub_lane`` seeds the
    executor's model-resolution table with a th#913 execution lane."""

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        compile_delay_s: float = 0.0,
        seed_forward_s: float = 0.0,
        tenant_forward_s: float = 0.02,
        weight_lane: str = "",
        hub_lane: str = "",
        seed_oom_after: int = 0,
    ) -> None:
        self.tmp_path = tmp_path
        self.compile_delay_s = compile_delay_s
        self.seed_forward_s = seed_forward_s
        self.tenant_forward_s = tenant_forward_s
        self.seed_oom_after = seed_oom_after
        self.compiles: List[Tuple[str, str, float, float]] = []
        self.in_compile = threading.Event()
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
                self.pipe = _Pipe(weight_lane=weight_lane)
                harness.pipes.append(self.pipe)
                gen_worker.arm_compile(self.pipe)

            @worker_function()
            def generate(self, ctx: RequestContext, payload: _In) -> _Out:
                if ctx.boot_warmup:
                    harness.seed_runs.append(payload.aspect_ratio)
                    if (harness.seed_oom_after
                            and len(harness.seed_runs)
                            > harness.seed_oom_after):
                        import torch

                        raise torch.cuda.OutOfMemoryError(
                            "CUDA out of memory (sim)")
                    deadline = time.monotonic() + harness.seed_forward_s
                    while time.monotonic() < deadline:
                        time.sleep(0.005)
                        ctx.raise_if_cancelled("seed preempted")
                    self.pipe.transformer.forward(payload.aspect_ratio)
                    return _Out()
                self.pipe.transformer.forward(payload.aspect_ratio)
                deadline = time.monotonic() + harness.tenant_forward_s
                while time.monotonic() < deadline:
                    time.sleep(0.002)
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
        if hub_lane:
            ref = wire_ref(self.spec.models["model"])
            self.ex._model_resolutions = {ref: (ref, "", hub_lane)}

    def _fake_enable_compiled(
        self, pipe: Any, cfg: Any, cache_dir: Any = None,
        artifact: Any = None, publisher: Any = None,
    ) -> fleet_cells.ArmOutcome:
        mint_root = self.tmp_path / f"mint-{id(pipe)}"
        capture = mint_root / "capture"
        (capture / "inductor" / "fxgraph").mkdir(parents=True, exist_ok=True)
        pending = fleet_cells.PendingSelfMint(
            family=FAMILY, cell_key="ck2-" + "a" * 56,
            ref=f"{cc.system_repo(FAMILY)}#ck2-{'a' * 56}",
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

        # Honor the one serveability brain exactly like the real arm path.
        fail_closed = cc.mandatory_serving(pipe) if hasattr(
            cc, "mandatory_serving") else False
        signal: Dict[str, Any] = {
            "callback": None,
            "lock": threading.Lock(),
            "successful_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "router": hot_swap.Router(fail_closed=fail_closed),
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

    def events(self, kind: str) -> List[pb.ActivityUpdate]:
        return [
            m.activity_update for m in self.sent
            if m.WhichOneof("msg") == "activity_update"
            and m.activity_update.kind == kind
        ]


# ---------------------------------------------------------------------------
# 1 — THE LIVE BREAK: a w8a8-stamped pipe on a hub-declared eager-serveable
#     execution lane must boot eager-first (background mint), not foreground
# ---------------------------------------------------------------------------


def test_w8a8_stamp_with_hub_execution_lane_boots_eager_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact live shape: pipe stamped ``w8a8-lora64`` (cell identity),
    hub resolution lane ``fp8-w8a16+eager`` (serveability). RED on 0.70.0:
    the weight-lane prefix classified this boot mandatory-quantized, the
    setup ran the FOREGROUND compile-then-serve mint (rec.background_mint
    is None), and the first tenant request sat behind the whole inline
    plan — the measured 26-minute cold-L4 starvation."""
    monkeypatch.setenv("GEN_WORKER_BG_YIELD", "1")
    h = _Harness(
        tmp_path, monkeypatch,
        compile_delay_s=0.4, seed_forward_s=0.05, tenant_forward_s=0.02,
        weight_lane="w8a8", hub_lane="fp8-w8a16+eager")

    async def _run() -> None:
        boot_t0 = time.monotonic()
        await h.boot()
        boot_wall = time.monotonic() - boot_t0
        # Eager-first: READY without paying the plan's compiles foreground.
        assert h.rec.background_mint is not None, (
            "w8a8-stamped pipe with hub lane fp8-w8a16+eager was refused "
            "eager-first — the foreground compile-then-serve mint is the "
            "reopen's measured 26-minute starvation")
        assert boot_wall < 2.0, f"boot paid the plan foreground: {boot_wall:.1f}s"
        # A tenant request mid-mint completes at serving latency.
        res, wall = await h.dispatch("r-live", aspect="16:9")
        assert res.status == pb.JOB_STATUS_OK, res.safe_message
        assert wall < 1.0, f"tenant starved during mint: {wall:.2f}s"
        await h.wait_mint()
        assert h.ex.serving_tiers() == {"generate": "compiled"}

    asyncio.run(_run())


def test_true_mandatory_lane_still_refuses_eager_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The qwen shape: hub lane says real w8a8 activations — eager is not
    a production tier there; the boot keeps the sequential foreground
    proof. Guards the fix from over-rotating."""
    monkeypatch.setenv("GEN_WORKER_BG_YIELD", "1")
    h = _Harness(
        tmp_path, monkeypatch,
        compile_delay_s=0.0, seed_forward_s=0.0, tenant_forward_s=0.01,
        weight_lane="w8a8", hub_lane="fp8-w8a8-dynamic+compiled")

    async def _run() -> None:
        await h.boot()
        assert h.rec.background_mint is None, (
            "a REAL w8a8-activation lane must keep the foreground proof")

    asyncio.run(_run())


def test_w8a8_stamp_without_lane_evidence_stays_foreground(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No hub lane evidence: the weight-lane stamp remains the fail-closed
    fallback — unchanged pre-reopen behavior."""
    monkeypatch.setenv("GEN_WORKER_BG_YIELD", "1")
    h = _Harness(
        tmp_path, monkeypatch,
        weight_lane="w8a8", hub_lane="")

    async def _run() -> None:
        await h.boot()
        assert h.rec.background_mint is None

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 2 — SIZING: a multi-minute unabortable compile + waiting tenant. The
#     compile lane must not steal against short demand; the tenant completes.
# ---------------------------------------------------------------------------


def test_multi_minute_compile_never_steals_against_live_demand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reopen's missing tape: background compile units are MINUTES of
    unabortable wall (here scaled to 1.2s vs a 20ms tenant forward), and a
    tenant stream is live the whole time. Post-fix, compile turns steal
    only after `_BG_COMPILE_STEAL_FLOOR_S` of continuous demand, so every
    tenant completes at serving latency. RED on 0.70.0: the single 30s
    floor (patched to 0.1s, as tape 3 always did) lets the compile steal
    almost immediately and a tenant waits out the whole unabortable unit."""
    monkeypatch.setenv("GEN_WORKER_BG_YIELD", "1")
    monkeypatch.setattr(executor_mod, "_BG_STEAL_FLOOR_S", 0.1)
    # raising=False: on the pre-fix tree this attribute does not exist and
    # the single floor governs — that IS the red run.
    monkeypatch.setattr(
        executor_mod, "_BG_COMPILE_STEAL_FLOOR_S", 30.0, raising=False)
    h = _Harness(
        tmp_path, monkeypatch,
        compile_delay_s=1.2, seed_forward_s=0.02, tenant_forward_s=0.02)

    async def _run() -> None:
        # A sustained, never-idle stream FROM THE FIRST ADMISSION: the next
        # dispatch is always admitted before the previous completes, so
        # `_bg_quiet` never sets and no compile turn can be idle-granted —
        # any multi-second tenant wall in this window is a STEAL. (The
        # arrive-mid-idle-compile residual is a separate, documented bound
        # and deliberately not provoked here.) The first request performs
        # setup and starts the mint; its wall is excluded.
        walls: List[float] = []
        pending_task: Any = None
        for i in range(12):
            nxt = asyncio.create_task(h.dispatch(f"r-{i}", aspect="16:9"))
            if pending_task is not None:
                res, wall = await pending_task
                assert res.status == pb.JOB_STATUS_OK, res.safe_message
                walls.append(wall)
            pending_task = nxt
            await asyncio.sleep(0)
        res, wall = await pending_task
        walls.append(wall)
        assert h.rec.background_mint is not None
        # Post-fix: no tenant ever waits out the 1.2s unabortable compile.
        steady = walls[1:]
        assert max(steady) < 1.0, (
            f"a tenant waited out a stolen multi-minute compile: {steady}")
        # Drain: with the stream gone, idle grants finish the mint.
        await h.wait_mint()

    asyncio.run(_run())


def test_compile_steal_is_announced_on_the_wire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the compile floor DOES elapse under truly continuous demand,
    the steal happens (minimum progress) and announces itself as a typed
    ``bg_turn_steal`` event."""
    monkeypatch.setenv("GEN_WORKER_BG_YIELD", "1")
    monkeypatch.setattr(executor_mod, "_BG_STEAL_FLOOR_S", 0.05)
    monkeypatch.setattr(
        executor_mod, "_BG_COMPILE_STEAL_FLOOR_S", 0.3, raising=False)
    monkeypatch.setattr(executor_mod, "_BG_STEAL_DEBT_FACTOR", 0.5)
    h = _Harness(
        tmp_path, monkeypatch,
        compile_delay_s=0.1, seed_forward_s=0.02, tenant_forward_s=0.02)

    async def _run() -> None:
        await h.boot()
        assert h.rec.background_mint is not None
        pending_task: Any = None
        deadline = time.monotonic() + 30.0
        i = 0
        while time.monotonic() < deadline:
            nxt = asyncio.create_task(h.dispatch(f"r-{i}", aspect="16:9"))
            if pending_task is not None:
                res, _wall = await pending_task
                assert res.status == pb.JOB_STATUS_OK, res.safe_message
            pending_task = nxt
            i += 1
            if h.events("bg_turn_steal"):
                break
        if pending_task is not None:
            await pending_task
        assert h.events("bg_turn_steal"), (
            "a compile steal under continuous demand must be announced")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 3 — THE PUBLISH BREAK: every refusal reason reaches the hub typed
# ---------------------------------------------------------------------------


def test_pack_or_closure_refusal_reaches_the_wire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ie#546 cycle lost its root cause because a finalize refusal
    (closure gate / pack) died in pod logs. Post-fix the verbatim reason
    rides the wire as a typed ``self_mint_abort`` event. RED on 0.70.0:
    no such event exists anywhere."""
    monkeypatch.setenv("GEN_WORKER_BG_YIELD", "1")

    def _refuse(*args: Any, **kwargs: Any) -> None:
        raise ValueError("closure refused: guard leak L['scale']")

    monkeypatch.setattr(cc, "finish_fleet_mint", _refuse)
    h = _Harness(tmp_path, monkeypatch, compile_delay_s=0.02)

    async def _run() -> None:
        await h.boot()
        bg = h.rec.background_mint
        assert bg is not None and bg.task is not None
        # The driver contains the failure (serving stays eager and alive);
        # the typed abort must still name the refusal verbatim.
        await asyncio.wait_for(bg.task, timeout=30.0)
        assert h.ex.serving_tiers() == {"generate": "eager"}
        aborts = h.events("self_mint_abort")
        assert aborts, "finalize refusal never reached the wire"
        assert any("L['scale']" in a.detail for a in aborts), (
            [a.detail for a in aborts])
        assert any(a.phase == "pack_failed" for a in aborts)

    asyncio.run(_run())


def test_withhold_and_no_sink_reach_the_wire(tmp_path: Path) -> None:
    """Unit tapes for the two publish-withhold doors (gw#612 gap and the
    missing publish sink): both emit typed events naming the reason."""
    from gen_worker import activity as activity_mod

    sent: List[pb.WorkerMessage] = []
    loop = asyncio.new_event_loop()

    async def _pump() -> None:
        pass

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    try:
        activity_mod.bind_sink(_send, loop)
        mint_root = tmp_path / "m"
        (mint_root / "capture").mkdir(parents=True)
        pending = fleet_cells.PendingSelfMint(
            family=FAMILY, cell_key="ck2-" + "b" * 56,
            ref=f"{cc.system_repo(FAMILY)}#ck2-{'b' * 56}",
            cfg=None, target=mint_root / "cell.tar.gz",
            capture_dir=mint_root / "capture", mint_root=mint_root,
            publisher=None, cache_dir=None,
        )
        pending._state["minted"] = object()
        fleet_cells.withhold_self_mint_publish(
            pending, "2/3 capture-sharing objects never proved")
        loop.run_until_complete(asyncio.sleep(0.05))
        events = [
            m.activity_update for m in sent
            if m.WhichOneof("msg") == "activity_update"
            and m.activity_update.kind == "self_mint_publish_withheld"
        ]
        assert events and "2/3" in events[0].detail

        # no-sink door
        mint_root2 = tmp_path / "m2"
        (mint_root2 / "capture").mkdir(parents=True)
        pending2 = fleet_cells.PendingSelfMint(
            family=FAMILY, cell_key="ck2-" + "c" * 56,
            ref=f"{cc.system_repo(FAMILY)}#ck2-{'c' * 56}",
            cfg=None, target=mint_root2 / "cell.tar.gz",
            capture_dir=mint_root2 / "capture", mint_root=mint_root2,
            publisher=None, cache_dir=None,
        )
        pending2._state["minted"] = object()
        pending2._state["meta"] = {}
        fleet_cells.publish_self_mint(pending2)
        loop.run_until_complete(asyncio.sleep(0.05))
        no_sink = [
            m.activity_update for m in sent
            if m.WhichOneof("msg") == "activity_update"
            and m.activity_update.kind == "self_mint_publish_withheld"
            and m.activity_update.phase == "no_sink"
        ]
        assert no_sink, "missing publish sink must be a typed event"
    finally:
        activity_mod.bind_sink(None, None)
        loop.close()


def test_oom_truncated_plan_never_finalizes_partial_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The finalize@8/18 shape: seeds OOM persistently mid-plan. Post-fix
    the driver retries bounded, then aborts LOUDLY with typed
    ``self_mint_abort`` events — it never converges into a finalize of a
    partial capture, and serving stays eager and alive. RED on 0.70.0:
    the OOM'd pass satisfied the stats-stable convergence and the mint
    finalized (phase=finalize) with nothing publishable and no event."""
    monkeypatch.setenv("GEN_WORKER_BG_YIELD", "1")
    h = _Harness(
        tmp_path, monkeypatch,
        compile_delay_s=0.02, seed_forward_s=0.01, tenant_forward_s=0.01,
        seed_oom_after=1)

    async def _run() -> None:
        await h.boot()
        bg = h.rec.background_mint
        assert bg is not None and bg.task is not None
        await asyncio.wait(
            {bg.task}, timeout=60.0, return_when=asyncio.ALL_COMPLETED)
        assert bg.task.done(), "mint driver hung on the OOM'd plan"
        # The mint must NOT have armed a partial capture.
        assert h.ex.serving_tiers() == {"generate": "eager"}, (
            "an OOM-truncated plan finalized a partial capture")
        aborts = h.events("self_mint_abort")
        assert aborts, "the OOM truncation never reached the wire"
        assert any(a.phase == "warmup_oom" for a in aborts)
        # Serving is alive: a tenant request still completes eager.
        res, _wall = await h.dispatch("r-after-oom", aspect="1:1")
        assert res.status == pb.JOB_STATUS_OK, res.safe_message

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 4 — Router seed-window holes (vocabulary overflow / dummy failure)
# ---------------------------------------------------------------------------


def test_seed_window_holes_never_compile_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED on 0.70.0: the _MAX_SIGS overflow and the dummy-build failure
    branches returned COMPILED even inside the seed window — an inline
    Dynamo+Inductor compile while holding the run gate. Post-fix both
    return EAGER and count ``seed_dropped`` so the driver aborts loudly."""
    router = hot_swap.Router()

    def compiled(*args: Any, **kwargs: Any) -> None:
        return None

    # Vocabulary overflow.
    monkeypatch.setattr(hot_swap, "_MAX_SIGS", 1)
    router.warm.add(("t", (("x",), ())))
    with hot_swap.mint_seed_window():
        verdict, _sig = router.route("t", compiled, ("overflow",), {})
    assert verdict == hot_swap.EAGER
    assert router.seed_dropped == 1

    # Dummy-build failure.
    monkeypatch.setattr(hot_swap, "_MAX_SIGS", 256)
    monkeypatch.setattr(
        hot_swap, "_dummy_value",
        lambda value, depth=0: (_ for _ in ()).throw(RuntimeError("boom")))
    fresh = hot_swap.Router()
    with hot_swap.mint_seed_window():
        verdict, _sig = fresh.route("t", compiled, ("nodummy",), {})
    assert verdict == hot_swap.EAGER
    assert fresh.seed_dropped == 1
    # Outside the window the legacy verdict stands.
    verdict, _sig = fresh.route("t", compiled, ("plain",), {})
    assert verdict == hot_swap.COMPILED
