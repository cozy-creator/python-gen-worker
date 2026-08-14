"""A background mint must not starve the tenant, oversize a stolen turn, or
lose its publish outcome. Three failure modes, each pinned here:

  1. ELIGIBILITY MISCLASSIFICATION (the starvation): sdxl's mixed
     ``#fp8-w8a8``-storage checkpoint stamps ``_cozy_weight_lane =
     "w8a8-lora64"`` while the hub serves it ``fp8-w8a16+eager``. The
     an eligibility check reading the weight-lane prefix classifies the
     boot mandatory-quantized and silently falls back to the FOREGROUND
     compile-then-serve mint, so the tenant sits inside ensure_setup for
     the whole inline-compile plan and the gate/preemption machinery
     never runs. ONE serveability brain
     (``compile_cache.mandatory_serving``): the hub-resolved execution
     lane outranks the weight-lane stamp.
  2. STOLEN-COMPILE SIZING: a stolen turn is not preemptible and a real
     inductor compile is 4-7 unabortable minutes — ~100x the advertised
     30-90s residual. Compile turns now steal only against MINUTES of
     continuous demand (``_BG_COMPILE_STEAL_FLOOR_S``) and announce the
     steal on the wire.
  3. THE PUBLISH BREAK: a mint-abort / publish-withhold /
     closure-refusal exit dying in unreachable pod logs. All of them
     ride the wire as typed events (``self_mint_abort``,
     ``self_mint_publish_withheld``, ``self_mint_publish_failed``), and
     an OOM-truncated warm plan cannot converge silently into a partial
     finalize.
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
from gen_worker import fleet_cells, hot_swap, mint_supervisor
from gen_worker.api.binding import Hub, wire_ref
from gen_worker.executor import Executor
from gen_worker.models.store import ModelStore
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs
from gen_worker.models import store as store_mod

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
        hub_execution_lane: str = "",
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

        monkeypatch.setattr(store_mod, "ensure_local", _fake_ensure_local)
        monkeypatch.setattr(
            fleet_cells, "enable_compiled", self._fake_enable_compiled)
        # Every mint is a CHILD mint now (the in-process capture only
        # ever built a dynamo cell). This harness has no child process, so the
        # child's OUTCOME is stubbed — what it is testing is the serving side
        # around it: eager-first, the tenant never starving, the router, and
        # the wire. The compile the harness performs is the one its own
        # `compiled` wrapper does, on the boot/warm thread, exactly as before.
        monkeypatch.setattr(
            mint_supervisor, "supervise", self._fake_build_cell)
        # The pgw#681 mint gate this simmed is deleted.
        # `guard_closure.closure_manifest` classified every compiled graph at
        # the MINT and wrote the result into the cell's metadata; it went with
        # the `torch-inductor-cache` format that carried it, so a rig whose
        # compiles never touch dynamo has no gate left to satisfy.
        self.ex = Executor(self.specs, _send, store=store)
        if hub_execution_lane:
            ref = wire_ref(self.spec.models["model"])
            self.ex._model_resolutions = {ref: (ref, "", hub_execution_lane)}

    async def _fake_build_cell(self, task: Any, **kwargs: Any) -> Any:
        """The child, minus the process: it adopts the pending it was given."""
        pending = task.pending
        minted = fleet_cells.SelfMint(
            family=pending.family, cell_key=pending.arm_token,
            ref=pending.ref, snapshot_digest="sha256:" + "b" * 64,
            artifact=pending.target)
        pending._state["minted"] = minted
        pending.target.parent.mkdir(parents=True, exist_ok=True)
        pending.target.write_bytes(b"stub-cell")
        return mint_supervisor.SupervisedResult(
            status=mint_supervisor.ADOPTED, minted=minted, attempts=1)

    def _fake_enable_compiled(
        self, pipe: Any, cfg: Any, cache_dir: Any = None,
        artifact: Any = None, publisher: Any = None,
    ) -> fleet_cells.ArmOutcome:
        mint_root = self.tmp_path / f"mint-{id(pipe)}"
        # OUTSIDE mint_root. The publish gate rmtree's mint_root when
        # a mint resolves with no sink, and this rig's simulated compile writes
        # its "graphs" long after that.
        capture = self.tmp_path / f"capture-{id(pipe)}"
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
    h = _Harness(
        tmp_path, monkeypatch,
        compile_delay_s=0.4, seed_forward_s=0.05, tenant_forward_s=0.02,
        weight_lane="w8a8", hub_execution_lane="fp8-w8a16+eager")

    async def _run() -> None:
        boot_t0 = time.monotonic()
        await h.boot()
        boot_wall = time.monotonic() - boot_t0
        # Eager-first: READY without paying the plan's compiles foreground.
        assert h.rec.background_mint is not None, (
            "w8a8-stamped pipe with hub lane fp8-w8a16+eager was refused "
            "eager-first — the foreground compile-then-serve mint is the "
            "reopen's measured 26-minute starvation")
        assert boot_wall < 6.0, f"boot paid the plan foreground: {boot_wall:.1f}s"
        # A tenant request mid-mint completes at serving latency.
        res, wall = await h.dispatch("r-live", aspect="16:9")
        assert res.status == pb.JOB_STATUS_OK, res.safe_message
        assert wall < 1.0, f"tenant starved during mint: {wall:.2f}s"
        await h.wait_mint()
        assert h.ex.serving_tiers() == {"generate": "compiled"}

    asyncio.run(_run())


def test_a_true_mandatory_lane_mints_in_a_child_and_is_fenced_meanwhile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The qwen shape: hub lane says real w8a8 activations — eager is not a
    production tier there.

    This used to assert the boot kept the SEQUENTIAL FOREGROUND proof, because
    the in-process capture needed a router the mandatory lane could not have.
    pgw#813 already dissolved that (a delegated pending arms NOTHING, so its
    eager tier is the untouched pipeline), and pgw#1010 removed the in-process
    shape entirely — so a mandatory lane mints in a child like every other
    lane. What protects the tenant is not a foreground proof any more, it is
    the dispatch fence: no cell, no active compile incarnation, no dispatch.
    """
    h = _Harness(
        tmp_path, monkeypatch,
        compile_delay_s=0.0, seed_forward_s=0.0, tenant_forward_s=0.01,
        weight_lane="w8a8", hub_execution_lane="fp8-w8a8-dynamic+compiled")

    async def _run() -> None:
        await h.boot()
        assert h.rec.background_mint is not None, (
            "a mandatory lane mints out of process like every other lane "
            "(pgw#813/pgw#1010) — the eager tier it serves from meanwhile is "
            "the untouched pipeline")
        await h.wait_mint()

    asyncio.run(_run())


def test_w8a8_stamp_without_execution_lane_evidence_stays_foreground(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No hub lane evidence: the weight-lane stamp remains the fail-closed
    fallback for SERVING — but, since pgw#813/pgw#1010, not for the mint shape:
    the cell is built in a child either way, because that is the only shape
    there is."""
    h = _Harness(
        tmp_path, monkeypatch,
        weight_lane="w8a8", hub_execution_lane="")

    async def _run() -> None:
        await h.boot()
        assert h.rec.background_mint is not None
        await h.wait_mint()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 2 — SIZING: a multi-minute unabortable compile + waiting tenant. The
#     compile lane must not steal against short demand; the tenant completes.
# ---------------------------------------------------------------------------


# `test_multi_minute_compile_never_steals_against_live_demand` stood
# here. Its mechanism was the IN-PROCESS mint's background compile turns
# competing with tenant demand — the mint compiles in a child process now, so
# there are no in-process mint turns to steal with. The steal floors
# (`_BG_STEAL_FLOOR_S` / `_BG_COMPILE_STEAL_FLOOR_S`) are untouched and still
# govern the pgw#622 router's background shape warms; the wire half of the
# doctrine is the test immediately below, which is unchanged.


def test_compile_steal_is_announced_on_the_wire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the compile floor DOES elapse under truly continuous demand,
    the steal happens (minimum progress) and announces itself as a typed
    ``bg_turn_steal`` event."""
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


# `test_pack_or_closure_refusal_reaches_the_wire` and
# `test_oom_truncated_plan_never_finalizes_partial_capture` stood here. Both
# assert a PACK — of an in-process inductor capture, into a dynamo cell — and
# both go with it. The surviving half of "a refusal reaches the wire" is the
# child's typed abort (`test_mint_abort_classification_th1299.py`) and the
# publish/withhold gate (`test_fleet_cells.py`), which the test between them
# still exercises here.


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
            family=FAMILY, arm_token="cg-key-v1-" + "b" * 56,
            ref=f"{cc.system_repo(FAMILY)}#cg-key-v1-{'b' * 56}",
            cfg=None, target=mint_root / "cell.tar.gz", mint_root=mint_root,
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
            family=FAMILY, arm_token="cg-key-v1-" + "c" * 56,
            ref=f"{cc.system_repo(FAMILY)}#cg-key-v1-{'c' * 56}",
            cfg=None, target=mint_root2 / "cell.tar.gz", mint_root=mint_root2,
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


# ---------------------------------------------------------------------------
# 4 — Router seed-window holes (vocabulary overflow / dummy failure)
# ---------------------------------------------------------------------------


def test_seed_window_holes_never_compile_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED on 0.70.0: the _MAX_SIGS overflow and the dummy-build failure
    branches returned COMPILED even inside the seed window — an inline
    Dynamo+Inductor compile while holding the run gate. Post-fix both
    return EAGER and count ``seed_dropped`` so the driver aborts loudly.

    pgw#1215 step 4: both routers wire the turn gate, exactly as the mint
    pipes do in production (`Executor._wire_turn_gate` runs before
    `hot_swap.enable` and therefore before the first seed can route). An
    ungated router is a typed refusal now, not a mode.
    """
    router = hot_swap.Router()
    router.set_turn_gate(lambda kind: contextlib.nullcontext())

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
    fresh.set_turn_gate(lambda kind: contextlib.nullcontext())
    with hot_swap.mint_seed_window():
        verdict, _sig = fresh.route("t", compiled, ("nodummy",), {})
    assert verdict == hot_swap.EAGER
    assert fresh.seed_dropped == 1
    # Outside the window the sequential verdict stands.
    verdict, _sig = fresh.route("t", compiled, ("plain",), {})
    assert verdict == hot_swap.COMPILED
