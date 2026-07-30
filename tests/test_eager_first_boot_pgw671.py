"""pgw#671 eager-first boot (worker half of th#1187).

The startup ladder no longer serializes pipeline_loading ->
self_mint_compile -> ready on eager-compatible lanes: after weights load
and the derived warm plan's eager pass, the record goes READY at the EAGER
tier while the self-mint runs as a background task through the pgw#622
routers; the record hot-swaps to compiled when the mint arms. Proven here
through the REAL executor ensure_setup codepath (fakes only at the
download and compile-arm leaves):

  1. READY is reached BEFORE the graph compiles finish (the 30-min block
     is gone); the background driver seeds the FULL plan, proves, packs,
     publishes and flips the tier to compiled with no capability-state
     flap. The self_mint_compile activity stays RUNNING past READY and
     terminates from the driver.
  2. RED-VERIFICATION via the kill switch: GEN_WORKER_EAGER_FIRST_BOOT=0
     restores today's sequential gate — every compile completes BEFORE
     READY. The elapsed-time split is the eager-vs-compiled boot latency
     evidence.
  3. Clean abandonment (adopt-on-arm shape): mid-build abandonment
     discards the capture wholesale, keeps serving eager, suspends
     concurrent routing, and completes (never fails) the activity.
  4. A failed background compile keeps the function serving eager and
     reports the typed activity failure — a mint failure never un-serves.
  5. Mandatory-quantized lanes are ineligible (gw#586: eager is not a
     production lane there).
  6. The capability projection carries serving_tier on READY only
     (th#1187 wire contract).
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Dict, List, Optional

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
from gen_worker.intent_registry import IntentRegistry
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs

FAMILY = "sdxl"
_ASPECT_AXIS = CompileAxis(classes=(
    AxisClass("sq", match=lambda v: v == "1:1", warm="1:1"),
    AxisClass("wide", match=lambda v: v == "16:9", warm="16:9"),
    AxisClass("tall", match=lambda v: v == "9:16", warm="9:16"),
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
    armed = cc._armed_pipelines()
    for pipe in list(armed):
        armed.discard(pipe)
    yield
    with cc._PROVEN_CELLS_LOCK:
        cc._PROVEN_CELLS.clear()
    with fleet_cells._PENDING_LOCK:
        fleet_cells._PENDING.clear()
    for pipe in list(cc._armed_pipelines()):
        cc._armed_pipelines().discard(pipe)


class _Harness:
    """One eager-first boot fixture over the real executor codepath."""

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        compile_delay_s: float = 0.0,
        compile_raises: bool = False,
        weight_lane: str = "",
    ) -> None:
        self.tmp_path = tmp_path
        self.compile_delay_s = compile_delay_s
        self.compile_raises = compile_raises
        self.weight_lane = weight_lane
        self.compile_log: List[float] = []
        self.ready_at: Optional[float] = None
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
                if harness.weight_lane:
                    self.pipe._cozy_weight_lane = harness.weight_lane
                gen_worker.arm_compile(self.pipe)

            @worker_function()
            def generate(self, ctx: RequestContext, payload: _In) -> _Out:
                self.pipe.transformer.forward(payload.aspect_ratio)
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
        # pgw#681 gate at its torch boundary, simmed: this rig's compiles
        # never touch dynamo, so extraction would honestly report closure
        # unprovable and refuse every finalize.
        monkeypatch.setattr(
            guard_closure, "closure_manifest",
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
            family=FAMILY, cell_key="ck5-" + "a" * 56,
            ref=f"{cc.system_repo(FAMILY)}#ck5-{'a' * 56}",
            cfg=cfg, target=mint_root / "cell.tar.gz",
            capture_dir=capture, mint_root=mint_root,
            publisher=None, cache_dir=cache_dir,
        )
        original = pipe.transformer.forward
        compile_log = self.compile_log

        seen_sigs: set = set()
        seen_lock = threading.Lock()

        def compiled(*args: Any, **kwargs: Any) -> Any:
            # First call per signature = the (simulated) inductor compile:
            # burn wall clock and write a capture entry; later calls serve
            # from the "in-memory code cache", like the real thing.
            if self.compile_raises:
                raise RuntimeError("synthetic inductor failure")
            sig = hot_swap.signature(args, kwargs)
            with seen_lock:
                novel = sig not in seen_sigs
                seen_sigs.add(sig)
            if novel:
                time.sleep(self.compile_delay_s)
                (capture / "inductor" / "fxgraph" / f"g{len(compile_log)}.bin"
                 ).write_bytes(b"graph")
                compile_log.append(time.monotonic())
            return original(*args, **kwargs)

        signal: Dict[str, Any] = {
            "callback": None,
            "lock": threading.Lock(),
            "successful_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "router": hot_swap.Router(
                fail_closed=self.weight_lane.startswith(("w8a8", "w4a4"))),
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
        t0 = time.monotonic()
        instance = await self.ex.ensure_setup(self.spec, {
            wire_ref(self.spec.models["model"]): pb.Snapshot(
                digest="blake3:" + "a" * 64),
        })
        self.ready_at = time.monotonic() - t0
        return instance

    @property
    def rec(self) -> Any:
        return self.ex._classes[self.spec.instance_key]

    async def wait_mint(self, timeout: float = 30.0) -> None:
        bg = self.rec.background_mint
        if bg is None or bg.task is None:
            return
        await asyncio.wait_for(asyncio.shield(bg.task), timeout)

    def activity_states(self, kind: str) -> List[int]:
        return [
            m.activity_update.state for m in self.sent
            if m.WhichOneof("msg") == "activity_update"
            and m.activity_update.kind == kind
        ]


def test_eager_first_boot_ready_before_compile_then_hot_swaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The headline: READY at eager tier BEFORE any graph finished
    compiling; the background driver seeds the full plan (3 aspect
    classes), proves, packs, and flips the tier — state never flaps."""
    h = _Harness(tmp_path, monkeypatch, compile_delay_s=0.3)

    async def _run() -> None:
        await h.boot()
        rec = h.rec
        assert rec.ready is True
        assert rec.background_mint is not None
        # The gate is gone: not every graph compiled before READY (full
        # plan = 3 aspect classes; foreground ran the eager pass only).
        assert len(h.compile_log) < 3
        assert h.ex.serving_tiers() == {"generate": "eager"}
        # Targets are registered active-less while the mint builds, so the
        # incarnation is addressable for peer-cell adoption. (The requested
        # cell key itself is not computable on a CUDA-less test host —
        # cell_key.compute needs the real runtime axes.)
        (target,) = h.ex.compile_targets()
        assert target.incarnation_id
        assert list(target.function_names) == ["generate"]
        assert target.active_compile_ref == ""

        await h.wait_mint()
        assert rec.background_mint is None
        # The full plan compiled in the background (one per aspect class).
        assert len(h.compile_log) == 3
        (target,) = h.ex.compile_targets()
        assert target.active_compile_ref.startswith(
            cc.system_repo(FAMILY))
        assert target.active_compile_snapshot_digest.startswith("blake3:")
        assert h.ex.serving_tiers() == {"generate": "compiled"}
        assert cc.cell_proven_in_process(target.active_compile_ref)
        # pgw#654 memory: the full plan is inheritable by the next instance.
        (memory,) = h.ex._warm_contract_runs.values()
        assert len(memory) == 3
        # The self_mint_compile activity outlived READY and COMPLETED from
        # the driver — never FAILED.
        states = h.activity_states("self_mint_compile")
        assert states.count(pb.ActivityState.ACTIVITY_STATE_COMPLETED) == 1
        assert pb.ActivityState.ACTIVITY_STATE_FAILED not in states

    asyncio.run(_run())


def test_kill_switch_restores_the_sequential_gate_and_measures_the_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED-VERIFICATION of the gate removal: with the env kill switch the
    old ladder is back — every compile lands BEFORE READY — and the
    elapsed-boot split is the eager-vs-compiled latency evidence."""
    delay = 0.3
    monkeypatch.setenv("GEN_WORKER_EAGER_FIRST_BOOT", "0")
    h_off = _Harness(tmp_path / "off", monkeypatch, compile_delay_s=delay)

    async def _off() -> None:
        await h_off.boot()
        assert h_off.rec.ready is True
        assert h_off.rec.background_mint is None
        # Sequential: the full plan compiled foreground, gating READY.
        assert len(h_off.compile_log) == 3
        assert h_off.ex.serving_tiers() == {"generate": "compiled"}

    asyncio.run(_off())
    assert h_off.ready_at is not None and h_off.ready_at >= 3 * delay

    monkeypatch.setenv("GEN_WORKER_EAGER_FIRST_BOOT", "1")
    h_on = _Harness(tmp_path / "on", monkeypatch, compile_delay_s=delay)

    async def _on() -> None:
        await h_on.boot()
        assert h_on.rec.background_mint is not None
        await h_on.wait_mint()

    asyncio.run(_on())
    assert h_on.ready_at is not None
    # The whole point: time-to-READY no longer pays the compile wall.
    assert h_on.ready_at < 3 * delay
    assert h_on.ready_at < h_off.ready_at


def test_mid_build_abandonment_is_clean_and_keeps_serving_eager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adopt-on-arm shape: abandoning the build mid-flight discards the
    capture wholesale (nothing half-packed, nothing published), suspends
    concurrent routing, keeps the function serving eager, and completes
    the activity (an abandoned mint is not a failure)."""
    h = _Harness(tmp_path, monkeypatch, compile_delay_s=2.0)

    async def _run() -> None:
        await h.boot()
        rec = h.rec
        bg = rec.background_mint
        assert bg is not None
        pendings = list(bg.pendings.values())
        assert pendings
        await h.ex.abandon_background_mint(rec, reason="peer cell adopting")
        assert rec.background_mint is None
        for pending in pendings:
            assert not pending.mint_root.exists()
            assert pending._state.get("minted") is None
        (pipe,) = h.pipes
        router = hot_swap.router_of(pipe)
        assert router is not None and router.concurrent is False
        assert rec.ready is True
        assert h.ex.serving_tiers() == {"generate": "eager"}
        (target,) = h.ex.compile_targets()
        assert target.active_compile_ref == ""
        states = h.activity_states("self_mint_compile")
        assert states.count(pb.ActivityState.ACTIVITY_STATE_COMPLETED) == 1
        assert pb.ActivityState.ACTIVITY_STATE_FAILED not in states

    asyncio.run(_run())


def test_failed_background_compile_stays_eager_and_reports_typed_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _Harness(tmp_path, monkeypatch, compile_raises=True)

    async def _run() -> None:
        await h.boot()
        rec = h.rec
        assert rec.ready is True
        bg = rec.background_mint
        assert bg is not None and bg.task is not None
        # The driver fails on the bg_failed signatures; serving is intact.
        try:
            await asyncio.wait_for(asyncio.shield(bg.task), timeout=30.0)
        except Exception:
            pass
        assert rec.background_mint is None
        assert rec.ready is True
        assert h.ex.serving_tiers() == {"generate": "eager"}
        states = h.activity_states("self_mint_compile")
        assert pb.ActivityState.ACTIVITY_STATE_FAILED in states

    asyncio.run(_run())


def test_mandatory_quantized_lane_keeps_the_sequential_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gw#586: eager is not a production lane for w8a8 — the boot must keep
    today's foreground proof (and its fail-closed policy)."""
    h = _Harness(tmp_path, monkeypatch, compile_delay_s=0.05,
                 weight_lane="w8a8")

    async def _run() -> None:
        await h.boot()
        rec = h.rec
        assert rec.ready is True
        assert rec.background_mint is None  # never deferred
        assert len(h.compile_log) == 3     # compiled foreground, pre-READY

    asyncio.run(_run())


def test_capability_projection_carries_serving_tier_on_ready_only() -> None:
    """th#1187 wire contract: serving_tier rides FunctionCapability field 9
    on READY capabilities; every other state (and a stub executor without
    the surface) projects the empty pre-0.65 tier."""
    registry = IntentRegistry("release-1", ["echo"], boot_config_generation=1)
    command = pb.DesiredStateCommand(
        worker_session_id=registry.worker_session_id,
        command_seq=1,
        goal_id="goal-1",
        release_id="release-1",
        config_generation=1,
        config_digest=b"digest-1",
        parameter_snapshot=msgspec.msgpack.encode({}),
        first_action_by_unix_ms=9_000_000_000_000,
        intents=[pb.DesiredIntent(
            intent_id="config-1",
            kind=pb.DESIRED_INTENT_KIND_CONFIG_APPLY,
            cause=pb.DESIRED_INTENT_CAUSE_CONFIG_CHANGE,
            mandatory=True,
        )],
        mandatory=True,
    )
    assert registry.apply_command(command).status == (
        pb.GOAL_RECEIPT_STATUS_ACCEPTED)
    registry.config_snapshot_applied(1)
    registry.bindings_applied(1)
    runtime_config = SimpleNamespace(
        generation=1, parameter_snapshot_generation=1)
    base = dict(
        runtime_config=runtime_config,
        store=SimpleNamespace(residency_snapshot=lambda: []),
        available_functions=lambda: ["echo"],
        compile_targets=lambda: [],
        unavailable={},
    )
    desired = pb.DesiredResidency(hot=[pb.DesiredInstance(function_name="echo")])

    executor = SimpleNamespace(
        **base, serving_tiers=lambda: {"echo": "eager"})
    registry.refresh_projection(executor, desired, {})
    (capability,) = registry.snapshot().capabilities
    assert capability.state == pb.FUNCTION_CAPABILITY_STATE_READY
    assert capability.serving_tier == "eager"

    executor = SimpleNamespace(
        **base, serving_tiers=lambda: {"echo": "compiled"})
    registry.refresh_projection(executor, desired, {})
    (capability,) = registry.snapshot().capabilities
    assert capability.serving_tier == "compiled"
    assert capability.state == pb.FUNCTION_CAPABILITY_STATE_READY

    # A legacy/stub executor without the surface keeps the empty tier.
    executor = SimpleNamespace(**base)
    registry.refresh_projection(executor, desired, {})
    (capability,) = registry.snapshot().capabilities
    assert capability.serving_tier == ""

    # Non-READY states never carry a tier.
    non_ready = dict(base, available_functions=lambda: [])
    executor = SimpleNamespace(
        **non_ready, serving_tiers=lambda: {"echo": "eager"})
    registry.refresh_projection(executor, desired, {})
    (capability,) = registry.snapshot().capabilities
    assert capability.state != pb.FUNCTION_CAPABILITY_STATE_READY
    assert capability.serving_tier == ""
