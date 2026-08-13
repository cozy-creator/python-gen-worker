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
from gen_worker import fleet_cells, guard_closure, hot_swap, mint_delegate
from gen_worker.api.binding import Hub, wire_ref
from gen_worker.executor import Executor, ModelStore
from gen_worker.lifecycle_intents import IntentRegistry
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
        # pgw#1010: every mint is a CHILD mint now. This rig has no child
        # process — without a stub it spawns a REAL one per attempt and the
        # test spends minutes importing torch to watch it fail. The serving
        # side is what pgw#671 is about: READY at eager tier first, the tier
        # flipping only when the mint lands.
        monkeypatch.setattr(
            mint_delegate, "build_cell", self._fake_build_cell)
        # pgw#1181: the pgw#681 mint gate this simmed is deleted with the
        # `torch-inductor-cache` format whose metadata carried its manifest.
        self.ex = Executor(self.specs, _send, store=store)

    # -- the child, minus the process ---------------------------------------

    async def _fake_build_cell(self, task: Any, **kwargs: Any) -> Any:
        pending = task.pending
        # The harness's own controls steer the CHILD now, exactly as they used
        # to steer the in-process compile: `compile_delay_s` is how long the
        # cell takes to appear (so a mid-build abandonment has something to
        # abandon), `compile_raises` is a child that produced nothing.
        abandon = kwargs.get("abandon")
        if self.compile_delay_s:
            if abandon is not None:
                try:
                    await asyncio.wait_for(
                        abandon.wait(), timeout=self.compile_delay_s)
                except asyncio.TimeoutError:
                    pass
                else:
                    # The parent pulled the plug mid-build: a real child is
                    # reaped and nothing is adopted.
                    return mint_delegate.DelegatedResult(
                        status=mint_delegate.ABANDONED, attempts=1,
                        detail="abandoned mid-build")
            else:
                await asyncio.sleep(self.compile_delay_s)
        if self.compile_raises:
            return mint_delegate.DelegatedResult(
                status=mint_delegate.FAILED, attempts=1,
                reason="synthetic_child_failure",
                detail="synthetic inductor failure")
        minted = fleet_cells.SelfMint(
            family=pending.family, cell_key=pending.arm_token,
            ref=pending.ref, snapshot_digest="sha256:" + "b" * 64,
            artifact=pending.target)
        pending._state["minted"] = minted
        pending.target.parent.mkdir(parents=True, exist_ok=True)
        pending.target.write_bytes(b"stub-cell")
        return mint_delegate.DelegatedResult(
            status=mint_delegate.ADOPTED, minted=minted, attempts=1)

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
    """The headline: READY at eager tier BEFORE any graph finished compiling;
    the background driver builds the cell and flips the tier — state never
    flaps.

    pgw#1010: the driver's compile is a CHILD PROCESS's now (the in-process
    seed/prove/pack phases only ever built a dynamo cell), so the assertions
    that counted this rig's own simulated compiles and its inherited warm
    memory are gone with them. What pgw#671 actually claims — READY does not
    wait for compiled serving, and the tier flips exactly once when the mint
    lands — is unchanged and is what stands below."""
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
        # the obligation identity needs the real runtime axes.)
        (target,) = h.ex.compile_targets()
        assert target.incarnation_id
        assert list(target.function_names) == ["generate"]
        assert target.active_compile_ref == ""

        await h.wait_mint()
        assert rec.background_mint is None
        (target,) = h.ex.compile_targets()
        assert target.active_compile_ref.startswith(
            cc.system_repo(FAMILY))
        assert target.active_compile_snapshot_digest.startswith("sha256:")
        assert h.ex.serving_tiers() == {"generate": "compiled"}
        assert cc.cell_proven_in_process(target.active_compile_ref)
        # The self_mint_compile activity outlived READY and COMPLETED from
        # the driver — never FAILED.
        states = h.activity_states("self_mint_compile")
        assert states.count(pb.ActivityState.ACTIVITY_STATE_COMPLETED) == 1
        assert pb.ActivityState.ACTIVITY_STATE_FAILED not in states

    asyncio.run(_run())


def test_eager_first_is_unconditional_and_no_env_restores_the_sequential_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#995 RED-VERIFICATION, inverted: the surviving path is the ONLY path.

    This test used to drive `GEN_WORKER_EAGER_FIRST_BOOT=0` to restore the
    pre-pgw#671 sequential ladder, and the elapsed-boot split between the two
    arms was the eager-vs-compiled latency evidence. That switch is deleted:
    it defaulted ON, no release ever declared it and no endpoint ever set it,
    so deleting it made the shape every pod already ran unconditional.

    An env used as a red-verification seam is the same defect as an env used as
    a feature gate — it is a behaviour switch that a rebuild can flip, and
    `GEN_WORKER_PREFER_AOT` is what that costs. So the assertion inverts: rather
    than proving the OFF arm still works, prove the OFF arm is UNREACHABLE.
    Setting the old name (or any plausible spelling of it) must change nothing.

    The latency evidence the old second arm carried is not lost — it is the
    absolute bound below, which is the half that actually stated the claim:
    time-to-READY does not pay the compile wall. The old test asserted BOTH
    `ready_at < 3 * delay` and `ready_at < h_off.ready_at`; the second is
    implied by the first whenever the sequential arm is honest, so the arm was
    paying for a comparison the bound already made.
    """
    delay = 0.3
    for name in (
        "GEN_WORKER_EAGER_FIRST_BOOT",
        "GEN_WORKER_EAGER_FIRST",
        "GEN_WORKER_EAGER_FIRST_BOOT_OFF",
    ):
        monkeypatch.setenv(name, "0")

    h = _Harness(tmp_path / "on", monkeypatch, compile_delay_s=delay)

    async def _run() -> None:
        await h.boot()
        # The pre-pgw#671 ladder would have compiled the full plan in the
        # foreground and left no background mint at all.
        assert h.rec.background_mint is not None, (
            "an env turned eager-first off — the pgw#995 deletion did not take, "
            "or a new switch was added on top of it")
        assert h.rec.ready is True
        await h.wait_mint()

    asyncio.run(_run())
    assert h.ready_at is not None
    # The whole point, stated as an absolute bound rather than a comparison
    # against an arm that no longer exists: READY does not wait for the wall.
    assert h.ready_at < 3 * delay


def test_no_env_read_survives_in_the_eager_first_and_bg_yield_paths() -> None:
    """pgw#995 structural guard: the deleted names are gone from the SOURCE.

    A behavioural assertion can pass while a second, unreached reader survives
    somewhere else in the module — which is exactly how `GEN_WORKER_PREFER_AOT`
    kept two live gates after one was believed removed. So read the source.
    """
    import gen_worker.executor as _ex
    import gen_worker.mint_delegate as _md
    import gen_worker.aot_wrapper_split as _ws

    src = "".join(
        Path(m.__file__).read_text() for m in (_ex, _md, _ws) if m.__file__)
    for gone in (
        "GEN_WORKER_EAGER_FIRST_BOOT",
        "GEN_WORKER_BG_YIELD",
        "GEN_WORKER_AOT_WRAPPER_SPLIT_OFF",
    ):
        # Prose that NAMES the deleted switch is fine and wanted; a read is not.
        assert f'environ.get("{gone}"' not in src, (
            f"{gone} is read again — pgw#995 deleted it because env must carry "
            f"config and secrets, never a branch selection")


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


# pgw#1010: `test_mandatory_quantized_execution_lane_keeps_the_sequential_gate`
# stood here — "eager is not a production lane for w8a8, so the boot keeps the
# foreground proof". Two rulings retired it: pgw#813 (a delegated pending arms
# NOTHING, so its eager tier is the untouched pipeline and a mandatory lane may
# defer) and pgw#1010 (there is no in-process foreground mint left to keep). The
# mandatory lane's protection is now stated where it belongs — it fails closed
# when the family declares no export, because the only cell is an AOT cell:
# `test_serve_finalize_pgw672.py::test_a_mandatory_lane_without_a_declaration_
# fails_closed_before_it_compiles`.


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
