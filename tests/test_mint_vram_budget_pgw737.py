"""pgw#737: the self-mint must never take the tenant request down.

The ie#535 wan-2.2 1.3.1 go-live measured, on gen-worker 0.75.1, an 80 GiB
H100 whose gw#587 background self-mint OOMed its warm plan three times and
whose TENANT request died with it — 26 of 40 denoise steps banked, 78.07 GiB
peak, `JOB_STATUS_RETRYABLE`, five hub re-dispatches of a deterministic
failure and a second H100 bought for nothing. Three tapes, one per defect:

  1. NO PRE-BUDGET: a capture was attempted on a card that could not hold
     it. Post-fix the mint reads free VRAM against a measured activation
     anchor and DECLINES (typed ``self_mint_skipped``, tier eager, cell
     absent) before a seed touches the card.
  2. NOT SURVIVABLE: the tenant OOM was terminal. Post-fix the mint is the
     evictable party — it is abandoned, its targets unwrapped, the allocator
     emptied, and the request re-runs eager on the clean card.
  3. RE-DISPATCHED: eager serving after a mint decline is a SUCCESS path.
     The request result is OK, not RETRYABLE — nothing for the hub's ladder
     to re-dispatch or to buy a pod for.

Plus the fence: an sdxl-class residency on the same rig still mints.

Self-contained (drop onto 0.75.1 to see it red).
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Annotated, Any, Dict, List

import msgspec
import pytest
import torch

import gen_worker
import gen_worker.executor as executor_mod
from gen_worker import (
    AxisClass,
    Compile,
    CompileAxis,
    RequestContext,
    Resources,
    endpoint,
    mint_budget,
    worker_function,
)
from gen_worker import compile_cache as cc
from gen_worker import fleet_cells, guard_closure, hot_swap
from gen_worker.api.binding import Hub, wire_ref
from gen_worker.executor import Executor, ModelStore
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs

GIB = 1 << 30
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
    monkeypatch.setattr(executor_mod, "_BG_COMPILE_QUIESCENCE_S", 0.02)
    monkeypatch.setattr(executor_mod, "_MINT_POLL_INTERVAL_S", 0.05)


def _card(
    monkeypatch: pytest.MonkeyPatch,
    *,
    total_gib: float,
    resident_gib: float,
    peak_gib: float,
    reserved_gib: float = 0.0,
) -> None:
    """Simulate ONE card reading. Only the four CUDA counters are faked —
    the free/reclaimable arithmetic, the activation anchor and the verdict
    are the shipped code. ``resident`` = weights on the card at the mint
    decision point, ``peak`` = the high-water the boot warm forward left."""
    total = int(total_gib * GIB)
    resident = int(resident_gib * GIB)
    reserved = int(max(reserved_gib, resident_gib) * GIB)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda dev=0: (total - reserved, total))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda dev=0: resident)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda dev=0: reserved)
    monkeypatch.setattr(
        torch.cuda, "max_memory_allocated", lambda dev=0: int(peak_gib * GIB))
    # flush_memory()/_peak_vram_bytes must stay inert on this fake card.
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)


class _Harness:
    """Real-executor rig (pgw#677 shape): ensure_setup boot, REAL
    handle_run_job tenant lane, instrumented compile leaf."""

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        compile_delay_s: float = 0.0,
        seed_forward_s: float = 0.0,
        tenant_forward_s: float = 0.01,
        tenant_oom_once: bool = False,
    ) -> None:
        self.tmp_path = tmp_path
        self.compile_delay_s = compile_delay_s
        self.seed_forward_s = seed_forward_s
        self.tenant_forward_s = tenant_forward_s
        self.tenant_oom_once = tenant_oom_once
        self.tenant_calls = 0
        self.compiles: List[str] = []
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
                harness.tenant_calls += 1
                if harness.tenant_oom_once and harness.tenant_calls == 1:
                    # The live shape: the OOM lands mid-denoise, with work
                    # already banked, while the mint holds the card.
                    raise torch.cuda.OutOfMemoryError(
                        "CUDA out of memory (sim, step 26/40)")
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
            guard_closure, "closure_manifest",
            lambda pipe, cfg, label="": {
                "v": 1, "graphs": [{"target": "transformer", "code": "sim",
                                    "entry": 0, "guards": []}],
                "verdicts": {}, "leaks": []})
        self.ex = Executor(self.specs, _send, store=store)

    def _fake_enable_compiled(
        self, pipe: Any, cfg: Any, cache_dir: Any = None,
        artifact: Any = None, publisher: Any = None,
    ) -> fleet_cells.ArmOutcome:
        mint_root = self.tmp_path / f"mint-{id(pipe)}"
        capture = mint_root / "capture"
        (capture / "inductor" / "fxgraph").mkdir(parents=True, exist_ok=True)
        pending = fleet_cells.PendingSelfMint(
            family=FAMILY, cell_key="ck1-" + "a" * 56,
            ref=f"{cc.system_repo(FAMILY)}#ck1-{'a' * 56}",
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
                # Recorded BEFORE the artifact write: a capture ATTEMPT is
                # what the budget must prevent, and an attempt whose capture
                # dir was already torn down still touched the card.
                harness.compiles.append(str(sig[0]))
                time.sleep(harness.compile_delay_s)
                try:
                    (capture / "inductor" / "fxgraph"
                     / f"g{len(harness.compiles)}.bin").write_bytes(b"graph")
                except OSError:
                    pass
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

    async def dispatch(
        self, request_id: str, aspect: str = "1:1",
    ) -> pb.JobResult:
        model_ref = wire_ref(self.spec.models["model"])
        run = pb.RunJob(
            request_id=request_id, attempt=1,
            function_name=self.spec.name,
            input_payload=msgspec.msgpack.encode(
                _In(prompt="a cat", aspect_ratio=aspect)),
            models=[pb.ModelBinding(slot="model", ref=model_ref)],
            snapshots={model_ref: pb.Snapshot(digest="blake3:" + "a" * 64)},
        )
        await self.ex.handle_run_job(run)
        job = self.ex.jobs[(run.request_id, run.attempt)]
        assert job.task is not None
        await job.task
        results = [m.job_result for m in self.sent
                   if m.WhichOneof("msg") == "job_result"
                   and m.job_result.request_id == request_id]
        assert results, f"no job_result for {request_id}"
        return results[-1]

    def events(self, kind: str) -> List[pb.ActivityUpdate]:
        return [
            m.activity_update for m in self.sent
            if m.WhichOneof("msg") == "activity_update"
            and m.activity_update.kind == kind
        ]


# ---------------------------------------------------------------------------
# The budget arithmetic itself, on the two measured cards
# ---------------------------------------------------------------------------


def test_wan22_card_declines_and_sdxl_card_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live wan-2.2 reading (80 GiB H100, ~54.2 GiB resident, a boot
    warm that peaked ~11 GiB above it) has no room for a capture the tenant
    must also fit around; an sdxl-class residency on the same card, and on a
    24 GiB card, does. Unprobeable never blocks."""
    _card(monkeypatch, total_gib=79.19, resident_gib=54.2, peak_gib=65.4)
    wan = mint_budget.probe()
    assert wan.probed and wan.measured and not wan.fits
    assert wan.need_bytes > wan.free_bytes
    assert "reason=insufficient_vram" in wan.line(
        "mint_skipped", "insufficient_vram")
    assert "headroom=" in wan.line("mint_skipped", "insufficient_vram")
    assert "needed~=" in wan.line("mint_skipped", "insufficient_vram")

    _card(monkeypatch, total_gib=79.19, resident_gib=7.0, peak_gib=9.5)
    assert mint_budget.probe().fits

    _card(monkeypatch, total_gib=23.6, resident_gib=7.0, peak_gib=9.5)
    assert mint_budget.probe().fits

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    blind = mint_budget.probe()
    assert blind.fits and not blind.probed


# ---------------------------------------------------------------------------
# 1 — PRE-BUDGET: decline before a seed touches the card
# ---------------------------------------------------------------------------


def test_mint_declines_on_a_card_that_cannot_hold_the_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED on 0.75.1: nothing checked, the plan ran, and the OOM took the
    tenant with it. Post-fix: no seed runs at all, one structured
    ``self_mint_skipped`` line on the wire, the tier stays eager, the cell
    stays absent, and the request SUCCEEDS."""
    h = _Harness(tmp_path, monkeypatch, compile_delay_s=0.05)
    _card(monkeypatch, total_gib=79.19, resident_gib=54.2, peak_gib=65.4)

    async def _run() -> None:
        await h.boot()
        await h.wait_mint()
        assert h.rec.background_mint is None, (
            "a mint was armed on a card that cannot hold its capture")
        assert not h.compiles, "a capture was attempted despite the budget"
        assert len(h.seed_runs) <= 1, (
            "the mint seeded its plan anyway (only the plain-eager boot "
            f"warm may run): {h.seed_runs}")
        skipped = h.events("self_mint_skipped")
        assert skipped, "the decline never reached the wire"
        assert skipped[-1].phase == "insufficient_vram"
        assert "mint_skipped reason=insufficient_vram" in skipped[-1].detail
        assert "headroom=" in skipped[-1].detail
        assert "needed~=" in skipped[-1].detail
        # Eager, cell absent, and no FAILED mint activity: a declined mint
        # is an outcome, not a worker failure (th#1228 — nothing here may
        # read as "re-dispatch me").
        assert h.ex.serving_tiers() == {"generate": "eager"}
        mints = [
            a for a in h.events("self_mint_compile")
            if a.state == pb.ActivityState.ACTIVITY_STATE_FAILED
        ]
        assert not mints, "a declined mint reported itself FAILED"
        res = await h.dispatch("r-declined")
        assert res.status == pb.JOB_STATUS_OK, res.safe_message

    asyncio.run(_run())


def test_small_resident_still_mints_on_the_same_rig(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fence: an sdxl-class residency has the headroom, so the budget
    must stay out of the way — the mint runs and the tier flips compiled."""
    h = _Harness(tmp_path, monkeypatch, compile_delay_s=0.02)
    _card(monkeypatch, total_gib=23.6, resident_gib=7.0, peak_gib=9.5)

    async def _run() -> None:
        await h.boot()
        await h.wait_mint()
        assert h.seed_runs, "the budget declined a mint that fits"
        assert not h.events("self_mint_skipped")
        assert h.ex.serving_tiers() == {"generate": "compiled"}

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 2 + 3 — SURVIVABLE, and a SUCCESS path
# ---------------------------------------------------------------------------


def test_tenant_oom_evicts_the_mint_and_the_request_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wan-2.2 tape, from the tenant's side: the request OOMs while the
    mint holds the card. RED on 0.75.1 — the request returned
    ``JOB_STATUS_RETRYABLE`` ("out of memory"), the hub re-dispatched it
    five times and bought a second H100 for a deterministic failure. Post
    fix the MINT loses: it is abandoned, its targets unwrapped, the card
    freed, and the same request re-runs eager to OK on this same worker."""
    h = _Harness(
        tmp_path, monkeypatch,
        compile_delay_s=1.5, seed_forward_s=0.05, tenant_oom_once=True)

    async def _run() -> None:
        await h.boot()
        assert h.rec.background_mint is not None
        res = await h.dispatch("r-oom", aspect="16:9")
        assert res.status == pb.JOB_STATUS_OK, (
            f"a tenant OOM against the worker's own mint was reported "
            f"{pb.JobStatus.Name(res.status)}: {res.safe_message}")
        assert h.tenant_calls == 2, "the request was not re-run in process"
        assert h.rec.background_mint is None, "the mint outlived the eviction"
        evicted = [
            a for a in h.events("self_mint_skipped") if a.phase == "tenant_oom"
        ]
        assert evicted, "the eviction never reached the wire"
        assert h.ex.serving_tiers() == {"generate": "eager"}
        # The targets are back to true eager: nothing left wrapped, so no
        # queued warm job can compile onto the card behind the tenant.
        for pipe in h.pipes:
            assert getattr(pipe, cc._MARKER_ATTR, None) is None, (
                "the evicted mint left its guarded wrapper installed")

    asyncio.run(_run())
