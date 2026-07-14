"""gw#463: CUDA OOM never fatals — degraded mode is the fit-ladder's terminal rung.

Real tiny DDPMPipeline (the gw#417 pattern) through the REAL Executor setup /
injection / job codepaths; only the device-placement boundary is shimmed — a
DDPMPipeline subclass whose ``.to("cuda")`` raises the real
``torch.cuda.OutOfMemoryError`` and whose offload hooks record instead of
touching CUDA — so every fallback line (place_pipeline ladder, executor
demotion bookkeeping, retry-once, FnDegraded re-emit) executes end to end on a
CUDA-less host.
"""

import asyncio
import logging
from pathlib import Path

import msgspec
import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

from gen_worker.api.binding import HF
from gen_worker.api.decorators import Resources
from gen_worker.executor import Executor, ModelStore, _map_exception
from gen_worker.models import memory
from gen_worker.models.serve_fit import RUN_OFFLOAD, ServePlan, demoted, plan_serve
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec

OOM = torch.cuda.OutOfMemoryError


def _save_tiny_ddpm(path: Path) -> None:
    unet = UNet2DModel(
        sample_size=8, in_channels=3, out_channels=3, layers_per_block=1,
        block_out_channels=(32, 32), norm_num_groups=8,
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )
    DDPMPipeline(unet=unet, scheduler=DDPMScheduler(num_train_timesteps=10)
                 ).save_pretrained(str(path))


class ShimPipe(DDPMPipeline):
    """Real pipeline; only the device-placement boundary is shimmed.

    ``.to('cuda')`` raises the real allocator exception while ``oom_on_cuda``
    holds; offload hooks record calls instead of touching CUDA.
    """

    oom_on_cuda = True
    to_cuda_attempts = 0
    offload_calls: list = []

    @classmethod
    def reset(cls, *, oom_on_cuda: bool = True) -> None:
        cls.oom_on_cuda = oom_on_cuda
        cls.to_cuda_attempts = 0
        cls.offload_calls = []

    def to(self, *args, **kwargs):
        wants_cuda = any("cuda" in str(v) for v in (*args, *kwargs.values()))
        if wants_cuda:
            type(self).to_cuda_attempts += 1
            if type(self).oom_on_cuda:
                raise OOM("CUDA out of memory (injected, gw#463 test)")
        return self

    def enable_model_cpu_offload(self, *a, **k):
        type(self).offload_calls.append("model_offload")

    def enable_group_offload(self, *a, **k):
        type(self).offload_calls.append("group_offload")

    def enable_sequential_cpu_offload(self, *a, **k):
        type(self).offload_calls.append("sequential")


@pytest.fixture()
def shim_env(monkeypatch, tmp_path_factory):
    """CUDA-less ladder harness: cuda 'present', 20 GB free."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(memory, "get_available_vram_gb", lambda *a, **k: 20.0)
    ShimPipe.reset()
    root = tmp_path_factory.mktemp("tiny-ddpm")
    _save_tiny_ddpm(root / "snap")
    return root / "snap"


# --------------------------------------------------------------------------- #
# Ladder vocabulary units
# --------------------------------------------------------------------------- #


def test_offload_rung_ordering() -> None:
    assert memory.next_offload_rung("") == "model_offload"
    assert memory.next_offload_rung("off") == "model_offload"
    assert memory.next_offload_rung("vae_only") == "model_offload"
    assert memory.next_offload_rung("model_offload") == "group_offload"
    assert memory.next_offload_rung("group_offload") == "sequential"
    assert memory.next_offload_rung("sequential") is None
    assert memory.deeper_offload_mode("model_offload", "sequential") == "sequential"
    assert memory.deeper_offload_mode("", "group_offload") == "group_offload"
    assert memory.deeper_offload_mode("model_offload", "") == "model_offload"


def test_is_cuda_oom_shapes() -> None:
    assert memory.is_cuda_oom(OOM("CUDA out of memory"))
    assert memory.is_cuda_oom(RuntimeError("CUDA error: out of memory"))
    assert memory.is_cuda_oom(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"))
    assert not memory.is_cuda_oom(RuntimeError("expected all tensors on the same device"))
    assert not memory.is_cuda_oom(ValueError("out of memory"))  # not allocator-shaped
    assert not memory.is_cuda_oom(None)


def test_allocator_runtime_error_maps_retryable_not_fatal() -> None:
    status, msg = _map_exception(RuntimeError("CUDA error: out of memory"))
    assert status == pb.JOB_STATUS_RETRYABLE and msg == "out of memory"


def test_demoted_is_a_ladder_transition() -> None:
    plan = ServePlan(serveable=True, run_mode="native", fit="fits",
                     wanted="bf16", ran="bf16")
    d = demoted(plan, detail="CUDA OOM", placement_mode="model_offload")
    assert d.degraded and d.run_mode == RUN_OFFLOAD
    assert d.ran == "offload:model_offload"
    d2 = demoted(d, detail="again", placement_mode="sequential")
    assert d2.ran == "offload:sequential"
    # No prior plan (gate never ran): still produces a reportable plan.
    assert demoted(None, detail="x", placement_mode="model_offload").degraded


# --------------------------------------------------------------------------- #
# Catch-site (a): setup/load — place_pipeline's OOM ladder
# --------------------------------------------------------------------------- #


def test_load_oom_demotes_to_model_offload(shim_env, caplog) -> None:
    pipe = ShimPipe.from_pretrained(str(shim_env))
    with caplog.at_level(logging.WARNING):
        applied = memory.place_pipeline(pipe, ref="acme/tiny")
    assert applied["mode"] == "model_offload"
    assert applied["oom_demotions"] == 1
    assert ShimPipe.to_cuda_attempts == 1
    assert ShimPipe.offload_calls == ["model_offload"]
    assert memory.low_vram_mode(pipe) == "model_offload"
    line = next(r.message for r in caplog.records if "DEGRADED_MODE" in r.message)
    assert "DEGRADED_MODE=engaged" in line and "phase=load" in line
    assert "model=acme/tiny" in line and "rung=off->model_offload" in line
    assert "needed_gb=" in line and "free_gb=" in line


def test_planned_offload_skips_doomed_resident_attempt(shim_env) -> None:
    pipe = ShimPipe.from_pretrained(str(shim_env))
    applied = memory.place_pipeline(pipe, mode="model_offload", ref="acme/tiny")
    assert applied["mode"] == "model_offload"
    assert "oom_demotions" not in applied
    assert ShimPipe.to_cuda_attempts == 0  # ie#369: no doomed resident attempt


def test_demote_pipeline_walks_and_terminates(shim_env) -> None:
    pipe = ShimPipe.from_pretrained(str(shim_env))
    assert memory.demote_pipeline(pipe) == "model_offload"
    assert memory.demote_pipeline(pipe) == "group_offload"
    assert memory.demote_pipeline(pipe) == "sequential"
    assert memory.demote_pipeline(pipe) is None  # terminal rung: stays put
    assert memory.low_vram_mode(pipe) == "sequential"


# --------------------------------------------------------------------------- #
# Executor end-to-end harness (real ModelStore + real injection/job paths)
# --------------------------------------------------------------------------- #


class _In(msgspec.Struct):
    x: str = "hi"


class _Out(msgspec.Struct):
    ok: bool = True


def _spec(cls) -> EndpointSpec:
    return EndpointSpec(
        name="fn", method=cls.run, kind="inference", payload_type=_In,
        output_mode="single", cls=cls, attr_name="run",
        models={"m": HF("acme/tiny")}, resources=Resources(vram_gb=1.0),
    )


def _executor(spec: EndpointSpec, tmp_path: Path, snap: Path, sent: list) -> Executor:
    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    store = ModelStore(_send, cache_dir=tmp_path, vram_budget_bytes=4 << 30)

    async def _fake_ensure_local(ref, snapshot=None, *, binding=None) -> Path:
        store.residency.track_disk(ref, snap)
        return snap

    store.ensure_local = _fake_ensure_local  # type: ignore[method-assign]
    return Executor([spec], _send, store=store)


async def _run_job(ex: Executor, rid: str) -> None:
    await ex.handle_run_job(pb.RunJob(
        request_id=rid, attempt=1, function_name="fn",
        input_payload=msgspec.msgpack.encode(_In())))
    job = ex.jobs[(rid, 1)]
    assert job.task is not None
    await job.task
    for _ in range(20):  # drain event coroutines from the handler thread
        await asyncio.sleep(0)


def _result(sent: list, rid: str) -> pb.JobResult:
    for m in sent:
        if m.WhichOneof("msg") == "job_result" and m.job_result.request_id == rid:
            return m.job_result
    raise AssertionError(f"no job_result for {rid}")


def test_setup_oom_learns_floor_and_skips_resident_on_reload(
        shim_env, tmp_path, caplog) -> None:
    class Endpoint:
        def setup(self, m: ShimPipe) -> None:
            self.m = m

        def run(self, ctx, payload: _In) -> _Out:  # pragma: no cover
            return _Out()

    spec = _spec(Endpoint)
    sent: list = []

    async def _go() -> None:
        ex = _executor(spec, tmp_path, shim_env, sent)
        with caplog.at_level(logging.WARNING):
            inst = await ex.ensure_setup(spec)
        assert memory.low_vram_mode(inst.m) == "model_offload"
        assert ShimPipe.to_cuda_attempts == 1
        # Learned floor + demoted ServePlan + loud warning.
        assert ex.degraded_floor["acme/tiny"] == "model_offload"
        plan = ex.serve_plans["fn"]
        assert plan.degraded and plan.ran == "offload:model_offload"
        assert any("DEGRADED_MODE=engaged" in r.message and "fn=fn" in r.message
                   for r in caplog.records)
        # Reload (eviction simulated): the floor drives placement — the
        # doomed resident attempt is NOT paid again (ie#369).
        rec = ex._classes[spec.instance_key]
        rec.ready, rec.instance = False, None
        await ex.ensure_setup(spec)
        assert ShimPipe.to_cuda_attempts == 1  # unchanged

    asyncio.run(_go())


def test_inference_oom_demotes_and_retries_once(shim_env, tmp_path, caplog) -> None:
    calls = {"n": 0}

    class Endpoint:
        def setup(self, m: ShimPipe) -> None:
            self.m = m

        def run(self, ctx, payload: _In) -> _Out:
            calls["n"] += 1
            if calls["n"] == 1:
                raise OOM("CUDA out of memory (injected mid-inference)")
            return _Out()

    ShimPipe.reset(oom_on_cuda=False)  # loads resident; the OOM comes mid-run
    spec = _spec(Endpoint)
    sent: list = []

    async def _go() -> None:
        ex = _executor(spec, tmp_path, shim_env, sent)
        with caplog.at_level(logging.WARNING):
            await _run_job(ex, "r1")
        assert _result(sent, "r1").status == pb.JOB_STATUS_OK
        assert calls["n"] == 2  # retried exactly once, in degraded mode
        pipe = ex.store.residency.obj("acme/tiny")
        assert memory.low_vram_mode(pipe) == "model_offload"
        assert ex.degraded_floor["acme/tiny"] == "model_offload"
        assert ex.serve_plans["fn"].ran == "offload:model_offload"
        assert any("DEGRADED_MODE=engaged" in r.message and "phase=inference" in r.message
                   for r in caplog.records)
        # ie#369 bar: the degradation is platform-visible as a request event.
        assert any(
            m.WhichOneof("msg") == "job_progress"
            and b"DEGRADED_MODE=engaged" in m.job_progress.data
            for m in sent)
        # Sticky: the next request stays degraded — no rung flapping.
        await _run_job(ex, "r2")
        assert _result(sent, "r2").status == pb.JOB_STATUS_OK
        assert memory.low_vram_mode(pipe) == "model_offload"

    asyncio.run(_go())


def test_inference_oom_in_degraded_mode_fails_retryable(shim_env, tmp_path) -> None:
    calls = {"n": 0}

    class Endpoint:
        def setup(self, m: ShimPipe) -> None:
            self.m = m

        def run(self, ctx, payload: _In) -> _Out:
            calls["n"] += 1
            raise OOM("CUDA out of memory (persistent)")

    ShimPipe.reset(oom_on_cuda=False)
    spec = _spec(Endpoint)
    sent: list = []

    async def _go() -> None:
        ex = _executor(spec, tmp_path, shim_env, sent)
        await _run_job(ex, "r1")
        # One degraded retry, then the mapped status — RETRYABLE, never FATAL
        # (a bigger card can serve it); the ladder floor deepened.
        assert calls["n"] == 2
        assert _result(sent, "r1").status == pb.JOB_STATUS_RETRYABLE
        assert _result(sent, "r1").safe_message == "out of memory"
        assert ex.degraded_floor["acme/tiny"] == "model_offload"
        # Next request: demotes one MORE rung (group_offload), retries, fails
        # RETRYABLE again — walking, not flapping.
        await _run_job(ex, "r2")
        assert calls["n"] == 4
        assert _result(sent, "r2").status == pb.JOB_STATUS_RETRYABLE
        assert ex.degraded_floor["acme/tiny"] == "group_offload"

    asyncio.run(_go())


def test_runtime_demotion_reemits_fn_degraded(shim_env, tmp_path) -> None:
    """The rung transition is telemetry, not just a log line: FnDegraded
    re-emits when the active rung changes (lifecycle dedupe is per-rung)."""
    from types import SimpleNamespace

    from gen_worker.lifecycle import Lifecycle

    class Endpoint:
        def setup(self, m: ShimPipe) -> None:
            self.m = m

        def run(self, ctx, payload: _In) -> _Out:  # pragma: no cover
            return _Out()

    class _StubTransport:
        def __init__(self) -> None:
            self.sent: list = []
            self.connected = True
            self.queue = SimpleNamespace(pending_result_keys=set())

        async def send(self, msg: pb.WorkerMessage) -> None:
            self.sent.append(msg)

    spec = _spec(Endpoint)
    sent: list = []

    async def _go() -> None:
        ex = _executor(spec, tmp_path, shim_env, sent)
        lc = Lifecycle(SimpleNamespace(worker_jwt="", worker_id="w", runpod_pod_id=""), ex)
        tp = _StubTransport()
        lc.transport = tp
        ex._record_demotion(spec, ref="acme/tiny", phase="inference",
                            from_rung="resident", to_rung="model_offload")
        await lc.maybe_send_state_delta()
        ex._record_demotion(spec, ref="acme/tiny", phase="inference",
                            from_rung="model_offload", to_rung="sequential")
        await lc.maybe_send_state_delta()
        events = [m.fn_degraded for m in tp.sent if m.WhichOneof("msg") == "fn_degraded"]
        assert [e.ran for e in events] == ["offload:model_offload", "offload:sequential"]
        assert all("DEGRADED_MODE=engaged" in e.reason for e in events)
        # Same rung again: deduped.
        await lc.maybe_send_state_delta()
        assert len([m for m in tp.sent if m.WhichOneof("msg") == "fn_degraded"]) == 2

    asyncio.run(_go())


def test_plan_serve_offload_rung_unchanged_by_reactive_path() -> None:
    """The reactive rung never preempts the quant rungs: a plan that fits
    fp8/emergency still plans those; offload stays the LAST rung."""
    from gen_worker.models.hub_policy import TensorhubWorkerCapabilities

    caps = TensorhubWorkerCapabilities(
        cuda_version="12.4", gpu_sm=89, torch_version="2.8", installed_libs=[])
    plan = plan_serve(Resources(vram_gb=12.0), caps, 8.0)
    assert plan.serveable and plan.run_mode == "fp8_storage"  # quant first
    huge = plan_serve(Resources(vram_gb=100.0), caps, 8.0)
    assert huge.serveable and huge.run_mode == RUN_OFFLOAD  # offload is last
