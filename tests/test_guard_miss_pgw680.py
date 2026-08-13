"""pgw#680: guard-miss doctrine — fail-on-recompile at serve time.

The 187s incident class (ie#546 retag cycle): a tenant request whose inputs
missed every cached guard set paid dynamo's INLINE recompile inside the
request, and single-flight stalled everything queued behind it. Paul's
design, verbatim: "instead of compiling [inline], it should throw an error,
which we catch, then run in eager mode + compile [in background] and note
the mismatch."

Doctrine under test:
  1. STANCE — tenant requests on compiled lanes run under
     ``torch._dynamo.config.error_on_recompile`` scoped to the executor's
     ``tenant_serve_window`` (thread-local ConfigModule ContextVar, torch
     2.13). Warm/mint/adopt/boot windows never enter the window, so they
     keep compiling; the stance cannot leak into them by construction.
  2. CATCH — the guard wrappers catch dynamo's ``RecompileError``, serve
     THIS request eager immediately, and record the mismatch loudly: the
     verbatim guard-failure reason, the shape/axis identity, the compiled graph key,
     the request id — as a typed ``guard_miss`` activity event the hub can
     count per (release, SKU, guard-reason).
  3. HEAL — the exact input class is recompiled through the existing
     background shape-warm driver (nice +10, own stream, dedup by
     signature), so the SECOND request of that shape hits compiled. A
     signature that keeps missing after its heals is per-request-volatile
     and gets routed eager permanently instead of thrashing.

The real-torch tapes drive REAL ``torch.compile(backend="eager")`` guards —
real dynamo guard evaluation, real RecompileError, the REAL wrappers and
the REAL background warm thread. The executor tape drives the REAL
``handle_run_job`` tenant path end-to-end with the torch.compile leaf
simulated at the same boundary as tests/test_serve_finalize_pgw672.py.

RED (verified before the fix landed): with the serve window neutralized —
exactly the pre-pgw#680 tree's behavior — the same guard-missing input
recompiles INLINE inside the request, silently
(test_outside_serve_window_recompiles_inline is that behavior, kept as the
permanent contrast tape for the warm/mint windows).
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import msgspec
import pytest
import torch
import torch.nn as nn
from torch._dynamo import exc as dexc

import gen_worker.executor as executor_mod
from gen_worker import Compile, activity as activity_mod
from gen_worker import compiled_graph_key as compiled_graph_key_mod
from gen_worker import compile_cache as cc
from gen_worker import fleet_compiled_graphs, hot_swap
from gen_worker.api.binding import Hub, wire_ref
from gen_worker.executor import Executor
from gen_worker.models import provision
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec

FAMILY = "sdxl"
MODEL_DIGEST = "blake3:" + "c" * 64


# ---------------------------------------------------------------------------
# Real-torch tapes: REAL dynamo guards, REAL RecompileError, REAL wrappers,
# REAL background warm thread. backend="eager" keeps full guard machinery
# without needing an inductor toolchain or a GPU.
# ---------------------------------------------------------------------------


class _Mod(nn.Module):
    """``k`` is an int attribute dynamo SPECIALIZES on: changing it after a
    compile guard-misses with a byte-identical call signature — exactly the
    "variable we're not understanding" class (scalar specialization)."""

    def __init__(self) -> None:
        super().__init__()
        self.k = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.k


def _signal(router: Optional[hot_swap.Router]) -> Dict[str, Any]:
    return {
        "callback": None,
        "lock": threading.Lock(),
        "successful_calls": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "guard_misses": 0,
        "on_guard_miss": None,
        "router": router,
    }


def _wait(predicate: Any, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("background heal never completed")


def test_serve_window_guard_miss_serves_eager_records_and_heals() -> None:
    """THE core tape: a warm-signature request whose guards miss under the
    stance (a) never raises to the caller and never pays an inline compile,
    (b) returns the correct EAGER result immediately, (c) confesses the
    verbatim torch reason through the typed callback, (d) heals in
    background so the SECOND request of the shape runs compiled."""
    mod = _Mod()
    original = mod.forward
    compiled = torch.compile(mod.forward, backend="eager", dynamic=False)
    router = hot_swap.Router()
    signal = _signal(router)
    misses: List[cc.GuardMiss] = []
    signal["on_guard_miss"] = misses.append
    wrapped = cc._guarded(
        original, compiled, "transformer.forward", failure_signal=signal)

    x = torch.ones(4)
    # Boot-window semantics: sequential router, first compile outside the
    # serve window (a first compile is not a recompile — it would not raise
    # even inside it); concurrency enables post-proof, like the executor.
    assert torch.equal(wrapped(x), x * 2)
    warm_sig = ("transformer.forward", hot_swap.signature((x,), {}))
    assert warm_sig in router.warm
    router.enable()

    mod.k = 3  # guard miss with an IDENTICAL call signature
    with cc.tenant_serve_window():
        out = wrapped(x)
    # (a)+(b): served, eager, correct — and the object is NOT degraded.
    assert torch.equal(out, x * 3)
    assert not signal.get("failed", False)
    assert signal["guard_misses"] == 1
    # (c): the confession carries torch's verbatim reason + identity.
    assert len(misses) == 1
    miss = misses[0]
    assert miss.target == "transformer.forward"
    assert "guard failure" in miss.reason
    assert miss.heal == "healing"
    assert miss.misses == 1
    assert cc.guard_miss_reason_class(miss.reason) != "unclassified"
    # The signature was demoted out of warm and routes EAGER while healing.
    assert warm_sig not in router.warm
    verdict, _sig = router.route(
        "transformer.forward", compiled, (x,), {})
    assert verdict == hot_swap.EAGER

    # (d): the background warm thread compiles the exact class; the second
    # request of the shape hits compiled under the stance, no new miss.
    _wait(lambda: warm_sig in router.warm)
    with cc.tenant_serve_window():
        out2 = wrapped(x)
    assert torch.equal(out2, x * 3)
    assert signal["guard_misses"] == 1, "second request must not miss"
    assert not misses[1:]


def test_outside_serve_window_recompiles_inline() -> None:
    """The warm/mint/boot-window contract (and the RED behavior): outside
    the serve window the identical guard-missing input pays the inline
    recompile silently — windows that exist to compile keep compiling."""
    mod = _Mod()
    compiled = torch.compile(mod.forward, backend="eager", dynamic=False)
    router = hot_swap.Router()
    signal = _signal(router)
    misses: List[cc.GuardMiss] = []
    signal["on_guard_miss"] = misses.append
    wrapped = cc._guarded(
        mod.forward, compiled, "transformer.forward", failure_signal=signal)

    x = torch.ones(4)
    wrapped(x)
    router.enable()
    mod.k = 5
    out = wrapped(x)  # NO serve window: dynamo recompiles inline, no raise
    assert torch.equal(out, x * 5)
    assert signal["guard_misses"] == 0
    assert misses == []


def test_repeatedly_volatile_signature_routes_eager_permanently() -> None:
    """A signature that guard-misses past the heal limit is per-request
    volatile (guards diverge on something our signatures do not capture):
    stop compile churn, route it eager permanently, keep serving."""
    mod = _Mod()
    compiled = torch.compile(mod.forward, backend="eager", dynamic=False)
    router = hot_swap.Router()
    signal = _signal(router)
    misses: List[cc.GuardMiss] = []
    signal["on_guard_miss"] = misses.append
    wrapped = cc._guarded(
        mod.forward, compiled, "transformer.forward", failure_signal=signal)

    x = torch.ones(4)
    wrapped(x)
    router.enable()
    sig = ("transformer.forward", hot_swap.signature((x,), {}))
    for step, k in enumerate((3, 4, 5), start=1):
        mod.k = k  # mutate per call: every heal compiles a stale guard set
        with cc.tenant_serve_window():
            out = wrapped(x)
        assert torch.equal(out, x * k)
        # miss recorded each time until the sig goes volatile
        assert signal["guard_misses"] == step
        if step <= hot_swap._GUARD_MISS_HEAL_LIMIT:
            _wait(lambda: sig in router.warm or sig in router.bg_failed)
    assert misses[-1].heal == "volatile"
    assert sig in router.volatile
    # Further requests: routed eager BEFORE the stance — served, no raise,
    # no new miss, no heal enqueued.
    mod.k = 9
    with cc.tenant_serve_window():
        out = wrapped(x)
    assert torch.equal(out, x * 9)
    assert signal["guard_misses"] == 3
    assert sig not in router.healing and sig not in router.pending


def test_regional_guard_miss_serves_eager_and_preserves_blocks() -> None:
    """Regional lanes (in-place block compiles, ie#381): the stance covers
    the block frames, the catch serves THIS request fully eager with dynamo
    disabled for the call, and the compiled block impls survive intact."""

    class _Stack(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blk = _Mod()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.blk(x)

    stack = _Stack()
    original = stack.forward
    stack.blk.compile(backend="eager", dynamic=False)  # in place, like
    # compile_repeated_blocks
    signal = _signal(hot_swap.Router())
    misses: List[cc.GuardMiss] = []
    signal["on_guard_miss"] = misses.append
    wrapped = cc._guarded_regional(
        stack, original, "transformer", failure_signal=signal)

    x = torch.ones(4)
    assert torch.equal(wrapped(x), x * 2)  # boot compile, outside window
    stack.blk.k = 3
    with cc.tenant_serve_window():
        out = wrapped(x)
    assert torch.equal(out, x * 3)
    assert signal["guard_misses"] == 1 and len(misses) == 1
    assert "guard failure" in misses[0].reason
    assert not signal.get("failed", False)
    # Blocks still compiled in place (not cleared like the permanent-degrade
    # path would): a later out-of-window call recompiles/serves normally.
    assert stack.blk._compiled_call_impl is not None
    assert torch.equal(wrapped(x), x * 3)


def test_reason_class_extraction() -> None:
    reason = (
        "Recompiling function forward in /x.py:5\n"
        "    triggered by the following guard failure(s):\n"
        "    - 0/1: tensor 'x' size mismatch at index 0. expected 3, actual 4\n"
        "    - 0/0: tensor 'x' size mismatch at index 0. expected 2, actual 4"
    )
    got = cc.guard_miss_reason_class(reason)
    assert got == "tensor 'x' size mismatch at index 0. expected 3, actual 4"
    assert cc.guard_miss_reason_class("") == "unclassified"
    assert cc.guard_miss_reason_class("plain text") == "plain text"


def test_stance_is_thread_scoped_background_compiles_freely() -> None:
    """The stance-choice invariant this lane's design stands on: the serve
    window arms only the serving thread — a concurrent background thread
    (the warm/mint driver) recompiles freely while the stance is up."""
    mod = _Mod()
    compiled = torch.compile(mod.forward, backend="eager", dynamic=False)
    compiled(torch.ones(4))
    mod.k = 7
    result: Dict[str, Any] = {}

    def bg() -> None:
        try:
            result["out"] = compiled(torch.ones(4))  # recompile, no stance
        except BaseException as exc:  # noqa: BLE001
            result["exc"] = exc

    with cc.tenant_serve_window():
        with cc._fail_on_recompile() as armed:
            assert armed
            t = threading.Thread(target=bg)
            t.start()
            t.join()
    assert "exc" not in result, f"stance leaked into a background thread: {result.get('exc')}"
    assert torch.equal(result["out"], torch.ones(4) * 7)


# ---------------------------------------------------------------------------
# Executor tape: the REAL handle_run_job tenant path end-to-end, torch leaf
# simulated at the pgw#672 boundary. Boot mints through warm windows (no
# stance, inline compiles), the tenant dispatch runs under the serve window,
# and the guard_miss activity event reaches the (bound) wire sink.
# ---------------------------------------------------------------------------


class _In(msgspec.Struct):
    prompt: str = ""


class _Out(msgspec.Struct):
    ok: bool = True


class _Denoiser:
    def forward(self, *args: Any, **kwargs: Any) -> None:
        return None


class _Pipe:
    def __init__(self) -> None:
        self.transformer = _Denoiser()

    @classmethod
    def from_pretrained(cls, path: Any, **kwargs: Any) -> "_Pipe":
        return cls()  # pragma: no cover - loader is patched


@pytest.fixture(autouse=True)
def _clean_registries() -> Any:
    # _Mod.forward.__code__ is class-shared: dynamo entries from one test's
    # compiles would serve (or guard-miss) another's. Fresh guard state per
    # test.
    torch._dynamo.reset()
    with cc._PROVEN_COMPILED_GRAPHS_LOCK:
        cc._PROVEN_COMPILED_GRAPHS.clear()
        cc._QUARANTINED_COMPILED_GRAPHS.clear()
    with fleet_compiled_graphs._PENDING_LOCK:
        fleet_compiled_graphs._PENDING.clear()
    for pipe in list(cc._armed_pipelines()):
        cc._armed_pipelines().discard(pipe)
    sink_before = activity_mod._sink
    yield
    with activity_mod._lock:
        activity_mod._sink = sink_before
    with cc._PROVEN_COMPILED_GRAPHS_LOCK:
        cc._PROVEN_COMPILED_GRAPHS.clear()
        cc._QUARANTINED_COMPILED_GRAPHS.clear()
    with fleet_compiled_graphs._PENDING_LOCK:
        fleet_compiled_graphs._PENDING.clear()
    for pipe in list(cc._armed_pipelines()):
        cc._armed_pipelines().discard(pipe)


class _Sim:
    """Dynamo/inductor semantics at the torch boundary, with guard EPOCHS.

    Entries are recorded at ``guard_epoch``; a call whose entry matches the
    current epoch is a HIT. An entry from an older epoch is a guard miss —
    dynamo must RECOMPILE: inside the tenant serve window the stance raises
    torch's real RecompileError (what error_on_recompile does); outside it
    the recompile happens inline (warm/mint windows, the warm thread's
    heal). ``window_seen`` records the serve-window flag at every compile
    so the mint-window tape can assert the stance never leaked in."""

    def __init__(self) -> None:
        self.guard_epoch = 0
        self.counters: Dict[str, int] = {
            "fxgraph_cache_hit": 0, "fxgraph_cache_miss": 0,
            "aot_cache_hit": 0,
        }
        self.compiles: List[Tuple[str, int]] = []
        self.window_seen: List[bool] = []
        self.served_compiled: List[str] = []

    def _entry(self, graph: str) -> Path:
        import os

        fx_dir = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]) / "fxgraph"
        fx_dir.mkdir(parents=True, exist_ok=True)
        return fx_dir / f"{graph}.bin"

    def compiled_call(self, original: Any, label: str,
                      args: tuple, kwargs: dict) -> Any:
        graph = "g-" + hashlib.blake2s(
            repr((label, args, sorted(kwargs))).encode()).hexdigest()[:16]
        entry = self._entry(graph)
        epoch: Optional[int] = None
        if entry.exists():
            epoch = int(entry.read_text() or 0)
        if epoch == self.guard_epoch:
            self.counters["fxgraph_cache_hit"] += 1
            self.served_compiled.append(graph)
            return original(*args, **kwargs)
        if epoch is not None and cc._SERVE_WINDOW.get():
            raise dexc.RecompileError(
                f"Recompiling function forward in sim.py:1\n"
                f"    triggered by the following guard failure(s):\n"
                f"    - 0/0: tensor 'sample' stride mismatch at index 1. "
                f"expected 4096, actual 1 (epoch {epoch} != "
                f"{self.guard_epoch})")
        self.window_seen.append(cc._SERVE_WINDOW.get())
        entry.write_text(str(self.guard_epoch))
        self.counters["fxgraph_cache_miss"] += 1
        self.compiles.append((graph, self.guard_epoch))
        return original(*args, **kwargs)


def _sim_apply_factory(sim: _Sim):
    def _sim_apply(pipeline: Any, cfg: Any, *, cache_ready: bool,
                   guard: bool = True, allow_cold: bool = False) -> bool:
        if getattr(pipeline, cc._MARKER_ATTR, None) is not None:
            return True
        original = pipeline.transformer.forward
        signal = _signal(hot_swap.Router(fail_closed=True))

        def compiled(*args: Any, **kwargs: Any) -> Any:
            return sim.compiled_call(original, "transformer", args, kwargs)

        setattr(pipeline, cc._MARKER_ATTR, {
            "targets": ["transformer"],
            "shapes": [tuple(s) for s in cfg.shapes],
            "cache": bool(cache_ready),
            "originals": [(pipeline.transformer, "forward", original)],
            "regional_mods": [],
            "failure_signal": signal,
        })
        pipeline.transformer.forward = cc._guarded(
            original, compiled, "transformer",
            fail_closed=True, failure_signal=signal)
        cc._armed_pipelines().add(pipeline)
        return True

    return _sim_apply


def _fake_arm_identity(family: str, weight_lane: str = "",
                       lora_bucket: int = 0, cfg: Any = None) -> Any:
    digest = hashlib.blake2s(
        f"{family}|{weight_lane}|{lora_bucket}".encode()
    ).hexdigest()[:56]
    return SimpleNamespace(token="arm1-" + digest, facts_dict=lambda: {})


SHAPE = (768, 768)


def _endpoint_cls() -> type:
    class _Ep:
        def setup(self, pipeline: _Pipe) -> None:
            self.pipeline = pipeline

        def warmup(self) -> None:
            self.pipeline.transformer.forward("warm", shape=SHAPE)

        def run(self, ctx: Any, payload: _In) -> _Out:
            # The tenant forward uses the SAME warm signature the boot
            # proved — the guard miss is on an axis signatures don't carry
            # (the sim's epoch: stride/autotune divergence in real life).
            self.pipeline.transformer.forward("warm", shape=SHAPE)
            return _Out()

    return _Ep


def _spec(name: str, cls: type) -> EndpointSpec:
    return EndpointSpec(
        name=name, method=cls.run, kind="inference",
        payload_type=_In, output_mode="single", cls=cls, attr_name="run",
        models={"pipeline": Hub("acme/sdxl-base")},
        compile=Compile(shapes=(SHAPE,), family=FAMILY, text_len=0),
    )


class _Rig:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                 specs: List[EndpointSpec]) -> None:
        # pgw#1010: no export declaration is registered, so the miss takes the
        # JIT INTAKE arm — the in-process compile the guard-miss doctrine
        # (mint window OFF while compiling, serve window ON while serving) is
        # about. It mints and publishes nothing, which is orthogonal to this
        # file's subject and asserted below where it shows.
        #
        # The sim keys its entries under TORCHINDUCTOR_CACHE_DIR. The retired
        # capture used to point that at a mint dir; an intake arm moves no env
        # (that was gw#608's whole root cause), so the rig states the cache
        # root itself — production leaves it wherever the image/worker set it.
        monkeypatch.setenv(
            "TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
        self.sim = _Sim()
        self.sent: List[pb.WorkerMessage] = []
        self.pipes: Dict[str, _Pipe] = {}
        model_dir = tmp_path / "model"
        model_dir.mkdir(exist_ok=True)

        async def _send(msg: pb.WorkerMessage) -> None:
            self.sent.append(msg)

        self.ex = Executor(specs, _send)
        self.ex.store._cache_dir = tmp_path / "cas"

        async def _download(ref: str, **kwargs: Any) -> Path:
            return model_dir

        def _load_slot(*args: Any, **kwargs: Any) -> Any:
            pipe = _Pipe()
            # pgw#1010: a PLAIN lane. A mandatory (w8a8/w4a4) lane serves only
            # from a compiled graph — the dispatch fence pins every request to an active
            # compile incarnation — so a family with no export declaration
            # fails closed there instead of arming JIT intake. The doctrine
            # this rig exercises is the intake compile itself, which is a plain
            # lane's shape.
            self.pipes[f"pipe-{len(self.pipes)}"] = pipe
            return provision.SlotLoad(obj=pipe, is_pipeline=True)

        def _mandatory_miss(*a: Any, **k: Any) -> bool:
            raise cc.CompiledExecutionLaneUnavailableError("no delivered compiled_graph")

        monkeypatch.setattr(executor_mod, "ensure_local", _download)
        monkeypatch.setattr(provision, "load_slot", _load_slot)
        monkeypatch.setattr(provision, "enable_compiled", _mandatory_miss)
        monkeypatch.setattr(fleet_compiled_graphs, "_cuda_ready", lambda: True)
        monkeypatch.setattr(cc, "toolchain_present", lambda: True)
        monkeypatch.setattr(cc, "apply", _sim_apply_factory(self.sim))
        monkeypatch.setattr(
            cc, "inductor_counters", lambda: dict(self.sim.counters))
        monkeypatch.setattr(fleet_compiled_graphs, "arm_identity", _fake_arm_identity)
        monkeypatch.setattr(cc, "reset_target_code", lambda pipeline: 0)
        # The sim never enters dynamo, so dynamo's cumulative compile clock
        # never moves. Stub it: what is under test is that the executor READS
        # it across the warm window and puts the number on the wire (pgw#1010 /
        # th#1322), not what torch would have measured.
        self.compile_wall = [0.0, 12.5]
        monkeypatch.setattr(
            cc, "compile_wall_seconds",
            lambda: self.compile_wall.pop(0) if self.compile_wall else 12.5)
        # pgw#681 coexistence: when the guard-closure mint gate is present,
        # neutralize it — it audits REAL dynamo graphs, which this rig's
        # simulated torch boundary never creates. Orthogonal to the
        # guard-miss doctrine under test here.
        try:
            from gen_worker import guard_closure as gc_mod

        except ImportError:
            pass
        monkeypatch.setattr(
            self.ex, "_enable_compiled",
            lambda p, cfg, artifact, delivered=None, arm=None, boot_local_key="":
                fleet_compiled_graphs.enable_compiled(
                    p, cfg, self.ex.store._cache_dir, artifact,
                    publisher=None))

    def boot(self, spec: EndpointSpec) -> None:
        model_ref = wire_ref(spec.models["pipeline"])
        asyncio.run(self.ex.ensure_setup(
            spec, {model_ref: pb.Snapshot(digest=MODEL_DIGEST)}))


async def _dispatch(rig: _Rig, run: pb.RunJob) -> pb.JobResult:
    loop = asyncio.get_running_loop()
    activity_mod.bind_sink(rig.ex._send, loop)
    await rig.ex.handle_run_job(run)
    job = rig.ex.jobs[(run.request_id, run.attempt)]
    assert job.task is not None
    await job.task
    results = [m.job_result for m in rig.sent
               if m.WhichOneof("msg") == "job_result"
               and m.job_result.request_id == run.request_id]
    assert results, f"no job_result for {run.request_id}"
    return results[-1]


def _run_job(rig: _Rig, spec: EndpointSpec, request_id: str) -> pb.RunJob:
    (target,) = rig.ex.compile_targets()
    model_ref = wire_ref(spec.models["pipeline"])
    job = pb.RunJob(
        request_id=request_id, attempt=1, function_name=spec.name,
        input_payload=msgspec.msgpack.encode(_In(prompt="a cat")),
        models=[pb.ModelBinding(slot="pipeline", ref=model_ref)],
        snapshots={model_ref: pb.Snapshot(digest=MODEL_DIGEST)},
    )
    if target.active_compile_ref:
        # The th#910 dispatch fence pins a request to the COMPILED GRAPH the worker
        # advertised. pgw#1010: an intake target advertises none — there is no
        # artifact to fence on — so a hub that sent one would be naming a compiled graph
        # this pod never claimed, and the worker's own
        # `required_compile_invalid` refusal is the correct answer to that.
        job.required_compile.CopyFrom(pb.RequiredCompileExecution(
            target_incarnation_id=target.incarnation_id,
            compiled_graph_ref=target.active_compile_ref,
            compiled_graph_snapshot_digest=target.active_compile_snapshot_digest,
            contract_digest=target.contract_digest,
        ))
    return job


def test_tenant_guard_miss_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full doctrine through the REAL tenant path:

    boot (mint window) compiles inline with the serve window OFF ->
    request 1 guard-misses under the stance -> completes OK served eager,
    guard_miss activity event on the wire (reason class in phase; compiled graph,
    request id, verbatim reason in detail), heal scheduled -> request 2 of
    the same shape serves COMPILED. The compiled identity is never revoked
    and nothing is disabled."""
    cls = _endpoint_cls()
    spec = _spec("generate", cls)
    rig = _Rig(tmp_path, monkeypatch, [spec])

    rig.boot(spec)
    # pgw#1010: JIT intake serves COMPILED CODE and names no artifact, so the
    # platform tier — which means "serving from a compiled graph" — is eager, and the
    # target advertises active-less. The per-request axis is where the
    # compiled-ness of an intake pod is stated (`serving_mode=jit_compiled_graph`).
    assert rig.ex.serving_tiers()["generate"] == "eager"
    # Mint-window contract: every boot compile ran with the window OFF.
    assert rig.sim.compiles, "boot must have compiled"
    assert rig.sim.window_seen and not any(rig.sim.window_seen), (
        "the serve window leaked into a mint/warm window")
    (target,) = rig.ex.compile_targets()
    compiled_graph_ref = target.active_compile_ref
    assert compiled_graph_ref == "", "an intake arm has no compiled_graph to advertise"
    # pgw#1010 / th#1322: and the compile it just paid for is a NUMBER on the
    # wire. The emitter used to be the mint parent's, and the mint no longer
    # runs JIT — so this boot is the only place an AOT-vs-JIT cost comparison
    # can get its JIT arm from.
    jit = [m.activity_update for m in rig.sent
           if m.WhichOneof("msg") == "activity_update"
           and m.activity_update.kind == activity_mod.KIND_JIT_COMPILE]
    assert jit and any(u.duration_ms > 0 for u in jit), (
        "an intake boot compiled and reported no duration")
    assert any("route=intake" in u.detail for u in jit)

    # The variable-we-don't-understand flips between mint and serve
    # (stride/autotune divergence between minting and serving workers).
    rig.sim.guard_epoch = 1
    boot_compiles = len(rig.sim.compiles)

    run1 = _run_job(rig, spec, "r-guard-miss")
    res1 = asyncio.run(_dispatch(rig, run1))
    assert res1.status == pb.JOB_STATUS_OK, res1.safe_message
    # Served EAGER: no inline compile happened inside the request window.
    (pipe,) = rig.pipes.values()
    assert cc._proof_count(pipe, "guard_misses") == 1
    # The heal ran on the background warm thread (window OFF there).
    _wait(lambda: len(rig.sim.compiles) > boot_compiles)
    assert not any(rig.sim.window_seen), (
        "a compile executed inside the tenant serve window")

    # The typed confession reached the wire, countable per reason class.
    events = [m.activity_update for m in rig.sent
              if m.WhichOneof("msg") == "activity_update"
              and m.activity_update.kind == activity_mod.KIND_GUARD_MISS]
    assert len(events) == 1
    ev = events[0]
    assert ev.state == pb.ActivityState.ACTIVITY_STATE_COMPLETED
    assert "stride mismatch" in ev.phase
    assert compiled_graph_ref in ev.detail
    assert "request=r-guard-miss" in ev.detail
    assert "stride mismatch at index 1" in ev.detail
    assert "heal=healing" in ev.detail

    # Nothing was revoked, degraded, or disabled — the lane is healthy.
    assert target.active_compile_ref == compiled_graph_ref
    assert rig.ex.serving_tiers()["generate"] == "eager"
    assert rig.ex.unavailable == {}

    # SECOND request of the shape: compiled, no new miss, no new event.
    served_before = len(rig.sim.served_compiled)
    res2 = asyncio.run(_dispatch(rig, _run_job(rig, spec, "r-healed")))
    assert res2.status == pb.JOB_STATUS_OK, res2.safe_message
    assert cc._proof_count(pipe, "guard_misses") == 1
    assert len(rig.sim.served_compiled) > served_before, (
        "the healed entry must serve the second request compiled")
    events_after = [m.activity_update for m in rig.sent
                    if m.WhichOneof("msg") == "activity_update"
                    and m.activity_update.kind == activity_mod.KIND_GUARD_MISS]
    assert len(events_after) == 1


# ---------------------------------------------------------------------------
# The heal must compile under the REQUESTING thread's intra-op count
# ---------------------------------------------------------------------------
#
# Dynamo's GLOBAL_STATE guard snapshots torch.get_num_threads() on the thread
# that COMPILES, and the OpenMP intra-op ICV is per-thread and sticky once
# initialized. hot_swap runs ONE process-global warm thread, so if it was
# created before something narrowed the serving thread's count, every entry it
# compiled carried a guard the serving thread could never satisfy
# ("GLOBAL_STATE changed: num_threads") — the heal marked the signature warm
# while its entry was unservable, the next request missed again, and
# _GUARD_MISS_HEAL_LIMIT made the signature permanently volatile.
#
# Latent rather than live on today's production ordering (both
# cpu_budget.impose_intra_op_threads call sites run at boot, before the warm
# thread exists) — but the warm job already carries the requesting thread's
# grad-mode and autocast, and the intra-op count is the same class of state.


def test_warm_job_carries_the_requesting_threads_intra_op_count() -> None:
    """Both job-creating paths record the count of the thread that ASKED."""
    router = hot_swap.Router()
    x = torch.ones(4)
    captured: List[Any] = []

    def fake_submit(job: Any) -> bool:
        captured.append(job)
        return True

    real_submit = hot_swap._submit
    hot_swap._submit = fake_submit  # type: ignore[assignment]
    try:
        router.record_guard_miss(
            ("transformer.forward", hot_swap.signature((x,), {})),
            "transformer.forward", lambda *a, **k: x, (x,), {})
    finally:
        hot_swap._submit = real_submit  # type: ignore[assignment]

    assert captured, "the heal must submit a warm job"
    assert captured[0].num_threads == torch.get_num_threads()


def test_the_warm_compile_imposes_that_count_before_compiling() -> None:
    """The whole point: the compile runs at the requesting thread's value, so
    the entry's GLOBAL_STATE guard matches the thread that will serve it."""
    router = hot_swap.Router()
    seen: List[int] = []

    def recording_compiled(*_a: Any, **_k: Any) -> Any:
        seen.append(int(torch.get_num_threads()))
        return None

    here = int(torch.get_num_threads())
    diverged = max(1, here - 1)  # what a stale warm thread would be sitting at
    job = hot_swap._WarmJob(
        router=router, label="transformer.forward", sig=("s",),
        compiled=recording_compiled, args=(torch.ones(4),), kwargs={},
        device=None, grad_mode="grad", autocast_dtype=None,
        num_threads=diverged, turn=None,
    )
    restore = int(torch.get_num_threads())
    try:
        hot_swap._run_warm_compile(job)
    finally:
        torch.set_num_threads(restore)

    assert seen == [diverged], (
        "the warm compile must impose the job's intra-op count before "
        f"compiling; compiled at {seen} instead of [{diverged}]")
