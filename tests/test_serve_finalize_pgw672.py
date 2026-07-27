"""pgw#672: the minted/attached ck2 compile object must serve its own warmup.

Live signature this closes (ie#546 burst rerun #2, gen-worker 0.64.0, L4):
a worker MINTS its cell (publish-intent 200, armed, obligation discharged),
then fails its own finalize —

    CompiledLaneUnavailableError: 1 attached compile object(s) did not serve
    their own warmup graph (warmups=18, calls=18, cache_hits=0,
    cache_misses=0, compile_seconds=0.0)

Zero hits AND zero misses with calls>0 means the FX cache was never even
consulted: dynamo's in-memory code cache (keyed on the class-shared
``__code__``; torch 2.13 inlined-module guards match any same-class
instance) served the warmup that a LATER same-family arm in the same warm
process was supposed to compile/look up. The pending capture stays empty,
finalize disproves it, the mandatory lane raised, both functions disabled,
pod retired, replacement re-mints the same key — 5 cycles, 4 dead workers.

These tests run the REAL executor ensure_setup codepath and the REAL
fleet_cells miss policy (arm_self_mint -> begin_fleet_mint env transaction
-> executor proof -> finalize_self_mint -> finish_fleet_mint pack), faking
only the torch boundary: ``compile_cache.apply``'s torch.compile leaf is a
simulator with dynamo-in-memory-code semantics, ``inductor_counters`` reads
the simulator, and ``torch._dynamo.reset_code`` drops simulator entries.

Fix under test (pgw#672):
  (a) honesty — a scoped per-code dynamo reset before every proof window
      forces the warmup through the real lookup path (mint: real capture;
      re-arm of an in-process finalized cell: real FX HIT);
  (b) finalize succeeds for the second same-family arm (its own graphs);
  (c) posture — when a compiled lane genuinely cannot serve, the worker
      DEGRADES to explicit eager (tier flips, loud) instead of
      quarantine -> disable -> die (also pgw#673's sm120 CantSplit shape).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import msgspec
import pytest

import gen_worker.executor as executor_mod
from gen_worker import Compile, cell_key as cell_key_mod
from gen_worker import compile_cache as cc
from gen_worker import guard_closure
from gen_worker import fleet_cells, hot_swap
from gen_worker.api.binding import Hub, wire_ref
from gen_worker.executor import Executor
from gen_worker.models import provision
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec

FAMILY = "sdxl"
MODEL_DIGEST = "blake3:" + "c" * 64


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


class _Sim:
    """Dynamo/inductor semantics at the torch boundary.

    ``inmem`` is dynamo's in-memory code cache (keyed on the class-shared
    ``__code__`` — every same-class pipeline instance shares it, exactly the
    torch 2.13 mechanism). A call whose code is resident serves WITHOUT any
    counter movement or capture write. Otherwise the FX cache dir named by
    ``TORCHINDUCTOR_CACHE_DIR`` is consulted: an existing content-addressed
    entry is a HIT, a missing one is a compile (entry written + MISS).
    """

    def __init__(self) -> None:
        self.inmem: set[Any] = set()
        self.counters: Dict[str, int] = {
            "fxgraph_cache_hit": 0, "fxgraph_cache_miss": 0,
            "aot_cache_hit": 0,
        }
        self.compiles: List[str] = []
        self.raise_on_compile: Optional[Exception] = None

    def compiled_call(self, code: Any, original: Any, label: str,
                      args: tuple, kwargs: dict) -> Any:
        if code in self.inmem:
            return original(*args, **kwargs)  # in-memory serve: 0/0 counters
        if self.raise_on_compile is not None:
            raise self.raise_on_compile
        graph = "g-" + hashlib.blake2s(
            repr((label, args, sorted(kwargs))).encode()).hexdigest()[:16]
        fx_dir = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]) / "fxgraph"
        fx_dir.mkdir(parents=True, exist_ok=True)
        entry = fx_dir / f"{graph}.bin"
        if entry.exists():
            self.counters["fxgraph_cache_hit"] += 1
        else:
            entry.write_bytes(b"graph")
            self.counters["fxgraph_cache_miss"] += 1
            self.compiles.append(graph)
        self.inmem.add(code)
        return original(*args, **kwargs)


def _sim_apply_factory(sim: _Sim):
    def _sim_apply(pipeline: Any, cfg: Any, *, cache_ready: bool,
                   guard: bool = True, allow_cold: bool = False) -> bool:
        if getattr(pipeline, cc._MARKER_ATTR, None) is not None:
            return True
        original = pipeline.transformer.forward
        code = original.__func__.__code__
        signal: Dict[str, Any] = {
            "callback": None,
            "lock": threading.Lock(),
            "successful_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "router": hot_swap.Router(fail_closed=True),
        }

        def compiled(*args: Any, **kwargs: Any) -> Any:
            return sim.compiled_call(code, original, "transformer", args, kwargs)

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


def _fake_key_compute(family: str, weight_lane: str = "", lora_bucket: int = 0,
                      *, contract: str, regional: bool = False,
                      image_digest: Optional[str] = None) -> Any:
    digest = hashlib.blake2s(
        f"{family}|{weight_lane}|{lora_bucket}|{contract}|{regional}".encode()
    ).hexdigest()[:56]
    return SimpleNamespace(digest="ck5-" + digest, axes={})


def _endpoint_cls(shapes: tuple) -> type:
    class _Ep:
        pipes: List[_Pipe] = []
        warmups = 0

        def setup(self, pipeline: _Pipe) -> None:
            self.pipeline = pipeline

        def warmup(self) -> None:
            type(self).warmups += 1
            # The endpoint's own serving call — through the REAL guard
            # wrapper installed by (sim) apply. One call per declared shape.
            for shape in shapes:
                self.pipeline.transformer.forward("warm", shape=tuple(shape))

        def run(self, ctx: Any, payload: _In) -> _Out:
            return _Out()

    return _Ep


def _spec(name: str, cls: type, shapes: tuple) -> EndpointSpec:
    return EndpointSpec(
        name=name, method=cls.run, kind="inference",
        payload_type=_In, output_mode="single", cls=cls, attr_name="run",
        models={"pipeline": Hub("acme/sdxl-base", flavor="fp8-w8a8")},
        compile=Compile(shapes=shapes, family=FAMILY, text_len=0),
    )


class _Rig:
    """The REAL executor + REAL fleet_cells miss policy, torch leaves faked."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                 specs: List[EndpointSpec]) -> None:
        self.sim = _Sim()
        self.pipes: Dict[str, _Pipe] = {}
        model_dir = tmp_path / "model"
        model_dir.mkdir(exist_ok=True)

        async def _send(_msg: pb.WorkerMessage) -> None:
            return None

        self.ex = Executor(specs, _send)
        self.ex.store._cache_dir = tmp_path / "cas"

        async def _download(ref: str, **kwargs: Any) -> Path:
            return model_dir

        def _load_slot(*args: Any, **kwargs: Any) -> Any:
            pipe = _Pipe()
            setattr(pipe, "_cozy_weight_lane", "w8a8")
            self.pipes[f"pipe-{len(self.pipes)}"] = pipe
            return provision.SlotLoad(obj=pipe, is_pipeline=True)

        def _mandatory_miss(*a: Any, **k: Any) -> bool:
            raise cc.CompiledLaneUnavailableError("no delivered cell")

        monkeypatch.setattr(executor_mod, "ensure_local", _download)
        monkeypatch.setattr(provision, "load_slot", _load_slot)
        monkeypatch.setattr(provision, "enable_compiled", _mandatory_miss)
        monkeypatch.setattr(fleet_cells, "_cuda_ready", lambda: True)
        monkeypatch.setattr(cc, "toolchain_present", lambda: True)
        monkeypatch.setattr(cc, "apply", _sim_apply_factory(self.sim))
        # pgw#681 gate at its torch boundary, simmed like apply's compile
        # leaf: _Sim never touches dynamo, so extraction would honestly
        # report closure unprovable and refuse every finalize.
        monkeypatch.setattr(
            guard_closure, "assert_closure",
            lambda pipe, cfg, label="": {
                "v": 1, "graphs": [{"target": "transformer", "code": "sim",
                                    "entry": 0, "guards": []}],
                "verdicts": {}, "leaks": []})
        monkeypatch.setattr(
            cc, "inductor_counters", lambda: dict(self.sim.counters))
        monkeypatch.setattr(cell_key_mod, "compute", _fake_key_compute)
        # torch boundary of the fix: the scoped reset drops the simulator's
        # in-memory code entries, exactly like torch._dynamo.reset_code.
        import torch._dynamo

        monkeypatch.setattr(
            torch._dynamo, "reset_code",
            lambda code: self.sim.inmem.discard(code))
        for spec in specs:
            monkeypatch.setattr(
                self.ex, "_enable_compiled",
                lambda p, cfg, artifact: fleet_cells.enable_compiled(
                    p, cfg, self.ex.store._cache_dir, artifact,
                    publisher=None))
            break

    def boot(self, spec: EndpointSpec) -> None:
        model_ref = wire_ref(spec.models["pipeline"])
        asyncio.run(self.ex.ensure_setup(
            spec, {model_ref: pb.Snapshot(digest=MODEL_DIGEST)}))


def test_second_same_family_arm_mints_its_own_graphs_and_finalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE L4 LOOP (red-verified against the pre-fix tree): generate mints
    fine; generate-turbo (same denoiser class, different cell key) then has
    its warmup served by dynamo's in-memory code — pre-fix its capture
    stays empty and finalize kills both functions with
    ``cache_hits=0, cache_misses=0``. Post-fix the proof-window reset makes
    the second mint REAL: its own graphs are captured, finalize succeeds,
    both functions serve compiled."""
    cls_a = _endpoint_cls(((768, 768),))
    cls_b = _endpoint_cls(((1024, 1024),))
    gen = _spec("generate", cls_a, ((768, 768),))
    turbo = _spec("generate-turbo", cls_b, ((1024, 1024),))
    rig = _Rig(tmp_path, monkeypatch, [gen, turbo])

    rig.boot(gen)
    assert rig.ex.serving_tiers()["generate"] == "compiled"
    compiles_after_a = len(rig.sim.compiles)
    assert compiles_after_a >= 1
    # The first mint's compiled code is resident in (sim) dynamo — the
    # exact precondition that starved the second capture live.
    assert rig.sim.inmem

    rig.boot(turbo)

    # The trace's failure mode is gone: turbo compiled its OWN graphs into
    # its OWN capture (real misses, not counter-silent in-memory service),
    # finalized, and serves compiled. Nothing was disabled.
    assert rig.ex.unavailable == {}
    tiers = rig.ex.serving_tiers()
    assert tiers["generate"] == "compiled"
    assert tiers["generate-turbo"] == "compiled"
    assert len(rig.sim.compiles) > compiles_after_a
    refs = {
        t.active_compile_ref for t in rig.ex.compile_targets()
    }
    assert len(refs) == 2, "two distinct cells, each proven by its own boot"
    for ref in refs:
        assert ref.startswith(f"root/family-{FAMILY}#ck5-")
        assert cc.cell_proven_in_process(ref)
    with fleet_cells._PENDING_LOCK:
        assert fleet_cells._PENDING == {}


def test_same_key_rearm_reuses_finalized_cell_with_a_real_fx_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requirement (a): the minted object SERVES — a later same-key arm in
    the same process re-arms from the folded live cache and its proof
    warmup records a REAL FX cache hit on the exact warm graph (the cache
    is consulted and hits; pre-fix it re-minted an empty capture and
    died)."""
    cls_a = _endpoint_cls(((768, 768),))
    cls_c = _endpoint_cls(((768, 768),))  # same shapes => same cell key
    gen = _spec("generate", cls_a, ((768, 768),))
    twin = _spec("generate-twin", cls_c, ((768, 768),))
    rig = _Rig(tmp_path, monkeypatch, [gen, twin])

    rig.boot(gen)
    (target_a,) = rig.ex.compile_targets()
    ref_a = target_a.active_compile_ref
    compiles_after_a = len(rig.sim.compiles)

    rig.boot(twin)

    assert rig.ex.unavailable == {}
    assert rig.ex.serving_tiers()["generate-twin"] == "compiled"
    # No second mint: the twin re-armed the in-process finalized cell and
    # PROVED it with a genuine FX-cache hit (counters moved, hit>0).
    assert len(rig.sim.compiles) == compiles_after_a
    twin_pipe = [
        p for p in rig.pipes.values()
        if cc.cache_hit_count(p) > 0
    ]
    assert twin_pipe, "the re-armed object must record a real cache hit"
    twin_target = [
        t for t in rig.ex.compile_targets()
        if "generate-twin" in t.function_names
    ]
    assert twin_target and twin_target[0].active_compile_ref == ref_a
    with fleet_cells._PENDING_LOCK:
        assert fleet_cells._PENDING == {}


def test_genuinely_unservable_lane_degrades_to_eager_not_death(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Requirement (c): when the compiled lane truly cannot prove itself
    (here: the reset is ineffective and stale in-memory code keeps serving
    — the th#1166 false-negative class), the worker DEGRADES: setup
    completes, the function stays dispatchable at explicit eager tier, the
    identity is quarantined, and a follow-up setup does NOT re-mint the
    same key (no churn loop)."""
    cls_a = _endpoint_cls(((768, 768),))
    gen = _spec("generate", cls_a, ((768, 768),))
    rig = _Rig(tmp_path, monkeypatch, [gen])
    # Break the fix's torch leaf: reset does nothing, and the denoiser code
    # is already "resident" — every warmup call serves counter-silently.
    import torch._dynamo

    monkeypatch.setattr(torch._dynamo, "reset_code", lambda code: None)
    rig.sim.inmem.add(_Denoiser.forward.__code__)

    with caplog.at_level(logging.ERROR):
        rig.boot(gen)

    # The 0.64.0 outcome was CompiledLaneUnavailableError -> generate
    # disabled -> pod retired. Now: alive, eager, loud.
    assert "generate" not in rig.ex.unavailable
    assert rig.ex.serving_tiers()["generate"] == "eager"
    assert rig.ex.compile_targets() == []
    degrade = [
        r.getMessage() for r in caplog.records
        if "did not serve their own warmup graph" in r.getMessage()
    ]
    assert degrade and "cache_hits=0, cache_misses=0" in degrade[0]
    assert "DEGRADED" in degrade[0]
    # The failed identity is quarantined: a fresh setup of the same spec
    # declines to re-mint the key (loop broken), still serves eager.
    rec = rig.ex._classes[gen.instance_key]
    rec.stale = True
    pending_before = len(rig.sim.compiles)
    rig.boot(gen)
    assert "generate" not in rig.ex.unavailable
    assert rig.ex.serving_tiers()["generate"] == "eager"
    assert len(rig.sim.compiles) == pending_before, "no re-mint of a quarantined key"
    with fleet_cells._PENDING_LOCK:
        assert fleet_cells._PENDING == {}


def test_compile_failure_on_mandatory_lane_degrades_per_sku(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """pgw#673 posture half: an InductorError-class failure compiling the
    W8A8 target (sm120 CantSplit) degrades THAT worker to eager serving —
    warmup completes, function stays up — instead of disabling every
    declared function and retiring the pod."""
    cls_a = _endpoint_cls(((768, 768),))
    gen = _spec("generate", cls_a, ((768, 768),))
    rig = _Rig(tmp_path, monkeypatch, [gen])
    rig.sim.raise_on_compile = RuntimeError(
        "InductorError: CantSplit: 640*(((s69 - 1)//2)) ... not divisible")

    with caplog.at_level(logging.ERROR):
        rig.boot(gen)

    assert "generate" not in rig.ex.unavailable
    assert rig.ex.serving_tiers()["generate"] == "eager"
    assert rig.ex.compile_targets() == []
    assert any(
        "CantSplit" in r.getMessage() and "eager" in r.getMessage()
        for r in caplog.records
    )
