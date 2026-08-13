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
fleet_cells miss policy (miss -> `compile_cache.arm_jit_intake` -> executor
proof), faking only the torch boundary: ``compile_cache.apply``'s torch.compile leaf is a
simulator with dynamo-in-memory-code semantics, ``inductor_counters`` reads
the simulator, and ``torch._dynamo.reset_code`` drops simulator entries.

Fix under test (pgw#672):
  (a) honesty — a scoped per-code dynamo reset before every proof window
      forces the warmup through the real lookup path (mint: real capture;
      re-arm of an in-process finalized cell: real FX HIT);
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
from gen_worker import Compile
from gen_worker import compile_cache as cc
from gen_worker import fleet_cells, hot_swap
from gen_worker.api.binding import Hub, wire_ref
from gen_worker.executor import Executor
from gen_worker.models import provision
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec
from gen_worker.models import store as store_mod

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


def _fake_arm_identity(family: str, weight_lane: str = "",
                       lora_bucket: int = 0, cfg: Any = None) -> Any:
    digest = hashlib.blake2s(
        f"{family}|{weight_lane}|{lora_bucket}".encode()
    ).hexdigest()[:56]
    return SimpleNamespace(token="arm1-" + digest, facts_dict=lambda: {})


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
        models={"pipeline": Hub("acme/sdxl-base")},
        compile=Compile(shapes=shapes, family=FAMILY, text_len=0),
    )


class _Rig:
    """The REAL executor + REAL fleet_cells miss policy, torch leaves faked."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                 specs: List[EndpointSpec]) -> None:
        # pgw#1010: no export declaration is registered, so this rig's miss
        # takes the JIT INTAKE arm — the in-process compile whose honesty
        # (requirement (a)) and degrade posture (requirement (c)) are what
        # survives here. It mints nothing, which is the point.
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
            # pgw#1010: a PLAIN lane. A mandatory (w8a8/w4a4) lane serves only
            # from a cell — the dispatch fence pins every request to an active
            # compile incarnation — so a family with no export declaration
            # fails closed there instead of arming JIT intake. The doctrine
            # this rig exercises is the intake compile itself, which is a plain
            # lane's shape.
            self.pipes[f"pipe-{len(self.pipes)}"] = pipe
            return provision.SlotLoad(obj=pipe, is_pipeline=True)

        def _mandatory_miss(*a: Any, **k: Any) -> bool:
            raise cc.CompiledExecutionLaneUnavailableError("no delivered cell")

        monkeypatch.setattr(store_mod, "ensure_local", _download)
        monkeypatch.setattr(provision, "load_slot", _load_slot)
        monkeypatch.setattr(provision, "enable_compiled", _mandatory_miss)
        monkeypatch.setattr(fleet_cells, "_cuda_ready", lambda: True)
        monkeypatch.setattr(cc, "toolchain_present", lambda: True)
        monkeypatch.setattr(cc, "apply", _sim_apply_factory(self.sim))
        # pgw#681 gate at its torch boundary, simmed like apply's compile
        # leaf: _Sim never touches dynamo, so extraction would honestly
        # report closure unprovable and refuse every finalize.
        # pgw#1181: the pgw#681 mint gate this simmed is deleted.
        # `guard_closure.closure_manifest` classified every compiled graph at
        # the MINT and wrote the result into the cell's metadata; it went with
        # the `torch-inductor-cache` format that carried it, so a rig whose
        # compiles never touch dynamo has no gate left to satisfy.
        monkeypatch.setattr(
            cc, "inductor_counters", lambda: dict(self.sim.counters))
        monkeypatch.setattr(fleet_cells, "arm_identity", _fake_arm_identity)
        # torch boundary of the fix: the scoped reset drops the simulator's
        # in-memory code entries, exactly like torch._dynamo.reset_code.
        import torch._dynamo

        monkeypatch.setattr(
            torch._dynamo, "reset_code",
            lambda code: self.sim.inmem.discard(code))
        for spec in specs:
            monkeypatch.setattr(
                self.ex, "_enable_compiled",
                lambda p, cfg, artifact, delivered=None, arm=None,
                       boot_local_key="":
                    fleet_cells.enable_compiled(
                        p, cfg, self.ex.store._cache_dir, artifact,
                        publisher=None))
            break

    def boot(self, spec: EndpointSpec) -> None:
        model_ref = wire_ref(spec.models["pipeline"])
        asyncio.run(self.ex.ensure_setup(
            spec, {model_ref: pb.Snapshot(digest=MODEL_DIGEST)}))


# pgw#1010: requirement (b) — "finalize succeeds for the second same-family
# arm (its own graphs)" — is deleted with the finalize it named. Both tests
# that carried it (`test_second_same_family_arm_mints_its_own_graphs_and_
# finalizes`, `test_same_key_rearm_reuses_finalized_cell_with_a_real_fx_hit`)
# packed and re-armed an in-process DYNAMO capture, which no longer exists:
# a family without an export declaration serves JIT intake and produces no
# artifact at all. Requirements (a) — the per-code reset that makes a warmup go
# through the real lookup path — and (c) — degrade instead of die — are what the
# live route still has, and they are asserted below.


# pgw#1010: `test_genuinely_unservable_execution_lane_degrades_to_eager_not_death`
# stood here. Its mechanism was the MINT proof: an in-process capture whose
# warmup was served counter-silently disproved itself, the identity was
# quarantined, and the pod degraded instead of dying. An intake arm has no
# artifact to prove — the compile happened in this process, so there are no
# bytes whose provenance a proof could certify — and `proves_inductor` is
# false without an active artifact, so that whole window does not run for it.
# What the pgw#672 posture requirement (c) still asserts is below: a compile
# that genuinely fails degrades THIS worker to eager, alive and loud. The
# proof-window honesty half (requirement (a)) keeps its coverage on the lane
# it still guards — an ADOPTED cell — in `test_executor_adopt.py`.


def test_a_mandatory_lane_without_a_declaration_fails_closed_before_it_compiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1010, the other half of requirement (c).

    A mandatory (w8a8/w4a4) lane serves only from a CELL — the th#910 dispatch
    fence pins every request to an active compile incarnation — so a JIT intake
    arm there would compile a whole boot's worth of graphs for a pod that then
    refuses every request `required_compile_missing`. It fails closed instead,
    typed, exactly as a mandatory lane did before self-mint existed.
    """
    from gen_worker.cell_adopt import AdoptOutcome

    monkeypatch.setattr(
        provision, "enable_compiled",
        lambda *a, **k: AdoptOutcome.miss("no_cell"))
    monkeypatch.setattr(fleet_cells, "_cuda_ready", lambda: True)
    monkeypatch.setattr(cc, "toolchain_present", lambda: True)
    monkeypatch.setattr(cc, "mandatory_serving", lambda pipe: True)
    monkeypatch.setattr(
        cc, "arm_jit_intake",
        lambda *a, **k: pytest.fail(
            "a mandatory lane must not compile an intake arm it cannot serve"))

    pipe = _Pipe()
    # pgw#888: this family declares NO export, so the refusal is PERMANENT —
    # no pod can ever hold a cell for it — and it is therefore the terminal
    # class, not the retryable one. Retrying it was pgw#888's own observation
    # (11 requests, five attempts each, one answer).
    with pytest.raises(cc.CompiledExecutionLaneImpossibleError, match="cell"):
        fleet_cells.enable_compiled(
            pipe, Compile(shapes=((768, 768),), family=FAMILY, text_len=0),
            tmp_path, None, publisher=None)


def test_compile_failure_on_a_compiled_lane_degrades_per_sku(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """pgw#673 posture half: an InductorError-class failure compiling the
    target (sm120 CantSplit) degrades THAT worker to eager serving —
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
    # pgw#1010: active-LESS, not absent — see the note in the test above.
    assert all(not t.active_compile_ref for t in rig.ex.compile_targets())
    assert any(
        "CantSplit" in r.getMessage() and "eager" in r.getMessage()
        for r in caplog.records
    )
