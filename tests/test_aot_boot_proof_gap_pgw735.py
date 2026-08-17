"""Pure-AOT boot arms run the boot warmup proof.

An executor proof that scores EXPORTED arms inside a loop gated on
``proves_inductor`` runs only when SOME dynamo selection coexists. A worker
whose ONLY arm is an adopted AOT cell then skips the boot proof entirely and
stays armed UNPROVEN. Same fail-closed rule as the dynamo lane, through the
REAL executor setup path (fakes only at the download + arming boundaries):

  1. an exercised pure-AOT arm is PROVEN — stays armed, cell recorded
     proven in-process (the pgw#637 registry);
  2. an unexercised pure-AOT arm KEEPS SERVING and banks no proof;
  3. the same on the MANDATORY (w8a8) lane does not kill the boot.

An unexercised arm must NOT be unwrapped, quarantined or dropped: an ADOPTED
cell arms before setup, so nothing has dispatched through it by construction,
and pods threw away artifacts they had just verified at `cos=1.00000`. The
fail-closed property lives elsewhere — the numerics gate refuses a cell that
does not reproduce eager, and a cell-attributable failure at SERVE time revokes
the arm in-request. What an absent measurement still decides is the PUBLISH,
which is what rows 2 and 3 pin.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, cast

import msgspec
import pytest

from gen_worker.serving_facts import FactsUnavailable as _FU

_NO_FACTS = _FU(owed_by="a test that resolves no catalog")

import gen_worker
from gen_worker import aot_serve, compile_cache, fleet_cells
from gen_worker.api.decorators import Compile
from gen_worker import RequestContext, Resources, Slot, endpoint, worker_function
from gen_worker.executor import Executor
from gen_worker.models.refs import normalize_model_ref
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs
from gen_worker.models import store as store_mod
from gen_worker._vendor.torchcg import CallIngress, CallInput, CompiledGraphRunner

FAMILY = "sdxl"


class GenIn(msgspec.Struct):
    prompt: str = "warm"
    num_inference_steps: int = 4


class Out(msgspec.Struct):
    y: str = "ok"


class _Unet:
    """The compile target: a module-shaped object with a bound forward."""

    def __init__(self) -> None:
        self.forwards = 0

    def forward(self, *args: Any, **kwargs: Any) -> None:
        self.forwards += 1


class _Pipe:
    def __init__(self) -> None:
        self.unet = _Unet()


class _FakeTCGRunner:
    bound = True
    declared_fqns: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.calls = 0


#: per-test wiring the endpoint instance reads at setup/handler time.
RIG: Dict[str, Any] = {}


@endpoint(
    models={"pipeline": Slot(str)},
    resources=Resources(gpu=True),
    compile=Compile(family=FAMILY, targets=("unet",), text_len=0,
                    shapes=((1024, 1024),)),
)
class AotFamily:
    def setup(self, pipeline: str) -> None:
        self.pipe = _Pipe()
        execution_lane = RIG.get("weight_lane")
        if execution_lane:
            setattr(self.pipe, "_cozy_weight_lane", execution_lane)
        RIG["pipe"] = self.pipe
        gen_worker.arm_compile(self.pipe)

    @worker_function()
    def generate(self, ctx: RequestContext, p: GenIn) -> Out:
        # The warmup exercising this handler "runs the artifact" only when
        # the rig says so — the unexercised direction models an armed .pt2
        # the warm plan never actually invoked.
        if RIG.get("exercise"):
            # `execution_count` sums the RUNNERS' own calls, so
            # "the artifact ran" means an artifact's counter moved — not a
            # state field production no longer reads. Bumping the old field
            # would model an exercise that never happened.
            marker = getattr(self.pipe, aot_serve._MARKER_ATTR, None)
            for row in ((marker or {}).get("targets") or {}).values():
                runner = (row.get("state") or {}).get("runner")
                for _name, art in getattr(runner, "runners", ()) or ():
                    art.runner.calls += 1
        return Out()


def _fake_arm(key: str, ref: str):
    """A fleet policy standing in for F1: arm an exported cell on the unet
    and return the adopted identity — the executor path from ArmOutcome to
    the boot proof is the code under test and runs REAL."""

    def _enable(pipe: Any, cfg: Any, cache_dir: Any, artifact: Any,
                publisher: Any = None) -> "fleet_cells.ArmOutcome":
        unet = pipe.unet
        # production wraps a REGISTRY; `is_armed` reads it.
        _runner = aot_serve.TCGEntryRunner(
            runner=cast(CompiledGraphRunner, _FakeTCGRunner()),
            contract=CallIngress(
                parameters=("sample",),
                flat_arity=1,
                inputs=(CallInput(
                    "sample", 0, "sample", 0, (), "sample", "float32", (1,),
                ),),
            ),
            module_name="unet", entry="unet/main", family=FAMILY)
        _dispatch = aot_serve.EntryDispatch(declared=("unet/main",))
        _dispatch.add("unet/main", _runner)
        state = {"successful_calls": 0, "failed": False,
                 "original": unet.forward, "runner": _dispatch}
        # The two markers are DIFFERENT SHAPES in production and
        # this rig now models that honestly. `wrap_module` writes a bare
        # `state` on the MODULE; `arm_entry` writes `targets` (+ `entries`) on
        # the PIPELINE. Sharing one dict between them was what kept a
        # `_marker_states` fallback alive for a shape nothing produces.
        setattr(unet, aot_serve._MARKER_ATTR, {
            "meta": {}, "state": state})
        setattr(pipe, aot_serve._MARKER_ATTR, {
            "meta": {},
            "targets": {"unet": {
                "module": unet, "attr": "forward", "state": state}},
            "entries": {"unet/main": {"key": ""}},
        })
        # An `aot_serve.note_aot_key(key)` stood here — the ONE line no
        # production arm route ever called, which is why these rows were green
        # while the pod served eager (pgw#1141b). It is DELETED, not moved: the
        # marker set above is what `arm_entry` publishes, so
        # `holds_exported_cell` answers the lane question off the OBJECT.
        # A fixture that needs a REAL boot-adopt drives tests/harness/adopt_rig.py.
        adopted = fleet_cells.SelfMint(
            family=FAMILY, compiled_graph_key=key, ref=ref,
            snapshot_digest="blake3:" + "ab" * 32,
            artifact=Path(cache_dir or ".") / "cell.tar")
        return fleet_cells.ArmOutcome(armed=True, self_mint=adopted)

    return _enable


def _executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Executor:
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    ex = Executor(extract_specs(AotFamily), _send)
    ex.store._cache_dir = tmp_path / "cas"

    async def _fake_download(ref: str, **kwargs: Any) -> Path:
        p = tmp_path / ref.replace("/", "_").replace(":", "_")
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(store_mod, "ensure_local", _fake_download)
    return ex


def _orders(run):
    """pgw#904: the driver reads neutral slot orders, never the wire message."""
    from gen_worker import dispatch

    return {
        b.slot: dispatch.SlotOrder(ref=b.ref.strip(), facts=_NO_FACTS)
        for b in run.models if b.slot
    }


def _boot(ex: Executor, ref: str = "acme/sdxl-base@prod") -> None:
    spec = ex.specs["generate"]
    run = pb.RunJob(function_name="generate",
                    models=[pb.ModelBinding(slot="pipeline", ref=ref)])
    eff = ex._dispatched_spec(spec, _orders(run))
    # Key the snapshot map by the NORMAL FORM the worker will actually look
    # up, not the raw string — th#1276 moved which tag the normal form elides
    # (`:prod` now, `:latest` before), and a hand-spelled key silently misses.
    snaps = {normalize_model_ref(ref): pb.Snapshot(digest="d1" * 16, files=[pb.SnapshotFile(
        path="model.safetensors", size_bytes=5, blake3="cd" * 32,
        url="http://r2.invalid/presigned")])}
    asyncio.run(ex.ensure_setup(eff, snaps))


def _rig(monkeypatch: pytest.MonkeyPatch, *, seed: str, exercise: bool,
         weight_lane: str = "") -> tuple[str, str]:
    RIG.clear()
    RIG["exercise"] = exercise
    if weight_lane:
        RIG["weight_lane"] = weight_lane
    key = "cg-key-v1-" + (seed * 56)[:56]
    ref = f"root/family-{FAMILY}#{key}"
    monkeypatch.setattr(fleet_cells, "enable_compiled", _fake_arm(key, ref))
    return key, ref


def test_exercised_pure_aot_arm_is_proven_at_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _key, ref = _rig(monkeypatch, seed="a", exercise=True)
    ex = _executor(tmp_path, monkeypatch)
    _boot(ex)
    pipe = RIG["pipe"]
    assert aot_serve.is_armed(pipe)
    assert aot_serve.execution_count(pipe) > 0
    # The gap: with no dynamo selection coexisting, the proof loop never
    # ran, so an honestly-exercised arm was never RECORDED proven.
    assert compile_cache.compiled_graph_proven_in_process(ref)
    assert not compile_cache.compiled_graph_quarantined_in_process(ref)


def test_unexercised_pure_aot_arm_keeps_serving_and_banks_no_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1141: absence of a dispatch is not a verdict about the artifact.
    The arm stands and the first real request is the proof; the cell is not
    recorded proven, which is what the publish gate reads."""
    _key, ref = _rig(monkeypatch, seed="b", exercise=False)
    ex = _executor(tmp_path, monkeypatch)
    _boot(ex)
    pipe = RIG["pipe"]
    assert aot_serve.is_armed(pipe)
    assert getattr(pipe.unet, aot_serve._MARKER_ATTR, None) is not None
    assert not compile_cache.compiled_graph_quarantined_in_process(ref)
    assert not compile_cache.compiled_graph_proven_in_process(ref)


def test_mandatory_execution_lane_without_a_dispatch_does_not_kill_the_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _key, ref = _rig(monkeypatch, seed="c", exercise=False,
                     weight_lane="w8a8-lora64")
    ex = _executor(tmp_path, monkeypatch)
    _boot(ex)  # never a load failure
    pipe = RIG["pipe"]
    assert aot_serve.is_armed(pipe)
    assert not compile_cache.compiled_graph_proven_in_process(ref)
