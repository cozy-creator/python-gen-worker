"""Pure-AOT boot arms run the boot warmup proof (pgw#722 finding 2 — the
#735 boot-proof gap).

The shipped #735 executor proof scored EXPORTED arms inside a loop gated on
``proves_inductor`` — it ran only when SOME dynamo selection coexisted. A
worker whose ONLY arm is an adopted AOT cell (the prod flip shape: F1
adopts, the delivered dynamo artifact is skipped) skipped the boot proof
entirely and stayed armed UNPROVEN: an artifact that never executed — or
executed and revoked — kept advertising itself. Same fail-closed rule as
the dynamo lane, through the REAL executor setup path (fakes only at the
download + arming boundaries):

  1. an exercised pure-AOT arm is PROVEN — stays armed, cell recorded
     proven in-process (the pgw#637 registry);
  2. an unexercised pure-AOT arm is DISARMED to true eager — artifact
     unwrapped (original forward restored), identity quarantined, no
     active selection kept;
  3. the same disarm on the MANDATORY (w8a8) lane rides the pgw#672
     degrade-to-eager posture without killing the boot.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List

import msgspec
import pytest

import gen_worker
from gen_worker import aot_serve, compile_cache, fleet_cells
from gen_worker.api.decorators import Compile
from gen_worker import RequestContext, Resources, Slot, endpoint, worker_function
from gen_worker.executor import Executor
from gen_worker.models.refs import normalize_model_ref
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs

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
            marker = getattr(self.pipe, aot_serve._MARKER_ATTR, None)
            if marker is not None:
                marker["state"]["successful_calls"] += 1
        return Out()


def _fake_arm(key: str, ref: str):
    """A fleet policy standing in for F1: arm an exported cell on the unet
    and return the adopted identity — the executor path from ArmOutcome to
    the boot proof is the code under test and runs REAL."""

    def _enable(pipe: Any, cfg: Any, cache_dir: Any, artifact: Any,
                publisher: Any = None) -> "fleet_cells.ArmOutcome":
        unet = pipe.unet
        state = {"successful_calls": 0, "failed": False,
                 "original": unet.forward}
        marker = {"module": unet, "state": state, "meta": {}}
        setattr(unet, aot_serve._MARKER_ATTR, marker)
        setattr(pipe, aot_serve._MARKER_ATTR, marker)
        aot_serve.note_aot_key(key)
        adopted = fleet_cells.SelfMint(
            family=FAMILY, cell_key=key, ref=ref,
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

    import gen_worker.executor as ex_mod

    monkeypatch.setattr(ex_mod, "ensure_local", _fake_download)
    return ex


def _boot(ex: Executor, ref: str = "acme/sdxl-base:prod") -> None:
    spec = ex.specs["generate"]
    run = pb.RunJob(function_name="generate",
                    models=[pb.ModelBinding(slot="pipeline", ref=ref)])
    eff = ex._effective_spec(spec, run)
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
    key = "ck1-" + (seed * 56)[:56]
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
    assert compile_cache.cell_proven_in_process(ref)
    assert not compile_cache.cell_quarantined_in_process(ref)


def test_unexercised_pure_aot_arm_disarms_to_true_eager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _key, ref = _rig(monkeypatch, seed="b", exercise=False)
    ex = _executor(tmp_path, monkeypatch)
    _boot(ex)
    pipe = RIG["pipe"]
    # The gap's fail-open half: pre-fix the arm stayed installed unproven.
    assert not aot_serve.is_armed(pipe)
    assert getattr(pipe.unet, aot_serve._MARKER_ATTR, None) is None
    assert pipe.unet.forward.__func__ is _Unet.forward
    assert compile_cache.cell_quarantined_in_process(ref)
    assert not compile_cache.cell_proven_in_process(ref)


def test_mandatory_execution_lane_disarm_degrades_without_killing_the_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _key, ref = _rig(monkeypatch, seed="c", exercise=False,
                     weight_lane="w8a8-lora64")
    ex = _executor(tmp_path, monkeypatch)
    _boot(ex)  # pgw#672: degrade, never a load failure
    pipe = RIG["pipe"]
    assert not aot_serve.is_armed(pipe)
    assert compile_cache.cell_quarantined_in_process(ref)
