"""pgw#1141: a boot-ADOPTED cell carries its own proof, and the setup warmup
proof gate must READ it instead of destroying the artifact that holds it.

MEASURED, twice, on two real pods (RTX 4000 Ada, gen-worker 0.106.0, hub
`7ae35d54a2`): `boot_adopt=hit` -> materialize -> `cell_numerics cos=1.00000
ret=1.0000 rel_l2=0.0000` on 3/3 axes -> and then the setup warmup scored the
same artifact `unexercised` (it dispatched nothing through it — the adopt arms
BEFORE setup, so by construction nothing has), folded it into `unproven`,
`function_proofs[id]=set()`, `aot_serve.unwrap`, `compile_cache.unwrap`. The
pod served eager for life and published nothing. The self-mint arm on the SAME
wheel, card and release served `+compiled` with an empty `fallback_reason`,
because it reaches its proof through a warm dispatch.

So the gate was destroying evidence STRONGER than the evidence it demands. The
pgw#868 numerics gate runs every packaged entry through its own runner against
the eager forward it replaces — that is "it executed and is still armed"
(`proven_since`'s whole test) plus the accuracy a dispatch counter cannot see.
Both arms take that measurement in `provision.arm_aot`, so banking it on the
arm makes adopted and self-minted cells symmetrical rather than special-casing
either.

Two directions, and both must hold:

* an adopted cell with a standing parity verdict is PROVEN — armed, target
  installed, serving compiled, and the pass is announced
  (`cell_numerics phase=serving_proof`);
* a cell with no verdict, or one whose artifact REVOKED after it was measured,
  is still disarmed — and says which half was missing
  (`cell_numerics phase=proof_absent`). A cell that fails numerics never
  reaches this gate at all: `arm_aot` unwraps it.

The gate itself is never stubbed. Part A drives the REAL `provision.arm_aot`
against a real packed artifact (the pgw#868 rig, whose ONE substitution is the
AOTI `.so`); part B drives the REAL `ensure_setup` warmup proof, faking only
the download and the arming policy — the same seam `test_aot_boot_proof_gap_
pgw735.py` uses.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Tuple

import msgspec
import pytest

import gen_worker
import gen_worker.executor as ex_mod
from gen_worker import RequestContext, Resources, Slot, endpoint, worker_function
from gen_worker import activity, aot_serve, cell_adopt, compile_cache, fleet_cells
from gen_worker import numerics_ladder
from gen_worker.api.decorators import Compile
from gen_worker.executor import Executor
from gen_worker.models.refs import normalize_model_ref
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs

# The rig of the REAL arm path (real artifact, real gate, real ladder).
import test_numerics_gate_pgw868 as rig868  # noqa: E402
from test_numerics_gate_pgw868 import ROWS, ProbePackage, arm, entry_name  # noqa: E402

#: pgw#868's fixtures, re-exported so this module collects them: `declared`
#: registers the real export declaration the probe feed is built from, and
#: `events` captures the typed rows.
declared = rig868.declared
events = rig868.events

FAMILY = "micro-diffusion"


# ---------------------------------------------------------------------------
# PART A — the parity verdict is BANKED on the arm that earned it
# ---------------------------------------------------------------------------


def test_a_faithful_arm_banks_its_parity_verdict(
        tmp_path, monkeypatch, declared, events):
    """The measurement the pod already paid for survives the function that
    took it. Pre-fix it was announced and dropped, which is why the warmup
    gate had nothing to read."""
    packages = {entry_name(h, w): ProbePackage() for h, w in ROWS}
    pipeline, _module, outcome = arm(tmp_path, monkeypatch, declared, packages)

    assert outcome.armed is True
    proof = aot_serve.numerics_proof(pipeline)
    assert proof is not None, "the arm banked no parity verdict"
    assert proof.axes == len(ROWS)
    assert proof.cell_key == "cell868"
    assert proof.verdict == numerics_ladder.VERDICT_HEALTHY
    assert proof.worst_cosine == pytest.approx(1.0, abs=1e-6)
    assert proof.worst_entry in {entry_name(h, w) for h, w in ROWS}


def test_a_gray_band_arm_banks_its_verdict_as_degraded(
        tmp_path, monkeypatch, declared, events):
    """A cell inside the declared gray band ARMS and confesses. It therefore
    serves, so it must carry a proof too — with the rung it actually reached,
    not a rounded-up one."""
    packages = {entry_name(h, w): ProbePackage(cosine=0.997) for h, w in ROWS}
    pipeline, _module, outcome = arm(tmp_path, monkeypatch, declared, packages)

    assert outcome.armed is True
    proof = aot_serve.numerics_proof(pipeline)
    assert proof is not None
    assert proof.verdict == numerics_ladder.VERDICT_DEGRADED
    assert proof.worst_cosine == pytest.approx(0.997, abs=1e-3)


def test_a_refused_cell_banks_nothing(tmp_path, monkeypatch, declared, events):
    """THE fail-closed half: a cell below its floor does not arm, so there is
    no marker, no proof, and nothing downstream can be carried by one."""
    packages = {entry_name(h, w): ProbePackage(cosine=0.99) for h, w in ROWS}
    pipeline, _module, outcome = arm(tmp_path, monkeypatch, declared, packages)

    assert outcome.armed is False
    assert aot_serve.is_armed(pipeline) is False
    assert aot_serve.numerics_proof(pipeline) is None
    assert aot_serve.numerics_measured(pipeline) is False


def test_the_proof_cannot_outlive_the_arm_it_is_about(
        tmp_path, monkeypatch, declared, events):
    """Scoped to the exact wrap that earned it: unwrap drops the marker, and a
    revoked target keeps the record while withdrawing the PROOF — the two are
    different facts and the disarm names which one it saw."""
    packages = {entry_name(h, w): ProbePackage() for h, w in ROWS}
    pipeline, _module, outcome = arm(tmp_path, monkeypatch, declared, packages)
    assert outcome.armed is True

    # A revoked target: measured, but no longer serving anything.
    for row in aot_serve.armed_targets(pipeline).values():
        row["state"]["failed"] = True
    assert aot_serve.numerics_proof(pipeline) is None
    assert aot_serve.numerics_measured(pipeline) is True

    aot_serve.unwrap(pipeline)
    assert aot_serve.numerics_measured(pipeline) is False
    assert aot_serve.record_numerics_proof(
        pipeline, aot_serve.NumericsProof("k", 1, "e", 1.0, "healthy")) is False


# ---------------------------------------------------------------------------
# PART B — the boot warmup proof gate, through the real `ensure_setup`
# ---------------------------------------------------------------------------


class GenIn(msgspec.Struct):
    prompt: str = "warm"
    num_inference_steps: int = 2


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
                    shapes=((256, 256),)),
)
class AdoptedFamily:
    def setup(self, pipeline: str) -> None:
        self.pipe = _Pipe()
        RIG["pipe"] = self.pipe
        gen_worker.arm_compile(self.pipe)

    @worker_function()
    def generate(self, ctx: RequestContext, p: GenIn) -> Out:
        # The pod shape: the boot warmup runs, and its payload lands on no
        # packaged entry of the adopted cell — so the artifact's own execution
        # counter does not move. `RIG["dispatch"]` models the OTHER arm, where
        # a warm dispatch does land.
        if RIG.get("dispatch"):
            marker = getattr(self.pipe, aot_serve._MARKER_ATTR, None)
            if marker is not None:
                marker["state"]["successful_calls"] += 1
        return Out()


#: The three dispatches the pgw#868 gate itself made at arm time. `aot_proof_
#: before` is snapshotted AFTER the arm, so these prove nothing to a gate that
#: only counts — which is exactly the pod's reading.
PROBE_CALLS = 3


def _fake_adopt_arm(key: str, ref: str, *, bank: bool, revoke: bool = False):
    """A fleet policy standing in for §4.27 boot-adopt: arm the resolved cell
    on the unet, run the parity gate (or not), and hand back the adopted
    identity. Everything from `ArmOutcome` to the boot proof runs REAL."""

    def _enable(pipe: Any, cfg: Any, cache_dir: Any, artifact: Any,
                publisher: Any = None, **_kw: Any) -> "fleet_cells.ArmOutcome":
        unet = pipe.unet
        state = {"successful_calls": PROBE_CALLS, "failed": False,
                 "original": unet.forward}
        marker = {"module": unet, "state": state, "meta": {"cell_key": key}}
        setattr(unet, aot_serve._MARKER_ATTR, marker)
        setattr(pipe, aot_serve._MARKER_ATTR, marker)
        aot_serve.note_aot_key(key)
        if bank:
            assert aot_serve.record_numerics_proof(pipe, aot_serve.NumericsProof(
                cell_key=key, axes=3, worst_entry="unet/h=256,w=256",
                worst_cosine=1.0, verdict=numerics_ladder.VERDICT_HEALTHY,
                elapsed_ms=158))
        if revoke:
            state["failed"] = True
        adopted = fleet_cells.SelfMint(
            family=FAMILY, cell_key=key, ref=ref,
            snapshot_digest="blake3:" + "ab" * 32,
            artifact=Path(cache_dir or ".") / "cell.tar")
        return fleet_cells.ArmOutcome(armed=True, self_mint=adopted)

    return _enable


def _rig(monkeypatch: pytest.MonkeyPatch, *, seed: str, bank: bool,
         revoke: bool = False, dispatch: bool = False) -> Tuple[str, str]:
    RIG.clear()
    RIG["dispatch"] = dispatch
    key = "ck1-" + (seed * 56)[:56]
    ref = f"root/family-{FAMILY}#{key}"
    monkeypatch.setattr(
        fleet_cells, "enable_compiled",
        _fake_adopt_arm(key, ref, bank=bank, revoke=revoke))
    return key, ref


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, str, str]]:
    said: List[Tuple[str, str, str]] = []
    real = activity.emit_event

    def _spy(kind: str, detail: str = "", **kw: Any) -> Any:
        said.append((kind, str(kw.get("phase", "")), detail))
        try:
            return real(kind, detail, **kw)
        except Exception:
            return None

    monkeypatch.setattr(activity, "emit_event", _spy)
    monkeypatch.setattr(ex_mod.activity_mod, "emit_event", _spy)
    return said


def _boot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
          ref: str = "acme/micro-diffusion:prod") -> Tuple[Executor, Any]:
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    ex = Executor(extract_specs(AdoptedFamily), _send)
    ex.store._cache_dir = tmp_path / "cas"

    async def _fake_download(target: str, **_kw: Any) -> Path:
        p = tmp_path / target.replace("/", "_").replace(":", "_")
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(ex_mod, "ensure_local", _fake_download)

    from gen_worker import dispatch as dispatch_mod

    spec = ex.specs["generate"]
    eff = ex._dispatched_spec(
        spec, {"pipeline": dispatch_mod.SlotOrder(ref=ref, components=())})
    snaps = {normalize_model_ref(ref): pb.Snapshot(
        digest="d1" * 16,
        files=[pb.SnapshotFile(
            path="model.safetensors", size_bytes=5, blake3="cd" * 32,
            url="http://r2.invalid/presigned")])}
    asyncio.run(ex.ensure_setup(eff, snaps))
    return ex, eff


def _phases(said: List[Tuple[str, str, str]], kind: str) -> List[str]:
    return [phase for k, phase, _d in said if k == kind]


def test_an_adopted_cell_with_a_standing_parity_proof_STAYS_ARMED(
        tmp_path, monkeypatch, spy):
    """THE RED. Pre-fix: `unexercised` -> `unproven` -> unwrap -> quarantine,
    `functions=()`, `target_applicability_incomplete`, eager for life. The
    artifact was verified at cos=1.00000 on this very pod first."""
    _key, ref = _rig(monkeypatch, seed="a", bank=True)
    ex, eff = _boot(tmp_path, monkeypatch)
    pipe = RIG["pipe"]

    assert aot_serve.is_armed(pipe) is True, (
        "the pod threw away a cell it had just verified against eager")
    assert getattr(pipe.unet, aot_serve._MARKER_ATTR, None) is not None
    assert compile_cache.cell_proven_in_process(ref)
    assert not compile_cache.cell_quarantined_in_process(ref)

    # ...and the arm is DISPATCHABLE, which is the half the pod actually lost:
    # a proof that leaves `function_proofs` empty installs no target and the
    # boot still ends `boot_ended_uncompiled`.
    rec = ex._classes[eff.instance_key]
    assert rec.compile_targets, "proven and still no installed compile target"
    target = next(iter(rec.compile_targets.values()))
    assert "generate" in target.function_names
    assert rec.eager_posture == ""
    assert cell_adopt.EagerPhase.ARMED_TARGET_UNRESOLVED.value not in _phases(
        spy, "serve_eager_posture")
    assert "target_applicability_incomplete" not in _phases(
        spy, "serve_eager_posture")
    assert ex._served_execution_lane(eff).endswith("+compiled")

    # An unannounced pass is indistinguishable from a gate that never ran.
    rows = [(phase, detail) for kind, phase, detail in spy
            if kind == activity.KIND_CELL_NUMERICS]
    assert [p for p, _d in rows] == [numerics_ladder.PHASE_SERVING_PROOF], rows
    assert "cos=1.00000" in rows[0][1]
    assert "axes=3" in rows[0][1]


def test_an_arm_with_NO_warm_dispatch_STILL_SERVES_and_withholds_its_publish(
        tmp_path, monkeypatch, spy):
    """THE DELETED BARRIER, in the direction that used to be its whole point.

    Nothing dispatched through this artifact and nothing measured it, so this
    pod holds no evidence either way — and under the serve-first ruling that is
    NOT a verdict. The cell serves; the first real request is the proof; a
    cell-attributable failure revokes it in-request. What absence of evidence
    still decides is PUBLISHING: serving optimistically costs this pod one
    eager fallback, publishing an unverified cell costs every pod that adopts
    it."""
    _key, ref = _rig(monkeypatch, seed="b", bank=False)
    ex, eff = _boot(tmp_path, monkeypatch)
    pipe = RIG["pipe"]

    assert aot_serve.is_armed(pipe) is True
    assert getattr(pipe.unet, aot_serve._MARKER_ATTR, None) is not None
    assert not compile_cache.cell_quarantined_in_process(ref)
    # ...but nothing proved it either: the publish gate reads this.
    assert not compile_cache.cell_proven_in_process(ref)
    rec = ex._classes[eff.instance_key]
    assert rec.compile_targets, "the arm stands, so its target must be installed"

    # LOUD, at the decision point — an unannounced posture is indistinguishable
    # from a gate that never ran.
    rows = [(phase, detail) for kind, phase, detail in spy
            if kind == activity.KIND_CELL_NUMERICS]
    assert [p for p, _d in rows] == [numerics_ladder.PHASE_PROOF_ABSENT], rows
    assert "STAYS ARMED" in rows[0][1]
    assert "no parity verdict was banked" in rows[0][1]


def test_a_REVOKED_artifact_is_not_carried_by_the_verdict_it_once_passed(
        tmp_path, monkeypatch, spy):
    """`proven_since`'s second half, applied to the parity proof: an artifact
    that was measured and then revoked has proven nothing. The arm is already
    gone (the wrapper revoked it), so what this pins is that the stale verdict
    does not vouch for it — the publish stays withheld."""
    _key, ref = _rig(monkeypatch, seed="c", bank=True, revoke=True)
    ex, eff = _boot(tmp_path, monkeypatch)
    pipe = RIG["pipe"]

    assert aot_serve.is_armed(pipe) is False
    assert not compile_cache.cell_proven_in_process(ref)
    rows = [(phase, detail) for kind, phase, detail in spy
            if kind == activity.KIND_CELL_NUMERICS]
    assert [p for p, _d in rows] == [numerics_ladder.PHASE_PROOF_ABSENT], rows
    assert "revoked since it was measured" in rows[0][1]


# ---------------------------------------------------------------------------
# PART C — try-serve: the first real request is the proof, and what it blames
# ---------------------------------------------------------------------------


class _Runner:
    """One entry's artifact runner: counts calls, raises what it is told to."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.calls = 0
        self.exc = exc

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return "compiled"


class _Module:
    def __init__(self) -> None:
        self.eager_calls = 0

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.eager_calls += 1
        return "eager"


def _wrapped(runner: _Runner, spy: List[Tuple[str, str, str]]) -> Any:
    module = _Module()
    aot_serve.wrap_module(
        module, runner, {"family": FAMILY, "cell_key": "ck1-x"},
        attr="forward", target="unet")
    return module


def test_a_cell_that_FAILS_at_serve_answers_the_request_eager_and_stays_down(
        spy, monkeypatch):
    """Paul's clause 2, exactly: the tenant whose request hit the bad cell
    still gets a correct answer, the disarm is STICKY for the process, and the
    revocation reaches the scheduler state through the wrapper's own callback.
    Nothing here is stubbed — this is the shipping wrapper."""
    runner = _Runner(RuntimeError("aoti: illegal memory access"))
    module = _wrapped(runner, spy)
    revoked: List[str] = []
    module._cozy_aot["state"]["failure_callback"] = revoked.append

    assert module.forward(1) == "eager", "the request must not raise"
    assert module.eager_calls == 1
    assert revoked and "illegal memory access" in revoked[0]

    # Sticky: no retry loop against a broken artifact.
    assert module.forward(2) == "eager"
    assert runner.calls == 1, "a revoked artifact was called again"
    assert module._cozy_aot["state"]["failed"] is True


def test_a_TRANSIENT_OOM_is_not_the_cells_fault_and_does_not_disarm_it(
        spy, monkeypatch):
    """Honest attribution, which is what makes serve-first safe: allocator
    exhaustion is a fact about the CARD at this instant (a sibling load, a
    rotation), not about the artifact. Condemning the cell for it would retire
    a correct one on the first busy moment and re-mint it on the replacement
    pod. The request is still answered."""
    runner = _Runner(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    module = _wrapped(runner, spy)
    revoked: List[str] = []
    module._cozy_aot["state"]["failure_callback"] = revoked.append

    assert module.forward(1) == "eager"
    assert revoked == [], "a transient OOM revoked the cell"
    assert module._cozy_aot["state"]["failed"] is False

    # ...and the artifact is still the serving lane once the pressure passes.
    runner.exc = None
    assert module.forward(2) == "compiled"
    assert [(k, p) for k, p, _d in spy if k == "aot_serve_oom"] == [
        ("aot_serve_oom", "cuda_oom")]


def test_a_warm_DISPATCH_still_proves_first_and_says_nothing_extra(
        tmp_path, monkeypatch, spy):
    """Symmetry, from the other side. The self-mint arm reaches its proof by a
    warm dispatch and is untouched by this issue: when a dispatch lands, it is
    the proof, and no parity row is emitted."""
    _key, ref = _rig(monkeypatch, seed="d", bank=True, dispatch=True)
    ex, eff = _boot(tmp_path, monkeypatch)
    pipe = RIG["pipe"]

    assert aot_serve.is_armed(pipe) is True
    assert aot_serve.execution_count(pipe) > PROBE_CALLS
    assert compile_cache.cell_proven_in_process(ref)
    rec = ex._classes[eff.instance_key]
    assert rec.compile_targets
    assert [phase for kind, phase, _d in spy
            if kind == activity.KIND_CELL_NUMERICS] == []
