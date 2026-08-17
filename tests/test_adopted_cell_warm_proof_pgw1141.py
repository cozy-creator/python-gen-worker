"""DESIGN-RULINGS §4.31 + §4.32: a boot-ADOPTED cell materializes, arms and
SERVES. No warmup barrier, and no quality gate at adoption.

§4.31: skip the warmup / arm check and serve right away; if serving raises and
the cell is the cause, de-arm the cell and serve eager. The barrier could never
pass for an adopt anyway — the adopt arms BEFORE setup, so the setup warmup
dispatches nothing through the artifact, scores it `unexercised`, folds it into
`unproven` and unwraps it. (The SELF-MINT arm is healthy because a mint's warmup
drives its own capture and therefore dispatches.)

§4.32: no adopt-side numerics re-check either. Every failure that gate ever
caught (a baked `conv_out.bias`, timestep dtype scars) was an AUTHOR defect in
endpoint code or config; re-measuring on every adopter taxes the fleet forever
for one author's one-time mistake. Adoption is materialize -> arm -> serve; the
gate runs ONCE, on the pod that minted the bytes, before they are published,
and it is STRICT.

Safety without re-measurement comes from CONSTRUCTION, not from checkpoint
identity (a `ck1` key is graph x envelope x sm x toolchain and carries no
checkpoint hash — one cell serves every checkpoint of the architecture, which
is the whole point of reuse): the cell is compiled CODE and weights flow
through it as data, so a mint-time parity proof proves the FUNCTION; a weight
value baked into the artifact is fenced fail-closed by the constant-folding
fence; and a checkpoint that changes the COMPUTATION hashes to a different
graph, hence a different key, hence no match.

What this file pins, in four parts:

A. the MINT gate — strict, identical-or-refuse, and it is the thing that
   decides whether bytes ship;
B. ADOPTION — a cell that would FAIL that gate still arms and serves, and no
   quality row is emitted at all, because nobody measured it;
C. the executor's setup warmup — an armed cell with no dispatch keeps its arm,
   its target and its aliases;
D. try-serve — a cell-attributable failure answers the request eager and
   de-arms sticky; a transient OOM does neither.

Nothing is stubbed. Parts A and B drive the REAL `provision.arm_aot` against a
real packed artifact (the rig's ONE substitution is the AOTI `.so`); part C
drives the REAL `ensure_setup`, faking only the download and the arming policy;
part D drives the real serving wrapper.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import msgspec
import pytest

from gen_worker.serving_facts import FactsUnavailable as _FU

_NO_FACTS = _FU(owed_by="a test that resolves no catalog")

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
from gen_worker.models import store as store_mod
from gen_worker._vendor.torchcg import CallIngress, CallInput, CompiledGraphRunner

#: pgw#868's fixtures, re-exported so this module collects them: `declared`
#: registers the real export declaration the probe feed is built from, and
#: `events` captures the typed rows.
declared = rig868.declared
events = rig868.events

FAMILY = "micro-diffusion"


# ---------------------------------------------------------------------------
# PART A — the MINT gate: strict, and it decides whether bytes ship
# ---------------------------------------------------------------------------


def test_the_MINTING_pod_proves_its_own_bytes_before_they_ship(
        tmp_path, monkeypatch, declared, events):
    """§4.32 item 2. The pod that compiled the artifact runs it against the
    eager forward it was traced from, on the same feed, and only then does the
    arm succeed — which is what `adopt_delegated_mint` needs to be true before
    `publish_self_mint` can ship anything."""
    packages = {entry_name(h, w): ProbePackage() for h, w in ROWS}
    pipeline, _module, outcome = arm(
        tmp_path, monkeypatch, declared, packages, verify_numerics=True)

    assert outcome.armed is True
    assert aot_serve.is_armed(pipeline) is True
    rows = [(p, d) for k, d, p in events if k == activity.KIND_COMPILED_GRAPH_NUMERICS]
    # ONE row per graph class — the gate runs at the moment that
    # entry exists, never "after all N" (DESIGN-RULINGS 4.32).
    assert [p for p, _d in rows] == ["checked", "checked"], rows
    # ONE axis per artifact, so the report is 1/1 twice, never
    # 2/2 once. A report over a collection is what the atom removed.
    assert "axes=1/1" in rows[0][1]


def test_the_MINT_gate_is_strict_a_gray_band_cell_does_not_ship(
        tmp_path, monkeypatch, declared, events):
    """§4.32: identical or refuse, no DEGRADED-publish band. An adopter runs no
    gate that could re-check what ships, so a gray-band publish would export an
    unmeasured degradation to every pod that pulls the key."""
    packages = {entry_name(h, w): ProbePackage(cosine=0.997) for h, w in ROWS}
    pipeline, _module, outcome = arm(
        tmp_path, monkeypatch, declared, packages, verify_numerics=True)

    assert outcome.armed is False, "a gray-band cell was published to the fleet"
    assert aot_serve.is_armed(pipeline) is False
    assert outcome.reason == "numerics_refused"
    # The refusal is per CLASS, so it says "is not published" of
    # that class rather than "nothing is published" of a bundle.
    assert "is not published" in outcome.detail
    # It still CONFESSES — a fleet-wide rate is only countable from rows.
    # One gate row PER GRAPH CLASS — this declaration has two,
    # so the vocabulary repeats rather than aggregating.
    assert [p for k, _d, p in events
            if k == activity.KIND_COMPILED_GRAPH_NUMERICS] == ["degraded", "degraded"]


def test_the_MINT_gate_refuses_a_cell_below_its_floor(
        tmp_path, monkeypatch, declared, events):
    """Unchanged by both rulings, and the reason the gate still exists at all:
    try-serve catches ERRORS, never wrong OUTPUT. A cell that runs cleanly and
    renders a bad image raises nothing."""
    packages = {entry_name(h, w): ProbePackage(cosine=0.99) for h, w in ROWS}
    pipeline, _module, outcome = arm(
        tmp_path, monkeypatch, declared, packages, verify_numerics=True)

    assert outcome.armed is False
    assert aot_serve.is_armed(pipeline) is False
    # One gate row PER GRAPH CLASS — this declaration has two,
    # so the vocabulary repeats rather than aggregating.
    assert [p for k, _d, p in events
            if k == activity.KIND_COMPILED_GRAPH_NUMERICS] == ["refused", "refused"]


# ---------------------------------------------------------------------------
# PART B — ADOPTION runs no quality gate at all
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cosine,label", [
    (0.99, "below the floor"),
    (0.997, "inside the gray band"),
])
def test_ADOPTION_arms_a_cell_the_mint_gate_would_have_REFUSED(
        tmp_path, monkeypatch, declared, events, cosine, label):
    """THE §4.32 RED, and it fails on master twice over: the adopt path used to
    run this gate and would have unwrapped both of these cells.

    Adoption is materialize -> arm -> serve. The bytes were proven once, at
    their mint; re-proving them on every adopter taxes the fleet forever for an
    author's one-time mistake, and it is the tax that made a pod throw away a
    cell it had just verified at `cos=1.00000`. Deliberately parametrized over
    a cell that would FAIL: the point is not that adoption is lucky, it is that
    adoption does not ASK."""
    packages = {entry_name(h, w): ProbePackage(cosine=cosine) for h, w in ROWS}
    pipeline, _module, outcome = arm(
        tmp_path, monkeypatch, declared, packages, verify_numerics=False)

    assert outcome.armed is True, f"an adopting pod re-judged a cell {label}"
    assert aot_serve.is_armed(pipeline) is True
    assert [k for k, _d, _p in events if k == activity.KIND_COMPILED_GRAPH_NUMERICS] == [], (
        "adoption emitted a quality verdict, so it ran a quality gate")


def test_ADOPTION_is_the_default_so_a_new_arm_path_cannot_inherit_the_tax(
        tmp_path, monkeypatch, declared, events):
    """`verify_numerics` defaults to False, and that direction is deliberate:
    the ONE caller that measures is the mint. A future arm path that forgets
    the flag adopts (correct); one that forgets it in the other direction would
    have silently re-imposed the per-adopter cost this issue deleted."""
    import inspect

    from gen_worker.models import provision

    sig = inspect.signature(provision.arm_aot)
    assert sig.parameters["verify_numerics"].default is False


# ---------------------------------------------------------------------------
# PART C — the setup warmup, through the real `ensure_setup`
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
                # `execution_count` sums the RUNNERS' own calls, so a
                # dispatch that only bumped the state counter moved a number
                # production does not read. Move the artifact's own.
                for row in (marker.get("targets") or {}).values():
                    runner = (row.get("state") or {}).get("runner")
                    for _name, art in getattr(runner, "runners", ()) or ():
                        art.runner.calls += 1
                    break
                marker["state"] = marker.get("state") or {}
        return Out()


#: Calls the artifact had already served before setup opened. Non-zero because
#: `aot_proof_before` is snapshotted AFTER the arm, so a counter-only gate can
#: never see them — which is exactly the pod's reading.
PROBE_CALLS = 3


def _fake_adopt_arm(key: str, ref: str, *, revoke: bool = False):
    """A fleet policy standing in for §4.27 boot-adopt: arm the resolved cell
    on the unet and hand back the adopted identity. Everything from
    `ArmOutcome` to the setup warmup runs REAL."""

    def _enable(pipe: Any, cfg: Any, cache_dir: Any, artifact: Any,
                publisher: Any = None, **_kw: Any) -> "fleet_cells.ArmOutcome":
        unet = pipe.unet
        # production's `arm_entry` puts an `EntryDispatch` REGISTRY
        # in `state["runner"]`, and `is_armed`/`armed_entries` read it — a
        # boolean was deleted precisely because it can claim more than the pod
        # serves. A double that sets the marker but no registry models a
        # pipeline production cannot produce, and every `is_armed` assertion
        # against it is testing a shape that does not exist.
        class _Runner:
            bound = True
            calls = PROBE_CALLS
            declared_fqns: tuple[str, ...] = ()

            def __call__(self, *_args: object) -> None:
                self.calls += 1

        _runner = aot_serve.TCGEntryRunner(
            runner=cast(CompiledGraphRunner, _Runner()),
            contract=CallIngress(
                parameters=("sample",),
                flat_arity=1,
                inputs=(CallInput(
                    "sample", 0, "sample", 0, (), "sample", "float32", (1,),
                ),),
            ),
            module_name="unet", entry="unet/main", family=FAMILY,
        )
        _dispatch = aot_serve.EntryDispatch(declared=("unet/main",))
        _dispatch.add("unet/main", _runner)
        state = {"successful_calls": PROBE_CALLS, "failed": False,
                 "original": unet.forward, "runner": _dispatch}
        # The two markers are DIFFERENT SHAPES in production and
        # this rig now models that honestly. `wrap_module` writes a bare
        # `state` on the MODULE; `arm_entry` writes `targets` (+ `entries`) on
        # the PIPELINE. Sharing one dict between them was what kept a
        # `_marker_states` fallback alive for a shape nothing produces.
        setattr(unet, aot_serve._MARKER_ATTR, {
            "meta": {"compiled_graph_key": key}, "state": state})
        setattr(pipe, aot_serve._MARKER_ATTR, {
            "meta": {"compiled_graph_key": key},
            "targets": {"unet": {
                "module": unet, "attr": "forward", "state": state}},
            "entries": {"unet/main": {"key": key, "target": "unet"}},
        })
        # An `aot_serve.note_aot_key(key)` stood here — the ONE line no
        # production arm route ever called, which is why these rows were green
        # while the pod served eager (pgw#1141b). It is DELETED, not moved: the
        # marker set above is what `arm_entry` publishes, so
        # `holds_exported_cell` answers the lane question off the OBJECT.
        # A fixture that needs a REAL boot-adopt drives tests/harness/adopt_rig.py.
        if revoke:
            # revoking is DE-ARMING. `is_armed` asks the registry,
            # so a flag left the runner armed and the pipeline claiming
            # compiled service it was not giving — the exact lie a cell-level
            # boolean makes possible.
            state["failed"] = True
            _dispatch.remove("unet/main", "revoked by the rig")
        adopted = fleet_cells.SelfMint(
            family=FAMILY, compiled_graph_key=key, ref=ref,
            snapshot_digest="blake3:" + "ab" * 32,
            artifact=Path(cache_dir or ".") / "cell.tar")
        return fleet_cells.ArmOutcome(armed=True, self_mint=adopted)

    return _enable


def _rig(monkeypatch: pytest.MonkeyPatch, *, seed: str,
         revoke: bool = False, dispatch: bool = False) -> Tuple[str, str]:
    RIG.clear()
    RIG["dispatch"] = dispatch
    key = "cg-key-v1-" + (seed * 56)[:56]
    ref = f"root/family-{FAMILY}#{key}"
    monkeypatch.setattr(
        fleet_cells, "enable_compiled",
        _fake_adopt_arm(key, ref, revoke=revoke))
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
          ref: str = "acme/micro-diffusion@prod") -> Tuple[Executor, Any]:
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    ex = Executor(extract_specs(AdoptedFamily), _send)
    ex.store._cache_dir = tmp_path / "cas"

    async def _fake_download(target: str, **_kw: Any) -> Path:
        p = tmp_path / target.replace("/", "_").replace(":", "_")
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(store_mod, "ensure_local", _fake_download)

    from gen_worker import dispatch as dispatch_mod

    spec = ex.specs["generate"]
    eff = ex._dispatched_spec(
        spec, {"pipeline": dispatch_mod.SlotOrder(ref=ref, facts=_NO_FACTS)})
    snaps = {normalize_model_ref(ref): pb.Snapshot(
        digest="d1" * 16,
        files=[pb.SnapshotFile(
            path="model.safetensors", size_bytes=5, blake3="cd" * 32,
            url="http://r2.invalid/presigned")])}
    asyncio.run(ex.ensure_setup(eff, snaps))
    return ex, eff


def _phases(said: List[Tuple[str, str, str]], kind: str) -> List[str]:
    return [phase for k, phase, _d in said if k == kind]


def test_an_ADOPTED_cell_serves_COMPILED_immediately_after_materialize(
        tmp_path, monkeypatch, spy):
    """THE HEADLINE RED, and on master it fails twice over — once for the
    warmup barrier (§4.31) and once for the adopt-side gate (§4.32).

    The pod's own reading, reproduced: the cell arms before setup, the boot
    warmup dispatches nothing through it, and pre-fix that made it
    `unexercised` -> `unproven` -> unwrap -> quarantine, `functions=()`,
    `target_applicability_incomplete`, `armed_target_unresolved`, eager for
    life. Now: armed, dispatchable, serving compiled, with NO quality
    measurement taken anywhere on this pod."""
    _key, ref = _rig(monkeypatch, seed="a")
    ex, eff = _boot(tmp_path, monkeypatch)
    pipe = RIG["pipe"]

    assert aot_serve.is_armed(pipe) is True, (
        "the pod threw away the cell it had just materialized")
    assert getattr(pipe.unet, aot_serve._MARKER_ATTR, None) is not None
    assert not compile_cache.compiled_graph_quarantined_in_process(ref)

    # ...and the arm is DISPATCHABLE, which is the half the pod actually lost:
    # an empty `function_proofs` installs no target and the boot still ends
    # `boot_ended_uncompiled` however armed the object is.
    rec = ex._classes[eff.instance_key]
    assert rec.compile_targets, "armed and still no installed compile target"
    target = next(iter(rec.compile_targets.values()))
    assert "generate" in target.function_names
    assert rec.eager_posture == ""
    assert cell_adopt.EagerPhase.ARMED_TARGET_UNRESOLVED.value not in _phases(
        spy, "serve_eager_posture")
    assert "target_applicability_incomplete" not in _phases(
        spy, "serve_eager_posture")
    assert ex._served_execution_lane(eff).endswith("+compiled")

    # §4.32: adoption measured nothing. The only compiled_graph_numerics row a boot like
    # this may emit is the posture one — never a verdict.
    rows = [(phase, detail) for kind, phase, detail in spy
            if kind == activity.KIND_COMPILED_GRAPH_NUMERICS]
    assert [p for p, _d in rows] == [
        numerics_ladder.PHASE_ARMED_UNDISPATCHED], rows
    assert "STAYS ARMED" in rows[0][1]
    assert "adoption runs no quality gate" in rows[0][1]


def test_a_REVOKED_artifact_is_not_re_armed_by_the_setup_pass(
        tmp_path, monkeypatch, spy):
    """The de-arm is STICKY (§4.31). An artifact the wrapper already revoked —
    it ran and failed — is not resurrected by anything downstream, and the boot
    lands on eager with its target omitted rather than advertising a lane that
    cannot serve."""
    _key, ref = _rig(monkeypatch, seed="c", revoke=True)
    ex, eff = _boot(tmp_path, monkeypatch)
    pipe = RIG["pipe"]

    assert aot_serve.is_armed(pipe) is False
    assert not compile_cache.compiled_graph_proven_in_process(ref)
    rec = ex._classes[eff.instance_key]
    assert not rec.compile_targets, (
        "a revoked artifact kept an installed target, so the wire would say "
        "aot_cell on a pipeline whose every call runs eager")
    # The revocation is NAMED on the wire, at the install that refused it.
    # (`rec.eager_posture` settles on the terminal `armed_target_unresolved`
    # from `_assert_armed_targets_installed`, which is the orphan report for
    # the same object; both rows are emitted and neither is silence.)
    assert cell_adopt.EagerPhase.COMPILED_DEGRADED.value in _phases(
        spy, "serve_eager_posture")
    assert rec.eager_posture


# ---------------------------------------------------------------------------
# PART D — try-serve: the first real request is the proof, and what it blames
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
        module, runner, {"family": FAMILY, "compiled_graph_key": "ck1-x"},
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
    """Symmetry, from the other side. A warm dispatch that DOES land still
    records the cell proven in-process (the pgw#637 registry the dynamo lane
    reads), and emits no posture row — there is nothing to explain."""
    _key, ref = _rig(monkeypatch, seed="d", dispatch=True)
    ex, eff = _boot(tmp_path, monkeypatch)
    pipe = RIG["pipe"]

    assert aot_serve.is_armed(pipe) is True
    assert aot_serve.execution_count(pipe) > PROBE_CALLS
    assert compile_cache.compiled_graph_proven_in_process(ref)
    rec = ex._classes[eff.instance_key]
    assert rec.compile_targets
    assert [phase for kind, phase, _d in spy
            if kind == activity.KIND_COMPILED_GRAPH_NUMERICS] == []
