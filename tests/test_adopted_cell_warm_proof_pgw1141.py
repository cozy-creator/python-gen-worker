"""pgw#1141 (DESIGN-RULINGS §4.31 + §4.32): a boot-ADOPTED cell materializes,
arms and SERVES. No warmup barrier, and no quality gate at adoption.

MEASURED, twice, on two real pods (RTX 4000 Ada, gen-worker 0.106.0, hub
`7ae35d54a2`): `boot_adopt=hit` -> materialize -> `cell_numerics cos=1.00000
ret=1.0000 rel_l2=0.0000` on 3/3 axes -> and then the setup warmup scored that
same artifact `unexercised` (it dispatched nothing through it — the adopt arms
BEFORE setup, so by construction nothing has), folded it into `unproven`, wrote
`function_proofs[id]=set()` and unwrapped it. `functions=()` then made the
target omission (`target_applicability_incomplete`) and the orphan report
(`armed_target_unresolved`) inevitable, and the pod served eager for life. The
SELF-MINT arm was healthy on the same wheel, card and release, because a mint's
warmup DRIVES its own capture and therefore dispatches.

§4.31 deleted the barrier: *"skip the warmup / arm check, so we can serve right
away … try to serve, if an error is encountered, and it's the cause of the
cell, de-arm the cell, and serve eager instead."*

§4.32 then deleted the adopt-side numerics re-check too, and moved the quality
question to where the defect is: every failure that gate ever caught (a baked
`conv_out.bias`, timestep dtype scars) was an AUTHOR defect in endpoint code or
config. Re-measuring on every adopter taxes the fleet forever for one author's
one-time mistake. Adoption is materialize -> arm -> serve; the gate runs ONCE,
on the pod that minted the bytes, before they are published, and it is STRICT.

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
real packed artifact (the pgw#868 rig, whose ONE substitution is the AOTI
`.so`); part C drives the REAL `ensure_setup`, faking only the download and the
arming policy (the seam `test_aot_boot_proof_gap_pgw735.py` uses); part D
drives the real serving wrapper.
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
from gen_worker import activity, aot_serve, cell_adopt, cell_key, compile_cache, fleet_cells
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
    rows = [(p, d) for k, d, p in events if k == activity.KIND_CELL_NUMERICS]
    # pgw#1176: ONE row per graph class — the gate runs at the moment that
    # entry exists, never "after all N" (DESIGN-RULINGS 4.32).
    assert [p for p, _d in rows] == ["checked", "checked"], rows
    # pgw#1176: ONE axis per artifact, so the report is 1/1 twice, never
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
    # pgw#1176: the refusal is per CLASS, so it says "is not published" of
    # that class rather than "nothing is published" of a bundle.
    assert "is not published" in outcome.detail
    # It still CONFESSES — a fleet-wide rate is only countable from rows.
    # pgw#1176: one gate row PER GRAPH CLASS — this declaration has two,
    # so the vocabulary repeats rather than aggregating.
    assert [p for k, _d, p in events
            if k == activity.KIND_CELL_NUMERICS] == ["degraded", "degraded"]


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
    # pgw#1176: one gate row PER GRAPH CLASS — this declaration has two,
    # so the vocabulary repeats rather than aggregating.
    assert [p for k, _d, p in events
            if k == activity.KIND_CELL_NUMERICS] == ["refused", "refused"]


def _pending(tmp_path: Path, decl: Any, publisher: Any = None):
    """A real `PendingSelfMint` pointing at a real packed cell."""
    target = tmp_path / "mint" / "cell.tar.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    return fleet_cells.PendingSelfMint(
        family=rig868.FAMILY, arm_token="arm1-" + "a" * 24,
        ref=f"root/family-{rig868.FAMILY}#cell868",
        cfg=decl, target=target, mint_root=tmp_path / "mint",
        publisher=publisher, cache_dir=tmp_path / "cache")


def _delegated_mint(tmp_path, monkeypatch, decl, packages, events):
    """Drive the REAL `adopt_delegated_mint` — the mint's publish gate — with
    the REAL `arm_aot` and the REAL numerics gate underneath it."""
    from gen_worker import aot_serve as aot

    monkeypatch.setattr(aot, "runtime_key", lambda: dict(rig868.RUNTIME))
    monkeypatch.setattr(aot, "_entry_admission_drift", lambda *a, **k: None)
    monkeypatch.setattr(aot, "_load_package", lambda path, entry="model": packages[entry])
    monkeypatch.setattr(fleet_cells, "arm_axis_divergence", lambda a, m, **_kw: "")
    monkeypatch.setattr(fleet_cells.activity_mod, "emit_event",
                        lambda kind, detail="", **kw: events.append(
                            (kind, detail, str(kw.get("phase", "")))))
    pending = _pending(tmp_path, decl)
    built = rig868.artifact(tmp_path)
    pending.target.write_bytes(Path(built).read_bytes())
    module = rig868.ProbeDenoiser()
    pipeline = rig868.ProbePipeline(module)
    return fleet_cells.adopt_delegated_mint(pipeline, pending, [pending.target])


def test_a_DIVERGENT_cell_is_not_published_by_the_pod_that_minted_it(
        tmp_path, monkeypatch, declared, events):
    """§4.32 item 2, end to end through the REAL publish gate.

    `publish_self_mint` can only ship what `adopt_delegated_mint` marked
    minted, so a refusal here IS the publish refusal.

    HONEST STATUS: this row is GREEN on master too, and that is the point of
    writing it. The move is a MOVE — the property "a divergent cell does not
    publish" must survive it unbroken, because §4.32's sequencing rule is that
    at no commit may zero parity gates exist. Master satisfied it by accident
    of placement (the gate lived in `arm_aot`, and a mint arms too, so the
    mint path inherited a check that was really aimed at adopters); this commit
    satisfies it on purpose, with the check aimed at the mint and nothing left
    on the adopt path. The rows that go RED are the ones that distinguish those
    two worlds: the gray band (master ships it), and every adopt row."""
    packages = {rig868.entry_name(h, w): ProbePackage(cosine=0.99)
                for h, w in ROWS}
    minted = _delegated_mint(tmp_path, monkeypatch, declared, packages, events)

    assert minted is None, "a cell that does not reproduce eager was published"
    # Typed, and the hub can count it: the ladder's own refusal plus the mint's
    # abort, which is the row that says nothing shipped.
    assert ("refused" in [p for k, _d, p in events
                          if k == activity.KIND_CELL_NUMERICS])
    aborts = [(d, p) for k, d, p in events if k == "self_mint_abort"]
    assert aborts, "the mint published nothing and said nothing"
    assert aborts[-1][1] == "numerics_refused"
    assert "nothing is published" in aborts[-1][0]


def test_a_FAITHFUL_cell_passes_the_mint_gate_and_is_publishable(
        tmp_path, monkeypatch, declared, events):
    """The control, without which the row above could pass for the wrong
    reason (an unreadable envelope, a divergence gate, a missing stamp)."""
    packages = {rig868.entry_name(h, w): ProbePackage() for h, w in ROWS}
    minted = _delegated_mint(tmp_path, monkeypatch, declared, packages, events)

    assert minted is not None, "a faithful cell was refused by the mint gate"
    # pgw#1176: the key is COMPUTED from the artifact's own recorded facts, not
    # a fixture placeholder. `"cell868"` was a stand-in from when the harness
    # stamped a literal; asserting it now would assert that the mint FAILED to
    # key its own product.
    assert cell_key.is_key(minted.cell_key), minted.cell_key
    # pgw#1176: ONE gate row here, and the count is load-bearing rather than
    # incidental — the DELEGATED mint adopts a single entry artifact, so one
    # class is gated. (My own "one row per class" sweep over-applied to this
    # row and expected two; the rows that DO see two arm two artifacts. A
    # blanket edit is a sweep, and sweeps damage the case that is different.)
    assert [p for k, _d, p in events
            if k == activity.KIND_CELL_NUMERICS] == ["checked"]


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
    assert [k for k, _d, _p in events if k == activity.KIND_CELL_NUMERICS] == [], (
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
                # pgw#1176: `execution_count` sums the RUNNERS' own calls, so a
                # dispatch that only bumped the state counter moved a number
                # production does not read. Move the artifact's own.
                for row in (marker.get("targets") or {}).values():
                    runner = (row.get("state") or {}).get("runner")
                    for _name, art in getattr(runner, "runners", ()) or ():
                        art.calls += 1
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
        # pgw#1176: production's `arm_entry` puts an `EntryDispatch` REGISTRY
        # in `state["runner"]`, and `is_armed`/`armed_entries` read it — a
        # boolean was deleted precisely because it can claim more than the pod
        # serves. A double that sets the marker but no registry models a
        # pipeline production cannot produce, and every `is_armed` assertion
        # against it is testing a shape that does not exist.
        _runner = aot_serve.ArtifactRunner(
            package=None,
            contract=aot_serve.ArtifactContract(inputs=(), symbols={}),
            constants=(), module_name="unet", entry="unet/main")
        _runner.calls = PROBE_CALLS
        _dispatch = aot_serve.EntryDispatch(declared=("unet/main",))
        _dispatch.add("unet/main", _runner)
        state = {"successful_calls": PROBE_CALLS, "failed": False,
                 "original": unet.forward, "runner": _dispatch}
        # pgw#1176: the two markers are DIFFERENT SHAPES in production and
        # this rig now models that honestly. `wrap_module` writes a bare
        # `state` on the MODULE; `arm_entry` writes `targets` (+ `entries`) on
        # the PIPELINE. Sharing one dict between them was what kept a
        # `_marker_states` fallback alive for a shape nothing produces.
        setattr(unet, aot_serve._MARKER_ATTR, {
            "meta": {"cell_key": key}, "state": state})
        setattr(pipe, aot_serve._MARKER_ATTR, {
            "meta": {"cell_key": key},
            "targets": {"unet": {
                "module": unet, "attr": "forward", "state": state}},
            "entries": {"unet/main": {"key": key, "target": "unet"}},
        })
        marker = getattr(pipe, aot_serve._MARKER_ATTR)
        # pgw#1152: an `aot_serve.note_aot_key(key)` stood here — the ONE line no
        # production arm route ever called, which is why these rows were green
        # while the pod served eager (pgw#1141b). It is DELETED, not moved: the
        # marker set above is what `arm_entry` publishes, so
        # `holds_exported_cell` answers the lane question off the OBJECT.
        # A fixture that needs a REAL boot-adopt drives tests/harness/adopt_rig.py.
        if revoke:
            # pgw#1176: revoking is DE-ARMING. `is_armed` asks the registry,
            # so a flag left the runner armed and the pipeline claiming
            # compiled service it was not giving — the exact lie a cell-level
            # boolean makes possible.
            state["failed"] = True
            _dispatch.remove("unet/main", "revoked by the rig")
        adopted = fleet_cells.SelfMint(
            family=FAMILY, cell_key=key, ref=ref,
            snapshot_digest="blake3:" + "ab" * 32,
            artifact=Path(cache_dir or ".") / "cell.tar")
        return fleet_cells.ArmOutcome(armed=True, self_mint=adopted)

    return _enable


def _rig(monkeypatch: pytest.MonkeyPatch, *, seed: str,
         revoke: bool = False, dispatch: bool = False) -> Tuple[str, str]:
    RIG.clear()
    RIG["dispatch"] = dispatch
    key = "ek1-" + (seed * 56)[:56]
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
    assert not compile_cache.cell_quarantined_in_process(ref)

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

    # §4.32: adoption measured nothing. The only cell_numerics row a boot like
    # this may emit is the posture one — never a verdict.
    rows = [(phase, detail) for kind, phase, detail in spy
            if kind == activity.KIND_CELL_NUMERICS]
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
    assert not compile_cache.cell_proven_in_process(ref)
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
    """Symmetry, from the other side. A warm dispatch that DOES land still
    records the cell proven in-process (the pgw#637 registry the dynamo lane
    reads), and emits no posture row — there is nothing to explain."""
    _key, ref = _rig(monkeypatch, seed="d", dispatch=True)
    ex, eff = _boot(tmp_path, monkeypatch)
    pipe = RIG["pipe"]

    assert aot_serve.is_armed(pipe) is True
    assert aot_serve.execution_count(pipe) > PROBE_CALLS
    assert compile_cache.cell_proven_in_process(ref)
    rec = ex._classes[eff.instance_key]
    assert rec.compile_targets
    assert [phase for kind, phase, _d in spy
            if kind == activity.KIND_CELL_NUMERICS] == []
