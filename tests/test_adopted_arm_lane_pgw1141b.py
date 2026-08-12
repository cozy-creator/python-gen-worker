"""pgw#1141b: a BOOT-ADOPTED cell is on the EXPORTED lane, and the setup pass
has to know that from the OBJECT — not from a process-global set of cell keys
somebody remembered to announce.

MEASURED on a real pod (RTX A4500, sm_86, gen-worker **0.111.0** — the wheel
carrying pgw#1141's fix — POD PROOF #4 in pgw#1108, first try)::

    seq  3  boot_adopt           hit                    key=ck1-329a6fbe… 10 291 ms
    seq 12  cell_numerics        armed_undispatched     "It STAYS ARMED and serves…"
    seq 13  serve_eager_posture  target_applicability_incomplete
              functions=() bindings=(('pipeline','tensorhub/micro-diffusion',…))
    seq 14  serve_eager_posture  armed_target_unresolved  armed=False targets_resolve=True
    seq 15  serve_degrade        armed_target_unresolved
    seq 17  self_mint_skipped    boot_ended_uncompiled

pgw#1141's half WORKS — seq 12 is 0.111.0's own new sentence, so the warmup
barrier really is deleted. And 677 ms later the object was unwrapped anyway.

THE LOCUS. ``aot_serve.is_aot_ref`` answers out of ``_KNOWN_AOT_KEYS``, a
process-global set fed by ``note_aot_key``. Before this issue it had exactly
two feeders, ``fleet_cells.arm_from_local_store`` and
``fleet_cells.adopt_delegated_mint`` — both SELF-PRODUCED routes. The ORDERED
arm (``fleet_cells.arm_ordered``: every hub Plan, and §4.27 boot-adopt) fed it
NOTHING. So for a boot-adopted cell, on the pod:

* ``_proves_by_fx(ref)`` was True -> the artifact went into ``proof_before``,
  the DYNAMO lane's cache-hit ledger, which an AOTI artifact can never move;
* ``aot_proof_before`` was EMPTY -> §4.31's "an exported arm keeps its arm"
  branch could not fire for the one object it exists for;
* the object scored calls=0, was folded into ``unproven``, got
  ``function_proofs[id]=set()`` and was unwrapped;
* ``exported_arm`` was then False at the install too, so ``permitted_names``
  came from that empty proof set -> ``functions=()`` ->
  ``target_applicability_incomplete`` -> ``armed_target_unresolved`` -> eager.

pgw#1141's own tests could not see it: their fleet-policy stand-ins call
``aot_serve.note_aot_key(key)`` BY HAND (``test_adopted_cell_warm_proof_
pgw1141.py`` ``_fake_adopt_arm``, ``test_aot_boot_proof_gap_pgw735.py``
``_fake_arm``), which is the one thing no production arm route did. The tests
entered one gate east of the bug — twice.

THE FIX, in two parts, because one of them is a convention and the other is a
structure:

1. ``aot_serve.load_and_wrap`` registers the key AT THE WRAP — the single seam
   every arm route passes, and the moment the fact becomes true;
2. the executor's three lane readers ask the OBJECT
   (``aot_serve.holds_exported_cell`` via ``executor._exported_arm``), so the
   disarm authority cannot be exercised over an object carrying a live cell
   even if no registry ever learned its key.

WHAT RUNS FOR REAL HERE. The whole chain from the ordered arm to the install:
``Executor.ensure_setup`` -> ``_injection_kwargs`` -> ``_enable_compiled`` with
a real ``_ArmOrder(adopt=…)`` -> ``fleet_cells.arm_ordered`` -> the real
receipt gate against a real RSA-signed receipt from a real HTTP hub ->
``provision.arm_aot`` -> ``aot_serve.load_and_wrap`` on a real packed artifact
-> the real boot warmup -> the real proof pass -> the real
``_install_compile_targets`` and ``_assert_armed_targets_installed``.

THREE SEAMS, all WEST of the locus and all named:

* ``Executor._boot_adopt`` returns a constructed HIT. Its derive+resolve half
  is driven end to end for real by ``test_boot_adopt_observability_pgw1116.py``
  against ``examples/micro-diffusion`` and a real hub; repeating it here would
  measure that, not this.
* ``provision.load_slot`` returns the probe pipeline instead of reading
  weights off disk (the class-annotated slot shape micro-diffusion uses —
  ``Slot(MicroPipeline, selected_by="model")`` — which is what routes the arm
  order to ``_enable_compiled`` at all).
* ``aot_serve._load_package``: an AOTI ``.so`` needs a GPU. pgw#868's rig owns
  this substitution and it is the only piece deferred to a pod.

Nothing about applicability, the lane split, the proof pass or the target
install is stubbed — those are the code under test.

Run: uv run pytest tests/test_adopted_arm_lane_pgw1141b.py -q
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Tuple

import msgspec
import pytest

from gen_worker import (
    RequestContext, Resources, Slot, activity, aot_identity, aot_serve,
    boot_adopt, cell_adopt, cell_key, cell_resolve, endpoint,
    env_seal, receipts, worker_function,
)
from gen_worker import executor as ex_mod
from gen_worker.executor import Executor
from gen_worker.models import provision
from gen_worker.models.refs import normalize_model_ref
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs

# The REAL artifact/arm rig: a real packed cell, a real declaration, real
# torch tensors. Its ONE substitution is the AOTI `.so`.
import test_numerics_gate_pgw868 as rig868  # noqa: E402
from test_numerics_gate_pgw868 import (  # noqa: E402
    FAMILY, ROWS, RUNTIME, ProbeDenoiser, ProbePackage, ProbePipeline,
    declaration, entry_name,
)
# The REAL receipt gate: a real RSA key, a real JWKS/receipt HTTP hub.
from test_receipts_pgw709 import (  # noqa: F401,E402 — fixtures come with it
    HubStub, hub, pub_map, rsa_key,
)

#: The publishing org. Platform tier, so the trust rule needs no viewer
#: identity — the identity half is pgw#1122's, fenced there.
ORG = "11111111-2222-3333-4444-555555555555"

MODEL_REF = "acme/micro-probe:prod"

#: Everything the endpoint instance and the load seam hand each other.
RIG: Dict[str, Any] = {}


class GenIn(msgspec.Struct):
    prompt: str = "warm"


class Out(msgspec.Struct):
    y: str = "ok"


#: pgw#868's declaration with the one field a `@endpoint` lint requires that a
#: bare `provision.arm_aot` rig never needed: the probe denoiser is
#: unconditioned, so the text axis is explicitly absent (ie#544).
CELL = msgspec.structs.replace(declaration(), text_len=0)


class AdoptedPipeline(ProbePipeline):
    """A WORKER-LOADED slot class — the annotation shape that routes the arm
    order into ``_enable_compiled``. ``from_pretrained`` is what the executor
    tests for; the load itself is the named seam (``provision.load_slot``)."""

    def __init__(self) -> None:
        super().__init__(ProbeDenoiser())

    @classmethod
    def from_pretrained(cls, *_a: Any, **_k: Any) -> "AdoptedPipeline":
        raise AssertionError("provision.load_slot is the seam, not this")


@endpoint(
    models={"pipeline": Slot(AdoptedPipeline)},
    resources=Resources(gpu=True),
    compile=CELL,
)
class AdoptedFamily:
    """A CLASS-annotated slot, which is micro-diffusion's shape and the reason
    the arm order reaches ``_enable_compiled`` at all: the arming-scope path a
    ``Slot(str)`` endpoint takes never sees an ``_ArmOrder``."""

    def setup(self, pipeline: AdoptedPipeline) -> None:
        self.pipe = pipeline

    @worker_function()
    def generate(self, ctx: RequestContext, p: GenIn) -> Out:
        # The pod's shape: the boot warmup runs and lands on no packaged entry
        # of the adopted cell, so the artifact's own counter does not move.
        return Out()


# ---------------------------------------------------------------------------
# the cell: a real packed artifact with a real, fully-stated identity
# ---------------------------------------------------------------------------


def _metadata() -> Dict[str, Any]:
    """pgw#868's envelope plus the two identity blocks an ADOPTED cell must
    carry: ``verify_declared_identity`` compares four axes and refuses a cell
    that is SILENT on any of them."""
    meta = rig868.metadata()
    meta["toolchain"] = {"torch": RUNTIME["torch"], "cuda": RUNTIME["cuda"],
                         "triton": "3.6.0"}
    meta[env_seal.SEAL_KEY] = {"PYTHONHASHSEED": "0", "TORCH_COMPILE_DEBUG": ""}
    meta[cell_key.EXPORT_ENVELOPE_KEY] = {
        "shapes": [list(row) for row in ROWS], "text_len": 0,
        "shape_strategy": "static-rows",
    }
    # The REAL key, restated from the artifact's own recorded facts — the same
    # recomputation admission runs (pgw#1059), so nothing here is a stamp the
    # bytes cannot back up.
    meta["cell_key"] = cell_key.from_exported_artifact_metadata(meta).digest
    return meta


def _resolved(meta: Dict[str, Any]) -> cell_resolve.ResolvedCell:
    """The hub's answer, stating exactly the identity the mint stamped."""
    have = aot_identity.artifact_identity(meta)
    return cell_resolve.ResolvedCell(
        family=FAMILY, cell_key=have.cell_key,
        cell_ref=f"root/family-{FAMILY}#{have.cell_key}",
        checkpoint_id="", content_digest="sha256:" + "ab" * 32,
        artifact_path="cell.tar.gz", size_bytes=0,
        publisher_org=ORG, publisher_tier="platform",
        graph_contract=have.graph_contract_digest,
        toolchain_digest=have.toolchain_digest,
        env_seal_digest=have.env_seal_digest,
        identity_axes={}, sm=RUNTIME["sm"], sku=RUNTIME["sku"], lane="",
        receipt="",
        transport=cell_resolve.Transport(
            snapshot_digest="blake3:" + "cd" * 32, files=()),
    )


def _derived(digest: str) -> Any:
    from gen_worker import boot_key

    return boot_key.DerivedKey(
        key=cell_key.CellKey(axes=(("adopted", digest),)),
        class_hashes={}, combined="", workers=1, width_reason="test",
        traced=len(ROWS), memo="miss", wall_ms=10_291,
    )


# ---------------------------------------------------------------------------
# the rig
# ---------------------------------------------------------------------------


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, str, str]]:
    """The typed rows, captured at the ONE sink the hub's
    ``worker_activity_events`` table is built from."""
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


def phases(said: List[Tuple[str, str, str]], kind: str) -> List[str]:
    return [phase for k, phase, _d in said if k == kind]


@pytest.fixture
def forget_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """``note_aot_key`` learns into a PROCESS-global set, so a sibling file
    that armed the same key would answer this file's question for it. Every
    row here starts with a runtime that has been told nothing."""
    monkeypatch.setattr(aot_serve, "_KNOWN_AOT_KEYS", set())


@pytest.fixture
def adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: HubStub,  # noqa: F811
    forget_keys: None,
) -> Dict[str, Any]:
    """A boot that RESOLVED a cell by its own derived key and is about to arm
    it — everything from the ``_ArmOrder`` down runs for real."""
    RIG.clear()
    meta = _metadata()
    artifact = rig868.artifact(tmp_path, meta)
    cell = _resolved(meta)

    # The hub countersigns these exact bytes. Platform tier: adoptable by any
    # pod, so the arm asks nobody who it is (pgw#1122 owns that direction).
    hub.serve_receipt_for(
        artifact, cell_key=cell.cell_key, family=FAMILY,
        publisher_tier="platform", publisher_org_id=ORG,
        owning_endpoint_id="")
    receipts.configure(base_url=hub.base_url, worker_jwt=lambda: "")

    # The one deferred piece (GPU) and the runtime axes a cardless box cannot
    # state — pgw#868's substitutions, verbatim.
    monkeypatch.setattr(aot_serve, "runtime_key", lambda: dict(RUNTIME))
    monkeypatch.setattr(aot_serve, "_entry_admission_drift",
                        lambda *a, **k: None)
    packages = {entry_name(h, w): ProbePackage() for h, w in ROWS}
    monkeypatch.setattr(
        aot_serve, "_load_package", lambda path, entry="model": packages[entry])

    # The class-annotated slot's weights load. `_inject_models` arms whatever
    # this returns, through the real ordered path.
    def _load_slot(annotation: Any, path: str, **_kw: Any) -> provision.SlotLoad:
        pipe = AdoptedPipeline()
        RIG["pipe"] = pipe
        return provision.SlotLoad(obj=pipe, is_pipeline=True, ran="bf16")

    monkeypatch.setattr(provision, "load_slot", _load_slot)
    monkeypatch.setattr(ex_mod.provision, "load_slot", _load_slot)

    def _adopt(self: Any, spec: Any, slots: Any) -> boot_adopt.BootAdoptOutcome:
        return boot_adopt.report(boot_adopt.BootAdoptOutcome(
            adoption=boot_adopt.BootAdoption(
                derived=_derived(cell.cell_key), cell=cell, artifact=artifact),
            reason=boot_adopt.HIT, derived_key=cell.cell_key, derive_ms=10_291,
            family=FAMILY, function="generate"))

    monkeypatch.setattr(Executor, "_boot_adopt", _adopt)
    RIG["meta"], RIG["cell"], RIG["artifact"] = meta, cell, artifact
    return RIG


def _boot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Tuple[Executor, Any]:
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
        spec, {"pipeline": dispatch_mod.SlotOrder(ref=MODEL_REF, components=())})
    snaps = {normalize_model_ref(MODEL_REF): pb.Snapshot(
        digest="d1" * 16,
        files=[pb.SnapshotFile(
            path="model.safetensors", size_bytes=5, blake3="cd" * 32,
            url="http://r2.invalid/presigned")])}
    asyncio.run(ex.ensure_setup(eff, snaps))
    return ex, eff


# ===========================================================================
# 1. THE HEADLINE RED — the pod's four-event chain, reproduced off-pod
# ===========================================================================


def test_a_boot_adopted_cell_is_NOT_scored_on_the_dynamo_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, adopted: Dict[str, Any],
    events: List[Tuple[str, str, str]],
) -> None:
    """THE RED. On master this reproduces POD PROOF #4's chain exactly:
    ``target_applicability_incomplete functions=()`` then
    ``armed_target_unresolved`` then ``serve_degrade``, and the pod serves
    eager for life having thrown away the cell it just materialized.

    The four assertions below are the four rows the pod emitted, in order.
    """
    ex, eff = _boot(tmp_path, monkeypatch)
    pipe = RIG["pipe"]

    assert "hit" in phases(events, activity.KIND_BOOT_ADOPT), (
        "the boot did not adopt at all — this row is testing the wrong thing")

    # seq 13: the object's applicability. `functions=()` is where the pod died.
    assert "target_applicability_incomplete" not in phases(
        events, "serve_eager_posture"), (
        "the boot-adopted cell computed EMPTY applicability — the ordered arm "
        "route never taught `is_aot_ref` its key, so the exported cell was "
        "scored on the dynamo lane's cache-hit ledger and disarmed")
    # seq 14 + 15: the orphan report and the degrade.
    assert cell_adopt.EagerPhase.ARMED_TARGET_UNRESOLVED.value not in phases(
        events, "serve_eager_posture")
    assert not [k for k, _p, _d in events if k == activity.KIND_SERVE_DEGRADE], (
        "a cell that materialized, armed and was never dispatched through "
        "degraded the whole record")

    # ...and the positive statement of the same fact: it SERVES.
    assert aot_serve.is_armed(pipe) is True
    rec = ex._classes[eff.instance_key]
    assert rec.compile_targets, "armed and still no installed compile target"
    target = next(iter(rec.compile_targets.values()))
    assert "generate" in target.function_names
    assert target.active_compile_ref == RIG["cell"].cell_ref
    assert rec.eager_posture == ""
    assert ex._served_execution_lane(eff).endswith("+compiled")


def test_the_ordered_arm_teaches_the_recognizer_its_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, adopted: Dict[str, Any],
) -> None:
    """THE LOCUS, stated as one fact. ``note_aot_key`` had two feeders and both
    were self-produced routes; the ordered arm — every hub Plan and every
    §4.27 boot-adopt — fed it nothing, so an adopted cell's own ref answered
    "not an AOT cell" on the pod that was serving it.
    """
    _boot(tmp_path, monkeypatch)

    ref = RIG["cell"].cell_ref
    assert aot_serve.is_aot_ref(ref), (
        "the process armed this exact artifact and still does not recognize "
        "its ref as an exported cell")
    assert aot_serve.is_aot_ref(ref, FAMILY)


def test_the_lane_is_read_off_the_OBJECT_not_a_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, adopted: Dict[str, Any],
) -> None:
    """The structural half. Registering the key is a convention a future arm
    route can forget again — twice now a disarm authority has survived one gate
    east of its fix. With the registry deliberately emptied AFTER the wrap, the
    readers must still put this object on the exported lane."""
    ex, eff = _boot(tmp_path, monkeypatch)
    pipe = RIG["pipe"]

    monkeypatch.setattr(aot_serve, "_KNOWN_AOT_KEYS", set())
    assert not aot_serve.is_aot_ref(RIG["cell"].cell_ref)
    assert aot_serve.holds_exported_cell(pipe) is True
    assert ex_mod._exported_arm(pipe, RIG["cell"].cell_ref) is True


def test_the_posture_row_names_the_adoption_not_the_dynamo_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, adopted: Dict[str, Any],
    events: List[Tuple[str, str, str]],
) -> None:
    """seq 12 on the pod carried §4.31's new sentence, which is why the leg
    read as "pgw#1141 works". It did — but the reason under it was the DYNAMO
    lane's ("this boot holds no evidence either way"), because the cell had
    been sorted onto that lane. An adopted cell's row must say what it is."""
    _boot(tmp_path, monkeypatch)

    rows = [d for k, _p, d in events if k == activity.KIND_CELL_NUMERICS]
    assert len(rows) == 1, rows
    assert "STAYS ARMED" in rows[0]
    assert "adoption runs no quality gate" in rows[0], (
        "the adopted cell was confessed under the dynamo lane's reason, which "
        "is the sorting error this issue is about")


# ===========================================================================
# 2. THE NEGATIVE — the honest de-arm paths keep their teeth
# ===========================================================================


def test_a_REVOKED_cell_still_de_arms_and_installs_no_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, adopted: Dict[str, Any],
    events: List[Tuple[str, str, str]],
) -> None:
    """§4.31's sticky de-arm, unchanged. The artifact ran and failed before any
    guard was bound, so nothing may advertise ``serving_mode=aot_cell`` for it.
    A fix that made a cell undisarmable would be worse than the bug."""
    real_wrap = aot_serve.load_and_wrap

    def _wrap_then_revoke(pipeline: Any, *a: Any, **k: Any) -> Any:
        meta = real_wrap(pipeline, *a, **k)
        for state in aot_serve._marker_states(pipeline):
            state["failed"] = True
        return meta

    monkeypatch.setattr(aot_serve, "load_and_wrap", _wrap_then_revoke)

    ex, eff = _boot(tmp_path, monkeypatch)
    pipe = RIG["pipe"]

    assert aot_serve.is_armed(pipe) is False
    rec = ex._classes[eff.instance_key]
    assert not rec.compile_targets, (
        "a revoked cell kept an installed target, so the wire would say "
        "aot_cell on a pipeline whose every call runs eager")
    assert cell_adopt.EagerPhase.COMPILED_DEGRADED.value in phases(
        events, "serve_eager_posture")
    assert rec.eager_posture


def test_an_operator_eager_only_order_still_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, adopted: Dict[str, Any],
) -> None:
    """§4.32 item 4: the consumer can always opt out of compiled serving. The
    exported-lane recognition must not route around the operator's order."""
    from gen_worker import serve_posture

    serve_posture.apply_command(True, actor="operator", reason="pgw#1141b row")
    try:
        ex, eff = _boot(tmp_path, monkeypatch)
        rec = ex._classes[eff.instance_key]
        # Nothing armed and nothing adopted: the order is obeyed at the
        # arming brain, before the lane question is ever asked.
        assert aot_serve.is_armed(RIG["pipe"]) is False
        assert aot_serve.holds_exported_cell(RIG["pipe"]) is False
        assert rec.eager_posture == cell_adopt.EagerPhase.OPERATOR_EAGER_ONLY
        assert ex.serving_tiers()["generate"] == "eager"
        assert not ex._served_execution_lane(eff).endswith("+compiled")
    finally:
        serve_posture.reset()


# ===========================================================================
# 3. THE FENCE — every route that wraps an exported cell feeds ONE recognizer
# ===========================================================================


def test_the_registration_lives_at_the_wrap_not_at_the_call_sites() -> None:
    """The fence against a FOURTH reader. ``note_aot_key`` was a convention
    ("whoever reads a cell_key off an aot-inductor envelope registers it") and
    the ordered arm simply did not keep it. Registration now happens inside
    ``load_and_wrap``, the one function every arm route passes, so a new route
    inherits it instead of having to remember it."""
    import inspect

    body = inspect.getsource(aot_serve.load_and_wrap)
    assert "note_aot_key(" in body, (
        "the wrap no longer registers the key it just armed; a new arm route "
        "is one convention away from pgw#1141b happening again")


def test_a_wrapped_object_answers_the_lane_question_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, adopted: Dict[str, Any],
) -> None:
    """``holds_exported_cell`` is deliberately NOT ``is_armed``: the install
    path must tell a REVOKED exported cell (never advertise it) apart from an
    object carrying no cell at all (an ordinary dynamo/eager object)."""
    _boot(tmp_path, monkeypatch)
    pipe = RIG["pipe"]

    assert aot_serve.holds_exported_cell(pipe) is True
    for state in aot_serve._marker_states(pipe):
        state["failed"] = True
    assert aot_serve.is_armed(pipe) is False
    assert aot_serve.holds_exported_cell(pipe) is True, (
        "a revoked cell stopped being recognizable as an exported one, which "
        "would route it back onto the dynamo lane's ledger")

    aot_serve.unwrap(pipe)
    assert aot_serve.holds_exported_cell(pipe) is False
