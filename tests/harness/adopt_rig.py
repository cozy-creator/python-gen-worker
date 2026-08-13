"""THE BOOT-ADOPT VEHICLE — the default rig for anything arming-adjacent.

WHY THIS EXISTS (pgw#1152). Four of the six gates the reuse circle hit —
pgw#1108, pgw#1122, pgw#1141, pgw#1141b — were ONE defect wearing different
clothes: **the boot-adopt path structurally differs from the self-mint path**.
It arms BEFORE setup, the compute child holds no JWT, and it is fed by
``fleet_cells.arm_ordered`` rather than by the self-produced routes
(``arm_from_local_store`` / ``adopt_delegated_mint``). Every one of those gates
was written and validated against self-mint.

And the TESTS could not see it. Their fixtures simulated an adoption using
self-mint machinery: thirteen rows called ``aot_serve.note_aot_key(key)`` BY
HAND — production's single missing line — another stubbed an accessor to raise,
another hand-set a marker production never writes. RED-first caught all six
*after* they shipped and prevented none, because a mutation proves your
assertion can fail, not that your fixture resembles production.

So: **a test may not construct anything production constructs differently.** No
hand-registration, no stubbed accessor, no hand-set marker. Where a test needs
to force an outcome it does so by REMOVING or BREAKING a real input — a hub that
serves no receipt, a package whose entry raises, a cell whose subject really
diverges — never by supplying a fact.

WHAT RUNS FOR REAL. The whole chain from the ordered arm to the target install::

    Executor.ensure_setup
      -> _injection_kwargs -> _enable_compiled with a real _ArmOrder(adopt=…)
      -> fleet_cells.arm_ordered
      -> the real receipt gate, against a real RSA-signed receipt from a real
            HTTP hub (harness.receipt_hub)
      -> provision.arm_aot -> aot_serve.arm_entry on a real packed cell
            (harness.exported_cell) whose ck1 key is restatable from its own
            recorded facts
      -> the real boot warmup, the real per-object proof pass, the real
            _install_compile_targets and _assert_armed_targets_installed

Nothing about applicability, the lane split, the proof pass, the arm decision
or the target install is stubbed — those are the code under test.

THREE SEAMS, all WEST of everything this rig is used to measure, all named:

* ``Executor._boot_adopt`` returns a constructed HIT. Its derive+resolve half is
  driven end to end for real by ``test_boot_adopt_observability_pgw1116.py``
  against ``examples/micro-diffusion`` and a real hub; repeating it here would
  measure that, not this.
* ``provision.load_slot`` returns the probe pipeline instead of reading weights
  off disk — the class-annotated slot shape micro-diffusion uses,
  ``Slot(MicroPipeline, selected_by="model")``, which is what routes the arm
  order to ``_enable_compiled`` at all.
* ``aot_serve._load_package``: an AOTI ``.so`` needs a GPU. pgw#868's rig owns
  this substitution and it is the only piece deferred to a pod.

Usage::

    def test_something(tmp_path, monkeypatch, hub):
        boot = AdoptRig(tmp_path, monkeypatch, hub).boot()
        assert boot.serves_compiled()

``hub`` is :mod:`harness.receipt_hub`'s fixture; import it into your module.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import msgspec
import pytest
import torch

from gen_worker import (
    RequestContext, Resources, Slot, activity, aot_identity, aot_serve,
    boot_adopt, boot_key, cell_key, cell_resolve, endpoint, env_seal, receipts,
    worker_function,
)
from gen_worker import executor as ex_mod
from gen_worker.executor import Executor
from gen_worker.models import provision
from gen_worker.models.refs import normalize_model_ref
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs

from harness import exported_cell as cell868
from harness.exported_cell import (
    ROWS, RUNTIME, ProbeDenoiser, ProbePackage, ProbePipeline, declaration,
    entry_name,
)

#: The publishing org. PLATFORM tier by default, so the trust rule needs no
#: viewer identity — the identity half is pgw#1122's and is fenced there.
ORG = "11111111-2222-3333-4444-555555555555"

FAMILY = cell868.FAMILY
MODEL_REF = "acme/micro-probe:prod"

#: Everything the endpoint instance (constructed by the executor, not by the
#: test) and the load seam hand each other for one boot.
RIG: Dict[str, Any] = {}


class GenIn(msgspec.Struct):
    prompt: str = "warm"


class Out(msgspec.Struct):
    y: str = "ok"


#: pgw#868's declaration with the one field an ``@endpoint`` lint requires that
#: a bare ``provision.arm_aot`` rig never needed: the probe denoiser is
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
        """The pod's default shape: the boot warmup runs and its payload lands
        on NO packaged entry of the adopted cell, so the artifact's own counter
        does not move.

        With ``warm_dispatches`` the handler really calls the wrapped forward
        at a DECLARED shape row — a genuine dispatch through the cell, which is
        what moves ``aot_serve.execution_count``. Nothing here writes a counter.
        """
        for _ in range(int(RIG.get("warm_dispatches", 0))):
            h, w = ROWS[0]
            self.pipe.denoiser(torch.zeros(h, w), torch.tensor(1.0))
        return Out()


# ---------------------------------------------------------------------------
# the cell: a real packed artifact with a real, fully-stated identity
# ---------------------------------------------------------------------------


def cell_metadata() -> Dict[str, Any]:
    """pgw#868's envelope plus the two identity blocks an ADOPTED cell must
    carry: ``verify_declared_identity`` compares four axes and refuses a cell
    that is SILENT on any of them."""
    meta = cell868.metadata()
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
    meta["cell_key"] = cell_key.from_entry_metadata(meta).digest
    return meta


def resolved_cell(
    meta: Dict[str, Any], *, publisher_tier: str = "platform",
    publisher_org: str = ORG,
) -> cell_resolve.ResolvedCell:
    """The hub's answer, stating exactly the identity the mint stamped."""
    have = aot_identity.artifact_identity(meta)
    return cell_resolve.ResolvedCell(
        family=FAMILY, cell_key=have.cell_key,
        cell_ref=f"root/family-{FAMILY}#{have.cell_key}",
        checkpoint_id="", content_digest="sha256:" + "ab" * 32,
        artifact_path="cell.tar.gz", size_bytes=0,
        publisher_org=publisher_org, publisher_tier=publisher_tier,
        graph_contract=have.graph_contract_digest,
        toolchain_digest=have.toolchain_digest,
        env_seal_digest=have.env_seal_digest,
        identity_axes={}, sm=RUNTIME["sm"], sku=RUNTIME["sku"], lane="",
        receipt="",
        transport=cell_resolve.Transport(
            snapshot_digest="blake3:" + "cd" * 32, files=()),
    )


def production_cfgs() -> Dict[str, Any]:
    """Every cfg object a PRODUCTION path hands the compile machinery.

    pgw#1150 (second pass): raw ``Compile`` never travels past the registry, and
    a suite that only ever passes one is testing a shape no fleet path builds —
    deleting ``registry.py``'s two ``numerics_floor=`` lines left that suite
    green. Both entries below come from a real production call site:

    * ``registry`` — ``EndpointSpec.compile_cell()``, what the executor hands
      ``_enable_compiled`` on every serving pod;
    * ``cli`` — ``cli.run``'s §4.28 desktop arm, the other call site.

    Both now route through ``CompileCell.from_declaration``, so a row
    parametrised over this dict proves the map is genuinely one map. Use it
    wherever a gate takes a cfg::

        @pytest.mark.parametrize("cfg", production_cfgs().values(), ids=...)
    """
    from gen_worker.registry import CompileCell

    spec = next(s for s in extract_specs(AdoptedFamily) if s.name == "generate")
    return {
        "registry": spec.compile_cell(),
        "cli": CompileCell.from_declaration(CELL),
    }


def _derived(digest: str) -> boot_key.DerivedKey:
    return boot_key.DerivedKey(
        # pgw#1176: the rig's derived manifest — one entry key per declared
        # class. `digest` stands in for the traced graph, as it always did.
        entry_keys={"rig": "ek1-" + (digest * 56)[:56]},
        class_hashes={}, manifest="", workers=1, width_reason="rig",
        traced=len(ROWS), memo="miss", wall_ms=10_291,
    )


# ---------------------------------------------------------------------------
# what a boot produced, read back through production's own accessors
# ---------------------------------------------------------------------------


@dataclass
class AdoptedBoot:
    """One completed ``ensure_setup`` over a boot-adopted cell.

    Every reader below asks PRODUCTION for the answer (``aot_serve.is_armed``,
    the class record's own compile targets, ``Executor._served_execution_lane``)
    rather than restating what the rig believes it did.
    """

    executor: Executor
    spec: Any
    pipeline: AdoptedPipeline
    cell: cell_resolve.ResolvedCell
    artifact: Path
    meta: Dict[str, Any]
    packages: Dict[str, ProbePackage]
    events: List[Tuple[str, str, str]] = field(default_factory=list)

    # -- the wire ---------------------------------------------------------
    def phases(self, kind: str) -> List[str]:
        return [phase for k, phase, _d in self.events if k == kind]

    def details(self, kind: str) -> List[str]:
        return [detail for k, _p, detail in self.events if k == kind]

    def adopted(self) -> bool:
        return "hit" in self.phases(activity.KIND_BOOT_ADOPT)

    # -- the object -------------------------------------------------------
    @property
    def record(self) -> Any:
        return self.executor._classes[self.spec.instance_key]

    def is_armed(self) -> bool:
        return aot_serve.is_armed(self.pipeline)

    def holds_cell(self) -> bool:
        return aot_serve.holds_exported_cell(self.pipeline)

    def compile_target(self) -> Any:
        targets = self.record.compile_targets
        return next(iter(targets.values())) if targets else None

    def serves_compiled(self) -> bool:
        return self.executor._served_execution_lane(self.spec).endswith(
            "+compiled")

    @property
    def armed_cfg(self) -> Any:
        """The cfg object PRODUCTION handed ``provision.arm_aot`` on this boot —
        observed at the call, never constructed here."""
        return RIG.get("armed_cfg")


# ---------------------------------------------------------------------------
# the rig
# ---------------------------------------------------------------------------


class AdoptRig:
    """Build and drive ONE boot-adopt.

    Every keyword forces an outcome by removing or breaking a REAL input:

    ``cosine``          the packaged subject really diverges from eager by that
                        much (adoption runs no quality gate — §4.32 — so this
                        must NOT refuse; it is here so a test can prove that).
    ``package_raises``  the packaged entry really raises when dispatched, which
                        is how a cell revokes itself for real.
    ``bind_oom_on``     that entry's constant bind really runs the card out of
                        device memory (pgw#1175) — the ONLY evidence a pod
                        cannot hold a cell, now that the estimate that used to
                        guess it is deleted.
    ``serve_receipt``   False = the hub answers 404 for these bytes, so the real
                        receipt gate refuses on a missing input.
    ``publisher_tier``  what the RECEIPT says, verbatim. ``"org"`` needs a
                        viewer identity this process does not hold (pgw#1122).
    ``warm_dispatches`` the handler really calls the wrapped forward N times.
    ``after_arm``       called with the pipeline immediately after the real
                        ``arm_entry`` returns — an observation point at the
                        exact moment production has one, for driving real
                        dispatches before setup's proof snapshot. It may not
                        write state the arm did not.
    """

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        hub: Any,
        *,
        cosine: float = 1.0,
        package_raises: str = "",
        bind_oom_on: str = "",
        serve_receipt: bool = True,
        publisher_tier: str = "platform",
        publisher_org: str = ORG,
        owning_endpoint: str = "",
        warm_dispatches: int = 0,
        after_arm: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.hub = hub
        self.cosine = cosine
        self.package_raises = package_raises
        self.bind_oom_on = bind_oom_on
        self.serve_receipt = serve_receipt
        self.publisher_tier = publisher_tier
        self.publisher_org = publisher_org
        self.owning_endpoint = owning_endpoint
        self.warm_dispatches = warm_dispatches
        self.after_arm = after_arm
        self.events: List[Tuple[str, str, str]] = []

    # -- assembly ---------------------------------------------------------

    def _install_event_spy(self) -> None:
        """Capture at the ONE sink the hub's ``worker_activity_events`` table
        is built from, and still let the real emitter run."""
        said = self.events
        real = activity.emit_event

        def _spy(kind: str, detail: str = "", **kw: Any) -> Any:
            said.append((kind, str(kw.get("phase", "")), detail))
            try:
                return real(kind, detail, **kw)
            except Exception:
                return None

        self.monkeypatch.setattr(activity, "emit_event", _spy)
        self.monkeypatch.setattr(ex_mod.activity_mod, "emit_event", _spy)

    def _forget_keys(self) -> None:
        """``_KNOWN_AOT_KEYS`` is PROCESS-global, so a sibling test file that
        armed the same key would answer this boot's question for it. Every boot
        starts with a runtime that has been told nothing.

        This REMOVES an input; it never supplies one. It is the only legitimate
        contact any test has with that set.
        """
        self.monkeypatch.setattr(aot_serve, "_KNOWN_AOT_KEYS", set())

    def _packages(self) -> Dict[str, ProbePackage]:
        return {
            entry_name(h, w): ProbePackage(
                cosine=self.cosine, raises=self.package_raises,
                bind_oom=entry_name(h, w) == self.bind_oom_on)
            for h, w in ROWS
        }

    def _install_seams(self, packages: Dict[str, ProbePackage]) -> None:
        mp = self.monkeypatch
        # SEAM 3 (GPU) + the runtime axes a cardless box cannot state —
        # pgw#868's substitutions, verbatim.
        mp.setattr(aot_serve, "runtime_key", lambda: dict(RUNTIME))
        mp.setattr(aot_serve, "_entry_admission_drift", lambda *a, **k: None)
        mp.setattr(
            aot_serve, "_load_package",
            lambda path, entry="model": packages[entry])

        # SEAM 2: the class-annotated slot's weights load. `_inject_models`
        # arms whatever this returns, through the real ordered path.
        def _load_slot(annotation: Any, path: str, **_kw: Any) -> provision.SlotLoad:
            pipe = AdoptedPipeline()
            RIG["pipe"] = pipe
            return provision.SlotLoad(obj=pipe, is_pipeline=True, ran="bf16")

        mp.setattr(provision, "load_slot", _load_slot)
        mp.setattr(ex_mod.provision, "load_slot", _load_slot)

        # OBSERVE, never construct: what cfg object did production actually
        # hand the arm? pgw#1150's third variant is a fixture passing a TYPE no
        # fleet path builds, so the rig records the real one and
        # `AdoptedBoot.armed_cfg` exposes it for assertion.
        real_arm = provision.arm_aot

        def _record_cfg(pipeline: Any, cfg: Any, *a: Any, **k: Any) -> Any:
            RIG["armed_cfg"] = cfg
            return real_arm(pipeline, cfg, *a, **k)

        mp.setattr(provision, "arm_aot", _record_cfg)

        if self.after_arm is not None:
            real_wrap = aot_serve.arm_entry
            hook = self.after_arm

            def _wrap_then_observe(pipeline: Any, *a: Any, **k: Any) -> Any:
                meta = real_wrap(pipeline, *a, **k)
                hook(pipeline)
                return meta

            mp.setattr(aot_serve, "arm_entry", _wrap_then_observe)

    def _install_boot_adopt(
        self, cell: cell_resolve.ResolvedCell, artifact: Path,
    ) -> None:
        """SEAM 1: the derive+resolve half, whose own coverage is pgw#1116's."""

        def _adopt(_self: Any, spec: Any, slots: Any) -> boot_adopt.BootAdoptOutcome:
            return boot_adopt.report(boot_adopt.BootAdoptOutcome(
                adoption=boot_adopt.BootAdoption(
                    derived=_derived(cell.cell_key), cell=cell,
                    artifact=artifact),
                reason=boot_adopt.HIT, derived_key=cell.cell_key,
                derive_ms=10_291, family=FAMILY, function="generate"))

        self.monkeypatch.setattr(Executor, "_boot_adopt", _adopt)

    # -- the drive --------------------------------------------------------

    def boot(self) -> AdoptedBoot:
        RIG.clear()
        RIG["warm_dispatches"] = self.warm_dispatches
        self._install_event_spy()
        self._forget_keys()

        meta = cell_metadata()
        artifact = cell868.artifact(self.tmp_path, meta)
        cell = resolved_cell(
            meta, publisher_tier=self.publisher_tier,
            publisher_org=self.publisher_org)

        if self.serve_receipt:
            # The hub countersigns these EXACT bytes.
            self.hub.serve_receipt_for(
                artifact, cell_key=cell.cell_key, family=FAMILY,
                publisher_tier=self.publisher_tier,
                publisher_org_id=self.publisher_org,
                owning_endpoint_id=self.owning_endpoint)
        receipts.configure(base_url=self.hub.base_url, worker_jwt=lambda: "")

        packages = self._packages()
        self._install_seams(packages)
        self._install_boot_adopt(cell, artifact)

        executor, spec = self._ensure_setup()
        return AdoptedBoot(
            executor=executor, spec=spec, pipeline=RIG["pipe"], cell=cell,
            artifact=artifact, meta=meta, packages=packages,
            events=self.events)

    def _ensure_setup(self) -> Tuple[Executor, Any]:
        sent: List[pb.WorkerMessage] = []

        async def _send(msg: pb.WorkerMessage) -> None:
            sent.append(msg)

        ex = Executor(extract_specs(AdoptedFamily), _send)
        ex.store._cache_dir = self.tmp_path / "cas"

        async def _fake_download(target: str, **_kw: Any) -> Path:
            p = self.tmp_path / target.replace("/", "_").replace(":", "_")
            p.mkdir(parents=True, exist_ok=True)
            return p

        self.monkeypatch.setattr(ex_mod, "ensure_local", _fake_download)

        from gen_worker import dispatch as dispatch_mod

        spec = ex.specs["generate"]
        eff = ex._dispatched_spec(
            spec,
            {"pipeline": dispatch_mod.SlotOrder(ref=MODEL_REF, components=())})
        snaps = {normalize_model_ref(MODEL_REF): pb.Snapshot(
            digest="d1" * 16,
            files=[pb.SnapshotFile(
                path="model.safetensors", size_bytes=5, blake3="cd" * 32,
                url="http://r2.invalid/presigned")])}
        asyncio.run(ex.ensure_setup(eff, snaps))
        return ex, eff


# ---------------------------------------------------------------------------
# the historical-defect bank — a rig that cannot re-find a bug we already fixed
# proves nothing
# ---------------------------------------------------------------------------


def _revert_pgw1141b(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put the tree back where ``f3ab710e`` found it, both halves.

    1. ``arm_entry`` stops registering the key it just armed, so
       ``_KNOWN_AOT_KEYS`` goes back to being fed only by the two SELF-PRODUCED
       routes — which the ordered/boot-adopt arm is not one of.
    2. ``executor._exported_arm`` goes back to asking the REF STRING through
       ``is_aot_ref`` instead of asking the object.

    Both are DELETIONS of the fix, not additions of a fault: this is master
    before the fix, reached by removing what the fix added.
    """
    real_wrap = aot_serve.arm_entry
    real_note = aot_serve.note_aot_key

    def _wrap_without_registering(pipeline: Any, *a: Any, **k: Any) -> Any:
        aot_serve.note_aot_key = lambda _key: None  # type: ignore[assignment]
        try:
            return real_wrap(pipeline, *a, **k)
        finally:
            aot_serve.note_aot_key = real_note  # type: ignore[assignment]

    monkeypatch.setattr(aot_serve, "arm_entry", _wrap_without_registering)
    monkeypatch.setattr(
        ex_mod, "_exported_arm",
        lambda pipeline, ref="": bool(ref) and aot_serve.is_aot_ref(ref))


#: name -> the deletion that puts the tree back where the fix found it.
HISTORICAL_DEFECTS: Dict[str, Callable[[pytest.MonkeyPatch], None]] = {
    "pgw1141b": _revert_pgw1141b,
}


def reintroduce(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Remove a landed fix so the rig can be asked to re-find its bug.

    A rig is only worth what it can catch, and the only honest evidence of that
    is a bug we ALREADY know the shape of. Every entry here is a deletion of
    production code, never an injected fault.
    """
    try:
        HISTORICAL_DEFECTS[name](monkeypatch)
    except KeyError:
        raise AssertionError(
            f"no historical defect named {name!r}; "
            f"have {sorted(HISTORICAL_DEFECTS)}") from None
