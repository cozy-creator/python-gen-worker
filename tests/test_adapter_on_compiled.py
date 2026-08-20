"""A LoRA on a compiled-armed module must never silently serve the base model.

pgw#1571 measured the defect on the v1 arm and fixed it there. pgw#1573's
reachability read then found that ``aot_serve.wrap_module`` — where that fix
lives — has ZERO non-test callers: the production arm is
``torchcg.adopt.AdoptSession``, which none of it touches. So the P0 was open on
the path that runs, and it was masked only by adoption itself being broken.
pgw#1573 fixed adoption; this file is the guard on the live path plus the proof
that it fires.

**These drive the real dispatcher.** ``AdoptSession`` is constructed with a
real document, a real store and a real artifact, and the module that comes back
is the one the author's ``ctx.compile`` returned — the same object a request is
dispatched against. The only stand-in is the bytes-to-callable loader, because
that is AOTInductor on a GPU; what it returns is a callable that is
DISTINGUISHABLE from the eager forward, which is the whole measurement.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest
import torch

import tcg_artifacts
from gen_worker import activity as activity_mod
from gen_worker._vendor.torchcg.adopt import AdoptSession
from gen_worker._vendor.torchcg.discovery import discover_lane
from gen_worker._vendor.torchcg.document import GraphSetDocument
from gen_worker._vendor.torchcg.graph_identity import EnvIdentity
from gen_worker.serving import adapter_guard
from gen_worker.serving.mint_store import graph_store

LANE = "toy.bf16@1"
SM = "sm_89"
STACK: tuple[tuple[str, str], ...] = (("torch", torch.__version__),)
ENV = EnvIdentity(stack=STACK, sm=SM)


class Toy(torch.nn.Module):
    """A denoiser-shaped module: one Linear a peft wrapper can sit on."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(4))

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.proj(sample)


def _document(module: Toy) -> GraphSetDocument:
    """The real publish-time derive over the real module."""
    def drive() -> None:
        module(torch.zeros(2, 4))

    lane = discover_lane(LANE, ("proj_owner",), {"proj_owner": module}, drive)
    return GraphSetDocument(stack=STACK, lanes=(lane,))


def _store_with_artifact(tmp_path: Path, document: GraphSetDocument) -> Any:
    from gen_worker._vendor.torchcg.requirements import RequirementsManifest

    store = graph_store(tmp_path / "cas", None, tmp_path / "no-baked")
    manifest = RequirementsManifest(
        include_set=(("torch", torch.__version__),), sm_compiled=SM)
    for index, record in enumerate(document.lanes[0].graphs):
        envelope = tcg_artifacts.build(
            tmp_path / "fleet" / f"{index}.tar.gz",
            graph_specialization=record.graph, sm=SM)
        store.publish_artifact(record.graph, ENV, envelope, manifest)
    return store


def _loader(calls: List[str]) -> Any:
    """A compiled callable that is DISTINGUISHABLE from the eager forward.

    It returns ``sample * 0`` — nothing the eager path can produce for a
    non-zero input — so "which forward served this call" is a value, not a
    counter somebody has to trust.
    """

    def load(path: Path, record: Any, module: Any) -> Any:
        def compiled(sample: torch.Tensor) -> torch.Tensor:
            calls.append(record.graph)
            return torch.zeros_like(sample)

        return compiled

    return load


@pytest.fixture()
def wire(monkeypatch: pytest.MonkeyPatch) -> List[tuple]:
    seen: List[tuple] = []
    monkeypatch.setattr(
        activity_mod, "emit_event",
        lambda kind, detail, phase="", **kw: seen.append((kind, phase, detail)))
    return seen


def _armed(tmp_path: Path, calls: List[str]) -> Toy:
    """One module, adopted through the GUARDED sink and armed for real."""
    module = Toy()
    document = _document(module)
    store = _store_with_artifact(tmp_path, document)
    session = AdoptSession(
        store, document, LANE, SM, loader=_loader(calls),
        artifacts_dir=tmp_path / "artifacts", stack=STACK)
    adapter_guard.sink(session.adopt)(module)
    assert adapter_guard.compiled_armed(module), (
        "nothing armed, so this file would prove nothing about a compiled "
        "module")
    return module


def test_the_compiled_graph_serves_when_no_adapter_is_attached(
    tmp_path: Path,
) -> None:
    """POLARITY. The guard must not turn compiled serving off by itself —
    otherwise every assertion below passes for the wrong reason."""
    calls: List[str] = []
    module = _armed(tmp_path, calls)

    out = module(torch.ones(2, 4))

    assert calls, "the guard swallowed a call no adapter had any claim on"
    assert torch.equal(out, torch.zeros(2, 4)), (
        "the eager forward served a module with nothing attached to it")


def test_a_LIVE_adapter_routes_EAGER_instead_of_serving_the_base_model(
    tmp_path: Path, wire: List[tuple]
) -> None:
    """THE P0 (pgw#1571), on the path that runs.

    peft wraps a denoiser's SUBMODULES; the armed graph replaces the PARENT's
    forward and never enters them. So an adapter attached after arming does not
    execute, and the artifact returns the base result with no refusal and no
    log — measured there as eager ``max|delta| = 2.2e-02`` against armed
    ``0.0`` with 32 wrappers attached.

    The guard's answer is EAGER: correctness over speed, stated once. What is
    forbidden is the silence.
    """
    calls: List[str] = []
    module = _armed(tmp_path, calls)

    # peft's own marker, and a wrapper on the submodule the graph traced
    # through — the exact shape `inject_adapter_in_model` leaves behind.
    # , not attribute assignment:  types its
    # own attributes, and peft writes this one from outside that contract.
    setattr(module, "peft_config", {"default": object()})
    with torch.no_grad():
        module.proj.weight.mul_(2.0)

    sample = torch.ones(2, 4)
    out = module(sample)

    assert calls == [], (
        "the compiled graph served a request with a live adapter attached — "
        "this is pgw#1571 exactly: the base model, bit-identically, with "
        "nothing said")
    assert torch.equal(out, sample * 2.0), (
        "the eager forward did not serve, so the adapter's effect is absent "
        "from the output")

    rows = [row for row in wire
            if row[0] == activity_mod.KIND_LORA_HYGIENE
            and row[1] == "adapter_ops_on_compiled"]
    assert len(rows) == 1, (
        f"a degradation that reaches only a pod log is the same silence this "
        f"guard exists to end (pgw#760): {wire}")
    assert "BASE MODEL" in rows[0][2] and "lora_fold" in rows[0][2], rows[0][2]

    # Said ONCE per module, not per request: an alarm on the hot path is an
    # alarm operators learn to ignore.
    module(sample)
    module(sample)
    assert len([row for row in wire
                if row[1] == "adapter_ops_on_compiled"]) == 1


def test_detaching_the_adapter_puts_the_compiled_graph_back(
    tmp_path: Path,
) -> None:
    """The degradation is FOR THE DURATION, not for the life of the pod.

    peft deletes `peft_config` on unload, so the next call is compiled again.
    A guard that latched would quietly cost every later request its speed.
    """
    calls: List[str] = []
    module = _armed(tmp_path, calls)

    # , not attribute assignment:  types its
    # own attributes, and peft writes this one from outside that contract.
    setattr(module, "peft_config", {"default": object()})
    module(torch.ones(2, 4))
    assert calls == []

    delattr(module, "peft_config")
    module(torch.ones(2, 4))
    assert calls, "the guard latched: a detached adapter left the module eager"


def test_the_v2_armed_marker_is_what_lora_fold_reads(tmp_path: Path) -> None:
    """``lora_fold._compiled_armed`` asked ``aot_serve``, whose marker no pod
    has carried since pgw#1373 — so on a real worker it answered False for
    every armed module and every compiled-aware branch behind it was dead.

    Red arm: point the predicate back at the v1 tier alone and this goes red
    on a module that IS armed.
    """
    from gen_worker.models import lora_fold

    calls: List[str] = []
    module = _armed(tmp_path, calls)
    assert lora_fold._compiled_armed(module) is True
    assert lora_fold._compiled_armed(Toy()) is False


def test_rearm_constants_refuses_by_name_rather_than_serving_stale_weights(
    tmp_path: Path,
) -> None:
    """The fold path's precondition, stated as a refusal.

    AOTI folds its constants once on the first ``run()`` and never re-folds on
    a bare tensor write, so folding an adapter INTO the weights needs the bound
    table re-installed. A runner that exposes no such table must say so — a
    folded constant that keeps serving pre-fold weights is a plausible wrong
    image and no error at all.
    """
    calls: List[str] = []
    module = _armed(tmp_path, calls)

    # This file's loader returns a bare closure, not a `CompiledGraphCall`, so
    # there is no runner to re-arm — which is exactly the shape the refusal is
    # written for.
    with pytest.raises(adapter_guard.ConstantRearmUnsupported, match="stale"):
        adapter_guard.rearm_constants(module)

    assert adapter_guard.rearm_constants(Toy()) == 0, (
        "an eager module has nothing to re-arm and that is not an error")


# --------------------------------------------------------------------------
# The WIRING: both ctx.compile hosts hand out the guarded sink
# --------------------------------------------------------------------------


def _fixture_host(tmp_path: Path, calls: List[str]) -> Any:
    """The real `EndpointHost`, booted with a real document over a real store.

    Everything above proves the guard WORKS. This proves the production host
    installs it — without which the guard is a correct object nothing calls,
    which is the exact shape of the defect it replaces (pgw#1571's fix lives in
    `aot_serve.wrap_module`, which has no caller).
    """
    from test_serving_adopt_first import (
        ENV as FIXTURE_ENV, LANE as FIXTURE_LANE, OVERRIDES, SM as FIXTURE_SM,
        STACK as FIXTURE_STACK, fresh_host, manifest, publish_document,
    )
    from gen_worker.serving import DeployBinding

    import json as _json
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text(
        _json.dumps({"seed": 7, "scheduler": {"prediction_type": "epsilon"}}))
    binding = DeployBinding(
        checkpoint_ref="ckpt:tiny@1", checkpoint_dir=root, model="sdxl",
        defaults=dict(OVERRIDES))

    derive_host = fresh_host(binding, tmp_path / "derive")
    derive_host.setup()
    document = publish_document(derive_host)

    store = graph_store(tmp_path / "cas", None, tmp_path / "no-baked")
    for index, record in enumerate(document.lanes[0].graphs):
        envelope = tcg_artifacts.build(
            tmp_path / "fleet" / f"{index}.tar.gz",
            graph_specialization=record.graph, sm=FIXTURE_SM)
        store.publish_artifact(record.graph, FIXTURE_ENV, envelope, manifest())

    host = fresh_host(binding, tmp_path)
    host.setup(store=store, document=document, sm=FIXTURE_SM,
               loader=_loader(calls), artifacts_dir=tmp_path / "artifacts",
               stack=FIXTURE_STACK)
    assert len(host.adoption.adopted) == 2, "nothing armed; the wiring is untestable"
    return host


def test_the_PRODUCTION_host_installs_the_guard(
    tmp_path: Path, wire: List[tuple]
) -> None:
    """RED ARM FOR THE WIRING. Hand `EndpointHost.setup` the bare
    `session.adopt` again and this goes red — which is what makes the guard a
    property of the serving path rather than of this test file."""
    calls: List[str] = []
    host = _fixture_host(tmp_path, calls)
    (instance,) = host.instances.values()
    unet = instance.model.pipe.unet
    assert adapter_guard.compiled_armed(unet), (
        "the marked module is not armed, so this row proves nothing")

    host.dispatch("generate", {"prompt": "x", "aspect_ratio": "1:1"},
                  request_id="warm")
    assert calls, "the compiled graph never served, before any adapter existed"
    calls.clear()

    setattr(unet, "peft_config", {"default": object()})
    host.dispatch("generate", {"prompt": "x", "aspect_ratio": "1:1"},
                  request_id="lora")

    assert calls == [], (
        "the production host handed out an UNGUARDED ctx.compile sink, so a "
        "LoRA request served the base model silently (pgw#1571)")
    assert [row for row in wire if row[1] == "adapter_ops_on_compiled"], wire


# --------------------------------------------------------------------------
# The OPERATOR'S ORDER (pgw#1589, carrying pgw#1587's intent to the live path)
# --------------------------------------------------------------------------


def test_an_operator_eager_only_order_suppresses_compiled_dispatch(
    tmp_path: Path, wire: List[tuple]
) -> None:
    """pgw#1589. The hub sends the order, `worker.on_message` applies it, and
    until now exactly ONE thing read it: `aot_serve`'s arm, which has no
    production caller. pgw#1587 added a second read in `arm_aot` — same
    orphaned tier. So an operator could issue `serve_posture{eager_only:true}`,
    get an ack, and watch the pod keep serving from its compiled graphs.

    Read here, on the path that dispatches, so the order is observed by the
    thing it is about.
    """
    from gen_worker import serve_posture

    calls: List[str] = []
    module = _armed(tmp_path, calls)
    serve_posture.reset()
    try:
        module(torch.ones(2, 4))
        assert calls, "compiled serving was off before any order was issued"
        calls.clear()

        serve_posture.apply_command(True, actor="operator@cozy", reason="drain")
        module(torch.ones(2, 4))
        assert calls == [], (
            "an operator ordered this worker EAGER and it dispatched to a "
            "compiled graph anyway (pgw#1589)")

        # The DISPATCH row, on its own phase — `serve_posture` emits its own
        # TRANSITION row on `PHASE_SUPPRESSED`, and one phase carrying two
        # vocabularies is a metric that means neither (pgw#1441's split).
        rows = [row for row in wire
                if row[0] == activity_mod.KIND_LORA_HYGIENE
                and row[1] == serve_posture.REASON]
        assert len(rows) == 1, f"the order must be a wire fact: {wire}"
        assert serve_posture.REASON == "operator_eager_only", (
            "this row finally wires EagerPhase.OPERATOR_EAGER_ONLY, which had "
            "no emitter since pgw#1142 defined it")
        assert "operator@cozy" in rows[0][2] and "drain" in rows[0][2]

        # REVERSIBLE, and with no re-arm: `apply_command` promises it and the
        # read is per call, never latched at arm time.
        serve_posture.apply_command(False, actor="operator@cozy")
        module(torch.ones(2, 4))
        assert calls, (
            "releasing the order did not resume compiled serving — the read "
            "latched, so the order is one-way")
    finally:
        serve_posture.reset()


# --------------------------------------------------------------------------
# pgw#1591: a WRAPPED dispatcher is not a displaced one, and a displaced one
# still reports what it measured
# --------------------------------------------------------------------------


def test_the_guard_does_not_read_as_a_DISPLACED_dispatcher(
    tmp_path: Path,
) -> None:
    """THE P1. `DispatchCounter` asked `module.forward is dispatcher`, and this
    guard installs itself AS `module.forward` — so every guarded module read
    DISPLACED the moment pgw#1573 landed.

    Measured in the field before it was understood: sd15, 12/12 requests of
    both benchmark arms, `dispatch: DISPLACED on UNet2DConditionModel`, and a
    whole GPU leg discarded because the lane could not tell eager from
    compiled. The arm was fine; the question was wrong.
    """
    from gen_worker.serving.dispatch_counter import DispatchCounter

    calls: List[str] = []
    module = _armed(tmp_path, calls)
    session = SimpleNamespace(_dispatchers=[
        (module, adapter_guard.dispatcher_of(module))])
    counter = DispatchCounter().install(SimpleNamespace(adoption=session))

    module(torch.ones(2, 4))
    counts = counter.take()

    assert counts.displaced_modules == (), (
        f"a GUARDED dispatcher is not a displaced one: {counts.facts()}")
    assert counts.compiled_graph_calls == 1 and counts.eager_calls == 0
    assert "DISPLACED" not in counts.summary()


def test_a_DISPLACED_module_still_reports_what_it_MEASURED(
    tmp_path: Path,
) -> None:
    """The second half, and the one that cost the #1548 lane its window.

    The displaced branch used to end "so all N call(s) ran eager" — a sentence
    it never measured. The same logs carried 120 AOTI wrapper invocations per
    arm, so the lane had two irreconcilable readings and correctly refused to
    pick one. Neither was right: compiled DID execute and the message
    overcounted eager.

    Displacement and dispatch are separate facts. Both get stated.
    """
    from gen_worker.serving.dispatch_counter import DispatchCounts

    displaced_but_served = DispatchCounts(
        module_calls=21, compiled_graph_calls=10, armed_modules=1,
        armed_graphs=14, displaced_modules=("UNet2DConditionModel",))
    line = displaced_but_served.summary()
    assert "DISPLACED" in line
    assert "10 of 21 call(s) still served COMPILED" in line, line
    assert "ran eager" not in line, (
        f"the summary asserted eager over a measurement that says otherwise: "
        f"{line}")

    truly_eager = DispatchCounts(
        module_calls=21, compiled_graph_calls=0, armed_modules=1,
        armed_graphs=14, displaced_modules=("UNet2DConditionModel",))
    assert "all 21 call(s) ran eager" in truly_eager.summary()
