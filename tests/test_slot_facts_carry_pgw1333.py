"""pgw#1333: the serving facts are RESOLVED once and CARRIED, never re-derived.

**The defect.** ``child_contract.MintSlot`` was ``(ref, path)``. Every mint and
boot-trace child re-derived its slot resolution through
``child_preflight.assert_slots_resolvable`` ->
``warmup.resolved_slots_kwargs(spec, None)`` — ``slots=None``, hardcoded — so
the objective the chain read was ``""`` and ``api.slot._finish_resolved``
refused the moment the invoked function declared ``objectives=``. Every AOT
mint and every boot adoption of every such function, on any catalog data,
forever. It cost e2e#1892 leg-A run 3 a paid L40S pod and cost the blocker its
correct attribution: the refusal says *"resolved checkpoint carries no training
objective"*, so it was filed against the checkpoint. The checkpoint was fine.

**The shape of the fix.** The facts are a TYPE
(:mod:`gen_worker.serving_facts`), they ride ``SlotOrder`` -> ``spec.slot_facts``
-> ``MintSlot`` -> back onto the child's rediscovered spec, and "nobody
resolved them" is a distinct member of that type which NAMES its owner. The
two states an empty string used to conflate — *the catalog says nothing* and
*nobody asked the catalog* — now CONFESS with different sentences, aimed at
the different people who can close them.

**AMENDED by pgw#1339 / th#2099.** This lane also made absence a refusal, and
that half was wrong: shipped as 0.120.0 it took `sd15` and `anima` down in
production on correctly-stamped checkpoints. Absence now degrades loudly
and serves; only a CONTRADICTION refuses. The carrying half — the type, the
chain, the attribution — is untouched and is what the rest of this file
proves.

Every test below fails if any link in that chain drops the facts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple, cast

import msgspec
import pytest

from gen_worker import child_contract, child_preflight, dispatch, registry, serving_facts
from gen_worker import warmup
from gen_worker.api.binding import ModelRef
from gen_worker.pb import worker_scheduler_pb2 as pb

CATALOG_MODULE = "harness.mint_catalog_slot_pgw969"
#: `wai-illustrious`' real stamp, read three ways off the standing hub when
#: pgw#1333 was filed: objective=epsilon, distilled=f, status=classified.
EPSILON = serving_facts.ServingFacts(
    objective="epsilon", distilled=False, distilled_status="classified")
PICKED = ModelRef(source="tensorhub", path="harness/catalog-pick", release="prod")


def _siblings() -> Tuple[Any, List[Any]]:
    """Exactly what a child does first: a FRESH discovery, holding the
    declaration and nothing the parent resolved."""
    specs = registry.collect_endpoints([CATALOG_MODULE])
    return child_preflight.select_specs(specs, "catalog-generate")


def _preflight(facts: serving_facts.SlotEvidence, tmp_path: Path) -> None:
    chosen, siblings = _siblings()
    assert chosen.objectives == ("epsilon", "v_prediction"), (
        "the harness endpoint must declare sdxl's serving contract, or this "
        "whole file is green on a shape that cannot exhibit the defect")
    slots = {"pipeline": child_contract.MintSlot(
        ref=PICKED, path=str(tmp_path), facts=facts)}
    child_preflight.bind_slots(siblings, slots)
    child_preflight.assert_slots_resolvable(
        siblings, slots, what="boot key derivation for 'catalog-generate'")


# ---------------------------------------------------------------------------
# 1. The type: a resolved slot cannot be written down without its facts
# ---------------------------------------------------------------------------


def test_a_slot_cannot_be_constructed_without_a_facts_stanza() -> None:
    """``ref`` and ``path`` alone are the pod's request. It must not decode.

    pgw#974 made bytes-without-identity unconstructable for the same reason
    and by the same means; this is the third half of one resolution.
    """
    with pytest.raises(TypeError):
        child_contract.MintSlot(ref=PICKED, path="/tree")  # type: ignore[call-arg]


def test_the_wire_carries_the_facts_and_which_kind_they_are() -> None:
    """The parent/child boundary is a JSON file, so the constructor is only
    half the guard. Both members of the union must survive the round trip
    DISTINGUISHABLY — collapsing them back to one is the defect."""
    for facts in (EPSILON, serving_facts.FactsUnavailable(owed_by="a sender")):
        slot = child_contract.MintSlot(ref=PICKED, path="/tree", facts=facts)
        back = msgspec.json.decode(
            msgspec.json.encode(slot), type=child_contract.MintSlot)
        assert back == slot
        assert type(back.facts) is type(facts)


def test_an_anonymous_gap_is_not_expressible() -> None:
    """``FactsUnavailable`` must name WHO owes the stamp. A blank owner is the
    empty string this type exists to abolish, wearing a struct."""
    with pytest.raises(ValueError):
        serving_facts.FactsUnavailable(owed_by="  ")


def test_a_bogus_objective_is_refused_at_construction() -> None:
    """Validated where it is built, not at the gate that reads it: a typo
    reaching ``_finish_resolved`` produces "not in the declared objectives",
    which reads as a compatibility refusal and is really a decode bug."""
    with pytest.raises(ValueError):
        serving_facts.ServingFacts(objective="epsilion")
    with pytest.raises(ValueError):
        serving_facts.ServingFacts(distilled_status="probably")


# ---------------------------------------------------------------------------
# 2. The defect itself: the carried facts RESOLVE the child's preflight
# ---------------------------------------------------------------------------


def test_the_parents_facts_resolve_the_childs_preflight(tmp_path: Path) -> None:
    """THE regression. An sdxl-shaped spec + an epsilon checkpoint resolves.

    Red without the fix: ``assert_slots_resolvable`` re-derived the objective
    from ``resolved_slots_kwargs(spec, None)`` and got ``""``, so this raised
    ``PreflightRefused`` on catalog data that satisfies the contract exactly.
    """
    _preflight(EPSILON, tmp_path)  # must not raise


def test_a_mismatch_still_refuses(tmp_path: Path) -> None:
    """The gate was not bought by deleting it. A ``flow`` checkpoint into an
    ``(epsilon, v_prediction)`` function is a real incompatibility and must
    still be a named refusal."""
    with pytest.raises(child_preflight.PreflightRefused) as exc:
        _preflight(
            serving_facts.ServingFacts(
                objective="flow", distilled=False,
                distilled_status="classified"),
            tmp_path)
    assert "is not in the invoked function's declared objectives" in str(exc.value)


def test_an_unclassified_checkpoint_SERVES(tmp_path: Path) -> None:
    """SUPERSEDED BY pgw#1339 / th#2099 — this asserted a refusal, and the
    refusal was the outage.

    This lane read §1.22 fail-closed as governing the objective axis. It does
    not: §4.20 says a gate whose evidence source is unavailable does the best
    it can *without displacing the platform*, and the DEGRADATION ruling says
    the worker complains loudly and still works. The objective backstop is not
    a safety gate — the hub gates checkpoint<->function compatibility at
    deploy and at request time, and this reader's own docstring calls itself a
    version-skew backstop. Shipping it fatal in 0.120.0 took `sd15` and
    `anima` down in production, both measured.

    So an unclassified checkpoint SERVES, and confesses. The gate that
    survives is the one against CONTRADICTION, asserted directly below and in
    `test_objective_absence_degrades_pgw1339.py`.

    **AFFIRMED, do not re-litigate from §1.22.** The supersession was reviewed
    and upheld against two of Paul's standing rulings in his own words: the
    machine-compatibility charter — *"it always runs, just possibly horribly
    inefficiently"* — and the CPU-offload ruling — *"we always allow it, and
    encourage it, although when it happens we should warn loudly so the error
    can be caught."* Serving-with-a-loud-warning on absent evidence is
    precisely that shape.
    """
    _preflight(serving_facts.ServingFacts(), tmp_path)  # must not raise


def test_a_distilled_mismatch_still_refuses(tmp_path: Path) -> None:
    """The orthogonal axis, on the same carried stanza."""
    with pytest.raises(child_preflight.PreflightRefused) as exc:
        _preflight(
            serving_facts.ServingFacts(
                objective="epsilon", distilled=True,
                distilled_status="classified"),
            tmp_path)
    assert "distilled=True" in str(exc.value)


def test_unstamped_distillation_evidence_SERVES(tmp_path: Path) -> None:
    """``distilled=False`` with an unstamped status is still not evidence of
    False — and, per pgw#1339, still not grounds to refuse a paid request.
    The orthogonal axis moves with the objective axis, for the same reason."""
    _preflight(
        serving_facts.ServingFacts(objective="epsilon", distilled=False),
        tmp_path)  # must not raise


def test_the_undeclared_sibling_is_unaffected(tmp_path: Path) -> None:
    """``catalog_generate_turbo`` declares nothing, so no facts are CHECKED
    for it — including when there are none. Refusing here would refuse every
    family that never opted into the serving contract."""
    specs = registry.collect_endpoints([CATALOG_MODULE])
    chosen, siblings = child_preflight.select_specs(
        specs, "catalog-generate-turbo")
    assert chosen.objectives is None and chosen.distilled is None
    # The sibling set includes the GOVERNED handler, so a gap still refuses —
    # the class is minted as a class (pgw#654). Resolve the governed one and
    # the ungoverned one must ride along.
    slots = {"pipeline": child_contract.MintSlot(
        ref=PICKED, path=str(tmp_path), facts=EPSILON)}
    child_preflight.bind_slots(siblings, slots)
    child_preflight.assert_slots_resolvable(siblings, slots, what="turbo")


def test_an_undeclared_class_resolves_with_no_facts_at_all(tmp_path: Path) -> None:
    """A family that declares NOTHING must mint with a bare
    ``FactsUnavailable`` — the unrestricted arm reads no facts, so a gap costs
    it nothing and must not refuse."""
    specs = registry.collect_endpoints(["harness.toy_endpoints"])
    chosen, siblings = child_preflight.select_specs(specs, "juggle-echo")
    assert chosen.objectives is None
    slots = {"pipeline": child_contract.MintSlot(
        ref=PICKED, path=str(tmp_path),
        facts=serving_facts.FactsUnavailable(owed_by="nothing asked"))}
    child_preflight.bind_slots(siblings, slots)
    child_preflight.assert_slots_resolvable(siblings, slots, what="juggle")


# ---------------------------------------------------------------------------
# 3. The gap fails LOUD, and blames the right repo
# ---------------------------------------------------------------------------


def test_a_missing_stamp_CONFESSES_by_name_and_names_the_hub(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The boot half is a HUB gap: ``hello_ack.go`` builds
    ``DesiredInstance.ModelBinding`` with the three serving-fact fields left
    zero, though the proto has them and the dispatch path stamps them.

    pgw#1339 turned the refusal into a confession — a gap on OUR side of the
    wire is not grounds to decline a paid request. **The attribution this test
    was written for is unchanged and still load-bearing**, because it is the
    whole reason the type has two members: the sentence must name the WIRE,
    not read as a verdict on the checkpoint. That mis-reading is what sent
    e2e#1892's blocker at the catalog, where three independent reads showed
    the stamp present and correct.
    """
    with caplog.at_level("WARNING"):
        _preflight(
            serving_facts.FactsUnavailable(owed_by=dispatch.BOOT_SENDER_OWES),
            tmp_path)  # serves
    msg = caplog.text
    assert "hello_ack.go" in msg
    assert "objective" in msg and "distilled_status" in msg
    assert "NOT a claim about the checkpoint's catalog row" in msg
    assert "carries no training objective" not in msg, (
        "the gap must not borrow the sentence that blames the checkpoint — "
        "that borrowing is what cost this bug its correct attribution")
    assert "stamp the serving facts on the binding this pod was sent" in msg, (
        "a wire gap must be pointed at the SENDER; telling the operator to go "
        "classify a checkpoint that is already classified is the same "
        "mis-attribution wearing a suggestion")


# ---------------------------------------------------------------------------
# 4. The parent half: one projection, and it reads every field
# ---------------------------------------------------------------------------


def test_the_dispatch_sender_stamps_so_its_zero_triple_is_an_answer() -> None:
    """``RunJob.models`` comes from the hub's catalog stamp, so an empty
    objective there is the catalog saying "unclassified" — a real answer, and
    a refusal against it is a real refusal."""
    order = dispatch.order_from_binding(
        pb.ModelBinding(slot="pipeline", ref="tensorhub/x@prod"))
    assert order.facts == serving_facts.ServingFacts()


def test_every_wire_field_is_read(tmp_path: Path) -> None:
    """The boot path built ``SlotOrder(ref=ref)`` and dropped three fields the
    message already carried. One projection now, for every sender."""
    order = dispatch.order_from_binding(pb.ModelBinding(
        slot="pipeline", ref="tensorhub/x@prod",
        inference_defaults='{"steps": 4}',
        objective="v_prediction", distilled=True,
        distilled_status="classified"))
    assert order.facts == serving_facts.ServingFacts(
        objective="v_prediction", distilled=True,
        distilled_status="classified")
    assert order.inference_defaults == '{"steps": 4}'


def test_the_boot_sender_zero_triple_reads_as_the_gap_it_is() -> None:
    """proto3 scalars have no presence, so a sender KNOWN not to stamp yet is
    read as the wire gap rather than as an unclassified row. The distinction
    is the whole point of ``owed_by``."""
    order = dispatch.order_from_binding(
        pb.ModelBinding(slot="pipeline", ref="tensorhub/x@prod"),
        owed_by=dispatch.BOOT_SENDER_OWES)
    assert isinstance(order.facts, serving_facts.FactsUnavailable)
    assert "hello_ack.go" in order.facts.owed_by
    # ...but the moment the hub DOES stamp, the same call reads the answer.
    stamped = dispatch.order_from_binding(
        pb.ModelBinding(
            slot="pipeline", ref="tensorhub/x@prod", objective="epsilon",
            distilled_status="classified"),
        owed_by=dispatch.BOOT_SENDER_OWES)
    assert stamped.facts == EPSILON


def test_an_order_cannot_be_built_without_saying_which_kind() -> None:
    """No default on ``SlotOrder.facts``: a construction site that has not
    thought about the facts is a compile error, not a blank objective."""
    with pytest.raises(TypeError):
        dispatch.SlotOrder(ref="tensorhub/x@prod")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 5. The seam between them: the parent's spec carries what it resolved
# ---------------------------------------------------------------------------


def test_the_dispatched_spec_folds_the_facts_in_beside_the_ref() -> None:
    """``_dispatched_spec`` is where the order becomes the spec every
    downstream consumer reads. It folded in the ref and dropped the facts —
    invisibly, because the only consumer that noticed was a child process."""
    from types import SimpleNamespace

    from gen_worker.executor import Executor

    specs = registry.collect_endpoints([CATALOG_MODULE])
    chosen, _ = child_preflight.select_specs(specs, "catalog-generate")
    order = dispatch.order_from_binding(pb.ModelBinding(
        slot="pipeline", ref="tensorhub/catalog-pick@prod",
        objective="epsilon", distilled_status="classified"))
    ex = object.__new__(Executor)
    ex._hub_bindings = {}
    ex.store = cast(Any, SimpleNamespace(register_binding=lambda *a, **k: None))
    effective = Executor._dispatched_spec(ex, chosen, {"pipeline": order})
    assert effective.slot_facts["pipeline"] == EPSILON
    # ...and the resolution chain now reads them from there, on the WARM
    # shape (`slots=None`) that every child uses.
    kwargs = warmup.resolved_slots_kwargs(effective, None)
    assert not kwargs["slot_errors"], kwargs["slot_errors"]
    assert kwargs["resolved_slots"]["pipeline"].objective == "epsilon"


def test_bind_slots_reinstalls_the_facts_in_a_rediscovered_child() -> None:
    """The child re-runs discovery, so ``spec.slot_facts`` comes back EMPTY —
    exactly as ``spec.models`` does. A binding installed without its facts is
    half a resolution, and the half that was missing."""
    _, siblings = _siblings()
    for spec in siblings:
        assert not spec.slot_facts
    child_preflight.bind_slots(siblings, {"pipeline": child_contract.MintSlot(
        ref=PICKED, path="/tree", facts=EPSILON)})
    for spec in siblings:
        assert spec.slot_facts["pipeline"] == EPSILON
        assert spec.models["pipeline"] == PICKED
