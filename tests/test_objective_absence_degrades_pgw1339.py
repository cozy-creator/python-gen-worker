"""pgw#1339 / th#2099 — a checkpoint whose objective the worker CANNOT SEE
serves anyway, loudly. Absence degrades; only a CONTRADICTION refuses.

**The outage.** gen-worker 0.120.0 shipped `_finish_resolved` with the guard
inverted. 0.118.0 read `if resolved.objective:` — an unseen objective was
simply nothing to check. 0.120.0 rewrote it as `if allowed_objectives is not
None:` with an inner `if not resolved.objective: raise`, so an ABSENT
objective went from "nothing to check" to a fatal. On 2026-08-17 `sd15` and
`anima` were republished onto that wheel, promoted, and pointed at `prod`;
every consumer request died

    DeclaredSlotResolutionError: slot 'pipeline': resolved checkpoint carries
    no training objective, so there is no evidence for the invoked function's
    declared objectives ('epsilon', 'v_prediction')

on a checkpoint the hub had stamped `objective='epsilon'` (th#2099 measured
the same digest completing on 0.118.0 against the same hub). 0.121.0 ships
the same reader byte for byte — `api/slot.py` sha256
`8191b9ba…` in both wheels — so the whole fleet campaign was blocked on it.

**Scope, stated precisely because the original filing over-associated it.**
`foundation-1` was named alongside the other two on the night and is NOT a
victim of this defect: it declares no `@worker_function(objectives=/
distilled=)` at all, so nothing here can reach it, and the fleet lane
subsequently measured ZERO `requests`/`request_state` and ZERO `worker_pods`
rows for it over 30 hours — it was never observed serving OR failing. The
defect is measured on `sd15` and `anima`; everything else is association.
Structural immunity is a claim this file can make; a cause of death for an
unobserved endpoint is not.

**The ruling this restores.** §DEGRADATION-IS-LOUD, Paul: *"the worker should
obviously complain loudly ... but it should still work"*; the machine-
compatibility charter, verbatim — *"it always runs, just possibly horribly
inefficiently"*; and the CPU-offload ruling — *"we always allow it, and
encourage it, although when it happens we should warn loudly so the error can
be caught."* Three statements of one shape, and its standing
gloss — the loudness is *diagnostics*, never a gate. pgw#1315 already settled
the identical shape for VRAM: a declared minimum *"gates one thing, a
config-WRITE, and that lives hub-side"*. A declared `objectives=` contract is
the same kind of declaration, and the hub already gates it twice upstream
(deploy-time `bindingcheck`, request-time `ServingContractGovernsSlot`). The
worker-side check is a BACKSTOP against version skew — its own docstring says
so — and a backstop must never be the thing that brings the platform down.

**So there are exactly two ways to fail this file, and they are opposite:**

1. **REFUSING ON ABSENCE.** Unknown evidence is the normal input to a degraded
   run. Refusing it is the outage.
2. **DEGRADING SILENTLY.** The confession is load-bearing — an unevidenced run
   with no `serve_degrade` row is the *other* defect, and it is the one that
   would let this recur invisibly. The emitter is disarmed in the red-verify
   below precisely so these assertions can be shown capable of going red.

**And the line that is NOT moved:** a checkpoint that positively CONTRADICTS
the declared contract (`flow` into an `(epsilon, v_prediction)` function, or a
`classified` distilled=True into a `distilled=False` function) still refuses,
by name. Absence is not evidence of compatibility; it is absence. Buying
always-runs by deleting the gate would be the vacuous pass pgw#1333 warned
about, and every arm of that warning is re-asserted here.

Torch-free by construction: the seam is the slot READER and its confession.
No compile, no mint, no card.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List, Tuple

import pytest

from gen_worker import activity as activity_mod
from gen_worker import child_contract, child_preflight, registry, serving_facts
from gen_worker import measured_posture as posture_mod
from gen_worker.api.binding import ModelRef
from gen_worker.api.slot import ObjectiveMismatchError, resolve_slot
from gen_worker.models import memory as memory_mod
from gen_worker.pb import worker_scheduler_pb2 as pb

CATALOG_MODULE = "harness.mint_catalog_slot_pgw969"
PICKED = ModelRef(source="tensorhub", path="harness/catalog-pick", release="prod")

#: The stamp `sd15`/`wai-illustrious` actually carry.
EPSILON = serving_facts.ServingFacts(
    objective="epsilon", distilled=False, distilled_status="classified")
#: The catalog answered, and the answer was "nothing measured this axis".
UNSTAMPED = serving_facts.ServingFacts()
#: Nobody ever asked the catalog — the wire gap pgw#1333 typed.
NO_ONE_ASKED = serving_facts.FactsUnavailable(owed_by="tensorhub hello_ack.go")

SD15_DECLARES = ("epsilon", "v_prediction")


class _Events:
    """The REAL activity sink the worker transport installs, so these
    assertions read the ActivityUpdates a hub would actually bank."""

    def __init__(self) -> None:
        self.sent: List[pb.WorkerMessage] = []
        self.loop = asyncio.new_event_loop()

    def __enter__(self) -> "_Events":
        async def _send(msg: pb.WorkerMessage) -> None:
            self.sent.append(msg)

        activity_mod.bind_sink(_send, self.loop)
        return self

    def __exit__(self, *exc: object) -> None:
        self.loop.run_until_complete(asyncio.sleep(0.02))
        activity_mod.reset_for_tests()
        self.loop.close()

    def unevidenced(self) -> List[pb.ActivityUpdate]:
        return [
            m.activity_update for m in self.sent
            if m.WhichOneof("msg") == "activity_update"
            and m.activity_update.kind == activity_mod.KIND_SERVE_DEGRADE
            and m.activity_update.phase == memory_mod.UNEVIDENCED_FACTS_PHASE
        ]


def _read(facts: serving_facts.ServingFacts) -> Any:
    """The real reader, on the exact shape the outage hit: sd15's declared
    contract against a resolved checkpoint."""
    return resolve_slot(
        "pipeline", _slot(), ref=PICKED,
        objective=facts.objective,
        distilled=facts.distilled,
        distilled_status=facts.distilled_status,
        allowed_objectives=SD15_DECLARES,
        allowed_distilled=False,
    )


def _slot() -> Any:
    from gen_worker.api.slot import Slot

    return Slot(str, selected_by="model")


def _siblings() -> Tuple[Any, List[Any]]:
    specs = registry.collect_endpoints([CATALOG_MODULE])
    return child_preflight.select_specs(specs, "catalog-generate")


def _preflight(evidence: serving_facts.SlotEvidence, tmp_path: Path) -> None:
    """The whole chain the paid pod walked: spec -> MintSlot -> preflight."""
    chosen, siblings = _siblings()
    assert chosen.objectives == SD15_DECLARES, (
        "the harness endpoint must declare a serving contract, or this file "
        "is green on a shape that cannot exhibit the defect")
    slots = {"pipeline": child_contract.MintSlot(
        ref=PICKED, path=str(tmp_path), facts=evidence)}
    child_preflight.bind_slots(siblings, slots)
    child_preflight.assert_slots_resolvable(
        siblings, slots, what="boot key derivation for 'catalog-generate'")


# ---------------------------------------------------------------------------
# 1. IT RUNS — the outage, inverted
# ---------------------------------------------------------------------------


def test_a_checkpoint_with_no_visible_objective_RESOLVES() -> None:
    """THE regression. This is the exact call that killed `sd15` in prod.

    Red before the fix with the production sentence verbatim: *"resolved
    checkpoint carries no training objective, so there is no evidence for the
    invoked function's declared objectives ('epsilon', 'v_prediction')"*.
    """
    with _Events():
        got = _read(UNSTAMPED)

    assert got.objective == "", (
        "the reader must not invent evidence it does not have — it serves "
        "WITHOUT the fact, it does not fabricate one")
    assert got.ref == PICKED


def test_unstamped_distillation_evidence_RESOLVES() -> None:
    """The orthogonal axis, same rule. `distilled=False` with an unstamped
    status is not evidence of False — and it is not grounds for refusal
    either."""
    with _Events():
        got = _read(serving_facts.ServingFacts(objective="epsilon"))

    assert got.objective == "epsilon"
    assert got.distilled_status == ""


def test_a_wire_gap_RESOLVES_and_still_names_who_owes_the_stamp(
    tmp_path: Path,
) -> None:
    """`FactsUnavailable` is a hub that never stamped, not a bad checkpoint.
    pgw#1333 made it refuse BY NAME; the name was right and the refusal was
    not. The pod serves, and the sentence still points at the gap."""
    with _Events() as ev:
        _preflight(NO_ONE_ASKED, tmp_path)  # must not raise

    rows = ev.unevidenced()
    assert rows, "a wire gap that serves silently is the other defect"
    assert "tensorhub hello_ack.go" in rows[0].detail, (
        "the confession must still blame the GAP, never the checkpoint")


# ---------------------------------------------------------------------------
# 2. AND IT IS LOUD — the confession is the deliverable, not a side effect
# ---------------------------------------------------------------------------


def test_the_degraded_run_confesses_through_the_one_serve_degrade_seam(
    tmp_path: Path,
) -> None:
    """pgw#1312's ONE home, extended with its own phase token — never a
    second emitter. The token IS the machine-readable cause, so the hub can
    count this across the fleet under one spelling."""
    with _Events() as ev:
        _preflight(UNSTAMPED, tmp_path)

    rows = ev.unevidenced()
    assert len(rows) == 1, f"expected exactly one confession, got {len(rows)}"
    assert rows[0].phase == posture_mod.REASON_SERVING_FACTS_UNEVIDENCED


def test_the_confession_names_what_why_and_what_would_be_better(
    tmp_path: Path,
) -> None:
    """The three required parts of every degraded run's contract. A complaint
    without a suggestion is noise."""
    with _Events() as ev:
        _preflight(UNSTAMPED, tmp_path)

    detail = ev.unevidenced()[0].detail
    # WHAT axis is unevidenced, and for which slot.
    assert "objective" in detail and "pipeline" in detail
    # WHY — the declared contract it could not be checked against.
    assert "epsilon" in detail
    # WHAT WOULD BE BETTER — actionable, and aimed at the catalog, which is
    # the only place that can close this.
    assert "classify" in detail.lower() or "stamp" in detail.lower()
    # And it must NOT read as a refusal.
    assert "still serves" in detail.lower() or "serves" in detail.lower()


def test_the_confession_is_capable_of_going_red(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The instrument proves it can fail. With the emitter disarmed every
    assertion that reads the sink must fail while always-runs stays green —
    otherwise this file's loudness half is decorative."""
    monkeypatch.setattr(
        memory_mod, "_confess_serve_degrade", lambda **kw: None)

    with _Events() as ev:
        _preflight(UNSTAMPED, tmp_path)  # still runs

    assert ev.unevidenced() == [], (
        "the sink must be genuinely fed by the emitter under test")


def test_the_REQUEST_TIME_serving_shape_serves_and_confesses() -> None:
    """The production path, at its own shape — not the warm/preflight one.

    `executor.py:10081` builds every dispatched `RequestContext` with
    `_resolve_slots_kwargs(spec, order.slots, order.adapters)`, i.e. WITH
    per-dispatch `SlotOrder`s, where the tests above go through the WARM
    shape. That distinction is exactly what th#2099 turned on — 0.118.0
    "completing" proved the request path worked, and the same digest fatally
    refused on 0.120.0 — so a fix asserted only on the warm shape would leave
    the surface that actually took customer traffic unproven.
    """
    from gen_worker import dispatch
    from gen_worker.warmup import resolved_slots_kwargs

    chosen, siblings = _siblings()
    # The ref reaches the spec the way a dispatch puts it there; only the
    # FACTS are the variable under test.
    child_preflight.bind_slots(siblings, {"pipeline": child_contract.MintSlot(
        ref=PICKED, path="/tree", facts=UNSTAMPED)})
    # `SlotOrder.ref` is the WIRE string; the typed ref reaches the reader off
    # the spec, which is why `bind_slots` above is the half that matters here.
    slots = {"pipeline": dispatch.SlotOrder(
        ref="tensorhub/harness/catalog-pick@prod", facts=UNSTAMPED)}

    with _Events() as ev:
        out = resolved_slots_kwargs(chosen, slots)

    assert out["slot_errors"] == {}, (
        "a dispatched request must not lose its slot — this dict becoming "
        "non-empty is precisely how sd15 returned FATAL to a paying consumer")
    assert out["declared_slot_errors"] == ()
    assert out["resolved_slots"]["pipeline"].objective == ""
    assert len(ev.unevidenced()) == 1


# ---------------------------------------------------------------------------
# 3. THE LINE THAT DOES NOT MOVE — contradiction still refuses
# ---------------------------------------------------------------------------


def test_a_positively_mismatched_objective_STILL_REFUSES() -> None:
    """Always-runs was not bought by deleting the gate. A `flow` checkpoint in
    an `(epsilon, v_prediction)` function is EVIDENCE of incompatibility, not
    an absence of it."""
    with pytest.raises(ObjectiveMismatchError) as exc:
        _read(serving_facts.ServingFacts(
            objective="flow", distilled=False, distilled_status="classified"))
    assert "is not in the invoked function's declared objectives" in str(exc.value)


def test_a_classified_distilled_mismatch_STILL_REFUSES() -> None:
    """The same rule on the distillation axis: `classified` is real evidence,
    and real evidence that contradicts the declaration is a refusal."""
    with pytest.raises(ObjectiveMismatchError) as exc:
        _read(serving_facts.ServingFacts(
            objective="epsilon", distilled=True, distilled_status="classified"))
    assert "distilled=True" in str(exc.value)


def test_a_bogus_distilled_status_STILL_REFUSES() -> None:
    """A value from outside the vocabulary is a DECODE bug, not a degraded
    machine, and must not be laundered into a warning."""
    with pytest.raises(ObjectiveMismatchError):
        resolve_slot(
            "pipeline", _slot(), ref=PICKED, objective="epsilon",
            distilled=False, distilled_status="probably",
            allowed_objectives=SD15_DECLARES, allowed_distilled=False)


# ---------------------------------------------------------------------------
# 4. THE POSITIVE CONTROL — a readable objective is untouched
# ---------------------------------------------------------------------------


def test_a_fully_evidenced_checkpoint_resolves_with_NO_confession(
    tmp_path: Path,
) -> None:
    """The healthy path must stay silent. A `serve_degrade` row on a clean
    checkpoint would make the signal unreadable at fleet scale — and would
    mean the assertions above pass for the wrong reason."""
    with _Events() as ev:
        got = _read(EPSILON)
        _preflight(EPSILON, tmp_path)

    assert got.objective == "epsilon"
    assert got.distilled_status == "classified"
    assert ev.unevidenced() == [], (
        "an evidenced checkpoint is not a degraded run")


def test_a_function_declaring_nothing_is_untouched_and_silent(
    tmp_path: Path,
) -> None:
    """No declaration means nothing is checked, so an absent objective is not
    even a degradation — it is simply irrelevant. Confessing here would emit a
    warning for every family that never opted in."""
    with _Events() as ev:
        got = resolve_slot(
            "pipeline", _slot(), ref=PICKED, objective="",
            distilled=False, distilled_status="",
            allowed_objectives=None, allowed_distilled=None)

    assert got.objective == ""
    assert ev.unevidenced() == []
