"""pgw#1175 / §4.33: the ADOPT'S device cost is ATTEMPTED, never predicted.

WHAT WAS DELETED, AND WHAT STANDS IN ITS PLACE.

``mint_budget.adopt_headroom`` refused an arm before it ran, on
``need = 2 * activation`` where ``activation`` was
``max(measured_high_water, 0.25 * allocated)`` — a quarter of the RESIDENT SET
whenever no forward had run yet, compared against a ``free_bytes`` those very
weights are already outside of. Its own docstring conceded the arithmetic
"CANNOT refuse a card that merely cannot hold 36 runners", i.e. it could not
refuse the failure it was written for (th#1825), and it COULD refuse cards that
were fine — stickily, for the life of the process. Two call sites ran it:
``provision.arm_aot`` (every arm route) and ``fleet_compiled_graphs.adopt_delegated_mint``
(the self-mint route). Both are gone.

What refuses now is the bind itself. ``aot_serve.arm_entry`` attempts every
entry inside ``_bind_headroom``; a real CUDA OOM comes back as a typed
``insufficient_adopt_vram`` ``AdoptError`` that NAMES the entry, before the
first live mutation, so the pod serves eager and stays up.

Three claims, and each is a DIFFERENT thing the deleted gate got wrong:

1. the refusal is CLASSIFIED (the deleted gate's token, on evidence);
2. it NAMES the entry that did not fit (the gate could not — it never got far
   enough to know, which is why th#1825 read as "1.9 MB free of 47.7 GB" with
   nothing to attribute it to);
3. it is NOT STICKY — a second arm of the same family re-attempts and succeeds
   when the card has room, where ``note_adopt_declined`` made the first refusal
   final for the process.

And the control: an adopt that fits still arms, so the guard is not a blanket
refusal wearing a classification.

The vehicle is :mod:`harness.adopt_rig` — the real boot-adopt chain onto a real
packed compiled graph. The OOM is forced by BREAKING A REAL INPUT (the packaged entry's
constant load really raises ``torch.OutOfMemoryError``, which is the device
work an adopt does), never by supplying a fact.

RED ON ``origin/master`` — the assertions below grafted onto master's UNMODIFIED
``src/`` (only this file and the harness's ``bind_oom_on`` input were copied
across), 5 of 7 fail:

* rows 1, 2, 4 and 6 — master turns a bind-time CUDA OOM into
  ``constants_injection_failed``, a CONTRACT verdict: ``ArtifactRunner.bind``
  re-labels every ``RuntimeError`` out of ``load_constants`` and
  ``torch.OutOfMemoryError`` is one. So on master a FULL CARD condemns a
  CORRECT compiled graph, and the pod records the compiled graph's identity as the culprit;
* row 6 also proves ``mint_budget`` is still importable there.

Row 3 (not-sticky) PASSES on master, and that is stated rather than hidden:
``adopt_headroom`` never declines on a cardless box, so ``note_adopt_declined``
never fires off-pod and the stickiness cannot be exhibited here. It is a
REGRESSION FENCE on the deletion, not a re-found bug.

Run: uv run pytest tests/test_bind_is_the_budget_pgw1175.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gen_worker import aot_serve
from gen_worker.models import provision

from harness import exported_compiled_graph as compiledgraph868
from harness.adopt_rig import AdoptRig, production_cfgs
from harness.receipt_hub import HubStub, hub, pub_map, rsa_key  # noqa: F401

# The SECOND of the compiled graph's two entries IN BIND ORDER (``sorted(entries)``, so
# h=8 follows h=16): a refusal must be attributable to the entry that did not
# fit, and picking the first could not tell "named the entry" apart from
# "named the first thing it saw".
OOM_ENTRY = compiledgraph868.entry_name(8, 8)

#: The classification, written LITERALLY so every row below is behavioural on
#: an unmodified tree — an assertion that reads a constant the fix introduces
#: goes red for the wrong reason, and proves nothing about what the tree does.
OOM_REASON = "insufficient_adopt_vram"


def _adopt_with_oom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: HubStub,  # noqa: F811
    entry: str = OOM_ENTRY
) -> object:
    """One real boot-adopt whose ``entry``'s bind really OOMs the card."""
    return AdoptRig(tmp_path, monkeypatch, hub, bind_oom_on=entry).boot()


# ===========================================================================
# 1. the refusal is CLASSIFIED — on evidence, not on an estimate
# ===========================================================================


def test_a_bind_that_ooms_is_a_classified_adopt_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: HubStub,  # noqa: F811
) -> None:
    boot = _adopt_with_oom(tmp_path, monkeypatch, hub)

    assert boot.is_armed() is False, (
        "a compiled_graph whose bind ran the card out of memory must not serve")
    said = [f"{k}|{phase}|{d}" for k, phase, d in boot.events]
    assert any(OOM_REASON in row for row in said), (
        f"the OOM was not classified as {OOM_REASON!r}; "
        f"the pod cannot tell 'this card is full' from 'this compiled_graph is broken'. "
        f"Saw: {said!r}")
    assert not any("constants_injection_failed" in row for row in said), (
        "the refusal reached the wire as a CONTRACT verdict — "
        "`ArtifactRunner.bind` re-labels every RuntimeError out of "
        "`load_constants`, and `torch.OutOfMemoryError` IS one, so a full "
        "card would condemn a correct compiled_graph")


def test_the_refusal_names_the_entry_that_did_not_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: HubStub,  # noqa: F811
) -> None:
    """The deleted gate could not do this: it refused BEFORE the load, so it
    never knew which entry it was protecting the card from. th#1825's whole
    forensic problem in one line."""
    boot = _adopt_with_oom(tmp_path, monkeypatch, hub)

    said = " ".join(d for _k, _p, d in boot.events)
    assert OOM_ENTRY in said, (
        f"nothing on the wire names entry {OOM_ENTRY!r}, so a reader cannot "
        f"tell a compiled_graph that is one entry too big from one that is wholly "
        f"unadoptable. Saw: {said!r}")
    # pgw#1176 RE-BASED THE POSITION, and the claim is sharper for it. The old
    # "(2 of 2)" measured how far through a bind-all-then-wrap sequence the OOM
    # landed — the only handle th#1825 had. Each class binds ALONE now, so that
    # index is always 1 of 1 and says nothing. What distinguishes "one class
    # too big" from "wholly unadoptable" is how many siblings are ALREADY
    # ARMED and still serving, which is a fact about what the pod is doing
    # rather than about a loop it happened to be in.
    assert "already armed)" in said, (
        "the refusal does not say WHERE in the bind it happened; an entry "
        "index is what makes 'nearly fit' measurable")


# ===========================================================================
# 2. NOT STICKY — `note_adopt_declined` made the first refusal final
# ===========================================================================


def test_a_refusal_does_not_poison_the_next_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: HubStub,  # noqa: F811
) -> None:
    """pgw#1169's stickiness is deleted with the bank it was made of.

    A card's free memory moves with what is resident — that is why the sticky
    set could not live in ``compile_cache.arming_block``, whose invariant is
    that every reason it names is deterministic for the life of the process.
    A refusal made of a NON-deterministic input must not be remembered.
    """
    refused = _adopt_with_oom(tmp_path, monkeypatch, hub)
    assert refused.is_armed() is False

    # Same process, same family, same lane, a card that now has room.
    ok = AdoptRig(tmp_path, monkeypatch, hub).boot()
    assert ok.is_armed() is True, (
        "the second arm inherited the first one's refusal — a decline made of "
        "free VRAM was remembered across a change in free VRAM")
    assert ok.serves_compiled()


# ===========================================================================
# 3. the control — the guard refuses only what actually fails
# ===========================================================================


def test_an_adopt_that_binds_still_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: HubStub,  # noqa: F811
) -> None:
    boot = AdoptRig(tmp_path, monkeypatch, hub).boot()

    assert boot.adopted()
    assert boot.is_armed() is True
    assert boot.serves_compiled()
    assert not [d for _k, phase, d in boot.events if phase == OOM_REASON]


# ===========================================================================
# 4. the guard is NARROW — only an OOM is a capacity verdict
# ===========================================================================


def test_a_non_oom_bind_failure_keeps_its_own_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: HubStub,  # noqa: F811
) -> None:
    """"the artifact is broken" and "this pod is full" are different verdicts,
    and only one of them says anything about the compiled graph. A guard that swallowed
    both would retire correct compiled graphs and excuse broken ones in the same line."""
    real = aot_serve.ArtifactRunner.bind

    def _boom(self: object, *a: object, **k: object) -> None:
        raise RuntimeError("the .so exports no such symbol")

    monkeypatch.setattr(aot_serve.ArtifactRunner, "bind", _boom)
    boot = AdoptRig(tmp_path, monkeypatch, hub).boot()
    assert real is not None

    assert boot.is_armed() is False
    said = " ".join(d for _k, _p, d in boot.events)
    assert OOM_REASON not in said, (
        "a structural bind failure was reported as a VRAM shortfall; the pod "
        "will now blame its card for a broken artifact forever")


# ===========================================================================
# 5. nothing predicts VRAM any more — the estimate surface is gone
# ===========================================================================


def test_no_module_predicts_adopt_vram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§4.33's standing rule, asserted structurally rather than by reading.

    ``mint_budget`` is deleted whole; ``provision.arm_aot`` and
    ``fleet_compiled_graphs`` reach no headroom verdict; and the pool's K carries no
    device term. Each is a thing that USED to compute a number nobody had
    measured, and would silently come back as a "safe" refusal.
    """
    with pytest.raises(ModuleNotFoundError):
        __import__("gen_worker.mint_budget")

    from gen_worker import aot_compile_pool, fleet_compiled_graphs

    for mod in (provision, fleet_compiled_graphs):
        assert not hasattr(mod, "mint_budget"), (
            f"{mod.__name__} still holds a budget module")

    width = aot_compile_pool.entry_workers(36, vcpus=32, available_bytes=64 * 1024**3,
        peak_rss_bytes=2 * 1024**3, device_lock=True)
    facts = width.facts()
    assert not [k for k in facts if "device" in k and k != "device_lock"], (
        f"K still reports a device term: {sorted(facts)!r}")
    assert width.workers > 1, (
        "a 32-core pod with 64 GiB and a 2 GiB measured child still compiles "
        "serially — the §4.33 arithmetic (36 entries / K) needs K to move")


def test_the_constant_is_the_token_downstream_already_reads() -> None:
    """One vocabulary. The deleted gates emitted this exact token, so the hub's
    phase column, ``fleet_compiled_graphs``' abort classification and every dashboard
    keep working — what changed is only that it is now said on evidence."""
    assert aot_serve.ADOPT_OOM_REASON == OOM_REASON


def test_torch_oom_is_what_the_guard_catches() -> None:
    """The type the guard is written against is the type a card raises —
    asserted here so a torch rename cannot make the guard silently inert."""
    from gen_worker.models.memory import is_cuda_oom

    assert is_cuda_oom(torch.OutOfMemoryError("CUDA out of memory."))
