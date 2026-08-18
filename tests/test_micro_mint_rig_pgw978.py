"""pgw#978 — the local micro-mint rig, as a row.

LOCAL-ONLY by classification, not by accident. CI has no GPU and no card-side
anything, and `scripts/skip_census.txt` carries these rows as `LOCAL-ONLY` so a
green CI never reads as coverage of them (pgw#966). Run them by hand:

    pytest tests/test_micro_mint_rig_pgw978.py -m localrig -p no:randomly

The rig's own driver (`scripts/micro_mint_rig.py`) is the thing under test. That
is deliberate: the driver is what a developer actually runs in the loop, so a
test that reimplemented the cycle beside it would drift from the thing it claims
to guard, and the first person to notice would be on a pod.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

torch = pytest.importorskip("torch")

#: pgw#966: the full cycle is OPT-IN, and its skip reason is stable so
#: `scripts/skip_census.txt` can classify it LOCAL-ONLY. Everything else in this
#: file is pure Python and RUNS IN CI — the gates, the carve-out bounds, the
#: handoff round-trip and the probe disarm are exactly the rows that must not
#: rot, and none of them needs a card.
RIG_OPT_IN = "PGW978_RIG"
_CYCLE_REASON = (
    "the pgw#978 micro-mint rig is opt-in: it runs a full local mint cycle "
    "(set PGW978_RIG=1)")


def _rig():
    import micro_mint_rig as rig

    return rig


def _gate() -> None:
    """The rig's own gates, applied to the test session.

    A row that ran the box over its load ceiling would be the same defect the
    rig refuses in its driver, so the refusal is honoured here rather than
    routed around with `--force-load`.
    """
    rig = _rig()
    try:
        rig.assert_load_gate()
        rig.assert_host_move_guard()
    except rig.RigRefused as exc:
        pytest.skip(f"micro-rig gate: {exc}")


# ---------------------------------------------------------------------------
# The gates are real refusals, and they are cheap to prove
# ---------------------------------------------------------------------------


def test_the_load_gate_refuses_a_busy_box() -> None:
    """The box is shared with several agent sessions; a compile started at load
    30 makes everyone slower including itself."""
    rig = _rig()
    with pytest.raises(rig.RigRefused, match="load"):
        rig.assert_load_gate(limit=-1.0)


def test_the_rig_refuses_to_run_with_the_host_move_guard_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A green cycle under a configuration production forbids is worse than no
    cycle. `GEN_WORKER_HOST_MOVE_GUARD=0` is a refusal, never a warning."""
    rig = _rig()
    monkeypatch.setenv("GEN_WORKER_HOST_MOVE_GUARD", "0")
    with pytest.raises(rig.RigRefused, match="HOST_MOVE_GUARD"):
        rig.assert_host_move_guard()


def test_the_device_is_resolved_and_reported_never_assumed() -> None:
    """pgw#983: this box's driver is CUDA 12.8 and the pinned torch is cu130, so
    the rig falls back to CPU — and must SAY which coverage it lost. A report
    whose green line implies device coverage it never had is the failure this
    field exists to prevent."""
    rig = _rig()
    dev = rig.resolve_device("auto")
    assert dev["device_kind"] in ("cuda", "cpu")
    assert dev["covers"]
    if dev["device_kind"] == "cpu":
        assert "NO VRAM cap" in dev["covers"]
        assert dev["why_not_cuda"]


# ---------------------------------------------------------------------------
# The toy checkpoint stays inside the policy carve-out
# ---------------------------------------------------------------------------


def test_the_generated_checkpoint_stays_under_the_carve_out_ceiling(
    tmp_path: Path,
) -> None:
    """The workspace policy's local-inference carve-out bounds the rig by SIZE
    and by ORIGIN. Both are enforced here rather than trusted: the tree is
    generated, and it is small."""
    from harness.tiny_diffusion import build_checkpoint, checkpoint_bytes

    rig = _rig()
    tree = build_checkpoint(tmp_path / "ckpt")
    size = checkpoint_bytes(tree)
    assert 0 < size < rig.MAX_WEIGHTS_BYTES
    # Deterministic: two builds agree, which is what lets a compiled graph key mean
    # anything across the two processes the rig runs.
    again = build_checkpoint(tmp_path / "ckpt2")
    assert checkpoint_bytes(again) == size


def test_the_device_budget_is_split_rather_than_assumed() -> None:
    """Two processes on one card must agree on the division up front. A leg
    written against 'the whole card' is how a co-resident pair OOMs the one that
    was merely second."""
    rig = _rig()
    # pgw#1175: the mint/adopt VRAM split is deleted with `vram_cap_bytes`,
    # the only thing that enforced it. The rig's remaining politeness levers
    # are the load gate and `compile_posture.USER_MACHINE`.
    assert not hasattr(rig, "MINT_VRAM_BYTES")
    assert not hasattr(rig, "ADOPT_VRAM_BYTES")


# ---------------------------------------------------------------------------
# The handoff is the real one — this is the pgw#969 guard
# ---------------------------------------------------------------------------


def test_a_slot_with_bytes_and_no_identity_cannot_reach_the_child() -> None:
    """pgw#969/pgw#974 in one row, at rig cost.

    The production crash was a request carrying a slot's PATH and no REF: it
    decoded, type-checked and looked complete, and the child died 0.0 s into
    `warmup_forward` at `ctx.slots["pipeline"]` on two L40S pods. The rig builds
    its slot through the real `MintSlot`, so the shape is unconstructable — and
    this row is what proves the rig did not route around the guard by
    hand-rolling a dict.
    """
    from gen_worker.child_contract import MintSlot

    with pytest.raises(TypeError):
        MintSlot(path="/tmp/x")  # type: ignore[call-arg]  # no ref


def test_the_request_the_rig_hands_the_child_carries_a_resolved_slot(
    tmp_path: Path,
) -> None:
    """Built through `mint_process.build_request` — the REAL parent chain, not
    a hand-written `MintRequest`. A hand-written request is the one shape the
    handoff can never produce, so it could not catch a handoff defect."""
    import msgspec

    from gen_worker.mint_process import MintRequest
    from harness.rig_vehicles import TINY
    from harness.tiny_diffusion import build_checkpoint

    rig = _rig()
    tree = build_checkpoint(tmp_path / "ckpt")
    # pgw#997: the vehicle is now an explicit argument — WHAT the rig mints is
    # a choice, so the handoff cannot read it off a module global.
    request = rig._mint_request(
        tmp_path / "mint", tree, TINY, ordinal=-1)
    # The boundary IS a JSON file, so round-trip it the way the child will.
    raw = msgspec.json.encode(request)
    decoded = msgspec.json.decode(raw, type=MintRequest)
    assert set(decoded.slots) == {"pipeline"}
    slot = decoded.slots["pipeline"]
    assert slot.ref.path == "rig/tiny-diffusion"
    assert slot.path == str(tree)
    # pgw#1010: no recipe axis — the child mints one artifact kind, so the
    # request carries the WORK ROOT it builds in instead of a choice.
    assert not hasattr(decoded, "recipe")
    assert decoded.work_root
    assert decoded.family == "microrig"


# ---------------------------------------------------------------------------
# The cycle itself
# ---------------------------------------------------------------------------


@pytest.mark.localrig
@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get(RIG_OPT_IN), reason=_CYCLE_REASON)
def test_the_full_machinery_cycle_runs_on_this_box(tmp_path: Path) -> None:
    """resolve -> handoff -> spawn -> load -> warm -> export -> seal -> publish
    -> adopt, end to end, in one process tree on this machine.

    This is the row that replaces four hours of publish/build/buy. Every leg is
    the production path; the only stand-ins are the hub (a local double speaking
    the real wire) and the model (generated, tiny, and never served to anyone).
    """
    _gate()
    rig = _rig()
    result = rig.run_cycle(tmp_path / "rig")

    names = [leg.name for leg in result.legs]
    assert names == ["gates", "weights", "handoff", "mint-child", "publish", "adopt"], (
        "a cycle that stops early must fail loudly, not silently report fewer legs: "
        + "\n".join(leg.line() for leg in result.legs))
    assert result.ok, "\n" + result.report()

    mint = next(leg for leg in result.legs if leg.name == "mint-child")
    phases = mint.facts["phase_seconds"]
    # The child really loaded the endpoint through `run_setup` — not a stub.
    assert float(phases.get("load", 0.0)) > 0.0
    # ...and really traced + compiled. pgw#984: the AOT recipe emits
    # `trace_graph`/`seal_publish`/`finalize` and NO `warmup_forward`, because
    # it exports rather than capturing a warm run. Asserting `warmup_forward`
    # here would be asserting a phase this recipe does not have.
    assert float(phases.get("trace_graph", 0.0)) > 0.0
    assert float(phases.get("seal_publish", 0.0)) > 0.0
    assert mint.facts["compiled_graph_key"], "the child sealed no compiled graph key"

    publish = next(leg for leg in result.legs if leg.name == "publish")
    routes = publish.facts["routes"]
    # The real seven-call publish protocol, not a one-shot POST.
    assert any(r.endswith("/publish-intent") for r in routes)
    assert any(r.endswith("/publishes") for r in routes)
    assert any(r.endswith("/complete") for r in routes)
    assert any(r.endswith("/publish-complete") for r in routes)
    assert publish.facts["cas_bytes"] > 0, "no compiled graph bytes reached the store"

    adopt = next(leg for leg in result.legs if leg.name == "adopt")
    assert adopt.facts["pid"] != os.getpid(), (
        "the adopt must run in a SECOND process — in-process adoption proves "
        "nothing about a compiled graph crossing pods")
    assert adopt.facts["compiled_graph_key"] == mint.facts["compiled_graph_key"], (
        "the compiled graph the second process adopted is not the compiled graph the first minted")


# ---------------------------------------------------------------------------
# pgw#980: the probe's publish disarm is structural
# ---------------------------------------------------------------------------


def test_a_probe_pod_cannot_publish_a_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    """A live-edit probe runs rsync'd code whose `gen_worker` version lies and
    whose `code_closure` no other pod can reproduce. Publishing from it would
    poison every pod that later adopts — so the refusal lives in the PARENT's
    action allowlist, which nothing swapped into the child can reach."""
    from gen_worker.procsplit import actions

    monkeypatch.setenv("GEN_WORKER_PROBE", "1")
    monkeypatch.delenv("GEN_WORKER_PROBE_PUBLISH_ARMED", raising=False)
    for path in ("/v1/worker/compiled-graphs/publish-intent",
                 "/v1/worker/compiled-graphs/publish-complete"):
        with pytest.raises(actions.ActionRefused, match="disarmed"):
            actions.authorize({"method": "POST", "path": path, "json": {}})
    # Discovery is untouched: a probe must still be able to ADOPT.
    action, _q, _b = actions.authorize({
        "method": "GET", "path": "/api/v1/repos/root/family-microrig/checkpoints",
        "query": {"limit": "50"}})
    assert action.name == "repo.checkpoints"


def test_arming_a_probes_publish_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two names, not one truthy flag: 'this is a probe' and 'this probe may
    write to the store' are different decisions, and the second must never be
    reachable by forgetting the first."""
    from gen_worker.procsplit import actions

    monkeypatch.setenv("GEN_WORKER_PROBE", "1")
    monkeypatch.setenv("GEN_WORKER_PROBE_PUBLISH_ARMED", "1")
    action, _q, _b = actions.authorize({
        "method": "POST", "path": "/v1/worker/compiled-graphs/publish-intent",
        "json": {"family": "microrig", "axes": {},
                 "entries": [{"compiled_graph_key": "cg-key-v1-" + "a" * 56,
                              "identity_axes": {}, "mint_duration_ms": 1}]}})
    assert action.name == "compiled_graphs.publish_intent"


def test_a_normal_pod_is_unaffected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard must cost a serving pod nothing. No probe marking, no change."""
    from gen_worker.procsplit import actions

    monkeypatch.delenv("GEN_WORKER_PROBE", raising=False)
    monkeypatch.delenv("GEN_WORKER_PROBE_PUBLISH_ARMED", raising=False)
    assert not actions.publish_disarmed()
    action, _q, _b = actions.authorize({
        "method": "POST", "path": "/v1/worker/compiled-graphs/publish-complete",
        "json": {"family": "microrig", "compiled_graph_key": "ck5-" + "a" * 56,
                 "checkpoint_id": "sha256:" + "b" * 64, "ok": True}})
    assert action.name == "compiled_graphs.publish_complete"
