"""TESTS MUST DRIVE THE PRODUCTION ENTRYPOINT, NOT THE UNIT BENEATH IT.

The defect class is correct code, green unit tests, and no route that reaches
it. Unit level is exactly where a wiring defect is invisible, because the unit
test IS the caller that production is not. This file is what enforces the rule.

The mechanism: declare each path's production call chain as a LADDER, drive a
real scenario, and assert the deepest rung was reached WITH THE WHOLE CHAIN
ABOVE IT — i.e. production did the calling. See ``entrypoint_ladder.py``.

The ledger below is the deliverable, in both directions:

* ``COVERED`` ladders are asserted green here. One that stops being driven from
  its front door — a route deleted, a rung renamed, a wrapper that stops being
  entered — fails, loudly, in CI.
* ``UNCOVERED`` ladders are the known gaps, each with the
  rung the traversal stops at and WHY. They are asserted to still be
  uncovered, so closing one is a deliberate edit to this file rather than a
  silent drift. That is the same contract ``scripts/check_registry_contract.py``
  holds the registry to: state changes in either direction must be deliberate.

An UNCOVERED entry is a debt, not a resting state. Each names its owner issue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import msgspec
import pytest

from entrypoint_ladder import _resolve, ladder
from gen_worker.pb import worker_scheduler_pb2 as pb
from harness.hub_double import hub_double, is_ready, is_result_for

# ---------------------------------------------------------------------------
# THE LEDGER
# ---------------------------------------------------------------------------

#: Path -> the production call chain, front door first.
LADDERS: Dict[str, Tuple[str, ...]] = {
    "boot": (
        "gen_worker.worker.Worker.arun",
        "gen_worker.lifecycle.Lifecycle.startup",
        "gen_worker.executor.Executor.ensure_setup",
    ),
    "serve": (
        "gen_worker.worker.Worker.arun",
        "gen_worker.transport.Transport._recv_loop",
        "gen_worker.lifecycle.Lifecycle.on_message",
        "gen_worker.executor.Executor.handle_run_job",
        "gen_worker.executor.Executor._execute",
    ),
    "mint": (
        "gen_worker.executor.Executor.ensure_setup",
        "gen_worker.executor.Executor._background_mint",
        "gen_worker.executor.Executor._supervise_mint",
        "gen_worker.mint_supervisor.supervise",
        "gen_worker.aot_mint.mint_graph_classes",
    ),
    "arm": (
        "gen_worker.executor.Executor._enable_compiled",
        "gen_worker.fleet_cells.enable_compiled",
        "gen_worker.models.provision.enable_compiled",
        "gen_worker.models.provision.arm_aot",
        "gen_worker.aot_serve.enable",
    ),
    "publish": (
        "gen_worker.executor.Executor._supervise_mint",
        "gen_worker.fleet_cells.adopt_delegated_mint",
        "gen_worker.fleet_cells.publish_self_mint",
        "gen_worker.executor.Executor._advertise_compiled_graphs",
    ),
}

#: The measured gaps. ``path -> (rung the traversal stops at, why, owner)``.
UNCOVERED: Dict[str, Tuple[str, str, str]] = {
    "mint": (
        "gen_worker.executor.Executor._background_mint",
        "No test boots a worker and lets ensure_setup start a real background "
        "mint. The chain is covered in disjoint segments and never joined: "
        "test_eager_first_boot_pgw671 enters at ensure_setup with faked "
        "leaves; the pgw#1215 family enters at mint_supervisor with a stubbed "
        "compile pool; test_aot_mint_pgw723 does a REAL torch.export at the "
        "bottom with no pod above it. tests/harness/mint_endpoints_pgw784.py "
        "says it outright — it 'spawns the mint child exactly as mint_supervisor "
        "does', i.e. the harness RE-IMPLEMENTS the production caller instead "
        "of driving it, which is this defect class inside the test layer.",
        "pgw#849 follow-up, owner: the mint lane",
    ),
    "arm": (
        "gen_worker.executor.Executor._enable_compiled",
        "Nothing enters at Executor._enable_compiled from a booted worker. "
        "fleet_cells.enable_compiled has ~29 direct callers in tests and "
        "aot_serve.enable has 2, all below the executor. The one file that "
        "tried to instrument the arm end to end, "
        "test_boot_phases_arm_pgw764.py, declares itself STAGED AND NOT "
        "COMMITTABLE in its own docstring. This is the exact shape of "
        "pgw#844: aot_serve.set_guard_failure_callback was unit-green and had "
        "no production caller, so every AOT arm was unadvertisable. MEASURED "
        "2026-08-01: the ladder was also driven against "
        "harness/boot_ladder_endpoints_pgw797.py, the one compile-declaring "
        "slot endpoint the harness owns, and _enable_compiled is NEVER "
        "REACHED there either — that boot does not run ensure_setup at all. "
        "So no available scenario arms.",
        "pgw#849 follow-up, owner: the arm lane",
    ),
    "publish": (
        "gen_worker.executor.Executor._supervise_mint",
        "CellPublisher.publish has a genuinely real test "
        "(test_cell_publish_v2_pgw807: real sockets, real multi-MB bytes, real "
        "chunked sha256) but it enters at the publisher. Nothing drives "
        "_supervise_mint -> adopt_delegated_mint -> publish_self_mint -> "
        "_advertise_compiled_graphs as one chain, and it is "
        "where active_compile_ref / active_self_mint become visible to the hub."
        " Blocked behind the mint ladder: publish has no input until a mint "
        "front-door test exists.",
        "pgw#849 follow-up, owner: the mint lane",
    ),
}


# ---------------------------------------------------------------------------
# 0. The ledger itself must be real.
# ---------------------------------------------------------------------------

def test_every_declared_rung_exists() -> None:
    """A renamed or deleted rung must break the ledger, not be skipped by it.

    RED by construction: change any dotted path below and this fails.
    """
    for path, rungs in LADDERS.items():
        for rung in rungs:
            owner, attr = _resolve(rung)
            assert callable(getattr(owner, attr)), f"{path}: {rung}"


def test_uncovered_entries_name_a_real_rung_and_an_owner() -> None:
    for path, (rung, why, owner) in UNCOVERED.items():
        assert path in LADDERS, f"{path} is not a declared ladder"
        assert rung in LADDERS[path], f"{path}: {rung} is not one of its rungs"
        assert len(why) > 80, f"{path}: a gap needs a reason, not a label"
        assert "owner" in owner, f"{path}: a gap needs an owner"


# ---------------------------------------------------------------------------
# 1. COVERED — asserted green. These must stay driven from the front door.
# ---------------------------------------------------------------------------

def _drive_boot_and_one_job() -> Tuple[object, object]:
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="pgw849-r1", attempt=1, function_name="echo",
            input_payload=msgspec.msgpack.encode({"text": "marco"})))
        conn.wait_for(is_result_for("pgw849-r1"))
    return scheduler, _harness


def test_boot_ladder_is_driven_from_the_process_front_door() -> None:
    with ladder(*LADDERS["boot"]) as rec:
        _drive_boot_and_one_job()
    assert rec.gap() is None, (
        "the boot ladder was not traversed from its front door:\n"
        + rec.report())


def test_serve_ladder_is_driven_from_the_wire() -> None:
    """A real job, over a real gRPC socket, into a real Worker — and every rung
    from the receive loop down entered BY PRODUCTION.

    This is the standard the other ladders are measured against. Shown RED by
    mutation: break any link in transport -> lifecycle -> executor and
    ``rec.gap()`` names it.
    """
    with ladder(*LADDERS["serve"]) as rec:
        _drive_boot_and_one_job()
    assert rec.gap() is None, (
        "the serve ladder was not traversed from the wire:\n" + rec.report())

    execs = rec.entries["gen_worker.executor.Executor._execute"]
    assert any(e.from_production for e in execs), (
        "_execute ran, but the TEST called it — that is a unit test wearing an "
        "integration test's clothes:\n" + rec.report())


# ---------------------------------------------------------------------------
# 1b. CALIBRATION — the instrument must be able to SEE the defect.
#
# A guard that cannot be shown failing is not evidence (pgw#853's standard, and
# pgw#784's red arm). So this drives the arm ladder the way a unit test drives
# it — by calling the leaf directly — and asserts the recording says so.
# ---------------------------------------------------------------------------

def test_the_instrument_sees_a_unit_level_test() -> None:
    """The red arm. Call ``aot_serve.enable`` the way ~29 files call
    ``fleet_cells.enable_compiled`` today — straight from the test, with no
    executor above it — and the ladder must report the chain broken and the
    call attributed to the TEST, not to production.

    If this ever passes silently, the instrument has stopped working and every
    green assertion above it is vacuous.
    """
    import gen_worker.aot_serve as aot_serve

    with ladder(*LADDERS["arm"]) as rec:
        try:
            aot_serve.enable(object(), object())      # signature is irrelevant
        except Exception:
            pass                                      # the ENTRY is the datum

    leaf = "gen_worker.aot_serve.enable"
    assert rec.reached(leaf), "the leaf never recorded — instrument is broken"
    calls = rec.entries[leaf]
    assert not any(c.from_production for c in calls), (
        "the leaf was entered from production in a test that called it "
        "directly — the caller attribution is wrong:\n" + rec.report())
    assert rec.gap() == LADDERS["arm"][0], (
        "the ladder did not notice that every rung above the leaf was "
        "skipped:\n" + rec.report())


# ---------------------------------------------------------------------------
# 1c. FRESH PROCESS — "entered at the production entrypoint" is not enough if
#     the test process already did the entrypoint's work.
#
# pytest shares sys.modules for a whole worker session, so an in-process boot
# starts with the imports, registrations and module-scope side effects a real
# boot performs ALREADY DONE — by an earlier test, or by this file's own
# imports. The pgw#853 lane's first attempt passed dishonestly for exactly that
# reason: an already-imported declaration module does not re-register, so the
# defect the test existed to catch was invisible to it. They moved the proof to
# a subprocess on the reasoning that a pod boot IS a fresh process, and that
# reasoning generalises: a guard that certifies an in-process test as covering
# a boot path certifies the false negative.
#
# So the boot ladder is certified twice, and the two are not interchangeable.
# ---------------------------------------------------------------------------

#: Per ladder: is its certification a fresh interpreter, or in-process only?
CERTIFICATION: Dict[str, str] = {
    "boot": "fresh-process",
    "serve": "in-process",     # needs a live hub-double wire; see note below
    "mint": "none",
    "arm": "none",
    "publish": "none",
}


# STATUS 2026-08-01, measured, NOT yet a gate. The mechanism is built —
# ``tests/harness/ladder_boot_main.py`` wraps the ladder, hands control to the
# real ``gen_worker.entrypoint`` through ``runpy``, and dumps the recording — and
# two facts came out of driving it:
#
#   1. ``Lifecycle.startup`` DOES NOT RUN until the transport has a hub. A fresh
#      boot against a dead address (the shape every existing boot-smoke test
#      uses) therefore certifies nothing below ``Worker.arun``. Any future
#      fresh-process ladder has to bind a real hub-double socket and point the
#      child at it — double in the test process, worker in a fresh one.
#   2. Doing exactly that did not complete inside a 120s budget and is not
#      root-caused. Shipping it as a gate would ship a flake, and a flaky guard
#      gets disabled, which is worse than not having it.
#
# So the boot ladder's certification stays declared IN-PROCESS below, which is
# the honest label, and the upgrade is a named follow-up rather than a skipped
# test sitting here going stale.


def test_certification_ledger_matches_the_ladders() -> None:
    for path in LADDERS:
        assert path in CERTIFICATION, f"{path} has no declared certification"
    for path, level in CERTIFICATION.items():
        assert level in ("fresh-process", "in-process", "none")
        if level == "none":
            assert path in UNCOVERED, (
                f"{path} claims no certification but is not recorded as a gap")
        else:
            assert path not in UNCOVERED, (
                f"{path} is certified {level} and also recorded as a gap")


# ---------------------------------------------------------------------------
# 2. UNCOVERED — the measured gaps, asserted to still be gaps.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", sorted(UNCOVERED))
def test_uncovered_ladders_are_still_uncovered(path: str) -> None:
    """Closing one of these is a WIN — and it must be a deliberate edit here.

    The assertion is inverted on purpose. A ladder recorded as uncovered that
    quietly starts passing means somebody built the front-door test and the
    ledger no longer describes the tree; a ledger nobody has to update is a
    ledger nobody reads. Delete the UNCOVERED entry in the same commit that
    covers the path.
    """
    stops_at, _why, _owner = UNCOVERED[path]
    with ladder(*LADDERS[path]) as rec:
        _drive_boot_and_one_job()
    gap = rec.gap()
    assert gap is not None, (
        f"{path} now traverses from its front door — DELETE its UNCOVERED "
        f"entry in this file:\n" + rec.report())
    assert gap == stops_at, (
        f"{path} stops at {gap!r}, but the ledger says {stops_at!r}. The "
        f"wiring moved; update the ledger:\n" + rec.report())
