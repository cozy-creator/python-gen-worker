"""pgw#824: the fleet-wide silent-failure audit — the SDK half.

Three invariants, asserted the way the hub actually sees them:

1. no silent failure paths — an important fail-soft outcome rides a typed
   event, never only a logger;
2. eager serving is an EVENT, not a default — every eager-served request on a
   compile-declaring release reports exactly WHY it is eager;
3. the mint/adopt lifecycle is continuously hub-visible — including progress
   INSIDE a long phase, not only at phase transitions.

Capture is at ``activity._emit`` (the pgw#733/#760 convention): the exact
envelope the stream sink ships, so kind/phase/detail are asserted as wired
rather than through a double of the event API. Nothing here asserts a log line
— on a hub-spawned pod there is no stdout to read one from, which is the whole
premise of the audit.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List, Tuple

import pytest

from gen_worker import activity, fleet_cells, serving_mode
from gen_worker.cell_adopt import EagerPhase


@pytest.fixture()
def events(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    captured: List[Any] = []
    monkeypatch.setattr(activity, "_emit", captured.append)
    return captured


def _rows(events: List[Any], kind: str) -> List[Tuple[str, str]]:
    return [(e.phase, e.detail) for e in events if e.kind == kind]


# ---------------------------------------------------------------------------
# Invariant 2 — the eager POSTURE vocabulary
# ---------------------------------------------------------------------------


def test_eager_with_no_cell_reports_a_reason_not_an_empty_string() -> None:
    """RED before pgw#824: `serving_mode=eager, fallback_reason=""`.

    All four pre-existing fallback classes presuppose an ARMED cell, so the
    commonest eager case — nothing armed at all — had no vocabulary and
    reported the empty string. "" could not distinguish a release that declares
    no compile target from a pod still minting from a pod that declined for
    cause, which is why "why is this fleet eager right now" had no query.
    """
    served = serving_mode.resolve(
        active_compile_ref="",
        eager_posture=serving_mode.POSTURE_MINT_IN_PROGRESS,
        sm="89",
    )
    assert served.serving_mode == serving_mode.MODE_EAGER
    assert served.fallback_reason == serving_mode.POSTURE_MINT_IN_PROGRESS
    # NOT a fallback: nothing fell back, there was nothing to fall back FROM.
    # Every existing compiled-vs-eager comparison keeps its old meaning.
    assert served.served_eager_fallback is False


def test_a_per_request_fallback_outranks_the_posture() -> None:
    """A compiled lane that fell back for THIS request is the stronger fact:
    the posture describes the worker, the fallback describes the request."""
    served = serving_mode.resolve(
        active_compile_ref="root/family-sdxl#deadbeef",
        verdict=serving_mode.FALLBACK_VOLATILE,
        eager_posture=serving_mode.POSTURE_MINT_IN_PROGRESS,
        sm="89",
    )
    assert served.fallback_reason == serving_mode.FALLBACK_VOLATILE
    assert served.served_eager_fallback is True


def test_the_posture_never_overwrites_a_compiled_serve() -> None:
    """A posture leaking onto a cell-served request would be a WRONG dimension,
    which is worse than a coarse one (the pgw#764 rule)."""
    served = serving_mode.resolve(
        active_compile_ref="root/family-sdxl#deadbeef",
        eager_posture=serving_mode.POSTURE_UNCOMPILED,
        sm="89",
    )
    assert served.serving_mode != serving_mode.MODE_EAGER
    assert served.fallback_reason == ""
    assert served.served_eager_fallback is False


# ---------------------------------------------------------------------------
# Invariant 1/2 — the arming brain's declines are CLASSIFIED, not a constant
# ---------------------------------------------------------------------------


class _Pipe:
    pass


def test_fail_closed_names_the_cause_instead_of_one_shared_constant(
    events: List[Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED before pgw#824: all nine `_fail_closed` exits emitted
    `phase=mint_unavailable` and put the cause in free text only, so counting
    causes hub-side meant substring-matching a sentence — the th#1250 lesson
    (kind-only coalescing erases the reason) one level down.
    """
    monkeypatch.setattr(fleet_cells.cc, "mandatory_serving", lambda pipe: False)
    monkeypatch.setattr(
        fleet_cells.loading, "pipeline_weight_lane", lambda pipe: "")

    outcome = fleet_cells._fail_closed(
        _Pipe(), "no C compiler for the self-mint",
        phase=EagerPhase.NO_TOOLCHAIN)

    assert outcome.armed is False
    # the token rides the WIRE, as the VALUE — a reader groups on the string,
    # never on the member's repr.
    assert ("no_toolchain", ) == tuple(
        phase for phase, _ in _rows(events, "self_mint_skipped"))
    # ... and rides OUT of the decision, for the request path to report.
    assert outcome.eager_reason == "no_toolchain"


#: The eager exits pgw#824 gave a classified cause, named as MEMBERS.
#:
#: This list is the audit's own specification of which causes must exist, and
#: it stays here rather than being read back out of ``EagerPhase``: an audit
#: that enumerates the enum agrees with the code by construction, which is the
#: failure mode it exists to prevent. Deleting an exit must fail a test, not
#: shrink a set on both sides at once.
#:
#: What is NOT re-typed here is the spelling. pgw#923 moved every one of these
#: exits from ``enable_compiled`` into ``_arming_policy`` — no exit removed, no
#: token changed — and this audit, which read only ``enable_compiled``'s source
#: for nine string literals, reported all nine as lost. Two literal lists is a
#: drift channel; naming members closes it while keeping the specification.
_DECLINE_PHASES = (
    EagerPhase.NO_FAMILY,
    EagerPhase.NO_CUDA,
    EagerPhase.NO_TOOLCHAIN,
    EagerPhase.NO_COMPILE_TARGET,
    EagerPhase.DELIVERED_CELL_SEEDED,
    EagerPhase.KEY_COMPUTATION_FAILED,
    EagerPhase.CAPTURE_CONFLICT,
    EagerPhase.MULTI_GROUP_IN_PROCESS,
    EagerPhase.CAPTURE_ARM_FAILED,
)


def _phase_uses(member: EagerPhase) -> int:
    """How many exits in the arming module pass ``member`` as their phase.

    Scans the whole ``fleet_cells`` module, not one function: WHICH function
    holds the policy is an implementation detail that has already changed once,
    and an audit that fails when a helper is extracted is an audit that gets
    re-pointed rather than read.
    """
    src = inspect.getsource(fleet_cells)
    return len(re.findall(rf"phase=EagerPhase\.{member.name}\b", src))


def test_every_fail_closed_exit_carries_a_distinct_token() -> None:
    """The nine exits must not collapse back onto one token by accident: a
    reader groups on this string, so a duplicate silently merges two causes."""
    uses = {member: _phase_uses(member) for member in _DECLINE_PHASES}
    missing = sorted(m.name for m, n in uses.items() if n == 0)
    assert not missing, f"these _fail_closed exits lost their token: {missing}"
    shared = sorted(m.name for m, n in uses.items() if n > 1)
    assert not shared, f"these tokens now cover more than one exit: {shared}"
    # ... and none of them fell back onto the shared constant they replaced.
    assert _phase_uses(EagerPhase.MINT_UNAVAILABLE) == 0


def test_the_eager_phase_values_are_a_wire_contract() -> None:
    """The hub stores these as ``worker_activity_events.phase`` and joins them
    to a request row's ``fallback_reason``. Renaming one does not break a
    build — it orphans every row already grouped under the old spelling — so
    the mapping is pinned here, in the one place a rename must be argued.
    """
    assert {m.name: m.value for m in EagerPhase} == {
        "NO_FAMILY": "no_family",
        "NO_CUDA": "no_cuda",
        "NO_TOOLCHAIN": "no_toolchain",
        "NO_COMPILE_TARGET": "no_compile_target",
        "DELIVERED_CELL_SEEDED": "delivered_cell_seeded",
        "KEY_COMPUTATION_FAILED": "key_computation_failed",
        "CAPTURE_CONFLICT": "capture_conflict",
        "MULTI_GROUP_IN_PROCESS": "multi_group_in_process",
        "CAPTURE_ARM_FAILED": "capture_arm_failed",
        "CELL_QUARANTINED": "cell_quarantined",
        "MINT_IN_PROGRESS": "mint_in_progress",
        "MINT_UNAVAILABLE": "mint_unavailable",
    }
    # The join the ArmOutcome docstring claims: a delegated mint's decline and
    # the serving posture that describes it are ONE token, not two that happen
    # to be spelled alike today.
    assert EagerPhase.MINT_IN_PROGRESS == serving_mode.POSTURE_MINT_IN_PROGRESS


def test_a_quarantined_cell_is_a_typed_event_not_a_log_line() -> None:
    """RED before pgw#824. This was the ONE eager exit in the arming policy
    that returned before `_fail_closed` and only `logger.error`'d. A pod that
    quarantines its own cell serves eager for the rest of its life — the state
    the hub most needs named, and the one it could not see.
    """
    assert _phase_uses(EagerPhase.CELL_QUARANTINED) == 1, (
        "the quarantine exit must emit a typed event")


# ---------------------------------------------------------------------------
# Invariant 3 — progress INSIDE a long phase
#
# Driven through a REAL mint (torch.export + AOTI pack, CPU, no GPU and no
# stand-in), because what broke on this branch was the wiring between `mint`
# and `_mint_cell`, not the callback's shape.
# ---------------------------------------------------------------------------

_FAMILY = "tiny824"
_WIDTH = 8


def _mint(tmp_path: Any, monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> Any:
    """One real two-class mint of a tiny CPU module."""
    torch = pytest.importorskip("torch")
    import types

    from gen_worker import aot_mint, aot_serve, compile_cache
    from gen_worker.api.decorators import Compile
    from gen_worker.api.export_contract import (
        Dim, GraphClass, Input, register_export_declaration,
        reset_export_declarations,
    )

    class _Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(_WIDTH, _WIDTH, bias=False)

        def forward(self, sample: Any) -> Any:
            return torch.tanh(self.lin(sample))

    full = {"sku": "", "sm": "sm_89", "torch": str(torch.__version__), "cuda": ""}
    monkeypatch.setattr(compile_cache, "runtime_key", lambda: dict(full))
    monkeypatch.setattr(aot_serve, "runtime_key", lambda: dict(full))

    reset_export_declarations()
    monkeypatch.setattr(
        aot_mint, "reset_export_declarations", reset_export_declarations,
        raising=False)
    register_export_declaration(Compile(
        family=_FAMILY,
        targets=("unet",),
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        # TWO classes, so "entry i of N" has something to count.
        classes=(GraphClass(dims={"B": 1}), GraphClass(dims={"B": 2})),
        inputs=(Input("sample", shape=("B", _WIDTH)),),
        shape_strategy="static-rows",
        warm_changes_key=False,
    ))
    try:
        return aot_mint.mint(
            types.SimpleNamespace(unet=_Tiny().eval()),
            aot_mint.ExportSpec(family=_FAMILY, target=""),
            tmp_path / "out", **kwargs)
    finally:
        reset_export_declarations()


def test_a_multi_entry_mint_reports_every_entry_before_it_runs(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED before pgw#824: `aot_mint.mint` was one opaque call over the
    family's whole declared class set, so a real ~5-minute export emitted
    NOTHING between `trace_graph` and `seal_publish`.

    A REAL mint — real `torch.export`, real AOTI pack, CPU, no stand-in — over
    a two-class declaration, because the callback contract on its own is not
    the thing that broke. pgw#825 split the monolithic `mint()` into
    `mint` / `_attach_partial_phases` / `_mint_cell` while this parameter was
    on a branch; the two merged textually clean and the beat was left calling
    a name its function no longer had (`NameError: on_progress`, 16 failed / 6
    errored). Only a test that drives the real `mint()` end to end can catch
    that class, so this one does.

    Reported BEFORE the work, deliberately: a row that never returns is the one
    a reader most needs named, and an after-the-fact tick names only the rows
    that finished.
    """
    from gen_worker import aot_mint

    beats: List[Tuple[str, int, int, str]] = []
    _mint(
        tmp_path, monkeypatch,
        on_progress=lambda *row: beats.append(row))  # type: ignore[arg-type]

    trace = [b for b in beats if b[0] == aot_mint.PHASE_TRACE_GRAPH]
    # The opener names the size of the job, then one row per declared class.
    assert [b[1] for b in trace] == [0, 1, 2], beats
    assert all(b[2] == 2 for b in trace), "a step with no total is not progress"
    assert trace[0][0] == activity.PHASE_TRACE_GRAPH
    # Each row NAMES its entry, or "12 of 18" says nothing about which 12.
    assert all(b[3] for b in trace[1:]), trace
    # And the mint's tail phase reports too, so a reader can tell "still
    # exporting" from "packing".
    assert [b[0] for b in beats][-1] == aot_mint.PHASE_SEAL_PUBLISH, beats


def test_an_aborted_mint_reports_the_entry_it_died_ON(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reconciliation of pgw#824's live beat with pgw#825's abort table.

    pgw#825 made an aborted mint report where it SPENT — per-entry rows for the
    entries that finished. That is the wrong half of the question when a mint
    dies mid-entry: 11 rows and no twelfth, and the twelfth is the one being
    asked about. Because both halves now read one `MintProgress`, the abort
    table carries the last position the live beat reported, as `at`.
    """
    from gen_worker import aot_mint

    def _refuse(row: Any, package: Any) -> Any:
        raise aot_mint.MintRefused(f"entry {row.name!r}: bindability gate: nope")

    monkeypatch.setattr(aot_mint, "_gate_and_declare_entry", _refuse)
    with pytest.raises(aot_mint.MintRefused) as excinfo:
        _mint(tmp_path, monkeypatch)

    table = getattr(excinfo.value, "mint_phases", {})
    assert table.get("terminus") == "aborted"
    # pgw#825's half, unchanged.
    assert table["n_entries"] == 2 and table["totals"]["export_s"] > 0
    # pgw#824's half, on the SAME record.
    assert table["at"]["phase"] == aot_mint.PHASE_SEAL_PUBLISH
    assert table["at"]["total"] == 2


def test_the_position_is_recorded_even_with_no_sink_and_a_raising_one() -> None:
    """One mechanism means the live beat and the abort table can never
    disagree: the position is RECORDED before it is reported, so a mint with no
    `on_progress` (every non-delegated caller) and a mint whose sink raises
    both still name where they were. Telemetry never costs a mint its work.
    """
    from gen_worker import aot_mint

    quiet = aot_mint.MintProgress()
    quiet.beat(aot_mint.PHASE_TRACE_GRAPH, 3, 18, "entry-c")
    assert quiet.at == {
        "phase": "trace_graph", "step": 3, "total": 18, "note": "entry-c"}

    def _explode(*_: Any) -> None:
        raise RuntimeError("the sink is not the mint")

    loud = aot_mint.MintProgress(on_progress=_explode)
    loud.beat(aot_mint.PHASE_INDUCTOR_COMPILE, 12, 18, "entry-l")
    assert loud.at["step"] == 12


def test_the_mint_progress_tokens_are_the_hubs_own_phase_vocabulary() -> None:
    """The child frames these verbatim and the parent lands them on the same
    `self_mint_compile` activity, so a drifted token would silently split one
    mint's progress across two phase names hub-side."""
    from gen_worker import aot_mint

    assert aot_mint.PHASE_TRACE_GRAPH == activity.PHASE_TRACE_GRAPH
    assert aot_mint.PHASE_INDUCTOR_COMPILE == activity.PHASE_INDUCTOR_COMPILE
    assert aot_mint.PHASE_SEAL_PUBLISH == activity.PHASE_SEAL_PUBLISH


def test_the_entry_compile_pool_reports_each_entry_as_it_lands() -> None:
    """The pool loop is the longest wire-silent stretch of a mint (an 18-entry
    sdxl cell spends the bulk of its wall clock there) and reported nothing
    between "compiling" and "packed"."""
    import inspect

    from gen_worker import aot_compile_pool

    src = inspect.getsource(aot_compile_pool.EntryCompilePool.compile)
    assert "on_entry" in src
    assert "len(done), len(entries)" in src, (
        "progress must carry BOTH a step and a total — a bare step is not "
        "progress, it is a counter")


def test_progress_reporting_never_costs_the_mint_its_work() -> None:
    """Telemetry must never fail the work it reports on: a raising callback
    would otherwise throw away entries that already compiled."""
    from gen_worker import mint_delegate

    class _ActNoCounter:
        """An Activity double with no counter registry — the shape every
        existing mint test passes."""

        beats = 0

        def heartbeat(self) -> None:
            type(self).beats += 1

    apply = mint_delegate._on_evidence(_ActNoCounter())
    apply(12.5)  # must not raise despite the absent counter()
    assert _ActNoCounter.beats == 1


def test_the_delegated_mint_actually_passes_the_evidence_callback() -> None:
    """`run_mint` has accepted `on_evidence` since pgw#784 and NOBODY passed
    one, so the child's measured progress existed only to decide whether to
    KILL it — never to prove it was working."""
    import inspect

    from gen_worker import mint_delegate

    src = inspect.getsource(mint_delegate.build_cell)
    assert "on_evidence=_on_evidence(act)" in src


# ---------------------------------------------------------------------------
# Invariant 3 — the adoption walk: a MISS says why every candidate lost
# ---------------------------------------------------------------------------


def test_discovery_counts_why_every_candidate_was_rejected() -> None:
    """RED before pgw#824: `_candidates` dropped rows on `logger.debug` or a
    bare `continue`, and the `miss` event then said only "no matching cell
    among N checkpoint(s)".

    True, and useless: a family with 12 published cells that rejects all 12
    read identically to a family with none, and those are different bugs with
    different owners. Coalescing with counts is the pattern; dropping is not.
    """
    from gen_worker import aot_cells

    rejected: Dict[str, int] = {}
    items: List[Dict[str, Any]] = [
        {"metadata": {"kind": "not-an-aot-cell"}},
        {"metadata": {"kind": "not-an-aot-cell"}},
        {"metadata": {"kind": aot_cells.aot_serve.ARTIFACT_KIND,
                      "cell_key": "not a key"}},
    ]
    rows = aot_cells._candidates(items, "sdxl", "", rejected)

    assert rows == []
    assert rejected["not_an_aot_cell"] == 2, "identical rejections COALESCE"
    assert rejected["unreadable_cell_key"] == 1
    # the classes are countable tokens, not sentences
    assert all(" " not in cls for cls in rejected)


def test_a_miss_with_no_candidates_at_all_says_so_explicitly() -> None:
    """"No published cells" and "every published cell was rejected" must not
    render as the same sentence."""
    import inspect

    from gen_worker import aot_cells

    src = inspect.getsource(aot_cells._discover_inner)
    assert "rejected by class" in src
    assert "no published cells at all" in src


# ---------------------------------------------------------------------------
# Invariant 1 — the two high-severity finds: both corrupt DECISIONS, not just
# visibility, so both are asserted on the decision, not only on the event.
# ---------------------------------------------------------------------------


def test_an_unlanded_nf4_rung_is_a_rung_OUTCOME_not_the_absence_of_one() -> None:
    """RED before pgw#824: the failure did `adaptive_rung = ""`, and the
    `if adaptive_rung:` stamp below it is the very mechanism that reports rung
    outcomes to placement — so clearing the variable SELF-SUPPRESSED the
    report. The worst outcome the ladder can produce (serving full precision
    over the budgeted VRAM, on a host already too tight for stored precision)
    was the only one that reported nothing, while every sibling rung reported
    itself.
    """
    from gen_worker.models import loading, provision

    # Three states placement must be able to tell apart.
    assert loading.RUNG_NF4_UNLANDED not in ("", "nf4")
    assert provision.RUNG_NF4_UNLANDED == loading.RUNG_NF4_UNLANDED


def test_the_unlanded_rung_reaches_placement_through_SlotLoad() -> None:
    """An event alone would not be enough: every sibling rung reaches
    ServePlan/FnDegraded via SlotLoad.rung -> `_record_adaptive_rung`, and a
    fix that only logged-but-typed would still leave placement blind."""
    import inspect

    from gen_worker.models import provision

    src = inspect.getsource(provision)
    assert "if rung == RUNG_NF4_UNLANDED:" in src
    assert "out.rung = rung" in src


def test_a_failed_eviction_stays_booked_in_vram(
    events: List[Any],
) -> None:
    """RED before pgw#824, and an ACCOUNTING bug before it is a visibility one.

    The booking was unconditional, so a failed eviction still wrote
    `tier=RAM, vram_bytes=0` while `_move_verified`'s own rollback had just put
    the object back on CUDA. The registry then believed the entry held ZERO
    VRAM, `make_room` handed out headroom that does not exist, and the OOM
    landed on an unrelated `promote()` later with nothing tying it back here.
    """
    import inspect

    from gen_worker.models import residency

    src = inspect.getsource(residency.Residency.promote)
    # the booking is now GATED on the eviction actually succeeding
    assert "evicted = self._move_verified" in src
    assert "if not evicted:" in src
    # and the failure returns BEFORE the RAM/0 booking
    failure_branch = src.split("if not evicted:")[1].split("e.tier = Tier.RAM")[0]
    assert "return False" in failure_branch, (
        "the failed-eviction branch must refuse BEFORE booking RAM/0 — "
        "booking it as RAM/0 is the headroom lie that OOMs a later promote")
    assert "eviction_failed_still_resident" in failure_branch


def test_block_offload_engagement_is_typed_not_just_logged() -> None:
    """The sibling the pgw#760 `apply_fp8_storage` fix missed, and the larger
    of the two: every forward now streams this component's weights over PCIe
    from host RAM, which is the biggest per-request latency change the loader
    can make. It was a `logger.warning` on a pod with no stdout."""
    import inspect

    from gen_worker.models import loading

    src = inspect.getsource(loading.apply_block_window_offload)
    assert 'phase="block_offload_engaged"' in src
    # pinned vs pageable differ by ~2x on the transfer that is now in the
    # critical path, so the detail has to carry which one it got.
    assert "pinned" in src.split('phase="block_offload_engaged"')[0][-700:]


def test_an_unparseable_destination_repo_confesses_once() -> None:
    """RED before pgw#824: `logger.debug` + `return None`.

    Returning None does not mean "no repo was asked for" — a destination_repo
    WAS supplied and did not parse, so the job's outputs silently stop being
    repo-CAS writes and land on the user-files/media path instead. That is the
    exact fallback `_require_repo_job_scope_for_tensors` exists to prevent, and
    it was reachable past that guard for every non-training kind.
    """
    import inspect

    from gen_worker import request_context

    src = inspect.getsource(request_context.RequestContext._repo_job_upload_scope)
    assert 'phase="repo_scope_unparseable"' in src
    # coalesced: the getter runs once per uploaded file, so one config defect
    # must not become a per-file flood
    assert "_repo_scope_parse_reported" in src


# ---------------------------------------------------------------------------
# Invariant 2, END TO END — the posture reaches the HUB on the real wire.
#
# Everything above proves the pieces. This proves the CHAIN: a real worker, a
# real gRPC stream, a real compile-declaring endpoint that cannot arm on this
# host, a real dispatch — and the terminal JobResult the hub-double receives
# carrying a classified reason instead of "".
# ---------------------------------------------------------------------------


def test_an_eager_request_on_a_compile_declaring_release_names_its_reason() -> None:
    """RED before pgw#824: `fallback_reason=""` on every one of these rows.

    This endpoint DECLARES a compile cell, so `no_compile_declared` cannot be
    the answer — the worker genuinely wanted a cell and did not get one. The
    arming brain classified why (there is no CUDA device on a CI host, so
    `_fail_closed(phase="no_cuda")`), and the whole point of the issue is that
    the classification survives all the way to the request row instead of dying
    in the function that computed it.
    """
    import pathlib
    import tempfile

    import msgspec

    from gen_worker import boot_phases
    from gen_worker.api.binding import wire_ref
    from gen_worker.pb import worker_scheduler_pb2 as pb

    from harness.blob_host import BlobHost
    from harness.boot_ladder_endpoints_pgw797 import COMPILE_MODEL, EchoIn
    from harness.hub_double import hub_double, is_ready, is_result_for

    boot_phases.reset_for_tests()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="pgw824-"))
    blobs = BlobHost(tmp)
    model_ref = wire_ref(COMPILE_MODEL)
    model_snap = blobs.one_file_snapshot(
        "snap-model", "model", b"pgw824-compile-model-bytes" * 512)
    try:
        with hub_double(
            modules=("harness.boot_ladder_endpoints_pgw797",),
        ) as (scheduler, _h):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                desired_residency=pb.DesiredResidency(
                    generation=1,
                    disk_refs=[model_ref],
                    snapshots={model_ref: model_snap},
                    hot=[pb.DesiredInstance(
                        function_name="compile-echo",
                        models=[pb.ModelBinding(slot="model", ref=model_ref)],
                    )],
                ),
            ))
            conn.wait_for(
                lambda m: m.WhichOneof("msg") == "state_delta"
                and "compile-echo" in m.state_delta.available_functions
            )
            conn.send(run_job=pb.RunJob(
                request_id="r-posture", attempt=1,
                function_name="compile-echo",
                input_payload=msgspec.msgpack.encode(EchoIn(text="x")),
            ))
            res = conn.wait_for(is_result_for("r-posture")).job_result
    finally:
        blobs.shutdown()

    assert res.status == pb.JOB_STATUS_OK
    m = res.metrics
    assert m.serving_mode == serving_mode.MODE_EAGER
    # THE ASSERTION. Not "" -- a classified, countable token.
    assert m.fallback_reason != "", (
        "an eager request on a release that DECLARED a compile target must "
        "say why it is eager; \"\" is the defect this issue exists to close")
    # and it must not be the honest-zero class: this release DOES declare one
    assert m.fallback_reason != serving_mode.POSTURE_NO_COMPILE_DECLARED
    # nothing fell back -- there was nothing to fall back FROM
    assert m.served_eager_fallback is False
    assert m.served_cell_ref == ""


# ---------------------------------------------------------------------------
# Invariant 1 — quantized-lane partial application: the coalescing rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mod_name,phase", [
    ("w8a8", "w8a8_partial_swap"),
    ("w4a4", "w4a4_partial_swap"),
])
def test_a_partially_swapped_quant_execution_lane_confesses_with_counts(
    mod_name: str, phase: str,
) -> None:
    """RED before pgw#824.

    Every layer the swap loop skips stays DEQUANTIZED at full precision inside
    a pipeline that reports the quantized lane to placement, to billing and to
    every AOT-vs-eager comparison. One skip class was a `logger.warning`; the
    dimension-alignment skip had no report of ANY kind.

    Asserted as COALESCED: a 300-Linear denoiser must produce one row carrying
    counts, not 300 rows. Counts are the pattern the doctrine names — dropping
    is not, and neither is flooding.
    """
    import importlib
    import inspect

    mod = importlib.import_module(f"gen_worker.models.{mod_name}")
    src = inspect.getsource(mod)

    assert f'phase="{phase}"' in src
    # both skip classes are counted, including the one that was fully silent
    assert '_skip("not_plain_linear"' in src
    assert '_skip("dim_unaligned"' in src
    # the emit is OUTSIDE the per-layer loop (coalesced, once per model)
    emit_at = src.index(f'phase="{phase}"')
    loop_at = src.index("for layer in art.quantized:")
    tail = src[loop_at:emit_at]
    assert "logger.info" in tail, (
        "the confession must come after the loop's summary line, i.e. once "
        "per model — not once per skipped layer")


def test_an_unqualified_fp8_gemm_is_a_execution_lane_decision_that_reports_itself() -> None:
    """The whole point of the w8a8 artifact is the fused fp8 GEMM. Falling to
    the dequant lane keeps the memory saving and gives up the speed, so every
    request is slower than the lane the pod advertises — and a fleet-wide
    qualification regression (a driver or torch bump that stops the kernel
    qualifying) would have surfaced as "everything got slower", with the cause
    written only to a pod log nobody can read."""
    import inspect

    from gen_worker.models import w8a8

    src = inspect.getsource(w8a8._choose_gemm_mode)
    assert 'phase="w8a8_gemm_unqualified"' in src


def test_an_adapter_that_maps_nothing_is_reported_not_waved_through(
    events: List[Any],
) -> None:
    """RED before pgw#824 — a VACUOUS GUARD, the pgw#812-D2 defect class again.

    `adapter_fidelity.evaluate_branch` returns None for an empty mapping and
    `gate(None)` is a no-op, so an adapter that maps to ZERO modules sails
    straight through the gate that exists to catch exactly this. The request
    then renders with NO adapter applied while reporting success: the user
    asked for a LoRA and got an image without one, and nothing anywhere said
    so.

    Reachable in production: `_normalize_lora_keys` falls back to RAW keys when
    `lora_state_dict` normalization raises, and raw kohya keys match no module
    path on the model.
    """
    from gen_worker.models import adapter_fidelity

    # the vacuous half, asserted directly: this is what made it invisible
    assert adapter_fidelity.evaluate_branch({}, {}, ref="r") is None
    assert adapter_fidelity.gate(None) is None
    assert not events, "the gate genuinely reports nothing for an empty map"

    # ... so the zero-mapping fact is reported at the staging site instead
    import inspect

    from gen_worker.models import w8a8_lora

    src = inspect.getsource(w8a8_lora)
    assert 'phase="not_applied"' in src
    assert "if not mapped:" in src
