"""pgw#1150 — ie#664's tier-2 checklist as ONE COMMAND.

ie#664 shipped the author-CI standard with its tier-2 leg as a RUNBOOK: a
checklist a human executes by hand on a rented pod, which is why all 13
`author-ci.toml` records read `blocker = "ie#664"`. Every primitive it
assembles already shipped — `rigcheck.assert_fleet_line`, the mint-parent
parity gate (pgw#1141), `numerics_probe`/`numerics_ladder`, pgw#1142's
serve-posture eager arm, `stage_ms.*` — and none of them were wired together.

WHAT IS REAL HERE: the endpoint load and function selection (`cli.run`'s own),
the dispatch, `ctx.stage` -> `stage_ms` timing, the serve-posture order and its
release, `Compile.blockers` read through `export_contract.open_blockers`, the
parity gate (`provision.gate_cell_numerics` against pgw#868's real armed cell,
real ladder, real declared floor), the record's TOML, and — where the sibling
repo is checked out — `inference-endpoints/scripts/lint_author_ci.py` itself,
imported and run against what this harness emitted.

WHAT IS FAKED, and why: the COMPILE and the model load. Paul's standing rule
(2026-08-10) is that no mint, compile or AOTI link runs on the shared dev box.
So `run_setup` is stubbed to hand back pgw#868's armed probe pipeline — the
same seam `test_two_run_reuse_pgw1096.py` fakes — and `assert_fleet_line` is
stubbed in the rows that are not about the preflight (this box HAS a driver and
no usable CUDA, so the real assertion refuses it, which two rows below assert
rather than work around). "A real AOTI cell arms on a real card and is faster"
is a claim only a pod can make; that is the first author's run, and it is owed.
"""

from __future__ import annotations

import importlib.util
import os
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from gen_worker import author_ci, rigcheck, serve_posture
from gen_worker.api.decorators import Compile

import test_numerics_gate_pgw868 as rig  # noqa: E402
from harness import author_ci_endpoints_pgw1150 as endpoints  # noqa: E402

MODULE = "harness.author_ci_endpoints_pgw1150"
REPO = Path(__file__).resolve().parents[1]

declared = rig.declared

#: A well-formed DECLARATION half — what tier 1 already refuses a compiling
#: family for not having. The harness judges itself against it and owns only
#: `[proof]`.
RECORD = '''\
# ie#664 — the AUTHOR's compile-vs-eager test story (DESIGN-RULINGS §4.32).
[parity]
floor = "declared"

[speed]
# Read the compute stage, never total_round_trip_ms (th#1795).
metric = "stage_ms.denoise"
min_speedup = 1.10

[proof]
status = "never-run"
blocker = "ie#664"
'''


@pytest.fixture(autouse=True)
def _clean_posture() -> Any:
    """No row may leak an eager-only order into the next one."""
    serve_posture.reset()
    endpoints.POSTURE_SEEN.clear()
    yield
    serve_posture.reset()


@pytest.fixture
def on_the_line(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    """The preflight, answered. Not the subject of these rows — two others own
    it — and this box genuinely cannot pass it."""
    env = {"device": "NVIDIA L4", "sm": "8.9", "torch": "2.13.0+cu130"}
    monkeypatch.setattr(
        author_ci.rigcheck, "assert_fleet_line", lambda *a, **k: dict(env))
    return env


def record_file(tmp_path: Path, body: str = RECORD) -> Path:
    path = tmp_path / "author-ci.toml"
    path.write_text(body, encoding="utf-8")
    return path


def invoke(record: Path, cls: str, *extra: str) -> Tuple[int, Any]:
    argv = ["--module", MODULE, "--class", cls, "--payload", '{"prompt": "x"}',
            "--record", str(record), "--commit", "deadbee", "--write",
            *extra]
    args = author_ci.build_parser().parse_args(argv)
    return author_ci.run(args)


def armed_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decl: Any,
               cosine: float, *, verify_numerics: bool) -> Any:
    """pgw#868's REAL arm, returned as the slot a stubbed `setup()` loads."""
    packages = {rig.entry_name(h, w): rig.ProbePackage(cosine=cosine)
                for h, w in rig.ROWS}
    pipeline, _module, outcome = rig.arm(
        tmp_path, monkeypatch, decl, packages,
        verify_numerics=verify_numerics)
    assert outcome.armed, f"the rig did not arm: {outcome.reason}"
    monkeypatch.setattr(
        author_ci._Subject, "_on_resolved",
        lambda self, resolved: self._loaded.update({"pipeline": pipeline}))
    return pipeline


# ---------------------------------------------------------------------------
# 1. the preflight is rigcheck's, exit vocabulary included
# ---------------------------------------------------------------------------

def test_a_rig_off_the_fleet_line_exits_90_and_a_dead_host_exits_91(
        tmp_path, monkeypatch):
    """RED on master: there is no command to exit at all.

    The numbers are rigcheck's own (`python -m gen_worker.rigcheck` exits 0/90/
    91). An author who wraps this must not have to learn a second table for the
    same preflight, and a wrapper that mapped both onto 1 would make "your card
    cannot run this wheel" indistinguishable from "your cell is slow".
    """
    record = record_file(tmp_path)
    base = ["--module", MODULE, "--class", "Fast", "--record", str(record)]

    def refuse(exc: Exception) -> Any:
        def _raise(*_a: Any, **_k: Any) -> Any:
            raise exc
        return _raise

    monkeypatch.setattr(author_ci.rigcheck, "assert_fleet_line",
                        refuse(rigcheck.FleetLineMismatch("off the line")))
    assert author_ci.main(base) == author_ci.EXIT_RIG_OFF_LINE == 90

    monkeypatch.setattr(author_ci.rigcheck, "assert_fleet_line",
                        refuse(rigcheck.FleetLineUnknown("no authority")))
    assert author_ci.main(base) == 90, "no authority is not permission to measure"

    monkeypatch.setattr(author_ci.rigcheck, "assert_fleet_line",
                        refuse(rigcheck.CudaUnusable("driver too old")))
    assert author_ci.main(base) == author_ci.EXIT_HOST_UNUSABLE == 91


def test_the_preflight_runs_before_the_endpoint_is_even_imported(
        tmp_path, monkeypatch):
    """A rig off the line has produced no evidence, so nothing else may run
    first — not the load, not the record read, not the git call."""
    order: List[str] = []
    monkeypatch.setattr(
        author_ci.rigcheck, "assert_fleet_line",
        lambda *a, **k: (order.append("preflight"), {"sm": "8.9"})[1])
    monkeypatch.setattr(
        author_ci._Subject, "load",
        lambda self: order.append("load") or (_ for _ in ()).throw(
            author_ci.HarnessError("stop here")))
    author_ci.main(["--module", MODULE, "--class", "Fast",
                    "--record", str(record_file(tmp_path))])
    assert order == ["preflight", "load"]


# ---------------------------------------------------------------------------
# 2. a declared block is a LEGAL state (ie#664 §6)
# ---------------------------------------------------------------------------

def test_a_blocked_family_records_never_run_and_continues_EAGER_ONLY(
        tmp_path, on_the_line):
    """`mint: blocked-by-declaration`, and the run does not stop.

    A family with open `Compile.blockers` said WHY it may not mint; repeating
    the sentence on a rented pod buys nothing. What it must not do is fail as
    if something broke, or leave the record unable to name the reason.
    """
    record = record_file(tmp_path)
    code, report = invoke(record, "Blocked")

    assert report.arm == author_ci.ARM_BLOCKED
    assert report.blocker == ", ".join(endpoints.BLOCKER_IDS)
    assert report.status == author_ci.STATUS_NEVER_RUN
    assert code == author_ci.EXIT_FAILED, "no proof was produced, and it says so"

    # Eager-only was STRUCTURAL, not hopeful: every request ran under the
    # order, so nothing could have adopted or minted behind the run's back.
    assert endpoints.POSTURE_SEEN and all(endpoints.POSTURE_SEEN)
    assert serve_posture.eager_only() is False, "the order outlived the run"

    proof = tomllib.loads(record.read_text(encoding="utf-8"))["proof"]
    assert proof["status"] == "never-run"
    assert proof["blocker"] == ", ".join(endpoints.BLOCKER_IDS)
    assert not (set(proof) & {"cosine", "eager_median_ms", "n"}), \
        "a blocked family recorded evidence it does not have"


# ---------------------------------------------------------------------------
# 3. the eager arm is pgw#1142's order, engaged in-process and RELEASED
# ---------------------------------------------------------------------------

def test_the_compiled_arm_runs_first_then_the_order_engages_and_releases(
        tmp_path, monkeypatch, declared, on_the_line):
    """The whole reason the two arms are comparable: ONE process, one set of
    weights, one pipeline — the arms differ only by the order.

    Engaged through `serve_posture.apply_command`, never the hub route: the
    author's pod may have no hub at all.
    """
    armed_cell(tmp_path, monkeypatch, declared, 1.0, verify_numerics=False)
    record = record_file(tmp_path)
    code, report = invoke(record, "Fast")

    n = 5
    assert len(endpoints.POSTURE_SEEN) == 2 * (n + 1), \
        "each arm runs n+1 requests — the first is the discarded one"
    assert endpoints.POSTURE_SEEN[:n + 1] == [False] * (n + 1), \
        "the compiled arm ran under an eager-only order"
    assert endpoints.POSTURE_SEEN[n + 1:] == [True] * (n + 1), \
        "the eager arm ran compiled"
    assert serve_posture.eager_only() is False
    assert report.compiled is not None and report.eager is not None
    assert len(report.compiled.samples) == n == len(report.eager.samples)
    assert report.compiled.discarded > 0, "the discarded request is recorded"
    assert code == author_ci.EXIT_OK and report.status == author_ci.STATUS_PROVEN


def test_the_order_is_released_even_when_a_leg_raises(
        tmp_path, monkeypatch, declared, on_the_line):
    """An abandoned eager-only order would silently un-compile the pod for
    every later run in the same process."""
    armed_cell(tmp_path, monkeypatch, declared, 1.0, verify_numerics=False)
    boom = RuntimeError("the handler died mid-arm")
    calls: List[int] = []

    def explode(subject: Any, name: str, stage: str, n: int) -> Any:
        calls.append(1)
        if len(calls) > 1:
            raise boom
        return author_ci.Leg(name=name, samples=(1.0,) * n, discarded=1.0)

    monkeypatch.setattr(author_ci, "measure_leg", explode)
    with pytest.raises(RuntimeError):
        invoke(record_file(tmp_path), "Fast")
    assert serve_posture.eager_only() is False


# ---------------------------------------------------------------------------
# 4. parity is the MINT-PARENT gate's verdict, at the family's DECLARED floor
# ---------------------------------------------------------------------------

def test_parity_uses_the_familys_own_declared_floor_and_refuses_below_it(
        tmp_path, monkeypatch, declared, on_the_line):
    """cos=0.99 is above the SDK default (0.98) and below this family's
    declared 0.995 — so the floor that decides IS the declaration's, or this
    row cannot tell the two apart. That is the whole of ie#664's *"the bar is
    the family's own numerics_floor, never a number typed into a harness"*.

    The cell is armed WITHOUT the mint gate (an adopt), so the harness takes
    the verdict through the gate itself — the same function, never a
    comparison re-implemented here.
    """
    armed_cell(tmp_path, monkeypatch, declared, 0.99, verify_numerics=False)
    record = record_file(tmp_path)
    code, report = invoke(record, "Fast")

    assert report.parity is not None
    assert report.parity.passed is False
    assert report.parity.cosine is not None and report.parity.cosine < 0.995
    assert report.parity.floor_source == "declared", \
        "the gate scored against the SDK default while the family declared one"
    assert report.status == author_ci.STATUS_FAILED
    assert code == author_ci.EXIT_FAILED
    assert "PARITY FAILED" in record.read_text(encoding="utf-8")


def test_a_healthy_cell_records_the_gates_own_cosine(
        tmp_path, monkeypatch, declared, on_the_line):
    armed_cell(tmp_path, monkeypatch, declared, 1.0, verify_numerics=False)
    record = record_file(tmp_path)
    _code, report = invoke(record, "Fast")

    assert report.parity is not None and report.parity.passed
    proof = tomllib.loads(record.read_text(encoding="utf-8"))["proof"]
    assert proof["cosine"] >= 0.999
    assert proof["cell"] == "cell868", "the record names WHICH cell was measured"


def test_the_declared_numerics_floor_reaches_the_gate_at_all(declared):
    """RED on master, and it is why this issue has a prerequisite.

    `numerics_ladder.declared_thresholds` has exactly ONE caller
    (`numerics_probe.probe_cell`) and its `cfg` is always a `registry
    .CompileCell` — which carried no `numerics_floor` field at all. So every
    gate on every path scored against the SDK default, `threshold_source` said
    `sdk-default` everywhere, and `Compile.numerics_floor` (pgw#812/#814,
    sdxl's measured 0.995/0.999) was a declaration nothing read.
    """
    from gen_worker import numerics_ladder
    from gen_worker.registry import extract_specs

    spec = extract_specs(endpoints.Fast)[0]
    cell = spec.compile_cell()
    assert cell is not None

    assert cell.numerics_floor == endpoints.FLOOR
    assert cell.numerics_warn == endpoints.WARN
    bar = numerics_ladder.declared_thresholds(cell)
    assert bar.floor == endpoints.FLOOR
    assert bar != numerics_ladder.DEFAULT_THRESHOLDS

    # A numerics band is not a graph axis: declaring one must never move a
    # cell key, or every family that states its floor re-mints the fleet.
    assert "numerics_floor" not in cell.contract_facts()
    assert "numerics_warn" not in cell.contract_facts()


# ---------------------------------------------------------------------------
# 5. the record — closed vocabulary, and a failure that reads as one
# ---------------------------------------------------------------------------

def test_a_below_bar_speedup_is_a_FAILURE_and_never_a_proof(
        tmp_path, monkeypatch, declared, on_the_line):
    """The `Regressed` family's compiled arm is slower than eager.

    The record keeps the evidence — a number that only reached a pod's stdout
    is a number we do not have — and says `failed`, which `lint_author_ci.py`
    refuses. What it must never do is round up to `proven`, and it must never
    be relabelled `never-run`: that is a measured gap becoming an unmeasured
    one.
    """
    armed_cell(tmp_path, monkeypatch, declared, 1.0, verify_numerics=False)
    record = record_file(tmp_path)
    code, report = invoke(record, "Regressed")

    assert report.speedup is not None and report.speedup < 1.10
    assert report.status == author_ci.STATUS_FAILED
    assert code == author_ci.EXIT_FAILED

    proof = tomllib.loads(record.read_text(encoding="utf-8"))["proof"]
    assert proof["status"] == "failed"
    assert proof["n"] == 5 and proof["compiled_median_ms"] > 0
    assert proof["eager_median_ms"] > 0 and proof["compiled_p95_ms"] > 0
    assert "SPEED BELOW THE DECLARED BAR" in proof["note"]


def test_a_round_trip_metric_is_refused_by_name(tmp_path, on_the_line):
    """th#1795: a sibling lane's '10.9x' corrected to 1.3x when it was read
    off the stage. This harness measures in-process, where a round trip does
    not exist at all."""
    for metric in ("total_round_trip_ms", "slot_held_ms", "denoise"):
        record = record_file(tmp_path, RECORD.replace(
            '"stage_ms.denoise"', f'"{metric}"'))
        with pytest.raises(author_ci.HarnessError, match="th#1795"):
            invoke(record, "Fast")


def test_the_bar_comes_from_the_declaration_first_then_the_record(tmp_path):
    """pgw#1149 moves the bar onto `Compile.speed_metric`/`min_speedup`. Those
    fields do not exist yet, so the resolver reads them by name — the day that
    lane lands, this follows it with no edit here and there is still exactly
    one reader per source."""
    record = tomllib.loads(RECORD)
    plain = Compile(family="f", targets=("t",), shapes=((8, 8),), text_len=0)
    bar = author_ci.resolve_bar(plain, record)
    assert (bar.metric, bar.min_speedup, bar.source) == (
        "stage_ms.denoise", 1.10, "author-ci.toml")

    class _Future:
        family = "f"
        speed_metric = "stage_ms.transformer"
        min_speedup = 1.5

    ahead = author_ci.resolve_bar(_Future(), record)
    assert (ahead.metric, ahead.min_speedup, ahead.source) == (
        "stage_ms.transformer", 1.5, "declaration")

    # No bar declared anywhere -> ie#664's fleet default, never a guessed
    # stage: 1.0 is met by denoise noise, and a stage nobody named is not one.
    del record["speed"]["min_speedup"]
    assert author_ci.resolve_bar(plain, record).source == "fleet-default"
    assert author_ci.resolve_bar(plain, record).min_speedup == 1.10
    record["speed"]["metric"] = ""
    with pytest.raises(author_ci.HarnessError, match="may not guess"):
        author_ci.resolve_bar(plain, record)


def test_a_metric_the_handler_does_not_bracket_names_what_it_did(
        tmp_path, on_the_line):
    record = record_file(tmp_path, RECORD.replace("denoise", "vae_decode"))
    with pytest.raises(author_ci.HarnessError, match="does not bracket"):
        invoke(record, "Fast")


def test_the_write_keeps_the_authors_declarations_verbatim(tmp_path):
    """`[parity]` and `[speed]` are the author's DECLARATIONS and the bar the
    run was judged against. Re-serializing them from a parse would silently
    drop their stated reasons, which is most of what tier 1 checks."""
    spliced = author_ci.splice_proof(RECORD, '[proof]\nstatus = "proven"\n')
    assert spliced.startswith(RECORD.split("[proof]")[0])
    assert "# Read the compute stage, never total_round_trip_ms" in spliced
    assert spliced.count("[proof]") == 1
    assert 'blocker = "ie#664"' not in spliced

    with pytest.raises(author_ci.HarnessError, match="after \\[proof\\]"):
        author_ci.splice_proof(RECORD + "\n[extra]\nx = 1\n", "[proof]\n")

    # A record with no [proof] table yet gains one rather than being refused.
    head = RECORD.split("[proof]")[0]
    assert author_ci.splice_proof(head, "[proof]\n").endswith("[proof]\n")


def test_median_and_p95_are_both_reported_and_the_tail_is_not_hidden():
    """A median alone hides the tail a guard miss produces (`lint_author_ci`
    requires both). At n=5 the nearest-rank p95 IS the max, deliberately."""
    leg = author_ci.Leg(name="compiled", samples=(10.0, 11.0, 12.0, 13.0, 90.0),
                        discarded=400.0)
    assert leg.median == 12.0
    assert leg.p95 == 90.0
    assert "discarded first: 400.0ms" in leg.line()


# ---------------------------------------------------------------------------
# 6. the round trip through the REAL lint — the schema is coordinated, not
#    forked
# ---------------------------------------------------------------------------

def _lint() -> Any:
    """`inference-endpoints/scripts/lint_author_ci.py`, imported for real."""
    candidates = [os.environ.get("AUTHOR_CI_LINT", "")]
    candidates += [str(base / "inference-endpoints" / "scripts"
                       / "lint_author_ci.py")
                   for base in (REPO.parent, REPO.parents[2], Path.home() / "cozy")
                   if base is not None]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            spec = importlib.util.spec_from_file_location(
                "lint_author_ci", candidate)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    pytest.skip("inference-endpoints is not checked out beside this repo")


def _through_the_lint(lint: Any, tmp_path: Path, record: Path) -> int:
    """The emitted record, in the tree shape the real gate reads."""
    root = lint._tree(tmp_path / "lint", record=record.read_text("utf-8"),
                      extra=", numerics_floor=0.995")
    return int(lint.check(root, quiet=True))


def test_a_proven_record_passes_the_real_lint(
        tmp_path, monkeypatch, declared, on_the_line):
    lint = _lint()
    armed_cell(tmp_path, monkeypatch, declared, 1.0, verify_numerics=False)
    record = record_file(tmp_path)
    _code, report = invoke(record, "Fast")
    assert report.status == author_ci.STATUS_PROVEN
    assert _through_the_lint(lint, tmp_path, record) == 0, \
        "the harness emitted a record its own repo's gate refuses"


def test_a_never_run_record_passes_the_real_lint(tmp_path, on_the_line):
    """A blocked family's record is a LEGAL one — the gap is countable, and
    CI is green because nothing is claimed."""
    lint = _lint()
    record = record_file(tmp_path)
    invoke(record, "Blocked")
    assert _through_the_lint(lint, tmp_path, record) == 0


def test_a_failed_record_is_REFUSED_by_the_real_lint(
        tmp_path, monkeypatch, declared, on_the_line):
    """The point of recording a failure: it stays red until the code (or the
    declaration) is fixed, and it cannot be skimmed as a proof."""
    lint = _lint()
    armed_cell(tmp_path, monkeypatch, declared, 1.0, verify_numerics=False)
    record = record_file(tmp_path)
    _code, report = invoke(record, "Regressed")
    assert report.status == author_ci.STATUS_FAILED
    assert _through_the_lint(lint, tmp_path, record) == 1


def test_the_harness_and_the_lint_agree_on_the_status_vocabulary():
    """Two spellings of a closed vocabulary is how one of them quietly grows a
    third word."""
    lint = _lint()
    assert set(lint.STATUSES) == {
        author_ci.STATUS_PROVEN, author_ci.STATUS_FAILED,
        author_ci.STATUS_NEVER_RUN}
    assert lint.STATUS_PROVEN == author_ci.STATUS_PROVEN
    assert lint.STATUS_NEVER_RUN == author_ci.STATUS_NEVER_RUN
    assert lint.STATUS_FAILED == author_ci.STATUS_FAILED
