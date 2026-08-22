"""pgw#1671 — the drive's step budget is the PLATFORM's, not the author's.

`ctx.step_callback()` hands an author a callable that raises `StepBudgetReached`
on the budget-th call, and until now that was the ONLY enforcement: an endpoint
that never invoked it drove the entire sampling schedule against fake weights,
with no error, no warning and no output. se#840 spent two lanes reading that as
a hung `torch.export` — 21 minutes with the export never entered.

Discovery now counts ROUNDS at the marked module itself (the first-seen call
signature coming round again is the author's next step, whatever its arm count)
and stops the drive on a round boundary, loudly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
import gen_worker._vendor.torchcg  # noqa: E402,F401

FIXTURES = Path(__file__).resolve().parent / "release_fixtures"

LOCK = (
    "version = 1\n"
    '\n[[package]]\nname = "torch"\nversion = "2.13.0"\n'
    '\n[[package]]\nname = "triton"\nversion = "3.7.1"\n'
    '\n[[package]]\nname = "nvidia-cublas"\nversion = "13.1.1.3"\n'
    '\n[[package]]\nname = "diffusers"\nversion = "0.39.0"\n'
)


@pytest.fixture(scope="module")
def tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sys.path.insert(0, str(FIXTURES))
    try:
        import tiny_tree
    finally:
        sys.path.remove(str(FIXTURES))
    return tiny_tree.save_config_only(tmp_path_factory.mktemp("round-guard-configs"))


def _derive(tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    from gen_worker.release.derive import derive_release

    tmp_path.mkdir(parents=True, exist_ok=True)
    probe = tmp_path / "unet-calls"
    monkeypatch.setenv("GEN_WORKER_ROUND_PROBE", str(probe))
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text(LOCK)

    sys.path.insert(0, str(FIXTURES))
    try:
        import budget_ignoring_endpoint

        result = derive_release(
            budget_ignoring_endpoint,
            checkpoint_dir=tree,
            lockfile=lockfile,
            trace_workers=1,
        )
    finally:
        sys.path.remove(str(FIXTURES))
    calls = len(probe.read_text().split()) if probe.exists() else 0
    return result, calls, budget_ignoring_endpoint.STEPS


def test_a_loop_that_ignores_the_budget_is_STOPPED_at_the_budget(
    tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measurement, not the intention: how many times did the marked module
    actually run — against a CONTROL run of the same fixture with no budget at
    all, so the number means something without knowing how many extra passes
    the export stage makes.

    The fixture asks for 8 steps and never calls the callback. Before this
    change the drive paid all 8, and on a real endpoint that is the difference
    between 5 s and 139 s PER PAYLOAD, times the whole auto-enumerated fan.
    """
    from gen_worker.release import derive as derive_module

    _result, guarded, steps = _derive(tree, tmp_path, monkeypatch)

    # The control: the same fixture, same drive, no budget in force.
    monkeypatch.setattr(derive_module, "TRACE_STEP_BUDGET", None)
    control_result, unguarded, _ = _derive(tree, tmp_path / "control", monkeypatch)

    assert steps == 8
    assert unguarded >= steps, (
        f"the control only ran the marked module {unguarded} times for "
        f"{steps} steps — the probe is not measuring the drive"
    )
    assert control_result.warnings == (), (
        "the control reported a stop it did not make"
    )
    # Measured on this fixture: 18 control vs 6 guarded, over 2 derive items —
    # per item 8 driven steps + 1 export pass, against 1 driven step + the
    # blocked entry to the next + 1 export pass. The probe cannot separate the
    # export's re-run from the drive's, so the claim is the RATIO: a guarded
    # drive costs at most half of an unguarded one, and grows with the export
    # stage rather than with the author's schedule.
    assert guarded * 2 <= unguarded, (
        f"the drive ran the marked module {guarded} times against a control's "
        f"{unguarded} for a step budget of 1 — the guard did not stop a loop "
        f"that ignores the callback"
    )


def test_the_stop_is_SAID_in_the_lock_and_names_what_it_may_have_cost(
    tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A silent stop would be the same defect one layer down: the author would
    see a derive that finished and never learn its loop was cut, nor that a
    shape only a later step introduces was not banked."""
    result, _calls, _steps = _derive(tree, tmp_path, monkeypatch)
    said = {w for w in result.warnings if "STOPPED at step" in w}
    assert said, f"no stop was reported; warnings were {list(result.warnings)!r}"
    (sentence,) = said
    assert "ctx.step_callback()" in sentence
    assert "will mint on first live encounter" in sentence


def test_the_completed_step_s_graphs_are_STILL_derived(
    tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stop lands on a round boundary, so everything the completed step
    showed is banked. A guard that cost the lane its graphs would be worse than
    the hang it replaces."""
    result, _calls, _steps = _derive(tree, tmp_path, monkeypatch)
    assert result.lane_graphs
    assert all(graphs for graphs in result.lane_graphs.values())
    assert result.unmarked_lanes == ()
