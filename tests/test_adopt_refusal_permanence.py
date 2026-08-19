"""pgw#1472: a permanent adoption refusal must be a STATE, not a log line.

`ServeAdoption.sink_for` swallows every failure into eager-forever, and that is
correct as a boot policy — adoption must never kill a pod. It is wrong as an
observability one: **an unconditional refusal is indistinguishable from a
correct one**, because both read as "serving eager". That is how an env-identity
mismatch could have made every locally derived boot eager forever while looking
healthy, and it is why the first version of pgw#1472's own filing was wrong for
an hour.

So the distinction is carried on the object.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker.serving.serve_adoption import PERMANENT_REFUSALS, ServeAdoption


@pytest.fixture()
def adoption(tmp_path: Path) -> ServeAdoption:
    return ServeAdoption("rel-1", sm="sm_89", artifacts_dir=tmp_path / "artifacts")


def test_a_fresh_session_claims_no_refusal_at_all(adoption: ServeAdoption) -> None:
    """Absent must not render as `False`-because-nothing-happened: before any
    refusal, BOTH the reason and the permanence read as unset."""

    assert adoption.refusal == ""
    assert adoption.refusal_permanent is False


def test_an_environment_mismatch_is_marked_PERMANENT(adoption: ServeAdoption) -> None:
    """A retry cannot fix the pod not being the release's env."""

    adoption._refuse("environment_mismatch", "resolved closure abc != stamped def")
    assert adoption.refusal_permanent is True
    assert "environment_mismatch" in adoption.refusal


def test_a_hub_failure_is_marked_RETRYABLE(adoption: ServeAdoption) -> None:
    """The counter-case, without which the flag could be a constant."""

    adoption._refuse("TimeoutError", "the adopt route did not answer in time")
    assert adoption.refusal_permanent is False


def test_the_permanent_set_is_named_and_every_member_is_reachable() -> None:
    """A vocabulary nothing emits is folklore. Each of these is a phase this
    module actually passes to `_refuse`, or an exception type it catches."""

    import gen_worker.serving.serve_adoption as module

    source = Path(module.__file__).read_text()
    for phase in PERMANENT_REFUSALS:
        assert f'"{phase}"' in source, (
            f"{phase!r} is declared permanent but this module never emits it; "
            f"a set that cannot fire is not a classification")


def test_the_refusal_is_readable_off_the_OBJECT_not_only_the_log(
    adoption: ServeAdoption, caplog: pytest.LogCaptureFixture
) -> None:
    """The whole point. A pod's stdout goes nowhere (pgw#760), so a fact that
    reaches only the logger is a fact nothing can count."""

    adoption._refuse("environment_mismatch", "closure mismatch")
    assert (adoption.refusal, adoption.refusal_permanent) == (
        "environment_mismatch: closure mismatch", True)
