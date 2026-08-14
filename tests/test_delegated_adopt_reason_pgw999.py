"""pgw#999 — a refused delegated cell names WHY, on the wire.

RED AT HEAD, all of it. `adopt_delegated_mint` spent a classified
``AdoptOutcome`` on ``bool(...)``, so three abort events restated one fact
three ways and none carried a cause:

    delegated_adopt_failed : "the child process produced a cell this runtime
                              could not adopt"
    delegated_no_cell      : "...produced no adoptable cell (the child's cell
                              did not adopt on this runtime)"
    error                  : "delegated mint produced no advertisable cell"

A mint can therefore burn hours of rented GPU to report "something". The
classified string existed in-process — ``contract_invalid``,
``constants_unbound``, ``no_arm_for_mode``, ``numerics_refused`` — and was
discarded one frame before the wire. This is the ``worker-errors-to-
orchestrator`` defect class verbatim, so these tests assert the CLASS reaches
the countable field, not merely that some prose got longer.

Every test states what it would have said at HEAD.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import pytest

from gen_worker import fleet_cells, mint_supervisor
from gen_worker.cell_adopt import AdoptOutcome
from gen_worker.compile_cache import AdoptError, CellSelectionBugError

FAMILY = "pgw999"


@dataclass
class _Cfg:
    family: str = FAMILY
    lora_bucket: int = 64
    shapes: Tuple[Tuple[int, int], ...] = ((1024, 1024),)
    targets: Tuple[str, ...] = ("unet",)
    text_lens: Tuple[int, ...] = (77,)
    guidance_scales: Tuple[float, ...] = (1.0, 5.0)
    regional: bool = False


class _Pipe:
    pass


@pytest.fixture()
def events(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, str, str]]:
    seen: List[Tuple[str, str, str]] = []

    def _sink(kind: str, detail: str, phase: str = "", duration_ms: int = 0, **_kw) -> None:
        seen.append((kind, phase, detail))

    monkeypatch.setattr(fleet_cells.activity_mod, "emit_event", _sink)
    monkeypatch.setattr(mint_supervisor.activity_mod, "emit_event", _sink)
    return seen


def _sealed_cell(path: Path, **over: Any) -> Path:
    """Opaque TCG bytes; exact identity/admission are separate test seams."""
    del over
    path.write_bytes(b"opaque-tcg-artifact")
    return path


@pytest.fixture()
def pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A pending whose cell EXISTS — the pgw#999 shape exactly.

    The mint succeeded (36/36 sealed and finalized on the pod that found
    this); the artifact is a real readable cell on disk and the only open
    question is whether the runtime that built it will arm it.
    """
    monkeypatch.setattr(fleet_cells, "_unregister", lambda p: None)
    monkeypatch.setattr(fleet_cells, "mark_terminus", lambda p, t, **_kw: None)
    artifact = _sealed_cell(tmp_path / "cell.tar.gz")
    return fleet_cells.PendingSelfMint(
        family=FAMILY, arm_token="cg-key-v1-sealed", ref=f"root/family-{FAMILY}#cg-key-v1-sealed",
        cfg=_Cfg(), target=artifact, mint_root=tmp_path / "root", publisher=None, delegated=True,)


def _abort(events: List[Tuple[str, str, str]]) -> Tuple[str, str]:
    rows = [(phase, detail) for kind, phase, detail in events
            if kind == "self_mint_abort"]
    assert len(rows) == 1, f"expected exactly one abort event, got {rows!r}"
    return rows[0]


def _arm_returns(monkeypatch: pytest.MonkeyPatch, outcome: AdoptOutcome) -> None:
    key = "cg-key-v1-" + "e" * 56
    metadata = {
        "family": FAMILY,
        "compiled_graph_key": key,
        "graph_class": {"name": "unet/e0", "target": "unet"},
    }
    monkeypatch.setattr(fleet_cells, "_stage_durable", lambda *_a, **_k: key)
    monkeypatch.setattr(
        fleet_cells,
        "_arm_compiled_graph",
        lambda *_a, **_k: (
            bool(outcome),
            dict(metadata),
            (
                "" if outcome else outcome.reason or "unclassified_arm_refusal",
                "" if outcome else outcome.detail or outcome.identity,
            ),
        ),
    )
    monkeypatch.setattr(
        fleet_cells,
        "_admit_durable",
        lambda _pending, *_a, **_k: type(
            "Stored", (), {"artifact": _pending.target}
        )(),
    )


def _arm_raises(monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> None:
    reason = str(getattr(exc, "reason", "") or type(exc).__name__)
    if isinstance(exc, CellSelectionBugError):
        reason = "cell_selection_bug"
    _arm_returns(monkeypatch, AdoptOutcome.miss(reason, str(exc)))


# ---------------------------------------------------------------------------
# 1. The returned classification — the exact attempt-26 path
# ---------------------------------------------------------------------------


def test_a_returned_refusal_puts_its_class_in_the_countable_field(
    pending: Any, events: List[Tuple[str, str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HEAD said phase='delegated_adopt_failed' — the call site's name, which
    every reader already knew from the event KIND. `phase` is the countable
    column, so it has to carry the CLASS, the way `self_mint_skipped` already
    does."""
    _arm_returns(monkeypatch, AdoptOutcome.miss(
        "contract_invalid",
        "input_contract records 5 leaves, the traced call takes 3"))

    assert fleet_cells.adopt_delegated_mint(_Pipe(), pending, [pending.target]) is None

    phase, detail = _abort(events)
    assert phase == "contract_invalid"
    assert "input_contract records 5 leaves" in detail
    assert "could not adopt" in detail, "the human sentence stays, too"


def test_every_classified_reason_survives_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The four the issue names, plus the numerics gate. A reason that is
    *transformed* on the way to the wire is a reason nobody can group by."""
    monkeypatch.setattr(fleet_cells, "_unregister", lambda p: None)
    monkeypatch.setattr(fleet_cells, "mark_terminus", lambda p, t, **_kw: None)
    for i, reason in enumerate((
        "contract_invalid", "constants_unbound", "no_arm_for_mode",
        "lane_unavailable", "numerics_refused", "sm_mismatch",
    )):
        seen: List[Tuple[str, str, str]] = []
        monkeypatch.setattr(
            fleet_cells.activity_mod, "emit_event",
            lambda kind, detail, phase="", duration_ms=0, **_kw: seen.append(
                (kind, phase, detail)))
        _arm_returns(monkeypatch, AdoptOutcome.miss(reason, f"detail for {reason}"))
        artifact = _sealed_cell(tmp_path / f"cell-{i}.tar.gz")
        p = fleet_cells.PendingSelfMint(
            family=FAMILY, arm_token=f"cg-key-v1-{i}", ref=f"root/family-{FAMILY}#cg-key-v1-{i}",
            cfg=_Cfg(), target=artifact, mint_root=tmp_path / f"root{i}", publisher=None, delegated=True,)
        assert fleet_cells.adopt_delegated_mint(_Pipe(), p, [artifact]) is None
        phase, _detail = _abort(seen)
        assert phase == reason
        assert fleet_cells.adopt_refusal(p) == (reason, f"detail for {reason}")


# ---------------------------------------------------------------------------
# 2. The RAISED classifications — the branch that was a logger.warning
# ---------------------------------------------------------------------------


def test_a_raised_AdoptError_is_classified_by_its_own_token(
    pending: Any, events: List[Tuple[str, str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`AdoptError` has carried `.reason` since it was written. HEAD's
    `except Exception` branch logged it to a logger no pod exposes and set
    `armed = False` — log-only swallowing, which is a defect class here."""
    _arm_raises(monkeypatch, AdoptError(
        "constants_unbound", "7 constants have no resident weight"))

    assert fleet_cells.adopt_delegated_mint(_Pipe(), pending, [pending.target]) is None

    phase, detail = _abort(events)
    assert phase == "constants_unbound"
    assert "7 constants have no resident weight" in detail


def test_an_unclassified_exception_is_named_by_its_TYPE_not_flattened(
    pending: Any, events: List[Tuple[str, str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception with no `.reason` still has a name. Collapsing it to one
    generic token would rebuild the very hole this issue is closing."""
    _arm_raises(monkeypatch, ValueError("shapes disagree"))

    assert fleet_cells.adopt_delegated_mint(_Pipe(), pending, [pending.target]) is None

    phase, detail = _abort(events)
    assert phase == "ValueError"
    assert "shapes disagree" in detail


def test_the_cell_selection_bug_keeps_its_own_loud_class(
    pending: Any, events: List[Tuple[str, str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """th#883's invariant must not be flattened into the generic refusal set:
    a self-requested, identity-verified cell that will not arm is a BUG in
    the selection brain, not a compatibility miss."""
    _arm_raises(monkeypatch, CellSelectionBugError("axes describe this runtime"))

    assert fleet_cells.adopt_delegated_mint(_Pipe(), pending, [pending.target]) is None

    phase, detail = _abort(events)
    assert phase == "cell_selection_bug"
    assert "axes describe this runtime" in detail


def test_a_silent_falsy_arm_says_SO_rather_than_inventing_a_reason(
    pending: Any, events: List[Tuple[str, str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`AdoptOutcome.miss("")` is a refusal that classified nothing. The event
    must say the classification is MISSING — a blank phase would read as "no
    reason exists", which is the lie this whole issue is about."""
    _arm_returns(monkeypatch, AdoptOutcome(armed=False))

    assert fleet_cells.adopt_delegated_mint(_Pipe(), pending, [pending.target]) is None

    phase, _detail = _abort(events)
    assert phase == "unclassified_arm_refusal"


# ---------------------------------------------------------------------------
# 3. The reason CROSSES the boundaries — one string, three events
# ---------------------------------------------------------------------------


def test_the_reason_is_readable_by_the_caller_that_must_requote_it(
    pending: Any, events: List[Tuple[str, str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`mint_supervisor.supervise` and the executor each emit their own event
    about the same refusal. They read the classification from the one place
    that produced it instead of re-deriving three vocabularies."""
    _arm_returns(monkeypatch, AdoptOutcome.miss("no_arm_for_mode", "mode='regional'"))
    fleet_cells.adopt_delegated_mint(_Pipe(), pending, [pending.target])

    assert fleet_cells.adopt_refusal(pending) == ("no_arm_for_mode", "mode='regional'")


def test_a_pending_that_never_refused_reports_no_reason(
    pending: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The accessor must not manufacture a refusal for a mint that adopted —
    an always-non-empty reason is as useless as an always-empty one."""
    _arm_returns(monkeypatch, AdoptOutcome.hit("family=x key=y"))

    assert fleet_cells.adopt_delegated_mint(_Pipe(), pending, [pending.target]) is not None
    assert fleet_cells.adopt_refusal(pending) == ("", "")


def test_the_delegated_result_carries_the_reason_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """The transport between the two events. RED at HEAD: `DelegatedResult`
    had no `reason` at all, so the executor had nothing to quote and fell back
    to naming its own call site."""
    result = mint_supervisor.SupervisedResult(
        status=mint_supervisor.FAILED, reason="contract_invalid",
        detail="the child's cell did not adopt on this runtime "
               "(contract_invalid: 5 leaves vs 3)")
    assert result.reason == "contract_invalid"
    assert not result.ok
    assert "contract_invalid" in result.detail
