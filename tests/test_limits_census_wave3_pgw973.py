"""pgw#973 wave 3 — §4.24 dispositions that changed behaviour.

Three shapes are proven here, all on the real code paths:

1. **Absence must be explicit** (§4.24 item 4). A limit that is not stated may
   become a stated default or a refusal; it may never become "unlimited", and
   it may certainly never INVERT into a purge.
2. **Nothing that can end real work keys on a wall clock**. The lane
   gate polls free VRAM four times a second — free VRAM IS the progress
   signal — and used to give up on a flat 45 s deadline anyway.
3. **A duplicated bound has one owner.** Four verbatim pairs collapse; the
   fifth is deliberately NOT collapsed and says why.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pytest


# ---------------------------------------------------------------------------
# transport.SendQueue — `maxsize <= 0` deleted the bound outright
# ---------------------------------------------------------------------------


def test_an_unbounded_send_queue_is_refused_at_construction() -> None:
    """`while self._maxsize > 0 and ...` in the enqueue path meant a queue
    built with 0 shed nothing and blocked nobody: an unbounded outbound buffer
    in a pod whose progress/event producers never wait. `maxsize` is a public
    `Transport(...)` / `worker(...)` parameter, so it was one caller away."""
    from gen_worker.transport import DEFAULT_QUEUE_MAXSIZE, SendQueue

    with pytest.raises(ValueError, match="maxsize must be positive"):
        SendQueue(maxsize=0)
    with pytest.raises(ValueError, match="maxsize must be positive"):
        SendQueue(maxsize=-1)
    assert DEFAULT_QUEUE_MAXSIZE > 0
    assert SendQueue()._maxsize == DEFAULT_QUEUE_MAXSIZE


def test_transport_and_worker_share_one_declared_queue_depth() -> None:
    """The default was written `1024` in three places. One declaration now."""
    import inspect

    from gen_worker import transport, worker

    depth = transport.DEFAULT_QUEUE_MAXSIZE
    assert inspect.signature(
        transport.Transport.__init__).parameters["queue_maxsize"].default == depth
    assert inspect.signature(
        worker.Worker.__init__).parameters["queue_maxsize"].default == depth


# ---------------------------------------------------------------------------
# 3. lane_residency_gate — a flat deadline over a live progress signal
# ---------------------------------------------------------------------------


class _FakeResidency:
    """Enough of `Residency` for the gate: a demoted lane that promotes only
    after N attempts. Nothing is mocked about the gate itself."""

    def __init__(self, promote_after: int) -> None:
        self._left = promote_after
        self.attempts = 0

    def executing(self, ref: str) -> Any:
        from contextlib import nullcontext

        return nullcontext()

    def movable(self, ref: str) -> bool:
        return True

    def obj(self, ref: str) -> Any:
        return None

    def tier(self, ref: str) -> Any:
        from gen_worker.models.residency import Tier

        return Tier.RAM

    def promote(self, ref: str) -> bool:
        self.attempts += 1
        self._left -= 1
        return self._left <= 0


@pytest.fixture()
def gated(monkeypatch: pytest.MonkeyPatch) -> Any:
    from gen_worker.models import lane_residency_gate as gate_mod

    monkeypatch.setattr(gate_mod, "_cuda_available", lambda: True)
    monkeypatch.setattr(gate_mod, "_POLL_S", 0.01)
    return gate_mod


def test_a_card_that_keeps_returning_memory_is_never_given_up_on(
    gated: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED against the flat deadline: the window here is 0.05 s and the promote
    takes ~0.4 s of polling, so a wall budget fails the request. Free VRAM
    climbs the whole time — the sibling IS demoting — so the silence window
    never fires and the lane serves."""
    free = [10.0]

    def climbing() -> float:
        free[0] += 1.0          # far above the 64 MiB quantum
        return free[0]

    monkeypatch.setattr(gated, "get_available_vram_gb", climbing)
    res = _FakeResidency(promote_after=40)
    gate = gated.LaneResidencyGate(
        ref="lane", residency=res, wait_s=0.05,
        retry_exc=RuntimeError)     # type: ignore[arg-type]

    with gate.ensure_resident():
        pass
    assert res.attempts == 40
    # The property, not the runner's speed: at the loop's DECLARED
    # cadence the window covers only a handful of polls, and the promote took
    # far more — so a wall budget would have fired here. No clock is read.
    polls_the_window_could_cover = gate.wait_s / gated._POLL_S
    assert res.attempts > polls_the_window_could_cover, (
        "the test did not actually outlive the window it is testing"
    )


def test_a_card_that_has_stopped_moving_still_gives_up(
    gated: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half — the bound still exists. A genuinely stuck card reports
    the same free VRAM forever, the window fires, and the request is failed
    RETRYABLE rather than executing a cpu-resident lane."""
    monkeypatch.setattr(gated, "get_available_vram_gb", lambda: 3.0)
    res = _FakeResidency(promote_after=10**9)
    gate = gated.LaneResidencyGate(
        ref="lane", residency=res, wait_s=0.05,
        retry_exc=RuntimeError)     # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="stopped moving"):
        with gate.ensure_resident():
            pass


def test_allocator_jitter_cannot_keep_the_window_alive(
    gated: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The window keys on a 64 MiB quantum, not on the raw float, so noise
    below a component's size is not progress."""
    free = [3.0]

    def jitter() -> float:
        free[0] += 1e-6
        return free[0]

    monkeypatch.setattr(gated, "get_available_vram_gb", jitter)
    gate = gated.LaneResidencyGate(
        ref="lane", residency=_FakeResidency(promote_after=10**9),
        wait_s=0.05, retry_exc=RuntimeError)   # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        with gate.ensure_resident():
            pass


# ---------------------------------------------------------------------------
# 4. Verbatim duplicates now have one owner each — structural pins, so a
#    re-declaration fails here rather than drifting for a release.
# ---------------------------------------------------------------------------


_SRC = Path(__file__).resolve().parents[1] / "src" / "gen_worker"


def _declaring(name: str) -> List[str]:
    """Modules that ASSIGN ``name`` at module level (an import is not a
    declaration)."""
    out: List[str] = []
    for path in _SRC.rglob("*.py"):
        for line in path.read_text().splitlines():
            if line.startswith(f"{name} ="):
                out.append(str(path.relative_to(_SRC)))
                break
    return out


def test_the_pinned_host_memory_cap_has_one_owner() -> None:
    from gen_worker import media_transfer
    from gen_worker.models import w8a8_lora

    assert w8a8_lora._PIN_MAX_BYTES is media_transfer.PIN_MAX_BYTES
    assert _declaring("PIN_MAX_BYTES") == ["media_transfer.py"]
    assert _declaring("_PIN_MAX_BYTES") == []


def test_the_nvidia_smi_budget_has_one_owner() -> None:
    from gen_worker import cuda_probe, hardware_report, host_canary

    assert host_canary._NVIDIA_SMI_TIMEOUT_S is cuda_probe.NVIDIA_SMI_TIMEOUT_S
    assert hardware_report._NVIDIA_SMI_TIMEOUT_S is cuda_probe.NVIDIA_SMI_TIMEOUT_S
    assert _declaring("NVIDIA_SMI_TIMEOUT_S") == ["cuda_probe.py"]


def test_the_host_ram_floor_has_one_owner_and_one_derivation() -> None:
    """`residency` and `staging` each declared the two constants AND re-derived
    the same min/max expression, with staging's comment promising it was "kept
    numerically identical" — a promise nothing enforced. They cannot import
    each other (`residency -> pinned_swap -> staging` already exists)."""
    from gen_worker.models import memory, residency, staging

    assert _declaring("_RAM_FLOOR_GB") == [str(Path("models/memory.py"))]
    assert _declaring("_RAM_FLOOR_FRACTION") == [str(Path("models/memory.py"))]
    assert residency._effective_ram_floor_gb() == memory.effective_ram_floor_gb()
    assert staging._floor_bytes() == int(memory.effective_ram_floor_gb() * 1024 ** 3)
    # The policy itself, exercised rather than asserted from the constants.
    assert memory.effective_ram_floor_gb(total_gb=1000.0) == 8.0   # flat above 40 GiB
    assert memory.effective_ram_floor_gb(total_gb=20.0) == 4.0     # adaptive below
    assert memory.effective_ram_floor_gb(total_gb=0.0) == 8.0      # unreadable host


# ---------------------------------------------------------------------------
# 5. Deletions — the bounds that were never read
# ---------------------------------------------------------------------------


def test_the_file_api_timeouts_are_gone() -> None:
    """Five `_FILE_API_*_TIMEOUT_S` constants in `request_context/_helpers.py`
    were defined, re-exported, and read by nothing — in `src`, in `tests`, or
    anywhere else. §4.24: a limit that reaches no call site prevents nothing."""
    hits = [
        str(p.relative_to(_SRC))
        for p in _SRC.rglob("*.py")
        if "_FILE_API" in p.read_text()
    ]
    assert hits == []


def test_the_mediated_action_ceiling_is_the_allowlists_own() -> None:
    """`_ACTION_HARD_TIMEOUT_S = 120.0` sat in a `min()` beside
    `action.timeout_s`, whose largest declared value is 60 s — so the third
    term could only ever reject nothing (§4.24 item 1). Behaviour is proven on
    the real mediation path in `test_procsplit.py`."""
    from gen_worker.procsplit import actions, parent

    assert not hasattr(parent, "_ACTION_HARD_TIMEOUT_S")
    assert max(a.timeout_s for a in actions.ACTIONS.values()) == 60.0


def test_the_parent_beat_fallback_is_declared_not_buried() -> None:
    """`beat_interval_s=0.0` means "adopt the child's cadence"; when no child
    declares one the loop fell back to a bare `10.0` inside its body."""
    from gen_worker.procsplit import parent

    assert parent._BEAT_INTERVAL_FALLBACK_S == 10.0


def test_the_mint_reap_grace_is_named_and_argues_its_exemption() -> None:
    """A fixed duration that is EXEMPT from gw#666, stated as such: it runs
    after `_terminate_group` has already signalled, so the kill decision was
    made upstream on `_EVIDENCE_WINDOW_S`, which is a progress signal."""
    from gen_worker import mint_process

    assert mint_process._REAP_GRACE_S == 15.0
    assert mint_process._EVIDENCE_WINDOW_S > 0


# ---------------------------------------------------------------------------
# 6. mint_child.cap_vram — DELETED WITH ITS INPUT
# ---------------------------------------------------------------------------
#
# "uncapped" is a state to STATE, not an
# absence to infer. The state it named is gone — §4.33 deleted the budget that
# computed `vram_cap_bytes`, so there is no ceiling to apply and no silence to
# distinguish from one. The RULE survives everywhere else in this census.
