"""pgw#973 wave 3 — §4.24 dispositions that changed behaviour.

Three shapes are proven here, all on the real code paths:

1. **Absence must be explicit** (§4.24 item 4). A limit that is not stated may
   become a stated default or a refusal; it may never become "unlimited", and
   it may certainly never INVERT into a purge.
2. **Nothing that can end real work keys on a wall clock** (gw#666). The lane
   gate polls free VRAM four times a second — free VRAM IS the progress
   signal — and used to give up on a flat 45 s deadline anyway.
3. **A duplicated bound has one owner.** Four verbatim pairs collapse; the
   fifth is deliberately NOT collapsed and says why.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

import pytest


# ---------------------------------------------------------------------------
# 1. aot_resume — TWO absence collapses in one expression, one of them an
#    inversion that DELETED the corpus it was supposed to bound.
# ---------------------------------------------------------------------------


def _area(tmp_path: Path, *names: str, size: int = 4096) -> List[Path]:
    """A real resume area on disk: one directory per scope with real bytes."""
    area = tmp_path / ".mint-resume"
    made: List[Path] = []
    for i, name in enumerate(names):
        scope = area / name
        scope.mkdir(parents=True)
        (scope / "blob.bin").write_bytes(b"\0" * size)
        # Distinct mtimes so "oldest first" is well-defined.
        os.utime(scope, (1_700_000_000 + i, 1_700_000_000 + i))
        made.append(scope)
    return made


def test_an_env_zero_no_longer_purges_every_banked_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE INVERSION. `cap = int(max_bytes or os.environ.get(ENV) or DEFAULT)`
    read the env as the string ``"0"``, which is TRUTHY — so
    ``GEN_WORKER_MINT_RESUME_MAX_BYTES=0``, the spelling an operator would
    reach for to mean "no cap", produced ``cap = 0`` and the sweep then removed
    every scope except ``keep``. A capacity bound turned into a delete-all.
    """
    from gen_worker import aot_resume

    keep, other_a, other_b = _area(tmp_path, "keep", "old-a", "old-b")
    monkeypatch.setenv(aot_resume.ENV_MAX_BYTES, "0")

    with pytest.raises(ValueError) as exc:
        aot_resume.sweep(keep)
    assert aot_resume.ENV_MAX_BYTES in str(exc.value)

    # The corpus is intact — which is the whole point. Under the old
    # expression this assertion fails with both siblings deleted.
    assert other_a.exists() and other_b.exists() and keep.exists()


def test_a_caller_computed_zero_is_refused_not_silently_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second collapse in the same expression, in the OPPOSITE direction:
    a caller passing ``max_bytes=0`` fell through to the 4 GiB default, so a
    budget that computed to nothing silently got the module's own number."""
    from gen_worker import aot_resume

    monkeypatch.delenv(aot_resume.ENV_MAX_BYTES, raising=False)
    with pytest.raises(ValueError, match="max_bytes must be positive"):
        aot_resume.resume_area_cap_bytes(0)
    with pytest.raises(ValueError, match="max_bytes must be positive"):
        aot_resume.resume_area_cap_bytes(-1)
    # Absence — and ONLY absence — takes the stated default.
    assert aot_resume.resume_area_cap_bytes(None) == aot_resume.DEFAULT_MAX_BYTES
    assert aot_resume.resume_area_cap_bytes(1024) == 1024


def test_a_malformed_env_names_itself_instead_of_raising_bare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gen_worker import aot_resume

    monkeypatch.setenv(aot_resume.ENV_MAX_BYTES, "4GiB")
    with pytest.raises(ValueError, match="not an integer byte count"):
        aot_resume.resume_area_cap_bytes()
    monkeypatch.setenv(aot_resume.ENV_MAX_BYTES, "-5")
    with pytest.raises(ValueError, match="must be positive"):
        aot_resume.resume_area_cap_bytes()


def test_the_cap_still_sweeps_when_it_is_stated(tmp_path: Path) -> None:
    """The bound is not merely safe — it still does its job, oldest first."""
    from gen_worker import aot_resume

    # `_area` stamps ascending mtimes, so `old-a` is the oldest non-keep.
    keep, old_a, old_b = _area(tmp_path, "old-a", "old-b", "keep", size=4096)
    dropped = aot_resume.sweep(keep, max_bytes=9000)   # room for two scopes
    assert dropped == 1
    assert keep.exists() and old_b.exists() and not old_a.exists(), (
        "the sweep evicted the NEWEST abandoned scope — the one most likely "
        "to be resumed next — instead of the oldest"
    )


def test_a_bad_env_disables_the_bank_rather_than_failing_the_mint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`open_bank` never fails a mint for a cache — the refusal surfaces as a
    warning and no bank, not as a dead mint."""
    from gen_worker import aot_resume

    monkeypatch.setenv(aot_resume.ENV_MAX_BYTES, "0")
    (keep,) = _area(tmp_path, "keep")
    assert aot_resume.open_bank(str(keep)) is None


# ---------------------------------------------------------------------------
# 2. transport.SendQueue — `maxsize <= 0` deleted the bound outright
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
    # The property, not the runner's speed (pgw#795): at the loop's DECLARED
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


def test_serving_headroom_is_one_number_pgw973() -> None:
    """The fifth pair is deliberately NOT collapsed: `aot_wrapper_split` is an
    inductor hook whose only in-repo imports are `host_isa` and
    `aot_run_impl_split`, and importing `aot_compile_pool` would drag the whole
    mint driver into it. The duplication is stated at the site; this pins the
    two values together so a drift is a failure and not a surprise."""
    from gen_worker import aot_compile_pool, aot_wrapper_split

    assert (aot_wrapper_split.SERVING_HEADROOM_CPUS
            == aot_compile_pool.SERVING_HEADROOM_CPUS)


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
    the real mediation path in `test_procsplit_security_pgw763`."""
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
# 6. mint_child.cap_vram — DELETED WITH ITS INPUT (pgw#1175)
# ---------------------------------------------------------------------------
#
# pgw#973 §4.24 item 4 was right: "uncapped" is a state to STATE, not an
# absence to infer. The state it named is gone — §4.33 deleted the budget that
# computed `vram_cap_bytes`, so there is no ceiling to apply and no silence to
# distinguish from one. The RULE survives everywhere else in this census.


