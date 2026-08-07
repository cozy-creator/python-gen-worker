"""pgw#784: the mint runs in its OWN OS process, and its death is not the
worker's.

th#1299's tape, restated as the contract this file pins (WORKER-CONTRACTS §2):
a compile-cell MISS must not put long-running GIL-holding Python on the loop
that carries the 10s beat and eager serving. So the mint becomes a child
process, and this file proves the SUPERVISOR half:

* the boundary is files + argv, never a pickled live object (spawn, not fork —
  a CUDA context cannot survive fork, and the child loads what it needs);
* every child death is CLASSIFIED, and the classification decides the retry:
  a named refusal is terminal, a resource shortfall and an unclassified crash
  get exactly one more attempt;
* the parent's liveness verdict is MEASURED (process-tree CPU + capture bytes),
  never a wall clock and never a frame the child prints — a busy child is never
  killed however long it takes, a silent one is killed quickly;
* abandonment reaps the whole process GROUP (inductor forks its own compile
  workers, and a mint that leaks them keeps billing for a cell nobody adopts).

The liveness proof itself — beats never missed while a >2min mint runs — is
``test_mint_liveness_pgw784.py``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import msgspec
import pytest
import torch

from gen_worker import mint_budget
from gen_worker import mint_process as mp

GIB = 1 << 30
STUB_MODULE = "harness.mint_child_stub"


def _request(tmp_path: Path, **over: Any) -> mp.MintRequest:
    fields: Dict[str, Any] = dict(
        function="gen",
        modules=("harness.toy_endpoints",),
        family="sdxl",
        cell_key="ck1-deadbeef",
        target=str(tmp_path / "cell.tar.gz"),
        work_root=str(tmp_path / "capture"),
        report=str(tmp_path / mp.REPORT_NAME),
        cfg=mp.CompileCellSpec(family="sdxl", shapes=((1024, 1024),),
                               targets=("unet",)),
    )
    fields.update(over)
    return mp.MintRequest(**fields)


def _env(mode: str, **extra: str) -> Dict[str, str]:
    """The stub child's environment: the parent's, plus the tests/ dir so
    ``harness.mint_child_stub`` is importable in a fresh interpreter."""
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), str(root / "tests"), env.get("PYTHONPATH", "")])
    env["MINT_STUB_MODE"] = mode
    env.update(extra)
    return env


async def _run(
    tmp_path: Path, mode: str, *, request: Optional[mp.MintRequest] = None,
    frames: Optional[list] = None, **kw: Any,
) -> mp.MintOutcome:
    req = request if request is not None else _request(tmp_path)
    return await mp.run_mint(
        req,
        workdir=tmp_path / "work",
        env=_env(mode, **{k: str(v) for k, v in kw.pop("child_env", {}).items()}),
        on_frame=(frames.append if frames is not None else None),
        **kw,
    )


@pytest.fixture(autouse=True)
def _stub_child(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mp, "MINT_CHILD_MODULE", STUB_MODULE)


# ---------------------------------------------------------------- protocol

def test_the_boundary_is_a_file_you_can_rerun(tmp_path: Path) -> None:
    """The whole request is one JSON file and one argv — that is what makes a
    failed mint reproducible by hand instead of only in a dead pod's logs."""
    req = _request(tmp_path)
    path = mp.write_request(tmp_path / "work", req)
    assert path.name == mp.REQUEST_NAME
    assert msgspec.json.decode(path.read_bytes(), type=mp.MintRequest) == req
    argv = mp.child_argv(path)
    assert argv[0] == sys.executable and argv[1] == "-m"
    assert argv[2] == mp.MINT_CHILD_MODULE and argv[3] == str(path)


def test_child_env_pins_the_card_and_declares_itself(tmp_path: Path) -> None:
    env = mp.child_env(_request(tmp_path, device=2), base={"PATH": "/bin"})
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["GEN_WORKER_MINT_CHILD"] == "1"
    # device=-1 means "leave the child's default" — never an accidental pin.
    assert "CUDA_VISIBLE_DEVICES" not in mp.child_env(
        _request(tmp_path), base={"PATH": "/bin"})


def test_a_stray_print_is_not_a_frame(tmp_path: Path) -> None:
    """Frames are prefixed. Endpoint code, torch and pip all print; none of
    them may steer a mint."""
    assert mp.frame_line(phase="load").startswith(mp.FRAME_PREFIX)
    frames: list = []
    out = asyncio.run(_run(tmp_path, "minted", frames=frames))
    assert out.minted
    assert [f.phase for f in frames] == ["load"]


# ------------------------------------------------------------ happy path

def test_a_minted_cell_comes_back_as_a_path_and_a_digest(tmp_path: Path) -> None:
    frames: list = []
    out = asyncio.run(_run(tmp_path, "minted", frames=frames))
    assert out.status == mp.MINTED and out.minted
    assert out.artifact == tmp_path / "cell.tar.gz"
    assert out.artifact.read_bytes() == b"stub-cell-bytes"
    assert out.report is not None and out.report.digest == "blake3:stub"
    assert out.report.cell_key == "ck1-deadbeef"
    assert not out.retryable
    assert "status=minted" in out.line()


def test_exit_zero_without_an_artifact_is_a_crash_not_a_mint(
    tmp_path: Path,
) -> None:
    """A report is a claim; the file is the fact. th#1299's whole class of bug
    is trusting a claim about work that did not land."""
    out = asyncio.run(_run(tmp_path, "no_artifact"))
    assert out.status == mp.CRASHED
    assert "wrote no artifact" in out.detail
    assert out.retryable


# --------------------------------------------------- failure inversion

@pytest.mark.parametrize(
    "mode, status, retryable",
    [
        ("refused", mp.REFUSED, False),
        ("resource", mp.RESOURCE, True),
        ("crash", mp.CRASHED, True),
        ("sigkill", mp.CRASHED, True),
        ("bad_request", mp.REFUSED, False),
    ],
)
def test_every_child_death_is_classified_and_drives_the_retry(
    tmp_path: Path, mode: str, status: str, retryable: bool,
) -> None:
    """A dead mint is an OUTCOME, never an exception that could take the
    worker with it — ``run_mint`` returns in every branch.

    And the class decides the retry: re-running a NAMED refusal buys a second
    billed compile for the same sentence, so it is terminal; a resource
    shortfall may have been the tenant's peak, which has since passed.
    """
    out = asyncio.run(_run(tmp_path, mode))
    assert out.status == status
    assert out.retryable is retryable
    assert out.detail, "a failed mint must always name its reason"


def test_a_signal_death_names_the_signal(tmp_path: Path) -> None:
    out = asyncio.run(_run(tmp_path, "sigkill"))
    assert out.exit_code is not None and out.exit_code < 0
    assert "SIGKILL" in out.detail


def test_a_crash_keeps_the_childs_stderr_for_diagnosis(tmp_path: Path) -> None:
    """A serve pod exposes no logs, so the child's last words have to ride
    back to the parent or they are lost."""
    out = asyncio.run(_run(tmp_path, "crash"))
    assert "stub child exploded" in out.stderr_tail


def test_max_attempts_is_two_not_a_loop(tmp_path: Path) -> None:
    """The retry policy is explicit and bounded. th#1288/ie#576: every retry
    is a billed pod, so 'retry until it works' is not a policy."""
    assert mp.MAX_ATTEMPTS == 2


# ------------------------------------------- observe the machine, not a clock

def test_a_busy_child_is_never_killed_however_long_it_takes(
    tmp_path: Path,
) -> None:
    """No wall-clock cap. A mint that keeps burning CPU keeps running — the
    9.5-minute sdxl mints in activity.py's own docstring are healthy work."""
    out = asyncio.run(_run(
        tmp_path, "busy",
        child_env={"MINT_STUB_SECONDS": 3},
        evidence_window_s=1.0, observe_interval_s=0.3))
    assert out.status == mp.MINTED, out.detail
    assert out.elapsed_s >= 2.5


def test_capture_bytes_alone_keep_a_cpu_quiet_child_alive(
    tmp_path: Path,
) -> None:
    """A compile parked in a C++ toolchain still grows the capture dir. Either
    signal alone is progress; demanding both would kill real work."""
    out = asyncio.run(_run(
        tmp_path, "grow",
        child_env={"MINT_STUB_SECONDS": 3},
        evidence_window_s=1.5, observe_interval_s=0.3))
    assert out.status == mp.MINTED, out.detail


def test_a_child_making_no_measured_progress_is_reaped(tmp_path: Path) -> None:
    """The other half of the same rule: silence with no CPU and no bytes is a
    wedge, and it dies quickly rather than billing the pod forever (pgw#786's
    complaint, applied here)."""
    out = asyncio.run(_run(
        tmp_path, "silent",
        child_env={"MINT_STUB_SECONDS": 120},
        evidence_window_s=1.0, observe_interval_s=0.3))
    assert out.status == mp.CRASHED
    assert "no measured progress" in out.detail
    assert out.retryable
    assert out.elapsed_s < 30, "the reap must not wait out the child"


def test_abandon_reaps_the_child_and_is_not_a_failure(tmp_path: Path) -> None:
    """Adopt-on-arm, vacate and shutdown all abandon a mint. That is an
    outcome, not a failure — and explicitly NOT retryable."""

    async def _go() -> mp.MintOutcome:
        stop = asyncio.Event()
        task = asyncio.ensure_future(_run(
            tmp_path, "silent", child_env={"MINT_STUB_SECONDS": 120},
            abandon=stop, observe_interval_s=0.2))
        await asyncio.sleep(0.8)
        stop.set()
        return await task

    out = asyncio.run(_go())
    assert out.status == mp.ABANDONED
    assert not out.retryable
    assert out.elapsed_s < 30


# ------------------------------------------------------ VRAM co-residency

def _fake_card(
    monkeypatch: pytest.MonkeyPatch, *, total_gib: float, resident_gib: float,
    peak_gib: float,
) -> None:
    total = int(total_gib * GIB)
    resident = int(resident_gib * GIB)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda dev=0: (total - resident, total))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda dev=0: resident)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda dev=0: resident)
    monkeypatch.setattr(
        torch.cuda, "max_memory_allocated", lambda dev=0: int(peak_gib * GIB))


def test_co_residency_asks_for_a_whole_second_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The child is its own process, so it holds its OWN weights. State that
    honestly rather than pretending the boundary is free."""
    _fake_card(monkeypatch, total_gib=24, resident_gib=6, peak_gib=8)
    budget = mint_budget.co_residency(0, family="sdxl", weight_lane="fp8")
    # resident 6 + activation (8-6=2) + 4 workspace + 1 context
    assert budget.need_bytes == pytest.approx(13 * GIB, rel=0.01)
    assert budget.fits


def test_a_weight_heavy_family_declines_instead_of_oomacking_the_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wan-2.2 shape (54 GiB resident on an 80 GiB card): a second copy
    does not fit, so the mint DECLINES. Eager serving, cell absent, a roomier
    pod mints it — pgw#737's policy, unchanged by the process boundary."""
    _fake_card(monkeypatch, total_gib=80, resident_gib=54, peak_gib=60)
    assert not mint_budget.co_residency(0, family="wan-2.2").fits


def test_an_unprobeable_card_never_blocks_a_mint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    budget = mint_budget.co_residency(0)
    assert budget.fits and not budget.probed
    assert "unprobeable" in budget.line("self_mint_skipped", "x")


def test_one_mint_teaches_the_next_and_the_ask_never_drifts_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NO MAGIC NUMBERS: the second ask on a pod is a MEASURED child peak, and
    it is monotone — a mint that peaked high once can peak that high again."""
    _fake_card(monkeypatch, total_gib=80, resident_gib=6, peak_gib=8)
    estimated = mint_budget.co_residency(0, family="f", weight_lane="l")
    mint_budget.record_child_peak("f", "l", 30 * GIB)
    measured = mint_budget.co_residency(0, family="f", weight_lane="l")
    assert measured.need_bytes > estimated.need_bytes
    assert measured.measured
    mint_budget.record_child_peak("f", "l", 2 * GIB)
    assert mint_budget.child_peak("f", "l") == 30 * GIB


def test_the_cap_is_bytes_expressed_as_the_cards_real_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The child's bound is the parent's byte reservation — enforcement, not a
    fraction anybody guessed. An under-estimate becomes the CHILD's OOM."""
    from gen_worker import mint_child

    seen: Dict[str, Any] = {}
    _fake_card(monkeypatch, total_gib=24, resident_gib=0, peak_gib=0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda dev: None)
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction",
        lambda frac, dev=0: seen.update(frac=frac, dev=dev))
    note = mint_child.cap_vram(0, 12 * GIB)
    assert seen["frac"] == pytest.approx(0.5, rel=0.01)
    assert "12.00GiB" in note
    seen.clear()
    assert mint_child.cap_vram(0, 0) == "" and not seen
