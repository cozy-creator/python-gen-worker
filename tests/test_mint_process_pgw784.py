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

from gen_worker import child_contract
from gen_worker import mint_process as mp

GIB = 1 << 30
STUB_MODULE = "harness.mint_child_stub"


def _request(tmp_path: Path, **over: Any) -> mp.MintRequest:
    fields: Dict[str, Any] = dict(
        function="gen",
        modules=("harness.toy_endpoints",),
        family="sdxl",
        arm_token="arm1-deadbeef",
        target=str(tmp_path / "cell.tar.gz"),
        work_root=str(tmp_path / "capture"),
        report=str(tmp_path / mp.REPORT_NAME),
        cfg=child_contract.CompileSpec(family="sdxl", shapes=((1024, 1024),),
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
    assert child_contract.frame_line(phase="load").startswith(child_contract.FRAME_PREFIX)
    frames: list = []
    out = asyncio.run(_run(tmp_path, "minted", frames=frames))
    assert out.minted
    assert [f.phase for f in frames] == ["load"]


# ------------------------------------------------------------ happy path

def test_a_minted_cell_comes_back_as_a_path_and_a_digest(tmp_path: Path) -> None:
    frames: list = []
    out = asyncio.run(_run(tmp_path, "minted", frames=frames))
    assert out.status == mp.MINTED and out.minted
    # the child reports its ENTRY SET, so the outcome carries the
    # artifacts it produced. This vehicle mints one class, and the unpack
    # asserts that arity rather than indexing past a set nobody checked.
    (only,) = out.artifacts
    assert only == tmp_path / "cell.tar.gz"
    assert only.read_bytes() == b"stub-cell-bytes"
    # the digest rides the ENTRY row, beside the key and the path —
    # a per-artifact fact belongs with its artifact, not on the report.
    assert out.report is not None
    ((_key, _path, digest),) = out.report.entries
    assert digest == "blake3:stub"
    assert out.report.cell_key == "arm1-deadbeef"
    assert not out.retryable
    assert "status=minted" in out.line()


def test_exit_zero_without_an_artifact_is_a_crash_not_a_mint(
    tmp_path: Path,
) -> None:
    """A report is a claim; the file is the fact. th#1299's whole class of bug
    is trusting a claim about work that did not land."""
    out = asyncio.run(_run(tmp_path, "no_artifact"))
    assert out.status == mp.CRASHED
    assert "wrote no entry artifact" in out.detail
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


# FIVE ROWS DELETED HERE, and what they asserted.
#
# `co_residency` priced a mint child as `resident weights + one activation set
# + a 4 GiB inductor workspace + a CUDA context`, and its first term was read
# off the PARENT — "a legitimate proxy and not a guess: the child loads the
# same weights at the same lane". `fc77b923` made every production mint
# weight-free, so the child loads no weights at all, and the term was being
# added to a `free_bytes` that already excluded them. The rows that covered it
# asserted that arithmetic faithfully:
#
#   * `..._asks_for_a_whole_second_copy` — 6 resident + 2 activation + 4 + 1 =
#     13 GiB for a child that now holds nothing;
#   * `..._declines_instead_of_oomacking_the_tenant` — the wan-2.2 shape, and
#     the exact verdict §4.33 RETRACTS (113.19 GiB, "hardware-unsatisfiable");
#   * `..._unprobeable_card_never_blocks_a_mint` — a permissive branch of a
#     function that no longer exists;
#   * `..._one_mint_teaches_the_next...` — the monotone device bank;
#   * `..._cap_is_bytes_expressed_as_the_cards_real_fraction` — the child's
#     `set_per_process_memory_fraction` ceiling, computed from that same
#     estimate, which pinned two real mints at 11.09 GiB on cards with
#     21.48 GiB free.
#
# Nothing replaces them, because nothing replaces the prediction: a mint is
# attempted, and a card that cannot take it kills the child, which comes back
# classified (`test_mint_oom_classification_pgw848`).
