"""pgw#1243: a compile that finished its work and then failed to die.

Two production mints, two families, two wheels, one signature::

    kind=self_mint_compile  phase=finalize  state=running
    counter=compile:mint_child_evidence  counter_done=1077.99  counter_total=0
    self_stalled=FALSE   worker HEARTBEAT: 9s old

sdxl compiled 36 graph classes in 45m08s, sealed them, said ``finalize`` — and
then sat there for 78.9 minutes. z-image did the same for 62 minutes on
0.114.2 and had to be terminated by a human. Nothing was ever published, and
the serving path was fine throughout: both pods answered eager requests while
their mint was dead.

The load-bearing evidence was ``aot_mint_phases`` rows = 0. That table is
emitted PARENT-side and unconditionally the moment the compile call returns, so
zero rows means the parent never returned from it — the compiling process said
its last word and then never terminated.

TWO DEFECTS, and the second is what made the first invisible:

1. **The parent waited on the child's EXIT, not on its TERMINUS.** A compile
   child writes its report as its last statement and returns; everything after
   is interpreter teardown, and a process that has just traced and
   AOTI-compiled has plenty that can hang there — a non-daemon thread, a
   subprocess pool, CUDA. The whole compile was on disk and the supervisor
   threw it away waiting for a corpse.
2. **Ambient CPU was admitted as progress after the work was done, so no
   silence window could fire.** One production wedge burned *exactly one core*
   — which is what its ``@ 1.00/s`` counter rate literally is — and that
   re-touched the window every single poll, forever. An axis that cannot go
   silent is not a progress axis.

Both live in the SUPERVISION of a compile child, so pgw#1215 step 4 moving the
supervision down a tier moved the defect with it, and made it worse: the pool's
drain loop has no give-up test of any kind — ``proc.poll()`` forever,
``time.sleep`` forever — because the three-tier stack used to get one for free
from the deleted middle-tier supervisor. The surviving fix lives in the
compile pool (``aot_compile_pool``), which is the only production path.

Paul's ruling on the fix shape (2026-08-14):

    "The worker was fine, but the graph-compilation process failed and just
    stayed dead. In that case, it should report back to its parent execution
    group, which reports back to the worker, which reports back to tensorhub
    that the build failed. Just because the compilation failed doesn't mean
    the worker should be killed; it can continue to serve eager forever, but
    we should be aware of, and log, and make aware to operators, when builds
    fail, precisely why they fail."

Every tape drives the REAL supervisor against REAL child processes that really
wedge — spinning and blocked, with and without a report. ``timeout=`` values
are the TAPE's guard so a hang cannot take the suite with it; nothing in
production keys on a duration, and the detection tapes prove that by driving
the silence window itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest

from gen_worker import aot_compile_pool as pool
from gen_worker import child_contract
from gen_worker import mint_supervisor
from gen_worker import progress as progress_mod
from harness import fake_compile_child

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

#: The tape's own guard. Generous against the fix (which returns in about one
#: poll) and tiny against the defect (62 minutes in production), so a RED run
#: fails fast instead of hanging the suite.
_TAPE_GUARD_S = 90.0

#: A silence window a tape can actually drive. The detection tapes need it
#: reachable; the terminus tapes set it far ABOVE their own runtime on purpose,
#: so a pass there cannot be the window firing.
_SHORT_WINDOW_S = 3.0
_UNREACHABLE_WINDOW_S = 600.0

_DECLARED = 6


# ======================================================================
# TIER 1 — the compile pool: the production path since pgw#1215 step 4
# ======================================================================

def _pool(
    tmp_path: Path, *, mode: str, window_s: float, workers: int = 1,
) -> pool.EntryCompilePool:
    os.environ["PGW_FAKE_CHILD"] = mode
    os.environ["PGW_FAKE_DECLARED"] = str(_DECLARED)
    width = pool.entry_workers(
        _DECLARED, limit=workers, vcpus=16, available_bytes=64 * 1024**3,
        device_lock=True)
    assert width.workers == workers
    return pool.EntryCompilePool(
        tmp_path / "pool", width=width, cache_dir=str(tmp_path / "cache"),
        python=fake_compile_child.script(tmp_path),
        entry_silence_window_s=window_s)


def _template(tmp_path: Path) -> pool.EntryJob:
    return pool.EntryJob(
        function="generate", modules=("harness.toy_endpoints",),
        out_dir=str(tmp_path / "artifacts"))


@pytest.fixture(autouse=True)
def _clean_fake_child_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGW_FAKE_CHILD", raising=False)
    monkeypatch.delenv("PGW_FAKE_DECLARED", raising=False)


def _survivors(script: str) -> List[int]:
    """Every live process still running THIS tape's child script.

    Matched on the command line, not the parent's process group: the pool
    spawns with ``start_new_session=True`` so it can kill a child's whole
    group, which means a leaked child is never in the parent's group and a
    pgrp sweep can only ever return the empty set.
    """
    out = subprocess.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout
    return [
        int(line.split(None, 1)[0]) for line in out.splitlines()
        if script in line and " -eo " not in line]


def _kill_survivors(script: str) -> None:
    """End every child of this tape, group first and then the bare pid.

    Both, because a leaked child here is a process at 100% CPU on a shared
    machine: ``killpg`` is the right instrument, and ``getpgid`` can lose a
    race with the process it is asking about, so the pid kill is the backstop.
    """
    for pid in _survivors(script):
        with contextlib.suppress(OSError):
            os.killpg(os.getpgid(pid), 9)
        with contextlib.suppress(OSError):
            os.kill(pid, 9)


def test_a_share_that_packed_and_reported_and_never_died_still_lands(
    tmp_path: Path,
) -> None:
    """RED before pgw#1243, and this is the production wedge.

    The child packs every graph class in its share, writes its report, and
    then burns a core forever. The old pool polled ``proc.poll()`` and slept,
    with nothing else in the loop at all: the compile was complete, on disk and
    reported, and the pool would have sat on it until the pod was destroyed.

    The window is ten minutes and the tape guard is ninety seconds, so a pass
    CANNOT be a silence window firing: the only thing that can end this run is
    the report.
    """
    script = fake_compile_child.script(tmp_path)
    p = _pool(tmp_path, mode="wedged-after-report",
              window_s=_UNREACHABLE_WINDOW_S)
    packed = p.compile(_template(tmp_path))

    assert sorted(packed) == sorted(f"cls/dim={i}" for i in range(_DECLARED)), (
        "a child that packed and REPORTED its whole share produced those "
        "graph classes; its teardown is not the pool's business")
    for row in packed.values():
        assert Path(row.artifact).exists()
    # ...and the corpse does not outlive the pool. A wedged child the parent
    # walked away from would keep burning a core on a serving pod.
    assert _survivors(script) == [], (
        "the pool reaped the child's GROUP at its terminus; a spinning "
        "survivor is a pod billing for a compile nobody is waiting for")


def test_a_share_that_stops_making_progress_is_detected_and_named(
    tmp_path: Path,
) -> None:
    """RED before pgw#1243 — the drain loop had NO give-up test of any kind.

    ``hang`` writes no report and blocks: flat CPU, flat bytes, alive forever.
    Before this the pool polled ``proc.poll()`` and slept, so this run never
    ended and no wire event was ever emitted. The three-tier stack got this
    watch for free from the mint child's own supervisor; pgw#1215 step 4
    deleted that tier without moving the watch down with it.

    The window is three seconds against a ninety-second guard, so a pass
    proves the window FIRED.
    """
    script = fake_compile_child.script(tmp_path)
    p = _pool(tmp_path, mode="hang", window_s=_SHORT_WINDOW_S)

    with pytest.raises(pool.EntryCompileFailed) as caught:
        p.compile(_template(tmp_path))

    # Paul: "make aware to operators ... precisely why they fail." The reason
    # must name the share, what is missing, and what was measured instead.
    detail = str(caught.value)
    assert "no measured progress" in detail
    assert "wrote no report" in detail
    assert "process-tree CPU" in detail
    assert "keeps serving eager" in detail
    assert caught.value.entry, "the failure must name WHICH share wedged"
    assert not caught.value.resource, (
        "a wedge is not a memory shortfall — mis-classifying it would buy a "
        "second billed compile at a narrower K for a defect K cannot fix")
    assert _survivors(script) == [], (
        "condemning a share must take its whole group with it")


def test_a_share_that_is_really_compiling_is_never_condemned(
    tmp_path: Path,
) -> None:
    """The positive control, and it is part of the method.

    A window that cannot tell a wedge from a working compile is useless in the
    other direction, so prove the identical loop with the identical
    three-second window passes a real run.
    """
    p = _pool(tmp_path, mode="ok", window_s=_SHORT_WINDOW_S)
    packed = p.compile(_template(tmp_path))
    assert sorted(packed) == sorted(f"cls/dim={i}" for i in range(_DECLARED))


def test_a_burning_share_with_no_report_is_deliberately_NOT_condemned(
    tmp_path: Path,
) -> None:
    """THE HONEST LIMIT, asserted rather than left as a surprise.

    A pool child that burns CPU and has written no report is, on every signal
    that exists, INDISTINGUISHABLE from a child forty minutes into one
    ``aot_compile`` — that is the whole reason the shape has no phase frames to
    key on. So it is not condemned, and that is correct: the doctrine's cost is
    that a spinning wedge here is only detectable once it stops burning, and
    the alternative (a duration bound) kills real 45-minute compiles.

    What makes this acceptable is that the OBSERVED wedge is not this shape.
    Both production instances had already REPORTED — every graph class packed,
    manifest named — and the terminus rule above ends those in one poll.
    Recorded so nobody reads the absence as an oversight and nobody "fixes" it
    with a timeout.
    """
    script = fake_compile_child.script(tmp_path)
    p = _pool(tmp_path, mode="wedged-no-report", window_s=_SHORT_WINDOW_S)

    done: List[Any] = []

    def _drive() -> None:
        # The pool will never return on its own; when the tape kills its
        # children below it raises, and that raise is the tape's own doing.
        with contextlib.suppress(Exception):
            done.append(p.compile(_template(tmp_path)))

    thread = threading.Thread(target=_drive, daemon=True)
    thread.start()
    # Several windows' worth: a burning child is never condemned, however long
    # it burns.
    thread.join(timeout=_SHORT_WINDOW_S * 3)
    assert thread.is_alive() and not done, (
        "a share burning CPU must NEVER be condemned — that is a live compile "
        "on every signal that exists, and killing it is the failure mode the "
        "no-magic-timeouts doctrine exists to prevent")
    # Tape hygiene: end the children by hand and let the pool unwind.
    _kill_survivors(script)
    thread.join(timeout=30.0)
    assert _survivors(script) == [], "the tape must not leak a spinning child"


def test_a_refusal_that_hangs_on_the_way_out_is_still_a_refusal(
    tmp_path: Path,
) -> None:
    """The report carries the CLASSIFICATION, not just the artifacts.

    ``refuse-midway`` writes a ``status=refused`` report naming the classes it
    did pack. Read off a signal exit instead, the pool would report a share
    that exited on SIGKILL — losing both the refusal and the artifacts that
    ARE on disk, which is pgw#1183's whole point.
    """
    os.environ["PGW_FAKE_CHILD"] = "refuse-midway"
    p = _pool(tmp_path, mode="refuse-midway", window_s=_UNREACHABLE_WINDOW_S)
    with pytest.raises(pool.EntryCompileFailed) as caught:
        p.compile(_template(tmp_path))
    assert "ARE packed and on disk" in str(caught.value), (
        f"the partial pack must survive the refusal: {caught.value}")


# ======================================================================
# The counter that let it hide
# ======================================================================

class _Act:
    """Just enough Activity to record what the hub would have seen."""

    def __init__(self) -> None:
        self.phases: List[str] = []
        self.notes: List[str] = []

    def phase(self, phase: str, step: int = 0, total: int = 0) -> None:
        self.phases.append(phase)

    def note(self, detail: str) -> None:
        self.notes.append(detail)

    def heartbeat(self) -> None:
        pass

    def counter(
        self, name: str, unit: str, total: float = 0.0,
    ) -> progress_mod.Counter:
        return progress_mod.counter(name, unit, total, owner="tape")


def test_the_mint_counts_against_an_honest_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``counter_total=0`` is why nothing upstream noticed.

    The evidence counter counts against no total, so no fraction exists and a
    frozen position is indistinguishable from a slow one — which is exactly
    what both pods reported for an hour. The frames have carried a real
    ``step/total`` all along; it just never reached a COUNTER, and a counter is
    the only thing the hub's liveness rule and ``progress.self_diagnosis``
    read. Driven on a fake clock: no sleeping, no wall-clock rule.
    """
    progress_mod.reset()
    clock = {"t": 1_000.0}
    monkeypatch.setattr(progress_mod, "_now", lambda: clock["t"])
    act = _Act()

    apply = mint_supervisor._on_frame(act)
    apply(child_contract.MintFrame(
        phase="inductor_compile", step=1, total=4, note="share-000"))

    snap = progress_mod.freshest("tape")
    assert snap is not None and snap.name == mint_supervisor.PROGRESS_COUNTER
    assert snap.total == 4.0 and snap.done == 1.0, (
        f"an unbounded counter is not progress evidence: {snap}")

    # Nothing lands after that — which is the wedge. Past its own window the
    # worker CONFESSES instead of reporting `self_stalled=FALSE` for an hour.
    assert progress_mod.self_diagnosis("tape") is None
    clock["t"] += progress_mod.window_for(
        mint_supervisor.PROGRESS_COUNTER) + 1.0
    stalled = progress_mod.self_diagnosis("tape")
    assert stalled is not None
    assert stalled.name == mint_supervisor.PROGRESS_COUNTER
    progress_mod.reset()


# ======================================================================
# The chain: pool -> execution group -> worker -> hub
# ======================================================================

def _cfg() -> Any:
    from gen_worker.api.decorators import DynamicDim
    from gen_worker.registry import CompileCell
    return CompileCell(
        shapes=((1024, 1024),), targets=("unet",), family="sdxl",
        regional=False, text_len=77,
        dynamic=(DynamicDim(dim="batch", min=2, max=8),),
        lora_bucket=64, guidance_scales=(5.0,), text_lens=(77,))


def _task(tmp_path: Path) -> Any:
    from gen_worker import fleet_cells
    pending = fleet_cells.PendingSelfMint(
        family="sdxl", arm_token="ck1-abc",
        ref="root/family-sdxl#cg-key-v1-abc", cfg=_cfg(),
        target=tmp_path / "cell.tar.gz",
        mint_root=tmp_path / "root", publisher=None, cache_dir=tmp_path)
    pending.mint_root.mkdir(parents=True, exist_ok=True)
    return mint_supervisor.MintTask(
        pending=pending, pipe=object(), function="gen",
        modules=("harness.toy_endpoints",), weight_lane="fp8", device=0,
        handler_proof="resident warm forward 'gen' (real weights)")


def test_a_wedged_compile_fails_the_BUILD_and_never_the_WORKER(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paul's ruling, on the worker seam, with the wedge's own sentence.

    A condemned share leaves the pool as ``EntryCompileFailed``, which
    ``aot_mint`` turns into ``MintRefused`` (a wedge is deterministic — a
    retry buys a second billed 45-minute compile for a defect a retry cannot
    change). This drives the REAL ``supervise`` with that exception and asserts
    the three properties the ruling asks for past detection:

    2. TYPED PROPAGATION carrying the precise reason, all the way to the hub;
    3. the worker SERVES EAGER — ``supervise`` returns, never raises;
    4. the mint honestly ENDS — the obligation is resolved, so every
       "is this pod minting" predicate reads false afterwards.

    Per Paul's 2026-08-14 amendment this asserts NO occupancy/hold semantics:
    retiring a minting pod is fine now that graph classes land incrementally,
    and that debt is th#1930's. What is owed here is the honest terminal state,
    and nothing more.
    """
    from gen_worker import aot_mint, fleet_cells
    from gen_worker import activity as activity_mod

    monkeypatch.setattr(
        mint_supervisor, "assert_family_mintable", lambda family: None)
    monkeypatch.setattr(
        fleet_cells, "aot_export_spec",
        lambda pipe, cfg: SimpleNamespace(
            family="sdxl", strict=True, lora_bucket=0))
    monkeypatch.setattr(
        mint_supervisor, "export_declaration", lambda family: object())
    monkeypatch.setattr(
        aot_mint, "declared_class_rows", lambda pipe, spec, decl: [object()] * 4)

    seen: List[tuple] = []
    monkeypatch.setattr(
        activity_mod, "emit_event",
        lambda kind, detail, phase="", **_kw: seen.append((kind, phase, detail)))
    abandoned: List[Any] = []
    monkeypatch.setattr(
        fleet_cells, "abandon_self_mint",
        lambda pending: abandoned.append(pending))

    wedge = (
        "share-000 (rows[0::1]): the compile child made no measured progress "
        "for 312s (window 300s) and wrote no report — process-tree CPU 41.2s "
        "and its work dir 18.4MiB are both flat. It is wedged, not compiling; "
        "this build FAILS and this worker keeps serving eager")

    calls: List[int] = []

    def _mint(template: Any, **kw: Any) -> Any:
        calls.append(1)
        raise aot_mint.MintRefused(wedge)

    monkeypatch.setattr(aot_mint, "mint_graph_classes", _mint)

    act = _Act()
    result = asyncio.run(mint_supervisor.supervise(
        _task(tmp_path), act=act, max_attempts=3))

    # 3. THE WORKER LIVES: a failed build is a returned value, never a raise.
    assert result.status == mint_supervisor.FAILED and not result.ok
    # ...and it does not buy a second 45-minute compile for a wedge.
    assert len(calls) == 1 and result.attempts == 1, (
        "a wedge is deterministic: re-running it is a second billed compile "
        "for the same sentence")

    # 2. TYPED PROPAGATION, carrying the precise reason to the hub.
    aborts = [e for e in seen if e[0] == "self_mint_abort"]
    assert aborts, "a failed build must be wire-visible, not a pod-log line"
    _kind, phase, detail = aborts[0]
    assert phase.startswith("supervised_"), phase
    assert "no measured progress" in detail and "wrote no report" in detail, (
        f"the event must carry the operator-actionable observation: {detail}")
    assert "kept serving eager" in detail, detail

    # 4. THE MINT ENDS. No hold semantics are asserted (Paul, 2026-08-14):
    #    only that the obligation is resolved, so nothing downstream still
    #    believes a build is running.
    assert abandoned, (
        "a declared build failure must resolve the mint obligation — a build "
        "that has reported failure is not running")
