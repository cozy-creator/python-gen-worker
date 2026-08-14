"""An SP follower must die with rank 0 — including on an ABORT.

Under the process split the compute child (rank 0) can die ``rc=-6``
(the NCCL abort), where atexit never runs, so ``daemon=True`` reaps
nothing. The parent then respawns the group onto the same cards in ~1 s while
the orphaned followers hold a full weight replica for their own 300 s
collective timeout — a crash loop seeded by its own orphans.

The fix is in the follower's OWN bootstrap (``_follower_main`` ->
``_die_with_rank0``): ``PR_SET_PDEATHSIG(SIGKILL)`` plus the spawn-race
re-parent check. This test drives the REAL bootstrap through the real
``multiprocessing`` spawn context and kills rank 0 with SIGKILL — the one
signal that cannot be handled, exactly like an abort for reaping purposes.
Reverting ``_die_with_rank0`` (or the ``rank0_pid`` plumbing in
``RankGroup.form``) turns this red: the follower survives its parent.

The ready signal is a FILE, not an mp queue, deliberately: the assertion must
only fire once the follower is provably past spawn bootstrap and inside
``entry`` (a follower killed mid-bootstrap dies of its half-built pipes and
would fake a reap), and a file does not put semaphores in the spawn payload —
the channel object is not what this test is about.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from harness import progress_wait

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="PR_SET_PDEATHSIG is Linux-only")

_SRC = str(Path(__file__).resolve().parent.parent / "src")

# Rank 0's stand-in: spawns ONE follower through the real _follower_main with
# the real spawn context and the real rank0_pid plumbing, then parks until the
# test SIGKILLs it. The entry reports ready (via the file channel) and parks —
# the property under test is the bootstrap, not the rendezvous.
_DRIVER = textwrap.dedent(
    """
    import multiprocessing as mp
    import os
    import sys
    import time

    from gen_worker.parallel.group import RankSpec, _follower_main


    class FileChannel:
        def __init__(self, path):
            self.path = path

        def report_ready(self, rank):
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                f.write(str(rank))
            os.replace(tmp, self.path)


    def entry(spec, channel):
        channel.report_ready(spec.rank)
        time.sleep(120)


    if __name__ == "__main__":
        ctx = mp.get_context("spawn")
        spec = RankSpec(1, 2, 1, "127.0.0.1", 0, "gloo", group_name="pgw820")
        proc = ctx.Process(
            target=_follower_main,
            args=(spec, entry, FileChannel(sys.argv[1]), None, os.getpid()),
            name="sp-pgw820-rank1",
            daemon=True,
        )
        proc.start()
        print(proc.pid, flush=True)
        time.sleep(120)
    """
)


def test_a_follower_is_reaped_when_rank0_aborts(tmp_path):
    driver = tmp_path / "rank0_driver.py"
    driver.write_text(_DRIVER)
    ready = tmp_path / "follower-ready"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([_SRC, env.get("PYTHONPATH", "")])
    rank0 = subprocess.Popen(
        [sys.executable, str(driver), str(ready)],
        stdout=subprocess.PIPE, env=env)
    try:
        follower_pid = int(rank0.stdout.readline().decode().strip())
        # Only a follower that is provably past bootstrap and inside `entry`
        # (pdeathsig armed) proves anything about surviving an abort.
        progress_wait.await_progress(
            ready.exists,
            lambda seen: seen,
            what="follower past bootstrap (ready file written from entry)",
            cadence=progress_wait.Cadence(),
            gone=lambda: (
                None if rank0.poll() is None
                else f"rank 0 exited early rc={rank0.returncode}"),
            poll_s=0.05,
        )
        assert ready.read_text() == "1"
        os.kill(follower_pid, 0)
        # Rank 0 "aborts": SIGKILL, like rc=-6, runs no atexit and no
        # multiprocessing daemon cleanup. Only the kernel can reap now.
        rank0.kill()
        rank0.wait(timeout=10)

        def _reaped() -> bool:
            try:
                os.kill(follower_pid, 0)
            except ProcessLookupError:
                return True
            return False

        progress_wait.await_progress(
            _reaped,
            lambda gone: gone,
            what=f"kernel reap of follower {follower_pid} after rank 0's abort",
            cadence=progress_wait.Cadence(),
            poll_s=0.05,
        )
    finally:
        if rank0.poll() is None:
            rank0.kill()
