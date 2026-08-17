"""pgw#833: a pre-Hello compute-child death must carry its OWN crash text.

The first hub-launched pod under the unconditional split died pre-Hello six
times and the wire carried only ``exit:1`` — the child's stderr (its crash
traceback, its last startup phase) lives only in the container log, and
RunPod exposes no container-logs API. Diagnosing that took three
paid probe pods. The fix: the parent captures the child's stderr (teeing
every byte back to its own stderr so the container log is unchanged) and
attaches the tail to the compute_process_exit post-mortem dial and to the
compute_boot_crash_loop give-up dial.

Also here: the pgw#826 follow-on ack. ``send_boot_fatal`` now waits (bounded)
for the parent's T_BOOT_FATAL_ACK, written only after the verdict is
recorded, so a fast-exiting child can no longer be reaped before the parent
reads the frame (CI run 30692482234's race).

All tests drive the REAL codepaths: real ParentControl, real child
subprocesses, real unix-socket frames; dials are captured in-process.
"""

from __future__ import annotations

import sys

from test_procsplit_pgw763 import (  # noqa: F401  (fixtures re-used)
    SplitHarness,
    captured_dials as captured_dials,
    captured_reports as captured_reports,
    isolated_postmortem as isolated_postmortem,
)

MARKER = "PGW833_THE_CHILD_SAYS_WHY_IT_DIED"


def test_pre_hello_death_dial_carries_the_childs_stderr(
    tmp_path, captured_dials, capfd,
):
    """The load-bearing row (RED without the fix): a child that dies before
    Hello with a distinctive stderr line gets that line into (1) every
    compute_process_exit dial, (2) the compute_boot_crash_loop give-up dial,
    and (3) the parent's own stderr — the container-log contract the pipe
    must not break."""
    code = (
        "import sys;"
        f"print('{MARKER}: BootExplosion: the real reason', file=sys.stderr);"
        "sys.exit(1)"
    )
    h = SplitHarness(tmp_path, child_cmd=[sys.executable, "-c", code])
    try:
        exit_code = h.wait_exit(120.0)
        assert exit_code == 1, f"expected the bounded boot-loop exit 1, got {exit_code}"

        exits = [d for d in captured_dials if "phase=compute_process_exit" in d]
        assert exits, "no compute_process_exit dial was made"
        assert all("child_stderr_tail" in d and MARKER in d for d in exits), (
            "a pre-Hello death dial does not carry the child's stderr tail — "
            "the pod is undiagnosable again (pgw#833):\n" + exits[-1]
        )
        giveups = [d for d in captured_dials if "compute_boot_crash_loop" in d]
        assert giveups and any(MARKER in d for d in giveups), (
            "the give-up dial should name the child's last stderr"
        )
    finally:
        h.close()
    # (3) the tee: every byte still reaches the parent's stderr = container log.
    assert MARKER in capfd.readouterr().err


def test_boot_fatal_ack_round_trips_before_the_child_exits(
    tmp_path, captured_dials, captured_reports,
):
    """The pgw#826 follow-on race, closed deterministically: the child spies on
    the ack wait and exits 7 unless the ack actually arrived, so a green run
    PROVES the parent recorded the verdict before the child died — the reap
    can no longer race the socket buffer. RED on the pre-fix tree (no ack
    protocol exists: the spy target is missing, the child crashes untyped,
    and the parent misclassifies the death as a crash loop)."""
    script = tmp_path / "ack_probe_child.py"
    script.write_text(
        "import sys\n"
        "from gen_worker.procsplit import child as c\n"
        "orig = c._wait_boot_fatal_ack\n"
        "seen = {}\n"
        "def spy(sock):\n"
        "    orig(sock)\n"
        "    seen['ok'] = True\n"
        "c._wait_boot_fatal_ack = spy\n"
        "c.send_boot_fatal({'reason_class': 'cuda_unavailable',"
        " 'detail': 'ack race probe'})\n"
        "sys.exit(1 if seen.get('ok') else 7)\n"
    )
    h = SplitHarness(tmp_path, child_cmd=[sys.executable, str(script)])
    try:
        exit_code = h.wait_exit(120.0)
        assert exit_code == 1
        assert h.pc._spawn_count == 1, "a terminal verdict must never respawn"
        assert h.pc.terminal_exit_reason == "boot_fatal:cuda_unavailable"
        # The child exited 1, which per the script means the ack ARRIVED
        # before it exited (exit 7 = frame sent but never acknowledged).
        assert any('"cause": "exit:1"' in d for d in captured_dials), (
            "child exited without seeing the ack: " + repr(captured_dials)
        )
        assert len(captured_reports) == 1
        assert captured_reports[0].reason_class == "cuda_unavailable"
    finally:
        h.close()


# ---------------------------------------------------------------------------
# pgw#1349 / pgw#932: the stderr tail is captured (above) — now it is READ
# ---------------------------------------------------------------------------


def test_the_grpc_fork_abort_names_itself_instead_of_being_rediagnosed():
    """pgw#932 has been diagnosed from first principles at least five times,
    because ``cause=signal:SIGABRT`` names the symptom and nothing else. The
    facts needed to tell it apart were already on the dial — ``saw_hello``, the
    OOM delta, and pgw#833's stderr tail — so the discriminator that lived in
    the issue text now lives in the parent.

    RED-VERIFIABLE both ways, which is the point of the negative rows: a
    classifier that fires on every pre-Hello abort would relabel real defects
    as "known, rerun it", which is worse than the confusion it replaces."""
    from gen_worker.procsplit.parent import is_grpc_fork_abort

    sighting = (
        "I0803 03:09:46.724278 11992 fork_posix.cc:71] Other threads are "
        "currently calling into gRPC, skipping fork() handlers\n"
        "E0803 03:09:46.728102 12039 ev_epoll1_linux.cc:373] (event_engine) "
        "Epoll1Poller:0x278b97e0 encountered epoll_wait error: Bad file "
        "descriptor\n"
    )
    poller_only = (
        "ev_epoll1_linux.cc:373 (event_engine) Epoll1Poller encountered "
        "epoll_wait error: Bad file descriptor"
    )
    def abort(*, saw_hello: bool = False, oom_delta: int = 0,
              cause: str = "signal:SIGABRT", tail: str = "") -> bool:
        return is_grpc_fork_abort(cause=cause, saw_hello=saw_hello,
                                  oom_delta=oom_delta, stderr_tail=tail)

    assert abort(tail=sighting)
    # The 2026-08-17 master red carried only the poller half of the tail.
    assert abort(tail=poller_only)

    # ...and every neighbouring shape stays UNEXPLAINED, by name:
    assert not abort(tail=sighting, saw_hello=True), (
        "a post-Hello abort is the tenant's process dying, not the launcher's")
    assert not abort(tail=sighting, oom_delta=2), (
        "an OOM kill has an owner and a cost; it must never read as 'rerun it'")
    assert not abort(tail=sighting, cause="signal:SIGSEGV"), (
        "a SIGSEGV is the pgw#676 class, which is a real serving defect")
    assert not abort(tail="free(): invalid pointer")
    assert not abort(tail="")
