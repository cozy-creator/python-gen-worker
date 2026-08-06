"""pgw#975: the pgw#763 split's OOM victim order, declared instead of emergent.

Every assertion here is read back off a REAL process's ``/proc`` entry — never
from the value we passed in — because "we called the setter" is exactly the
claim that was true of nothing before this issue.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gen_worker import postmortem
from gen_worker.procsplit import oom_rank

from harness.hub_double import is_ready
from test_procsplit_pgw763 import (  # noqa: E402  (shared real-spawn harness)
    SplitHarness,
    captured_dials,  # noqa: F401  (fixture)
    isolated_postmortem,  # noqa: F401  (autouse fixture)
)

GIB = 1024 ** 3
linux_only = pytest.mark.skipif(
    sys.platform != "linux", reason="oom_score_adj is a Linux /proc interface"
)


def _read_oom_score_adj(pid: int) -> int:
    return int(Path(f"/proc/{pid}/oom_score_adj").read_text().strip())


@pytest.fixture()
def split(tmp_path, captured_dials):  # noqa: F811
    h = SplitHarness(tmp_path)
    try:
        yield h
    finally:
        h.close()


# --------------------------------------------------------------------------
# The property: a genuinely spawned compute child outranks its control parent
# --------------------------------------------------------------------------


@linux_only
def test_a_real_compute_child_outranks_the_control_parent(split):
    """RED before this issue: the child inherited the parent's value exactly, so
    the reporter's survival came down to which interpreter had imported more.
    The child is spawned by the REAL ParentControl through a real fork+exec; the
    number is read out of the kernel's own file, not from what we passed in."""
    split.scheduler.wait_connection(0).wait_for(is_ready)
    proc = split.pc._proc
    assert proc is not None, "no compute child was spawned"

    child_adj = _read_oom_score_adj(proc.pid)
    parent_adj = _read_oom_score_adj(os.getpid())

    delta = oom_rank.score_adj_delta_for_domain(
        oom_rank.oom_domain_bytes(), oom_rank.parent_ceiling_bytes()
    )
    assert child_adj == min(1000, parent_adj + delta), (
        f"child oom_score_adj={child_adj}, expected parent({parent_adj}) + "
        f"{delta} — read back off /proc/{proc.pid}/oom_score_adj"
    )
    assert child_adj > parent_adj, (
        f"the compute child ({child_adj}) does not outrank the control parent "
        f"({parent_adj}); a kernel OOM can take the reporter"
    )


@linux_only
def test_the_whole_compute_subtree_inherits_it(tmp_path):
    """The mint child (pgw#784) and the AOT pool's entry children are spawned
    BELOW a compute child and get no call of their own. That is only correct if
    the value survives fork and exec, so prove it on a real grandchild rather
    than asserting the man page."""
    child = tmp_path / "child.py"
    child.write_text(
        "import subprocess, sys\n"
        "from gen_worker.procsplit.oom_rank import raise_own_oom_score_adj\n"
        "rank = raise_own_oom_score_adj()\n"
        "assert rank.applied, rank.format()\n"
        "grand = subprocess.run(\n"
        "    [sys.executable, '-c',\n"
        "     'print(open(\"/proc/self/oom_score_adj\").read().strip())'],\n"
        "    capture_output=True, text=True, check=True)\n"
        "print(open('/proc/self/oom_score_adj').read().strip())\n"
        "print(grand.stdout.strip())\n"
    )
    out = subprocess.run(
        [sys.executable, str(child)],
        capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(
            [str(Path(__file__).resolve().parent.parent / "src"),
             os.environ.get("PYTHONPATH", "")])},
    )
    mine, grandchild = (int(v) for v in out.stdout.split())
    delta = oom_rank.score_adj_delta_for_domain(
        oom_rank.oom_domain_bytes(), oom_rank.parent_ceiling_bytes()
    )
    assert mine == min(1000, _read_oom_score_adj(os.getpid()) + delta)
    assert grandchild == mine, (
        "a grandchild did not inherit the rank — the mint child and every "
        "inductor entry child would be unranked"
    )


# --------------------------------------------------------------------------
# The value: derived, not tidy
# --------------------------------------------------------------------------


def test_the_parent_ceiling_still_matches_the_constants_it_was_derived_from():
    """`oom_rank` writes the reship buffer out literally so it can run before
    grpc exists in the child. This is the anti-drift guard for that copy — if
    either source constant moves, the ceiling is wrong and this goes red."""
    from gen_worker.procsplit.seam import CONTROL_FRAME_CEILING_BYTES
    from gen_worker.transport import RESHIP_WINDOW

    assert oom_rank._PARENT_BUFFER_BYTES == (
        RESHIP_WINDOW * CONTROL_FRAME_CEILING_BYTES
    )


@pytest.mark.parametrize(
    "domain_gib, expected, shape",
    [
        (755.07, 1, "RunPod CPU pod, measured live 2026-07-30"),
        (124.91, 2, "RunPod 4090 SECURE, measured live 2026-07-30"),
        (14.9, 15, "tightest cgroup cap observed (0.56.2 ram_total_gb report)"),
        (2.0, 110, "a hypothetical 2 GiB container"),
    ],
)
def test_the_value_is_derived_from_the_domain_not_picked(domain_gib, expected, shape):
    """§4.24: the number has to be re-derivable. Pinning the real shapes means a
    future edit that rounds this to 100 or 500 goes red and has to argue."""
    got = oom_rank.score_adj_delta_for_domain(
        int(domain_gib * GIB), oom_rank.parent_ceiling_bytes()
    )
    assert got == expected, f"{shape}: expected {expected}, got {got}"


def test_the_margin_always_covers_the_parents_whole_ceiling_twice():
    """The property the table is an instance of: one point is 0.1% of the
    domain, so `adj` points must be worth at least twice everything the control
    parent can hold. Checked across six orders of magnitude of domain."""
    ceiling = oom_rank.parent_ceiling_bytes()
    for domain in (1 * GIB, 8 * GIB, 15 * GIB, 64 * GIB, 125 * GIB, 755 * GIB):
        adj = oom_rank.score_adj_delta_for_domain(domain, ceiling)
        assert adj * (domain / 1000) >= 2 * ceiling, (
            f"domain={domain / GIB:.0f}GiB adj={adj} buys "
            f"{adj * domain / 1000 / 1024 ** 2:.0f}MiB against a "
            f"{ceiling / 1024 ** 2:.0f}MiB parent ceiling"
        )


def test_an_unreadable_domain_degrades_toward_protecting_the_reporter():
    """Guessing the roomiest host would silently produce adj=1 on a tight
    container. The fallback is the tightest domain we have ever run in."""
    tight = oom_rank.score_adj_delta_for_domain(0, oom_rank.parent_ceiling_bytes())
    assert tight == oom_rank.score_adj_delta_for_domain(
        oom_rank._TIGHTEST_OBSERVED_DOMAIN_BYTES, oom_rank.parent_ceiling_bytes()
    )
    assert tight > oom_rank.score_adj_delta_for_domain(755 * GIB,
                                                 oom_rank.parent_ceiling_bytes())


# --------------------------------------------------------------------------
# Failure is typed and loud
# --------------------------------------------------------------------------


@linux_only
def test_a_failed_set_is_a_named_degradation_never_a_silent_pass(
    monkeypatch, tmp_path, caplog,
):
    """A hardened container with a read-only /proc must not leave us believing
    the guarantee holds. Driven by a REAL unwritable path producing a real
    OSError, not a patched-out writer."""
    unwritable = tmp_path / "nonexistent-dir" / "oom_score_adj"
    monkeypatch.setattr(oom_rank, "_SELF_OOM_SCORE_ADJ", unwritable)

    with caplog.at_level(logging.ERROR, logger=oom_rank.__name__):
        rank = oom_rank.raise_own_oom_score_adj()

    assert not rank.applied
    assert rank.unprotected, "the degradation did not name what is unprotected"
    assert "control parent" in rank.unprotected
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert oom_rank.DEGRADE_PHASE in logged, "the failure was not logged typed"
    assert "errno" in rank.reason


@linux_only
def test_the_gap_is_cut_over_whatever_baseline_was_inherited(monkeypatch, tmp_path):
    """The bug a real spawn caught: `oom_score_adj` is INHERITED, so the value
    the child reads is the parent's own. An absolute set is a no-op whenever the
    ambient baseline already exceeds it — this box inherits 200 from
    `gnome-terminal` and an absolute 6 ranked parent and child identically."""
    fake = tmp_path / "oom_score_adj"
    fake.write_text("200\n")
    monkeypatch.setattr(oom_rank, "_SELF_OOM_SCORE_ADJ", fake)

    rank = oom_rank.raise_own_oom_score_adj()
    delta = oom_rank.score_adj_delta_for_domain(
        oom_rank.oom_domain_bytes(), oom_rank.parent_ceiling_bytes()
    )

    assert rank.applied
    assert int(fake.read_text()) == 200 + delta > 200
    assert rank.previous == 200


@linux_only
def test_a_baseline_already_at_the_kernel_maximum_is_reported_unprotected(
    monkeypatch, tmp_path, caplog,
):
    """At 1000 the parent is already maximally killable and no child can be
    ranked above it. Nothing can be done about that — but it must not be
    reported as a working control (§4.24 item 2)."""
    fake = tmp_path / "oom_score_adj"
    fake.write_text("1000\n")
    monkeypatch.setattr(oom_rank, "_SELF_OOM_SCORE_ADJ", fake)

    with caplog.at_level(logging.ERROR, logger=oom_rank.__name__):
        rank = oom_rank.raise_own_oom_score_adj()

    assert not rank.applied
    assert rank.reason == "baseline_at_kernel_maximum"
    assert rank.unprotected
    assert oom_rank.DEGRADE_PHASE in "\n".join(
        r.getMessage() for r in caplog.records
    )


# --------------------------------------------------------------------------
# memory.oom.group: the fact that decides whether any of this reports anything
# --------------------------------------------------------------------------


def test_the_container_facts_now_carry_memory_oom_group():
    """Read off this box's real cgroup chain. The VALUE here proves nothing
    about RunPod — the point is that the key now rides the existing
    `worker_fatal` dial and boot record, so a real pod answers it."""
    facts = postmortem.container_limits()
    assert "memory_oom_group" in facts
    assert facts["memory_oom_group"] in (0, 1, None)


def test_a_group_kill_is_called_out_in_the_death_dial():
    detail = postmortem.format_detail(
        phase="compute_process_exit",
        verdict={"exit_code": None, "signaled": True, "signal": 9,
                 "signal_name": "SIGKILL", "core_dumped": False},
        limits={"memory_oom_group": 1},
    )
    assert "memory.oom.group=1" in detail
    assert "GROUP KILL" in detail

    benign = postmortem.format_detail(
        phase="compute_process_exit",
        verdict={"exit_code": 1, "signaled": False},
        limits={"memory_oom_group": 0},
    )
    assert "memory.oom.group=0" in benign
    assert "GROUP KILL" not in benign


# --------------------------------------------------------------------------
# The production entrypoint really is where it runs
# --------------------------------------------------------------------------


def test_the_entrypoint_ranks_the_child_before_its_heavy_imports():
    """The harness enters one layer below `entrypoint`, so the placement itself
    is asserted on the source: a child ranked after `import torch` is unranked
    for the seconds pgw#833 measured it dying in."""
    src = (
        Path(__file__).resolve().parent.parent
        / "src" / "gen_worker" / "entrypoint.py"
    ).read_text()
    call = src.index("raise_own_oom_score_adj()")
    assert src.index("is_compute_child") < call
    for heavy in ("from .worker import Worker", "import msgspec"):
        assert call < src.index(heavy), (
            f"the OOM rank is declared after {heavy!r}"
        )
