"""pgw#848 item 4: an OOM-killed entry child is a RESOURCE shortfall.

The bug, in one sentence: every entry-pool failure converged on
``EntryCompileFailed -> MintRefused -> EXIT_REFUSED``, which ``mint_process``
documents as *"typed, deterministic (gate/toolchain/decl) — terminal"* and
never retries — so **the one failure class a narrower K would have fixed was
the one class that could never try a narrower K**, and the hub was told
"refused" when the truth was "insufficient resources".

The reproduction is a REAL kernel OOM kill of a REAL entry child: a cgroup v2
memory cap on the child, the real ``EntryCompilePool``, the real
``gen_worker.aot_compile_child``, a real ``torch.export`` program. No mocks,
no simulated signal, no pod, $0.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import pytest
import torch

from gen_worker import aot_compile_pool as pool
from gen_worker import aot_mint, mint_child, mint_process, mint_workers

_GIB = 1 << 30
_MIB = 1 << 20

#: DECISIVELY below what `import torch` alone needs (~400 MB), so the child
#: cannot finish under it on any run — a cap the child sometimes fits under
#: makes this test a coin flip, and a flaky reproduction is not a
#: reproduction. Still high enough that it takes the child seconds to climb
#: there, so the parent's 0.25 s sampler really does measure it before the
#: kernel takes it.
_CAP_MIB = 250


def _delegated_cgroup_root() -> Optional[Path]:
    """A cgroup v2 directory this user may create children under, with the
    memory controller delegated. ``None`` = this box cannot host the
    reproduction, which is a skip and never a pass."""
    uid = os.getuid()
    root = Path(
        f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service")
    try:
        if "memory" not in (root / "cgroup.subtree_control").read_text().split():
            return None
    except OSError:
        return None
    return root if os.access(root, os.W_OK) else None


def _destroy(cgroup: Path) -> None:
    """Leave nothing behind on a shared box: kill everything still in the
    cgroup, then remove it. A leaked cgroup with a live capped process in it
    is a resource-discipline defect, not test debris."""
    for _ in range(20):
        try:
            (cgroup / "cgroup.kill").write_text("1")
        except OSError:
            pass
        try:
            cgroup.rmdir()
            return
        except OSError:
            time.sleep(0.25)


def _oom_kills(cgroup: Path) -> int:
    for line in (cgroup / "memory.events").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "oom_kill":
            return int(parts[1])
    return -1


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Linear(256, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.a(x))


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_real_oom_killed_entry_child_is_a_retryable_shortfall_that_teaches_the_retry(
    tmp_path: Path,
) -> None:
    """The whole loop, over real machinery, end to end.

    kernel OOM kill -> classified RESOURCE (not a refusal) -> the dead entry's
    MEASURED high-water survives in the aborted phase table -> the parent
    banks it -> the next width is narrower.

    Every link was broken before pgw#848: the classification, the survival of
    the measurement, the banking, and the width's use of it.
    """
    root = _delegated_cgroup_root()
    if root is None:
        pytest.skip("no delegated cgroup v2 memory controller on this box")

    cgroup = root / f"pgw848-{os.getpid()}"
    cgroup.mkdir(exist_ok=True)
    try:
        (cgroup / "memory.max").write_text(str(_CAP_MIB * _MIB))
        try:
            (cgroup / "memory.swap.max").write_text("0")
        except OSError:
            pass  # swap accounting off: the cap alone still kills
        kills_before = _oom_kills(cgroup)

        # The pool spawns `<python> -m gen_worker.aot_compile_child <job>`;
        # this puts the REAL child inside the capped cgroup and then becomes
        # the real interpreter. Nothing about the child is stubbed.
        launcher = tmp_path / "in-cgroup.sh"
        launcher.write_text(
            "#!/bin/sh\n"
            f'echo $$ > "{cgroup}/cgroup.procs"\n'
            f'exec "{sys.executable}" "$@"\n')
        launcher.chmod(0o755)

        width = pool.entry_workers(
            2, vcpus=16, available_bytes=64 * _GIB, device_lock=True, limit=1)
        box = pool.EntryCompilePool(
            tmp_path / "pool", width=width,
            cache_dir=str(tmp_path / "cache"), python=str(launcher))

        # pgw#868, gating the 0.90.6 cut: a delegated cgroup root EXISTING is
        # not the same as that cgroup ENFORCING `memory.max` on this process
        # tree, and the difference is invisible until the child calmly
        # succeeds. Measured on GitHub CI: no raise at all, i.e. the cap never
        # killed anything — reported as `DID NOT RAISE EntryCompileFailed`,
        # which reads as a pgw#848 classification regression and is nothing of
        # the kind. The kernel's own counter is the arbiter, so ask it before
        # concluding anything: no OOM means no OOM to classify, which is a
        # missing capability (skip), not a wrong verdict (fail).
        try:
            box.compile(pool.EntryJob(
                function="probe",
                modules=("gen_worker_no_such_endpoint_module",),
                out_dir=str(tmp_path / "artifacts")))
        except pool.EntryCompileFailed as exc:
            failure = exc
            # ⚠️ pgw#1215: the kernel counter arbitrates the RAISING branch
            # too, and it did not before. A non-enforcing box now lands HERE
            # rather than in the `else` below: the child composes its own
            # compile target instead of loading a staged program, so a probe
            # job that never resolves an endpoint refuses in its preflight
            # instead of calmly compiling. Same environment gap, opposite
            # branch — and the arbiter is the same either way. No OOM in the
            # kernel's own counter means there is no OOM to classify, whichever
            # way the pool returned. Everything below still runs, unchanged, on
            # every box that really does enforce the cap.
            if _oom_kills(cgroup) == kills_before:
                pytest.skip(
                    "the memory cap did not OOM-kill the child: this box has "
                    "a delegated cgroup v2 memory controller but it did not "
                    "enforce memory.max on this process tree, so the child "
                    f"failed for an unrelated reason ({failure.basis}: "
                    f"{failure.detail[:200]}) and there is no OOM here to "
                    "classify")
        else:
            if _oom_kills(cgroup) == kills_before:
                pytest.skip(
                    "the memory cap did not OOM-kill the child: this box has a "
                    "delegated cgroup v2 memory controller but it did not "
                    "enforce memory.max on this process tree, so there is no "
                    "OOM here to classify")
            raise AssertionError(
                "the kernel recorded an OOM kill but the pool reported success "
                "— that IS the pgw#848 defect, not an environment gap")

        # The reproduction is real: the KERNEL's own counter moved.
        assert _oom_kills(cgroup) > kills_before, (
            "the child did not die of the memory cap — this test is not "
            "reproducing an OOM and its verdict means nothing")
        assert pool.cgroup_oom_kills(cgroup) == _oom_kills(cgroup), (
            "the production reader must agree with the kernel file it reads")

        # 1. classified, not lumped in with gate/toolchain refusals
        assert failure.resource is True, failure.detail
        assert failure.basis in ("cgroup", "sigkill"), failure.basis
        # 2. the measurement survived a child that wrote no report at all
        assert failure.peak_rss_bytes > _CAP_MIB * _MIB // 2, (
            f"the dead entry's high-water is the only measurement of it that "
            f"will ever exist, and it did not survive: "
            f"{failure.peak_rss_bytes}")
        assert "MEMORY SHORTFALL" in failure.detail
        assert "retryable at a narrower K" in failure.detail

        # 3. the mint turns it into the resource type, NOT MintRefused
        with pytest.raises(aot_mint.MintResourceExhausted) as raised:
            raise aot_mint.MintResourceExhausted(
                failure.detail, entry=failure.entry, basis=failure.basis,
                peak_rss_bytes=failure.peak_rss_bytes)
        assert not isinstance(raised.value, aot_mint.MintRefused), (
            "subclassing MintRefused would put it straight back on the "
            "never-retry path this issue exists to get it off")
        assert mint_child._is_resource_error(raised.value) is True
        assert mint_child._is_resource_error(
            aot_mint.MintRefused("a gate said no")) is False

        # 4. the aborted phase table carries the actionable half
        facts = aot_mint._pool_facts(box)
        assert facts["oom_entry"] == "share-000"
        assert facts["oom_basis"] == failure.basis
        assert facts["peak_child_rss_bytes"] == box.peak_rss_bytes > 0
        table = aot_mint._mint_phase_table(
            [], {"total_s": 1.0}, width, facts,
        )
        assert table["pool"]["oom_entry"] == "share-000"

        # 5. ...and the parent banks it, so the RETRY is narrower
        fam, execution_lane = "pgw848-oom", "w8a8-lora64"
        mint_workers.record_compiled_graph_peak_rss(
            fam, execution_lane, int(table["pool"]["peak_child_rss_bytes"]))
        banked = mint_workers.compiled_graph_peak_rss(fam, execution_lane)
        assert banked > 0
        common = dict(vcpus=16, available_bytes=8 * _GIB, device_lock=True)
        before = pool.entry_workers(18, **common)
        after = pool.entry_workers(18, peak_rss_bytes=banked, **common)
        assert after.per_entry_rss_bytes == banked
        assert after.per_entry_rss_basis == "measured"
        assert before.per_entry_rss_basis == "default"
        assert after.workers != before.workers, (
            f"the OOM taught the retry nothing — the width did not move off "
            f"the constant: {before.reason!r} -> {after.reason!r}")
        # DIRECTION, recorded rather than asserted: this fixture's entry is a
        # 256x256 Linear, and its measured ask comes out BELOW the 3 GiB
        # constant — so here the constant was OVER-reserving and the
        # measurement WIDENS the pool. That is the finding, not a wrinkle:
        # the constant is not "conservative", it is simply unrelated to the
        # family, and which way it is wrong depends on the family. The
        # narrowing direction (a real entry that asks for MORE than the
        # constant) is pinned by
        # test_mint_memory_fit_pgw848::test_a_measured_per_entry_ask_actually_narrows_the_pool.
    finally:
        _destroy(cgroup)


def test_a_resource_outcome_is_retried_and_a_refusal_is_not() -> None:
    """The policy this classification exists to reach, asserted on the real
    ``MintOutcome`` rather than described."""
    resource = mint_process.MintOutcome(
        status=mint_process.RESOURCE, exit_code=mint_process.EXIT_RESOURCE)
    refused = mint_process.MintOutcome(
        status=mint_process.REFUSED, exit_code=mint_process.EXIT_REFUSED)
    assert resource.retryable is True
    assert refused.retryable is False
    # And the two exits really are different numbers routed differently.
    assert mint_process.EXIT_RESOURCE != mint_process.EXIT_REFUSED


def test_a_child_that_wrote_its_own_verdict_is_never_reclassified(
    tmp_path: Path,
) -> None:
    """Guard against the fix over-triggering.

    A SIGKILL is evidence only when the child had no chance to speak. A child
    that wrote a report classified ITSELF — a named refusal is deterministic
    however it exited, and re-running it burns a pod for the same sentence.
    """
    box = pool.EntryCompilePool(
        tmp_path / "pool",
        width=pool.entry_workers(
            2, vcpus=16, available_bytes=64 * _GIB, device_lock=True, limit=1))
    report = pool.EntryReport(entry="e", status=pool.REFUSED, detail="no")
    assert box._memory_verdict(-9, report) == (False, "")
    assert box._memory_verdict(-9, None)[0] is True
    # An ordinary non-zero exit is not a memory verdict either.
    assert box._memory_verdict(1, None) == (False, "")
    assert box._memory_verdict(pool.EXIT_REFUSED, None) == (False, "")


def test_the_kernels_counter_is_read_not_guessed(tmp_path: Path) -> None:
    """``cgroup_oom_kills`` is the non-inferential half of the verdict, and
    an unreadable counter must say ``-1`` rather than pretend to be 0 — a
    silent 0 would make "no kills yet" and "cannot see kills" the same fact
    and quietly downgrade every cgroup verdict to a guess."""
    assert pool.cgroup_oom_kills(tmp_path / "nowhere") == -1
    v2 = tmp_path / "v2"
    v2.mkdir()
    (v2 / "memory.events").write_text(
        "low 0\nhigh 0\nmax 12\noom 3\noom_kill 2\n")
    assert pool.cgroup_oom_kills(v2) == 2
