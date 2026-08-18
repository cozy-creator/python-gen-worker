"""K must be a FIT against a measured high-water, not a constant.

``aot_compile_pool.entry_workers`` bounds K on three readings, one of which is
``peak_rss_bytes`` — one entry child's host high-water. Four things must hold
for that argument to carry a real measurement:

1. the entry child measures ``RUSAGE_CHILDREN``, not ``RUSAGE_SELF``, which
   cannot see ``cc1plus``;
2. the pool's live sampler walks ``/proc/<pid>/children`` RECURSIVELY —
   ``cc1plus`` is at depth 2;
3. the pool's own ``peak_child_rss_bytes`` is banked;
4. ``aot_mint`` passes ``peak_rss_bytes`` to ``entry_workers``; without it
   ``mem_workers`` divides available RAM by a 3 GiB constant and
   ``per_entry_rss_basis`` reads ``"default"`` forever. This is the ONLY
   per-entry footprint K divides by.

Why (1) and (2) matter — measured on the real sdxl AOTI wrapper TU (6,324,290
bytes, production flags, g++ 13.3)::

    ground truth, whole process tree          2.052 GiB
      of which ONE process, cc1plus           2.049 GiB
    RUSAGE_CHILDREN                           2.045 GiB
    RUSAGE_SELF          (instrument 1)       0.012 GiB   <- 171x low
    one-level /proc walk (instrument 2)       0.015 GiB   <- 133x low

``test_a_measured_per_entry_ask_actually_narrows_the_pool`` is green either
way: ``entry_workers(peak_rss_bytes=...)`` always did the right thing with a
measurement; the defect was that no caller handed it one.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

from gen_worker import child_contract
from gen_worker import aot_compile_pool as pool
from gen_worker import mint_workers

from harness import progress_wait

_GIB = 1 << 30
_MIB = 1 << 20


# ---------------------------------------------------------------------------
# (1) + (2): the instruments
# ---------------------------------------------------------------------------


_ALLOC = """
import sys, time
from pathlib import Path
n = int(sys.argv[1])
buf = bytearray(n)
buf[::4096] = b'x' * len(range(0, len(buf), 4096))
Path(sys.argv[2]).write_text("up")
time.sleep(float(sys.argv[3]))
"""

_SPAWN = """
import subprocess, sys, time
subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2], sys.argv[3],
                  sys.argv[4]])
time.sleep(float(sys.argv[4]) + 5)
"""


def _scripts(tmp_path: Path) -> tuple[Path, Path]:
    alloc = tmp_path / "alloc.py"
    alloc.write_text(_ALLOC)
    spawn = tmp_path / "spawn.py"
    spawn.write_text(_SPAWN)
    return alloc, spawn


def test_the_pools_sampler_sees_the_process_that_holds_the_memory(
    tmp_path: Path,
) -> None:
    """The allocation that bounds K is a GRANDCHILD of the sampled process.

    Not a hypothetical: on a real ``aoti_compile_and_package`` the entry
    child's direct children are ``g++`` (a driver that allocates nothing) and
    inductor's ``async_compile`` workers, while ``cc1plus`` — the 2.04 GiB —
    sits at depth 2 alongside ``as``/``collect2``, and ``ld`` at depth 3.
    A one-level reading is therefore blind to the only number it exists for.
    """
    alloc_py, spawn_py = _scripts(tmp_path)
    marker = tmp_path / "up"
    alloc = 320 * _MIB
    # sh (proc.pid) -> python spawn.py (depth 1) -> python alloc.py (depth 2):
    # the topology a real entry compile has, where the memory is at depth 2.
    proc = subprocess.Popen(
        ["/bin/sh", "-c",
         f"{sys.executable} {spawn_py} {alloc_py} {alloc} {marker} 20 & wait"],
        start_new_session=True)
    try:
        # bounded on PROGRESS, never on a clock. The advance
        # is the grandchild's RSS climbing — which is the very quantity under
        # test — and `gone` ends the wait immediately and definitively if the
        # fixture dies, with no duration involved.
        progress_wait.await_progress(
            lambda: (marker.exists(), pool._peak_rss_bytes(proc)),
            lambda seen: bool(seen[0]),
            what="the grandchild to finish allocating",
            cadence=progress_wait.Cadence(),
            gone=lambda: (
                f"the fixture exited {proc.returncode} before allocating"
                if proc.poll() is not None else None),
            render=lambda seen: (
                f"marker={seen[0]} tree_rss={seen[1] / _MIB:.0f}MiB"),
        )

        depth1 = [proc.pid] + [
            int(p) for p in Path(
                f"/proc/{proc.pid}/task/{proc.pid}/children"
            ).read_text().split()]
        transitive = pool._descendants(proc.pid)
        assert len(transitive) > len(depth1), (
            f"the tree walk found nothing the one-level read missed "
            f"(depth1={depth1} transitive={transitive}) — the fixture is not "
            f"reproducing the topology under test")

        one_level = sum(pool._vmhwm_bytes(pid) for pid in depth1)
        whole_tree = pool._peak_rss_bytes(proc)
        assert whole_tree - one_level > alloc * 0.8, (
            f"the sampler must see the grandchild's {alloc / _MIB:.0f} MiB: "
            f"one-level={one_level / _MIB:.0f} MiB "
            f"whole-tree={whole_tree / _MIB:.0f} MiB")
    finally:
        proc.kill()
        proc.wait(timeout=10)
        for pid in pool._descendants(proc.pid):
            try:
                os.kill(pid, 9)
            except OSError:
                pass


_ENTRY_PROBE = """
import json, resource, subprocess, sys
from gen_worker import aot_compile_child
before = aot_compile_child._peak_rss()
subprocess.run([sys.executable, sys.argv[1], sys.argv[2], sys.argv[3], "0"],
               check=True)
print(json.dumps({
    "before": before,
    "after": aot_compile_child._peak_rss(),
    "self_only": int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
}))
"""


def test_an_entry_childs_reported_peak_includes_the_compiler_it_ran(
    tmp_path: Path,
) -> None:
    """``aot_compile_child._peak_rss`` is the number banked as the per-entry
    ask. A reading that excludes the compile is not a reading of the entry.

    Run in a FRESH interpreter: ``RUSAGE_CHILDREN`` is a process-lifetime
    high-water, so a pytest session that has already forked something large
    would mask the very thing under test.
    """
    alloc_py, _ = _scripts(tmp_path)
    probe = tmp_path / "probe.py"
    probe.write_text(_ENTRY_PROBE)
    alloc = 512 * _MIB
    out = subprocess.run(
        [sys.executable, str(probe), str(alloc_py), str(alloc),
         str(tmp_path / "up2")],
        check=True, capture_output=True, text=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(filter(None, (
                str(Path(pool.PACKAGE_ROOT)), os.environ.get("PYTHONPATH", ""),
            ))),
        })
    got = __import__("json").loads(out.stdout.strip().splitlines()[-1])

    assert got["after"] - got["before"] > alloc * 0.8, (
        f"the child's own peak did not move when a {alloc / _MIB:.0f} MiB "
        f"compile ran under it: {got}")
    assert got["after"] > got["self_only"], (
        f"RUSAGE_SELF alone is the pre-pgw#848 reading and it is blind "
        f"here: {got}")


# ---------------------------------------------------------------------------
# (3) + (4): the feedback loop
# ---------------------------------------------------------------------------


def test_a_measured_per_entry_ask_actually_narrows_the_pool() -> None:
    """The width must MOVE on the measurement, and must say that it did.

    Same host, same card, same entry count. The only difference is whether the
    caller passed the pod's own measured per-entry high-water.
    """
    common: Dict[str, Any] = dict(
        vcpus=32, available_bytes=32 * _GIB, device_lock=True)
    guessed = pool.entry_workers(18, **common)
    measured = pool.entry_workers(
        18, peak_rss_bytes=7 * _GIB, **common)

    assert guessed.per_entry_rss_basis == "default"
    assert measured.per_entry_rss_basis == "measured"
    # (32 - 4 reserve) / 3 GiB default = 9 ; / 7 GiB measured = 4
    assert guessed.mem_workers == 9, guessed.reason
    assert measured.mem_workers == 4, measured.reason
    assert measured.workers < guessed.workers, (
        f"a pod whose entries really need 7 GiB apiece must not run the width "
        f"a 3 GiB guess licensed: {guessed.reason!r} vs {measured.reason!r}")
    assert measured.binding == "host-memory", measured.binding


def test_the_measured_ask_is_banked_and_survives_to_the_next_mint() -> None:
    """The bank is monotone and keyed per (family, lane). pgw#1175: its three
    device twins are deleted; this is the last one, and the one K needs."""
    fam, execution_lane = "pgw848-fam", "w8a8-lora64"
    assert mint_workers.compiled_graph_peak_rss(fam, execution_lane) == 0
    mint_workers.record_compiled_graph_peak_rss(fam, execution_lane, 5 * _GIB)
    assert mint_workers.compiled_graph_peak_rss(fam, execution_lane) == 5 * _GIB
    # A luckier run must not talk the ask down.
    mint_workers.record_compiled_graph_peak_rss(fam, execution_lane, 2 * _GIB)
    assert mint_workers.compiled_graph_peak_rss(fam, execution_lane) == 5 * _GIB
    mint_workers.record_compiled_graph_peak_rss(fam, execution_lane, 9 * _GIB)
    assert mint_workers.compiled_graph_peak_rss(fam, execution_lane) == 9 * _GIB
    # Keyed, not global.
    assert mint_workers.compiled_graph_peak_rss(fam, "plain") == 0


def test_the_pools_own_measurement_reaches_the_bank_over_the_real_relay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """END TO END over the path that was dark.

    The pool has published ``peak_child_rss_bytes`` in its phase table since
    pgw#830 and the parent has relayed that table since pgw#842 — and nothing
    ever read the field. This drives the REAL ``build_compiled_graph`` bookkeeping
    against a real ``MintOutcome`` and asserts the ask lands in the bank.
    """
    from gen_worker import mint_process
    from gen_worker import aot_mint

    fam, execution_lane = "pgw848-relay", "w8a8-lora64"
    width = pool.entry_workers(
        2, vcpus=16, available_bytes=64 * _GIB, device_lock=True, limit=2)
    table = aot_mint._mint_phase_table(
        {}, {"total_s": 1.0}, width,
        {"peak_child_rss_bytes": 6 * _GIB, "pool_workers": 2},
    )
    assert table["pool"]["peak_child_rss_bytes"] == 6 * _GIB, (
        "the pool's measurement must be IN the table the parent relays")

    outcome = mint_process.MintOutcome(
        status=mint_process.MINTED, elapsed_s=1.0,
        report=mint_process.MintReport(
            status=mint_process.MINTED, elapsed_s=1.0, mint_phases=table))

    assert mint_workers.compiled_graph_peak_rss(fam, execution_lane) == 0
    # The exact statement `build_compiled_graph` runs, against the real structures.
    pool_block = (outcome.report.mint_phases or {}).get("pool")
    mint_workers.record_compiled_graph_peak_rss(
        fam, execution_lane, int((pool_block or {}).get("peak_child_rss_bytes") or 0))
    assert mint_workers.compiled_graph_peak_rss(fam, execution_lane) == 6 * _GIB

    # ...and the request the NEXT attempt builds carries it to the child.
    banked = mint_workers.compiled_graph_peak_rss(fam, execution_lane)
    assert banked == 6 * _GIB
    request = mint_process.MintRequest(
        function="f", modules=(), family=fam, arm_token="k", target="t",
        work_root="c", report="r",
        cfg=child_contract.CompileSpec(),
        compiled_graph_peak_rss_bytes=banked)
    assert request.compiled_graph_peak_rss_bytes == 6 * _GIB, (
        "a banked measurement that cannot cross the process boundary is a "
        "measurement the width will never see — the width is computed in the "
        "child, whose memory dies with it")


def test_the_harness_import_cannot_depend_on_collection_order() -> None:
    """pgw#848: this file's `from harness import progress_wait` errored at
    COLLECTION under `-n 4` while passing 5/5 standalone.

    Mechanism, reproduced: pytest's `prepend` import mode puts a test file's
    rootdir on `sys.path` as it imports that file, so whichever module in
    `tests/` is imported FIRST is what makes `harness` importable for the ~15
    modules that assume it. Nobody declared that dependency and nothing
    enforced it. Exec'ing this module with `src/` on the path but not `tests/`
    raises `ModuleNotFoundError: No module named 'harness'` — and an import
    error at collection fails a whole run, not a test, which is why a release
    cut hit it and a standalone run did not.

    Fixed in `tests/conftest.py`, which pytest guarantees to import before any
    test module in this directory, in every mode including each xdist worker.
    Asserted here rather than trusted: a guarantee nothing checks is the
    ordering assumption again, one level up.
    """
    import sys

    conftest = Path(__file__).resolve().parent
    assert str(conftest) in sys.path, (
        f"{conftest} is not on sys.path — every `from harness import ...` in "
        f"this suite is back to depending on which module pytest happened to "
        f"import first")
    assert progress_wait.await_progress is not None
