"""pgw#868 A4: exporting a row OUT OF PROCESS must produce the same artifact.

The whole parallel-export scheme rests on one claim: a row exported by a
worker, from its own module copy in a fresh interpreter, compiles to
byte-identical files against the same row exported in the parent. Proven here
rather than assumed — pgw#846 governs, and this lane has twice found an
"obviously inert" difference reaching the artifact (the row in node ARGUMENTS,
the device in node META).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from torch import nn

from gen_worker import aot_export_parallel, aot_mint, aot_wrapper_split, host_isa

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

WORKER = textwrap.dedent("""
    import sys, torch
    sys.path.insert(0, sys.argv[1])
    from gen_worker import host_isa
    host_isa.impose()
    torch.manual_seed(0)
    from torch import nn
    m = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1), nn.SiLU(),
                      nn.Conv2d(8, 4, 3, padding=1)).eval()
    h, w = int(sys.argv[3]), int(sys.argv[4])
    with torch.no_grad():
        ep = torch.export.export(m, (torch.randn(1, 4, h, w),), strict=False)
    torch.export.save(ep, sys.argv[2])
""")


def _module():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(4, 8, 3, padding=1), nn.SiLU(),
                         nn.Conv2d(8, 4, 3, padding=1)).eval()


def _digests(program, tag, cache: Path):
    """Compile in the SAME cleared build dir — different dirs can never be
    byte-equal (the dir and expanded -march are embedded in the object)."""
    shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    import torch._inductor.codecache as codecache
    for n in ("cache_dir", "default_cache_dir"):
        f = getattr(codecache, n, None)
        if getattr(f, "cache_clear", None):
            f.cache_clear()
    aot_wrapper_split.install()
    out = {}
    for handle in aot_mint.compile_entry_files(program, tag):
        p = Path(str(handle))
        if p.is_file() and p.suffix == ".cpp":
            body = p.read_text().split("// Compile cmd")[0].replace(
                str(cache), "<c>")
            out["".join(p.suffixes[-2:])] = hashlib.sha256(
                body.encode()).hexdigest()[:16]
    return out


def test_an_out_of_process_export_is_byte_identical(tmp_path):
    host_isa.impose()
    src = str(Path(aot_mint.__file__).resolve().parents[1])
    worker_py = tmp_path / "w.py"
    worker_py.write_text(WORKER)
    out = tmp_path / "worker.pt2"

    rc = subprocess.run(
        [sys.executable, str(worker_py), src, str(out), "24", "32"],
        capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr[-2000:]

    worker_program = torch.export.load(str(out))
    with torch.no_grad():
        parent_program = torch.export.export(
            _module(), (torch.randn(1, 4, 24, 32),), strict=False)

    assert worker_program.graph_module.code == parent_program.graph_module.code

    cache = tmp_path / "build"
    parent = _digests(parent_program, "parent", cache)
    worker = _digests(worker_program, "worker", cache)
    assert parent and worker
    assert parent == worker, f"parent={parent} worker={worker}"


def test_groups_split_only_at_an_arm_change():
    rows = [("p", True)] * 3 + [("p", False)] * 4
    assert aot_export_parallel.groups(rows) == [[0, 1, 2], [3, 4, 5, 6]]
    assert aot_export_parallel.groups([("p", True)]) == [[0]]
    # an alternating declaration must NOT be merged across the mutation
    alt = [("p", True), ("p", False), ("p", True)]
    assert aot_export_parallel.groups(alt) == [[0], [1], [2]]


def test_width_refuses_to_guess_an_unmeasured_footprint():
    """The failure mode of guessing is an OOM that kills a 65-minute phase."""
    g = 18
    for kw in ({"per_export_device_bytes": 0}, {"budget_bytes": 0}):
        base = {"budget_bytes": 40 << 30,
                "per_export_device_bytes": 5 << 30, "cpu_workers": 32}
        base.update(kw)
        assert aot_export_parallel.width_for(g, **base)["workers"] == 1

    w = aot_export_parallel.width_for(
        g, budget_bytes=40 << 30, per_export_device_bytes=5 << 30,
        cpu_workers=32)
    assert w["workers"] == 8 and w["binding"] == "ceiling"
    # the EXPORT footprint, not the compile pool's 11.07 GiB estimate
    w = aot_export_parallel.width_for(
        g, budget_bytes=27 << 30, per_export_device_bytes=5 << 30,
        cpu_workers=32, ceiling=8)
    assert w["workers"] == 5 and w["binding"] == "vram"
    assert aot_export_parallel.width_for(
        2, budget_bytes=40 << 30, per_export_device_bytes=1 << 30,
        cpu_workers=32)["workers"] == 1


def test_the_budget_is_pgw992s_card_bound_never_a_free_sample(monkeypatch):
    """pgw#1000: export workers are priced exactly as entry children are.

    The whole reason pgw#992 exists is that a momentary free-VRAM sample does
    not bound what K children hold at their simultaneous peaks. Export has no
    claim to a weaker rule — same card, same residents, same mint.

    The property that makes the budget REAL rather than a restatement of
    `free`: the census is taken before the export phase, and the own-footprint
    term is this process's high-water read after it. `total - co-tenant - own`
    collapses to exactly `free` when those two coincide, so a census taken at
    decision time would price nothing at all. This asserts the gap.
    """
    from gen_worker import aot_compile_pool

    total, free, own_at_open = 80 << 30, 30 << 30, 10 << 30   # 40 GiB co-tenant
    census = aot_compile_pool.CardCensus(total, free, own_at_open, "sampled")
    assert census.resident_other_bytes == 40 << 30

    # The mint child grew to 26 GiB tracing 36 rows — the pgw#992 threat,
    # measured on the real pod as 5.36 GiB at open and 16.20 GiB at the OOM.
    monkeypatch.setattr(
        aot_compile_pool, "own_device_high_water", lambda device=-1: 26 << 30)
    budget = aot_export_parallel._card_budget(census)
    assert budget == total - (40 << 30) - (26 << 30) == 14 << 30
    assert budget < free, (
        "the card-wide budget must be TIGHTER than the free sample — that gap "
        "IS the resident growth a free reading cannot see")

    # Coincident readings collapse to `free`, which is the honest floor of
    # this rule and the reason the census must be taken early.
    monkeypatch.setattr(
        aot_compile_pool, "own_device_high_water", lambda device=-1: own_at_open)
    assert aot_export_parallel._card_budget(census) == free

    unreadable = aot_compile_pool.CardCensus(0, 0, 0, "absent")
    assert aot_export_parallel._card_budget(unreadable) == -1
    assert aot_export_parallel.width_for(
        18, budget_bytes=-1, per_export_device_bytes=5 << 30,
        cpu_workers=32)["binding"] == "no-card-budget"


def test_the_census_is_taken_before_the_export_phase_not_at_the_decision():
    """The ordering the budget depends on, asserted at the call site."""
    import inspect

    from gen_worker import aot_mint

    src = inspect.getsource(aot_mint._mint_cell)
    opened = src.index("_ExportFootprint.open()")
    decided = src.index("aot_export_parallel.decide(")
    assert opened < decided, (
        "the census must predate the rows whose growth it prices")
    assert "census=export_footprint.census" in src


def test_there_is_no_env_gate_left_to_turn_this_on():
    """pgw#1000: the decision lives in code. Width 1 is the natural floor when
    the budget says so — not a flag, and not a module that cannot be reached.

    The gate was justified as "the export footprint is unmeasured". It was not
    unmeasured; it was measured WRONG (the phase high-water, not one row's
    delta), so `width_for` could only ever return 1 and the flag guarded a
    computation with one possible answer.
    """
    import inspect

    src = inspect.getsource(aot_export_parallel)
    assert "ENV_FLAG" not in src
    assert "environ" not in src and "getenv" not in src
    assert not hasattr(aot_export_parallel, "enabled")


def test_the_CALL_SITE_exists_and_reads_the_shipped_measurement():
    """pgw#868 A4 / pgw#1000: the connection, proven as an OBSERVABLE.

    Every instance of this program's signature defect passed its own unit
    tests — a module with a correct API that nothing imports. So this asserts
    the JOIN: `aot_mint` imports the module, and `decide()` consumes exactly
    the key `aot_mint` ships and produces a width decision from it.

    The key CHANGED in pgw#1000, and that is the fix: `export_peak_device_bytes`
    is the phase high-water (15.4 GiB on the only complete sdxl mint) and sized
    nothing but a width of 1; `per_export_device_bytes` is one row's delta over
    the resident baseline, which is what a worker actually adds to the card.
    """
    import inspect

    from gen_worker import aot_mint

    # 1. aot_mint actually imports it, and actually calls it
    assert aot_mint.aot_export_parallel is aot_export_parallel
    src = inspect.getsource(aot_mint)
    assert "aot_export_parallel.decide(" in src
    # ...and it ships the key `decide` reads, from the per-row instrument.
    assert "per_export_device_bytes" in inspect.getsource(
        aot_mint._ExportFootprint)
    assert "export_footprint.facts()" in src

    # 2. the consumer reads the producer's key, and the width MOVES with it
    rows = [("p", True)] * 18 + [("p", False)] * 18
    unmeasured = aot_export_parallel.decide(
        rows, {}, budget_bytes=40 << 30, cpu_workers=32)
    assert unmeasured["export_parallel_width"] == 1.0
    assert unmeasured["export_parallel_binding"] == 5.0   # unmeasured

    measured = aot_export_parallel.decide(
        rows, {"per_export_device_bytes": float(5 << 30)},
        budget_bytes=40 << 30, cpu_workers=32)
    assert measured["export_parallel_width"] == 8.0
    assert measured["export_parallel_groups"] == 2.0
    assert measured["export_parallel_largest_group"] == 18.0
    assert measured["export_parallel_per_export_bytes"] == float(5 << 30)
    assert measured["export_parallel_budget_bytes"] == float(40 << 30)

    # 3. the OLD key no longer moves the width — the wrong number is not
    #    merely deprecated, it is disconnected.
    stale = aot_export_parallel.decide(
        rows, {"export_peak_device_bytes": float(5 << 30)},
        budget_bytes=40 << 30, cpu_workers=32)
    assert stale["export_parallel_width"] == 1.0


def test_the_phase_highwater_and_one_rows_delta_are_different_facts():
    """pgw#1000, stated as the arithmetic that kept this dark for 3 releases.

    On attempt 26's real sdxl mint the phase high-water was 15.4 GiB. Dividing
    a card-wide budget by THAT asks "how many whole mint children fit", which
    is 1 on every card the fleet rents. One row's delta is a different and much
    smaller quantity, and it is the one a pool is sized by.
    """
    budget = 18 << 30                      # a realistic L40S mint budget
    phase_highwater = 16558897664          # attempt 26, measured
    assert aot_export_parallel.width_for(
        18, budget_bytes=budget, per_export_device_bytes=phase_highwater,
        cpu_workers=64)["workers"] == 1

    # ...and any per-row figure under a third of the budget buys real width.
    assert aot_export_parallel.width_for(
        18, budget_bytes=budget, per_export_device_bytes=4 << 30,
        cpu_workers=64)["workers"] == 4
