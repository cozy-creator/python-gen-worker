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
    """The failure mode of guessing is an OOM that kills a 74-minute phase."""
    g = 18
    for kw in ({"per_export_device_bytes": 0}, {"free_device_bytes": 0}):
        base = {"free_device_bytes": 40 << 30,
                "per_export_device_bytes": 5 << 30, "cpu_workers": 32}
        base.update(kw)
        assert aot_export_parallel.width_for(g, **base)["workers"] == 1

    w = aot_export_parallel.width_for(
        g, free_device_bytes=40 << 30, per_export_device_bytes=5 << 30,
        cpu_workers=32)
    assert w["workers"] == 8 and w["binding"] == "ceiling"
    # the EXPORT footprint, not the compile pool's 11.07 GiB estimate
    w = aot_export_parallel.width_for(
        g, free_device_bytes=27 << 30, per_export_device_bytes=5 << 30,
        cpu_workers=32, ceiling=8)
    assert w["workers"] == 5 and w["binding"] == "vram"
    assert aot_export_parallel.width_for(
        2, free_device_bytes=40 << 30, per_export_device_bytes=1 << 30,
        cpu_workers=32)["workers"] == 1


def test_flag_is_off_by_default(monkeypatch):
    monkeypatch.delenv(aot_export_parallel.ENV_FLAG, raising=False)
    assert aot_export_parallel.enabled() is False
    monkeypatch.setenv(aot_export_parallel.ENV_FLAG, "1")
    assert aot_export_parallel.enabled() is True


def test_the_CALL_SITE_exists_and_reads_the_shipped_measurement():
    """pgw#868 A4: the connection, proven as an OBSERVABLE not as a construction.

    Every instance of this program's signature defect passed its own unit
    tests — a module with a correct API that nothing imports. So this asserts
    the JOIN: `aot_mint` imports the module, and `decide()` consumes exactly
    the key `aot_mint` ships (`export_peak_device_bytes`, 0.90.6) and produces
    a width decision from it.
    """
    import inspect

    from gen_worker import aot_mint

    # 1. aot_mint actually imports it, and actually calls it
    assert aot_mint.aot_export_parallel is aot_export_parallel
    src = inspect.getsource(aot_mint)
    assert "aot_export_parallel.decide(rows, timings)" in src

    # 2. the consumer reads the producer's key, and the width MOVES with it
    rows = [("p", True)] * 18 + [("p", False)] * 18
    unmeasured = aot_export_parallel.decide(
        rows, {}, free_device_bytes=40 << 30, cpu_workers=32)
    assert unmeasured["export_parallel_width"] == 1.0
    assert unmeasured["export_parallel_binding"] == 5.0   # unmeasured

    measured = aot_export_parallel.decide(
        rows, {"export_peak_device_bytes": float(5 << 30)},
        free_device_bytes=40 << 30, cpu_workers=32)
    assert measured["export_parallel_width"] == 8.0
    assert measured["export_parallel_groups"] == 2.0
    assert measured["export_parallel_largest_group"] == 18.0
    assert measured["export_parallel_per_export_bytes"] == float(5 << 30)

    # 3. the flag has a real caller now, and is reported either way
    assert measured["export_parallel_enabled"] == 0.0
