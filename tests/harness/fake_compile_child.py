"""A stand-in for ``gen_worker.aot_compile_child`` that runs no compile.

pgw#1215 made the compile child build its OWN weight-free pipeline and trace
its own share, so there is no longer a cheap already-exported program a test
can hand it: a green end-to-end pool run now requires a real AOTI compile of a
real endpoint's declared class, which is a POD leg (pgw#1215 step 3) and is
forbidden on a developer box.

What is still testable off-pod is everything the PARENT does — width, spawn,
group-wide teardown, reap, report decode, code-identity refusal, the span
partition across the process boundary, per-share durability, and the
class-set-wholeness proof. All of that reads a report file. So this writes one.

It is a REAL process spawned by the REAL pool through the real
``child_argv``/``child_env``: only the trace-and-compile interior is replaced,
and the replacement is a separate executable rather than a monkeypatch, so
nothing about the parent under test is stubbed. Behaviour is steered by the
``PGW_FAKE_CHILD`` environment variable (see :func:`script`).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_BODY = '''#!{python}
"""Fake compile child: reads the job, writes a report, exits. No compile."""
import json, os, sys, time
from pathlib import Path

job = json.loads(Path(sys.argv[-1]).read_bytes())
mode = os.environ.get("PGW_FAKE_CHILD", "ok")
share = job["share"]
index, count = int(job["share_index"]), int(job["share_count"])
report = Path(job["report"])
report.parent.mkdir(parents=True, exist_ok=True)

if mode in ("die", "die-and-hang") and index == 0:
    sys.stderr.write("fake child: refusing share 0 on purpose\\n")
    sys.exit(2)
if mode in ("hang", "die-and-hang"):
    # A share that will NOT exit on its own: only a group-wide teardown ends
    # it, which is the property the pool's failure path has to hold.
    time.sleep(600)
if mode == "wedged-no-report":
    # pgw#1243: wedged BEFORE writing a report, and SPINNING while it is —
    # the shape that defeated every evidence window the platform had, because
    # a burning core is admitted as progress. One core, exactly, which is
    # what instance A's `@ 1.00/s` counter rate literally was.
    acc = 0
    while True:
        for i in range(200000):
            acc += i * i

declared = int(os.environ.get("PGW_FAKE_DECLARED", "6"))
mine = [f"cls/dim={{i}}" for i in range(index, declared, count)]
if mode == "short" and index == 0:
    names = []
elif mode == "collide":
    names = ["cls/dim=0"]
elif mode == "refuse-midway":
    # pgw#1183 at the graph class: pack the first half, then refuse. The
    # artifacts that already exist must be REPORTED, not discarded.
    names = mine[:1]
else:
    names = mine

out = Path(job["out_dir"])
out.mkdir(parents=True, exist_ok=True)


def _position(phase, detail=""):
    # pgw#1371: the real child's position beat, faked with the same file
    # protocol so the parent's harvest is exercised for real.
    path = report.parent / "position.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(json.dumps(
        {{"phase": phase, "detail": detail, "epoch": time.time()}}).encode())
    os.replace(tmp, path)


def _stream_row(idx, row):
    rows = report.parent / "classes"
    rows.mkdir(parents=True, exist_ok=True)
    tmp = rows / f".{{idx:03d}}.tmp"
    tmp.write_bytes(json.dumps(row).encode())
    os.replace(tmp, rows / f"{{idx:03d}}.json")


_position("start")
if mode == "slow-positions":
    # pgw#1371: flat on every pgw#1243 axis — ~no CPU, no bytes in the work
    # dir — but honestly beating its position while it "works". The window
    # must admit the beat as evidence or it condemns a live share.
    for i in range(12):
        time.sleep(0.75)
        _position("working", str(i))
classes = []
stream_hang_after = int(os.environ.get("PGW_FAKE_STREAM_HANG_AFTER", "-1"))
if mode == "stream-then-hang":
    # Burn a parent-visible sliver of CPU first, like a real compile child
    # does from its first second — it is what the pool's observed-CPU ledger
    # line (pgw#1371) records for a share that never reaches its report.
    acc, t_burn = 0, time.time()
    while time.time() - t_burn < 0.6:
        acc += 1
for name in names:
    if mode == "stream-then-hang" and len(classes) == max(0, stream_hang_after):
        # pgw#1371: the mid-flight shape — some classes landed and streamed,
        # the share never reaches its report. The parent must already know
        # everything this child packed.
        while True:
            time.sleep(60)
    artifact = out / (name.replace("/", "__") + ".tar.gz")
    artifact.write_text("packed graph class " + name)
    row = {{
        "name": name, "key": "ek1-" + name.replace("/", "-"),
        "artifact": str(artifact), "metadata": json.dumps({{"name": name}}),
        "spans": {{"export_s": 1.0, "compile_s": 2.0}},
    }}
    classes.append(row)
    _stream_row(len(classes) - 1, row)
    _position("packed", name)

now = time.time()
refused = mode == "refuse-midway"
report.write_bytes(json.dumps({{
    "entry": share, "status": "refused" if refused else "compiled",
    "classes": classes,
    "declared_classes": declared,
    "detail": (
        f"{{len(classes)}} graph class(es) packed before the share refused: "
        f"graph class {{mine[1] if len(mine) > 1 else '?'}} refused"
        if refused else f"{{len(classes)}} packed graph class(es)"),
    "elapsed_s": 0.05, "peak_rss_bytes": 1024 * 1024,
    "phases": {{"lowering_s": 0.5, "codegen_s": 0.5, "graph_passes_s": 0.5,
               "host_compile_s": 0.5}},
    "spans": {{"child_wall_s": 5.0, "child_seal_s": 0.5,
              "child_torch_import_s": 0.5, "child_devlock_s": 0.0,
              "child_setup_s": 0.5, "child_trace_s": 1.0,
              "compile_wall_s": 2.0, "child_pack_s": 0.0,
              "child_other_s": 0.5,
              "lowering_s": 0.5, "codegen_s": 0.5, "graph_passes_s": 0.5,
              "host_compile_s": 0.5, "compile_other_s": 0.0}},
    "overlays": {{"seal_libhash_s": 0.01}},
    "spans_v": 2,
    "module_import_epoch": now - 5.0,
    "run_start_epoch": now - 5.0,
    "report_epoch": now,
    "code_digest": {digest!r}, "code_dir": {code_dir!r},
}}).encode())
if mode == "wedged-after-report":
    # pgw#1243, the observed wedge: every graph class packed, the report
    # written — and then the interpreter never comes down. Two production
    # mints did exactly this for 78.9 and 62 minutes.
    acc = 0
    while True:
        for i in range(200000):
            acc += i * i
sys.exit(2 if refused else 0)
'''


def script(tmp_path: Path, *, digest: Optional[str] = None) -> str:
    """Write the fake child and return the path the pool's ``python=`` takes.

    ``digest`` defaults to the parent's own, so the pool's pgw#840 code-identity
    check passes; pass something else to exercise the refusal.
    """
    from gen_worker import aot_compile_pool as pool

    path = tmp_path / "fake-compile-child.py"
    path.write_text(_BODY.format(
        python=sys.executable,
        digest=pool.CODE_DIGEST if digest is None else digest,
        code_dir=pool.PACKAGE_ROOT))
    path.chmod(0o755)
    return str(path)
