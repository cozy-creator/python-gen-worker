"""pgw#1371: a mint's progress is visible per GRAPH CLASS, live.

THE FLEET SIGNATURE THIS PINS. Two 0.124.0 pods (rzz5p4e7b2kcpp,
c7bx4yxbh3wx87, e2e#1892 runs 7/8) ran the 36-class sdxl runtime mint and
rolled up ``pool_wall_s 2798 / pool_busy_s 0 / pool_efficiency 0.0`` and
``status=abandoned n_entries=0`` — which the tracker read as "the pool never
dispatches". It dispatches fine: pod zco8e1bx0t1jgk completed the IDENTICAL
shape the day before at 0.9989 pool efficiency in 3608 s (hub
``worker_activity_events``, read 2026-08-18). The defect is that a share of
36/K classes reported ONCE, at its end, so a pool torn down 46 minutes into
real compile work was indistinguishable — on every wire fact — from a pool
that did nothing, and the hub's stall rule read the healthy mint as
``self_stalled=t`` for its whole life.

So the child streams a row per packed class and a position beat per phase, the
parent harvests them every poll, and every consumer of "how far along is this
mint" — the beat counter, the on-disk snapshot, the abandon table, the pool
ledger, the silence window — advances at the granularity the work completes
at. These tapes drive the REAL pool against REAL child processes; only the
compile interior is faked (the local-testing rule: no local inductor).
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

import pytest

from gen_worker import aot_compile_pool as pool_mod
from harness import fake_compile_child

torch = pytest.importorskip("torch")

from gen_worker import aot_mint  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

_DECLARED = 6

#: Tape guard only — generous against the fix, tiny against the defect. No
#: production code keys on it.
_TAPE_GUARD_S = 60.0


@pytest.fixture(autouse=True)
def _clean_fake_child_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGW_FAKE_CHILD", raising=False)
    monkeypatch.delenv("PGW_FAKE_DECLARED", raising=False)
    monkeypatch.delenv("PGW_FAKE_STREAM_HANG_AFTER", raising=False)


def _pool(
    tmp_path: Path, *, mode: str, workers: int = 1,
    window_s: float = 600.0,
) -> pool_mod.EntryCompilePool:
    os.environ["PGW_FAKE_CHILD"] = mode
    os.environ["PGW_FAKE_DECLARED"] = str(_DECLARED)
    width = pool_mod.entry_workers(
        _DECLARED, limit=workers, vcpus=16, available_bytes=64 * 1024**3,
        device_lock=True)
    return pool_mod.EntryCompilePool(
        tmp_path / "pool", width=width, cache_dir=str(tmp_path / "cache"),
        python=fake_compile_child.script(tmp_path),
        entry_silence_window_s=window_s)


def _template(tmp_path: Path) -> pool_mod.EntryJob:
    return pool_mod.EntryJob(
        function="generate", modules=("harness.toy_endpoints",),
        out_dir=str(tmp_path / "artifacts"))


def test_every_class_beats_as_it_lands_not_once_per_share(
    tmp_path: Path,
) -> None:
    """RED before pgw#1371: the only pool callback was per SHARE.

    One worker, six classes, so the old contract fired once — after
    everything. The class contract fires six times with a monotonically
    rising landed count against the class GOAL, which is what the hub's
    stall-rule counter needs to stay honest through a share that takes an
    hour on the fleet.
    """
    p = _pool(tmp_path, mode="ok")
    landed: List[Tuple[str, int, int]] = []
    packed = p.compile(
        _template(tmp_path),
        on_class=lambda name, done, total: landed.append((name, done, total)))

    assert sorted(packed) == sorted(f"cls/dim={i}" for i in range(_DECLARED))
    assert [done for _, done, _ in landed] == list(range(1, _DECLARED + 1)), (
        "the landed count must rise by exactly one per streamed class — a "
        "single end-of-share burst is the old share-granular blindness with "
        "a new name")
    assert all(total == _DECLARED for _, _, total in landed)
    facts = p.ledger.facts()
    assert facts["pool_classes_landed"] == _DECLARED
    assert "pool_child_cpu_s" in facts


def test_a_pool_torn_down_mid_share_reports_what_actually_landed(
    tmp_path: Path,
) -> None:
    """THE fleet shape: abandoned before any share reached its report.

    The child streams two classes and then never reports (the artifacts are
    on disk — that is pgw#1183's per-class durability). The supervisor
    abandons, exactly as the worker shutdown did on the two 0.124.0 pods.
    Before pgw#1371 the pool's whole account of this was `busy_s 0` and an
    empty class table; now the abandon sentence, the class spans and the
    ledger all name the two landed classes, and the ledger carries the CPU
    its children were observed burning.
    """
    os.environ["PGW_FAKE_STREAM_HANG_AFTER"] = "2"
    p = _pool(tmp_path, mode="stream-then-hang")
    seen: List[int] = []
    t0 = time.monotonic()

    def _abandon() -> bool:
        # Abandon once the parent has SEEN two classes land; the deadline is
        # the tape's own guard, and the assertions below catch it firing.
        return len(seen) >= 2 or time.monotonic() - t0 > _TAPE_GUARD_S

    with pytest.raises(pool_mod.EntryCompileAbandoned) as caught:
        p.compile(
            _template(tmp_path),
            on_class=lambda name, done, total: seen.append(done),
            should_abandon=_abandon)

    assert len(seen) >= 2, (
        "the parent never harvested the streamed classes — the abandon came "
        "from the tape guard, which is exactly the fleet's blind mid-flight "
        "teardown")
    assert "2 graph class(es) are already packed" in str(caught.value)
    assert p.ledger.classes_landed == 2
    assert set(p.class_spans) == {"cls/dim=0", "cls/dim=1"}
    # The observed-CPU line: the number that distinguishes "torn down
    # mid-flight" from "did nothing" on a roll-up whose busy_s is 0.
    assert p.ledger.facts()["pool_child_cpu_s"] > 0.0
    # And the artifacts really are on disk for the next attempt's accretion.
    for row in p.class_spans:
        assert (tmp_path / "artifacts" /
                (row.replace("/", "__") + ".tar.gz")).exists()


def test_a_sleeping_share_that_streams_positions_is_not_condemned(
    tmp_path: Path,
) -> None:
    """The silence window admits the child's OWN streamed evidence.

    This child burns ~no CPU and writes nothing into its work dir — on the
    pgw#1243 axes it is flat — but it beats its position file while it
    works. RED with the harvest ignored by the window: the pool condemns a
    share that is demonstrably advancing through its phases.
    """
    script = fake_compile_child.script(tmp_path)
    os.environ["PGW_FAKE_CHILD"] = "slow-positions"
    os.environ["PGW_FAKE_DECLARED"] = str(_DECLARED)
    width = pool_mod.entry_workers(
        _DECLARED, limit=1, vcpus=16, available_bytes=64 * 1024**3,
        device_lock=True)
    p = pool_mod.EntryCompilePool(
        tmp_path / "pool", width=width, cache_dir=str(tmp_path / "cache"),
        python=script, entry_silence_window_s=3.0)

    packed = p.compile(_template(tmp_path))
    assert sorted(packed) == sorted(f"cls/dim={i}" for i in range(_DECLARED))


def test_the_child_stream_protocol_roundtrips_into_the_harvest(
    tmp_path: Path,
) -> None:
    """The REAL child-side writers against the REAL parent-side harvest.

    The fake child above re-implements the file protocol; this drives the
    production writer functions themselves, so the two sides cannot drift
    apart silently.
    """
    from gen_worker import aot_compile_child as child_mod

    slot = tmp_path / "share-000"
    slot.mkdir(parents=True)
    job = pool_mod.EntryJob(
        function="generate", modules=("m",),
        report=str(slot / pool_mod.ENTRY_REPORT_NAME),
        work=str(slot / "work"), out_dir=str(tmp_path / "artifacts"))
    artifact = tmp_path / "artifacts" / "a.tar.gz"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("x")

    child_mod._mark_position(job, "compile", detail="cls/one (1 of 3)")
    child_mod._stream_class_row(job, 0, pool_mod.PackedGraphClass(
        name="cls/one", key="ek1-one", artifact=str(artifact),
        metadata="{}", spans={"export_s": 1.5, "compile_s": 2.5}))

    width = pool_mod.entry_workers(
        3, limit=1, vcpus=16, available_bytes=64 * 1024**3, device_lock=True)
    p = pool_mod.EntryCompilePool(tmp_path / "pool", width=width)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        row = pool_mod._Running(
            entry="share-000", proc=proc, job=job,
            started=time.monotonic(), stderr_path=slot / "stderr.log")
        beats: List[Tuple[str, int, int]] = []
        p._on_class = lambda name, done, total: beats.append(
            (name, done, total))

        assert p._harvest_progress(row) is True
        assert p.class_spans["cls/one"] == {"export_s": 1.5, "compile_s": 2.5}
        assert p.ledger.classes_landed == 1
        assert beats == [("cls/one", 1, 3)]
        position = json.loads(row.position)
        assert position["phase"] == "compile"
        assert "cls/one" in position["detail"]
        # Idempotent: nothing new means no advance and no double count.
        assert p._harvest_progress(row) is False
        assert p.ledger.classes_landed == 1
    finally:
        with contextlib.suppress(OSError):
            proc.kill()
        proc.wait()


def test_a_killed_mints_snapshot_names_the_landed_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """END TO END through `mint_graph_classes`: the on-disk phase snapshot of
    an abandoned mint carries n_entries > 0, real per-class spans and the
    final pool ledger — everything runs 7/8 reported as zero.
    """
    monkeypatch.setenv("PGW_FAKE_CHILD", "stream-then-hang")
    monkeypatch.setenv("PGW_FAKE_DECLARED", str(_DECLARED))
    monkeypatch.setenv("PGW_FAKE_STREAM_HANG_AFTER", "2")
    snapshot = tmp_path / "phases.json"
    width = pool_mod.entry_workers(
        _DECLARED, limit=1, vcpus=16, available_bytes=64 * 1024**3,
        device_lock=True)
    seen: List[str] = []
    t0 = time.monotonic()

    def _progress(phase: str, step: int, total: int, note: str) -> None:
        seen.append(f"{phase} {step}/{total} {note}")

    def _abandon() -> bool:
        return len([s for s in seen if "cls/" in s]) >= 2 \
            or time.monotonic() - t0 > _TAPE_GUARD_S

    with pytest.raises(pool_mod.EntryCompileAbandoned):
        aot_mint.mint_graph_classes(
            _template(tmp_path),
            workdir=tmp_path / "pool",
            width=width,
            spec=aot_mint.ExportSpec(family="tiny1371", target="unet"),
            python=fake_compile_child.script(tmp_path),
            on_progress=_progress,
            phase_snapshot=snapshot,
            should_abandon=_abandon)

    table = json.loads(snapshot.read_text())
    assert table["n_entries"] == 2, (
        "an abandoned mint must name the classes that landed — n_entries=0 "
        "on 46 minutes of real work is the exact fleet mis-read this exists "
        "to end")
    assert set(table["entries"]) == {"cls/dim=0", "cls/dim=1"}
    assert table["totals"]["export_s"] > 0
    pool_block = table["pool"]
    assert pool_block["pool_classes_landed"] == 2
    assert pool_block["pool_child_cpu_s"] > 0.0
    # The terminal beat stamped the final position.
    assert table["at"]["note"].startswith("abandoned")
    # The counter axis is CLASSES: the total the hub's stall rule sees is the
    # class goal, never the worker count — and the class-named beats really
    # reached the sink (the wire the hub's `self_stalled` verdict reads).
    assert any(f"/{_DECLARED} " in s for s in seen)
    assert any("cls/dim=0" in s for s in seen), (
        "no per-class beat reached the progress sink — the hub would read "
        "this healthy mint as self_stalled=t, which is the fleet signature")
