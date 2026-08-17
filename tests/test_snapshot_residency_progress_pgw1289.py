"""The residency scan must be visible WHILE it runs, not only once it ends."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from gen_worker._vendor.tensorfs import CASRef, LocalCAS

import projection_fixture
from gen_worker.models.cozy_snapshot import ensure_snapshot_async
from gen_worker.models.hub_client import WorkerResolvedRepo, WorkerResolvedRepoFile
from gen_worker.models.refs import TensorhubRef

_GATE_TIMEOUT_S = 10.0


def _ref() -> TensorhubRef:
    return TensorhubRef(owner="acme", repo="model", release="latest")


def test_a_resident_grant_advances_progress_before_the_scan_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pgw#1289: `_ensure_objects` emitted `progress` once, AFTER the residency
    loop. `LocalCAS.contains` is `verify_object`, a full sha256 re-hash of every
    byte, so a resident 31.6 GB snapshot re-hashed 31.6 GB with the hub-visible
    counter pinned at zero and `download` never reached — and the hub's 6-minute
    transfer freshness window killed a pod that had already done all the work.

    Totals were always correct; only their arrival was wrong, so any test that
    checks the final callback set passes on the broken tree. This asserts the
    arrival: the second grant's residency check refuses to complete until the
    first grant's callback has been observed, which is impossible if emission
    waits for the loop. A regression fails here with a named message on a
    10 s bound rather than wedging the suite.
    """

    first = b"resident-first"
    second = b"resident-second-object"
    store = LocalCAS(tmp_path)
    first_digest = store.put_bytes(first)
    second_digest = store.put_bytes(second)
    first_reported = threading.Event()

    real_contains = LocalCAS.contains

    def gated_contains(self: Any, ref: Any, *, size: int | None = None) -> bool:
        if CASRef.parse(ref) == second_digest and not first_reported.wait(
            _GATE_TIMEOUT_S
        ):
            raise AssertionError(
                "no progress was reported for the first resident object while "
                "the second was still being verified — the whole residency "
                "scan is invisible until it ends"
            )
        return real_contains(self, ref, size=size)

    monkeypatch.setattr(LocalCAS, "contains", gated_contains)

    resolved = WorkerResolvedRepo(
        snapshot_digest="sha256:" + "d" * 64,
        files=[
            WorkerResolvedRepoFile(
                "config.json",
                len(first),
                "http://127.0.0.1:1/must-not-fetch",
                digest=str(first_digest),
            ),
            WorkerResolvedRepoFile(
                "weights.safetensors",
                len(second),
                "http://127.0.0.1:1/must-not-fetch",
                digest=str(second_digest),
            ),
        ],
    )

    total = len(first) + len(second)
    caller = threading.current_thread()
    beats: list[tuple[int, int | None]] = []
    threads: list[threading.Thread] = []

    def progress(done: int, reported: int | None) -> None:
        beats.append((done, reported))
        threads.append(threading.current_thread())
        # The gate opens on a RECORDED mid-scan beat, so it cannot be satisfied
        # by the leading 0/total or by the final one.
        if 0 < done < total:
            first_reported.set()

    path = asyncio.run(
        ensure_snapshot_async(
            base_dir=tmp_path, ref=_ref(), resolved=resolved, progress=progress
        )
    )

    assert (path / "config.json").read_bytes() == first
    assert projection_fixture.bytes_at(path, "weights.safetensors") == second
    # The total is known before any byte is hashed, and the counter reaches it.
    assert beats[0] == (0, total)
    assert beats[-1] == (total, total)
    # And it moved while the scan was still running.
    assert any(0 < done < total for done, _total in beats)
    assert set(threads) == {caller}


def test_the_scan_yields_the_event_loop_between_grants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An emission that cannot leave the process is not an emission.

    The residency scan hashes on the worker's own event-loop thread, so
    without a turn between grants the heartbeat coroutine and the DOWNLOADING
    events the progress callback queues are stranded until the whole scan
    ends — measured at 1.0 s of blocked loop per GiB scanned, on a warm page
    cache. This pins the yield: a ticker task must observe wall time passing
    while the scan is still in flight.
    """

    bodies = [b"grant-%d" % index for index in range(4)]
    store = LocalCAS(tmp_path)
    digests = [store.put_bytes(body) for body in bodies]
    ticks: list[float] = []
    scanned: list[float] = []

    real_contains = LocalCAS.contains

    def slow_contains(self: Any, ref: Any, *, size: int | None = None) -> bool:
        scanned.append(time.monotonic())
        time.sleep(0.05)
        return real_contains(self, ref, size=size)

    monkeypatch.setattr(LocalCAS, "contains", slow_contains)

    resolved = WorkerResolvedRepo(
        snapshot_digest="sha256:" + "e" * 64,
        files=[
            WorkerResolvedRepoFile(
                f"part{index}.safetensors",
                len(body),
                "http://127.0.0.1:1/must-not-fetch",
                digest=str(digest),
            )
            for index, (body, digest) in enumerate(zip(bodies, digests, strict=True))
        ],
    )

    async def scenario() -> None:
        stop = asyncio.Event()

        async def ticker() -> None:
            while not stop.is_set():
                await asyncio.sleep(0.005)
                ticks.append(time.monotonic())

        beat = asyncio.create_task(ticker())
        try:
            await ensure_snapshot_async(
                base_dir=tmp_path, ref=_ref(), resolved=resolved
            )
        finally:
            stop.set()
            await beat

    asyncio.run(scenario())

    assert len(scanned) == len(bodies)
    assert any(scanned[0] < tick < scanned[-1] for tick in ticks), (
        "the event loop never ran between residency checks — every heartbeat "
        "and every queued progress event is stranded for the whole scan "
        f"(scan {scanned[0]:.3f}..{scanned[-1]:.3f}, ticks {ticks})"
    )
