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
    beats: list[tuple[int, int | None]] = []

    def progress(done: int, reported: int | None) -> None:
        beats.append((done, reported))
        if 0 < done < total:
            first_reported.set()

    path = asyncio.run(
        ensure_snapshot_async(
            base_dir=tmp_path, ref=_ref(), resolved=resolved, progress=progress
        )
    )

    assert (path / "config.json").read_bytes() == first
    assert projection_fixture.bytes_at(path, "weights.safetensors") == second
    assert beats[0] == (0, total)
    assert beats[-1] == (total, total)
    assert any(0 < done < total for done, _total in beats)
    assert [done for done, _t in beats] == sorted(done for done, _t in beats), (
        f"the reported position went BACKWARDS across scan threads: {beats}")


def test_the_scan_yields_the_event_loop_between_grants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An emission that cannot leave the process is not an emission."""

    bodies = [b"grant-%d" % index for index in range(4)]
    store = LocalCAS(tmp_path)
    digests = [store.put_bytes(body) for body in bodies]
    ticks: list[float] = []
    scanned: list[float] = []
    finished: list[float] = []

    real_contains = LocalCAS.contains

    grant_digests = {str(digest) for digest in digests}

    def slow_contains(self: Any, ref: Any, *, size: int | None = None) -> bool:
        if str(ref) not in grant_digests:
            return bool(real_contains(self, ref, size=size))
        scanned.append(time.monotonic())
        time.sleep(0.05)
        try:
            return real_contains(self, ref, size=size)
        finally:
            finished.append(time.monotonic())

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
    assert len(finished) == len(bodies)
    first_entry, last_return = min(scanned), max(finished)
    assert any(first_entry < tick < last_return for tick in ticks), (
        "the event loop never ran WHILE the residency scan was in flight — "
        "every heartbeat and every queued progress event is stranded for the "
        f"whole scan (scan {first_entry:.3f}..{last_return:.3f}, ticks {ticks})"
    )
