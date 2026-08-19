"""The weight download's byte POSITION, off the real fetch loop.

# pgw#1455: rental #6 (e2e#1910) parked a request on `model_download_pending` for
# 49 minutes, ~$0.59, with 0 rows in `worker_activity_events`. The scheduler's
# `attempts=2963` is its own 1 Hz placement poll and tracks wall-clock seconds
# 1:1, so it reads identically whether a download is healthy, slow or wedged.
# Nothing anywhere said whether the download was PROGRESSING.

Everything here drives the REAL pull: objects are served by a real HTTP origin
that TRICKLES them out, fetched by the real `gen_worker.transfer.grants.download`
through the real `ensure_snapshot_async`, and the positions are read back off a
bound activity sink as `ActivityUpdate` envelopes — the same wire the th#1839
route serves and the hub decodes.

The reporter is wired by handing `FetchPosition.progress` to the download layer's
own `progress=` callback, which is exactly what `models/store.py` does at the one
funnel every materialization path passes through. So the object under test here
is the object that runs in production, on the loop that runs in production.
"""

from __future__ import annotations

import asyncio
import hashlib
import http.server
import re
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from gen_worker import activity, weight_position
from gen_worker.models.cozy_snapshot import ensure_snapshot_async
from gen_worker.models.hub_client import WorkerResolvedRepo, WorkerResolvedRepoFile
from gen_worker.models.refs import TensorhubRef
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.weight_position import MIB, FetchPosition

#: Six objects at 1.5 MiB. Small enough to stay CPU-cheap, large enough that the
#: position moves through several whole MiB while the transfer is in flight.
OBJECT_BYTES = 3 * MIB // 2
OBJECTS = 6


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802
        key = self.path.rsplit("/", 1)[-1]
        body = self.server.blobs.get(key)  # type: ignore[attr-defined]
        if body is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        # THE THROTTLE. A source that answers instantly cannot tell a position
        # that advances DURING a transfer from one computed after it.
        chunk = len(body) // 4 or len(body)
        for start in range(0, len(body), chunk):
            self.wfile.write(body[start:start + chunk])
            self.wfile.flush()
            time.sleep(0.01)


class _Origin:
    """A real, slow HTTP origin."""

    def __init__(self) -> None:
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.blobs = {}  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def put(self, data: bytes) -> str:
        digest = _sha(data)
        self.server.blobs[digest] = data  # type: ignore[attr-defined]
        host, port = self.server.server_address[:2]
        return f"http://{host!s}:{port!s}/{digest}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class _Wire:
    """A bound activity sink, collecting what a real pull emits."""

    def __init__(self) -> None:
        self.updates: list[pb.ActivityUpdate] = []

    async def send(self, msg: pb.WorkerMessage) -> None:
        if msg.WhichOneof("msg") == "activity_update":
            self.updates.append(msg.activity_update)

    def positions(self) -> list[pb.ActivityUpdate]:
        return [u for u in self.updates if u.kind == activity.KIND_WEIGHT_FETCH]


def _detail(update: pb.ActivityUpdate) -> dict[str, str]:
    """The wire grammar every reader parses, `(\\w+)=(\\S+)`."""
    return dict(re.findall(r"(\w+)=(\S+)", update.detail))


def _resolved(files: list[tuple[str, bytes, str]]) -> WorkerResolvedRepo:
    fingerprint = hashlib.sha256(
        b"|".join(f"{path}:{_sha(body)}".encode() for path, body, _url in files)
    ).hexdigest()
    return WorkerResolvedRepo(
        snapshot_digest="sha256:" + fingerprint,
        files=[
            WorkerResolvedRepoFile(path, len(body), url, digest=_sha(body))
            for path, body, url in files
        ],
    )


def _pull(wire: _Wire, base: Path, repo: str, resolved: WorkerResolvedRepo,
          position: FetchPosition) -> None:
    """One real snapshot pull with the position reporter wired the way
    `models/store.py` wires it: its `progress` IS the download's callback."""

    async def run() -> None:
        activity.bind_sink(wire.send, asyncio.get_running_loop())
        position.open()
        try:
            await ensure_snapshot_async(
                base_dir=base,
                ref=TensorhubRef(owner="acme", repo=repo, release="latest"),
                resolved=resolved,
                progress=position.progress,
            )
        finally:
            position.close(ok=True)
        # The sink ships through `create_task`; give those tasks their turn.
        for _ in range(4):
            await asyncio.sleep(0)

    asyncio.run(run())


@pytest.fixture(autouse=True)
def _clean_sink() -> Any:
    activity.reset_for_tests()
    yield
    activity.reset_for_tests()


@pytest.fixture()
def _fine_cadence(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Production emits one row per 256 MiB or per minute; this suite moves 9 MiB
    in under a second. The CADENCE is policy and is asserted separately — what
    this fixture buys is that the ADVANCEMENT mechanism is exercised at test
    scale instead of being read."""
    monkeypatch.setattr(weight_position, "STRIDE_MIB", 1)
    monkeypatch.setattr(weight_position, "MIN_INTERVAL_S", 0.0)
    yield


def test_the_position_advances_while_the_fetch_runs(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """THE FACT rental #6 did not have. A trickling multi-MiB pull leaves a
    STRICTLY INCREASING sequence of byte positions on the activity stream, so
    'downloading' and 'downloading is PROGRESSING' stop being the same
    observation."""
    origin = _Origin()
    try:
        blobs = [
            (f"unet/shard-{i}.safetensors", bytes([i + 1]) * OBJECT_BYTES)
            for i in range(OBJECTS)
        ]
        resolved = _resolved([(p, b, origin.put(b)) for p, b in blobs])
        wire = _Wire()
        position = FetchPosition("tensorhub:acme/model-a@latest",
                                 total_bytes=OBJECTS * OBJECT_BYTES)
        _pull(wire, tmp_path / "pod", "model-a", resolved, position)

        rows = wire.positions()
        phases = [r.phase for r in rows]
        assert phases[0] == weight_position.PHASE_STARTED
        assert phases[-1] == weight_position.PHASE_FETCHED
        # The advancement itself, on the TYPED column the hub reads.
        steps = [r.step for r in rows]
        assert steps[0] == 0, "the opening row states position zero"
        assert steps[-1] == (OBJECTS * OBJECT_BYTES) // MIB
        assert len(rows) >= 4, f"a 9 MiB trickle left only {len(rows)} position(s)"
        # STRICT increase across every row the transfer produced while running.
        # The terminal `fetched` row is deliberately outside this: it restates
        # where the fetch stopped and may repeat the last position, which is the
        # only way a fetch that ends between strides gets a closing row at all.
        running = [r.step for r in rows if r.phase != weight_position.PHASE_FETCHED]
        assert all(b > a for a, b in zip(running, running[1:])), (
            f"positions must strictly increase; got {steps}")
        assert steps[-1] >= running[-1]
        # A hub reading these differences the way th#1243 does sees movement.
        assert steps[-1] > steps[0]

        for row in rows:
            assert row.state == pb.ActivityState.ACTIVITY_STATE_COMPLETED
            got = _detail(row)
            assert got["ref"] == "tensorhub:acme/model-a@latest"
            # The typed column and the sentence can never disagree.
            assert int(got["pos_mib"]) == row.step
            assert int(got["total_mib"]) == row.total_steps
    finally:
        origin.close()


def test_a_sub_mib_fetch_still_leaves_rows_at_zero(tmp_path: Path) -> None:
    """Absence renders as a row that says zero, at the PRODUCTION cadence (no
    fixture here). A transfer too small to move a whole MiB emits no `fetching`
    row — and would emit nothing at all without the unconditional `started` and
    `fetched` pair, which is the 0-rows reading that cost rental #6."""
    origin = _Origin()
    try:
        body = b"c" * 4096
        resolved = _resolved([("config/model.safetensors", body, origin.put(body))])
        wire = _Wire()
        position = FetchPosition("tensorhub:acme/tiny@latest", total_bytes=len(body))
        _pull(wire, tmp_path / "pod", "tiny", resolved, position)

        rows = wire.positions()
        assert [r.phase for r in rows] == [
            weight_position.PHASE_STARTED, weight_position.PHASE_FETCHED]
        assert [r.step for r in rows] == [0, 0]
        assert [r.total_steps for r in rows] == [0, 0]
    finally:
        origin.close()


def test_the_position_is_integral_mib_never_a_fraction() -> None:
    """pgw#1397's documented trap, which this reuses the shape of.

    The hub parses an INTEGER off the position field. A position carried in GiB
    spends the first 1073741823 bytes truncating to `int(0.97) == 0`, so a
    healthy multi-GB transfer would report a frozen position — the wedge it is
    not. Integral MiB is the unit for that reason and nothing may make it a
    float."""
    position = FetchPosition("tensorhub:acme/big@latest", total_bytes=7 * 1024 * MIB)
    for done, expected in (
        (0, 0),
        (MIB - 1, 0),
        (MIB, 1),
        (4212 * MIB + 900_000, 4212),
        (7 * 1024 * MIB, 7168),
    ):
        position._pos_bytes = done
        assert position.position_mib == expected
        assert isinstance(position.position_mib, int)


def test_a_frozen_or_regressing_position_emits_nothing_new(_fine_cadence: Any) -> None:
    """The hub advances on STRICT INCREASE, so the worker must never manufacture
    an increase. A flat position (the wedge) and a position that goes backwards
    (a retry re-reporting fewer materialized bytes) both leave the last row
    standing — its age is then the evidence, which is the hub's to read."""
    wire: list[pb.ActivityUpdate] = []
    original = weight_position.FetchPosition._emit

    def spy(self: FetchPosition, phase: str) -> None:
        wire.append(pb.ActivityUpdate(phase=phase, step=self.position_mib))
        original(self, phase)

    position = FetchPosition("tensorhub:acme/model-a@latest", total_bytes=8 * MIB)
    try:
        weight_position.FetchPosition._emit = spy  # type: ignore[method-assign]
        position.open()
        position.progress(4 * MIB, 8 * MIB)
        emitted = len(wire)
        position.progress(4 * MIB, 8 * MIB)          # frozen
        position.progress(4 * MIB + MIB - 1, 8 * MIB)  # under a whole MiB
        position.progress(1 * MIB, 8 * MIB)          # a retry, backwards
        assert len(wire) == emitted, f"{wire[emitted:]} should not have been emitted"
        position.progress(6 * MIB, 8 * MIB)          # real movement resumes
        assert [u.step for u in wire] == [0, 4, 6]
    finally:
        weight_position.FetchPosition._emit = original  # type: ignore[method-assign]
