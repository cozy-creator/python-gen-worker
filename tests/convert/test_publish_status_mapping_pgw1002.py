"""pgw#1002 A: a publish failure the HUB tagged `retryable: true` must be
reported RETRYABLE, and the durable-write / re-cast-skip halves of pgw#1003.

The defect: `HubPublishError` carries the hub's own th#1301 `retryable` bit and
`_map_exception` had no branch for the type at all — `grep -n HubPublishError
src/gen_worker/executor.py` was empty — so every hub-tagged retryable publish
failure fell through to the generic tail and reported JOB_STATUS_FATAL.
Downstream that is terminal: only JOB_STATUS_RETRYABLE is requeued, so the
orchestrator's five-attempt budget was never spent on the final artifact of a
two-hour cast. Intermediate checkpoints, which raise `ArtifactTransferError`
with an honest retryable flag, WERE requeued — the asymmetry that named the
bug.

These drive the real `publish_v2` against the fake hub and feed the exception
it actually raises into the real `_map_exception`.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from fake_hub import _FakeHub, _client
from gen_worker.hubio.client import CommitFile, HubPublishError
from gen_worker.executor import _map_exception
from gen_worker.pb import worker_scheduler_pb2 as pb

CS = 4096


def payload(n: int, seed: int = 1) -> bytes:
    out = bytearray(n)
    x = (seed * 2654435761 + 1) & 0xFFFFFFFF
    for i in range(n):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = (x >> 24) & 0xFF
    return bytes(out)


def write(tmp: Path, name: str, data: bytes) -> CommitFile:
    p = tmp / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return CommitFile(path=name, local_path=p, size_bytes=len(data))


@pytest.fixture()
def small_chunks(monkeypatch):
    monkeypatch.setattr("gen_worker.models.chunk_upload.CAS_CHUNK_SIZE_BYTES", CS)


def _publish_and_capture(fake_hub, tmp_path, verdict) -> HubPublishError:
    _FakeHub.state["complete_failure"] = verdict
    with pytest.raises(HubPublishError) as err:
        _client(fake_hub).publish_v2(
            destination_repo="acme/model",
            files=[write(tmp_path, "w.safetensors", payload(CS * 2))])
    return err.value


def test_a_hub_tagged_RETRYABLE_publish_failure_is_reported_RETRYABLE(
    fake_hub, tmp_path, small_chunks,
):
    exc = _publish_and_capture(fake_hub, tmp_path, {
        "code": "verification_backlog", "retryable": True,
        "message": "verifier is behind; retry",
    })
    status, detail = _map_exception(exc)
    assert status == pb.JOB_STATUS_RETRYABLE
    # The hub's own code LEADS the detail, so the refusal groups by a stable
    # token rather than by prose (th#1259's provenance-typing shape).
    assert detail.startswith("verification_backlog: ")


def test_a_hub_tagged_REPUDIATION_stays_FATAL(fake_hub, tmp_path, small_chunks):
    exc = _publish_and_capture(fake_hub, tmp_path, {
        "code": "invalid_manifest_for_kind", "retryable": False,
        "message": "missing_diffusers_single_file_safetensors",
    })
    status, detail = _map_exception(exc)
    assert status == pb.JOB_STATUS_FATAL
    assert detail.startswith("invalid_manifest_for_kind: ")


def test_a_publish_error_the_hub_never_classified_stays_FATAL():
    """`None` honestly means "the hub named nothing". Inventing a retry the
    hub did not offer is how a permanently-broken publish burns five attempts."""
    exc = HubPublishError("publish declare failed (400): ...", status=400,
                          code="invalid_manifest")
    assert exc.retryable is None
    status, detail = _map_exception(exc)
    assert status == pb.JOB_STATUS_FATAL
    assert detail.startswith("invalid_manifest: ")


def test_the_detail_is_bounded_and_never_leaks_a_stack():
    exc = HubPublishError("x" * 4000, retryable=True)
    status, detail = _map_exception(exc)
    assert status == pb.JOB_STATUS_RETRYABLE
    assert len(detail) <= 512


# ---------------------------------------------------------------------------
# pgw#1003: the writer's durable finalize
# ---------------------------------------------------------------------------


def test_the_incremental_writer_finalizes_ATOMICALLY_and_fsyncs(tmp_path, monkeypatch):
    """`close()` used to just close the handle — no fsync, no temp+rename — so
    a hard-killed pod could leave a truncated cast output under the real name
    that nothing re-verifies. The download side has done this correctly since
    gw#408 (`s3_transfer` fsync -> os.replace -> fsync_dir)."""
    from gen_worker.convert import writer as w

    synced: list[str] = []
    monkeypatch.setattr(w, "fsync_file", lambda p: synced.append(f"file:{p.name}"))
    monkeypatch.setattr(w, "fsync_dir", lambda p: synced.append("dir"))

    out = tmp_path / "model.safetensors"
    seen: list[bool] = []
    with w.IncrementalSafetensorsWriter(out, metadata={"k": "v"}) as writer:
        writer.add_tensor_metadata("a", dtype="F32", shape=[2])
        writer.write_header()
        seen.append(out.exists())  # nothing under the real name yet
        writer.write_tensor("a", b"\x00" * 8)

    assert seen == [False], "bytes must not appear under the final name mid-write"
    assert out.is_file()
    assert synced == [f"file:.{out.name}.partial", "dir"]
    assert not (tmp_path / f".{out.name}.partial").exists()
    # And it is a readable safetensors file.
    from safetensors import safe_open

    with safe_open(str(out), framework="pt", device="cpu") as f:
        assert f.metadata()["k"] == "v"
        assert list(f.keys()) == ["a"]


def test_a_writer_body_that_RAISES_leaves_no_output_at_all(tmp_path):
    from gen_worker.convert.writer import IncrementalSafetensorsWriter

    out = tmp_path / "model.safetensors"
    with pytest.raises(RuntimeError):
        with IncrementalSafetensorsWriter(out) as writer:
            writer.add_tensor_metadata("a", dtype="F32", shape=[2])
            writer.add_tensor_metadata("b", dtype="F32", shape=[2])
            writer.write_header()
            writer.write_tensor("a", b"\x00" * 8)
            raise RuntimeError("cast blew up")
    assert not out.exists()
    assert os.listdir(tmp_path) == []


def test_an_INCOMPLETE_tensor_set_is_never_committed(tmp_path):
    """A truncated artifact under the real name is worse than no artifact: the
    publish path proves digests from bytes in hand, so it would happily ship
    it."""
    from gen_worker.convert.writer import IncrementalSafetensorsWriter

    out = tmp_path / "model.safetensors"
    writer = IncrementalSafetensorsWriter(out)
    writer.add_tensor_metadata("a", dtype="F32", shape=[2])
    writer.add_tensor_metadata("b", dtype="F32", shape=[2])
    writer.write_header()
    writer.write_tensor("a", b"\x00" * 8)
    writer.close()  # commit=True, but only 1 of 2 tensors is written
    assert not out.exists()
    assert os.listdir(tmp_path) == []


def test_the_writer_output_is_byte_identical_to_what_it_used_to_produce(tmp_path):
    """The finalize changed; the bytes must not have."""
    from gen_worker.convert.writer import IncrementalSafetensorsWriter

    out = tmp_path / "m.safetensors"
    with IncrementalSafetensorsWriter(out, metadata={"b": "2", "a": "1"}) as w:
        w.add_tensor_metadata("t", dtype="F16", shape=[4])
        w.write_header()
        w.write_tensor("t", b"\x01\x02" * 4)
    raw = out.read_bytes()
    # Header is sorted-key JSON for byte determinism (unchanged contract).
    assert b'"__metadata__":{"a":"1","b":"2"}' in raw
    assert raw.endswith(b"\x01\x02" * 4)
    assert hashlib.sha256(raw).hexdigest() == hashlib.sha256(out.read_bytes()).hexdigest()
