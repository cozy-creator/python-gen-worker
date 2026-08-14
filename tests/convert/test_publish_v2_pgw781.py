"""Tensorhub publish policy composed over HashRepo's public API."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fake_hub import _client, _FakeHub
from hashrepo import MAX_CHUNK_SIZE

from gen_worker.hubio.client import CommitFile, HubPublishError


def _write(root: Path, name: str, data: bytes) -> CommitFile:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return CommitFile(path=name, local_path=path, size_bytes=len(data))


def test_small_file_publishes_as_one_hashrepo_object(fake_hub, tmp_path: Path) -> None:
    data = b'{"model":"hashrepo"}'
    result = _client(fake_hub).publish_v2(
        destination_repo="acme/model",
        files=[_write(tmp_path, "config.json", data)],
        tags=["prod"],
    )
    assert result.checkpoint_id
    declaration = next(iter(_FakeHub.state["publishes"].values()))["files"][0]
    assert declaration == {
        "path": "config.json",
        "size_bytes": len(data),
        "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
    }
    assert result.uploaded == 1


def test_non_safetensors_file_uses_bounded_hashrepo_chunks(fake_hub, tmp_path: Path) -> None:
    data = b"a" * MAX_CHUNK_SIZE + b"tail"
    result = _client(fake_hub).publish_v2(
        destination_repo="acme/model",
        files=[_write(tmp_path, "weights.safetensors", data)],
        tags=["prod"],
    )
    declaration = next(iter(_FakeHub.state["publishes"].values()))["files"][0]
    assert [chunk["len"] for chunk in declaration["chunks"]] == [MAX_CHUNK_SIZE, 4]
    assert all("sha256:" not in chunk["digest"] for chunk in declaration["chunks"])
    assert result.uploaded == 2


def test_valid_safetensors_publishes_header_and_tensor_chunks_at_small_size(
    fake_hub, tmp_path: Path
) -> None:
    body = b"tensor-bytes"
    header = json.dumps(
        {
            "weight": {
                "dtype": "U8",
                "shape": [len(body)],
                "data_offsets": [0, len(body)],
            }
        },
        separators=(",", ":"),
    ).encode()
    header += b" " * (-len(header) % 8)
    data = len(header).to_bytes(8, "little") + header + body

    result = _client(fake_hub).publish_v2(
        destination_repo="acme/model",
        files=[_write(tmp_path, "weights.safetensors", data)],
        tags=["prod"],
    )

    declaration = next(iter(_FakeHub.state["publishes"].values()))["files"][0]
    assert [chunk["len"] for chunk in declaration["chunks"]] == [8 + len(header), len(body)]
    assert result.uploaded == 2


def test_republishing_identical_bytes_is_a_remote_dedup_hit(
    fake_hub, tmp_path: Path
) -> None:
    file = _write(tmp_path, "config.json", b"same")
    client = _client(fake_hub)
    client.publish_v2(destination_repo="acme/model", files=[file], tags=["prod"])
    result = client.publish_v2(
        destination_repo="acme/model", files=[file], tags=["prod"]
    )
    assert result.uploaded == 0
    assert result.deduped == 1


def test_by_reference_add_is_refused(fake_hub) -> None:
    with pytest.raises(HubPublishError, match="by-reference"):
        _client(fake_hub).publish_v2(
            destination_repo="acme/model",
            files=[CommitFile(path="weights.safetensors", size_bytes=10)],
        )


def test_tensorhub_metadata_and_provenance_stay_in_the_adapter(
    fake_hub, tmp_path: Path
) -> None:
    _client(fake_hub).publish_v2(
        destination_repo="acme/model",
        files=[_write(tmp_path, "config.json", b"{}")],
        tags=["prod"],
        dtype="int8:awq",
        metadata={"placement": "l4"},
        provenance={"upstream_revision": "abc123", "parents": "forbidden"},
    )
    declaration = next(iter(_FakeHub.state["publishes"].values()))
    assert declaration["dtype"] == "int8-awq"
    assert declaration["metadata"] == {"placement": "l4"}
    assert declaration["provenance"] == {"upstream_revision": "abc123"}
