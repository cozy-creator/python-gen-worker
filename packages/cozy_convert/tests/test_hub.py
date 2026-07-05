"""HubClient against a threaded fake of tensorhub's /commits API.

Exercises the real HTTP code path: POST /commits (presign), part PUTs,
per-upload complete, finalize (202 poll -> 200), blake3 dedup skips, and
bounded retries on transient failures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cozy_convert.hub import CommitFile, HubPublishError, blake3_file, files_from_tree

from conftest import _FakeHub, _client


def test_commit_uploads_completes_and_finalizes(fake_hub, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)
    (tmp_path / "sub").mkdir()
    (tmp_path / "config.json").write_text('{"a":1}')
    (tmp_path / "sub" / "weights.safetensors").write_bytes(b"\x00" * 64)

    result = _client(fake_hub).commit(
        destination_repo="acme/my-model",
        files=files_from_tree(tmp_path),
        tags=["prod"],
        mode="replace",
        flavor="bf16",
        dtype="bf16",
        file_layout="diffusers",
        file_type="safetensors",
        lineage=[{"parent_repo": "external-sources/upstream",
                  "parent_checkpoint_id": "hf:org/name",
                  "relationship_kind": "import"}],
        auto_create_external_parent=True,
    )
    assert result.revision_id == "rev-1"
    assert result.uploaded == 2 and result.deduped == 0

    st = _FakeHub.state
    req = st["commit_request"]
    assert req["mode"] == "replace"
    assert req["tags"] == [{"tag": "prod"}]
    assert req["flavor"] == "bf16"
    assert req["auto_create_external_parent"] is True
    ops = {op["path"]: op for op in req["operations"]}
    assert set(ops) == {"config.json", "sub/weights.safetensors"}
    assert ops["config.json"]["blake3"] == blake3_file(tmp_path / "config.json")
    # Bytes actually PUT + parts echoed on complete.
    assert len(st["put_bytes"]) == 2
    assert st["complete_bodies"][0]["parts"][0] == {"part_number": 1, "etag": "etag-1"}
    assert st["finalize_calls"] == 2  # 202 then 200
    assert st["auth"] == "Bearer cap-token"


def test_commit_uses_sdk_transfer_grant_when_offered(fake_hub, tmp_path: Path, monkeypatch) -> None:
    # R2 path (found live in e2e J7): the server returns a scoped temporary
    # credential (transfer_grant) instead of presigned multipart part URLs;
    # the client must SDK-upload and complete with the transfer block.
    monkeypatch.setattr("time.sleep", lambda *_: None)
    _FakeHub.state["grant_mode"] = True
    _FakeHub.state["finalize_calls"] = 1  # finalize 200 immediately

    calls: list[dict] = []

    def _fake_upload(*, file_path, grant, blake3_hex, size_bytes, on_progress=None):
        from gen_worker.s3_transfer import S3TransferResult
        calls.append({"file_path": str(file_path), "bucket": grant.bucket,
                      "key": grant.key, "blake3": blake3_hex, "size": size_bytes})
        return S3TransferResult(bucket=grant.bucket, key=grant.key,
                                size_bytes=size_bytes, blake3=blake3_hex, etag="sdk-etag")

    import gen_worker.s3_transfer as s3t
    monkeypatch.setattr(s3t, "upload_file_with_grant", _fake_upload)

    f = tmp_path / "model.safetensors"
    f.write_bytes(b"\x02" * 48)
    result = _client(fake_hub).commit(
        destination_repo="acme/my-model",
        files=[CommitFile(path="model.safetensors", local_path=f)],
    )
    assert result.uploaded == 1 and result.deduped == 0
    assert len(calls) == 1
    assert calls[0]["bucket"] == "repo-cas"
    assert calls[0]["size"] == 48
    # No raw part PUTs happened; complete carried the transfer block.
    assert "put_bytes" not in _FakeHub.state
    body = _FakeHub.state["complete_bodies"][0]
    assert body["transfer"]["mode"] == "s3_sdk"
    assert body["transfer"]["etag"] == "sdk-etag"
    assert body["transfer"]["blake3"] == blake3_file(f)


def test_dedup_skips_put(fake_hub, tmp_path: Path) -> None:
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"\x01" * 32)
    _FakeHub.state["existing_blobs"] = {blake3_file(f)}
    _FakeHub.state["finalize_calls"] = 1  # finalize returns 200 immediately

    result = _client(fake_hub).commit(
        destination_repo="acme/my-model",
        files=[CommitFile(path="model.safetensors", local_path=f)],
    )
    assert result.deduped == 1 and result.uploaded == 0
    assert "put_bytes" not in _FakeHub.state


def test_destination_must_be_owner_repo(fake_hub) -> None:
    with pytest.raises(HubPublishError, match="owner/repo"):
        _client(fake_hub).commit(
            destination_repo="just-a-name",
            files=[CommitFile(path="x", local_path=Path("/nonexistent"))],
        )


def test_transient_failures_are_retried(fake_hub, tmp_path: Path, monkeypatch) -> None:
    """One 503 on the commit POST, one 500 on a part PUT, one 500 on complete:
    the commit still lands (bounded retries, no restart of the whole job)."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"\x02" * 48)
    _FakeHub.state.update(
        {"fail_commit_posts": 1, "fail_puts": 1, "fail_completes": 1,
         "finalize_calls": 1})

    result = _client(fake_hub).commit(
        destination_repo="acme/my-model",
        files=[CommitFile(path="model.safetensors", local_path=f)],
    )
    assert result.revision_id == "rev-1"
    assert result.uploaded == 1
    # the successful PUT actually carried the bytes
    assert list(_FakeHub.state["put_bytes"].values()) == [b"\x02" * 48]


def test_files_from_tree_skips_hf_cache_junk(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    junk = tmp_path / ".cache" / "huggingface" / "download"
    junk.mkdir(parents=True)
    (junk / "config.json.metadata").write_text("etag")
    (tmp_path / ".cache" / "huggingface" / ".gitignore").write_text("*")
    # a real dotfile at repo root is content, not junk
    (tmp_path / ".gitignore").write_text("logs/")

    paths = [f.path for f in files_from_tree(tmp_path)]
    assert paths == [".gitignore", "config.json"]
