"""HubClient against a threaded fake of tensorhub's /commits API.

Exercises the real HTTP code path: POST /commits (presign), part PUTs,
per-upload complete, finalize (202 poll -> 200), blake3 dedup skips, and
bounded retries on transient failures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker.convert.hub import CommitFile, HubPublishError, blake3_file, files_from_tree

from fake_hub import _FakeHub, _client


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
        provenance={"upstream_revision": "abc123",
                    "quantization_method": "",  # empty values are dropped
                    "step_number": 0},
    )
    assert result.revision_id == "rev-1"
    # THE queryable id: minted at finalize from the snapshot manifest.
    assert result.checkpoint_id == "blake3:abc"
    assert result.uploaded == 2 and result.deduped == 0

    st = _FakeHub.state
    req = st["commit_request"]
    assert req["mode"] == "replace"
    assert req["tags"] == [{"tag": "prod"}]
    assert req["flavor"] == "bf16"
    # th#606: only non-empty worker-addable provenance fields are sent;
    # the old worker-declared lineage contract is gone.
    assert req["provenance"] == {"upstream_revision": "abc123"}
    assert "lineage" not in req and "auto_create_external_parent" not in req
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


def test_complete_polls_through_upload_complete_in_progress_race(
    fake_hub, tmp_path: Path, monkeypatch,
) -> None:
    """Regression for e2e tracker #110: /complete verifies large single
    files synchronously and can outlast the client's timeout, so a retry can
    race the still-running first attempt into 409 upload_complete_in_progress.
    That must be polled through (idempotent once finalized), not treated as
    a fatal error that aborts the whole commit."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    _FakeHub.state["complete_race_count"] = 2
    _FakeHub.state["finalize_calls"] = 1  # finalize returns 200 immediately

    f = tmp_path / "model.safetensors"
    f.write_bytes(b"\x03" * 48)
    result = _client(fake_hub).commit(
        destination_repo="acme/my-model",
        files=[CommitFile(path="model.safetensors", local_path=f)],
    )
    assert result.revision_id == "rev-1"
    assert result.uploaded == 1
    assert len(_FakeHub.state["complete_race_polls"]) == 2
    assert list(_FakeHub.state["put_bytes"].values()) == [b"\x03" * 48]


def test_complete_polls_through_race_on_sdk_transfer_grant_path(
    fake_hub, tmp_path: Path, monkeypatch,
) -> None:
    """Same race, but on the R2 SDK transfer-grant complete path (the shape
    actually used by clone-huggingface/clone-civitai mirrors)."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    _FakeHub.state["grant_mode"] = True
    _FakeHub.state["complete_race_count"] = 1
    _FakeHub.state["finalize_calls"] = 1

    def _fake_upload(*, file_path, grant, blake3_hex, size_bytes, on_progress=None):
        from gen_worker.s3_transfer import S3TransferResult
        return S3TransferResult(bucket=grant.bucket, key=grant.key,
                                size_bytes=size_bytes, blake3=blake3_hex, etag="sdk-etag")

    import gen_worker.s3_transfer as s3t
    monkeypatch.setattr(s3t, "upload_file_with_grant", _fake_upload)

    f = tmp_path / "model.safetensors"
    f.write_bytes(b"\x04" * 48)
    result = _client(fake_hub).commit(
        destination_repo="acme/my-model",
        files=[CommitFile(path="model.safetensors", local_path=f)],
    )
    assert result.uploaded == 1
    assert len(_FakeHub.state["complete_race_polls"]) == 1
    assert _FakeHub.state["complete_bodies"][0]["transfer"]["etag"] == "sdk-etag"


def test_complete_gives_up_after_deadline_if_race_never_resolves(
    fake_hub, tmp_path: Path, monkeypatch,
) -> None:
    """A genuinely stuck server (never finalizes) must still fail eventually
    rather than polling forever."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    monkeypatch.setattr("gen_worker.convert.hub._COMPLETE_NETWORK_MAX_WAIT_S", 0.0)
    _FakeHub.state["complete_race_count"] = 10_000

    f = tmp_path / "model.safetensors"
    f.write_bytes(b"\x05" * 48)
    with pytest.raises(HubPublishError, match="upload complete failed"):
        _client(fake_hub).commit(
            destination_repo="acme/my-model",
            files=[CommitFile(path="model.safetensors", local_path=f)],
        )


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


def test_complete_repost_through_network_severed_attempts(monkeypatch) -> None:
    """te#44 J9 runs 7+8: the idle multi-minute /complete verify gets severed
    by middleboxes; the client must re-POST (idempotent) instead of failing
    the commit after the quick generic retries are exhausted."""
    from gen_worker.convert.hub import HubClient, HubPublishError

    monkeypatch.setattr("time.sleep", lambda *_: None)
    client = HubClient(base_url="http://hub", token="t", owner="acme")
    calls = {"n": 0}

    class _OK:
        status_code = 200
        text = "{}"

        @staticmethod
        def json():
            return {}

    def _post(path, payload=None, *, timeout=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise HubPublishError(f"POST {path} failed (network): severed")
        return _OK()

    monkeypatch.setattr(client, "_post", _post)
    resp = client._post_complete("/complete", {})
    assert resp.status_code == 200 and calls["n"] == 3


def test_complete_network_severed_raises_after_deadline(monkeypatch) -> None:
    from gen_worker.convert.hub import HubClient, HubPublishError

    monkeypatch.setattr("time.sleep", lambda *_: None)
    monkeypatch.setattr("gen_worker.convert.hub._COMPLETE_NETWORK_MAX_WAIT_S", 0.0)
    client = HubClient(base_url="http://hub", token="t", owner="acme")

    def _post(path, payload=None, *, timeout=None):
        raise HubPublishError("POST /complete failed (network): severed")

    monkeypatch.setattr(client, "_post", _post)
    with pytest.raises(HubPublishError, match="network"):
        client._post_complete("/complete", {})
