"""gw#462 publish resilience against the fake tensorhub (real HTTP codepaths).

The J24 qwen postmortem scenarios: a lost staged object must cost ONE file's
re-upload (never the job); transient finalize 5xx must retry through; a single
failing part must retry only that part.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker.convert.hub import (
    _REUPLOAD_ATTEMPTS,
    HubPublishError,
    files_from_tree,
)

from fake_hub import _FakeHub, _client


def _tree(tmp_path: Path) -> Path:
    (tmp_path / "config.json").write_text('{"a":1}')
    (tmp_path / "shard-00004.safetensors").write_bytes(b"\x04" * 96)
    return tmp_path


def test_staging_missing_reuploads_only_that_file(fake_hub, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)
    _FakeHub.state["staging_missing"] = {"shard-00004.safetensors": 1}

    result = _client(fake_hub).commit(
        destination_repo="acme/qwen-image",
        files=files_from_tree(_tree(tmp_path)),
    )
    assert result.uploaded == 2

    st = _FakeHub.state
    # Exactly one re-open, for the poisoned file only.
    assert st["reopens"] == ["shard-00004.safetensors"]
    # The shard's bytes were PUT twice (original + re-upload), the other file once.
    puts_by_uid = {p: n for p, n in st["put_counts"].items()}
    assert sum(n for p, n in puts_by_uid.items() if "/re-1/" in p) == 1
    assert sum(puts_by_uid.values()) == 3
    # The whole commit still finalized once.
    assert st["finalize_calls"] >= 2


def test_staging_missing_is_bounded_then_typed(fake_hub, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)
    _FakeHub.state["staging_missing"] = {"shard-00004.safetensors": 10_000}

    with pytest.raises(HubPublishError, match=r"staged bytes lost server-side"):
        _client(fake_hub).commit(
            destination_repo="acme/qwen-image",
            files=files_from_tree(_tree(tmp_path)),
        )
    # Bounded: one original attempt + _REUPLOAD_ATTEMPTS re-opens, then typed failure.
    assert _FakeHub.state["reopen_count"] == _REUPLOAD_ATTEMPTS


def test_staging_missing_reopen_dedup_hit_short_circuits(fake_hub, tmp_path: Path, monkeypatch) -> None:
    """If the blob landed in CAS between the loss and the re-open, the re-open
    answers exists:true and no bytes move."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    st = _FakeHub.state
    st["staging_missing"] = {"shard-00004.safetensors": 1}
    st["reopen_dedup"] = True

    result = _client(fake_hub).commit(
        destination_repo="acme/qwen-image",
        files=files_from_tree(_tree(tmp_path)),
    )
    assert result.uploaded == 2
    assert st["reopens"] == ["shard-00004.safetensors"]
    # No re-upload bytes: only the two original PUTs happened.
    assert sum(st["put_counts"].values()) == 2


def test_finalize_503s_then_succeeds(fake_hub, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)
    _FakeHub.state["fail_finalizes"] = 3

    result = _client(fake_hub).commit(
        destination_repo="acme/qwen-image",
        files=files_from_tree(_tree(tmp_path)),
    )
    assert result.checkpoint_id == "blake3:abc"
    # 3 x 503 (retried inside _send_with_retries) then 200.
    assert _FakeHub.state["finalize_calls"] == 4


def test_single_failing_part_retries_only_that_part(fake_hub, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)
    (tmp_path / "weights.bin").write_bytes(b"\xab" * 90)
    _FakeHub.state["force_parts"] = 3  # 3 parts of 30 bytes

    client = _client(fake_hub)
    # Prime a commit to learn the part URLs, then fail part #2 once on a
    # fresh state — simplest deterministic targeting: the fake's URLs are
    # stable (/put/up-0/<n>) for a single-add commit.
    _FakeHub.state["fail_put_paths"] = {"/put/up-0/2": 1}

    result = client.commit(
        destination_repo="acme/qwen-image",
        files=files_from_tree(tmp_path),
    )
    assert result.uploaded == 1
    counts = _FakeHub.state["put_counts"]
    assert counts["/put/up-0/2"] == 2  # failed once, retried once
    assert counts["/put/up-0/1"] == 1
    assert counts["/put/up-0/3"] == 1
