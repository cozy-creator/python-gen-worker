"""HashRepo journals transfer identity; worker policy retains produced trees."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fake_hub import _client, _FakeHub

from gen_worker.convert.clone import _reusable_flavor_tree
from gen_worker.hubio.client import CommitFile, HubPublishError
from gen_worker.hubio.publish_state import JOURNAL_NAME, ProducerRecovery


def _file(root: Path, data: bytes = b"artifact") -> CommitFile:
    path = root / "weights.safetensors"
    path.write_bytes(data)
    return CommitFile(path=path.name, local_path=path, size_bytes=len(data))


def _sessions(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return list(json.loads(path.read_text()).get("sessions") or [])


def test_transport_failure_keeps_hashrepo_session_and_staged_bytes(
    fake_hub, tmp_path: Path
) -> None:
    _FakeHub.state["fail_puts"] = 999
    journal = tmp_path / JOURNAL_NAME
    with pytest.raises(HubPublishError, match="failed to upload"):
        _client(fake_hub).publish_v2(
            release="r1",
            destination_repo="acme/model",
            files=[_file(tmp_path)],
            journal_path=journal,
        )
    assert len(_sessions(journal)) == 1
    assert _FakeHub.state.get("aborts", []) == []


def test_retry_replans_the_same_hashrepo_session(fake_hub, tmp_path: Path) -> None:
    journal = tmp_path / JOURNAL_NAME
    file = _file(tmp_path)
    _FakeHub.state["fail_puts"] = 999
    with pytest.raises(HubPublishError):
        _client(fake_hub).publish_v2(
            release="r1",
            destination_repo="acme/model", files=[file], journal_path=journal
        )
    (session,) = _sessions(journal)

    _FakeHub.state["fail_puts"] = 0
    result = _client(fake_hub).publish_v2(
        release="r1",
        destination_repo="acme/model", files=[file], journal_path=journal
    )
    assert result.revision_id == session["session_id"]
    assert list(_FakeHub.state["publishes"]) == [session["session_id"]]
    assert _sessions(journal) == []


def test_terminal_refusal_aborts_and_clears_both_records(
    fake_hub, tmp_path: Path
) -> None:
    _FakeHub.state["complete_failure"] = {
        "code": "invalid_manifest_for_kind",
        "retryable": False,
        "message": "bad artifact",
    }
    journal = tmp_path / JOURNAL_NAME
    with pytest.raises(HubPublishError) as caught:
        _client(fake_hub).publish_v2(
            release="r1",
            destination_repo="acme/model",
            files=[_file(tmp_path)],
            journal_path=journal,
            journal_state={"spec_label": "fp8"},
        )
    assert caught.value.retryable is False
    assert _sessions(journal) == []
    assert ProducerRecovery(journal).count() == 0
    assert _FakeHub.state.get("aborted_publishes") == ["pub-1"]


def test_retryable_refusal_retains_producer_recovery_policy(
    fake_hub, tmp_path: Path
) -> None:
    _FakeHub.state["complete_failure"] = {
        "code": "verification_backlog",
        "retryable": True,
        "message": "try again",
    }
    journal = tmp_path / JOURNAL_NAME
    state = {"spec_label": "fp8", "tree": str(tmp_path), "attrs": {"dtype": "fp8"}}
    with pytest.raises(HubPublishError):
        _client(fake_hub).publish_v2(
            release="r1",
            destination_repo="acme/model",
            files=[_file(tmp_path)],
            journal_path=journal,
            journal_state=state,
        )
    recovered = ProducerRecovery(journal).find(spec_label="fp8", tree=str(tmp_path))
    assert recovered is not None
    assert recovered.producer_state["attrs"] == {"dtype": "fp8"}


def test_different_artifact_never_adopts_another_session(
    fake_hub, tmp_path: Path
) -> None:
    journal = tmp_path / JOURNAL_NAME
    _FakeHub.state["fail_puts"] = 999
    with pytest.raises(HubPublishError):
        _client(fake_hub).publish_v2(
            release="r1",
            destination_repo="acme/model",
            files=[_file(tmp_path, b"first")],
            journal_path=journal,
        )
    first = _sessions(journal)[0]["session_id"]
    _FakeHub.state["fail_puts"] = 0
    result = _client(fake_hub).publish_v2(
        release="r1",
        destination_repo="acme/model",
        files=[_file(tmp_path, b"second")],
        journal_path=journal,
    )
    assert result.revision_id != first


def test_different_declaration_never_adopts_another_session(
    fake_hub, tmp_path: Path
) -> None:
    journal = tmp_path / JOURNAL_NAME
    file = _file(tmp_path)
    _FakeHub.state["fail_puts"] = 999
    with pytest.raises(HubPublishError):
        _client(fake_hub).publish_v2(
            release="r1",
            destination_repo="acme/model",
            files=[file],
            journal_path=journal,
        )
    first = _sessions(journal)[0]["session_id"]

    _FakeHub.state["fail_puts"] = 0
    result = _client(fake_hub).publish_v2(
        release="r2",
        destination_repo="acme/model",
        files=[file],
        journal_path=journal,
    )
    assert result.revision_id != first


def test_lost_complete_success_reconciles_through_idempotent_complete(
    fake_hub, tmp_path: Path
) -> None:
    _FakeHub.state["lose_complete_responses"] = 1
    result = _client(fake_hub).publish_v2(
        release="r1",
        destination_repo="acme/model",
        files=[_file(tmp_path)],
        journal_path=tmp_path / JOURNAL_NAME,
    )
    assert result.checkpoint_id == "sha256:" + "ab" * 32
    assert _FakeHub.state["complete_attempts"] == ["pub-1", "pub-1"]


def test_publish_state_files_are_not_repository_content(tmp_path: Path) -> None:
    from gen_worker.hubio.client import files_from_tree
    from gen_worker.hubio.publish_state import STATE_NAME

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / JOURNAL_NAME).write_text("{}")
    (tmp_path / STATE_NAME).write_text("{}")
    (tmp_path / f".{JOURNAL_NAME}.lock").write_text("")
    (tmp_path / f".{STATE_NAME}.lock").write_text("")
    assert [file.path for file in files_from_tree(tmp_path)] == ["config.json"]


def test_valid_retained_tree_reuses_the_recorded_session(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    tree = tmp_path / "fp8"
    tree.mkdir()
    _file(tree)
    ProducerRecovery(tmp_path / JOURNAL_NAME).record(
        "session-valid",
        paths=["weights.safetensors"],
        producer_state={
            "spec_label": "fp8",
            "tree": str(tree),
            "attrs": {"dtype": "float8_e4m3fn"},
        },
    )

    with caplog.at_level("WARNING"):
        attrs = _reusable_flavor_tree(tmp_path, "fp8", tree)

    assert attrs == {"dtype": "float8_e4m3fn"}
    assert "session-valid" in caplog.text


def test_retained_tree_with_a_different_file_set_rebuilds(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    tree = tmp_path / "fp8"
    tree.mkdir()
    _file(tree)
    ProducerRecovery(tmp_path / JOURNAL_NAME).record(
        "session-mismatch",
        paths=["weights.safetensors", "missing.json"],
        producer_state={
            "spec_label": "fp8",
            "tree": str(tree),
            "attrs": {"dtype": "float8_e4m3fn"},
        },
    )

    with caplog.at_level("INFO"):
        attrs = _reusable_flavor_tree(tmp_path, "fp8", tree)

    assert attrs is None
    assert "session-mismatch" in caplog.text
