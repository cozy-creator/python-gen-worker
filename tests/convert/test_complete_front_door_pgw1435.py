from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gen_worker.hubio import client as hub_mod
from gen_worker.hubio.client import CommitFile, HubPublishError

from fake_hub import _FakeHub, _client


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hub_mod, "_RETRY_BASE_DELAY_S", 0.01)
    monkeypatch.setattr(hub_mod, "_RETRY_MAX_DELAY_S", 0.05)
    monkeypatch.setattr(hub_mod, "_COMPLETE_SILENCE_WINDOW_S", 2.0)


def _one_file(tmp_path: Path) -> CommitFile:
    p = tmp_path / "model.safetensors"
    p.write_bytes(b"weights-bytes")
    return CommitFile(path="model.safetensors", local_path=p)


def test_transient_front_door_503_on_complete_does_not_destroy_the_publish(
    fake_hub: Any, tmp_path: Path,
) -> None:
    """The measured incident, made to end the other way."""
    _FakeHub.state["proxy_completes"] = 3
    _FakeHub.state["proxy_status"] = 503

    res = _client(fake_hub).publish_v2(
        release="r1", destination_repo="acme/repo", files=[_one_file(tmp_path)])

    assert res.checkpoint_id
    attempts = _FakeHub.state["complete_attempts"]
    assert len(attempts) == 4, attempts
    assert len(set(attempts)) == 1, "retried a DIFFERENT session — bytes abandoned"
    assert _FakeHub.state["proxy_completes"] == 0


def test_front_door_that_never_recovers_is_classified_never_pasted(
    fake_hub: Any, tmp_path: Path,
) -> None:
    """A front door that stays down still must not put HTML in `cause`."""
    _FakeHub.state["proxy_completes"] = 10_000
    _FakeHub.state["proxy_status"] = 503

    with pytest.raises(HubPublishError) as caught:
        _client(fake_hub).publish_v2(
            release="r1", destination_repo="acme/repo", files=[_one_file(tmp_path)])

    exc = caught.value
    assert exc.code == "front_door_unavailable"
    assert exc.retryable is True, "the bytes are staged; a new pod is not the remedy"
    assert exc.status == 503
    message = str(exc)
    assert "DOCTYPE" not in message and "<html" not in message, message
    assert "HTTP 503, text/html" in message
    assert len(_FakeHub.state["complete_attempts"]) > 1, "never retried at all"


def test_typed_repudiation_stays_terminal_on_one_attempt(
    fake_hub: Any, tmp_path: Path,
) -> None:
    """The reason the escape hatch existed, kept working."""
    _FakeHub.state["complete_failure"] = {
        "code": "audit_findings", "retryable": False,
        "message": "the artifact did not pass the pre-publication audit",
    }

    with pytest.raises(HubPublishError) as caught:
        _client(fake_hub).publish_v2(
            release="r1", destination_repo="acme/repo", files=[_one_file(tmp_path)])

    exc = caught.value
    assert exc.code == "audit_findings"
    assert exc.retryable is False
    assert len(_FakeHub.state["complete_attempts"]) == 1, \
        _FakeHub.state["complete_attempts"]
