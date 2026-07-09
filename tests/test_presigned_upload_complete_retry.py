"""Regression tests for the /complete false-negative-on-retry bug (e2e
tracker #110): tensorhub verifies large single files synchronously and can
outlast an intermediary's timeout, so a client retry can race the still-running
first attempt and get 409 upload_complete_in_progress. The client must poll
that specific condition to the server's actual `Finalized` outcome instead of
treating it as fatal (which previously aborted the whole commit)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from gen_worker.presigned_upload import (
    _complete_upload_session,
    _error_code_of,
    _poll_until_finalized,
)
from gen_worker.api.errors import ArtifactTransferError, CanceledError


class _FakeSession:
    """Session double: the poll/complete helpers now take a per-save
    requests.Session; only .post is used here."""

    def __init__(self, post):
        self.post = post


class _FakeResponse:
    def __init__(self, status_code: int, body: dict | None = None, text: str | None = None):
        self.status_code = status_code
        self._body = body
        self.text = text if text is not None else (json.dumps(body) if body is not None else "")

    def json(self):
        if self._body is None:
            raise ValueError("no json body")
        return self._body


def test_error_code_of_extracts_structured_error_code():
    resp = _FakeResponse(409, {"error": {"code": "upload_complete_in_progress", "message": "x"}})
    assert _error_code_of(resp) == "upload_complete_in_progress"


def test_error_code_of_returns_empty_for_non_structured_body():
    assert _error_code_of(_FakeResponse(500, text="internal error")) == ""
    assert _error_code_of(_FakeResponse(200, {"ok": True})) == ""


def test_complete_upload_session_polls_through_upload_complete_in_progress(monkeypatch):
    """First attempt races a concurrent finalize (409); the poll loop must
    keep re-POSTing /complete (idempotent per tensorhub's `sess.Finalized`
    fast path) until it succeeds, rather than raising immediately."""
    responses = [
        _FakeResponse(409, {"error": {"code": "upload_complete_in_progress"}}),
        _FakeResponse(409, {"error": {"code": "upload_complete_in_progress"}}),
        _FakeResponse(200, {"destination_repo": "tensorhub/sdxl-illustrious", "published": []}),
    ]
    calls = {"n": 0}

    def fake_post(*_a, **_kw):
        resp = responses[calls["n"]]
        calls["n"] += 1
        return resp

    with patch("gen_worker.presigned_upload.time.sleep"):
        result = _complete_upload_session(
            complete_url="https://tensorhub.test/complete",
            headers={"Authorization": "Bearer x"},
            payload={"transfer": {"mode": "s3_sdk"}},
            cancel_check=None,
            session=_FakeSession(fake_post),
        )
    assert result == {"destination_repo": "tensorhub/sdxl-illustrious", "published": []}
    assert calls["n"] == 3


def test_poll_until_finalized_gives_up_after_deadline(monkeypatch):
    """A genuinely stuck server (never finalizes) must eventually raise a
    retryable ArtifactTransferError rather than polling forever."""
    monkeypatch.setattr(
        "gen_worker.presigned_upload._COMPLETE_IN_PROGRESS_MAX_WAIT_S", 0.01,
    )
    always_409 = _FakeResponse(409, {"error": {"code": "upload_complete_in_progress"}})
    with patch("gen_worker.presigned_upload.time.sleep"):
        with pytest.raises(ArtifactTransferError) as exc_info:
            _poll_until_finalized(
                complete_url="https://tensorhub.test/complete",
                complete_headers={},
                payload={},
                cancel_check=None,
                session=_FakeSession(lambda *a, **kw: always_409),
            )
    assert exc_info.value.retryable is True


def test_poll_until_finalized_respects_cancel_check():
    with pytest.raises(CanceledError):
        _poll_until_finalized(
            complete_url="https://tensorhub.test/complete",
            complete_headers={},
            payload={},
            cancel_check=lambda: True,
            session=_FakeSession(lambda *a, **kw: None),
        )


def test_complete_upload_session_other_409_is_not_polled():
    """A different 409 (e.g. the session was aborted) is a real terminal
    error, not the in-progress race — must not enter the poll loop."""
    aborted = _FakeResponse(409, {"error": {"code": "upload_session_aborted"}})
    with pytest.raises(ArtifactTransferError) as exc_info:
        _complete_upload_session(
            complete_url="https://tensorhub.test/complete",
            headers={},
            payload={},
            cancel_check=None,
            session=_FakeSession(lambda *a, **kw: aborted),
        )
    assert exc_info.value.retryable is False
    assert exc_info.value.status_code == 409

def test_complete_upload_session_reposts_through_network_severed_attempts():
    """te#44 J9 runs 7+8: the idle multi-minute /complete verify gets severed
    by middleboxes; the client must re-POST (idempotent once finalized) on the
    long network deadline instead of failing the save/commit after a handful
    of quick generic retries."""
    import requests

    calls = {"n": 0}

    def fake_post(*_a, **_kw):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise requests.ConnectionError("severed mid-verify")
        return _FakeResponse(200, {"ok": True})

    with patch("gen_worker.presigned_upload.time.sleep"):
        result = _complete_upload_session(
            complete_url="https://tensorhub.test/complete",
            headers={},
            payload={"parts": []},
            cancel_check=None,
            session=_FakeSession(fake_post),
        )
    assert result == {"ok": True}
    assert calls["n"] == 3


def test_complete_upload_session_network_severed_raises_after_deadline(monkeypatch):
    import requests

    monkeypatch.setattr("gen_worker.presigned_upload._COMPLETE_NETWORK_MAX_WAIT_S", 0.0)

    def fake_post(*_a, **_kw):
        raise requests.ConnectionError("severed")

    with patch("gen_worker.presigned_upload.time.sleep"):
        with pytest.raises(ArtifactTransferError, match="finalize request failed"):
            _complete_upload_session(
                complete_url="https://tensorhub.test/complete",
                headers={},
                payload={"parts": []},
                cancel_check=None,
                session=_FakeSession(fake_post),
            )
