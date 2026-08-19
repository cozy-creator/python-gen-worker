"""pgw#1435 / th#2182: the front door is not the hub, on the LAST call too.

Two ingests — `microsoft/TRELLIS.2-4B` (16.2 GB) and `unsloth/Qwen3.6-27B-MTP-GGUF`
(17.9 GB) — plus a third lane's release publish were destroyed in one window by
`publish complete failed (503): <!DOCTYPE html>… ngrok…`, each after every byte
was uploaded, staged and AUDITED. `_post_v2_complete` passed
``definite=lambda resp: True``, so a proxy's offline page counted as the hub's
verdict and the silence-bounded retry that wraps every other call never ran.

Three behaviours, and the third is why the escape hatch was there in the first
place — a v2 refusal is a PROJECTION, not an error envelope, and must stay
terminal or a repudiation turns into a later 409 `publish_repudiated` with the
cause gone (pgw#743's defect, one route over).

Red-verified against the pre-fix module: reinstating ``definite=lambda resp: True``
fails the first two (the transient case raises `publish complete failed (503)`
with the HTML page in the message; the classified case has code "" and the page
in `str(exc)`), and leaves the third green.
"""

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
    """The measured incident, made to end the other way.

    Everything is staged; only `complete` meets the offline page, three times.
    The publish must land, and it must land on the SAME session — a second
    declare would mean the staged bytes were abandoned, which is the cost this
    issue exists to stop.
    """
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
    """A front door that stays down still must not put HTML in `cause`.

    The operator-visible failure carries a token to group by and the hub's
    absence as a fact — not 690 bytes of someone else's web page.
    """
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
    """The reason the escape hatch existed, kept working.

    A v2 refusal answers with the th#1301 projection and no `{"error": ...}`
    envelope. The generic shape heuristic reads that as proxy-shaped, so
    without an explicit projection arm the fix would retry a repudiation and
    report the 409 that follows it instead of the reason.
    """
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
