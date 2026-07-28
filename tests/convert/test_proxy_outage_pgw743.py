"""pgw#743: a proxy-shaped answer is never the publisher's verdict.

Two independent clones downloaded 53 GiB over ~58 minutes and were both killed
at the FIRST upload by ``upload complete failed (503) ... <!DOCTYPE html>`` —
ngrok's page while the hub container was being rebuilt, classified FATAL. The
guard is behavioural: an outage that ends must let the job finish, a real hub
refusal must still end it immediately, and neither may depend on WHICH status
the proxy chose to put on its HTML.

Red-verified against the pre-fix module (404-only discrimination): the 403 and
503 cases raise ``commit create failed`` instead of publishing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gen_worker.convert import hub as hub_mod
from gen_worker.convert.hub import CommitFile, HubPublishError
from gen_worker.convert.keepalive import HubKeepalive

from fake_hub import _FakeHub, _client


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hub_mod, "_RETRY_BASE_DELAY_S", 0.01)
    monkeypatch.setattr(hub_mod, "_RETRY_MAX_DELAY_S", 0.05)
    monkeypatch.setattr(hub_mod, "_SEND_SILENCE_WINDOW_S", 5.0, raising=False)


def _one_file(tmp_path: Path) -> CommitFile:
    p = tmp_path / "model.safetensors"
    p.write_bytes(b"weights-bytes")
    return CommitFile(path="model.safetensors", local_path=p)


@pytest.mark.parametrize("proxy_status", [503, 502, 403, 404])
def test_proxy_outage_of_any_status_is_ridden_out(
    fake_hub: Any, tmp_path: Path, proxy_status: int,
) -> None:
    """The publish survives a proxy outage and lands once the hub is back.

    Eight refusals is past the old five-attempt cap on purpose: the cap, not
    the status, is what discarded the paid download.
    """
    _FakeHub.state["proxy_posts"] = 8
    _FakeHub.state["proxy_status"] = proxy_status

    res = _client(fake_hub).commit(
        destination_repo="acme/repo", files=[_one_file(tmp_path)])

    assert res.uploaded == 1
    assert res.checkpoint_id == "blake3:abc"
    assert _FakeHub.state["proxy_posts"] == 0


def test_hub_refusal_stays_terminal_and_is_not_retried(
    fake_hub: Any, tmp_path: Path,
) -> None:
    """A 404 carrying the hub's own error envelope is a VERDICT.

    Biasing toward retry must not turn real refusals into retry loops: one
    attempt, then a typed failure.
    """
    posts: list[str] = []
    session = hub_mod._http_session()
    real_post = session.post

    def counting_post(url: str, **kw: Any) -> Any:
        posts.append(url)
        return real_post(url.replace("/commits", "/no-such-route"), **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(session, "post", counting_post)
        with pytest.raises(HubPublishError, match=r"commit create failed \(404\)"):
            _client(fake_hub).commit(
                destination_repo="acme/repo", files=[_one_file(tmp_path)])

    assert len(posts) == 1, f"a definite hub 404 must not be retried: {posts}"


def test_outage_outliving_the_window_fails_typed(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patience is bounded: an outage that never ends still fails, typed, with
    the last proxy status visible for forensics."""
    monkeypatch.setattr(hub_mod, "_SEND_SILENCE_WINDOW_S", 0.3)
    _FakeHub.state["proxy_posts"] = 10_000
    _FakeHub.state["proxy_status"] = 503

    with pytest.raises(HubPublishError, match=r"commit create failed \(503\)"):
        _client(fake_hub).commit(
            destination_repo="acme/repo", files=[_one_file(tmp_path)])


# ---- the keepalive half: the hub-silent hour is what set the trap ----


def test_keepalive_dates_the_outage_and_never_raises(fake_hub: Any) -> None:
    """The probe reports reachability and records how long the hub was gone.

    It must never raise: a job is not allowed to die of its own liveness
    probe, whatever the probe finds.
    """
    clock = {"t": 0.0}
    lines: list[str] = []
    ka = HubKeepalive(
        _client(fake_hub), "/api/v1/repos/acme/repo",
        log=lines.append, now=lambda: clock["t"])

    assert ka.probe() is True

    _FakeHub.state["proxy_gets"] = 2  # the tunnel loses its backend
    clock["t"] = 100.0
    assert ka.probe() is False
    clock["t"] = 340.0
    assert ka.probe() is False

    assert ka.probes == 3
    assert ka.unreachable_since == 100.0
    assert any("hub NOT answering" in line for line in lines)
    assert sum("hub NOT answering" in line for line in lines) == 1, \
        "one line per transition, not one per probe"


def test_keepalive_measures_recovery(fake_hub: Any) -> None:
    clock = {"t": 0.0}
    lines: list[str] = []
    ka = HubKeepalive(
        _client(fake_hub), "/api/v1/repos/acme/repo",
        log=lines.append, now=lambda: clock["t"])

    ka._record(False, "probe failed")
    clock["t"] = 1500.0
    ka._record(True, "status=200")

    assert ka.longest_outage_s == 1500.0
    assert ka.unreachable_since is None
    assert any("reachable again after 1500s" in line for line in lines)
