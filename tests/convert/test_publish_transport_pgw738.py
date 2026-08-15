"""pgw#738/pgw#743: the artifact publish must be origin-discriminating,
silence-bounded, typed, and OBSERVABLE.

te#125's edit run was declared dead ~10 min into a publish phase that emitted
zero signals by construction; #743's clone runs were killed FATAL by
proxy-shaped 5xx after ~2 min of bounded retries. These tests drive the real
``HubClient.commit`` path against the fake hub with injected proxy pages,
connection resets, and outage exhaustion.

Red-verified against the pre-fix ``hubio/client.py`` (5-attempt cap, no origin
discrimination): the proxy-404 test failed immediately with
``commit create failed (404)``.

The two publish-observability rows are DELETED (§4.34): they were staged on
``_throttled_part_progress``, a sibling lane's uncommitted work that never
landed, so they had never run anywhere.
"""

from __future__ import annotations

import time
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
    # raising=False: red-verification runs against the pre-fix module,
    # which has no silence window at all.
    monkeypatch.setattr(hub_mod, "_SEND_SILENCE_WINDOW_S", 5.0, raising=False)


def _one_file(tmp_path: Path, name: str = "model.safetensors") -> CommitFile:
    p = tmp_path / name
    p.write_bytes(b"weights-bytes")
    return CommitFile(path=name, local_path=p)


def test_proxy_404_on_publish_declare_is_retried_then_succeeds(
    fake_hub: Any, tmp_path: Path,
) -> None:
    """A proxy answering 404 (hub restarting behind ngrok) is an outage, not
    'route missing'. Pre-fix: instant HubPublishError, paid work discarded."""
    _FakeHub.state["proxy_posts"] = 2
    _FakeHub.state["proxy_status"] = 404
    res = _client(fake_hub).publish_v2(
        release="r1",
        destination_repo="acme/repo", files=[_one_file(tmp_path)])
    assert res.uploaded == 1
    assert res.checkpoint_id


def test_proxy_503_on_publish_is_retried_then_succeeds(
    fake_hub: Any, tmp_path: Path,
) -> None:
    """#743's exact shape: proxy-shaped 503 (HTML body). Pre-fix the 5-attempt
    cap exhausted in ~2 min and classified FATAL at the finish line."""
    _FakeHub.state["proxy_posts"] = 7  # > the old 5-attempt budget
    _FakeHub.state["proxy_status"] = 503
    res = _client(fake_hub).publish_v2(
        release="r1",
        destination_repo="acme/repo", files=[_one_file(tmp_path)])
    assert res.uploaded == 1


def test_hub_origin_404_stays_terminal(fake_hub: Any, tmp_path: Path) -> None:
    """The HUB saying 404 (real error envelope) is definite: no retry storm,
    typed HubPublishError."""
    calls: list[float] = []
    client = _client(fake_hub)
    real_post = hub_mod._http_session().post

    def counting_post(url: str, **kw: Any) -> Any:
        calls.append(time.monotonic())
        return real_post(url.replace("/publishes", "/definitely-missing"), **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hub_mod._http_session(), "post", counting_post)
        with pytest.raises(HubPublishError, match=r"publish declare failed \(404\)"):
            client.publish_v2(destination_repo="acme/repo",
                              files=[_one_file(tmp_path)], release="r1")
    assert len(calls) == 1, "a definite hub 404 must not be retried"


def test_sustained_outage_exhausts_typed_never_raw(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An outage that outlives the silence window fails TYPED (HubPublishError
    carrying the last proxy status), never a raw requests exception."""
    monkeypatch.setattr(hub_mod, "_SEND_SILENCE_WINDOW_S", 0.3)
    _FakeHub.state["proxy_posts"] = 10_000
    _FakeHub.state["proxy_status"] = 503
    with pytest.raises(HubPublishError, match=r"publish declare failed \(503\)"):
        _client(fake_hub).publish_v2(
            release="r1",
            destination_repo="acme/repo", files=[_one_file(tmp_path)])
