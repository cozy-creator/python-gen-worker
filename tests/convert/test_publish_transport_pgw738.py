"""pgw#738/pgw#743: the artifact publish must be origin-discriminating,
silence-bounded, typed, and OBSERVABLE.

te#125's edit run was declared dead ~10 min into a publish phase that emitted
zero signals by construction; #743's clone runs were killed FATAL by
proxy-shaped 5xx after ~2 min of bounded retries. These tests drive the real
``HubClient.commit`` / ``publish_flavors`` path against the fake hub with
injected proxy pages, connection resets, and outage exhaustion.

Red-verified against the pre-fix ``convert/hub.py`` (5-attempt cap, no origin
discrimination): the proxy-404 test failed immediately with
``commit create failed (404)`` and the observability test found no log lines.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from gen_worker.convert import hub as hub_mod
from gen_worker.convert.hub import CommitFile, HubPublishError
from gen_worker.convert.produced import ProducedFlavor
from gen_worker.convert.publish import publish_flavors

from fake_hub import _FakeHub, _client

# pgw#797 adjudication (test-health audit): six of these eight cases pass
# against HEAD and pin real, landed behaviour — this file is NOT a dead-lane
# leftover. The two observability cases assert logging that lives only in a
# sibling lane's uncommitted `convert/publish.py` (HEAD has no
# `_throttled_part_progress`), so they are marked staged rather than left red.
# REMOVE THE MARK when that publish.py work lands.
_PUBLISH_OBSERVABILITY_LANDED = hasattr(
    __import__("gen_worker.convert.publish", fromlist=["publish"]),
    "_throttled_part_progress",
)
_needs_publish_observability = pytest.mark.skipif(
    not _PUBLISH_OBSERVABILITY_LANDED,
    reason="pgw#738 STAGED: publish-progress observability "
           "(_throttled_part_progress) has not landed in convert/publish.py",
)


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


def test_proxy_404_on_commit_create_is_retried_then_succeeds(
    fake_hub: Any, tmp_path: Path,
) -> None:
    """A proxy answering 404 (hub restarting behind ngrok) is an outage, not
    'route missing'. Pre-fix: instant HubPublishError, paid work discarded."""
    _FakeHub.state["proxy_posts"] = 2
    _FakeHub.state["proxy_status"] = 404
    res = _client(fake_hub).commit(
        destination_repo="acme/repo", files=[_one_file(tmp_path)])
    assert res.uploaded == 1
    assert res.checkpoint_id == "blake3:abc"


def test_proxy_503_on_commit_is_retried_then_succeeds(
    fake_hub: Any, tmp_path: Path,
) -> None:
    """#743's exact shape: proxy-shaped 503 (HTML body). Pre-fix the 5-attempt
    cap exhausted in ~2 min and classified FATAL at the finish line."""
    _FakeHub.state["proxy_posts"] = 7  # > the old 5-attempt budget
    _FakeHub.state["proxy_status"] = 503
    res = _client(fake_hub).commit(
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
        return real_post(url.replace("/commits", "/definitely-missing"), **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hub_mod._http_session(), "post", counting_post)
        with pytest.raises(HubPublishError, match=r"commit create failed \(404\)"):
            client.commit(destination_repo="acme/repo",
                          files=[_one_file(tmp_path)])
    assert len(calls) == 1, "a definite hub 404 must not be retried"


def test_connection_reset_mid_part_put_recovers(
    fake_hub: Any, tmp_path: Path,
) -> None:
    """The peer severs the socket mid-PUT (tunnel drop): retry, then land."""
    _FakeHub.state["reset_puts"] = 2
    res = _client(fake_hub).commit(
        destination_repo="acme/repo", files=[_one_file(tmp_path)])
    assert res.uploaded == 1
    assert _FakeHub.state.get("completed"), "the upload must complete"


def test_sustained_outage_exhausts_typed_never_raw(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An outage that outlives the silence window fails TYPED (HubPublishError
    carrying the last proxy status), never a raw requests exception."""
    monkeypatch.setattr(hub_mod, "_SEND_SILENCE_WINDOW_S", 0.3)
    _FakeHub.state["proxy_posts"] = 10_000
    _FakeHub.state["proxy_status"] = 503
    with pytest.raises(HubPublishError, match=r"commit create failed \(503\)"):
        _client(fake_hub).commit(
            destination_repo="acme/repo", files=[_one_file(tmp_path)])


class _CtxDouble:
    """Just enough ctx for publish_flavors: hub identity + a log capture."""

    def __init__(self, base_url: str) -> None:
        self._file_api_base_url = base_url
        self._worker_capability_token = "cap-token"
        self.owner = "acme"
        self.lines: list[str] = []

    def log(self, message: str, **fields: Any) -> None:
        self.lines.append(message)


@_needs_publish_observability
def test_publish_flavors_is_observable(fake_hub: Any, tmp_path: Path) -> None:
    """pgw#738: the publish phase must EMIT — phase transitions and progress
    ride ctx.log (the channel watchdogs read). Pre-fix: zero lines, and the
    dead-signature (gpu idle + log silence) killed a possibly-live publish."""
    ctx = _CtxDouble(f"http://127.0.0.1:{fake_hub.server_port}")
    (tmp_path / "sub").mkdir()
    p = tmp_path / "sub" / "diffusion.safetensors"
    p.write_bytes(b"q" * 1024)
    results = publish_flavors(
        ctx,
        [ProducedFlavor(path=str(tmp_path / "sub"), flavor="fp8-w8a8")],
        destination_repo="acme/quant",
    )
    assert results and results[0].uploaded == 1
    joined = "\n".join(ctx.lines)
    assert "publish: committing flavor fp8-w8a8" in joined
    assert "publish: uploading parts=" in joined
    assert "publish: flavor fp8-w8a8 committed" in joined
    assert "checkpoint=blake3:abc" in joined


def test_sdk_grant_lane_forwards_progress(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The R2 SDK lane (boto3) was progress-BLIND. The grant entry must now
    forward byte progress into part_progress."""
    import gen_worker.s3_transfer as s3t

    seen: list[tuple[int, int, int]] = []

    def fake_upload(*, file_path: Any, grant: Any, blake3_hex: str,
                    size_bytes: int, on_progress: Any = None) -> Any:
        assert on_progress is not None, "SDK lane must wire progress"
        on_progress(0, 1, size_bytes // 2)
        on_progress(1, 1, size_bytes)
        return s3t.S3TransferResult(
            bucket=grant.bucket, key=grant.key,
            size_bytes=size_bytes, blake3=blake3_hex)

    monkeypatch.setattr(s3t, "upload_file_with_grant", fake_upload)
    _FakeHub.state["grant_mode"] = True
    res = _client(fake_hub).commit(
        destination_repo="acme/repo", files=[_one_file(tmp_path)],
        part_progress=lambda d, t, b: seen.append((d, t, b)))
    assert res.uploaded == 1
    assert (1, 1, 13) in seen, f"byte progress must reach part_progress: {seen}"


@_needs_publish_observability
def test_throttled_progress_logs_on_interval() -> None:
    from gen_worker.convert.publish import _throttled_part_progress

    clock = {"t": 0.0}
    lines: list[str] = []
    pp = _throttled_part_progress(lines.append, interval_s=30.0,
                                  now=lambda: clock["t"])
    clock["t"] = 31.0
    pp(1, 10, 100)          # past interval -> logs
    pp(2, 10, 200)          # within interval -> silent
    clock["t"] = 62.0
    pp(3, 10, 300)          # past interval -> logs
    pp(10, 10, 1000)        # final part always logs
    assert lines == [
        "publish: uploading parts=1/10 bytes=100",
        "publish: uploading parts=3/10 bytes=300",
        "publish: uploading parts=10/10 bytes=1000",
    ]
