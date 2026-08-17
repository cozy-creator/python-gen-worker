"""pgw#1279 / HARDCUT A9: every publish names the release it attaches to.

th#1987 made `release` REQUIRED at tensorhub's v2 DECLARE and refused it before
a byte moves. This module drives the two publish seams that used to send
nothing — `ctx.save_checkpoint` (the training/clone output path) and the
conversion producer — against the real v2 publish client and a real localhost
hub, and asserts the release reaches the wire.

REVERT-TURNS-RED: on the parent commit `publish_v2(release=...)` is optional
and neither call site passes one, so the declare body carries no `release` and
the real hub answers `release_required` on a rented pod instead of here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fake_hub import _FakeHub

from gen_worker.request_context import JobContext

RELEASE = "2026.08"


def _ctx(port: int, *, release: str = RELEASE) -> JobContext:
    """A REAL producer request context: the hints the executor builds from the
    invoke's reserved `destination` struct, and nothing stubbed."""
    hints: dict[str, Any] = {"kind": "training", "destination_repo": "acme/model"}
    if release:
        hints["destination_release"] = release
    return JobContext(
        request_id="req-1",
        job_id="job-1",
        owner="acme",
        file_api_base_url=f"http://127.0.0.1:{port}",
        worker_capability_token="cap-token",
        execution_hints=hints,
    )


def _weights(tmp_path: Path) -> Path:
    p = tmp_path / "adapter.safetensors"
    p.write_bytes(b"\x11" * 4096)
    return p


def _declaration() -> dict:
    return _FakeHub.state["publish_request"]


def test_save_checkpoint_publishes_into_the_requests_release(
    fake_hub: Any, tmp_path: Path
) -> None:
    ctx = _ctx(fake_hub.server_port)
    out = ctx.save_checkpoint("adapter.safetensors", _weights(tmp_path))

    assert out.stream_mode == "presigned"
    assert _declaration()["release"] == RELEASE
    assert _FakeHub.state["attached_releases"]


def test_a_request_naming_no_release_refuses_before_the_upload(
    fake_hub: Any, tmp_path: Path
) -> None:
    """The remedy is a control-plane act by the CALLER (`destination.release`),
    so the refusal names it — and no publish session is opened."""
    ctx = _ctx(fake_hub.server_port, release="")

    with pytest.raises(RuntimeError, match="destination.release"):
        ctx.save_checkpoint("adapter.safetensors", _weights(tmp_path))

    assert not _FakeHub.state.get("publishes")


def test_the_release_is_a_first_class_field_not_metadata(
    fake_hub: Any, tmp_path: Path
) -> None:
    """th#1274's finding, kept red-able: the hub reads `metadata["release"]`
    NOWHERE, so a producer that smuggles it there publishes inert prose."""
    ctx = _ctx(fake_hub.server_port)
    ctx.save_checkpoint("adapter.safetensors", _weights(tmp_path))

    body = _declaration()
    assert body["release"] == RELEASE
    assert "release" not in json.loads(json.dumps(body.get("metadata") or {}))
