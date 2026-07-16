"""th#826 call-out primitive: SDK client + ctx surface against a fake hub."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

import pytest

from gen_worker.api.errors import (
    CanceledError,
    ChildCallRefusedError,
    ChildCallTimeoutError,
    ChildRequestCanceledError,
    ChildRequestFailedError,
)
from gen_worker.callout import CalloutClient, ChildRequest
from gen_worker.request_context import RequestContext

PARENT_ID = "req-parent-1"
TOKEN = "cap-token-1"


class _FakeHub(BaseHTTPRequestHandler):
    """Scriptable fake of the platform surface the callout client speaks to."""

    state: Dict[str, Any] = {}

    def log_message(self, *args: Any) -> None:  # silence
        pass

    def _json(self, code: int, doc: Any) -> None:
        body = json.dumps(doc).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self) -> bool:
        return self.headers.get("Authorization") == f"Bearer {TOKEN}"

    def do_POST(self) -> None:  # noqa: N802
        st = _FakeHub.state
        if not self._authed():
            self._json(401, {"error": {"code": "unauthorized", "message": "no token"}})
            return
        if self.path.endswith("/cancel"):
            rid = self.path.split("/")[-2]
            st.setdefault("cancels", []).append(rid)
            self._json(200, {"ok": True})
            return
        # invoke: /{owner}/{endpoint}/{function}:{tag}
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        st.setdefault("submits", []).append({"path": self.path, "payload": payload})
        refusal = st.get("refusal")
        if refusal:
            self._json(refusal["status"], {"error": {"code": refusal["code"], "message": refusal.get("message", "")}})
            return
        self._json(200, {"request_id": st.get("next_request_id", "child-1"), "status": "queued"})

    def do_GET(self) -> None:  # noqa: N802
        st = _FakeHub.state
        if not self._authed():
            self._json(401, {"error": {"code": "unauthorized", "message": "no token"}})
            return
        if "/checkpoints/" in self.path:
            key = self.path.rsplit("/", 1)[-1]
            ckpts = st.setdefault("checkpoints", {})
            if key in ckpts:
                self._json(200, ckpts[key])
            else:
                self._json(404, {"error": {"code": "not_found", "message": "checkpoint not found"}})
            return
        # /v1/requests/{id}
        rid = self.path.rsplit("/", 1)[-1]
        polls = st.setdefault("polls", {})
        n = polls.get(rid, 0)
        polls[rid] = n + 1
        seq = st.get("statuses", ["completed"])
        doc = seq[min(n, len(seq) - 1)]
        self._json(200, {"id": rid, **doc})

    def do_PUT(self) -> None:  # noqa: N802
        st = _FakeHub.state
        if not self._authed():
            self._json(401, {"error": {"code": "unauthorized", "message": "no token"}})
            return
        key = self.path.rsplit("/", 1)[-1]
        length = int(self.headers.get("Content-Length") or 0)
        value = json.loads(self.rfile.read(length) or b"null")
        st.setdefault("checkpoints", {})[key] = value
        st.setdefault("checkpoint_puts", []).append(key)
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture()
def hub():
    _FakeHub.state = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", _FakeHub.state
    server.shutdown()
    thread.join(timeout=5)


def _client(base: str, cancel_event: Optional[threading.Event] = None) -> CalloutClient:
    return CalloutClient(
        base_url=base,
        parent_request_id=PARENT_ID,
        get_token=lambda: TOKEN,
        cancel_event=cancel_event,
    )


def _ctx(base: str) -> RequestContext:
    return RequestContext(
        request_id=PARENT_ID,
        file_api_base_url=base,
        worker_capability_token=TOKEN,
    )


def test_submit_and_wait_returns_output(hub):
    base, state = hub
    state["statuses"] = [
        {"status": "queued", "output": []},
        {"status": "in_progress", "output": []},
        {"status": "completed", "output": [{"type": "video", "ref": "media/abc.mp4"}]},
    ]
    out = _ctx(base).call_endpoint(
        "tensorhub/music-analysis", "analyze-quick", {"audio": "ref-1"},
        poll_interval_s=0.01,
    )
    assert out == [{"type": "video", "ref": "media/abc.mp4"}]
    sub = state["submits"][0]
    assert sub["path"] == "/tensorhub/music-analysis/analyze-quick:prod"
    assert sub["payload"] == {"audio": "ref-1"}


def test_submit_carries_cheaper_tier_and_tag(hub):
    base, state = hub
    state["statuses"] = [{"status": "completed", "output": []}]
    _ctx(base).call_endpoint(
        "tensorhub/ltx-video-2.3", "audio-reactive", {"p": 1},
        tag="dev", tier="flex", poll_interval_s=0.01,
    )
    sub = state["submits"][0]
    assert sub["path"] == "/tensorhub/ltx-video-2.3/audio-reactive:dev"
    assert sub["payload"]["availability_tier"] == "flex"


@pytest.mark.parametrize(
    "code",
    [
        "call_depth_exceeded",
        "call_cycle_detected",
        "tree_budget_exceeded",
        "tier_escalation_denied",
        "parent_not_running",
        "budget_not_root",
    ],
)
def test_typed_admission_refusals(hub, code):
    base, state = hub
    state["refusal"] = {"status": 403, "code": code, "message": "refused"}
    with pytest.raises(ChildCallRefusedError) as exc:
        _ctx(base).call_endpoint("tensorhub/some-ep", "step", {})
    assert exc.value.code == code


def test_undeclared_child_calls_maps_to_typed_refusal(hub):
    base, state = hub
    state["refusal"] = {"status": 403, "code": "insufficient_scope", "message": "forbidden"}
    with pytest.raises(ChildCallRefusedError) as exc:
        _ctx(base).call_endpoint("tensorhub/some-ep", "step", {})
    assert exc.value.code == "child_calls_not_declared"


def test_missing_token_is_typed_refusal():
    ctx = RequestContext(request_id=PARENT_ID, file_api_base_url="http://127.0.0.1:9")
    with pytest.raises(ChildCallRefusedError) as exc:
        ctx.call_endpoint("tensorhub/some-ep", "step", {})
    assert exc.value.code == "child_calls_not_declared"


def test_wait_false_returns_handle_and_cancel(hub):
    base, state = hub
    state["next_request_id"] = "child-9"
    state["statuses"] = [{"status": "in_progress", "output": []}]
    handle = _ctx(base).call_endpoint("tensorhub/some-ep", "step", {}, wait=False)
    assert isinstance(handle, ChildRequest)
    assert handle.request_id == "child-9"
    assert handle.status() == "in_progress"
    handle.cancel()
    assert state["cancels"] == ["child-9"]


def test_child_failure_and_cancel_raise_typed(hub):
    base, state = hub
    state["statuses"] = [
        {"status": "failed", "error": {"type": "oom", "message": "boom"}},
    ]
    with pytest.raises(ChildRequestFailedError) as exc:
        _ctx(base).call_endpoint("tensorhub/some-ep", "step", {}, poll_interval_s=0.01)
    assert exc.value.error_type == "oom"
    assert exc.value.error_message == "boom"

    state["polls"] = {}
    state["statuses"] = [{"status": "canceled"}]
    with pytest.raises(ChildRequestCanceledError):
        _ctx(base).call_endpoint("tensorhub/some-ep", "step", {}, poll_interval_s=0.01)


def test_wait_timeout_raises_and_child_keeps_running(hub):
    base, state = hub
    state["statuses"] = [{"status": "in_progress", "output": []}]
    with pytest.raises(ChildCallTimeoutError):
        _ctx(base).call_endpoint(
            "tensorhub/some-ep", "step", {}, timeout_s=0.05, poll_interval_s=0.01
        )


def test_parent_cancellation_interrupts_wait(hub):
    base, state = hub
    state["statuses"] = [{"status": "in_progress", "output": []}]
    ctx = _ctx(base)
    handle = ctx.call_endpoint("tensorhub/some-ep", "step", {}, wait=False)

    timer = threading.Timer(0.1, ctx._cancel)
    timer.start()
    try:
        with pytest.raises(CanceledError):
            handle.result(timeout_s=30, poll_interval_s=0.5)
    finally:
        timer.cancel()


def test_workflow_checkpoint_memoizes(hub):
    base, state = hub
    ctx = _ctx(base)
    calls = {"n": 0}

    def step() -> Dict[str, Any]:
        calls["n"] += 1
        return {"clip": "media/clip-1.mp4", "score": 0.97}

    first = ctx.workflow_checkpoint("scene-1", step)
    second = ctx.workflow_checkpoint("scene-1", step)
    assert first == second == {"clip": "media/clip-1.mp4", "score": 0.97}
    assert calls["n"] == 1
    assert state["checkpoint_puts"] == ["scene-1"]
    # A different key computes independently.
    other = ctx.workflow_checkpoint("scene-2", lambda: {"clip": "media/clip-2.mp4"})
    assert other == {"clip": "media/clip-2.mp4"}


def test_bad_endpoint_shape_rejected(hub):
    base, _ = hub
    with pytest.raises(ValueError):
        _ctx(base).call_endpoint("music-analysis", "step", {})


import msgspec


class _Out(msgspec.Struct):
    ok: bool


class _In(msgspec.Struct):
    x: int


def test_child_calls_declaration_reaches_manifest():
    from gen_worker import RequestContext as Ctx  # noqa: F401 (import sanity)
    from gen_worker.api.decorators import ATTR, endpoint
    from gen_worker.discovery.discover import _extract_entries

    @endpoint(kind="inference", child_calls=True)
    def workflow(ctx: RequestContext, payload: _In) -> _Out:  # pragma: no cover
        return _Out(ok=True)

    @endpoint(kind="inference")
    def plain(ctx: RequestContext, payload: _In) -> _Out:  # pragma: no cover
        return _Out(ok=True)

    assert getattr(workflow, ATTR).child_calls is True
    assert getattr(plain, ATTR).child_calls is False

    wf_entry = _extract_entries(workflow, "tests.test_callout")[0]
    assert wf_entry["child_calls"] is True
    plain_entry = _extract_entries(plain, "tests.test_callout")[0]
    assert "child_calls" not in plain_entry
