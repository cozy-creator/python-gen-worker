from __future__ import annotations

import inspect
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, List, Optional, Tuple, cast

import pytest

from gen_worker.callout import CalloutClient, semver_major_segment
from gen_worker.request_context import RequestContext


class _Recorder:

    def __init__(self) -> None:
        self.paths: List[str] = []
        recorder = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                recorder.paths.append(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                body = json.dumps({"request_id": "child-1"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "_Recorder":
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        host, port = cast(Tuple[str, int], self._server.server_address[:2])
        return f"http://{host}:{port}"


def _client(base_url: str) -> CalloutClient:
    return CalloutClient(
        base_url=base_url, parent_request_id="req-parent", get_token=lambda: "tok"
    )


@pytest.mark.parametrize("semver_major,segment", [(0, "v0"), (1, "v1"), (12, "v12")])
def test_submit_addresses_the_semver_major_segment(
    semver_major: int, segment: str
) -> None:
    """The URL that goes out, over a real socket."""
    with _Recorder() as rec:
        _client(rec.base_url).submit(
            "acme/child", "generate", {"prompt": "x"}, semver_major=semver_major
        )
    assert rec.paths == [f"/acme/child/{segment}/generate"]
    assert ":" not in rec.paths[0]


def test_submit_without_semver_major_refuses_naming_the_parameter() -> None:
    with _Recorder() as rec:
        client = _client(rec.base_url)
        with pytest.raises(TypeError, match="semver_major"):
            client.submit("acme/child", "generate", {})  # type: ignore[call-arg]
        assert rec.paths == [], "a refused call must not reach the platform"


def test_semver_major_none_raises_rather_than_defaulting() -> None:
    with _Recorder() as rec:
        client = _client(rec.base_url)
        with pytest.raises(TypeError, match="semver_major"):
            client.submit("acme/child", "generate", {}, semver_major=None)  # type: ignore[arg-type]
        assert rec.paths == [], "a refused call must not reach the platform"


@pytest.mark.parametrize("bad", ["0", "v0", 1.0, True, False])
def test_non_int_semver_major_refuses(bad: Any) -> None:
    """`True` would render `vTrue`; `"v0"` would render `vv0`."""
    with pytest.raises(TypeError, match="semver_major"):
        semver_major_segment(bad)


def test_negative_semver_major_refuses() -> None:
    with pytest.raises(ValueError, match="semver_major"):
        semver_major_segment(-1)


def _param(fn: Any, name: str) -> Optional[inspect.Parameter]:
    return inspect.signature(fn).parameters.get(name)


@pytest.mark.parametrize(
    "fn", [RequestContext.call_endpoint, CalloutClient.submit], ids=["ctx", "client"]
)
def test_semver_major_is_required_keyword_only_and_tag_is_gone(fn: Any) -> None:
    """The wheel contract ie/te/pie call at runtime."""
    assert _param(fn, "tag") is None, "the endpoint-tag axis is dead (th#2044)"
    p = _param(fn, "semver_major")
    assert p is not None
    assert p.kind is inspect.Parameter.KEYWORD_ONLY
    assert p.default is inspect.Parameter.empty, "required, no default, no `latest`"


def test_ctx_call_endpoint_addresses_a_non_zero_major_over_the_real_client() -> None:
    """`ctx` is a pass-through: the major it is handed is the one on the wire."""
    with _Recorder() as rec:
        ctx = RequestContext.__new__(RequestContext)
        ctx._canceled = False  # type: ignore[attr-defined]
        ctx._cancel_event = threading.Event()  # type: ignore[attr-defined]
        ctx._request_id = "req-parent"  # type: ignore[attr-defined]
        ctx._worker_capability_token = "tok"  # type: ignore[attr-defined]
        ctx._file_api_base_url = rec.base_url  # type: ignore[attr-defined]
        ctx._child_calls = True  # type: ignore[attr-defined]

        handle = ctx.call_endpoint(
            "acme/child", "generate", {}, semver_major=7, wait=False
        )
    assert handle is not None
    assert rec.paths == ["/acme/child/v7/generate"]


@pytest.mark.parametrize("n,want", [(0, "v0"), (3, "v3"), (10, "v10")])
def test_segment_grammar(n: int, want: str) -> None:
    assert semver_major_segment(n) == want
