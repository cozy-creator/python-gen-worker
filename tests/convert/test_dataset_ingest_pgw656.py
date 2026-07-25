"""pgw#656: dataset ingest never creates a duplicate it could have found.

Drives the real ``publish_dataset_revision`` over a real HTTP server that
implements tensorhub's actual listing contract (``?org=``, ``?limit=`` capped
at 200, ``?cursor=``, ``next_cursor``). Every row here is a way the lookup
used to come back empty — a torn response, a 500, a second page — each of
which fell straight through to CREATE.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from gen_worker.convert.hub import publish_dataset_revision
from gen_worker.api.errors import AuthError


class _DatasetHub(BaseHTTPRequestHandler):
    server_version = "FakeTensorhubDatasets/1.0"
    state: dict[str, Any] = {}

    def log_message(self, *args: Any) -> None:  # silence
        pass

    def _send(self, code: int, payload: Any = None, raw: bytes | None = None) -> None:
        body = raw if raw is not None else json.dumps(payload or {}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        st = _DatasetHub.state
        parsed = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        st.setdefault("list_calls", []).append(query)

        status = int(st.get("list_status", 200))
        if status != 200:
            self._send(status, {"error": "boom"})
            return
        if st.get("list_garbage"):
            self._send(200, raw=b"<html>gateway</html>")
            return

        # tensorhub semantics: org filter, limit capped at 200, integer cursor.
        rows = [r for r in st.get("rows", []) if r["org"] == query.get("org")]
        limit = min(int(query.get("limit") or 50), 200)
        offset = int(query.get("cursor") or 0)
        page = rows[offset:offset + limit]
        next_cursor = str(offset + len(page)) if len(page) == limit else ""
        self._send(200, {
            "items": [
                {"dataset_id": r["dataset_id"], "name": r["name"]} for r in page
            ],
            "next_cursor": next_cursor,
        })

    def do_POST(self) -> None:  # noqa: N802
        _DatasetHub.state.setdefault("creates", []).append(self.path)
        self._send(200, {"dataset_id": "created-id"})

    def do_PATCH(self) -> None:  # noqa: N802
        _DatasetHub.state.setdefault("patches", []).append(self.path)
        self._send(200, {})


@pytest.fixture
def hub():
    _DatasetHub.state = {"rows": [], "creates": [], "patches": [], "list_calls": []}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DatasetHub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", _DatasetHub.state
    finally:
        server.shutdown()
        server.server_close()


def _publish(base: str, dataset: str = "acme/faces"):
    return publish_dataset_revision(
        base_url=base,
        token="t",
        destination_dataset=dataset,
        features_json={"image": "bytes"},
        kind="image",
    )


def test_absent_dataset_is_created_once(hub) -> None:
    base, state = hub
    out = _publish(base)
    assert out["existed"] is False
    assert len(state["creates"]) == 1


def test_existing_dataset_beyond_the_first_page_is_found(hub) -> None:
    """The unpaginated read saw one page and called it the org's whole set —
    dataset 250 of 300 then got a duplicate row."""
    base, state = hub
    state["rows"] = [
        {"org": "acme", "dataset_id": f"id-{i}", "name": f"ds-{i}"} for i in range(300)
    ]
    state["rows"][250] = {"org": "acme", "dataset_id": "the-one", "name": "faces"}

    out = _publish(base)
    assert out == {
        "ok": True, "dataset_id": "the-one", "owner": "acme",
        "name": "faces", "existed": True,
    }
    assert state["creates"] == []
    assert len(state["list_calls"]) == 2  # 200 + the remainder
    assert state["list_calls"][0]["org"] == "acme"  # not the ignored ?tenant=


def test_a_500_on_the_listing_refuses_instead_of_creating(hub) -> None:
    base, state = hub
    state["list_status"] = 500
    with pytest.raises(RuntimeError, match="dataset list failed"):
        _publish(base)
    assert state["creates"] == []


def test_unreadable_json_refuses_instead_of_creating(hub) -> None:
    """The exact `except Exception: pass` this issue names: a transient
    proxy/HTML body read as 'no existing dataset'."""
    base, state = hub
    state["list_garbage"] = True
    with pytest.raises(RuntimeError, match="unreadable JSON"):
        _publish(base)
    assert state["creates"] == []


def test_unauthorized_listing_is_typed(hub) -> None:
    base, state = hub
    state["list_status"] = 403
    with pytest.raises(AuthError):
        _publish(base)
    assert state["creates"] == []
