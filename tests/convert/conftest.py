"""Fixtures for gen_worker.convert tests (fake tensorhub lives in fake_hub.py)."""

from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer

import pytest

from fake_hub import _FakeHub


@pytest.fixture()
def fake_hub():
    _FakeHub.state = {"existing_blobs": set()}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHub)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield server
    server.shutdown()
