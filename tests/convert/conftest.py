"""Fixtures for gen_worker.convert tests (fake tensorhub lives in fake_hub.py)."""

from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer

import pytest

from fake_hub import _FakeHub


@pytest.fixture()
def fake_hub():
    # th#1987: a publish attaches to an ALREADY-CUT release. The default repo
    # this fake serves has these cut; a test that wants the `release_not_found`
    # refusal empties the set explicitly.
    _FakeHub.state = {"existing_blobs": set(), "releases": {"r1", "r2", "2026.08"}}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHub)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield server
    server.shutdown()
