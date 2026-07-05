"""Input-asset materialization (#379): URL-ref Assets in the decoded payload
are downloaded to local_path before the handler runs. Real HTTP socket, no
transport mocks."""

from __future__ import annotations

import http.server
import threading

import msgspec
import pytest

from gen_worker import input_assets
from gen_worker.api.errors import ValidationError
from gen_worker.api.types import Asset, ImageAsset

PNG = (
    b"\x89PNG\r\n\x1a\n" + b"\x00" * 64  # magic bytes + filler
)


class Payload(msgspec.Struct):
    image: ImageAsset
    prompt: str = ""
    extra: list[Asset] = []


@pytest.fixture()
def http_root(tmp_path):
    (tmp_path / "in.png").write_bytes(PNG)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(tmp_path), **kw)

        def log_message(self, *a):  # quiet
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture(autouse=True)
def allow_localhost(monkeypatch):
    monkeypatch.setattr(input_assets, "_url_is_blocked", lambda url: False)


def test_materializes_url_ref_assets(http_root, tmp_path):
    rid = "req-mat-1"
    p = Payload(image=ImageAsset(ref=f"{http_root}/in.png"))
    try:
        n = input_assets.materialize_input_assets(p, rid)
        assert n == 1
        assert p.image.local_path and p.image.local_path.endswith(".png")
        with open(p.image.local_path, "rb") as f:
            assert f.read() == PNG
        assert p.image.size_bytes == len(PNG)
        assert p.image.mime_type == "image/png"
    finally:
        input_assets.cleanup_input_assets(rid)
    assert not input_assets.inputs_dir_for_request(rid).exists()


def test_non_url_refs_untouched(tmp_path):
    p = Payload(image=ImageAsset(ref="outputs/foo.png"))
    assert input_assets.materialize_input_assets(p, "req-mat-2") == 0
    assert p.image.local_path is None


def test_size_cap_enforced(http_root):
    p = Payload(image=ImageAsset(ref=f"{http_root}/in.png", url_max_bytes=8))
    with pytest.raises(ValidationError, match="size cap"):
        input_assets.materialize_input_assets(p, "req-mat-3")
    input_assets.cleanup_input_assets("req-mat-3")


def test_mime_allowlist_enforced(http_root):
    p = Payload(
        image=ImageAsset(
            ref=f"{http_root}/in.png", url_allowed_mime_types=("audio/mpeg",)
        )
    )
    with pytest.raises(ValidationError, match="not allowed"):
        input_assets.materialize_input_assets(p, "req-mat-4")
    input_assets.cleanup_input_assets("req-mat-4")


def test_wildcard_mime_allowed(http_root):
    p = Payload(
        image=ImageAsset(ref=f"{http_root}/in.png", url_allowed_mime_types=("image/*",))
    )
    assert input_assets.materialize_input_assets(p, "req-mat-5") == 1
    input_assets.cleanup_input_assets("req-mat-5")


def test_nested_list_assets(http_root):
    p = Payload(
        image=ImageAsset(ref="not-a-url", local_path="/tmp/x"),
        extra=[Asset(ref=f"{http_root}/in.png"), Asset(ref="plain-ref")],
    )
    assert input_assets.materialize_input_assets(p, "req-mat-6") == 1
    assert p.extra[0].local_path
    assert p.extra[1].local_path is None
    input_assets.cleanup_input_assets("req-mat-6")


def test_private_url_blocked(http_root, monkeypatch):
    monkeypatch.setattr(
        input_assets, "_url_is_blocked", lambda url: True
    )
    p = Payload(image=ImageAsset(ref=f"{http_root}/in.png"))
    with pytest.raises(ValidationError, match="not allowed"):
        input_assets.materialize_input_assets(p, "req-mat-7")
