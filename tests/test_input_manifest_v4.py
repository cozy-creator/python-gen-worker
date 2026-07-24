"""gw#585: Tensorhub v4 private-input manifests (th#886 hard cut).

Hermetic: the resolver is a fake injected transport; downloads hit a local
loopback HTTP server only. Covers deterministic traversal, manifest fencing,
the single strict resolver POST, response strict-decoding, byte verification,
attempt-scoped cleanup/result fencing, discovery build rejection, and the
current proto descriptors.
"""

from __future__ import annotations

import asyncio
import base64
import http.server
import json
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import blake3
import msgspec
import pytest

from gen_worker import input_assets
from gen_worker.api.errors import CanceledError, RetryableError, ValidationError
from gen_worker.api.types import Asset, AudioAsset, ImageAsset, MediaAsset, VideoAsset
from gen_worker.discovery.discover import _collect_payload_moderation_metadata
from gen_worker.executor import Executor
from gen_worker.input_assets import (
    InputManifestEntry,
    cleanup_input_assets,
    inputs_dir_for_request,
    manifest_from_run_job,
    materialize_input_assets,
)
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec

PNG_A = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAYAAAD0In+KAAAAEUlEQVR4nGP8z8Dwn4GBgQEADQUCAOAHawIAAAAASUVORK5CYII="
)
PNG_B = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAYAAAD0In+KAAAAEUlEQVR4nGNkYPj/n4GBgQEACwcCAKXJIuMAAAAASUVORK5CYII="
)
CORRUPT_PNG = b"\x89PNG\r\n\x1a\n" + b"not-a-decodable-image"
MP3 = b"ID3" + b"audio" * 16
_SERVER_CHUNK = 1 << 20
SLOW_BODY = PNG_A + b"x" * (2 * _SERVER_CHUNK)

BASE_URL = "http://hub.internal.test"
CAPABILITY = "cap-token-SECRET"
EXPIRES = "2099-01-01T00:00:00+00:00"

REF_A = "tenant/private/a"
REF_B = "tenant/private/b"
REF_C = "tenant/private/audio"


def b3(data: bytes) -> str:
    return blake3.blake3(data).hexdigest()


def make_entry(ref: str, body: bytes, kind: str, mime: str, **overrides: Any) -> InputManifestEntry:
    fields: dict[str, Any] = {
        "asset_id": f"asset-{ref.rsplit('/', 1)[-1]}",
        "source_ref": ref,
        "blake3": b3(body),
        "size_bytes": len(body),
        "kind": kind,
        "mime_type": mime,
    }
    fields.update(overrides)
    return InputManifestEntry(**fields)


@dataclass
class FakeResolver:
    """Injected resolver transport: records calls, serves a canned response."""

    manifest: tuple[InputManifestEntry, ...] = ()
    urls: dict[str, str] = field(default_factory=dict)  # source_ref -> url
    status: int = 200
    raw_body: bytes | None = None
    mutate: Any = None  # callable(list[dict]) -> list[dict]
    exc: Exception | None = None
    on_call: Any = None
    calls: list[tuple[str, dict[str, str], bytes]] = field(default_factory=list)

    def __call__(self, url: str, headers: Any, body: bytes) -> tuple[int, bytes]:
        self.calls.append((url, dict(headers), bytes(body)))
        if self.on_call is not None:
            self.on_call()
        if self.exc is not None:
            raise self.exc
        if self.raw_body is not None:
            return self.status, self.raw_body
        if self.status != 200:
            return self.status, b'{"error":"resolver-error-body"}'
        assets = [
            {
                "asset_id": e.asset_id,
                "source_ref": e.source_ref,
                "blake3": e.blake3,
                "size_bytes": e.size_bytes,
                "kind": e.kind,
                "mime_type": e.mime_type,
                "url": self.urls[e.source_ref],
                "url_expires_at": EXPIRES,
            }
            for e in self.manifest
        ]
        if self.mutate is not None:
            assets = self.mutate(assets)
        return 200, json.dumps({"assets": assets}).encode()


@dataclass
class HTTPRoot:
    base_url: str
    hits: dict[str, int]
    slow_started: threading.Event
    slow_release: threading.Event

    def url(self, name: str) -> str:
        return f"{self.base_url}/{name}"


@pytest.fixture()
def http_root() -> Any:
    hits: dict[str, int] = {}
    slow_started = threading.Event()
    slow_release = threading.Event()
    bodies = {
        "/a.png": ("image/png", PNG_A),
        "/b.png": ("image/png", PNG_B),
        "/audio.mp3": ("audio/mpeg", MP3),
        "/corrupt.png": ("image/png", CORRUPT_PNG),
    }

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            pass

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            path = urllib.parse.urlsplit(self.path).path
            hits[path] = hits.get(path, 0) + 1
            if path == "/slow.bin":
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(SLOW_BODY)))
                self.end_headers()
                try:
                    self.wfile.write(SLOW_BODY[:_SERVER_CHUNK])
                    self.wfile.flush()
                    slow_started.set()
                    slow_release.wait(timeout=5)
                    self.wfile.write(SLOW_BODY[_SERVER_CHUNK:])
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            content = bodies.get(path)
            if content is None:
                self.send_error(404)
                return
            mime, body = content
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = HTTPRoot(
        base_url=f"http://127.0.0.1:{server.server_address[1]}",
        hits=hits,
        slow_started=slow_started,
        slow_release=slow_release,
    )
    yield root
    slow_release.set()
    server.shutdown()
    thread.join(timeout=5)


@pytest.fixture(autouse=True)
def allow_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(input_assets, "_url_is_blocked", lambda _url: False)


class Payload(msgspec.Struct):
    image: ImageAsset | None = None
    prompt: str = ""
    extra: list[Asset] = []
    audio: AudioAsset | None = None
    mapped: dict[str, Asset] = {}


def materialize(
    payload: Any,
    request_id: str,
    *,
    attempt: int = 1,
    manifest: tuple[InputManifestEntry, ...] = (),
    resolver: FakeResolver | None = None,
    cancel_check: Any = None,
) -> int:
    return materialize_input_assets(
        payload,
        request_id,
        attempt=attempt,
        manifest=manifest,
        file_base_url=BASE_URL,
        capability_token=CAPABILITY,
        cancel_check=cancel_check,
        resolve_transport=resolver,
    )


def assert_no_leak(exc: BaseException, *secrets: str) -> None:
    text = str(exc)
    for secret in (CAPABILITY, BASE_URL, *secrets):
        assert secret not in text


# ---------------------------------------------------------------------------
# Deterministic traversal + classification
# ---------------------------------------------------------------------------


def test_reverse_insertion_map_keys_traverse_sorted(http_root: HTTPRoot) -> None:
    mapped: dict[str, Asset] = {}
    mapped["zzz"] = Asset(ref=REF_B)
    mapped["aaa"] = Asset(ref=REF_A)
    payload = Payload(mapped=mapped)
    assert [a.ref for a in input_assets._iter_assets(payload)] == [REF_A, REF_B]

    manifest = (
        make_entry(REF_A, PNG_A, "media", "image/png"),
        make_entry(REF_B, PNG_B, "media", "image/png"),
    )
    resolver = FakeResolver(
        manifest=manifest,
        urls={REF_A: http_root.url("a.png"), REF_B: http_root.url("b.png")},
    )
    assert materialize(payload, "req-map", manifest=manifest, resolver=resolver) == 2
    assert Path(mapped["aaa"].local_path or "").read_bytes() == PNG_A
    assert Path(mapped["zzz"].local_path or "").read_bytes() == PNG_B
    cleanup_input_assets("req-map", 1)

    # Insertion-order manifest (reversed) is a dispatch reorder: rejected.
    reordered = (manifest[1], manifest[0])
    with pytest.raises(ValidationError, match="^input_asset_manifest_mismatch"):
        materialize(Payload(mapped=dict(mapped)), "req-map-2", manifest=reordered)


def test_non_string_map_key_rejected_at_runtime() -> None:
    class Loose(msgspec.Struct):
        mapped: dict

    payload = Loose(mapped={1: Asset(ref=REF_A)})
    with pytest.raises(ValidationError, match="^input_asset_map_key_invalid"):
        materialize_input_assets(payload, "req-badkey", attempt=1)


def test_asset_in_set_rejected_at_runtime() -> None:
    # Assets are unhashable, so a real decoded payload can never place one in
    # a set; the defensive runtime branch still refuses hostile lookalikes.
    class HashableAsset(Asset):
        def __hash__(self) -> int:  # pragma: no cover - identity for the test
            return 7

    class Loose(msgspec.Struct):
        items: Any

    for container in ({HashableAsset(ref=REF_A)}, frozenset({HashableAsset(ref=REF_A)})):
        with pytest.raises(ValidationError, match="^input_asset_unordered_container"):
            materialize_input_assets(Loose(items=container), "req-set", attempt=1)


def test_padded_or_empty_ref_rejected() -> None:
    for ref in ("", " tenant/private/a", "tenant/private/a "):
        payload = Payload(extra=[Asset(ref=ref)])
        with pytest.raises(ValidationError, match="^invalid_input_asset_ref"):
            materialize_input_assets(payload, "req-pad", attempt=1)


@pytest.mark.parametrize(
    "ref", ["file:///etc/passwd", "s3://bucket/key", "data:image/png,x", "//host/x"]
)
def test_non_http_schemes_rejected(ref: str) -> None:
    payload = Payload(extra=[Asset(ref=ref)])
    with pytest.raises(ValidationError, match="^unsupported_input_asset_scheme") as caught:
        materialize_input_assets(payload, "req-scheme", attempt=1)
    assert ref not in str(caught.value)


# ---------------------------------------------------------------------------
# Manifest fencing (before any resolver call or GET)
# ---------------------------------------------------------------------------


def test_zero_assets_zero_manifest() -> None:
    assert materialize(Payload(prompt="text"), "req-zero") == 0
    assert not inputs_dir_for_request("req-zero", 1).exists()


def test_zero_assets_with_manifest_rejected() -> None:
    manifest = (make_entry(REF_A, PNG_A, "media", "image/png"),)
    with pytest.raises(ValidationError, match="^input_asset_manifest_mismatch"):
        materialize(Payload(prompt="text"), "req-zero-extra", manifest=manifest)


def test_public_only_never_calls_resolver(http_root: HTTPRoot) -> None:
    resolver = FakeResolver()
    payload = Payload(image=ImageAsset(ref=http_root.url("a.png")))
    assert materialize(payload, "req-public", resolver=resolver) == 1
    assert resolver.calls == []
    assert Path(payload.image.local_path or "").read_bytes() == PNG_A
    cleanup_input_assets("req-public", 1)


@pytest.mark.parametrize(
    "case,payload_refs,manifest_refs",
    [
        ("missing", [REF_A], []),
        ("extra", [REF_A], [REF_A, REF_B]),
        ("reorder", [REF_A, REF_B], [REF_B, REF_A]),
    ],
)
def test_dispatch_manifest_count_order_fenced(
    case: str, payload_refs: list[str], manifest_refs: list[str]
) -> None:
    bodies = {REF_A: PNG_A, REF_B: PNG_B}
    payload = Payload(extra=[Asset(ref=r) for r in payload_refs])
    manifest = tuple(make_entry(r, bodies[r], "media", "image/png") for r in manifest_refs)
    resolver = FakeResolver(manifest=manifest)
    with pytest.raises(ValidationError, match="^input_asset_manifest_mismatch") as caught:
        materialize(payload, f"req-{case}", manifest=manifest, resolver=resolver)
    assert resolver.calls == []
    assert_no_leak(caught.value, REF_A, REF_B)


def test_dispatch_kind_mismatch_fenced() -> None:
    payload = Payload(image=ImageAsset(ref=REF_A))
    manifest = (make_entry(REF_A, PNG_A, "video", "image/png"),)
    resolver = FakeResolver(manifest=manifest)
    with pytest.raises(ValidationError, match="^input_asset_manifest_mismatch"):
        materialize(payload, "req-kindfence", manifest=manifest, resolver=resolver)
    assert resolver.calls == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"asset_id": ""},
        {"blake3": "abc"},
        {"blake3": "Z" * 64},
        {"size_bytes": 0},
        {"mime_type": ""},
    ],
)
def test_invalid_manifest_metadata_fenced(overrides: dict[str, Any]) -> None:
    payload = Payload(extra=[Asset(ref=REF_A)])
    manifest = (make_entry(REF_A, PNG_A, "media", "image/png", **overrides),)
    resolver = FakeResolver(manifest=manifest)
    with pytest.raises(ValidationError, match="^input_asset_manifest_invalid"):
        materialize(payload, "req-meta", manifest=manifest, resolver=resolver)
    assert resolver.calls == []


def test_private_inputs_require_positive_attempt() -> None:
    payload = Payload(extra=[Asset(ref=REF_A)])
    manifest = (make_entry(REF_A, PNG_A, "media", "image/png"),)
    with pytest.raises(ValidationError, match="^input_asset_attempt_invalid"):
        materialize(payload, "req-attempt", attempt=0, manifest=manifest)


# ---------------------------------------------------------------------------
# Resolver call contract
# ---------------------------------------------------------------------------


def test_exactly_one_strict_resolver_post(http_root: HTTPRoot) -> None:
    payload = Payload(
        image=ImageAsset(ref=REF_A),
        extra=[Asset(ref=REF_B)],
        audio=AudioAsset(ref=REF_C),
    )
    manifest = (
        make_entry(REF_A, PNG_A, "image", "image/png"),
        make_entry(REF_B, PNG_B, "media", "image/png"),
        make_entry(REF_C, MP3, "audio", "audio/mpeg"),
    )
    resolver = FakeResolver(
        manifest=manifest,
        urls={
            REF_A: http_root.url("a.png"),
            REF_B: http_root.url("b.png"),
            REF_C: http_root.url("audio.mp3"),
        },
    )
    assert materialize(payload, "req-one-post", attempt=7, manifest=manifest, resolver=resolver) == 3
    assert len(resolver.calls) == 1
    url, headers, body = resolver.calls[0]
    assert url == f"{BASE_URL}/api/v1/worker/input-assets/resolve"
    assert headers["Authorization"] == f"Bearer {CAPABILITY}"
    assert json.loads(body) == {"request_id": "req-one-post", "attempt": 7}
    cleanup_input_assets("req-one-post", 7)


def test_one_private_asset_preserves_all_fields(http_root: HTTPRoot) -> None:
    asset = ImageAsset(
        ref=REF_A,
        local_path="/etc/passwd",
        mime_type="image/png",
        size_bytes=len(PNG_A),
        blake3=b3(PNG_A),
        sha256="caller-provided-sha",
    )
    payload = Payload(image=asset)
    manifest = (make_entry(REF_A, PNG_A, "image", "image/png"),)
    resolver = FakeResolver(manifest=manifest, urls={REF_A: http_root.url("a.png")})
    assert materialize(payload, "req-priv-one", manifest=manifest, resolver=resolver) == 1
    assert asset.ref == REF_A  # opaque ref NEVER rewritten
    assert asset.local_path and Path(asset.local_path).read_bytes() == PNG_A
    assert Path(asset.local_path).parent == inputs_dir_for_request("req-priv-one", 1)
    assert asset.local_path != "/etc/passwd"
    assert asset.mime_type == "image/png"
    assert asset.size_bytes == len(PNG_A)
    assert asset.sha256 == "caller-provided-sha"

    cleanup_input_assets("req-priv-one", 1)
    assert asset.local_path is None


def test_duplicate_private_occurrences_share_one_download(http_root: HTTPRoot) -> None:
    a1, b, a2 = Asset(ref=REF_A), Asset(ref=REF_B), Asset(ref=REF_A)
    payload = Payload(extra=[a1, b, a2])
    manifest = (
        make_entry(REF_A, PNG_A, "media", "image/png"),
        make_entry(REF_B, PNG_B, "media", "image/png"),
    )
    resolver = FakeResolver(
        manifest=manifest,
        urls={REF_A: http_root.url("a.png"), REF_B: http_root.url("b.png")},
    )
    assert materialize(payload, "req-aba", manifest=manifest, resolver=resolver) == 2
    assert a1.local_path == a2.local_path and a1.local_path != b.local_path
    assert http_root.hits == {"/a.png": 1, "/b.png": 1}
    cleanup_input_assets("req-aba", 1)


def test_mixed_private_and_public(http_root: HTTPRoot) -> None:
    private = ImageAsset(ref=REF_A)
    public = ImageAsset(ref=http_root.url("b.png"))
    payload = Payload(image=private, extra=[public])
    manifest = (make_entry(REF_A, PNG_A, "image", "image/png"),)
    resolver = FakeResolver(manifest=manifest, urls={REF_A: http_root.url("a.png")})
    assert materialize(payload, "req-mixed", manifest=manifest, resolver=resolver) == 2
    assert len(resolver.calls) == 1
    assert Path(private.local_path or "").read_bytes() == PNG_A
    assert Path(public.local_path or "").read_bytes() == PNG_B
    assert public.ref == http_root.url("b.png")
    cleanup_input_assets("req-mixed", 1)


def test_resolver_409_cancels_attempt() -> None:
    payload = Payload(extra=[Asset(ref=REF_A)])
    manifest = (make_entry(REF_A, PNG_A, "media", "image/png"),)
    resolver = FakeResolver(manifest=manifest, status=409)
    with pytest.raises(CanceledError):
        materialize(payload, "req-409", manifest=manifest, resolver=resolver)
    assert not inputs_dir_for_request("req-409", 1).exists()


@pytest.mark.parametrize("status", [400, 403, 500, 503])
def test_resolver_failure_statuses_are_retryable(status: int) -> None:
    payload = Payload(extra=[Asset(ref=REF_A)])
    manifest = (make_entry(REF_A, PNG_A, "media", "image/png"),)
    resolver = FakeResolver(manifest=manifest, status=status)
    with pytest.raises(RetryableError, match="^input_asset_resolution_unavailable") as caught:
        materialize(payload, f"req-{status}", manifest=manifest, resolver=resolver)
    assert "resolver-error-body" not in str(caught.value)
    assert_no_leak(caught.value, REF_A)


def test_resolver_network_failure_is_retryable() -> None:
    payload = Payload(extra=[Asset(ref=REF_A)])
    manifest = (make_entry(REF_A, PNG_A, "media", "image/png"),)
    resolver = FakeResolver(manifest=manifest, exc=ConnectionError("boom"))
    with pytest.raises(RetryableError, match="^input_asset_resolution_unavailable"):
        materialize(payload, "req-net", manifest=manifest, resolver=resolver)


def test_missing_resolver_credentials_is_retryable() -> None:
    payload = Payload(extra=[Asset(ref=REF_A)])
    manifest = (make_entry(REF_A, PNG_A, "media", "image/png"),)
    with pytest.raises(RetryableError, match="^input_asset_resolution_unavailable"):
        materialize_input_assets(
            payload, "req-nocreds", attempt=1, manifest=manifest, file_base_url="",
            capability_token="",
        )


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"{}",
        b'{"assets": [{"bogus": 1}]}',
        b'{"assets": [], "extra_top": 1}',
    ],
)
def test_malformed_resolver_response_is_retryable(raw: bytes) -> None:
    payload = Payload(extra=[Asset(ref=REF_A)])
    manifest = (make_entry(REF_A, PNG_A, "media", "image/png"),)
    resolver = FakeResolver(manifest=manifest, raw_body=raw)
    with pytest.raises(RetryableError, match="^input_asset_response_mismatch"):
        materialize(payload, "req-malformed", manifest=manifest, resolver=resolver)


def test_unknown_response_field_is_rejected(http_root: HTTPRoot) -> None:
    payload = Payload(extra=[Asset(ref=REF_A)])
    manifest = (make_entry(REF_A, PNG_A, "media", "image/png"),)

    def add_unknown(assets: list[dict]) -> list[dict]:
        assets[0]["surprise"] = True
        return assets

    resolver = FakeResolver(
        manifest=manifest, urls={REF_A: http_root.url("a.png")}, mutate=add_unknown
    )
    with pytest.raises(RetryableError, match="^input_asset_response_mismatch"):
        materialize(payload, "req-unknown", manifest=manifest, resolver=resolver)
    assert http_root.hits == {}


@pytest.mark.parametrize(
    "case,mutate",
    [
        ("missing", lambda assets: assets[:-1]),
        ("extra", lambda assets: assets + [dict(assets[0])]),
        ("reorder", lambda assets: list(reversed(assets))),
    ],
)
def test_response_count_order_rejected(
    http_root: HTTPRoot, case: str, mutate: Any
) -> None:
    payload = Payload(extra=[Asset(ref=REF_A), Asset(ref=REF_B)])
    manifest = (
        make_entry(REF_A, PNG_A, "media", "image/png"),
        make_entry(REF_B, PNG_B, "media", "image/png"),
    )
    resolver = FakeResolver(
        manifest=manifest,
        urls={REF_A: http_root.url("a.png"), REF_B: http_root.url("b.png")},
        mutate=mutate,
    )
    with pytest.raises(RetryableError, match="^input_asset_response_mismatch"):
        materialize(payload, f"req-resp-{case}", manifest=manifest, resolver=resolver)
    assert http_root.hits == {}


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("asset_id", "asset-other"),
        ("source_ref", "tenant/private/other"),
        ("blake3", "0" * 64),
        ("size_bytes", 999999),
        ("kind", "video"),
        ("mime_type", "image/jpeg"),
        ("url", ""),
        ("url_expires_at", "not-a-timestamp"),
    ],
)
def test_changed_immutable_response_field_rejected(
    http_root: HTTPRoot, field_name: str, value: Any
) -> None:
    payload = Payload(extra=[Asset(ref=REF_A)])
    manifest = (make_entry(REF_A, PNG_A, "media", "image/png"),)

    def change(assets: list[dict]) -> list[dict]:
        assets[0][field_name] = value
        return assets

    resolver = FakeResolver(
        manifest=manifest, urls={REF_A: http_root.url("a.png")}, mutate=change
    )
    with pytest.raises(RetryableError, match="^input_asset_response_mismatch"):
        materialize(payload, f"req-chg-{field_name}", manifest=manifest, resolver=resolver)
    assert http_root.hits == {}


def test_cancel_observed_at_resolve_boundary() -> None:
    canceled = threading.Event()
    payload = Payload(extra=[Asset(ref=REF_A)])
    manifest = (make_entry(REF_A, PNG_A, "media", "image/png"),)
    resolver = FakeResolver(manifest=manifest, urls={REF_A: "http://127.0.0.1:1/a"})
    resolver.on_call = canceled.set
    with pytest.raises(CanceledError):
        materialize(
            payload,
            "req-cancel-resolve",
            manifest=manifest,
            resolver=resolver,
            cancel_check=canceled.is_set,
        )
    assert len(resolver.calls) == 1
    assert not inputs_dir_for_request("req-cancel-resolve", 1).exists()


def test_blocked_resolved_url_rejected_without_download(
    http_root: HTTPRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(input_assets, "_url_is_blocked", lambda _url: True)
    payload = Payload(extra=[Asset(ref=REF_A)])
    manifest = (make_entry(REF_A, PNG_A, "media", "image/png"),)
    resolver = FakeResolver(manifest=manifest, urls={REF_A: http_root.url("a.png")})
    with pytest.raises(ValidationError, match="^input_asset_url_not_allowed") as caught:
        materialize(payload, "req-blocked", manifest=manifest, resolver=resolver)
    assert http_root.hits == {}
    assert_no_leak(caught.value, REF_A, http_root.base_url)


def test_internal_object_host_allowlist_only_for_resolver_minted(
    http_root: HTTPRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Deployment-declared internal object store: resolver-minted URLs to the
    # exact host pass; the same host stays blocked without the declaration.
    monkeypatch.setattr(input_assets, "_url_is_blocked", lambda _url: True)
    host = "127.0.0.1"
    payload = Payload(extra=[Asset(ref=REF_A)])
    manifest = (make_entry(REF_A, PNG_A, "media", "image/png"),)

    monkeypatch.delenv("GEN_WORKER_INTERNAL_OBJECT_HOSTS", raising=False)
    resolver = FakeResolver(manifest=manifest, urls={REF_A: http_root.url("a.png")})
    with pytest.raises(ValidationError, match="^input_asset_url_not_allowed"):
        materialize(payload, "req-internal-denied", manifest=manifest, resolver=resolver)

    monkeypatch.setenv("GEN_WORKER_INTERNAL_OBJECT_HOSTS", host)
    resolver = FakeResolver(manifest=manifest, urls={REF_A: http_root.url("a.png")})
    count = materialize(
        Payload(extra=[Asset(ref=REF_A)]),
        "req-internal-allowed",
        manifest=manifest,
        resolver=resolver,
    )
    assert count == 1


def test_internal_object_host_never_unblocks_caller_public_urls(
    http_root: HTTPRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(input_assets, "_url_is_blocked", lambda _url: True)
    monkeypatch.setenv("GEN_WORKER_INTERNAL_OBJECT_HOSTS", "127.0.0.1")
    payload = Payload(extra=[Asset(ref=http_root.url("pub.png"))])
    with pytest.raises(ValidationError, match="^input_asset_url_not_allowed"):
        materialize(payload, "req-public-still-blocked")
    assert http_root.hits == {}


# ---------------------------------------------------------------------------
# Download verification
# ---------------------------------------------------------------------------


def test_size_mismatch_is_retryable_integrity_failure(http_root: HTTPRoot) -> None:
    payload = Payload(extra=[Asset(ref=REF_A)])
    manifest = (
        make_entry(REF_A, PNG_A, "media", "image/png", size_bytes=len(PNG_A) + 1),
    )
    resolver = FakeResolver(manifest=manifest, urls={REF_A: http_root.url("a.png")})
    with pytest.raises(RetryableError, match="^input_asset_integrity_failed"):
        materialize(payload, "req-size", manifest=manifest, resolver=resolver)
    assert not inputs_dir_for_request("req-size", 1).exists()


def test_blake3_mismatch_is_retryable_integrity_failure(http_root: HTTPRoot) -> None:
    payload = Payload(extra=[Asset(ref=REF_A)])
    manifest = (make_entry(REF_A, PNG_A, "media", "image/png", blake3="0" * 64),)
    resolver = FakeResolver(manifest=manifest, urls={REF_A: http_root.url("a.png")})
    with pytest.raises(RetryableError, match="^input_asset_integrity_failed") as caught:
        materialize(payload, "req-hash", manifest=manifest, resolver=resolver)
    assert_no_leak(caught.value, REF_A, http_root.base_url)


def test_private_kind_and_mime_limits_enforced(http_root: HTTPRoot) -> None:
    # Declared AudioAsset (manifest kind audio) but the stored bytes are a PNG.
    payload = Payload(audio=AudioAsset(ref=REF_A))
    manifest = (make_entry(REF_A, PNG_A, "audio", "audio/mpeg"),)
    resolver = FakeResolver(manifest=manifest, urls={REF_A: http_root.url("a.png")})
    with pytest.raises(ValidationError, match="^input_asset_kind_mismatch"):
        materialize(payload, "req-kind", manifest=manifest, resolver=resolver)

    # Caller MIME allowlist still applies to private inputs.
    payload2 = Payload(
        extra=[Asset(ref=REF_A, url_allowed_mime_types=("audio/*",))]
    )
    manifest2 = (make_entry(REF_A, PNG_A, "media", "image/png"),)
    resolver2 = FakeResolver(manifest=manifest2, urls={REF_A: http_root.url("a.png")})
    with pytest.raises(ValidationError, match="^input_asset_kind_mismatch"):
        materialize(payload2, "req-allow", manifest=manifest2, resolver=resolver2)


def test_private_caller_byte_cap_precedes_download(http_root: HTTPRoot) -> None:
    payload = Payload(extra=[Asset(ref=REF_A, url_max_bytes=8)])
    manifest = (make_entry(REF_A, PNG_A, "media", "image/png"),)
    resolver = FakeResolver(manifest=manifest, urls={REF_A: http_root.url("a.png")})
    with pytest.raises(ValidationError, match="^input_asset_too_large"):
        materialize(payload, "req-cap", manifest=manifest, resolver=resolver)
    assert http_root.hits == {}


def test_private_corrupt_image_fails_decode(http_root: HTTPRoot) -> None:
    payload = Payload(image=ImageAsset(ref=REF_A))
    manifest = (make_entry(REF_A, CORRUPT_PNG, "image", "image/png"),)
    resolver = FakeResolver(manifest=manifest, urls={REF_A: http_root.url("corrupt.png")})
    with pytest.raises(ValidationError, match="^input_asset_decode_failed"):
        materialize(payload, "req-decode", manifest=manifest, resolver=resolver)
    assert not inputs_dir_for_request("req-decode", 1).exists()


def test_private_download_cancel_cleans_partial_file(http_root: HTTPRoot) -> None:
    asset = Asset(ref=REF_A)
    payload = Payload(extra=[asset])
    manifest = (make_entry(REF_A, SLOW_BODY, "media", "application/octet-stream"),)
    resolver = FakeResolver(manifest=manifest, urls={REF_A: http_root.url("slow.bin")})
    canceled = threading.Event()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            materialize,
            payload,
            "req-dl-cancel",
            manifest=manifest,
            resolver=resolver,
            cancel_check=canceled.is_set,
        )
        assert http_root.slow_started.wait(timeout=5)
        canceled.set()
        http_root.slow_release.set()
        with pytest.raises(CanceledError):
            future.result(timeout=10)

    assert asset.local_path is None
    assert not inputs_dir_for_request("req-dl-cancel", 1).exists()


def test_download_failure_cleans_prior_private_success(http_root: HTTPRoot) -> None:
    ok, failing = Asset(ref=REF_A), Asset(ref=REF_B)
    payload = Payload(extra=[ok, failing])
    manifest = (
        make_entry(REF_A, PNG_A, "media", "image/png"),
        make_entry(REF_B, PNG_B, "media", "image/png"),
    )
    resolver = FakeResolver(
        manifest=manifest,
        urls={REF_A: http_root.url("a.png"), REF_B: http_root.url("missing.bin")},
    )
    with pytest.raises(RetryableError, match="^input_asset_download_failed"):
        materialize(payload, "req-partial", manifest=manifest, resolver=resolver)
    assert ok.local_path is None and failing.local_path is None
    assert not inputs_dir_for_request("req-partial", 1).exists()


# ---------------------------------------------------------------------------
# Attempt fencing: N and N+1 never touch each other's scope
# ---------------------------------------------------------------------------


def test_attempt_scopes_are_isolated(http_root: HTTPRoot) -> None:
    first = Payload(image=ImageAsset(ref=http_root.url("a.png")))
    second = Payload(image=ImageAsset(ref=http_root.url("b.png")))
    materialize(first, "req-fence", attempt=1)
    materialize(second, "req-fence", attempt=2)
    dir1 = inputs_dir_for_request("req-fence", 1)
    dir2 = inputs_dir_for_request("req-fence", 2)
    try:
        assert dir1.exists() and dir2.exists() and dir1 != dir2
        assert first.image.local_path and second.image.local_path

        cleanup_input_assets("req-fence", 2)
        assert dir1.exists() and not dir2.exists()
        assert first.image.local_path is not None
        assert second.image.local_path is None
    finally:
        cleanup_input_assets("req-fence", 1)
        cleanup_input_assets("req-fence", 2)
    assert not dir1.exists()


class WorkerInput(msgspec.Struct):
    images: list[ImageAsset] = []


class WorkerOutput(msgspec.Struct):
    count: int


def test_executor_passes_manifest_and_capability(monkeypatch: pytest.MonkeyPatch, http_root: HTTPRoot) -> None:
    """The executor threads run.input_assets + file_base_url + capability +
    (request_id, attempt) + the cancel callback into the materializer."""
    manifest_rows = [
        pb.InputAsset(
            asset_id="asset-a",
            source_ref=REF_A,
            blake3=b3(PNG_A),
            size_bytes=len(PNG_A),
            kind="image",
            mime_type="image/png",
        )
    ]
    resolver = FakeResolver(
        manifest=manifest_from_run_job(manifest_rows),
        urls={REF_A: http_root.url("a.png")},
    )
    monkeypatch.setattr(input_assets, "_default_resolve_transport", resolver)

    seen: dict[str, Any] = {}

    def handler(ctx: Any, payload: WorkerInput) -> WorkerOutput:
        seen["ref"] = payload.images[0].ref
        seen["bytes"] = Path(payload.images[0].local_path or "").read_bytes()
        return WorkerOutput(count=len(payload.images))

    async def scenario() -> pb.JobResult:
        sent: list[pb.WorkerMessage] = []

        async def send(message: pb.WorkerMessage) -> None:
            sent.append(message)

        spec = EndpointSpec(
            name="edit",
            method=handler,
            kind="inference",
            payload_type=WorkerInput,
            output_mode="single",
        )
        executor = Executor([spec], send)
        executor.file_base_url = BASE_URL
        await executor.handle_run_job(
            pb.RunJob(
                request_id="exec-manifest",
                attempt=3,
                function_name="edit",
                input_payload=msgspec.msgpack.encode(
                    WorkerInput(images=[ImageAsset(ref=REF_A)])
                ),
                capability_token=CAPABILITY,
                input_assets=manifest_rows,
            )
        )
        job = executor.jobs[("exec-manifest", 3)]
        assert job.task is not None
        await job.task
        results = [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"]
        assert len(results) == 1
        return results[0]

    result = asyncio.run(scenario())
    assert result.status == pb.JOB_STATUS_OK
    assert seen["ref"] == REF_A and seen["bytes"] == PNG_A
    assert len(resolver.calls) == 1
    _, headers, body = resolver.calls[0]
    assert headers["Authorization"] == f"Bearer {CAPABILITY}"
    assert json.loads(body) == {"request_id": "exec-manifest", "attempt": 3}
    assert not inputs_dir_for_request("exec-manifest", 3).exists()


def test_executor_supersession_fences_result_and_scopes(http_root: HTTPRoot) -> None:
    """Attempt N canceled by N+1 publishes nothing and never touches N+1's
    inputs; N+1 completes normally."""
    handled: list[int] = []

    def handler(ctx: Any, payload: WorkerInput) -> WorkerOutput:
        handled.append(len(payload.images))
        return WorkerOutput(count=len(payload.images))

    async def scenario() -> list[tuple[int, int]]:
        sent: list[pb.WorkerMessage] = []

        async def send(message: pb.WorkerMessage) -> None:
            sent.append(message)

        spec = EndpointSpec(
            name="edit",
            method=handler,
            kind="inference",
            payload_type=WorkerInput,
            output_mode="single",
        )
        executor = Executor([spec], send)
        await executor.handle_run_job(
            pb.RunJob(
                request_id="exec-fence",
                attempt=1,
                function_name="edit",
                input_payload=msgspec.msgpack.encode(
                    WorkerInput(images=[ImageAsset(ref=http_root.url("slow.bin"))])
                ),
            )
        )
        assert await asyncio.to_thread(http_root.slow_started.wait, 5)
        await executor.handle_run_job(
            pb.RunJob(
                request_id="exec-fence",
                attempt=2,
                function_name="edit",
                input_payload=msgspec.msgpack.encode(
                    WorkerInput(images=[ImageAsset(ref=http_root.url("a.png"))])
                ),
            )
        )
        http_root.slow_release.set()
        for key in (("exec-fence", 1), ("exec-fence", 2)):
            job = executor.jobs[key]
            assert job.task is not None
            await asyncio.wait_for(job.task, timeout=10)
        return [
            (m.job_result.attempt, m.job_result.status)
            for m in sent
            if m.WhichOneof("msg") == "job_result"
        ]

    results = asyncio.run(scenario())
    # Attempt 1 was superseded: its terminal result is suppressed. Attempt 2
    # published exactly one OK result and both scopes are cleaned.
    assert results == [(2, pb.JOB_STATUS_OK)]
    assert handled == [1]
    assert not inputs_dir_for_request("exec-fence", 1).exists()
    assert not inputs_dir_for_request("exec-fence", 2).exists()


# ---------------------------------------------------------------------------
# Discovery: build-time schema contract
# ---------------------------------------------------------------------------


class DiscNested(msgspec.Struct):
    clips: list[VideoAsset] = []


class DiscPayload(msgspec.Struct):
    generic: Asset | None = None
    media: MediaAsset | None = None
    image: ImageAsset | None = None
    nested: DiscNested | None = None
    mapped: dict[str, AudioAsset] = {}


def test_discovery_emits_media_kind_for_generic_assets() -> None:
    meta = _collect_payload_moderation_metadata(DiscPayload)
    assert meta["media"] == [
        {"field": "generic", "kind": "media"},
        {"field": "media", "kind": "media"},
        {"field": "image", "kind": "image"},
        {"field": "nested.clips[]", "kind": "video"},
        {"field": "mapped.*", "kind": "audio"},
    ]


def test_discovery_rejects_asset_bearing_sets() -> None:
    class BadSet(msgspec.Struct):
        items: set[ImageAsset] = set()

    class BadFrozen(msgspec.Struct):
        items: frozenset[Asset] = frozenset()

    class NestedBad(msgspec.Struct):
        wrapped: set[DiscNested] = set()

    for bad in (BadSet, BadFrozen, NestedBad):
        with pytest.raises(ValueError, match="unordered set/frozenset"):
            _collect_payload_moderation_metadata(bad)

    class OkSet(msgspec.Struct):
        tags: set[str] = set()

    assert _collect_payload_moderation_metadata(OkSet) == {}


def test_discovery_rejects_non_string_key_asset_maps() -> None:
    class BadMap(msgspec.Struct):
        items: dict[int, ImageAsset] = {}

    with pytest.raises(ValueError, match="string keys"):
        _collect_payload_moderation_metadata(BadMap)

    class OkMap(msgspec.Struct):
        counts: dict[int, int] = {}

    assert _collect_payload_moderation_metadata(OkMap) == {}


# ---------------------------------------------------------------------------
# Proto descriptors: v4 wire contract
# ---------------------------------------------------------------------------


def test_protocol_version_is_5() -> None:
    assert pb.PROTOCOL_VERSION_CURRENT == 5


def test_run_job_input_assets_field_descriptor() -> None:
    # 15, not 14: th#913 claimed RunJob.lane=14 on chaos before the v4 landing.
    fd = pb.RunJob.DESCRIPTOR.fields_by_name["input_assets"]
    assert fd.number == 15
    assert fd.is_repeated
    assert fd.message_type is pb.InputAsset.DESCRIPTOR


def test_input_asset_field_numbers() -> None:
    fields = [(f.name, f.number) for f in pb.InputAsset.DESCRIPTOR.fields]
    assert fields == [
        ("asset_id", 1),
        ("source_ref", 2),
        ("blake3", 3),
        ("size_bytes", 4),
        ("kind", 5),
        ("mime_type", 6),
    ]


def test_manifest_from_run_job_round_trip() -> None:
    rows = [
        pb.InputAsset(
            asset_id="asset-a",
            source_ref=REF_A,
            blake3=b3(PNG_A),
            size_bytes=len(PNG_A),
            kind="image",
            mime_type="image/png",
        ),
        pb.InputAsset(
            asset_id="asset-b",
            source_ref=REF_B,
            blake3=b3(PNG_B),
            size_bytes=len(PNG_B),
            kind="media",
            mime_type="image/png",
        ),
    ]
    manifest = manifest_from_run_job(rows)
    assert manifest == (
        InputManifestEntry(
            asset_id="asset-a",
            source_ref=REF_A,
            blake3=b3(PNG_A),
            size_bytes=len(PNG_A),
            kind="image",
            mime_type="image/png",
        ),
        InputManifestEntry(
            asset_id="asset-b",
            source_ref=REF_B,
            blake3=b3(PNG_B),
            size_bytes=len(PNG_B),
            kind="media",
            mime_type="image/png",
        ),
    )
    assert manifest_from_run_job(()) == ()
