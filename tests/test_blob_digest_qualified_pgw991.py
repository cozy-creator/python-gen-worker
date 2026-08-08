"""pgw#991: a bare-hex blob address is refused, not tagged `blake3:`.

`_download_blob_by_digest` used to promote a bare hex string to a digest by
prefixing `blake3:`. That was wrong twice over: the repo-CAS has been sha256
since th#1303, so the guess addressed a namespace the blob was not in; and the
hub's own `storage.ParseDigest` (fronted by `validateDigestParam` on the
by-digest route th#1641 added) REFUSES bare hex outright, so even a correct
guess is not the contract.

The hub in this rig answers the by-digest route exactly like the real one —
`400 invalid_digest` for anything `ParseDigest` rejects — so a regression that
re-introduces the guess fails here on the hub's own rule rather than on a
worker-side assertion about it.
"""
from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote

import pytest
from blake3 import blake3

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.api.errors import (
    BlobDigestMalformedError,
    PayloadRefError,
    ValidationError,
)
from gen_worker.executor import _map_exception
from gen_worker.request_context import (
    REF_ORIGIN_PLATFORM,
    ConversionContext,
)

BLOB_BYTES = b"real blob bytes"
# pgw#1013: both addresses are derived from the bytes the rig serves, because
# `_download_blob_by_digest` now verifies that the bytes hash to the digest
# that named them. The subject of this file is unchanged — which ADDRESSES are
# well-formed — but a rig serving one blob under an unrelated digest describes
# a content-addressed store that does not content-address.
BARE_HEX = hashlib.sha256(BLOB_BYTES).hexdigest()
SHA256_DIGEST = f"sha256:{BARE_HEX}"
BLAKE3_DIGEST = "blake3:" + blake3(BLOB_BYTES).hexdigest()

# Every algorithm tensorhub `storage.casSupportedAlgos` keys on, and its width.
_SUPPORTED = {"blake3": 64, "sha256": 64}

_REQUESTED: list[str] = []


def _hub_parse_digest(ref: str) -> str | None:
    """tensorhub `internal/storage/cas_paths.go` ParseDigest, transcribed."""
    ref = ref.strip()
    if not ref:
        return None
    algo, sep, hexpart = ref.partition(":")
    if not sep:
        return None  # "bare hex is refused"
    algo = algo.strip().lower()
    hexpart = hexpart.strip()
    width = _SUPPORTED.get(algo)
    if width is None:
        return None
    if any(c not in "0123456789abcdefABCDEF" for c in hexpart):
        return None
    if len(hexpart) != width:
        return None
    return f"{algo}:{hexpart.lower()}"


class _Hub(BaseHTTPRequestHandler):
    def log_message(self, *_a: object) -> None:
        pass

    def _send(self, code: int, body: bytes = b"") -> None:
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path
        if "/blobs/" not in path or not path.endswith("/content"):
            return self._send(404, b"")
        raw = unquote(path.split("/blobs/", 1)[1][: -len("/content")])
        _REQUESTED.append(raw)
        if _hub_parse_digest(raw) is None:
            # api.WriteError(c, http.StatusBadRequest, "invalid_digest", "")
            return self._send(400, b'{"error":{"code":"invalid_digest"}}')
        return self._send(200, BLOB_BYTES)


@pytest.fixture()
def hub() -> Iterator[str]:
    _REQUESTED.clear()
    srv = HTTPServer(("127.0.0.1", 0), _Hub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_port}"
    finally:
        srv.shutdown()
        srv.server_close()


def _ctx(hub_url: str) -> ConversionContext:
    ctx = ConversionContext(request_id="r-pgw991")
    ctx._file_api_base_url = hub_url
    ctx._worker_capability_token = "test-token"
    return ctx


def test_bare_hex_is_refused_before_any_request(hub: str, tmp_path: Path) -> None:
    """The filed defect. Bare hex is a malformed ADDRESS, and the caller owns
    it — so it fails the REQUEST typed, and no HTTP call is made at all."""
    ctx = _ctx(hub)

    with pytest.raises(BlobDigestMalformedError) as ei:
        ctx.materialize_blob(BARE_HEX, tmp_path / "out.bin")

    exc = ei.value
    assert exc.code == "blob_digest_malformed"
    assert exc.ref == BARE_HEX
    assert isinstance(exc, PayloadRefError)
    assert isinstance(exc, ValidationError)
    # The refusal names the fix, in the hub's own words.
    assert "sha256:<64 hex>" in str(exc)
    assert _REQUESTED == [], "a malformed address must not reach the network"


def test_bare_hex_maps_to_INVALID_not_FATAL(hub: str, tmp_path: Path) -> None:
    """th#1259's rule holds for this class too: a caller's bad address is
    never model-health evidence."""
    ctx = _ctx(hub)
    with pytest.raises(BlobDigestMalformedError) as ei:
        ctx.materialize_blob(BARE_HEX, tmp_path / "out.bin")
    status, message = _map_exception(ei.value)
    assert status == pb.JOB_STATUS_INVALID
    assert message.startswith("blob_digest_malformed: ")


def test_the_old_blake3_guess_is_what_the_hub_rejects(hub: str, tmp_path: Path) -> None:
    """Guard on the regression itself. Had the code kept tagging bare hex
    `blake3:`, this sha256 blob would have been fetched from the blake3
    namespace — a wrong address that the route answers, not an error. The
    qualified form is the ONLY thing that reaches the right one."""
    ctx = _ctx(hub)
    out = ctx.materialize_blob(SHA256_DIGEST, tmp_path / "ok.bin")
    assert out.read_bytes() == BLOB_BYTES
    assert _REQUESTED == [SHA256_DIGEST], (
        "the digest must reach the hub verbatim and algorithm-tagged; "
        f"a blake3: guess would have sent blake3:{BARE_HEX}"
    )


def test_blake3_still_works_for_the_dataset_cas(hub: str, tmp_path: Path) -> None:
    """blake3 is not dead — it is the dataset-CAS contract. A caller that
    declares it explicitly is still served."""
    ctx = _ctx(hub)
    out = ctx.materialize_blob(BLAKE3_DIGEST, tmp_path / "ds.bin")
    assert out.read_bytes() == BLOB_BYTES
    assert _REQUESTED == [BLAKE3_DIGEST]


def test_platform_origin_malformed_stays_fatal(hub: str, tmp_path: Path) -> None:
    """A malformed address the PLATFORM produced is a platform fault, so it
    keeps the fatal classification instead of blaming the caller."""
    ctx = _ctx(hub)
    with pytest.raises(RuntimeError) as ei:
        ctx.materialize_blob(BARE_HEX, tmp_path / "p.bin", origin=REF_ORIGIN_PLATFORM)
    assert not isinstance(ei.value, PayloadRefError)
    assert "malformed platform blob digest" in str(ei.value)
    assert _REQUESTED == []


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        BARE_HEX,                       # not algorithm-tagged
        f"md5:{BARE_HEX}",              # unsupported algorithm
        "sha256:" + "zz" * 32,          # non-hex character
        "sha256:" + "ab" * 16,          # wrong width
        "sha256:",                      # empty hex
    ],
)
def test_worker_refusal_matches_the_hub_exactly(hub: str, tmp_path: Path, bad: str) -> None:
    """Parity, asserted rather than asserted-about: every address the worker
    refuses is one the hub's ParseDigest also refuses. A worker that refused
    MORE would strand a legal address; one that refused LESS would ship a 400
    to a pod that could have named the fault locally."""
    assert _hub_parse_digest(bad) is None, "rig disagrees with the hub's rule"
    ctx = _ctx(hub)
    with pytest.raises(BlobDigestMalformedError):
        ctx.materialize_blob(bad, tmp_path / "x.bin")
    assert _REQUESTED == []


@pytest.mark.parametrize(
    "good",
    [SHA256_DIGEST, SHA256_DIGEST.upper().replace("SHA256", "sha256"), BLAKE3_DIGEST],
)
def test_worker_accepts_everything_the_hub_accepts(hub: str, tmp_path: Path, good: str) -> None:
    canonical = _hub_parse_digest(good)
    assert canonical is not None
    ctx = _ctx(hub)
    ctx.materialize_blob(good, tmp_path / "y.bin")
    # Canonicalised the same way the hub canonicalises it (lowercased hex).
    assert _REQUESTED == [canonical]
