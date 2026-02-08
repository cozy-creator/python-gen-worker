import hashlib
import os
import tempfile
import unittest
from typing import Any, Dict, Optional
from unittest.mock import patch

from gen_worker.types import Asset
from gen_worker.worker import ActionContext, Worker


class _FakeHeaders(dict):
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:  # type: ignore[override]
        return super().get(key, default)


class _FakeHTTPResponse:
    def __init__(self, body: bytes, status: int = 200, headers: Optional[Dict[str, str]] = None) -> None:
        self._body = body
        self._pos = 0
        self.status = status
        self.headers: Any = _FakeHeaders(headers or {})

    def read(self, n: int = -1) -> bytes:
        if self._pos >= len(self._body):
            return b""
        if n is None or n < 0:
            n = len(self._body) - self._pos
        chunk = self._body[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class TestAssetMaterialization(unittest.TestCase):
    def _worker(self, owner: str = "tenant-1") -> Worker:
        w = Worker.__new__(Worker)
        w.owner = owner
        return w

    def test_materialize_external_url(self) -> None:
        w = self._worker()
        # Use a literal public IP so SSRF/DNS resolution doesn't depend on DNS working.
        a = Asset(ref="https://1.1.1.1/a.png")
        data = b"\x89PNG\r\n\x1a\nhello"

        with tempfile.TemporaryDirectory() as td:
            os.environ["WORKER_RUN_DIR"] = td
            os.environ["WORKER_CACHE_DIR"] = os.path.join(td, "cache")

            class _Opener:
                def open(self, req: Any, timeout: int = 0) -> _FakeHTTPResponse:
                    _ = req, timeout
                    return _FakeHTTPResponse(data)

            with patch("urllib.request.build_opener", return_value=_Opener()) as _mock:
                w._materialize_asset(ActionContext("run-1", owner=w.owner), a)
                self.assertGreaterEqual(_mock.call_count, 1)

            self.assertIsNotNone(a.local_path)
            assert a.local_path is not None
            with open(a.local_path, "rb") as f:
                self.assertEqual(f.read(), data)

            self.assertEqual(a.size_bytes, len(data))
            self.assertEqual(a.sha256, hashlib.sha256(data).hexdigest())
            self.assertEqual(a.mime_type, "image/png")

    def test_materialize_external_url_size_cap(self) -> None:
        w = self._worker()
        a = Asset(ref="https://1.1.1.1/a.bin")
        data = b"1234"

        with tempfile.TemporaryDirectory() as td:
            os.environ["WORKER_RUN_DIR"] = td
            os.environ["WORKER_CACHE_DIR"] = os.path.join(td, "cache")
            os.environ["WORKER_MAX_INPUT_FILE_BYTES"] = "1"

            class _Opener:
                def open(self, req: Any, timeout: int = 0) -> _FakeHTTPResponse:
                    _ = req, timeout
                    return _FakeHTTPResponse(data)

            with patch("urllib.request.build_opener", return_value=_Opener()):
                with self.assertRaises(Exception):
                    w._materialize_asset(ActionContext("run-1", owner=w.owner), a)

    def test_materialize_cozy_hub_ref(self) -> None:
        w = self._worker(owner="tenant-1")
        a = Asset(ref="my-uploads/cat.png")
        body = b"\x89PNG\r\n\x1a\ncat"
        sha = hashlib.sha256(body).hexdigest()

        def fake_urlopen(req: Any, timeout: int = 0) -> _FakeHTTPResponse:
            # HEAD request (urllib.request.Request)
            method = getattr(req, "method", None)
            items = getattr(req, "header_items", None)
            hdrs: Dict[str, str] = {}
            if callable(items):
                for k, v in items():
                    hdrs[str(k).lower()] = str(v)
            if method == "HEAD":
                self.assertEqual(hdrs.get("authorization"), "Bearer tok")
                self.assertEqual(hdrs.get("x-cozy-owner"), "tenant-1")
                return _FakeHTTPResponse(
                    b"",
                    status=200,
                    headers={
                        "X-Cozy-Size-Bytes": str(len(body)),
                        "X-Cozy-SHA256": sha,
                        "X-Cozy-Mime-Type": "image/png",
                    },
                )
            # GET request
            self.assertEqual(hdrs.get("authorization"), "Bearer tok")
            self.assertEqual(hdrs.get("x-cozy-owner"), "tenant-1")
            return _FakeHTTPResponse(body, status=200, headers={"Content-Type": "image/png"})

        with tempfile.TemporaryDirectory() as td:
            os.environ["WORKER_RUN_DIR"] = td
            os.environ["WORKER_CACHE_DIR"] = os.path.join(td, "cache")
            os.environ["WORKER_MAX_INPUT_FILE_BYTES"] = "9999999"

            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                ctx = ActionContext(
                    "run-1",
                    owner="tenant-1",
                    file_api_base_url="https://cozy-hub.example",
                    file_api_token="tok",
                )
                w._materialize_asset(ctx, a)

            self.assertIsNotNone(a.local_path)
            assert a.local_path is not None
            with open(a.local_path, "rb") as f:
                self.assertEqual(f.read(), body)
            self.assertEqual(a.mime_type, "image/png")
            self.assertEqual(a.size_bytes, len(body))
            self.assertEqual(a.sha256, sha)


if __name__ == "__main__":
    unittest.main()
