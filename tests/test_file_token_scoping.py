"""Tests for per-run file token scoping (issue #50)."""
import os
import tempfile
import unittest
from typing import Any, Dict, Optional
from unittest.mock import patch
import urllib.error

from gen_worker.errors import AuthError
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


def _make_http_error(code: int, msg: str = "error") -> urllib.error.HTTPError:
    """Create a real HTTPError for testing."""
    import io
    return urllib.error.HTTPError(
        url="https://example.com",
        code=code,
        msg=msg,
        hdrs={},  # type: ignore
        fp=io.BytesIO(b""),
    )


class TestFileTokenScoping(unittest.TestCase):
    """Test that per-run file tokens are used instead of env vars."""

    def test_save_bytes_uses_per_run_token(self) -> None:
        """save_bytes should use the token from ActionContext, not env."""
        captured_auth: list[str] = []

        def fake_urlopen(req: Any, timeout: int = 0) -> _FakeHTTPResponse:
            items = getattr(req, "header_items", None)
            if callable(items):
                for k, v in items():
                    if str(k).lower() == "authorization":
                        captured_auth.append(str(v))
            return _FakeHTTPResponse(b'{"size_bytes": 5, "sha256": "abc123"}', status=200)

        # Set env var to a different value to prove it's not used
        os.environ["FILE_API_TOKEN"] = "env-token-should-not-be-used"
        os.environ["FILE_API_BASE_URL"] = "https://should-not-be-used.example"

        ctx = ActionContext(
            "run-123",
            owner="tenant-1",
            file_api_base_url="https://cozy-hub.example",
            file_api_token="per-run-token",
        )

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            asset = ctx.save_bytes("runs/run-123/outputs/test.bin", b"hello")

        self.assertEqual(len(captured_auth), 1)
        self.assertEqual(captured_auth[0], "Bearer per-run-token")
        self.assertIsNotNone(asset)

    def test_save_bytes_falls_back_to_env_when_no_per_run_token(self) -> None:
        """save_bytes should fall back to env var if no per-run token provided."""
        captured_auth: list[str] = []

        def fake_urlopen(req: Any, timeout: int = 0) -> _FakeHTTPResponse:
            items = getattr(req, "header_items", None)
            if callable(items):
                for k, v in items():
                    if str(k).lower() == "authorization":
                        captured_auth.append(str(v))
            return _FakeHTTPResponse(b'{"size_bytes": 5, "sha256": "abc123"}', status=200)

        os.environ["FILE_API_TOKEN"] = "env-fallback-token"
        os.environ["FILE_API_BASE_URL"] = "https://cozy-hub.example"

        ctx = ActionContext(
            "run-456",
            owner="tenant-1",
            # No file_api_token provided
        )

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            asset = ctx.save_bytes("runs/run-456/outputs/test.bin", b"hello")

        self.assertEqual(len(captured_auth), 1)
        self.assertEqual(captured_auth[0], "Bearer env-fallback-token")
        self.assertIsNotNone(asset)


class TestAuthErrorHandling(unittest.TestCase):
    """Test that 401/403 errors raise AuthError (non-retryable)."""

    def test_save_bytes_raises_auth_error_on_401(self) -> None:
        """save_bytes should raise AuthError on 401 response."""
        def fake_urlopen(req: Any, timeout: int = 0) -> _FakeHTTPResponse:
            raise _make_http_error(401, "Unauthorized")

        ctx = ActionContext(
            "run-789",
            owner="tenant-1",
            file_api_base_url="https://cozy-hub.example",
            file_api_token="expired-token",
        )

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(AuthError) as cm:
                ctx.save_bytes("runs/run-789/outputs/test.bin", b"hello")

        self.assertIn("401", str(cm.exception))
        self.assertIn("file_token", str(cm.exception))

    def test_save_bytes_raises_auth_error_on_403(self) -> None:
        """save_bytes should raise AuthError on 403 response."""
        def fake_urlopen(req: Any, timeout: int = 0) -> _FakeHTTPResponse:
            raise _make_http_error(403, "Forbidden")

        ctx = ActionContext(
            "run-abc",
            owner="tenant-1",
            file_api_base_url="https://cozy-hub.example",
            file_api_token="wrong-scope-token",
        )

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(AuthError) as cm:
                ctx.save_bytes("runs/run-abc/outputs/test.bin", b"hello")

        self.assertIn("403", str(cm.exception))

    def test_save_bytes_create_raises_auth_error_on_401(self) -> None:
        """save_bytes_create should raise AuthError on 401 response."""
        def fake_urlopen(req: Any, timeout: int = 0) -> _FakeHTTPResponse:
            raise _make_http_error(401, "Unauthorized")

        ctx = ActionContext(
            "run-def",
            owner="tenant-1",
            file_api_base_url="https://cozy-hub.example",
            file_api_token="bad-token",
        )

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(AuthError):
                ctx.save_bytes_create("runs/run-def/outputs/new.bin", b"data")

    def test_materialize_asset_raises_auth_error_on_401(self) -> None:
        """_materialize_asset should raise AuthError on 401 for HEAD request."""
        w = Worker.__new__(Worker)
        w.owner = "tenant-1"

        def fake_urlopen(req: Any, timeout: int = 0) -> _FakeHTTPResponse:
            raise _make_http_error(401, "Unauthorized")

        a = Asset(ref="some-file.png")

        with tempfile.TemporaryDirectory() as td:
            os.environ["WORKER_RUN_DIR"] = td
            os.environ["WORKER_CACHE_DIR"] = os.path.join(td, "cache")

            ctx = ActionContext(
                "run-ghi",
                owner="tenant-1",
                file_api_base_url="https://cozy-hub.example",
                file_api_token="expired-token",
            )

            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                with self.assertRaises(AuthError) as cm:
                    w._materialize_asset(ctx, a)

            self.assertIn("401", str(cm.exception))

    def test_materialize_asset_raises_auth_error_on_403(self) -> None:
        """_materialize_asset should raise AuthError on 403 for HEAD request."""
        w = Worker.__new__(Worker)
        w.owner = "tenant-1"

        def fake_urlopen(req: Any, timeout: int = 0) -> _FakeHTTPResponse:
            raise _make_http_error(403, "Forbidden - path not in allowed prefixes")

        a = Asset(ref="other-run/outputs/secret.png")

        with tempfile.TemporaryDirectory() as td:
            os.environ["WORKER_RUN_DIR"] = td
            os.environ["WORKER_CACHE_DIR"] = os.path.join(td, "cache")

            ctx = ActionContext(
                "run-jkl",
                owner="tenant-1",
                file_api_base_url="https://cozy-hub.example",
                file_api_token="scoped-token",
            )

            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                with self.assertRaises(AuthError) as cm:
                    w._materialize_asset(ctx, a)

            self.assertIn("403", str(cm.exception))


class TestAuthErrorMapping(unittest.TestCase):
    """Test that AuthError is mapped to non-retryable error type."""

    def test_auth_error_is_non_retryable(self) -> None:
        """AuthError should map to 'auth' error type with retryable=False."""
        w = Worker.__new__(Worker)
        w.owner = "tenant-1"

        exc = AuthError("token expired")
        error_type, retryable, safe_msg, internal_msg = w._map_exception(exc)

        self.assertEqual(error_type, "auth")
        self.assertFalse(retryable)
        # safe_msg contains the exception message
        self.assertIn("token expired", safe_msg.lower())

    def test_auth_error_default_message(self) -> None:
        """AuthError with empty message should use 'authentication failed'."""
        w = Worker.__new__(Worker)
        w.owner = "tenant-1"

        exc = AuthError("")
        error_type, retryable, safe_msg, internal_msg = w._map_exception(exc)

        self.assertEqual(error_type, "auth")
        self.assertFalse(retryable)
        self.assertIn("authentication", safe_msg.lower())


if __name__ == "__main__":
    unittest.main()
