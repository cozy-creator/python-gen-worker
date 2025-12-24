import unittest

from gen_worker.errors import CanceledError, FatalError, ResourceError, RetryableError, ValidationError
from gen_worker.worker import Worker


class TestErrorMapping(unittest.TestCase):
    def _worker(self) -> Worker:
        # Avoid running Worker.__init__ (network/env-heavy); these helpers don't depend on init state.
        return Worker.__new__(Worker)

    def test_sanitize_safe_message_redacts_tokens_urls_paths(self) -> None:
        w = self._worker()
        msg = "Bearer abc.def.ghi https://example.com/secret /home/user/token.txt"
        out = w._sanitize_safe_message(msg)
        self.assertNotIn("abc.def.ghi", out)
        self.assertNotIn("https://example.com/secret", out)
        self.assertNotIn("/home/user/token.txt", out)

    def test_map_exception_validation(self) -> None:
        w = self._worker()
        error_type, retryable, safe, internal = w._map_exception(ValidationError("bad input"))
        self.assertEqual(error_type, "validation")
        self.assertFalse(retryable)
        self.assertIn("bad input", safe)
        self.assertIn("ValidationError", internal)

    def test_map_exception_retryable(self) -> None:
        w = self._worker()
        error_type, retryable, safe, _ = w._map_exception(RetryableError("temporary"))
        self.assertEqual(error_type, "retryable")
        self.assertTrue(retryable)
        self.assertIn("temporary", safe)

    def test_map_exception_fatal(self) -> None:
        w = self._worker()
        error_type, retryable, safe, _ = w._map_exception(FatalError("no"))
        self.assertEqual(error_type, "fatal")
        self.assertFalse(retryable)
        self.assertIn("no", safe)

    def test_map_exception_resource(self) -> None:
        w = self._worker()
        error_type, retryable, safe, _ = w._map_exception(ResourceError("oom"))
        self.assertEqual(error_type, "resource")
        self.assertFalse(retryable)
        self.assertIn("oom", safe)

    def test_map_exception_canceled(self) -> None:
        w = self._worker()
        error_type, retryable, safe, _ = w._map_exception(CanceledError("stop"))
        self.assertEqual(error_type, "canceled")
        self.assertFalse(retryable)
        self.assertEqual(safe, "canceled")


if __name__ == "__main__":
    unittest.main()

