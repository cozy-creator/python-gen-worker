"""Tests for Worker's handling of EndpointConfig disabled_functions and
ref_availability_by_function.

These tests poke directly at Worker's public predicate methods
(is_function_runnable, payload_key_status) after seeding the internal
availability state — isolated from the full EndpointConfig proto-decoding
path so we don't need a running orchestrator.
"""

import unittest
from unittest.mock import MagicMock


class _WorkerAvailabilityMixin:
    """Helper that creates a Worker shell with only the availability state
    we care about. Avoids the cost of constructing a full Worker (which
    expects a live orchestrator connection)."""

    def make_worker_shell(self) -> object:
        # Construct via __new__ so we can skip the full __init__ network
        # setup — the predicate methods only read dict state.
        from gen_worker.worker import Worker

        w = Worker.__new__(Worker)
        w._disabled_functions_by_name = {}
        w._payload_ref_availability_by_function = {}
        return w


class TestWorkerIsFunctionRunnable(unittest.TestCase, _WorkerAvailabilityMixin):
    def test_empty_state_defaults_to_runnable(self) -> None:
        w = self.make_worker_shell()
        self.assertTrue(w.is_function_runnable("any_function"))

    def test_disabled_function_reports_not_runnable(self) -> None:
        w = self.make_worker_shell()
        w._disabled_functions_by_name = {
            "fn_risky": {
                "function_name": "fn_risky",
                "ref": "other/private-model",
                "reason": "read_access_revoked",
            }
        }
        self.assertFalse(w.is_function_runnable("fn_risky"))
        self.assertTrue(w.is_function_runnable("fn_safe"))

    def test_blank_function_name_allowed(self) -> None:
        w = self.make_worker_shell()
        # Defensive — a missing/blank name isn't a match against the map.
        self.assertTrue(w.is_function_runnable(""))


class TestWorkerPayloadKeyStatus(unittest.TestCase, _WorkerAvailabilityMixin):
    def test_unknown_function_returns_none(self) -> None:
        w = self.make_worker_shell()
        self.assertIsNone(w.payload_key_status("fn_picker", "small"))

    def test_known_function_and_key_returns_status(self) -> None:
        w = self.make_worker_shell()
        w._payload_ref_availability_by_function = {
            "fn_picker": {
                "small": {"ref": "paul/model-small", "status": "resolved"},
                "large": {"ref": "paul/model-large", "status": "repo_deleted"},
            }
        }
        self.assertEqual(w.payload_key_status("fn_picker", "small"), "resolved")
        self.assertEqual(w.payload_key_status("fn_picker", "large"), "repo_deleted")

    def test_unknown_key_on_known_function_returns_none(self) -> None:
        w = self.make_worker_shell()
        w._payload_ref_availability_by_function = {
            "fn_picker": {"small": {"ref": "paul/model-small", "status": "resolved"}}
        }
        # An unregistered key means "no status tracked" — callers should
        # proceed as-normal (request will hit the release's model spec for
        # actual resolution). Distinct from "resolved".
        self.assertIsNone(w.payload_key_status("fn_picker", "unknown"))


class TestFilterPrefetchForDisabledFunctions(unittest.TestCase, _WorkerAvailabilityMixin):
    def test_no_disabled_returns_refs_unchanged(self) -> None:
        w = self.make_worker_shell()
        refs = ["cozy:owner/repo@sha256:abc", "cozy:owner/other@sha256:def"]
        self.assertEqual(w._filter_prefetch_for_disabled_functions(refs), refs)

    def test_disabled_function_refs_logged_but_retained(self) -> None:
        # Current implementation is conservative — doesn't have enough
        # info to know which refs are shared across functions, so it
        # retains the set but emits a log line. This test documents the
        # current conservative behavior.
        w = self.make_worker_shell()
        w._disabled_functions_by_name = {
            "fn_risky": {"ref": "cozy:owner/repo@sha256:abc", "reason": "read_access_revoked"}
        }
        refs = ["cozy:owner/repo@sha256:abc", "cozy:owner/other@sha256:def"]
        # Function should still return all refs; orchestrator already narrowed.
        self.assertEqual(w._filter_prefetch_for_disabled_functions(refs), refs)


class TestCanonicalizeResolvedModelsMap(unittest.TestCase):
    def test_digest_ref_with_flavor_adds_matching_tag_alias(self) -> None:
        from gen_worker.worker import Worker

        resolved = object()
        out = Worker._canonicalize_resolved_models_map(
            {"cozy:owner/repo@blake3:snap#int4": resolved}
        )

        self.assertIs(out["cozy:owner/repo@blake3:snap#int4"], resolved)
        self.assertIs(out["cozy:owner/repo:latest#int4"], resolved)
        self.assertNotIn("cozy:owner/repo:latest", out)


if __name__ == "__main__":
    unittest.main()
