"""Pytest pinning the `attributes=` kwarg contract on save_checkpoint /
open_checkpoint_stream.

Issue #63: a prior session shipped the `attributes` wire end-to-end after a
TypeError surfaced in production; this test guards the kwarg-passthrough
half of the fix so a future refactor can't quietly drop it again.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest


# -----------------------------------------------------------------------
# Signature contracts
# -----------------------------------------------------------------------

def test_save_checkpoint_accepts_attributes_kwarg() -> None:
    """`RequestContext.save_checkpoint` must expose an `attributes`
    keyword argument typed as a Mapping[str, str]. The conversion-dispatch
    path passes attributes={...}; if this kwarg disappears the contract
    we shipped is broken."""
    from gen_worker.request_context import RequestContext

    sig = inspect.signature(RequestContext.save_checkpoint)
    assert "attributes" in sig.parameters, (
        "save_checkpoint must accept `attributes` kwarg (issue #63 contract)"
    )
    param = sig.parameters["attributes"]
    assert param.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD), (
        f"`attributes` must be keyword-callable; got {param.kind}"
    )
    # Default is None (Optional). Most importantly: the parameter exists at all.
    assert param.default is None, (
        f"`attributes` should default to None for backward-compat; got {param.default!r}"
    )


def test_open_checkpoint_stream_accepts_attributes_kwarg() -> None:
    """Mirror contract for the streaming entry point — same kwarg lives on
    `open_checkpoint_stream`, which is what the conversion dispatch
    actually calls when streaming a flavor."""
    from gen_worker.request_context import RequestContext

    sig = inspect.signature(RequestContext.open_checkpoint_stream)
    assert "attributes" in sig.parameters, (
        "open_checkpoint_stream must accept `attributes` kwarg"
    )
    assert sig.parameters["attributes"].default is None


# -----------------------------------------------------------------------
# Stream stores attributes verbatim and threads them onto /complete
# -----------------------------------------------------------------------

class _StubCtx:
    """Minimal stand-in for RequestContext just so _RequestOutputStream
    can construct without touching real upload code paths.

    The stream constructor only needs:
      - request_id (string used in tempfile prefix)
      - _should_stream_output_to_file_api(ref) → bool
      - _repo_job_upload_scope() → Optional[tuple]
    None of these need to dispatch to live infra in this test."""

    def __init__(self, repo_scope: Optional[tuple[str, str, str]] = None) -> None:
        self.request_id = "req-test-123"
        self._scope = repo_scope

    def _should_stream_output_to_file_api(self, _ref: str) -> bool:
        # Force the local-fallback path so finalize doesn't attempt a real
        # upload. We're testing attribute storage, not network code.
        return False

    def _repo_job_upload_scope(self) -> Optional[tuple[str, str, str]]:
        return self._scope

    def is_canceled(self) -> bool:
        return False


def _make_stream(attributes, repo_scope=None):
    from gen_worker.request_context._stream import _RequestOutputStream

    ctx = _StubCtx(repo_scope=repo_scope)
    return _RequestOutputStream(
        ctx=ctx,
        ref="weights/checkpoint.safetensors",
        kind="checkpoint",
        format="safetensors",
        expected_size_bytes=4,
        attributes=attributes,
    )


def test_stream_stores_lineage_attributes_normalized() -> None:
    """Attributes should be stored as a {str: str} dict regardless of the
    concrete Mapping type the caller hands in. Whitespace-only keys are
    dropped (same shape used by `_lineage_attributes` consumer code)."""
    stream = _make_stream({"recipe": "fp8_wo", "calib_dataset": "partiprompts-256", "  ": "drop"})
    try:
        assert stream._lineage_attributes == {
            "recipe": "fp8_wo",
            "calib_dataset": "partiprompts-256",
        }
    finally:
        # Clean up the tempfile the stream creates.
        try:
            os.remove(stream._tmp_path)
        except OSError:
            pass


def test_stream_lineage_attributes_empty_when_unset() -> None:
    stream = _make_stream(None)
    try:
        assert stream._lineage_attributes == {}
    finally:
        try:
            os.remove(stream._tmp_path)
        except OSError:
            pass


def test_stream_lineage_attributes_coerces_non_string_values() -> None:
    """Tenants sometimes pass numeric/bool values; the wire is
    string-keyed string-valued (server schema), so coerce at the boundary
    rather than 400ing the request late."""
    stream = _make_stream({"step": 100, "is_final": True})
    try:
        # Both values stringified to match the wire schema.
        assert stream._lineage_attributes == {"step": "100", "is_final": "True"}
    finally:
        try:
            os.remove(stream._tmp_path)
        except OSError:
            pass


# -----------------------------------------------------------------------
# Complete-payload extra carries the attributes when repo_job scope is set
# -----------------------------------------------------------------------

def test_complete_extra_includes_attributes_when_scoped() -> None:
    """The `complete_extra` payload (sent on the upload-session /complete
    endpoint at finalize) is the wire that carries attributes to
    tensorhub. When the request is scoped to a repo job AND attributes
    are set, the dict must include `attributes`."""
    # We can't easily exercise the live upload path, but the payload
    # construction logic is straightforward to mirror via the field
    # snapshot. Test it by reading the same fields the stream uses.
    stream = _make_stream(
        {"recipe": "fp8_wo"},
        repo_scope=("owner-x", "repo-y", "session-z"),
    )
    try:
        # Mirror the inline complete_extra-building shape from
        # request_context._stream._upload_to_file_api (around line 380).
        # The point of this test is that whatever we construct from the
        # snapshot matches what the wire shape requires.
        extra: dict = {}
        if stream._lineage_produced_by_kind:
            extra["produced_by_kind"] = stream._lineage_produced_by_kind
        if stream._lineage_flavor:
            extra["flavor"] = stream._lineage_flavor
        if stream._lineage_attributes:
            extra["attributes"] = dict(stream._lineage_attributes)

        assert "attributes" in extra
        assert extra["attributes"] == {"recipe": "fp8_wo"}
    finally:
        try:
            os.remove(stream._tmp_path)
        except OSError:
            pass


def test_complete_extra_omits_attributes_when_empty() -> None:
    """Empty attributes mapping should NOT add an `attributes` key —
    tensorhub's strict JSON decoder treats `{}` differently from absent,
    and we want the wire to stay minimal."""
    stream = _make_stream(
        None,
        repo_scope=("owner-x", "repo-y", "session-z"),
    )
    try:
        extra: dict = {}
        if stream._lineage_attributes:
            extra["attributes"] = dict(stream._lineage_attributes)
        assert "attributes" not in extra
    finally:
        try:
            os.remove(stream._tmp_path)
        except OSError:
            pass
