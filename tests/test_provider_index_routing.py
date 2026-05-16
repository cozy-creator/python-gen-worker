"""Tests for issue #17: HF-provider refs from supported_repo_refs must
route to HuggingFaceHubDownloader, not the tensorhub resolved_repos lookup.

The bug: `ModelRefDownloader.download(model_ref)` called
`parse_model_ref(model_ref)` with no `provider=` kwarg, so every ref was
parsed as `provider="tensorhub"`. For an HF-provider binding like
`black-forest-labs/FLUX.2-klein-base-4B#bf16` that hit the tensorhub branch
in `_download_async` and raised the "ref not in resolved_repos_by_id" error.

The fix: build a `{bare_ref -> provider}` index from the baked endpoint.lock
manifest at worker startup, set it as a contextvar before invoking model
downloads, and have `ModelRefDownloader.download` consult it to pass the
right `provider=` kwarg into `parse_model_ref`. Refs not in the index
default to `provider="tensorhub"` — same as the wire-format contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from gen_worker._worker_support import (
    _binding_canonical_ref,
    _collect_binding_entries,
    build_provider_index_from_manifest,
)
from gen_worker.models.hf_downloader import HuggingFaceDownloadResult, HuggingFaceRef
from gen_worker.models.ref_downloader import (
    ModelRefDownloader,
    lookup_provider_for_ref,
    reset_provider_by_ref,
    set_provider_by_ref,
)


# -----------------------------------------------------------------------------
# build_provider_index_from_manifest
# -----------------------------------------------------------------------------


def _make_manifest_with_bindings(bindings_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Helper: wrap a list of bindings dicts in the endpoint.lock shape."""
    return {
        "functions": [
            {
                "name": f"fn{i}",
                "bindings": b,
            }
            for i, b in enumerate(bindings_blocks)
        ]
    }


def test_provider_index_extracts_fixed_hf_binding() -> None:
    manifest = _make_manifest_with_bindings(
        [
            {
                "pipeline": {
                    "kind": "fixed",
                    "provider": "hf",
                    "ref": "black-forest-labs/FLUX.2-klein-base-4B",
                    "flavor": "bf16",
                    "tag": "prod",
                }
            }
        ]
    )
    index = build_provider_index_from_manifest(manifest)

    # The bare ref + flavor (the form the worker sees on the wire) must
    # be in the index pointing at hf.
    assert "black-forest-labs/FLUX.2-klein-base-4B#bf16" in index
    assert index["black-forest-labs/FLUX.2-klein-base-4B#bf16"] == "hf"
    # The bare ref without flavor must also be there for convenience.
    assert index["black-forest-labs/FLUX.2-klein-base-4B"] == "hf"


def test_provider_index_extracts_fixed_tensorhub_binding() -> None:
    manifest = _make_manifest_with_bindings(
        [
            {
                "pipeline": {
                    "kind": "fixed",
                    "provider": "tensorhub",
                    "ref": "acme/flux",
                    "flavor": "fp8",
                    "tag": "prod",
                }
            }
        ]
    )
    index = build_provider_index_from_manifest(manifest)

    # Tensorhub canonical form uses "owner/repo:tag" — match TensorhubRef.canonical().
    assert "acme/flux:prod#fp8" in index
    assert index["acme/flux:prod#fp8"] == "tensorhub"
    # Bare ref + flavor (without :tag) is also indexed for runtime payloads
    # that drop the tag.
    assert index["acme/flux#fp8"] == "tensorhub"


def test_provider_index_extracts_dispatch_table() -> None:
    """Dispatch tables can mix providers across variants — each entry's
    provider must be indexed independently.
    """
    manifest = _make_manifest_with_bindings(
        [
            {
                "pipeline": {
                    "kind": "dispatch",
                    "field": "variant",
                    "table": {
                        "bf16": {
                            "provider": "hf",
                            "ref": "owner/flux",
                            "flavor": "bf16",
                            "tag": "prod",
                        },
                        "fp8": {
                            "provider": "tensorhub",
                            "ref": "owner/flux",
                            "flavor": "fp8",
                            "tag": "prod",
                        },
                    },
                }
            }
        ]
    )
    index = build_provider_index_from_manifest(manifest)

    assert index.get("owner/flux#bf16") == "hf"
    # Tensorhub canonical wins for the tensorhub variant.
    assert index.get("owner/flux:prod#fp8") == "tensorhub"


def test_provider_index_empty_for_missing_manifest() -> None:
    assert build_provider_index_from_manifest(None) == {}
    assert build_provider_index_from_manifest({}) == {}
    assert build_provider_index_from_manifest({"functions": []}) == {}


def test_provider_index_skips_malformed_entries() -> None:
    """Manifest entries missing `ref` or with non-dict shape must not
    crash the index builder — defensive against unexpected schemas.
    """
    manifest = {
        "functions": [
            {"bindings": {"pipeline": {"kind": "fixed", "provider": "hf"}}},  # no ref
            {"bindings": "not-a-dict"},
            {"bindings": {"pipeline": "not-a-dict"}},
            {
                "bindings": {
                    "pipeline": {
                        "kind": "fixed",
                        "provider": "hf",
                        "ref": "ok/ref",
                    }
                }
            },
        ]
    }
    index = build_provider_index_from_manifest(manifest)
    # Only the well-formed entry should be indexed.
    assert index == {"ok/ref": "hf"}


# -----------------------------------------------------------------------------
# lookup_provider_for_ref + contextvar plumbing
# -----------------------------------------------------------------------------


def test_lookup_provider_default_when_no_context() -> None:
    # Outside any contextvar set, lookups must return the default.
    assert lookup_provider_for_ref("foo/bar") == "tensorhub"
    assert lookup_provider_for_ref("foo/bar", default="hf") == "hf"


def test_lookup_provider_reads_contextvar() -> None:
    index = {"acme/flux#bf16": "hf", "acme/tensorhub-ref": "tensorhub"}
    tok = set_provider_by_ref(index)
    try:
        assert lookup_provider_for_ref("acme/flux#bf16") == "hf"
        assert lookup_provider_for_ref("acme/tensorhub-ref") == "tensorhub"
        # Refs not in the index fall through to the default.
        assert lookup_provider_for_ref("not/in-index") == "tensorhub"
    finally:
        reset_provider_by_ref(tok)


def test_lookup_provider_after_reset_falls_back_to_default() -> None:
    """After reset_provider_by_ref the lookup must return the default
    again — no lingering state across requests.
    """
    tok = set_provider_by_ref({"foo/bar": "hf"})
    reset_provider_by_ref(tok)
    assert lookup_provider_for_ref("foo/bar") == "tensorhub"


# -----------------------------------------------------------------------------
# ModelRefDownloader.download routes via the provider index
# -----------------------------------------------------------------------------


class _FakeHFDownloader:
    """Records every HuggingFaceHubDownloader.download call so the test
    can assert routing without hitting the network.
    """

    def __init__(self) -> None:
        self.calls: List[HuggingFaceRef] = []

    def download(self, ref: HuggingFaceRef) -> HuggingFaceDownloadResult:
        self.calls.append(ref)
        return HuggingFaceDownloadResult(local_dir=Path("/tmp/fake-hf-dir"))


def test_hf_ref_with_provider_index_routes_to_hf_downloader(tmp_path: Path) -> None:
    """The end-to-end fix for issue #17: a worker that registered with
    `supported_repo_refs = ["black-forest-labs/FLUX.2-klein-base-4B#bf16"]`
    and whose endpoint.lock declared that binding as `provider: hf` must
    download via HuggingFaceHubDownloader — not raise the tensorhub
    "not in resolved_repos_by_id" error.
    """
    downloader = ModelRefDownloader()
    fake_hf = _FakeHFDownloader()
    downloader._hf = fake_hf  # type: ignore[assignment]

    ref = "black-forest-labs/FLUX.2-klein-base-4B#bf16"

    # Set the provider index the way the worker does at startup.
    tok = set_provider_by_ref({ref: "hf"})
    try:
        out = downloader.download(ref, str(tmp_path))
    finally:
        reset_provider_by_ref(tok)

    # HF downloader must have been invoked with the right repo_id.
    assert len(fake_hf.calls) == 1, f"expected 1 HF download, got {len(fake_hf.calls)}"
    assert fake_hf.calls[0].repo_id == "black-forest-labs/FLUX.2-klein-base-4B"
    # The downloader returns the local dir from the HF result.
    assert out == "/tmp/fake-hf-dir"


def test_hf_ref_without_provider_index_falls_back_to_tensorhub(tmp_path: Path) -> None:
    """Without the provider index set, a bare HF-shaped ref still defaults
    to tensorhub (matching the wire-format contract). This is the
    pre-fix behavior — kept as a contract guarantee so the default branch
    doesn't silently change.
    """
    downloader = ModelRefDownloader()
    fake_hf = _FakeHFDownloader()
    downloader._hf = fake_hf  # type: ignore[assignment]

    # No contextvar set — default provider applies.
    with pytest.raises(RuntimeError, match="not in resolved_repos_by_id"):
        downloader.download("acme/some-tensorhub-ref", str(tmp_path))

    # HF downloader must NOT have been invoked.
    assert fake_hf.calls == []


def test_tensorhub_ref_with_provider_index_still_works(tmp_path: Path) -> None:
    """The tensorhub path must keep working when the index marks a ref
    as tensorhub. Verifies the fix didn't accidentally break the
    tensorhub flow (regression guard).
    """
    downloader = ModelRefDownloader()
    fake_hf = _FakeHFDownloader()
    downloader._hf = fake_hf  # type: ignore[assignment]

    ref = "acme/tensorhub-only#fp8"

    tok = set_provider_by_ref({ref: "tensorhub"})
    try:
        # Tensorhub without orchestrator-resolved repos raises — this is the
        # contract: workers can't resolve tensorhub refs by themselves.
        with pytest.raises(RuntimeError, match="not in resolved_repos_by_id"):
            downloader.download(ref, str(tmp_path))
    finally:
        reset_provider_by_ref(tok)

    # And critically: HF downloader was NOT invoked for a tensorhub ref.
    assert fake_hf.calls == []


def test_invoker_override_not_in_index_defaults_to_tensorhub(tmp_path: Path) -> None:
    """Acceptance criterion from the issue: refs absent from the build-time
    index (e.g. invoker-supplied overrides) keep the tensorhub default —
    matches the wire-format contract.
    """
    downloader = ModelRefDownloader()
    fake_hf = _FakeHFDownloader()
    downloader._hf = fake_hf  # type: ignore[assignment]

    # Index has one entry; the ref under test isn't in it.
    tok = set_provider_by_ref({"other/ref": "hf"})
    try:
        with pytest.raises(RuntimeError, match="not in resolved_repos_by_id"):
            downloader.download("invoker/override-ref", str(tmp_path))
    finally:
        reset_provider_by_ref(tok)

    # HF downloader stays unused for tensorhub-defaulted refs.
    assert fake_hf.calls == []


# -----------------------------------------------------------------------------
# _binding_canonical_ref + _collect_binding_entries
# -----------------------------------------------------------------------------


def test_binding_canonical_ref_tensorhub_with_tag_and_flavor() -> None:
    entry = {
        "provider": "tensorhub",
        "ref": "acme/flux",
        "tag": "prod",
        "flavor": "fp8",
    }
    assert _binding_canonical_ref(entry) == "acme/flux:prod#fp8"


def test_binding_canonical_ref_hf_keeps_bare_ref() -> None:
    entry = {
        "provider": "hf",
        "ref": "owner/repo",
        "flavor": "bf16",
        "tag": "prod",
    }
    # HF refs ignore tag entirely; flavor appended via #.
    assert _binding_canonical_ref(entry) == "owner/repo#bf16"


def test_binding_canonical_ref_handles_missing_provider() -> None:
    """Defaults to tensorhub when provider is missing — matches the
    wire-format contract.
    """
    entry = {"ref": "acme/flux"}
    out = _binding_canonical_ref(entry)
    assert out == "acme/flux:latest"


def test_collect_binding_entries_handles_dispatch_table() -> None:
    bindings = {
        "pipeline": {
            "kind": "dispatch",
            "field": "variant",
            "table": {
                "v1": {"provider": "hf", "ref": "a/b"},
                "v2": {"provider": "tensorhub", "ref": "c/d"},
            },
        }
    }
    entries = _collect_binding_entries(bindings)
    refs = sorted(e["ref"] for e in entries)
    assert refs == ["a/b", "c/d"]


def test_collect_binding_entries_handles_fixed() -> None:
    bindings = {
        "pipeline": {
            "kind": "fixed",
            "provider": "hf",
            "ref": "a/b",
        }
    }
    entries = _collect_binding_entries(bindings)
    assert len(entries) == 1
    assert entries[0]["ref"] == "a/b"
    assert entries[0]["provider"] == "hf"


# -----------------------------------------------------------------------------
# Worker._provider_for_ref: instance-side lookup (no contextvar required)
# -----------------------------------------------------------------------------


def test_worker_provider_for_ref_reads_instance_index() -> None:
    """The worker exposes ``_provider_for_ref`` for callers that need to
    look up provider without setting/resetting the contextvar (e.g. the
    ``_try_find_existing_cozy_snapshot_dir`` cache pre-check that runs
    on a worker thread before the contextvar is established).
    """
    from gen_worker.worker import Worker

    w = Worker.__new__(Worker)
    w._provider_by_ref_index = {
        "owner/repo": "hf",
        "owner/repo#bf16": "hf",
        "acme/cozy": "tensorhub",
    }

    assert w._provider_for_ref("owner/repo") == "hf"
    assert w._provider_for_ref("owner/repo#bf16") == "hf"
    assert w._provider_for_ref("acme/cozy") == "tensorhub"
    # Missing refs fall back to the default.
    assert w._provider_for_ref("not/here") == "tensorhub"
    assert w._provider_for_ref("not/here", default="hf") == "hf"


def test_worker_provider_for_ref_empty_index() -> None:
    from gen_worker.worker import Worker

    w = Worker.__new__(Worker)
    w._provider_by_ref_index = {}
    assert w._provider_for_ref("any/ref") == "tensorhub"


def test_hf_parse_strips_flavor_suffix() -> None:
    """Issue #17: runtime payloads carry "owner/repo#flavor" for HF refs
    (flavor is binding metadata, not an HF concept). The HF parser must
    strip the flavor before producing the repo_id used by huggingface_hub.
    """
    from gen_worker.models.refs import parse_model_ref

    parsed = parse_model_ref("black-forest-labs/FLUX.2-klein-base-4B#bf16", provider="hf")
    assert parsed.provider == "hf"
    assert parsed.hf is not None
    assert parsed.hf.repo_id == "black-forest-labs/FLUX.2-klein-base-4B"
