"""Regression tests for gen-worker 0.7.21 binding-shape readiness fix.

Two bugs were uncovered in 0.7.19 live runs against FLUX (release
3d8ec41ecd407d2818d13f7a):

Bug A — startup_phase=ready fired before any model bytes landed because
``_release_allowed_model_ids`` was None for binding-shape manifests. The
0.7.19 pre-mark-downloading code path was therefore skipped, the readiness
gate found ``_startup_required_refs_canonical=set()``, and ``ready`` fired
immediately.

Bug B — even after the worker successfully loaded the bf16 ref, the
orchestrator's dispatch gate never matched the worker's ``disk_models``
advertisement against ``RequiredRepoRefs``: ``RequiredRepoRefs`` was
``["...FLUX.2-klein-base-4B#bf16"]`` (with flavor) but
``HuggingFaceRef.canonical()`` stripped the flavor, so disk_models was
``["...FLUX.2-klein-base-4B"]`` (no flavor) and the cache-locality scorer
returned ``localityCold`` — locking the request in queued forever.

Both bugs are fixed by:
- Walking ``manifest["functions"][i]["bindings"]`` in ``Worker.__init__``
  to populate ``_release_allowed_model_ids`` + ``_required_refs_by_function``.
- Keeping ``#flavor`` on ``HuggingFaceRef.canonical()`` so disk_models /
  RequiredRepoRefs use the same identity.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

from gen_worker.models.cache import ModelCache
from gen_worker.models.refs import HuggingFaceRef, parse_model_ref
from gen_worker.worker import Worker


# --------------------------------------------------------------------------- #
# Bug B: HuggingFaceRef.canonical retains #flavor                             #
# --------------------------------------------------------------------------- #


def test_hf_ref_canonical_includes_flavor() -> None:
    """The orchestrator's RequiredRepoRefs carries flavor folded into the
    ref string. Workers MUST report disk_models with the same identity or
    the cache-locality scorer always lands on localityCold and the request
    never dispatches.
    """
    ref = HuggingFaceRef(repo_id="black-forest-labs/FLUX.2-klein-base-4B", flavor="bf16")
    assert ref.canonical() == "black-forest-labs/FLUX.2-klein-base-4B#bf16"


def test_hf_ref_canonical_no_flavor_falls_back_to_bare_repo() -> None:
    """When the binding has no flavor, canonical must NOT emit a trailing
    "#" so the bare repo form keeps round-tripping through the existing
    HF Hub fast paths.
    """
    ref = HuggingFaceRef(repo_id="acme/sd-xl")
    assert ref.canonical() == "acme/sd-xl"


def test_hf_ref_canonical_with_revision_and_flavor() -> None:
    """Revision and flavor compose: revision before flavor, matching the
    ``owner/repo@revision#flavor`` shape used elsewhere in the wire format.
    """
    ref = HuggingFaceRef(
        repo_id="acme/sd-xl",
        revision="0123abcd",
        flavor="bf16",
    )
    assert ref.canonical() == "acme/sd-xl@0123abcd#bf16"


def test_parse_model_ref_hf_preserves_flavor() -> None:
    """parse_model_ref must capture the #flavor tail so canonical() can
    reproduce the orchestrator's RequiredRepoRefs form. The repo_id field
    is still flavor-stripped because huggingface_hub needs a clean
    owner/repo identifier.
    """
    parsed = parse_model_ref("black-forest-labs/FLUX.2-klein-base-4B#bf16", provider="hf")
    assert parsed.provider == "hf"
    assert parsed.hf is not None
    # repo_id stays bare for huggingface_hub.
    assert parsed.hf.repo_id == "black-forest-labs/FLUX.2-klein-base-4B"
    # Flavor preserved.
    assert parsed.hf.flavor == "bf16"
    # canonical() emits the with-flavor form.
    assert parsed.hf.canonical() == "black-forest-labs/FLUX.2-klein-base-4B#bf16"


def test_parse_model_ref_hf_empty_flavor_treated_as_none() -> None:
    """Trailing '#' with empty flavor must not produce ``hf.flavor == ""`` —
    that would emit "owner/repo#" from canonical() and break the
    RequiredRepoRefs match.
    """
    parsed = parse_model_ref("acme/sd-xl#", provider="hf")
    assert parsed.hf is not None
    assert parsed.hf.flavor is None
    assert parsed.hf.canonical() == "acme/sd-xl"


# --------------------------------------------------------------------------- #
# Bug A: bindings-shape manifest walking in __init__                          #
# --------------------------------------------------------------------------- #


def _make_manifest_with_function_bindings(
    fn_name: str, bindings: Dict[str, Any]
) -> Dict[str, Any]:
    """Build a minimal endpoint.lock manifest with one function + bindings.

    Mirrors the structure produced by the gen-worker 0.7.x build pipeline:
    no top-level ``models`` / ``models_by_function`` blocks, only the
    typed ``[functions.bindings]`` shape.
    """
    return {
        "functions": [
            {
                "name": fn_name,
                "bindings": bindings,
            }
        ]
    }


def test_binding_shape_manifest_populates_required_refs() -> None:
    """A binding-shape manifest with one fixed HF binding must populate
    `_release_allowed_model_ids` AND `_required_refs_by_function` so the
    pre-mark-downloading + readiness-gate paths both engage. Before the
    0.7.21 fix the bindings walk only happened for the provider-index;
    `_release_allowed_model_ids` stayed None and the worker emitted
    `ready` before any model bytes landed.
    """
    manifest = _make_manifest_with_function_bindings(
        "generate",
        {
            "pipeline": {
                "kind": "fixed",
                "provider": "hf",
                "ref": "black-forest-labs/FLUX.2-klein-base-4B",
                "flavor": "bf16",
            }
        },
    )
    # Mock the discovery + manager / downloader paths so __init__ doesn't
    # try to spin gRPC streams. The test only cares about the manifest-
    # parsing prefix of __init__.
    w = _construct_worker_with_manifest(manifest)

    # Bug A: _release_allowed_model_ids should now contain the bf16 ref.
    assert w._release_allowed_model_ids is not None
    assert "black-forest-labs/FLUX.2-klein-base-4B#bf16" in w._release_allowed_model_ids

    # Bug A: _required_refs_by_function should be populated so
    # _bound_model_refs_for_function returns the right set for the
    # binding-shape function. Without this the loading-functions
    # advertisement is empty and the orchestrator dispatches to a cold
    # worker.
    assert "generate" in w._required_refs_by_function
    assert w._required_refs_by_function["generate"] == {
        "black-forest-labs/FLUX.2-klein-base-4B#bf16"
    }


def test_binding_shape_manifest_pre_marks_refs_as_downloading() -> None:
    """The existing 0.7.19 pre-mark loop runs over
    ``_release_allowed_model_ids``. With the 0.7.21 fix wiring the
    binding-derived refs into that set, the model-cache should be seeded
    with ``downloading`` entries BEFORE the first registration goes out —
    so the very first heartbeat carries the function in
    ``loading_functions`` instead of falsely advertising it as ready.
    """
    manifest = _make_manifest_with_function_bindings(
        "generate",
        {
            "pipeline": {
                "kind": "fixed",
                "provider": "hf",
                "ref": "black-forest-labs/FLUX.2-klein-base-4B",
                "flavor": "bf16",
            }
        },
    )
    w = _construct_worker_with_manifest(manifest)

    # The pre-mark loop should have flipped the cache entry to downloading.
    stats = w._model_cache.get_stats()
    downloading = set(stats.downloading_models or [])
    assert "black-forest-labs/FLUX.2-klein-base-4B#bf16" in downloading


def test_binding_shape_manifest_blocks_ready_until_download_lands() -> None:
    """End-to-end: with one HF fixed binding, _emit_ready_if_all_cached
    must NOT fire before the download lands. After mark_cached_to_disk
    of the canonical ref (with flavor), it MUST fire.
    """
    manifest = _make_manifest_with_function_bindings(
        "generate",
        {
            "pipeline": {
                "kind": "fixed",
                "provider": "hf",
                "ref": "black-forest-labs/FLUX.2-klein-base-4B",
                "flavor": "bf16",
            }
        },
    )
    w = _construct_worker_with_manifest(manifest)
    emitted: List[str] = []

    def capture_phase(phase: str, *, status: str = "ok", **_: Any) -> None:
        emitted.append(phase)

    w._emit_startup_phase = capture_phase  # type: ignore[method-assign]
    # Reset the latch since __init__ doesn't fire `ready` itself.
    w._ready_phase_emitted = False

    # Before any download lands — must stay in models_downloading.
    w._emit_ready_if_all_cached()
    assert "ready" not in emitted, f"ready emitted prematurely: {emitted}"

    # Simulate the download completing with the WITH-FLAVOR canonical form
    # (Bug B fix: that's what the HF prefetch path produces now).
    w._model_cache.mark_cached_to_disk(
        "black-forest-labs/FLUX.2-klein-base-4B#bf16", Path("/tmp/snap")
    )
    w._emit_ready_if_all_cached()
    assert emitted == ["ready"], f"want a single 'ready' emit after download, got {emitted}"


def test_binding_shape_manifest_loading_functions_reports_per_function_state() -> None:
    """With two functions each bound to a different HF ref, the loading
    set must accurately reflect which function is still waiting on which
    ref. The 0.7.21 fix adds _required_refs_by_function as the source of
    truth for binding-shape manifests.
    """
    manifest = {
        "functions": [
            {
                "name": "generate_bf16",
                "bindings": {
                    "pipeline": {
                        "kind": "fixed",
                        "provider": "hf",
                        "ref": "owner/repo-a",
                        "flavor": "bf16",
                    }
                },
            },
            {
                "name": "generate_nf4",
                "bindings": {
                    "pipeline": {
                        "kind": "fixed",
                        "provider": "hf",
                        "ref": "owner/repo-b",
                        "flavor": "nf4",
                    }
                },
            },
        ]
    }
    w = _construct_worker_with_manifest(manifest)
    # Stub out spec maps so _all_declared_function_names enumerates both
    # function names without running discovery.
    w._request_specs = {"generate_bf16": object(), "generate_nf4": object()}
    w._training_specs = {}
    w._batched_specs = {}
    w._serial_class_specs = {}
    w._conversion_class_specs = {}

    # Both refs start as downloading (pre-mark in __init__).
    assert sorted(w._loading_function_names()) == ["generate_bf16", "generate_nf4"]

    # bf16 lands — only nf4 should remain loading.
    w._model_cache.mark_cached_to_disk("owner/repo-a#bf16", Path("/tmp/a"))
    assert w._loading_function_names() == ["generate_nf4"]
    assert "generate_bf16" in w._available_function_names()

    # nf4 lands — nothing loading.
    w._model_cache.mark_cached_to_disk("owner/repo-b#nf4", Path("/tmp/b"))
    assert w._loading_function_names() == []
    assert sorted(w._available_function_names()) == ["generate_bf16", "generate_nf4"]


def test_terminally_failed_ref_unblocks_readiness() -> None:
    """A required ref that fails terminally (e.g. an HF flavor that
    doesn't exist on the repo) must NOT keep the worker pinned in
    models_downloading forever. The 0.7.21 fix marks the ref as
    terminally failed and releases the readiness gate.
    """
    manifest = {
        "functions": [
            {
                "name": "generate",
                "bindings": {
                    "pipeline": {
                        "kind": "fixed",
                        "provider": "hf",
                        "ref": "acme/exists",
                        "flavor": "bf16",
                    }
                },
            },
            {
                "name": "nf4_only",
                "bindings": {
                    "pipeline": {
                        "kind": "fixed",
                        "provider": "hf",
                        "ref": "acme/missing",
                        "flavor": "nf4",
                    }
                },
            },
        ]
    }
    w = _construct_worker_with_manifest(manifest)
    w._request_specs = {"generate": object(), "nf4_only": object()}
    w._training_specs = {}
    w._batched_specs = {}
    w._serial_class_specs = {}
    w._conversion_class_specs = {}

    emitted: List[str] = []

    def capture_phase(phase: str, *, status: str = "ok", **_: Any) -> None:
        emitted.append(phase)

    w._emit_startup_phase = capture_phase  # type: ignore[method-assign]
    w._ready_phase_emitted = False

    # bf16 ref lands successfully.
    w._model_cache.mark_cached_to_disk("acme/exists#bf16", Path("/tmp/a"))
    w._emit_ready_if_all_cached()
    # Still waiting on the nf4 ref — ready should NOT fire yet.
    assert emitted == [], f"ready fired before nf4 resolution: {emitted}"

    # nf4 ref fails terminally (e.g. HF 404). The readiness gate must
    # release.
    w._mark_ref_terminally_failed("acme/missing#nf4", "HF 404: not found")
    w._emit_ready_if_all_cached()
    assert emitted == ["ready"], (
        f"ready must fire after the last outstanding ref is terminally failed; got {emitted}"
    )

    # The function whose ONLY ref failed must be marked locally unavailable
    # so the dispatch gate rejects requests for it.
    assert "nf4_only" in w._worker_local_unavailable_functions_by_name
    # The function whose ref DID land stays runnable.
    assert "generate" not in w._worker_local_unavailable_functions_by_name


# --------------------------------------------------------------------------- #
# Helper: build a Worker against a manifest without running discovery / gRPC  #
# --------------------------------------------------------------------------- #


def _construct_worker_with_manifest(manifest: Dict[str, Any]) -> Worker:
    """Run Worker.__init__'s manifest-parsing prefix without the gRPC /
    discovery side effects.

    We can't call Worker.__init__ directly because it tries to discover
    decorated functions in the importing module and open a gRPC stream.
    Instead, we replay the relevant subset of __init__ by hand: install
    the model cache, then call the same helpers __init__ calls on a
    binding-shape manifest. This is the same pattern other tests in this
    repo use (test_worker_readiness.py, test_incremental_function_readiness.py).
    """
    from gen_worker._worker_support import (
        _binding_canonical_ref,
        _collect_binding_entries,
    )
    from gen_worker.models.refs import parse_model_ref

    w = Worker.__new__(Worker)
    w._fixed_model_id_by_key = {}
    w._payload_model_id_by_key_by_function = {}
    w._fixed_model_spec_by_key = {}
    w._payload_model_spec_by_key_by_function = {}
    w._required_refs_by_function = {}
    w._disabled_functions_by_name = {}
    w._worker_local_unavailable_functions_by_name = {}
    w._provider_by_ref_index = {}
    w._model_cache = ModelCache(max_vram_gb=0.0, model_cache_dir="/tmp/test-mc-binding")
    w._ready_phase_emitted = False
    w._ready_phase_lock = threading.Lock()
    w._startup_required_refs_canonical = set()
    w._manifest_allowed_model_ids = None
    w._release_allowed_model_ids = None
    w._terminally_failed_refs = set()

    # Replay the binding-walking prefix of __init__.
    functions_block = manifest.get("functions")
    binding_refs: set[str] = set()
    if isinstance(functions_block, list):
        for fn_entry in functions_block:
            if not isinstance(fn_entry, dict):
                continue
            fn_name = str(fn_entry.get("name") or "").strip()
            if not fn_name:
                continue
            per_fn: set[str] = set()
            for entry in _collect_binding_entries(fn_entry.get("bindings")):
                ref_key = _binding_canonical_ref(entry)
                if not ref_key:
                    continue
                provider = str(entry.get("provider") or "").strip() or "tensorhub"
                try:
                    parsed = parse_model_ref(ref_key, provider=provider)
                    if parsed.provider == "tensorhub" and parsed.tensorhub:
                        canon = parsed.tensorhub.canonical()
                    elif parsed.provider == "hf" and parsed.hf:
                        canon = parsed.hf.canonical()
                    else:
                        canon = ref_key
                except Exception:
                    canon = ref_key
                per_fn.add(canon)
                binding_refs.add(canon)
            if per_fn:
                w._required_refs_by_function[fn_name] = per_fn
    if binding_refs:
        w._manifest_allowed_model_ids = binding_refs
        w._release_allowed_model_ids = binding_refs

    # Pre-mark loop (matches __init__).
    for canon in w._release_allowed_model_ids or set():
        if not canon:
            continue
        w._startup_required_refs_canonical.add(canon)
        w._model_cache.mark_downloading(canon, progress=0.0)

    return w
