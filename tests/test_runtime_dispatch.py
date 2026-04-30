"""Issue #67 task 13: dispatch on `runtime_library` not file sniffing."""
from __future__ import annotations

import pytest

from gen_worker.pipeline.runtime_dispatch import (
    UnsupportedRuntimeLibrary,
    pick_loader_for_runtime_library,
)


def test_diffusers_runtime_library_picks_diffusers_loader() -> None:
    assert pick_loader_for_runtime_library(runtime_library="diffusers") == "diffusers"


def test_diffusers_single_file_picks_diffusers_loader() -> None:
    """diffusers-single-file is the AIO-style layout — same diffusers
    loader (load_pipeline.from_single_file)."""
    assert pick_loader_for_runtime_library(runtime_library="diffusers-single-file") == "diffusers"


def test_diffusers_lora_picks_diffusers_loader() -> None:
    """A native LoRA loads via `pipeline.load_lora_weights()` on a
    diffusion pipeline — same `diffusers` loader."""
    assert pick_loader_for_runtime_library(runtime_library="diffusers-lora") == "diffusers"


def test_library_name_fallback_when_runtime_library_empty() -> None:
    """tensorhub's server-derived `library_name` can fill in when a
    checkpoint's per-row `runtime_library` attribute is missing."""
    assert pick_loader_for_runtime_library(library_name="diffusers") == "diffusers"


def test_runtime_library_wins_over_library_name() -> None:
    """When both are present, `runtime_library` wins (finer grain)."""
    # `diffusers-lora` maps to loader_kind=diffusers (still in SUPPORTED)
    assert (
        pick_loader_for_runtime_library(
            runtime_library="diffusers-lora",
            library_name="diffusers",
        )
        == "diffusers"
    )


def test_transformers_raises_unsupported_with_clear_message() -> None:
    """No transformers inference loader is registered yet — the
    exception must name the loader_kind so the caller can surface a
    specific 424 (`no_loader_for_kind:transformers`)."""
    with pytest.raises(UnsupportedRuntimeLibrary) as ei:
        pick_loader_for_runtime_library(runtime_library="transformers")
    assert ei.value.runtime_library == "transformers"
    assert ei.value.loader_kind == "transformers"


def test_unknown_runtime_library_raises_with_recognized_set() -> None:
    """Garbage values surface the recognized set in the error so
    operators can debug malformed metadata writes."""
    with pytest.raises(UnsupportedRuntimeLibrary) as ei:
        pick_loader_for_runtime_library(runtime_library="some-future-runtime")
    assert ei.value.loader_kind is None


def test_empty_inputs_raise() -> None:
    with pytest.raises(UnsupportedRuntimeLibrary):
        pick_loader_for_runtime_library()


def test_case_insensitive_normalization() -> None:
    assert pick_loader_for_runtime_library(runtime_library="DIFFUSERS") == "diffusers"
    assert pick_loader_for_runtime_library(library_name="  Diffusers  ") == "diffusers"
