"""Inference-side dispatch on tensorhub-recorded `runtime_library`.

Issue #67 task 13: replace file-sniffing in inference workers with a stamp
read from the resolved checkpoint attributes. Today the only runtime we
support is `diffusers` (loaded by `PipelineLoader`); the other runtime
families have no inference-side loader yet (transformers / peft /
sentence-transformers / llama.cpp). This module is the dispatch point: as
new loaders land, they hook in here and the worker picks the right one
without re-introducing file sniffing.

Usage:

    from gen_worker.pipeline.runtime_dispatch import pick_loader_for_runtime_library

    loader_kind = pick_loader_for_runtime_library(
        runtime_library=resolved_checkpoint_attrs.get("runtime_library", ""),
    )
    if loader_kind == "diffusers":
        return PipelineLoader().load(...)
    elif loader_kind == "transformers":
        raise NotImplementedError("transformers loader not implemented")
    ...

The raised exceptions name the missing loader and the recorded runtime,
so the caller (the worker request handler) can surface a 424 with a
specific reason instead of a generic "load failed".
"""
from __future__ import annotations

from typing import Optional


# Mirror of the gen-worker → tensorhub repo_kind → library_name map so
# inference-side dispatch doesn't have to re-derive it from `repo_kind`.
# Keep the values in sync with `tensorhub.repos.library_name` enum.
_LIBRARY_NAME_TO_LOADER_KIND: dict[str, str] = {
    "diffusers": "diffusers",
    "transformers": "transformers",
    "peft": "peft",
    "sentence-transformers": "sentence-transformers",
    "llama.cpp": "llama-cpp",
}

# Same mapping but keyed on the gen-worker classifier's `runtime_library`
# value, since the classifier emits a slightly broader set than tensorhub
# stores (it distinguishes `diffusers-single-file` and `diffusers-lora`).
_RUNTIME_LIBRARY_TO_LOADER_KIND: dict[str, str] = {
    "diffusers": "diffusers",
    "diffusers-single-file": "diffusers",
    "diffusers-lora": "diffusers",  # load_lora_weights() on a diffusion pipeline
    "transformers": "transformers",
    "peft": "peft",
    "sentence-transformers": "sentence-transformers",
    "llama-cpp": "llama-cpp",
}


# Loaders that ship today. Add a key here when a new loader lands so
# `pick_loader_for_runtime_library` stops raising NotImplementedError
# for that runtime.
SUPPORTED_LOADERS: frozenset[str] = frozenset({"diffusers"})


class UnsupportedRuntimeLibrary(Exception):
    """Raised when the resolved checkpoint's runtime_library has no
    inference-side loader registered. Carries the recorded value plus
    the loader-kind that *would* handle it, so callers can surface a
    specific 424 to the user."""

    def __init__(self, runtime_library: str, loader_kind: Optional[str]) -> None:
        self.runtime_library = runtime_library
        self.loader_kind = loader_kind
        if loader_kind is None:
            msg = (
                f"runtime_library={runtime_library!r} is not in the recognized set. "
                f"Recognized: {sorted(_RUNTIME_LIBRARY_TO_LOADER_KIND.keys())}."
            )
        else:
            msg = (
                f"runtime_library={runtime_library!r} maps to loader_kind={loader_kind!r}, "
                f"but no inference-side loader is registered for that kind. "
                f"Supported: {sorted(SUPPORTED_LOADERS)}."
            )
        super().__init__(msg)


def pick_loader_for_runtime_library(*, runtime_library: str = "", library_name: str = "") -> str:
    """Return the loader_kind to use for a checkpoint.

    Pass either `runtime_library` (the gen-worker classifier value, set on
    the per-checkpoint `attributes` row) or `library_name` (tensorhub's
    server-derived value on the `repos` row). At least one must be
    non-empty; `runtime_library` wins when both are present because it
    carries finer-grained shape distinctions (`diffusers-lora` vs
    `diffusers`).

    Returns one of: `diffusers`, `transformers`, `peft`,
    `sentence-transformers`, `llama-cpp`. Raises
    `UnsupportedRuntimeLibrary` when no loader is registered for the
    resolved kind.
    """
    rl = (runtime_library or "").strip().lower()
    ln = (library_name or "").strip().lower()
    if rl:
        loader_kind = _RUNTIME_LIBRARY_TO_LOADER_KIND.get(rl)
    elif ln:
        loader_kind = _LIBRARY_NAME_TO_LOADER_KIND.get(ln)
    else:
        raise UnsupportedRuntimeLibrary("", None)
    if loader_kind is None:
        raise UnsupportedRuntimeLibrary(rl or ln, None)
    if loader_kind not in SUPPORTED_LOADERS:
        raise UnsupportedRuntimeLibrary(rl or ln, loader_kind)
    return loader_kind
