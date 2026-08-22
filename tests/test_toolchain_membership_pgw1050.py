"""The ``toolchain`` axis's MEMBERSHIP: the compiler, not the model libraries."""

from __future__ import annotations

from typing import Dict

import pytest

from gen_worker._vendor.torchcg import identity as tcg_identity

from gen_worker import dist_records
from gen_worker import toolchain as cc

from harness.compiled_graph_meta import exported_compiled_graph_meta

TOOLCHAIN: Dict[str, str] = {
    "settings_declaration": "5" * 16,
    "loaded_libs": "6" * 16,
    "torch": "7" * 16,
    "triton": "8" * 16,
    "nvidia-cuda-nvrtc-cu13": "9" * 16,
    "bin:ptxas": "a" * 16,
    "diffusers": "b" * 16,
    "transformers": "c" * 16,
    "peft": "d" * 16,
}

COMPILER_COMPONENTS = (
    "settings_declaration", "loaded_libs", "torch", "triton",
    "nvidia-cuda-nvrtc-cu13", "bin:ptxas",
)

MODEL_LIBRARIES = ("diffusers", "transformers", "peft")


def _bumped(component: str) -> Dict[str, str]:
    block = dict(TOOLCHAIN)
    block[component] = "f" * 16
    return block


def _key(block: Dict[str, str]) -> str:
    return str(exported_compiled_graph_meta(toolchain=block)["key"])


@pytest.mark.parametrize("library", MODEL_LIBRARIES)
def test_the_env_block_keys_WHATEVER_IT_IS_GIVEN(library: str) -> None:
    """tcg#90 moved WHERE membership is decided, and this test says where.

    torchcg used to carry a `_NOT_TOOLCHAIN` deny-list and evict diffusers /
    transformers / peft from the axis itself. That was a SECOND producer of the
    membership rule, disagreeing with `require_stack`'s allow-list one level
    away, and it is gone: the env block's members are the caller's to choose and
    torchcg keys exactly what it is handed.

    So a model library reaching the block DOES split the key. That is not a
    regression — it is the responsibility being unambiguous. The rule now lives
    in exactly one place, `gen_worker.toolchain.toolchain_digest`, and the test
    below proves it never emits one.
    """

    assert _key(_bumped(library)) != _key(dict(TOOLCHAIN))


@pytest.mark.parametrize("library", MODEL_LIBRARIES)
def test_the_compile_stack_SELECTOR_drops_the_library(library: str) -> None:
    """The surviving torchcg-side statement of membership, and it is an
    ALLOW-list: `compile_stack` selects torch/triton/nvidia-* out of a lockfile
    and a model library is simply not selected."""

    from gen_worker._vendor.torchcg.identity import compile_stack

    selected = compile_stack({"torch": "2.13.0", library: "9.9.9"})
    assert library not in selected
    assert selected == {"torch": "2.13.0"}


@pytest.mark.parametrize("component", COMPILER_COMPONENTS)
def test_a_genuine_toolchain_bump_still_splits(component: str) -> None:
    """torch, triton, the CUDA runtime wheels, the bundled ptxas, the settings declaration and the boot-frozen native manifest are the compiler."""
    assert _key(_bumped(component)) != _key(dict(TOOLCHAIN))


def test_an_unclassified_component_stays_in_the_key() -> None:
    """The eviction is a DENY-list, not an allow-list."""
    block = dict(TOOLCHAIN)
    block["nvidia-cublas-cu14"] = "e" * 16
    assert _key(block) != _key(dict(TOOLCHAIN))


def test_the_producer_does_not_collect_the_model_libraries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED on `origin/master`: ``toolchain_digest`` hashed the three RECORDs alongside torch/triton/nvidia."""
    env = {
        "torch": "torch RECORD", "triton": "triton RECORD",
        "nvidia-cuda-nvrtc-cu13": "nvrtc RECORD",
        "diffusers": "diffusers RECORD", "transformers": "transformers RECORD",
        "peft": "peft RECORD", "requests": "requests RECORD",
    }
    monkeypatch.setattr(dist_records, "record_texts", lambda: env)
    cc.toolchain_digest.cache_clear()
    try:
        collected = dict(cc.toolchain_digest())
    finally:
        cc.toolchain_digest.cache_clear()
    for library in MODEL_LIBRARIES:
        assert library not in collected, library
    for component in ("torch", "triton", "nvidia-cuda-nvrtc-cu13"):
        assert component in collected, component
    assert "requests" not in collected


def test_producer_and_reader_agree_on_membership() -> None:
    """One axis, one membership: everything the producer collects REACHES the key.

    Under tcg#90 the reader drops NOTHING — the env block is digested whole —
    so this is now the strong direction of the same property, and the eviction
    it used to skip past (`MODEL_LIBRARIES`) is asserted at the producer in
    `test_the_producer_does_not_collect_the_model_libraries`.
    """
    collected = dict(cc.toolchain_digest())
    assert collected, "the producer must collect something to prove anything"
    assert "torch" in collected, "the producer must name the compiler"
    full = _key(collected)
    for component in collected:
        without = {k: v for k, v in collected.items() if k != component}
        if "torch" not in without:
            continue  # torch's absence is a refusal, tested elsewhere
        assert _key(without) != full, component

