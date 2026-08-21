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
    return str(exported_compiled_graph_meta(toolchain=block)["compiled_graph_key"])


@pytest.mark.parametrize("library", MODEL_LIBRARIES)
def test_a_model_library_bump_does_not_rekey(library: str) -> None:
    """Two compiled graphs identical in graph x envelope x sm, differing ONLY in a model library's content, are ONE compiled graph and must carry ONE key."""
    assert _key(_bumped(library)) == _key(dict(TOOLCHAIN))


@pytest.mark.parametrize("library", MODEL_LIBRARIES)
def test_the_axis_ignores_the_library_even_when_a_graph_records_it(
    library: str,
) -> None:
    """Membership is a property of the AXIS, not of whichever producer wrote the block: a recorded block that carries the component digests the same as one that does not."""
    without = {k: v for k, v in TOOLCHAIN.items() if k not in MODEL_LIBRARIES}
    with_one = dict(without)
    with_one[library] = TOOLCHAIN[library]
    assert (tcg_identity.toolchain_axis_digest(with_one)
            == tcg_identity.toolchain_axis_digest(without))
    assert (tcg_identity.toolchain_axis_digest({library: TOOLCHAIN[library]})
            == tcg_identity.toolchain_axis_digest({}))


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
    """One axis, one membership: everything the producer collects REACHES the axis, and nothing it collects is dropped on the way in."""
    collected = dict(cc.toolchain_digest())
    assert collected, "the producer must collect something to prove anything"
    full = tcg_identity.toolchain_axis_digest(collected)
    for component in collected:
        if component in MODEL_LIBRARIES:
            continue
        without = {k: v for k, v in collected.items() if k != component}
        assert tcg_identity.toolchain_axis_digest(without) != full, component

