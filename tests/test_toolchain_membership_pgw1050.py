"""The ``toolchain`` axis's MEMBERSHIP: the compiler, not the model libraries.

Two failure modes; this file pins the expensive one. The key must not
UNDER-split — that is the arm token's and the local-store verdict's problem,
and both stay pessimistic. It must also not OVER-split:
``diffusers``/``transformers``/``peft`` do not belong on the ``toolchain`` axis
because everything they can do to a cell already arrives through the traced
``graph`` axis (the axis is the COMPUTATION, not the ingress). Otherwise every
model-library patch release re-keys every cell in the fleet for a graph that
has not moved.

The two tests that matter are a matched pair, and neither is meaningful
without the other:

* an evicted component moves NO key (the over-split);
* a genuine compiler component still moves the key (the under-split, which
  must never happen) — GREEN on both sides, and it is what makes the first
  test a narrowing rather than a deletion.
"""

from __future__ import annotations

from typing import Dict

import pytest

from torch_compiled_graphs import identity as tcg_identity

from gen_worker import compile_cache as cc, dist_records

from harness.cell_meta import exported_cell_meta

#: A recorded ``toolchain`` block of the shape a real mint records: the
#: compiler proper (content digests of the wheels' RECORDs + the bundled CUDA
#: tool binaries), the settings declaration, the boot-frozen native manifest —
#: and, on a pre-pgw#1050 cell, the three model libraries.
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

#: What the axis IS. Every other component of a recorded block is not.
COMPILER_COMPONENTS = (
    "settings_declaration", "loaded_libs", "torch", "triton",
    "nvidia-cuda-nvrtc-cu13", "bin:ptxas",
)

MODEL_LIBRARIES = ("diffusers", "transformers", "peft")


def _bumped(component: str) -> Dict[str, str]:
    """The same toolchain block with exactly one component's CONTENT moved —
    a wheel bump, identified the way the amendment requires (the RECORD's
    content digest), never a version string."""
    block = dict(TOOLCHAIN)
    block[component] = "f" * 16
    return block


def _key(block: Dict[str, str]) -> str:
    """The ck1 key of a cell whose graph, envelope and sm are held fixed and
    whose toolchain block is ``block``."""
    return str(exported_cell_meta(toolchain=block)["compiled_graph_key"])


# ---------------------------------------------------------------------------
# The over-split — RED on origin/master, for each of the three
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("library", MODEL_LIBRARIES)
def test_a_model_library_bump_does_not_rekey(library: str) -> None:
    """Two cells identical in graph x envelope x sm, differing ONLY in a model
    library's content, are ONE cell and must carry ONE key.

    RED on `origin/master`: the axis folded the library's RECORD digest, so
    the bump moved the key and the whole fleet re-minted for a computation
    that had not changed.
    """
    assert _key(_bumped(library)) == _key(dict(TOOLCHAIN))


@pytest.mark.parametrize("library", MODEL_LIBRARIES)
def test_the_axis_ignores_the_library_even_when_a_cell_records_it(
    library: str,
) -> None:
    """Membership is a property of the AXIS, not of whichever producer wrote
    the block: a recorded block that carries the component digests the same as
    one that does not."""
    without = {k: v for k, v in TOOLCHAIN.items() if k not in MODEL_LIBRARIES}
    with_one = dict(without)
    with_one[library] = TOOLCHAIN[library]
    assert (tcg_identity.toolchain_axis_digest(with_one)
            == tcg_identity.toolchain_axis_digest(without))
    # ...and the component carries NO bits of its own: an axis over the model
    # library alone is the axis over an empty block. Stated through the
    # authority rather than by reading its canonical form, which pgw#1277
    # deleted from this repo — a worker-side copy of TCG's membership would be
    # the second authority this unit exists to remove.
    assert (tcg_identity.toolchain_axis_digest({library: TOOLCHAIN[library]})
            == tcg_identity.toolchain_axis_digest({}))


# ---------------------------------------------------------------------------
# The under-split — must stay GREEN on both sides of the change
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("component", COMPILER_COMPONENTS)
def test_a_genuine_toolchain_bump_still_splits(component: str) -> None:
    """torch, triton, the CUDA runtime wheels, the bundled ptxas, the settings
    declaration and the boot-frozen native manifest are the compiler. A cell is
    a ``dlopen``-ed ELF linking torch's AOTI runtime: an ABI mismatch is a
    segfault or silent numerics, never slowness. Each of them moves the key."""
    assert _key(_bumped(component)) != _key(dict(TOOLCHAIN))


def test_an_unclassified_component_stays_in_the_key() -> None:
    """The eviction is a DENY-list, not an allow-list. A component nobody has
    classified — a new ``nvidia-*`` runtime wheel, a new bundled tool — keys
    by default, because the axiom's expensive failure is the over-split and
    its UNSAFE failure is the under-split."""
    block = dict(TOOLCHAIN)
    block["nvidia-cublas-cu14"] = "e" * 16
    assert _key(block) != _key(dict(TOOLCHAIN))


# ---------------------------------------------------------------------------
# The producer collects the same membership the reader states
# ---------------------------------------------------------------------------


def test_the_producer_does_not_collect_the_model_libraries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED on `origin/master`: ``toolchain_digest`` hashed the three RECORDs
    alongside torch/triton/nvidia.

    Driven off a synthetic env rather than this box's site-packages, so the
    test cannot go falsely green on a machine that simply has no diffusers
    installed.
    """
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
    # ...and it still collects the compiler, so the test is not passing by
    # collecting nothing.
    for component in ("torch", "triton", "nvidia-cuda-nvrtc-cu13"):
        assert component in collected, component
    # A distribution that is neither is not swept in by the narrowing.
    assert "requests" not in collected


def test_producer_and_reader_agree_on_membership() -> None:
    """One axis, one membership: everything the producer collects REACHES the
    axis, and nothing it collects is dropped on the way in.

    Proved by mutation rather than by comparing against a second canonical
    form: drop any one collected component and the axis must move. That is the
    same claim the old ``toolchain_facts`` equality made, stated against the
    authority that now owns membership."""
    collected = dict(cc.toolchain_digest())
    assert collected, "the producer must collect something to prove anything"
    full = tcg_identity.toolchain_axis_digest(collected)
    for component in collected:
        if component in MODEL_LIBRARIES:
            continue  # evicted by design — the tests above own that claim
        without = {k: v for k, v in collected.items() if k != component}
        assert tcg_identity.toolchain_axis_digest(without) != full, component


# ---------------------------------------------------------------------------
# The wire fact follows the key
# ---------------------------------------------------------------------------
