"""pgw#1512: the lane's dtype governs the lane's component, and nothing else.

Paul's ruling, 2026-08-19. The trace used to read one dtype off the lane and
hand it to `from_pretrained`, casting EVERY component of the tree to it. The
serving loader does the opposite and says so:

    "No `torch_dtype=` (the lane contract IS the dtype)"   serving/context.py
    "bytes land verbatim in the container's own dtype ... any conversion is
     the STORE's contract-negotiation job, NEVER load time. A container that
     disagrees with the active lane is reported, not silently repaired."
                                          serving/streaming/engine.py

So the trace performed exactly the conversion serve refuses, and a DiT lane's
bf16 landed on the VAE beside it — a bf16 bias meeting an fp32 activation in a
decode block that is fine on a pod (h3's `MiniMaxH3VideoDecodeStep`).

Precision is now per component: the lane governs the components its CONTRACT
DESCRIBES, everything else takes its own declared dtype, and a component the
lane covers with nothing anywhere to state its precision REFUSES rather than
defaulting (pgw#1448 — and at trace a default silently re-keys a graph).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
import gen_worker._vendor.torchcg  # noqa: E402,F401
import torch  # noqa: E402

from gen_worker.release.trace_context import (  # noqa: E402
    TraceLoadContext,
    TraceSurfaceUnavailable,
)

FIXTURES = Path(__file__).resolve().parent / "release_fixtures"


def _contract(*patterns: str, dtype: str = "bfloat16") -> Any:
    """A real tensorfs contract covering exactly `patterns`."""

    from gen_worker._vendor.tensorfs.contract import Contract

    return Contract.from_document(
        json.dumps(
            {
                "format": "tensorfs-contract-v1",
                "name": "probe.lane",
                "version": 1,
                "description": "pgw#1512 probe",
                "dtype": dtype,
                "tensors": [
                    {"role": p, "pattern": f"{p}.weight", "rank": 4}
                    for p in patterns
                ],
            }
        )
    )


@pytest.fixture(scope="module")
def tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import sys

    sys.path.insert(0, str(FIXTURES))
    try:
        import tiny_tree
    finally:
        sys.path.remove(str(FIXTURES))
    return tiny_tree.save_config_only(tmp_path_factory.mktemp("dtype-config-only"))


def _ctx(tree: Path, lane: Any) -> TraceLoadContext:
    return TraceLoadContext(lane=lane, checkpoint_dir=tree)


def test_the_lane_governs_the_component_its_CONTRACT_NAMES_and_no_other(
    tree: Path,
) -> None:
    """The ruling in one assertion.

    A contract covering `unet` says bf16 about the UNET and nothing else.
    The three components beside it show all three outcomes at once:

      * `vae` — uncovered, declares nothing: NOT cast, which is what a pod
        does with bytes it was given no contract for;
      * `text_encoder` — uncovered, but its own transformers config declares
        `dtype: float32`, so passthrough honours the packager. Under the old
        global cast this component was forced to the DENOISER's bf16 despite
        saying float32 about itself — the fabrication in one line.
    """

    ctx = _ctx(tree, _contract("unet.conv_out"))

    assert ctx.component_dtype(tree, "unet") is torch.bfloat16
    assert ctx.component_dtype(tree, "vae") is None
    assert ctx.component_dtype(tree, "text_encoder") is torch.float32


def test_a_component_directory_passed_WITHOUT_a_subfolder_is_still_that_component(
    tree: Path,
) -> None:
    """diffusers hands the directory over, not tree+subfolder.

    `AutoencoderKL.from_pretrained(<tree>/vae)` carries no subfolder, so an
    absent subfolder must not read as "the root" — that mistake made every
    component look lane-governed and refused a contractless endpoint's derive.
    """

    ctx = _ctx(tree, _contract("unet.conv_out"))

    assert ctx.component_dtype(tree / "unet", None) is torch.bfloat16
    assert ctx.component_dtype(tree / "vae", None) is None


def test_a_components_OWN_declared_dtype_wins_over_no_coverage(
    tree: Path, tmp_path: Path
) -> None:
    """Passthrough: what the packager said about THIS component."""

    declared = tmp_path / "declared"
    (declared / "vae").mkdir(parents=True)
    config = json.loads((tree / "vae" / "config.json").read_text())
    config["torch_dtype"] = "float16"
    (declared / "vae" / "config.json").write_text(json.dumps(config))

    ctx = _ctx(declared, _contract("unet.conv_out"))
    assert ctx.component_dtype(declared, "vae") is torch.float16


def test_the_lane_covering_a_component_nothing_can_speak_for_REFUSES(
    tree: Path,
) -> None:
    """No silent fp32 where precision IS identity (pgw#1448 / pgw#1458).

    The contract covers `unet` but declares no dtype, the tree declares none,
    and a config-only tree has no headers — so nothing can answer for the one
    component whose precision becomes the graph key.
    """

    from gen_worker._vendor.tensorfs.contract import Contract

    dtypeless = Contract.from_document(
        json.dumps(
            {
                "format": "tensorfs-contract-v1",
                "name": "probe.dtypeless",
                "version": 1,
                "description": "covers unet, states no dtype",
                "tensors": [
                    {"role": "unet.conv_out", "pattern": "unet.conv_out.weight",
                     "rank": 4}
                ],
            }
        )
    )
    ctx = _ctx(tree, dtypeless)

    with pytest.raises(TraceSurfaceUnavailable) as refusal:
        ctx.component_dtype(tree, "unet")
    message = str(refusal.value)
    assert "graph identity" in message
    assert "'unet'" in message


def test_an_UNCOVERED_component_that_nothing_declares_is_NOT_a_refusal(
    tree: Path,
) -> None:
    """The absence of a conversion is not a default, and this is the line.

    Measured on the real trees: sd15's and sdxl's `[derive]
    checkpoint_configs` declare no dtype in any component config, none in
    `model_index.json`, and ship no safetensors. Refusing here would make
    every endpoint in the fleet underivable, so an uncovered component simply
    is not cast — exactly what the streaming loader does with bytes it was
    given no contract for.
    """

    ctx = _ctx(tree, _contract("unet.conv_out"))
    assert ctx.component_dtype(tree, "scheduler") is None


def test_a_lane_with_no_readable_patterns_governs_NOTHING(tree: Path) -> None:
    """A derived lane has no contract, so it speaks for no component."""

    ctx = _ctx(tree, None)
    assert ctx._lane_components() == frozenset()


def test_the_governed_set_comes_from_the_patterns_not_a_second_list(
    tree: Path,
) -> None:
    """Derived, so it cannot go stale against the contract it describes."""

    ctx = _ctx(tree, _contract("unet.conv_out", "vae.decoder.conv_out"))
    assert ctx._lane_components() == frozenset({"unet", "vae"})
