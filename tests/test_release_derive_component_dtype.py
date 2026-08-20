"""What precision each component loads at, and WHERE that fact comes from.

pgw#1530 corrects pgw#1512. That issue read a lane contract's `tensors` list
as a roster of the COMPONENTS the contract describes — first path segment of
each pattern — and applied the lane's dtype only to those.

**No shipped contract is spelled that way.** Every one states the DENOISER's
own parameter names: sd15 and sd2 say `conv_in.weight` / `down_blocks.…`,
minimax-h3 says `transformer_blocks.…`, flux says `context_embedder.…` — never
a `unet.` or `transformer.` prefix. So the derived roster matched no component
on any endpoint in the fleet, the lane's dtype was never applied anywhere, and
two things followed:

  * a config-only derive traced sd15's UNet at **float32 under a bfloat16
    lane** — silently, and precision IS graph identity (pgw#1458), so a bf16
    mint refuses those keys by name;
  * a tree whose UNet declares bf16 while its text encoder's config still says
    float32 (the publisher's config in a store-converted tree) put an fp32
    activation into a bf16 denoiser: `Input type (float) and bias type
    (c10::BFloat16)` on payload 0.

The rule is now about WHERE the answer comes from, in order of how directly
the source knows: the component's own BYTES, then the LANE, then the
component's config — with the config last precisely because it is the one
source a conversion does not rewrite.

These tests use the REAL library contracts on purpose. pgw#1512's fixture
invented `unet.conv_out.weight`, and that invention is what let every check
pass while the fleet's actual spelling did the opposite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")
import torch  # noqa: E402

from gen_worker._vendor import tensorfs as _vendored_tensorfs  # noqa: E402
from gen_worker.release.trace_context import TraceLoadContext  # noqa: E402

#: The shipped contracts, from the VENDORED tensorfs that ships in this wheel
#: — not a path outside the repo. An earlier draft read
#: `~/cozy/tensorfs/spec/...` and `~/cozy/serverless-endpoints/sd15/...`, which
#: made every assertion here SKIP in CI: the tests that prove this fix would
#: not have run in the gate that is supposed to catch its regression.
CONTRACTS = Path(_vendored_tensorfs.__file__).resolve().parent / "_contracts"

FIXTURES = Path(__file__).resolve().parent / "release_fixtures"


def _contract(name: str) -> Any:
    from gen_worker._vendor.tensorfs.contract import Contract

    return Contract.from_document((CONTRACTS / name).read_text())


def _ctx(tree: Path, lane: Any) -> TraceLoadContext:
    return TraceLoadContext(lane=lane, checkpoint_dir=tree)


@pytest.fixture(scope="module")
def sd15_shaped_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """sd15's shape, built in-repo: a config-only pipeline tree whose TEXT
    ENCODER config declares float32 while nothing else declares anything.

    That is the real sd15 fixture's shape — `text_encoder/config.json` says
    `float32` because it is the PUBLISHER's config, and a store conversion
    rewrites bytes, not configs.
    """

    import sys

    sys.path.insert(0, str(FIXTURES))
    try:
        import tiny_tree
    finally:
        sys.path.remove(str(FIXTURES))

    tree = tiny_tree.save_config_only(tmp_path_factory.mktemp("sd15-shaped"))
    config = json.loads((tree / "text_encoder" / "config.json").read_text())
    config["torch_dtype"] = "float32"
    (tree / "text_encoder" / "config.json").write_text(json.dumps(config))
    return tree


def test_EVERY_shipped_contract_names_the_DENOISERS_OWN_parameters() -> None:
    """The fact that falsified pgw#1512, pinned so it cannot be re-assumed.

    If a contract ever DID lead with a component name this would go red, and
    whoever wrote it would find this test explaining why that matters.
    """

    # True COMPONENT names as `model_index.json` spells them. `encoder` and
    # `decoder` are deliberately absent: they are the VAE's own submodules,
    # and sdxl's contract leads with them — which is itself the point of the
    # next assertion.
    component_words = {"unet", "vae", "transformer", "text_encoder", "tokenizer"}
    offenders = []
    for path in sorted(CONTRACTS.glob("*.json")):
        heads = {
            t["pattern"].split(".", 1)[0]
            for t in json.loads(path.read_text()).get("tensors", [])
        }
        if heads & component_words:
            offenders.append(path.name)
    assert offenders == []


def test_a_contract_can_cover_SEVERAL_components_which_settles_the_question() -> None:
    """sdxl's contract enumerates the UNet, the VAE **and** the text encoder.

    `down_blocks`/`conv_in` (denoiser), `encoder`/`decoder`/`quant_conv`
    (VAE), `text_model`/`text_projection` (text encoder) — all at one
    `dtype: bfloat16`, none of them component-prefixed.

    So "a denoiser contract says nothing about the VAE beside it" — pgw#1512's
    premise, and mine — is not merely unprovable, it is contradicted by a
    shipped contract. A lane's dtype is the precision of the TREE the store
    converts, which is why it is the right answer for every component that has
    no bytes of its own to speak with.
    """

    heads = {
        t["pattern"].split(".", 1)[0]
        for t in json.loads(
            (CONTRACTS / "sdxl.diffusers-bf16.v1.json").read_text()
        )["tensors"]
    }
    assert {"down_blocks", "conv_in"} & heads, "the denoiser's own parameters"
    assert {"encoder", "decoder", "quant_conv"} & heads, "the VAE's"
    assert {"text_model", "text_projection"} & heads, "the text encoder's"


def test_sd15s_UNET_traces_at_the_LANES_dtype_not_at_float32(sd15_shaped_tree: Path) -> None:
    """The regression, at the seam. Was float32 under a bfloat16 lane."""

    ctx = _ctx(sd15_shaped_tree, _contract("sd15.diffusers-bf16.v1.json"))
    assert ctx.component_dtype(sd15_shaped_tree, "unet") is torch.bfloat16


def test_the_WHOLE_sd15_pipeline_agrees_so_no_activation_crosses_a_boundary(
    sd15_shaped_tree: Path,
) -> None:
    """Uniformity is the point: a mixed pipeline is what raised.

    `text_encoder/config.json` in this tree says float32. Under the corrected
    order the lane answers first, so the encoder and the denoiser agree and
    `prompt_embeds` can reach the UNet.
    """

    ctx = _ctx(sd15_shaped_tree, _contract("sd15.diffusers-bf16.v1.json"))
    seen = {
        name: ctx.component_dtype(sd15_shaped_tree, name)
        for name in ("unet", "vae", "text_encoder")
    }
    assert set(seen.values()) == {torch.bfloat16}, seen


def test_a_components_own_BYTES_outrank_the_lane(tmp_path: Path) -> None:
    """A converted tree's headers are what the pod actually runs.

    Built with real safetensors so the stub-aware reader is the thing under
    test, not a stand-in.
    """

    safetensors = pytest.importorskip("safetensors.torch")

    tree = tmp_path / "tree"
    (tree / "vae").mkdir(parents=True)
    (tree / "vae" / "config.json").write_text(json.dumps({"_class_name": "AutoencoderKL"}))
    safetensors.save_file(
        {"decoder.conv_out.weight": torch.zeros(2, 2, dtype=torch.float16)},
        str(tree / "vae" / "diffusion_pytorch_model.safetensors"),
    )

    ctx = _ctx(tree, _contract("sd15.diffusers-bf16.v1.json"))
    # The lane says bfloat16; these bytes say float16, and the bytes win.
    assert ctx.component_dtype(tree, "vae") is torch.float16


def test_a_stale_component_CONFIG_does_not_outrank_the_lane(
    sd15_shaped_tree: Path,
) -> None:
    """The precedence inversion that produced the crash.

    sd15's `text_encoder/config.json` says float32 in a tree the store
    converts to bf16 — a conversion rewrites bytes, not configs.
    """

    declared = json.loads(
        (sd15_shaped_tree / "text_encoder" / "config.json").read_text()
    )
    assert declared.get("torch_dtype") == "float32", "premise: the config says fp32"

    ctx = _ctx(sd15_shaped_tree, _contract("sd15.diffusers-bf16.v1.json"))
    assert ctx.component_dtype(sd15_shaped_tree, "text_encoder") is torch.bfloat16


def test_the_config_is_still_read_when_NOTHING_else_can_speak(tmp_path: Path) -> None:
    """Last, but not discarded — a derived lane has no dtype to offer."""

    tree = tmp_path / "tree"
    (tree / "thing").mkdir(parents=True)
    (tree / "thing" / "config.json").write_text(json.dumps({"torch_dtype": "float16"}))

    ctx = _ctx(tree, None)  # no lane, no bytes
    assert ctx.component_dtype(tree, "thing") is torch.float16


def test_nothing_anywhere_means_NO_CAST_rather_than_a_default(tmp_path: Path) -> None:
    """The absence of a conversion is not a default precision (pgw#1448)."""

    tree = tmp_path / "tree"
    (tree / "thing").mkdir(parents=True)
    (tree / "thing" / "config.json").write_text(json.dumps({"_class_name": "X"}))

    ctx = _ctx(tree, None)
    assert ctx.component_dtype(tree, "thing") is None


def test_a_component_directory_passed_WITHOUT_a_subfolder_still_resolves(
    sd15_shaped_tree: Path,
) -> None:
    """diffusers hands the directory over: `from_pretrained(<tree>/vae)`."""

    ctx = _ctx(sd15_shaped_tree, _contract("sd15.diffusers-bf16.v1.json"))
    assert ctx.component_dtype(sd15_shaped_tree / "unet", None) is torch.bfloat16
    assert (
        ctx.component_dtype(sd15_shaped_tree / "text_encoder", None) is torch.bfloat16
    )
