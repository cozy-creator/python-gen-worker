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

**pgw#1567 (Paul, 2026-08-20) then settled the ORDER, which is what was still
wrong.** The ladder led with the component's own BYTES, so a derive against a
tree the store had not converted — a stock fp16 dreamshaper on a dev box —
traced fp16 graphs under a bf16 lane and armed a fleet no runtime could enter.
The lane declaration now answers FIRST for every component of the tree it
governs; the checkpoint speaks only for a DERIVED lane, which has no contract
to read a dtype from. The trace dtype is a property of the contract-template,
never of the bytes that happen to be mounted.

These tests use REAL RATIFIED lanes on purpose. pgw#1512's fixture invented
`unet.conv_out.weight`, and that invention is what let every check pass while
the fleet's actual spelling did the opposite.

**pgw#1621 made the mistake pgw#1512 made INEXPRESSIBLE, which is why two of
these tests are gone rather than re-keyed.** They enumerated every v1 contract
document's `tensors[].pattern` list and refused a pattern whose first dotted
segment was a COMPONENT WORD (`unet.`, `vae.`), because a v1 document had one
flat pattern list and nothing else to say which component a pattern belonged
to — so "is this a directory name or a parameter name?" had to be INFERRED from
the spelling, and pgw#1512 inferred it wrong for the entire fleet. A v2
TOPOLOGY carries `components[]` as a first-class field, each with its own
`{key -> shape}` map, so the component is DECLARED and there is no inference to
get wrong. `test_a_v2_topology_declares_its_components_so_the_pgw1512_
inference_cannot_recur` is what replaces both, and it states the same two facts
positively: components are named, and one layout really does cover several.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")
import torch  # noqa: E402

from gen_worker.release.trace_context import TraceLoadContext  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "release_fixtures"

#: sd15's REAL lane, from the VENDORED v2 corpus that ships in this wheel —
#: not a path outside the repo. An earlier draft read `~/cozy/tensorfs/spec/...`
#: and `~/cozy/serverless-endpoints/sd15/...`, which made every assertion here
#: SKIP in CI: the tests that prove this fix would not have run in the gate
#: that is supposed to catch its regression.
SD15_BF16 = ("sd15.diffusers@1", "plain.bf16@1")


def _contract(pair: Any = SD15_BF16) -> Any:
    """The lane as the trace sees it: a READ `DeclaredLane`.

    pgw#1621: the trace's dtype source is `DeclaredLane.dtype`, which is the
    ratified QUANT RULE's `declared_dtype` — so this helper builds the lane the
    way `Model.__init_subclass__` does rather than parsing a document, and a
    test can never disagree with the platform about a lane's precision.
    """
    from gen_worker.demand import GiB, const
    from gen_worker.models.tensor_layout_contract import (
        LayoutId,
        capability_floor_for_rule,
        rule_dtype,
    )
    from gen_worker.serving.lane_spec import DeclaredLane, lane

    topology, quant = pair
    return DeclaredLane(
        layout=LayoutId(topology=topology, quant=quant),
        topology=topology, quant=quant,
        dtype=rule_dtype(quant), min_sm=capability_floor_for_rule(quant),
        spec=lane(request=const(GiB(1))),
    )


def _ctx(tree: Path, lane: Any) -> TraceLoadContext:
    return TraceLoadContext(lane=lane, checkpoint_dir=tree)


@pytest.fixture(scope="module")
def sd15_shaped_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """sd15's shape, built in-repo: a config-only pipeline tree whose TEXT ENCODER config declares float32 while nothing else declares anything."""

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


def test_a_v2_topology_declares_its_components_so_pgw1512_cannot_recur() -> None:
    """The two fences this file used to carry, made one POSITIVE statement.

    **What they were.** `test_EVERY_shipped_contract_names_the_DENOISERS_OWN_
    parameters` enumerated every v1 document's flat `tensors[].pattern` list
    and refused a pattern whose first dotted segment was a component word
    (`unet.`, `vae.`) — because a v1 document had ONE flat list and no way to
    say which component a pattern belonged to, so "directory name or parameter
    name?" had to be INFERRED from the spelling. pgw#1512 inferred it wrong for
    the whole fleet: it read the first segment as a component roster, matched
    nothing on any endpoint, never applied the lane's dtype anywhere, and
    traced sd15's UNet at float32 under a bfloat16 lane. The fence even needed
    an EXEMPTION — `musicgen.transformers-fp16`, a single-file checkpoint whose
    tensors genuinely ARE named `text_encoder.*` / `audio_encoder.*` — which is
    the tell that the inference was never sound, only usually right.
    `test_a_contract_can_cover_SEVERAL_components` was the companion: sdxl's
    one document covered UNet, VAE and text encoder, which falsified pgw#1512's
    premise that a denoiser contract says nothing about the VAE beside it.

    **Why neither can be re-keyed.** A v2 TOPOLOGY carries `components[]` as a
    first-class field, each with its own `{key -> shape}` map. The component is
    DECLARED, not spelled into a prefix, so there is no inference to get wrong
    and no exemption to grant — a single-file checkpoint is simply a topology
    with ONE unnamed component, which is what `musicgen.transformers@1` is.
    Refusing a pattern by its first segment is not a check that got weaker; it
    is a check with nothing left to check.

    **What survives is both facts, stated positively**, because they are still
    what makes the dtype policy below correct: the components are named, and
    one layout really does govern several of them.
    """
    from gen_worker.models.tensor_layout_contract import topologies

    corpus = topologies()

    # sd15: ONE component, the denoiser, addressed by its own parameter names.
    # No `unet.` prefix anywhere in the keys — the component is the KEY of the
    # map, which is exactly the distinction pgw#1512 could not draw.
    (sd15_component,) = corpus["sd15.diffusers@1"]
    assert sd15_component == "unet"
    sd15_keys = corpus["sd15.diffusers@1"]["unet"]
    assert "conv_in.weight" in sd15_keys
    assert not [k for k in sd15_keys if k.startswith("unet.")]

    # sdxl: FOUR components under ONE layout — the fact that falsified
    # pgw#1512's premise, now readable instead of inferred from a pattern list.
    sdxl = corpus["sdxl.diffusers@1"]
    assert set(sdxl) == {"unet", "vae", "text_encoder", "text_encoder_2"}
    assert "conv_in.weight" in sdxl["unet"], "the denoiser's own parameters"
    assert "decoder.conv_in.weight" in sdxl["vae"], "the VAE's"
    # ...and the VAE's `decoder.*` keys live under the VAE component, so the
    # v1 collision between "a submodule named decoder" and "a component named
    # decoder" cannot be constructed at all.
    assert not [k for k in sdxl["unet"] if k.startswith("decoder.")]

    # musicgen: the SINGLE-FILE shape that needed a v1 exemption. It is one
    # component with no name, and its keys carry the submodule prefixes that
    # used to look like component words. No exemption; nothing to exempt.
    (musicgen_component,) = corpus["musicgen.transformers@1"]
    assert musicgen_component == ""
    assert [k for k in corpus["musicgen.transformers@1"][""]
            if k.startswith("audio_encoder.")]

    # A topology carries NO dtype — the precision half is the quant rule's,
    # which is why the lane and not the topology answers the dtype question
    # below.
    for handle, components in corpus.items():
        for component, keys in components.items():
            for shape in list(keys.values())[:1]:
                assert all(isinstance(d, int) for d in shape), (handle, component)


def test_sd15s_UNET_traces_at_the_LANES_dtype_not_at_float32(sd15_shaped_tree: Path) -> None:
    """The regression, at the seam."""

    ctx = _ctx(sd15_shaped_tree, _contract())
    assert ctx.component_dtype(sd15_shaped_tree, "unet") is torch.bfloat16


def test_the_WHOLE_sd15_pipeline_agrees_so_no_activation_crosses_a_boundary(
    sd15_shaped_tree: Path,
) -> None:
    """Uniformity is the point: a mixed pipeline is what raised."""

    ctx = _ctx(sd15_shaped_tree, _contract())
    seen = {
        name: ctx.component_dtype(sd15_shaped_tree, name)
        for name in ("unet", "vae", "text_encoder")
    }
    assert set(seen.values()) == {torch.bfloat16}, seen


def _tree_at(root: Path, dtype: Any) -> Path:

    safetensors = pytest.importorskip("safetensors.torch")

    root.mkdir(parents=True, exist_ok=True)
    for component, weight in (
        ("unet", "conv_in.weight"),
        ("vae", "decoder.conv_out.weight"),
    ):
        (root / component).mkdir(parents=True, exist_ok=True)
        (root / component / "config.json").write_text(json.dumps({"_class_name": "X"}))
        safetensors.save_file(
            {weight: torch.zeros(2, 2, dtype=dtype)},
            str(root / component / "diffusion_pytorch_model.safetensors"),
        )
    return root


def test_the_LANE_outranks_the_mounted_checkpoints_own_bytes(tmp_path: Path) -> None:

    tree = _tree_at(tmp_path / "tree", torch.float16)
    ctx = _ctx(tree, _contract())
    assert ctx.component_dtype(tree, "vae") is torch.bfloat16
    assert ctx.component_dtype(tree, "unet") is torch.bfloat16


def test_FLIPPING_the_checkpoints_dtype_does_not_move_the_traced_precision(
    tmp_path: Path,
) -> None:
    """The deliverable, stated as a property: same lane, two checkpoints."""

    lane = _contract()
    answers = []
    for name, dtype in (("fp16", torch.float16), ("fp32", torch.float32)):
        tree = _tree_at(tmp_path / name, dtype)
        ctx = _ctx(tree, lane)
        answers.append(
            tuple(ctx.component_dtype(tree, part) for part in ("unet", "vae"))
        )
    assert answers[0] == answers[1] == (torch.bfloat16, torch.bfloat16)


def test_a_checkpoint_that_disagrees_with_the_lane_is_NAMED(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Following the lane silently would hide an unconverted tree."""

    import logging

    tree = _tree_at(tmp_path / "tree", torch.float16)
    ctx = _ctx(tree, _contract())
    with caplog.at_level(logging.WARNING, logger="gen_worker.release.trace"):
        assert ctx.component_dtype(tree, "unet") is torch.bfloat16
        assert ctx.component_dtype(tree, "unet") is torch.bfloat16
    said = [r for r in caplog.records if "TRACE FOLLOWS THE LANE" in r.getMessage()]
    assert len(said) == 1, "named once per component, not once per ask"
    assert "float16" in said[0].getMessage()


def test_a_checkpoint_that_AGREES_with_the_lane_says_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The red arm's other half: no warning when there is nothing to warn about."""

    import logging

    tree = _tree_at(tmp_path / "tree", torch.bfloat16)
    ctx = _ctx(tree, _contract())
    with caplog.at_level(logging.WARNING, logger="gen_worker.release.trace"):
        assert ctx.component_dtype(tree, "unet") is torch.bfloat16
    assert not [r for r in caplog.records if "TRACE FOLLOWS THE LANE" in r.getMessage()]


def test_a_stale_component_CONFIG_does_not_outrank_the_lane(
    sd15_shaped_tree: Path,
) -> None:
    """The precedence inversion that produced the crash."""

    declared = json.loads(
        (sd15_shaped_tree / "text_encoder" / "config.json").read_text()
    )
    assert declared.get("torch_dtype") == "float32", "premise: the config says fp32"

    ctx = _ctx(sd15_shaped_tree, _contract())
    assert ctx.component_dtype(sd15_shaped_tree, "text_encoder") is torch.bfloat16


def test_the_config_is_still_read_when_NOTHING_else_can_speak(tmp_path: Path) -> None:
    """Last, but not discarded — a derived lane has no dtype to offer."""

    tree = tmp_path / "tree"
    (tree / "thing").mkdir(parents=True)
    (tree / "thing" / "config.json").write_text(json.dumps({"torch_dtype": "float16"}))

    ctx = _ctx(tree, None)
    assert ctx.component_dtype(tree, "thing") is torch.float16


def test_nothing_anywhere_means_NO_CAST_rather_than_a_default(tmp_path: Path) -> None:

    tree = tmp_path / "tree"
    (tree / "thing").mkdir(parents=True)
    (tree / "thing" / "config.json").write_text(json.dumps({"_class_name": "X"}))

    ctx = _ctx(tree, None)
    assert ctx.component_dtype(tree, "thing") is None


def test_a_component_directory_passed_WITHOUT_a_subfolder_still_resolves(
    sd15_shaped_tree: Path,
) -> None:
    """diffusers hands the directory over: `from_pretrained(<tree>/vae)`."""

    ctx = _ctx(sd15_shaped_tree, _contract())
    assert ctx.component_dtype(sd15_shaped_tree / "unet", None) is torch.bfloat16
    assert (
        ctx.component_dtype(sd15_shaped_tree / "text_encoder", None) is torch.bfloat16
    )
