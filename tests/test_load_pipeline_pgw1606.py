from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from gen_worker.serving import lane_ladder as L  # noqa: E402
from gen_worker.serving import lane_materialize as M  # noqa: E402


BF16_BODY = "bf16-w16a16"
FP8_BODY = "fp8-w8a8-dynamic"


#: The lanes these tests resolve, as pgw#1621 stamp pairs. Both halves are
#: ratified documents in the vendored `spec/v2` corpus, so the dtype and the
#: floor below are READ rather than chosen — a lane whose dtype a test picked
#: could not prove anything about the loader that must honour it.
BF16_LANE = ("sdxl.diffusers@1", "plain.bf16@1")
FP8_LANE = ("sdxl.diffusers@1", "cozy.fp8-rowwise@1")


def _lane(pair: Any = BF16_LANE) -> L.DeclaredLane:
    """A REAL pgw#1599/pgw#1621 `DeclaredLane` — the ladder's own value object.

    `dtype` and `min_sm` come off the ratified QUANT RULE through the same two
    producers the declaration surface uses (`rule_dtype`,
    `capability_floor_for_rule`), so this fixture cannot disagree with the
    platform about either.
    """
    from gen_worker.demand import GiB, const
    from gen_worker.models.tensor_layout_contract import (
        LayoutId,
        capability_floor_for_rule,
        rule_dtype,
    )
    from gen_worker.serving.lane_spec import lane

    topology, quant = pair
    return L.DeclaredLane(
        layout=LayoutId(topology=topology, quant=quant),
        topology=topology, quant=quant,
        dtype=rule_dtype(quant), min_sm=capability_floor_for_rule(quant),
        spec=lane(request=const(GiB(1.0))),
    )


def _resolved(body: str, pair: Any = BF16_LANE) -> L.ResolvedLane:
    return L.ResolvedLane(
        declared=_lane(pair), body=body,
        reason=L.CHOSE_BASELINE, card=L.CardFacts(sm=89, name="fixture"),
    )


def _tiny_unet() -> Any:
    from diffusers import UNet2DModel

    unet: Any = UNet2DModel(
        sample_size=8, in_channels=3, out_channels=3,
        block_out_channels=(32, 32), layers_per_block=1,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"), norm_num_groups=8,
    )
    return unet.to(torch.bfloat16).eval()


class _Pipe:

    def __init__(self, unet: Any) -> None:
        self.unet = unet


def test_census_counts_plain_linears_on_an_unquantized_denoiser():
    counts = M.lane_census(_Pipe(_tiny_unet()))
    assert counts["w8a8"] == 0 and counts["w4a4"] == 0
    assert counts["plain"] > 0, "the fixture must actually contain Linears"


def test_census_sees_a_real_fp8_module_by_its_marker_not_its_dtype():
    """A quantized leaf reports the LOGICAL dtype it emulates, so a dtype census reads per-row fp8 as bf16 and answers 'no fp8 here' about a fully quantized model."""
    from gen_worker.models.w8a8 import fp8_scaled_linear_class

    cls = fp8_scaled_linear_class()
    leaf = cls(16, 16, bias=False, compute_dtype=torch.bfloat16,
               static_input_scale=False)
    unet = _tiny_unet()
    name = next(n for n, m in unet.named_modules()
                if type(m).__name__ == "Linear")
    parent = unet.get_submodule(name.rsplit(".", 1)[0]) if "." in name else unet
    setattr(parent, name.rsplit(".", 1)[-1], leaf)

    counts = M.lane_census(_Pipe(unet))
    assert counts["w8a8"] == 1
    assert M.lane_of(_Pipe(unet)) == FP8_BODY


def test_a_quantized_lane_over_an_unquantized_pipeline_REFUSES():
    """The matched-nothing refusal."""
    pipe = _Pipe(_tiny_unet())
    with pytest.raises(M.LaneMaterializationError, match="ZERO quantized leaves"):
        M._assert_lane(pipe, _resolved(FP8_BODY), swapped=17)


def test_a_baseline_lane_carrying_quantized_leaves_REFUSES():
    """The other direction, which is just as silent: a baseline lane's pricing, compiled-graph key and executed-lane claim all say bf16."""
    from gen_worker.models.w8a8 import fp8_scaled_linear_class

    unet = _tiny_unet()
    name = next(n for n, m in unet.named_modules()
                if type(m).__name__ == "Linear")
    parent = unet.get_submodule(name.rsplit(".", 1)[0]) if "." in name else unet
    setattr(parent, name.rsplit(".", 1)[-1],
            fp8_scaled_linear_class()(16, 16, bias=False,
                                      compute_dtype=torch.bfloat16,
                                      static_input_scale=False))
    with pytest.raises(M.LaneMaterializationError, match="quantized leaf"):
        M._assert_lane(_Pipe(unet), _resolved(BF16_BODY), swapped=0)


def test_a_baseline_lane_over_a_clean_pipeline_passes():
    M._assert_lane(_Pipe(_tiny_unet()), _resolved(BF16_BODY), swapped=0)


def test_the_fp8_arm_refuses_a_tree_that_carries_no_fp8_artifact(
    tmp_path: Path,
) -> None:
    """The ladder resolved fp8 because the deploy said the bytes were staged."""
    (tmp_path / "unet").mkdir(parents=True)
    with pytest.raises(M.LaneMaterializationError, match="no fp8 artifact"):
        M.materialize(_Pipe(_tiny_unet()), _resolved(FP8_BODY), tree=tmp_path)


def test_the_nvfp4_arm_refuses_the_same_way_the_fp8_arm_does(
    tmp_path: Path,
) -> None:
    """Symmetry is the point."""
    (tmp_path / "transformer").mkdir(parents=True)
    with pytest.raises(M.LaneMaterializationError, match="no fp4 artifact"):
        M.materialize(_Pipe(_tiny_unet()), _resolved("nvfp4-w4a4-static"),
                      tree=tmp_path)


def test_a_body_with_no_materializer_is_refused_rather_than_ignored():
    """If the lane table can name a body nothing can build, the table and the materializer disagree and one of them is wrong."""
    with pytest.raises(M.LaneMaterializationError, match="no materializer"):
        M.materialize(_Pipe(_tiny_unet()), _resolved("svdq-fp4-w4a4"),
                      tree=Path("."))


@pytest.fixture(scope="module")
def tiny_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from diffusers import DDPMPipeline, DDPMScheduler

    root = tmp_path_factory.mktemp("lp") / "src"
    pipe: Any = DDPMPipeline(unet=_tiny_unet().to(torch.float32),
                             scheduler=DDPMScheduler(num_train_timesteps=10))
    pipe.save_pretrained(root)
    return root


def test_load_pipeline_materializes_the_baseline_lane_and_confesses(
    tiny_tree: Path, caplog: Any
) -> None:
    """End to end on the eager bridge: a resolved baseline lane, a real tree, a real pipeline — and the confession on the log naming the lane, the reason and the rejected rungs."""
    import logging

    from diffusers import DDPMPipeline

    from gen_worker.serving.context import DeployBinding, LoadContext

    resolved = L.ResolvedLane(
        declared=_lane(BF16_LANE), body=BF16_BODY,
        reason=L.CHOSE_BASELINE, card=L.CardFacts(sm=89, name="fixture"),
        rejected=(L.RejectedRung(body=FP8_BODY, contract_id="sdxl.diffusers@1+cozy.fp8-rowwise@1",
                                 reason=L.REJECT_SM_FLOOR,
                                 detail="needs sm89, card is sm86"),),
    )
    ctx: LoadContext = LoadContext(
        binding=DeployBinding(checkpoint_ref="t@1", checkpoint_dir=tiny_tree),
        resolved=resolved,
    )
    with caplog.at_level(logging.INFO, logger="gen_worker.serving.context"):
        pipe: Any = ctx.load_pipeline(DDPMPipeline)
    assert pipe is not None
    assert M.lane_of(_Pipe(pipe.unet)) == BF16_BODY
    line = next(r.message for r in caplog.records if "LANE=" in r.message)
    assert "LANE=bf16-w16a16" in line
    assert "rejected=fp8-w8a8-dynamic(sm_floor: needs sm89, card is sm86)" in line


def test_load_pipeline_without_a_resolved_lane_degrades_to_load(
    tiny_tree: Path,
) -> None:
    """A fixture, a derive and the local CLI all build a context with no ladder behind it."""
    from diffusers import DDPMPipeline

    from gen_worker.serving.context import DeployBinding, LoadContext

    ctx: LoadContext = LoadContext(
        binding=DeployBinding(checkpoint_ref="t@1", checkpoint_dir=tiny_tree))
    assert ctx.resolved_lane is None
    assert ctx.load_pipeline(DDPMPipeline) is not None


def test_the_resolved_lane_selects_its_own_tree_out_of_a_multi_lane_binding(
    tiny_tree: Path,
) -> None:
    """The binding half of the multi-lane fix: `checkpoint_dir` answers the tree for the lane the ladder picked, and for the UPCAST rung it answers the tree whose BYTES are fetched, not the lane's own."""
    from gen_worker.serving.context import DeployBinding, LoadContext

    binding = DeployBinding(
        checkpoint_ref="t@1", checkpoint_dir=Path("/nonexistent/default"),
        lane_trees={"sdxl.diffusers@1+plain.bf16@1": Path("/trees/bf16"),
                    "sdxl.diffusers@1+cozy.fp8-rowwise@1": tiny_tree},
    )
    plain: LoadContext = LoadContext(
        binding=binding,
        resolved=L.ResolvedLane(declared=_lane(BF16_LANE),
                                body=BF16_BODY, reason=L.CHOSE_BASELINE),
    )
    assert plain.checkpoint_dir == Path("/trees/bf16")

    upcast: LoadContext = LoadContext(
        binding=binding,
        resolved=L.ResolvedLane(declared=_lane(BF16_LANE),
                                body=BF16_BODY, reason=L.CHOSE_UPCAST,
                                fetch_contract="sdxl.diffusers@1+cozy.fp8-rowwise@1",
                                transfer_saved_bytes=3_400_000_000),
    )
    assert upcast.checkpoint_dir == tiny_tree, (
        "the upcast rung fetches the QUANTIZED tree and serves baseline "
        "modules out of it")

    single: LoadContext = LoadContext(
        binding=DeployBinding(checkpoint_ref="t@1", checkpoint_dir=tiny_tree),
        resolved=plain.resolved_lane,
    )
    assert single.checkpoint_dir == tiny_tree
