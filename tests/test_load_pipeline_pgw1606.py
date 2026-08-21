"""pgw#1606 — `ctx.load_pipeline` materializes the lane, and PROVES it did.

CPU-only, on a real tiny diffusers pipeline and the real executor module
classes. Nothing here is mocked: `fp8_scaled_linear_class()` builds the actual
`_Fp8ScaledLinear` the fp8 lane serves on, so the census is reading the same
marker production reads.

The point of the file is the pair of refusals. A loader that only reports its
successes cannot be audited, and the audit behind this issue found four places
where a lane's numerics could silently be something other than the lane's name.
Both refusals are exercised, in both directions.
"""

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


def _lane(contract_id: str, dtype: str = "bfloat16") -> L.DeclaredLane:
    """A REAL pgw#1599 `DeclaredLane` — the ladder's own value object."""
    from gen_worker.demand import GiB, const
    from gen_worker.models.tensor_layout_contract import capability_floor_for_dtype
    from gen_worker.serving.lane_spec import lane

    return L.DeclaredLane(
        contract=contract_id, contract_id=contract_id, dtype=dtype,
        min_sm=int(capability_floor_for_dtype(dtype) or 0),
        spec=lane(request=const(GiB(1.0))),
    )


def _resolved(body: str, contract_id: str = "t.lane@1") -> L.ResolvedLane:
    return L.ResolvedLane(
        declared=_lane(contract_id), body=body,
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
    """A pipeline shaped the way `_denoisers` looks for one."""

    def __init__(self, unet: Any) -> None:
        self.unet = unet


# --------------------------------------------------------------------------
# The census reads MARKERS, not dtypes
# --------------------------------------------------------------------------


def test_census_counts_plain_linears_on_an_unquantized_denoiser():
    counts = M.lane_census(_Pipe(_tiny_unet()))
    assert counts["w8a8"] == 0 and counts["w4a4"] == 0
    assert counts["plain"] > 0, "the fixture must actually contain Linears"


def test_census_sees_a_real_fp8_module_by_its_marker_not_its_dtype():
    """A quantized leaf reports the LOGICAL dtype it emulates, so a dtype
    census reads per-row fp8 as bf16 and answers 'no fp8 here' about a fully
    quantized model. The census must key on the structural marker instead —
    this is `_cozy_w8a8_linear`, set by the executor itself."""
    from gen_worker.models.w8a8 import fp8_scaled_linear_class

    cls = fp8_scaled_linear_class()
    leaf = cls(16, 16, bias=False, compute_dtype=torch.bfloat16,
               static_input_scale=False)
    unet = _tiny_unet()
    # Replace one real Linear in the tree with a real fp8 module.
    name = next(n for n, m in unet.named_modules()
                if type(m).__name__ == "Linear")
    parent = unet.get_submodule(name.rsplit(".", 1)[0]) if "." in name else unet
    setattr(parent, name.rsplit(".", 1)[-1], leaf)

    counts = M.lane_census(_Pipe(unet))
    assert counts["w8a8"] == 1
    assert M.lane_of(_Pipe(unet)) == FP8_BODY


# --------------------------------------------------------------------------
# THE TWO REFUSALS — and both must be able to fire
# --------------------------------------------------------------------------


def test_a_quantized_lane_over_an_unquantized_pipeline_REFUSES():
    """The matched-nothing refusal. This is the one that stops a bf16 model
    from being served, priced and compile-keyed as fp8."""
    pipe = _Pipe(_tiny_unet())
    with pytest.raises(M.LaneMaterializationError, match="ZERO quantized leaves"):
        M._assert_lane(pipe, _resolved(FP8_BODY), swapped=17)


def test_a_baseline_lane_carrying_quantized_leaves_REFUSES():
    """The other direction, which is just as silent: a baseline lane's
    pricing, compiled-graph key and executed-lane claim all say bf16."""
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
    """The ladder resolved fp8 because the deploy said the bytes were staged.
    Detection reads the safetensors HEADERS, so an empty result means the tree
    on disk is not the one the contract names — and the honest move is to
    refuse, not to quietly serve whatever is there."""
    (tmp_path / "unet").mkdir(parents=True)
    with pytest.raises(M.LaneMaterializationError, match="no fp8 artifact"):
        M.materialize(_Pipe(_tiny_unet()), _resolved(FP8_BODY), tree=tmp_path)


def test_the_nvfp4_arm_refuses_the_same_way_the_fp8_arm_does(
    tmp_path: Path,
) -> None:
    """Symmetry is the point. The nvfp4 rung is proven against a FABRICATED
    document (the flat nvfp4 layout pgw serves has no registered contract —
    `models/w4a4.py:500-510` says so deliberately, since it is NOT
    `bfl.nvfp4-preswizzled@1` and conflating them measured LPIPS 1.11), so what
    can be proven today is that its arm refuses on the same terms rather than
    being the one path that quietly serves whatever is on disk."""
    (tmp_path / "transformer").mkdir(parents=True)
    with pytest.raises(M.LaneMaterializationError, match="no fp4 artifact"):
        M.materialize(_Pipe(_tiny_unet()), _resolved("nvfp4-w4a4-static"),
                      tree=tmp_path)


def test_a_body_with_no_materializer_is_refused_rather_than_ignored():
    """If the lane table can name a body nothing can build, the table and the
    materializer disagree and one of them is wrong. Silence would pick the
    wrong one."""
    with pytest.raises(M.LaneMaterializationError, match="no materializer"):
        M.materialize(_Pipe(_tiny_unet()), _resolved("svdq-fp4-w4a4"),
                      tree=Path("."))


# --------------------------------------------------------------------------
# ctx.load_pipeline
# --------------------------------------------------------------------------


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
    """End to end on the eager bridge: a resolved baseline lane, a real tree,
    a real pipeline — and the confession on the log naming the lane, the
    reason and the rejected rungs."""
    import logging

    from diffusers import DDPMPipeline

    from gen_worker.serving.context import DeployBinding, LoadContext

    resolved = L.ResolvedLane(
        declared=_lane("t.bf16@1"), body=BF16_BODY,
        reason=L.CHOSE_BASELINE, card=L.CardFacts(sm=89, name="fixture"),
        rejected=(L.RejectedRung(body=FP8_BODY, contract_id="t.fp8@1",
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
    """A fixture, a derive and the local CLI all build a context with no
    ladder behind it. No silent default lane — the call just builds."""
    from diffusers import DDPMPipeline

    from gen_worker.serving.context import DeployBinding, LoadContext

    ctx: LoadContext = LoadContext(
        binding=DeployBinding(checkpoint_ref="t@1", checkpoint_dir=tiny_tree))
    assert ctx.resolved_lane is None
    assert ctx.load_pipeline(DDPMPipeline) is not None


def test_the_resolved_lane_selects_its_own_tree_out_of_a_multi_lane_binding(
    tiny_tree: Path,
) -> None:
    """The binding half of the multi-lane fix: `checkpoint_dir` answers the
    tree for the lane the ladder picked, and for the UPCAST rung it answers
    the tree whose BYTES are fetched, not the lane's own."""
    from gen_worker.serving.context import DeployBinding, LoadContext

    binding = DeployBinding(
        checkpoint_ref="t@1", checkpoint_dir=Path("/nonexistent/default"),
        lane_trees={"t.bf16@1": Path("/trees/bf16"),
                    "t.fp8@1": tiny_tree},
    )
    plain: LoadContext = LoadContext(
        binding=binding,
        resolved=L.ResolvedLane(declared=_lane("t.bf16@1"),
                                body=BF16_BODY, reason=L.CHOSE_BASELINE),
    )
    assert plain.checkpoint_dir == Path("/trees/bf16")

    upcast: LoadContext = LoadContext(
        binding=binding,
        resolved=L.ResolvedLane(declared=_lane("t.bf16@1"),
                                body=BF16_BODY, reason=L.CHOSE_UPCAST,
                                fetch_contract="t.fp8@1",
                                transfer_saved_bytes=3_400_000_000),
    )
    assert upcast.checkpoint_dir == tiny_tree, (
        "the upcast rung fetches the QUANTIZED tree and serves baseline "
        "modules out of it")

    # A single-lane binding carries no map and is unchanged.
    single: LoadContext = LoadContext(
        binding=DeployBinding(checkpoint_ref="t@1", checkpoint_dir=tiny_tree),
        resolved=plain.resolved_lane,
    )
    assert single.checkpoint_dir == tiny_tree
