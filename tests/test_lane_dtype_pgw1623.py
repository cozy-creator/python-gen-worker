"""pgw#1623: the streamed load lands the LANE's dtype, or it says why not.

The article is the production layout, measured off the hub's own catalog rather
than invented: `tensorhub/wai-illustrious@prod-fp16vae`, the checkpoint sdxl
0.4.1 bound on pod `xvuc95yqw6buzz`, carries

    component_dtypes {text_encoder: fp16, text_encoder_2: fp16,
                      unet: fp16, vae: fp32}   (source: safetensors-headers)

on a lane (`sdxl.diffusers@1+plain.bf16@1`) whose QUANT RULE declares
**bfloat16**. Three dtypes, one pipeline. The eager `from_pretrained(torch_dtype=…)` bridge flattens that
to the lane's dtype and always has; the streaming engine passed it through, and
the result died `Input type (c10::Half) and bias type (float) should be the
same` in the VAE's first conv — a real request, burned on a rented card.

No mocks: a real diffusers pipeline saved per-component at those dtypes, into a
real chunked CAS, projected the way the chokepoint projects it, and loaded
through the production `StreamingLoader.build`. The forward arm runs the
mismatch that actually killed the request — the unet's own output dtype into
`vae.decode` — so the fix is proven by serving, not only by inspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from cas_fixture import ingest_repository  # noqa: E402
from gen_worker._vendor.tensorfs import LocalCAS, project_snapshot  # noqa: E402
from gen_worker.models.projection import REF_PREFIX, SNAPSHOTS_DIR  # noqa: E402
from gen_worker.serving.streaming import (  # noqa: E402
    BridgeWeightStore,
    StreamingLoader,
)
from gen_worker.serving.streaming.engine import LaneDtypeUnmet  # noqa: E402
from streaming_fixture import Lane, tiny_pipeline_class  # noqa: E402

STORED: Dict[str, Any] = {
    "unet": torch.float16,
    "vae": torch.float32,
    "text_encoder": torch.float16,
    "text_encoder_2": torch.float16,
}


def _heterogeneous_source(target: Path) -> type:
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel

    torch.manual_seed(1623)
    unet = UNet2DConditionModel(
        sample_size=16, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(32, 64), norm_num_groups=4, cross_attention_dim=32,
        attention_head_dim=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
    )
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=4, norm_num_groups=4,
        block_out_channels=(32,), down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
    )
    text_config = CLIPTextConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, vocab_size=256, max_position_embeddings=32,
        projection_dim=32,
    )
    pipeline_cls = tiny_pipeline_class()
    def at(module: Any, dtype: Any) -> Any:
        module.to(dtype)
        return module

    pipeline = pipeline_cls(
        unet=at(unet, STORED["unet"]),
        vae=at(vae, STORED["vae"]),
        text_encoder=at(CLIPTextModel(text_config), STORED["text_encoder"]),
        text_encoder_2=at(CLIPTextModel(text_config), STORED["text_encoder_2"]),
        scheduler=DDIMScheduler(),
    )
    pipeline.save_pretrained(str(target), safe_serialization=True)
    return pipeline_cls


def _project(base: Path, source: Path, key: str) -> Path:
    cas = LocalCAS(base)
    manifest = ingest_repository(cas, source)
    cas.compare_and_swap_ref(
        REF_PREFIX + key, cas.store_manifest(manifest), expected=None
    )
    tree = base / SNAPSHOTS_DIR / key
    project_snapshot(cas, manifest, tree)
    return tree


def _cas_manifest(tree: Path) -> Tuple[Any, Any]:
    from gen_worker.models import projection

    projected = projection.resolve_projection(tree)
    assert projected is not None, f"{tree} has no chunk store behind it"
    return projected.cas, projected.manifest


@pytest.fixture(scope="module")
def loaded(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """The production load: heterogeneous tree, bf16 lane, real engine."""
    base = tmp_path_factory.mktemp("pgw1623")
    source = base / "source-model"
    pipeline_cls = _heterogeneous_source(source)
    tree = _project(base, source, key="c" * 64)
    loader = StreamingLoader(
        BridgeWeightStore(*_cas_manifest(tree)), device="cpu", buffer_bytes=4096
    )
    pipeline = loader.build(pipeline_cls, checkpoint_dir=tree, lane=Lane)
    return {"pipeline": pipeline, "report": loader.last_report, "tree": tree,
            "source": source, "pipeline_cls": pipeline_cls}


def _wide_floats(module: Any) -> Dict[str, Any]:
    held = dict(module.named_parameters(remove_duplicate=False))
    held.update(dict(module.named_buffers(remove_duplicate=False)))
    return {
        name: tensor.dtype for name, tensor in held.items()
        if tensor is not None
        and tensor.dtype in (torch.float64, torch.float32,
                             torch.float16, torch.bfloat16)
    }


def test_the_article_really_is_three_dtypes_on_disk(loaded: Dict[str, Any]) -> None:
    """The control arm: without it a green suite proves only that the fixture is uniform, which is the fixture bug this whole class hides behind."""
    from safetensors import safe_open

    stored: Dict[str, set] = {}
    source = loaded["source"]
    for container in sorted(source.rglob("*.safetensors")):
        with safe_open(str(container), framework="pt") as handle:
            for name in handle.keys():
                stored.setdefault(container.parent.name, set()).add(
                    handle.get_slice(name).get_dtype()
                )
    assert stored["unet"] == {"F16"}, stored
    assert stored["vae"] == {"F32"}, stored
    assert stored["text_encoder"] == {"F16"}, stored


def test_the_loaded_pipeline_is_one_dtype_and_it_is_the_lanes(
    loaded: Dict[str, Any],
) -> None:
    pipeline = loaded["pipeline"]
    for component in STORED:
        dtypes = set(_wide_floats(getattr(pipeline, component)).values())
        assert dtypes == {Lane.dtype}, (
            f"{component} came back {sorted(str(d) for d in dtypes)} on a lane "
            f"declaring {Lane.dtype}"
        )


def test_the_repair_is_counted_not_silent(loaded: Dict[str, Any]) -> None:
    """A store defect that is repaired and NOT reported is the same defect one layer down."""
    report = loaded["report"]
    assert report.cast_to_lane > 0
    assert report.attributes()["cast_to_lane"] == report.cast_to_lane


def test_the_unet_output_survives_the_vae_pgw1623(loaded: Dict[str, Any]) -> None:
    """THE REQUEST THAT BURNED."""
    pipeline = loaded["pipeline"]
    latents = torch.randn(
        1, 4, 8, 8, dtype=next(pipeline.unet.parameters()).dtype)
    decoded = pipeline.vae.decode(latents, return_dict=False)[0]
    assert decoded.dtype == Lane.dtype


def test_a_tensor_that_dodges_the_cast_refuses_by_name(
    loaded: Dict[str, Any],
) -> None:
    """The fence can go RED."""
    pipeline = loaded["pipeline"]
    module = pipeline.vae
    leaf = next(name for name, _ in module.named_parameters())
    owner = module.get_submodule(leaf.rpartition(".")[0]) if "." in leaf else module
    key = leaf.rpartition(".")[2]
    held = owner._parameters[key]
    owner._parameters[key] = torch.nn.Parameter(
        held.to(torch.float32), requires_grad=False)
    try:
        with pytest.raises(LaneDtypeUnmet) as caught:
            StreamingLoader._assert_lane_dtype({"vae": module}, Lane.dtype)
        assert leaf in str(caught.value)
        assert "float32" in str(caught.value)
    finally:
        owner._parameters[key] = held
