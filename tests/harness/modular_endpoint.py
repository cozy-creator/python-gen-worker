"""pgw#1036 harness: a REAL tiny diffusers ModularPipeline (0.39 modular
API, real safetensors, CPU) plus an endpoint serving it, for the modular
hydration guard's integration tests.

The trees this builds reproduce the mirror disease exactly: every component
spec's ``pretrained_model_name_or_path`` in ``modular_model_index.json``
names an UPSTREAM repo id (``upstream/pgw1036-owned-by-hf``) that does not
exist — any load that is not re-pointed at the local tree fails loudly, and
the tests additionally run under ``HF_HUB_OFFLINE=1``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.modular_pipelines import ModularPipeline
from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec

from gen_worker import RequestContext, Slot, endpoint
from gen_worker.api.binding import Hub

from .toy_endpoints import EchoIn, EchoOut, _ToyDefaults

UPSTREAM_REPO_ID = "upstream/pgw1036-owned-by-hf"


class TinyBlocks(ModularPipelineBlocks):
    """Blocks that only declare component expectations — construction and
    hydration are what pgw#1036 tests, never a denoise."""

    # Truth-tested by ModularPipeline.__init__'s blocks resolution.
    block_classes = ["tiny"]

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(name="unet", type_hint=UNet2DConditionModel),
            ComponentSpec(name="vae", type_hint=AutoencoderKL),
            ComponentSpec(name="vae_ref", type_hint=AutoencoderKL),
            ComponentSpec(name="scheduler", type_hint=DDPMScheduler),
        ]


class TinyModularPipeline(ModularPipeline):
    default_blocks_name = "TinyBlocks"


# ModularPipeline.__init__ resolves the blocks class from the DIFFUSERS
# namespace — the vendored-package registration pattern (minimax-h3 does the
# same for MiniMaxH3Blocks).
diffusers.TinyBlocks = TinyBlocks
diffusers.TinyModularPipeline = TinyModularPipeline


def tiny_unet(fill: float) -> UNet2DConditionModel:
    m = UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(4, 8), norm_num_groups=4, cross_attention_dim=8,
        attention_head_dim=2,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
    )
    for p in m.parameters():
        p.data.fill_(fill)
    return m


class KeepFp32Unet(UNet2DConditionModel):
    """A DiT-shaped component in miniature (pgw#1071): a stack that stores
    bf16 plus heads the class declares must stay fp32 — H3's
    ``_keep_in_fp32_modules`` (patch/timestep/output projections) at test
    scale."""

    _keep_in_fp32_modules = ["conv_in"]


def tiny_keep_fp32_unet(fill: float) -> "KeepFp32Unet":
    base = tiny_unet(fill)
    m = KeepFp32Unet(**{
        k: v for k, v in base.config.items() if not str(k).startswith("_")
    })
    m.load_state_dict(base.state_dict())
    return m


diffusers.KeepFp32Unet = KeepFp32Unet


def tiny_vae(fill: float) -> AutoencoderKL:
    m = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=4, sample_size=8,
        layers_per_block=1, block_out_channels=(4,), norm_num_groups=4,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
    )
    for p in m.parameters():
        p.data.fill_(fill)
    return m


def _spec_entry(class_name: str, subfolder: str) -> list:
    return ["diffusers", class_name, {
        "pretrained_model_name_or_path": UPSTREAM_REPO_ID,
        "subfolder": subfolder,
        "variant": None,
        "revision": None,
        "type_hint": ["diffusers", class_name],
    }]


def build_base_tree(root: Path, *, fill: float = 1.0) -> Path:
    """The mirror shape: full components locally, every index spec naming
    the upstream repo id, and ``vae_ref`` as the config-only unselected
    partition (H3's ``transformer_ref`` shape)."""
    root.mkdir(parents=True, exist_ok=True)
    tiny_unet(fill).save_pretrained(root / "unet")
    vae = tiny_vae(fill)
    vae.save_pretrained(root / "vae")
    DDPMScheduler().save_pretrained(root / "scheduler")
    # config-only partition: config.json, no weights
    (root / "vae_ref").mkdir(exist_ok=True)
    (root / "vae_ref" / "config.json").write_text(
        (root / "vae" / "config.json").read_text())
    index = {
        "_class_name": "TinyModularPipeline",
        "_diffusers_version": diffusers.__version__,
        "_blocks_class_name": "TinyBlocks",
        "unet": _spec_entry("UNet2DConditionModel", "unet"),
        "vae": _spec_entry("AutoencoderKL", "vae"),
        "vae_ref": _spec_entry("AutoencoderKL", "vae_ref"),
        "scheduler": _spec_entry("DDPMScheduler", "scheduler"),
    }
    (root / "modular_model_index.json").write_text(json.dumps(index))
    # model_index.json too — the executor's override-name validation and the
    # loader's dtype/component helpers read it (the real mirror carries both).
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "TinyModularPipeline",
        "_diffusers_version": diffusers.__version__,
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "vae_ref": ["diffusers", "AutoencoderKL"],
        "scheduler": ["diffusers", "DDPMScheduler"],
    }))
    return root


def build_mixed_precision_tree(
    root: Path, *, fill: float = 1.0, unet_dtype: str = "bf16",
    vae_dtype: str = "fp32", extra_fp32_parts: int = 0,
) -> Path:
    """ie#615's minimax-h3 shape at test scale (pgw#1071): a big narrow
    denoiser (bf16 stack, fp32 ``_keep_in_fp32_modules`` heads) beside a VAE
    the checkpoint stores WIDE, which a pipeline-level bf16 must not
    downcast.

    ``extra_fp32_parts`` adds fp32 siblings until the snapshot-wide majority
    vote flips — the condition under which the old tree-wide sniff produced
    no dtype at all and diffusers' fp32 default upcast the bf16 stack."""
    import torch

    tree = build_base_tree(root, fill=fill)
    unet = tiny_keep_fp32_unet(fill).to(getattr(torch, _TORCH_DTYPES[unet_dtype]))
    unet.conv_in.to(torch.float32)  # the checkpoint's own wide heads
    unet.save_pretrained(tree / "unet")
    tiny_vae(fill).to(getattr(torch, _TORCH_DTYPES[vae_dtype])).save_pretrained(
        tree / "vae")

    index = json.loads((tree / "modular_model_index.json").read_text())
    model_index = json.loads((tree / "model_index.json").read_text())
    index["unet"] = _spec_entry("KeepFp32Unet", "unet")
    model_index["unet"] = ["diffusers", "KeepFp32Unet"]
    for i in range(extra_fp32_parts):
        name = f"vae_x{i}"
        tiny_vae(fill).to(torch.float32).save_pretrained(tree / name)
        index[name] = _spec_entry("AutoencoderKL", name)
        model_index[name] = ["diffusers", "AutoencoderKL"]
    (tree / "modular_model_index.json").write_text(json.dumps(index))
    (tree / "model_index.json").write_text(json.dumps(model_index))
    return tree


_TORCH_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def build_override_vae_tree(root: Path, *, fill: float = 2.0,
                            subdir: bool = True) -> Path:
    """A th#980 component-override tree: ``<root>/vae/`` (subdir layout) or
    the component at the root (both platform conventions; deliberately NO
    model_index.json — the encoder-trunc-fp8 shape)."""
    dest = (root / "vae") if subdir else root
    tiny_vae(fill).save_pretrained(dest)
    return root


def tree_files(root: Path) -> Dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


MODULAR_DECLARED = Hub("harness/tiny-modular", tag="prod")


@endpoint(models={
    "pipeline": Slot(TinyModularPipeline, default_checkpoint=MODULAR_DECLARED),
})
class ModularEndpoint:
    """Reports the hydration outcome so the wire result IS the assertion."""

    def setup(self, pipeline: TinyModularPipeline) -> None:
        self.pipe = pipeline

    def modular_echo(self, ctx: RequestContext[_ToyDefaults],
                     data: EchoIn) -> EchoOut:
        p = self.pipe
        vae_fill = float(next(iter(p.vae.parameters())).flatten()[0]) \
            if p.vae is not None else float("nan")
        prov = dict(getattr(p, "_cozy_modular_hydration", {}) or {})
        return EchoOut(response=(
            f"unet={'set' if p.unet is not None else 'none'}"
            f"|vae_fill={vae_fill:g}"
            f"|vae_ref={'set' if p.vae_ref is not None else 'none'}"
            f"|scheduler={'set' if p.scheduler is not None else 'none'}"
            f"|prov_vae={prov.get('vae', '')}"
        ))
