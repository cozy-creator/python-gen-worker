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
