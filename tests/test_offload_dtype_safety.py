"""gw#441/gw#469: no offload rung may render with broken dtype.

A ``force_upcast`` VAE (SDXL family) mutates dtype at decode (``upcast_vae``
-> fp32 -> back); hook-managed weights miss the runtime cast and decode fatals
with Half/float mismatches. Every offload rung must keep such a VAE resident:
group offload excludes it, model/sequential rungs exclude it via diffusers'
``_exclude_from_cpu_offload`` (which moves excluded components to the
execution device itself).
"""

from __future__ import annotations

import logging

import pytest

from gen_worker.models.memory import _dtype_fragile_vae, _pin_fragile_vae

diffusers = pytest.importorskip("diffusers")


def _tiny_vae(force_upcast: bool):
    from diffusers import AutoencoderKL

    return AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(4,), layers_per_block=1,
        latent_channels=4, norm_num_groups=2, sample_size=8,
        force_upcast=force_upcast,
    )


class _Pipe:
    def __init__(self, vae) -> None:
        self.vae = vae


def test_force_upcast_vae_detected() -> None:
    assert _dtype_fragile_vae(_Pipe(_tiny_vae(True))) is not None
    assert _dtype_fragile_vae(_Pipe(_tiny_vae(False))) is None  # fp16-fix vae
    assert _dtype_fragile_vae(_Pipe(None)) is None


def test_pin_excludes_vae_from_cpu_offload_hooks() -> None:
    pipe = _Pipe(_tiny_vae(True))
    applied: dict = {}
    _pin_fragile_vae(pipe, applied, logging.getLogger(__name__))
    assert "vae" in pipe._exclude_from_cpu_offload
    assert applied.get("vae_resident") is True
    # idempotent — a second application must not duplicate the entry
    _pin_fragile_vae(pipe, applied, logging.getLogger(__name__))
    assert pipe._exclude_from_cpu_offload.count("vae") == 1


def test_pin_leaves_safe_vae_hook_managed() -> None:
    pipe = _Pipe(_tiny_vae(False))
    applied: dict = {}
    _pin_fragile_vae(pipe, applied, logging.getLogger(__name__))
    assert "vae" not in (getattr(pipe, "_exclude_from_cpu_offload", None) or [])
    assert "vae_resident" not in applied


def test_real_pipeline_exclusion_is_honored_by_diffusers() -> None:
    """The mechanism contract: DiffusionPipeline's offload rungs consult
    ``self._exclude_from_cpu_offload`` (and move excluded components to the
    execution device). Guard the attribute so a diffusers rename cannot
    silently re-hook the fragile VAE."""
    from diffusers import DiffusionPipeline

    assert hasattr(DiffusionPipeline, "_exclude_from_cpu_offload")
    import inspect

    src = inspect.getsource(DiffusionPipeline.enable_sequential_cpu_offload)
    assert "_exclude_from_cpu_offload" in src
    src = inspect.getsource(DiffusionPipeline.enable_model_cpu_offload)
    assert "_exclude_from_cpu_offload" in src
