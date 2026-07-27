"""Three defects the SDK genericity audit found (pgw#740).

None of these is a migration — they are bugs that per-family hardcoding caused:

1. `repackage.detect_diffusers_family` routed the SDXL two-text-encoder
   signature to the SD1.5 converter, silently. Held shut only by an unrelated
   allowlist in `clone.py`, while the function is publicly exported.
2. `utils/lora._denoiser_fingerprint` digested UNet-only config fields, so every
   DiT fingerprinted to `"|||"` — a normalized-split cache COLLISION between
   structurally different transformers.
3. `models/memory`'s group-offload loop iterated a fixed component list, so
   Wan's `transformer_2` and LTX's `connectors` were never offloaded — silently
   resident, with the caller believing otherwise.
"""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")

from gen_worker.convert import repackage  # noqa: E402
from gen_worker.models import memory  # noqa: E402
from gen_worker.utils import lora as lora_util  # noqa: E402


# --------------------------------------------------------------------------
# 1. converter/signature divergence
# --------------------------------------------------------------------------

def _sdxl_tree(tmp_path):
    for part in ("unet", "vae", "text_encoder", "text_encoder_2", "tokenizer_2"):
        (tmp_path / part).mkdir(parents=True)
    return tmp_path


def _sd15_tree(tmp_path):
    for part in ("unet", "vae", "text_encoder"):
        (tmp_path / part).mkdir(parents=True)
    return tmp_path


def test_sdxl_signature_detects_as_sdxl(tmp_path):
    """A second text encoder IS the SDXL signature — `convert/layout.py` has
    always said so for the identical signal."""
    assert repackage.detect_diffusers_family(_sdxl_tree(tmp_path)) == "sdxl"


def test_sd15_signature_is_unchanged(tmp_path):
    assert repackage.detect_diffusers_family(_sd15_tree(tmp_path)) == "sd15_sd2"


def test_converter_refuses_a_signature_it_does_not_match(tmp_path):
    """The guard is independent of the detection fix: a CALLER passing the
    wrong family must not get the wrong converter silently either."""
    sdxl = _sdxl_tree(tmp_path / "sdxl")
    with pytest.raises(repackage.ConverterSignatureError) as exc:
        repackage.convert_diffusers_to_singlefile(
            sdxl, tmp_path / "out.safetensors", family="sd15_sd2")
    message = str(exc.value)
    assert "sd15_sd2" in message and "SDXL signature" in message

    sd15 = _sd15_tree(tmp_path / "sd15")
    with pytest.raises(repackage.ConverterSignatureError) as exc:
        repackage.convert_diffusers_to_singlefile(
            sd15, tmp_path / "out2.safetensors", family="sdxl")
    assert "NO second text encoder" in str(exc.value)


# --------------------------------------------------------------------------
# 2. DiT fingerprint collision
# --------------------------------------------------------------------------

class _Cfg:
    def __init__(self, **fields):
        for k, v in fields.items():
            setattr(self, k, v)


class _Denoiser(torch.nn.Module):
    def __init__(self, cfg, width=4):
        super().__init__()
        self.config = cfg
        self.lin = torch.nn.Linear(width, width)


class _Pipe:
    pass


def _dit(**fields):
    pipe = _Pipe()
    pipe.transformer = _Denoiser(_Cfg(**fields))
    return pipe


def test_two_different_dits_do_not_share_a_fingerprint():
    """The bug: UNet-only fields meant every DiT digested to '|||'."""
    a = lora_util._denoiser_fingerprint(_dit(num_layers=19, num_attention_heads=24))
    b = lora_util._denoiser_fingerprint(_dit(num_layers=48, num_attention_heads=16))
    assert a and b
    assert a != b, "two structurally different DiTs share a cache fingerprint"
    assert "|||" not in a


def test_unet_fingerprint_still_discriminates():
    a = lora_util._denoiser_fingerprint(_dit(
        down_block_types=("A", "B"), block_out_channels=(320, 640),
        layers_per_block=2))
    b = lora_util._denoiser_fingerprint(_dit(
        down_block_types=("A", "B"), block_out_channels=(320, 1280),
        layers_per_block=2))
    assert a != b


def test_a_config_with_no_known_field_still_fingerprints_structurally():
    """A fingerprint that resolves to nothing is not a key, it is a
    collision — so an unrecognized config falls back to structure."""
    a = lora_util._denoiser_fingerprint(_dit(mystery_field="x"))
    pipe = _Pipe()
    pipe.transformer = _Denoiser(_Cfg(mystery_field="x"), width=8)
    b = lora_util._denoiser_fingerprint(pipe)
    assert a and b and a != b


# --------------------------------------------------------------------------
# 3. group offload skipping components silently
# --------------------------------------------------------------------------

class _OffloadableModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        self.offloaded = False

    def enable_group_offload(self, **_kwargs):
        self.offloaded = True


class _WanLikePipe:
    """Two MoE experts plus a VAE — `transformer_2` is the one that used to be
    skipped."""

    def __init__(self):
        self.transformer = _OffloadableModule()
        self.transformer_2 = _OffloadableModule()
        self.vae = _OffloadableModule()


def _run_group_offload(pipe, monkeypatch, caplog):
    """`_apply_group_offload` needs a CUDA-looking process; nothing here
    allocates on a device (the stubs record the call), so faking the probe is
    honest for testing the ENUMERATION, which is what these tests are about."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    applied: dict = {}
    with caplog.at_level(logging.WARNING, logger=memory._LOG.name):
        memory._apply_group_offload(pipe, applied, offload_to_disk_path=None)
    return applied


def test_second_expert_and_connectors_are_offloaded(monkeypatch, caplog):
    pipe = _WanLikePipe()
    pipe.connectors = _OffloadableModule()   # the LTX-shaped component
    applied = _run_group_offload(pipe, monkeypatch, caplog)

    assert pipe.transformer.offloaded
    assert pipe.transformer_2.offloaded, (
        "Wan's second MoE expert was skipped — it stays fully resident")
    assert pipe.connectors.offloaded, "LTX's connectors were skipped"
    assert applied.get("group_offload") is True


def test_a_component_that_cannot_offload_is_named_loudly(monkeypatch, caplog):
    """A component with no `enable_group_offload` normally still gets covered
    by diffusers' generic `apply_group_offloading`. When even that fails, the
    component stays FULLY RESIDENT while the caller believes offload was
    applied — the silence the fail-loud rule exists to remove."""
    import diffusers.hooks as hooks

    class _Stubborn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

    def _refuse(module, **_kwargs):
        if isinstance(module, _Stubborn):
            raise RuntimeError("unsupported module layout")

    monkeypatch.setattr(hooks, "apply_group_offloading", _refuse)

    pipe = _WanLikePipe()
    pipe.image_encoder = _Stubborn()
    _run_group_offload(pipe, monkeypatch, caplog)

    warnings = [r.getMessage() for r in caplog.records
                if "did NOT cover" in r.getMessage()]
    assert warnings, "a component left fully resident was not reported"
    assert "image_encoder" in warnings[0]
    assert "stay fully resident" in warnings[0]
