"""Two defects the SDK genericity audit found.

(The third — detection/converter signature divergence routing SDXL to the
SD1.5 converter — is pinned generically in tests/test_repack_engine_pgw740.py:
detection and the guard both read the ONE declared signature, dual-encoder
trees are refused by single-encoder converters and vice versa.)

1. `utils/lora._denoiser_fingerprint` digested UNet-only config fields, so every
   DiT fingerprinted to `"|||"` — a normalized-split cache COLLISION between
   structurally different transformers.
2. `models/memory`'s group-offload loop iterated a fixed component list, so
   Wan's `transformer_2` and LTX's `connectors` were never offloaded — silently
   resident, with the caller believing otherwise.
"""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")

from gen_worker.models import memory  # noqa: E402
from gen_worker.utils import lora as lora_util  # noqa: E402


# --------------------------------------------------------------------------
# 1. DiT fingerprint collision
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


def test_structurally_different_denoisers_never_share_a_fingerprint():
    """The bug: UNet-only fields meant every DiT digested to '|||' — a
    normalized-split cache collision. Three axes of the same invariant: DiT
    configs discriminate, UNet configs still discriminate, and a config with
    no known field falls back to structure (a fingerprint that resolves to
    nothing is not a key, it is a collision)."""
    a = lora_util._denoiser_fingerprint(_dit(num_layers=19, num_attention_heads=24))
    b = lora_util._denoiser_fingerprint(_dit(num_layers=48, num_attention_heads=16))
    assert a and b
    assert a != b, "two structurally different DiTs share a cache fingerprint"
    assert "|||" not in a

    u1 = lora_util._denoiser_fingerprint(_dit(
        down_block_types=("A", "B"), block_out_channels=(320, 640),
        layers_per_block=2))
    u2 = lora_util._denoiser_fingerprint(_dit(
        down_block_types=("A", "B"), block_out_channels=(320, 1280),
        layers_per_block=2))
    assert u1 != u2

    s1 = lora_util._denoiser_fingerprint(_dit(mystery_field="x"))
    pipe = _Pipe()
    pipe.transformer = _Denoiser(_Cfg(mystery_field="x"), width=8)
    s2 = lora_util._denoiser_fingerprint(pipe)
    assert s1 and s2 and s1 != s2


# --------------------------------------------------------------------------
# 2. group offload skipping components silently
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

    # The outcome is the fail-loud report: a WARNING-or-worse record naming the
    # component that stayed resident. The wording is the module's own business.
    warnings = [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING]
    assert any("image_encoder" in w for w in warnings), (
        f"a component left fully resident was not reported: {warnings}")
