"""pgw#566 — a converter that half-converts must not outrank the raw keys.

Live find on the 5090 rig (real `StableDiffusionXLPipeline` on a w8a8 tree,
nerijs/pixel-art-xl r32 kohya adapter): `normalize_adapter_state_dict` ->
`StableDiffusionXLPipeline.lora_state_dict` emitted HALF-converted keys —
`unet.down_blocks.4.1.proj_in.lora.down.weight`, an SGM block index RENAMED
but never REMAPPED (SDXL has down_blocks 0-2) — so `map_adapter` failed loud
with 2166 unresolved keys and the w8a8-lane request died typed. The RAW dict
through `map_adapter`'s own kohya/SGM grammar mapped 722/722 modules on the
same model.

Two causes, both closed here:

1. `unet_config` reached only converters with a NAMED `unet_config` parameter.
   SDXL's takes `**kwargs`, so the SGM block remap never ran on the one class
   that needs it. (Covered by the un-xfailed row in
   `tests/convert/test_p8_convert_publish_contract.py`.)
2. The only guard on a bad conversion was gw#627's `.processor.` NAME test,
   which catches the family it was written from and nothing else. Half-
   converted SGM indices contain no `.processor.` and sailed through. The
   general question — does the converted dict actually RESOLVE against this
   model — is asked here with `map_adapter`'s own oracle, so the check and the
   mapper cannot drift.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch

from gen_worker.models import w8a8_lora


class _Denoiser(torch.nn.Module):
    """SDXL-shaped enough for the block-index question: down_blocks 0-2."""

    def __init__(self) -> None:
        super().__init__()
        self.down_blocks = torch.nn.ModuleList([
            torch.nn.ModuleList([torch.nn.Linear(8, 8)]) for _ in range(3)
        ])
        self.config = {"down_block_types": ("a", "b", "c")}


def _pipe(converter) -> Any:
    class _Pipe:
        unet = None

        @staticmethod
        def lora_state_dict(sd: Dict[str, Any], **kwargs: Any):
            return converter(sd, **kwargs)

    p = _Pipe()
    p.unet = _Denoiser()
    return p


def _raw() -> Dict[str, Any]:
    """Raw kohya-flat keys that DO resolve against the model's own paths."""
    return {
        "unet.down_blocks.0.0.lora_down.weight": torch.zeros(4, 8),
        "unet.down_blocks.0.0.lora_up.weight": torch.zeros(8, 4),
    }


def _half_converted() -> Dict[str, Any]:
    """The live failure shape: block index renamed, never remapped. SDXL has
    no `down_blocks.4`, and nothing here contains `.processor.`."""
    return {
        "unet.down_blocks.4.1.lora.down.weight": torch.zeros(4, 8),
        "unet.down_blocks.4.1.lora.up.weight": torch.zeros(8, 4),
    }


def test_a_half_converted_dict_does_NOT_outrank_the_raw_keys() -> None:
    """RED before pgw#566: the converted dict was preferred unconditionally
    (unless it happened to contain `.processor.`), and `map_adapter` then
    raised RefCompatibilitySurprise on every key."""
    pipe = _pipe(lambda sd, **kw: _half_converted())
    out = w8a8_lora.normalize_adapter_state_dict(pipe, _raw(), ref="pixel-art-xl")

    assert set(out) == set(_raw()), (
        "a strictly worse conversion must not be preferred")
    # and the raw keys the fallback keeps are the ones that actually map
    mapped = w8a8_lora.map_adapter(
        {k.split("unet.", 1)[1]: v for k, v in out.items()},
        pipe.unet, ref="pixel-art-xl")
    assert mapped, "the retained raw keys must resolve against the model"


def test_a_GOOD_conversion_is_still_preferred() -> None:
    """te#81's zero-drift path stays the default — the guard must only fire on
    a conversion that is strictly WORSE, never on one that works."""
    good = {
        "unet.down_blocks.1.0.lora_down.weight": torch.zeros(4, 8),
        "unet.down_blocks.1.0.lora_up.weight": torch.zeros(8, 4),
    }
    pipe = _pipe(lambda sd, **kw: dict(good))
    out = w8a8_lora.normalize_adapter_state_dict(pipe, _raw(), ref="ok")
    assert set(out) == set(good)


def test_an_equally_unresolvable_conversion_is_still_preferred() -> None:
    """The rule is "strictly worse", not "not perfect". When both forms fail,
    `map_adapter` must get the converted dict and raise its own typed error —
    silently swapping to raw keys would only move where the failure appears."""
    bad = {
        "unet.down_blocks.9.0.lora.down.weight": torch.zeros(4, 8),
        "unet.down_blocks.9.0.lora.up.weight": torch.zeros(8, 4),
    }
    raw_bad = {
        "unet.down_blocks.8.0.lora_down.weight": torch.zeros(4, 8),
        "unet.down_blocks.8.0.lora_up.weight": torch.zeros(8, 4),
    }
    pipe = _pipe(lambda sd, **kw: dict(bad))
    out = w8a8_lora.normalize_adapter_state_dict(pipe, raw_bad, ref="x")
    assert set(out) == set(bad)


def test_unet_config_reaches_a_kwargs_only_converter() -> None:
    """The direct cause. `inspect.signature(...).parameters` on diffusers'
    real SDXL converter has no `unet_config` — it has `**kwargs`."""
    seen: Dict[str, Any] = {}

    def _conv(sd, **kw):
        seen.update(kw)
        return dict(sd)

    w8a8_lora.normalize_adapter_state_dict(_pipe(_conv), _raw(), ref="x")
    assert "unet_config" in seen


@pytest.mark.parametrize("fn, expected", [
    (lambda sd, unet_config=None: sd, True),   # named
    (lambda sd, **kw: sd, True),               # VAR_KEYWORD — the SDXL shape
    (lambda sd: sd, False),                    # neither: do not invent a kwarg
])
def test_the_signature_rule_itself(fn, expected: bool) -> None:
    assert w8a8_lora._accepts_unet_config(fn) is expected


def test_the_probe_never_breaks_an_overlay() -> None:
    """An unanswerable probe must not decide. A pipeline with no resident
    denoiser cannot be asked whether keys resolve, so the conversion stands —
    the old behaviour, unchanged, for every case the guard cannot judge."""
    class _NoDenoiser:
        @staticmethod
        def lora_state_dict(sd, **kw):
            return _half_converted()

    out = w8a8_lora.normalize_adapter_state_dict(_NoDenoiser(), _raw(), ref="x")
    assert set(out) == set(_half_converted())


def test_the_guard_uses_map_adapters_own_oracle_not_a_second_grammar() -> None:
    """Two spellings of "does this key resolve" is how a guard passes an
    adapter the mapper then refuses. The probe must call `_group_keys` against
    `branch_modules`, which is exactly what `map_adapter` does."""
    from pathlib import Path

    src = Path(w8a8_lora.__file__).read_text()
    body = src[src.index("def _unresolved_count"):src.index("def _converted_resolves")]
    assert "_group_keys(" in body and "branch_modules(" in body
