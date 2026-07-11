"""Cross-era config/weights parse regression (the ie#393 rope trap, generalized).

The serving stack must parse BOTH serialization eras to identical effective
values:

  - "old era": trees serialized by transformers 4.x — ``rope_scaling`` +
    ``rope_theta`` config keys, slow-tokenizer files (vocab.json/merges.txt,
    no tokenizer.json), ``text_model.``-prefixed text-encoder state dicts.
  - "new era": trees serialized by transformers 5.x — ``rope_parameters``,
    fast-only tokenizer.json, unprefixed state dicts.

History: transformers 4.57 silently parsed a 5.x-serialized Gemma3 TE config
as rope_scaling=None (ie#393 — would have changed LTX TE behavior in prod),
and 4.x loads of 5.x-saved CLIP text encoders re-initialized ALL TE weights
randomly (te#45). These tests pin the inverse direction now that the stack is
5.x-primary: old-era artifacts keep loading with identical semantics, and
round-trips through the current serializer stay stable. Any transformers bump
that breaks either direction fails here, at unit speed, before it can reach a
serve image.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig,
    AutoTokenizer,
    CLIPTextConfig,
    CLIPTextModel,
)


def _effective_rope(cfg) -> dict:
    """Per-layer-type effective rope values (ignores stray top-level keys)."""
    rp = cfg.rope_parameters
    return {
        layer: {k: v for k, v in params.items() if v is not None}
        for layer, params in rp.items()
        if isinstance(params, dict)
    }


def test_gemma3_legacy_rope_scaling_parses_like_rope_parameters(tmp_path: Path) -> None:
    """4.x-era Gemma3 config (rope_scaling/rope_theta) == its 5.x re-serialization.

    This is the exact LTX-2.3 TE shape: linear factor-8 scaling on the
    full-attention rope, default local rope for sliding attention.
    """
    legacy = {
        "architectures": ["Gemma3TextModel"],
        "model_type": "gemma3_text",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "vocab_size": 100,
        "rope_theta": 1000000.0,
        "rope_local_base_freq": 10000.0,
        "rope_scaling": {"rope_type": "linear", "factor": 8.0},
    }
    old_dir = tmp_path / "old-era"
    old_dir.mkdir()
    (old_dir / "config.json").write_text(json.dumps(legacy))

    cfg_old = AutoConfig.from_pretrained(old_dir)
    rope_old = _effective_rope(cfg_old)

    # The trap: rope scaling silently dropped -> factor missing entirely.
    assert rope_old["full_attention"]["rope_type"] == "linear"
    assert rope_old["full_attention"]["factor"] == 8.0
    assert rope_old["full_attention"]["rope_theta"] == 1000000.0
    assert rope_old["sliding_attention"]["rope_theta"] == 10000.0

    # New era: re-serialize under the current stack, re-parse, compare.
    new_dir = tmp_path / "new-era"
    cfg_old.save_pretrained(new_dir)
    serialized = json.loads((new_dir / "config.json").read_text())
    assert "rope_parameters" in serialized  # 5.x vocabulary on disk
    cfg_new = AutoConfig.from_pretrained(new_dir)
    assert _effective_rope(cfg_new) == rope_old


def test_clip_slow_tokenizer_files_load_and_reserialize_identically(tmp_path: Path) -> None:
    """4.x-era slow CLIP tokenizer tree (vocab.json+merges.txt) still loads
    and encodes identically to its 5.x fast-only re-serialization (te#45)."""
    old_dir = tmp_path / "old-era"
    old_dir.mkdir()
    vocab = {
        "<|startoftext|>": 0,
        "<|endoftext|>": 1,
        "a</w>": 2,
        "b</w>": 3,
        "ab</w>": 4,
        "a": 5,
        "b": 6,
    }
    (old_dir / "vocab.json").write_text(json.dumps(vocab))
    (old_dir / "merges.txt").write_text("#version: 0.2\na b</w>\n")
    (old_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "CLIPTokenizer",
                "add_bos_token": True,
                "add_eos_token": True,
                "bos_token": "<|startoftext|>",
                "eos_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "model_max_length": 77,
            }
        )
    )

    tok_old = AutoTokenizer.from_pretrained(old_dir)
    ids_old = tok_old("a b ab")["input_ids"]
    # bos/eos wrapping must survive (ie#393: explicit add_bos/eos flags lost).
    assert ids_old[0] == 0 and ids_old[-1] == 1
    assert ids_old == [0, 2, 3, 4, 1]

    new_dir = tmp_path / "new-era"
    tok_old.save_pretrained(new_dir)
    tok_new = AutoTokenizer.from_pretrained(new_dir)
    assert tok_new("a b ab")["input_ids"] == ids_old


def test_clip_text_encoder_prefixed_state_dict_loads_identically(tmp_path: Path) -> None:
    """4.x-era ``text_model.``-prefixed TE state dict loads to the same
    forward as the current-era unprefixed save (te#45 silent-corruption case:
    a mismapped load re-initializes every TE weight randomly)."""
    cfg = CLIPTextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        vocab_size=100,
        bos_token_id=0,
        eos_token_id=1,
    )
    torch.manual_seed(0)
    model = CLIPTextModel(cfg)

    new_dir = tmp_path / "new-era"
    model.save_pretrained(new_dir)
    sd = load_file(new_dir / "model.safetensors")

    old_dir = tmp_path / "old-era"
    old_dir.mkdir()
    (old_dir / "config.json").write_text((new_dir / "config.json").read_text())
    save_file(
        {"text_model." + k: v for k, v in sd.items()},
        str(old_dir / "model.safetensors"),
        metadata={"format": "pt"},
    )

    x = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        ref = model(x).last_hidden_state
        out_old = CLIPTextModel.from_pretrained(old_dir)(x).last_hidden_state
        out_new = CLIPTextModel.from_pretrained(new_dir)(x).last_hidden_state

    # Random re-init would make these wildly unequal.
    assert torch.equal(ref, out_old)
    assert torch.equal(ref, out_new)


def test_scheduler_config_roundtrip_preserves_time_shift_type(tmp_path: Path) -> None:
    """Scheduler config re-serialization keeps non-default fields (ie#393:
    the OzzyGT re-serialization dropped time_shift_type et al)."""
    pytest.importorskip("diffusers")
    from diffusers import FlowMatchEulerDiscreteScheduler

    sched = FlowMatchEulerDiscreteScheduler(shift=3.0, time_shift_type="linear")
    sched.save_pretrained(tmp_path / "sched")
    reloaded = FlowMatchEulerDiscreteScheduler.from_pretrained(tmp_path / "sched")
    assert reloaded.config.time_shift_type == "linear"
    assert reloaded.config.shift == 3.0
