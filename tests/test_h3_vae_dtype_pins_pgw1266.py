"""MiniMax-H3's two VAEs are fp32-pinned, and the producer half now knows it.

ie#718 proposed casting H3's `vae/` to bf16 for −5.21 GB of every cold boot.
The free provenance check refused it, and this is the refusal made structural
— because until this row existed `families.facts` held exactly ONE entry
(`AutoencoderKLWan`), so `cast_exempt_components` would have cast H3's VAEs
and `verify_produced_tree` would have published the result without a word.

The evidence, at primary source in the vendored upstream bytes
(`minimax-h3/src/minimax_h3/vendored/diffusers_h3/models/`):

* `autoencoder_kl_minimax_h3.py:529` — *"The released checkpoint is float32
  and the verified decode recipe is float16 autocast over float32 weights …
  a pipeline-level `torch_dtype=torch.bfloat16` must therefore not downcast
  the weights"*, backed by `_keep_in_fp32_modules`. The decode compute is
  ALREADY fp16 (`modular/decoders.py:188`), so narrowing the weights buys
  bandwidth at 3 fewer mantissa bits than the arithmetic uses.
* `autoencoder_kl_minimax_h3_audio.py:527` — the DAC/BigVGAN stack
  *"degrades audibly under bfloat16 (roughly 20 dB quieter decodes)"*, and
  it decodes at its own parameter dtype with no autocast to widen it back.
* Our own banked arm (ie#621, research/h3-acceleration.md): fp16 DECODER
  WEIGHTS — strictly wider than bf16 — measured 74.97 dB PSNR min /
  0.0186 max abs against the fp32 decode at 0.94x. Worse on both axes.

`_keep_in_fp32_modules` is not a substitute for this table: it stops a
DOWNCAST of fp32-on-disk bytes and can restore nothing from bf16-on-disk
bytes. The truncation would be permanent and invisible.

Real codepaths throughout: the real `build_flavor_tree` cast over real
safetensors, the real `verify_produced_tree` publish gate, the real
`families.facts` table.

Run: pytest tests/test_h3_vae_dtype_pins_pgw1266.py -q
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from gen_worker.convert.clone import OutputSpec, build_flavor_tree  # noqa: E402
from gen_worker.convert.dtype_pins import (  # noqa: E402
    ComponentDtypePinError,
    ComponentDtypePinViolation,
    cast_exempt_components,
    component_dtypes_on_disk,
    component_pins,
    verify_produced_tree,
)
from gen_worker.convert.ingest import IngestedSource  # noqa: E402
from gen_worker.families.facts import component_dtype_for_class  # noqa: E402

#: Published bytes for `tensorhub/minimax-h3:serve-narrowed`, from
#: `repo_artifact_file_metadata` (ie#681 §2 / ie#718). The video VAE is the
#: whole of the proposed −5.21 GB; the audio VAE is 0.6 GB of it.
H3_VIDEO_VAE_BYTES = 10_415_561_155
H3_AUDIO_VAE_BYTES = 605_431_611

H3_MODEL_INDEX = {
    "_class_name": "MiniMaxH3ModularPipeline",
    "audio_vae": ["diffusers", "AutoencoderKLMiniMaxH3Audio"],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
    "transformer": ["diffusers", "MiniMaxH3Transformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLMiniMaxH3"],
}


def _save(path: Path, tensors: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(tensors, str(path))


def _h3_tree(root: Path) -> Path:
    """An H3 tree in miniature, in the shipped mixed-precision shape: both
    VAEs fp32 (upstream's bytes, mirrored `dtype: "source"`) beside a bf16
    transformer and a bf16 text encoder."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(json.dumps(H3_MODEL_INDEX), encoding="utf-8")
    g = torch.Generator().manual_seed(1266)
    _save(root / "vae" / "diffusion_pytorch_model.safetensors",
          {f"decoder.blocks.{i}.attn.to_q.weight": torch.randn(16, 16, generator=g)
           for i in range(4)})
    _save(root / "audio_vae" / "diffusion_pytorch_model.safetensors",
          {f"decoder.resblocks.{i}.conv.weight": torch.randn(16, 16, generator=g)
           for i in range(4)})
    _save(root / "transformer" / "diffusion_pytorch_model.safetensors",
          {f"blocks.{i}.attn.to_q.weight":
           torch.randn(16, 16, generator=g).to(torch.bfloat16) for i in range(4)})
    _save(root / "text_encoder" / "model.safetensors",
          {"embed_tokens.weight": torch.randn(16, 16, generator=g).to(torch.bfloat16)})
    for comp, cls in (("vae", "AutoencoderKLMiniMaxH3"),
                      ("audio_vae", "AutoencoderKLMiniMaxH3Audio")):
        (root / comp / "config.json").write_text(
            json.dumps({"_class_name": cls}), encoding="utf-8")
    (root / "scheduler").mkdir(exist_ok=True)
    (root / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    return root


def _source(root: Path) -> IngestedSource:
    return IngestedSource(
        provider="huggingface",
        source_ref="MiniMaxAI/MiniMax-H3",
        source_revision="bfc8ed0353f5a9733be73e6b2c98ec0948195b86",
        dir=root,
        layout="multi-file",
        attrs={"dtype": "fp32", "file_layout": "multi-file", "file_type": "safetensors"},
        metadata={},
        model_family="minimax",
        model_family_variant="h3",
    )


# ---------------------------------------------------------------------------
# The facts themselves
# ---------------------------------------------------------------------------

def test_both_h3_vae_classes_carry_an_fp32_fact_with_its_reason() -> None:
    """RED on the parent commit: `families.facts` knew only the Wan VAE, so
    both of these returned None and every gate downstream was a no-op."""
    for cls in ("AutoencoderKLMiniMaxH3", "AutoencoderKLMiniMaxH3Audio"):
        fact = component_dtype_for_class(cls)
        assert fact is not None, f"{cls} carries no load-dtype fact"
        assert fact.dtype == "fp32"
        assert fact.reason.strip()
    # The reasons must name the measurement, not restate the conclusion: the
    # audio row exists on upstream's ~20 dB, the video row on ie#621's arm.
    audio = component_dtype_for_class("AutoencoderKLMiniMaxH3Audio")
    video = component_dtype_for_class("AutoencoderKLMiniMaxH3")
    assert audio is not None and video is not None
    assert "20 dB" in audio.reason
    assert "74.97" in video.reason


def test_the_h3_tree_declares_both_pins_and_only_narrowing_is_exempt(tmp_path: Path) -> None:
    root = _h3_tree(tmp_path / "src")
    pins = component_pins(root)
    assert set(pins) == {"vae", "audio_vae"}
    assert {p.dtype for p in pins.values()} == {"fp32"}

    assert set(cast_exempt_components(root, "bf16")) == {"vae", "audio_vae"}
    assert set(cast_exempt_components(root, "fp8")) == {"vae", "audio_vae"}
    # Widening is a no-op, so nothing is exempt from it.
    assert cast_exempt_components(root, "fp32") == {}


# ---------------------------------------------------------------------------
# THE RED TEST — ie#718's proposed produce, refused where it is created
# ---------------------------------------------------------------------------

def test_a_bf16_cast_leaves_both_h3_vaes_byte_identical(tmp_path: Path) -> None:
    """ie#718's `cast-dtype dtypes=["bf16"]` over the H3 tree. On the parent
    commit this halved both VAEs and published a checkpoint whose audio
    decodes ~20 dB quiet — valid, classified, complete, and wrong."""
    src = _h3_tree(tmp_path / "src")
    before = {
        comp: (src / comp / "diffusion_pytorch_model.safetensors").read_bytes()
        for comp in ("vae", "audio_vae")
    }

    out, attrs = build_flavor_tree(
        _source(src),
        OutputSpec(dtype="bf16", file_layout="multi-file", file_type="safetensors"),
        tmp_path / "out",
    )

    for comp, raw in before.items():
        assert (out / comp / "diffusion_pytorch_model.safetensors").read_bytes() == raw
    dtypes = component_dtypes_on_disk(out)
    assert dtypes["vae"] == "fp32" and dtypes["audio_vae"] == "fp32", dtypes
    # …and the cast still does its job on everything that is not pinned.
    assert dtypes["transformer"] == "bf16"

    # It is REPORTED, not silent — this is what makes a −5.21 GB claim that
    # never materialised auditable rather than mysterious.
    assert attrs["dtype"] == "bf16"
    assert "vae:fp32" in attrs["dtype_pinned_components"]
    assert "audio_vae:fp32" in attrs["dtype_pinned_components"]


def test_naming_an_h3_vae_as_a_quant_target_is_refused_by_name(tmp_path: Path) -> None:
    """An explicit per-component instruction contradicting the pin is a
    refusal, not a skip: the caller can repair it."""
    src = _h3_tree(tmp_path / "src")
    with pytest.raises(ComponentDtypePinError) as excinfo:
        build_flavor_tree(
            _source(src),
            OutputSpec(dtype="fp8", file_layout="multi-file", file_type="safetensors"),
            tmp_path / "out",
            quantize_components=["transformer", "audio_vae"],
        )
    err = excinfo.value
    assert err.component == "audio_vae"
    assert err.class_name == "AutoencoderKLMiniMaxH3Audio"
    assert err.pin == "fp32" and err.requested == "fp8"
    assert "20 dB" in str(err)


def test_the_publish_gate_refuses_an_h3_tree_whose_vae_we_narrowed(tmp_path: Path) -> None:
    """The belt to the cast's braces: however a bf16 H3 VAE were produced —
    a hand-run leg, a future producer, a path this table does not gate — the
    publish refuses it."""
    src = _h3_tree(tmp_path / "src")
    bad = _h3_tree(tmp_path / "bad")
    g = torch.Generator().manual_seed(7)
    _save(bad / "vae" / "diffusion_pytorch_model.safetensors",
          {f"decoder.blocks.{i}.attn.to_q.weight":
           torch.randn(16, 16, generator=g).to(torch.bfloat16) for i in range(4)})

    with pytest.raises(ComponentDtypePinViolation) as excinfo:
        verify_produced_tree(bad, source_dir=src)
    err = excinfo.value
    assert (err.component, err.pin, err.produced, err.source) == (
        "vae", "fp32", "bf16", "fp32")


def test_the_pin_never_refuses_a_mirror_of_someone_elses_narrow_bytes(tmp_path: Path) -> None:
    """The fact is about the architecture, not a licence to refuse an
    upstream. H3 does not ship this way today — the assertion is that the
    guard could not block the mirror if it ever did."""
    src = _h3_tree(tmp_path / "src")
    g = torch.Generator().manual_seed(9)
    narrow = {f"decoder.blocks.{i}.attn.to_q.weight":
              torch.randn(16, 16, generator=g).to(torch.bfloat16) for i in range(4)}
    _save(src / "vae" / "diffusion_pytorch_model.safetensors", narrow)
    mirror = _h3_tree(tmp_path / "mirror")
    _save(mirror / "vae" / "diffusion_pytorch_model.safetensors", narrow)
    verify_produced_tree(mirror, source_dir=src)


def test_the_diet_leg_arithmetic_this_refusal_declines(tmp_path: Path) -> None:
    """State the number that was on the table, so the refusal is a TRADE and
    not a shrug: 5.21 GB of every cold boot, 94.5% of it the video VAE."""
    saving = (H3_VIDEO_VAE_BYTES + H3_AUDIO_VAE_BYTES) / 2
    assert round(saving / 1e9, 2) == 5.51
    assert round(H3_VIDEO_VAE_BYTES / 2 / 1e9, 2) == 5.21
    # The audio half is 5.5% of the leg and carries the ~20 dB regression.
    assert H3_AUDIO_VAE_BYTES / (H3_VIDEO_VAE_BYTES + H3_AUDIO_VAE_BYTES) < 0.06
