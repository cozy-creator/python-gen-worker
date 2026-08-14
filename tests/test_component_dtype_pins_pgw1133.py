"""A tree-wide dtype cast must not narrow an fp32-pinned component.

A plain ``dtype: "bf16"`` clone halves ``vae/diffusion_pytorch_model.safetensors``
(507,591,892 B fp32 -> 253,806,966 B) even though ``families.facts`` pins
``AutoencoderKLWan -> fp32`` and ``models.loading`` honours that on every
materialize. The resulting checkpoint is valid, classified, complete and served
by every gate; the truncation is visible only as a byte count nobody reads.

Real codepaths: the real `build_flavor_tree` cast over real safetensors files,
the real `verify_produced_tree` publish gate, the real `families.facts` table.
The VAE tensors are shaped so the fp32 file is exactly the live 507,591,892 :
253,806,966 ratio in miniature — 4 bytes per parameter vs 2.

Run: pytest tests/test_component_dtype_pins_pgw1133.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

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

# The live wan-2.2 A14B numbers this issue was filed on.
WAN_VAE_FP32_BYTES = 507_591_892
WAN_VAE_BF16_BYTES = 253_806_966
WAN_VAE_PARAMS = 126_892_531  # 194 tensors, read off the published headers

WAN_MODEL_INDEX = {
    "_class_name": "WanPipeline",
    "_diffusers_version": "0.35.0.dev0",
    "boundary_ratio": 0.875,
    "scheduler": ["diffusers", "UniPCMultistepScheduler"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "T5TokenizerFast"],
    "transformer": ["diffusers", "WanTransformer3DModel"],
    "transformer_2": ["diffusers", "WanTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
}


def _save(path: Path, tensors: dict[str, "torch.Tensor"]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(tensors, str(path))


def _wan_tree(root: Path) -> Path:
    """An fp32-VAE wan-2.2 tree in miniature: the pinned component fp32, the
    two experts fp32 (what upstream ships), plus a bf16 text encoder — the
    exact mixed-precision shape the live A14B mirror has."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(json.dumps(WAN_MODEL_INDEX), encoding="utf-8")
    g = torch.Generator().manual_seed(1133)
    _save(root / "vae" / "diffusion_pytorch_model.safetensors",
          {f"decoder.block.{i}.weight": torch.randn(16, 16, generator=g)
           for i in range(4)})
    for comp in ("transformer", "transformer_2"):
        _save(root / comp / "diffusion_pytorch_model.safetensors",
              {f"blocks.{i}.attn.to_q.weight": torch.randn(16, 16, generator=g)
               for i in range(4)})
    _save(root / "text_encoder" / "model.safetensors",
          {"shared.weight": torch.randn(16, 16, generator=g).to(torch.bfloat16)})
    (root / "vae" / "config.json").write_text(
        json.dumps({"_class_name": "AutoencoderKLWan"}), encoding="utf-8")
    (root / "scheduler").mkdir(exist_ok=True)
    (root / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    return root


def _source(root: Path) -> IngestedSource:
    return IngestedSource(
        provider="huggingface",
        source_ref="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        source_revision="5be7df9619b54f4e2667b2755bc6a756675b5cd7",
        dir=str(root),
        layout="multi-file",
        attrs={"dtype": "fp32", "file_layout": "multi-file", "file_type": "safetensors"},
        metadata={},
        model_family="wan",
        model_family_variant="wan22",
    )


# ---------------------------------------------------------------------------
# The fact, read from the tree the producer is about to convert
# ---------------------------------------------------------------------------

def test_the_tree_declares_the_pin_and_only_narrowing_is_exempt(tmp_path):
    root = _wan_tree(tmp_path / "src")
    pins = component_pins(root)
    assert set(pins) == {"vae"} and pins["vae"].dtype == "fp32"

    # A narrowing cast exempts it; a widening / equal one is a no-op and does not.
    assert set(cast_exempt_components(root, "bf16")) == {"vae"}
    assert set(cast_exempt_components(root, "fp8")) == {"vae"}
    assert cast_exempt_components(root, "fp32") == {}


# ---------------------------------------------------------------------------
# THE RED TEST — revert `_cast_tree`'s pin consultation and this fails
# ---------------------------------------------------------------------------

def test_bf16_cast_keeps_the_pinned_vae_byte_identical_and_casts_the_experts(tmp_path):
    src = _wan_tree(tmp_path / "src")
    before = (src / "vae" / "diffusion_pytorch_model.safetensors").read_bytes()

    out, attrs = build_flavor_tree(
        _source(src),
        OutputSpec(dtype="bf16", file_layout="multi-file", file_type="safetensors"),
        tmp_path / "out",
    )

    vae_out = out / "vae" / "diffusion_pytorch_model.safetensors"
    # The live assertion, in miniature: the pinned component is passed through
    # byte-for-byte, at 4 bytes/param — never the 2 the flavor label implies.
    assert vae_out.read_bytes() == before
    dtypes = component_dtypes_on_disk(out)
    assert dtypes["vae"] == "fp32", dtypes
    assert dtypes["transformer"] == "bf16" and dtypes["transformer_2"] == "bf16"

    # The live bytes, stated exactly: the same 126,892,531 parameters at 4
    # bytes each (what the pin requires) or at 2 (what the cast produced).
    assert WAN_VAE_FP32_BYTES == WAN_VAE_PARAMS * 4 + 21_768
    assert WAN_VAE_BF16_BYTES == WAN_VAE_PARAMS * 2 + 21_904

    # And it is REPORTED, not silent.
    assert attrs["dtype"] == "bf16"
    assert attrs["dtype_pinned_components"] == "vae:fp32"


def test_an_explicit_request_to_quantize_the_pinned_component_is_refused_by_name(tmp_path):
    src = _wan_tree(tmp_path / "src")
    with pytest.raises(ComponentDtypePinError) as excinfo:
        build_flavor_tree(
            _source(src),
            OutputSpec(dtype="fp8", file_layout="multi-file", file_type="safetensors"),
            tmp_path / "out",
            quantize_components=["transformer", "vae"],
        )
    err = excinfo.value
    assert err.component == "vae" and err.class_name == "AutoencoderKLWan"
    assert err.pin == "fp32" and err.requested == "fp8"
    assert "AutoencoderKLWan" in str(err) and "numerically fragile" in str(err)


def test_an_fp32_cast_of_a_pinned_component_is_not_refused(tmp_path):
    """Widening is free — the skip only exists to stop truncation."""
    src = _wan_tree(tmp_path / "src")
    out, _ = build_flavor_tree(
        _source(src),
        OutputSpec(dtype="fp32", file_layout="multi-file", file_type="safetensors"),
        tmp_path / "out",
    )
    assert component_dtypes_on_disk(out)["vae"] == "fp32"


# ---------------------------------------------------------------------------
# The publish gate — the belt to the cast's braces
# ---------------------------------------------------------------------------

def test_the_publish_gate_refuses_a_tree_whose_pinned_component_we_narrowed(tmp_path):
    src = _wan_tree(tmp_path / "src")
    bad = _wan_tree(tmp_path / "bad")
    # Exactly what leg A published: a bf16 VAE where the source had fp32.
    g = torch.Generator().manual_seed(7)
    _save(bad / "vae" / "diffusion_pytorch_model.safetensors",
          {f"decoder.block.{i}.weight": torch.randn(16, 16, generator=g).to(torch.bfloat16)
           for i in range(4)})

    with pytest.raises(ComponentDtypePinViolation) as excinfo:
        verify_produced_tree(bad, source_dir=src)
    err = excinfo.value
    assert (err.component, err.pin, err.produced, err.source) == ("vae", "fp32", "bf16", "fp32")


def test_the_gate_still_mirrors_an_upstream_that_ships_the_component_narrow(tmp_path):
    """The pin is a fact about the architecture, not a licence to refuse
    someone else's bytes: a source ALREADY at bf16 is mirrorable, and the
    load side widens it on materialize as it always did."""
    src = _wan_tree(tmp_path / "src")
    g = torch.Generator().manual_seed(9)
    narrow = {f"decoder.block.{i}.weight": torch.randn(16, 16, generator=g).to(torch.bfloat16)
              for i in range(4)}
    _save(src / "vae" / "diffusion_pytorch_model.safetensors", narrow)
    mirror = _wan_tree(tmp_path / "mirror")
    _save(mirror / "vae" / "diffusion_pytorch_model.safetensors", narrow)

    dtypes = verify_produced_tree(mirror, source_dir=src)
    assert dtypes["vae"] == "bf16"


def test_a_tree_with_no_model_index_has_no_pins(tmp_path):
    """Single-file / transformers layouts carry no component class vocabulary,
    so the gate is a no-op there rather than a guess."""
    root = tmp_path / "bare"
    _save(root / "model.safetensors", {"w": torch.randn(8, 8).to(torch.bfloat16)})
    assert component_pins(root) == {}
    assert verify_produced_tree(root) == {"model": "bf16"}
