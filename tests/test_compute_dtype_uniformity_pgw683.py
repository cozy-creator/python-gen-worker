"""Guards compute-dtype uniformity across a composition: a lane-aware dtype
default is not sufficient on its own, because a composition can be handed a
foreign-precision component with NO component override anywhere on the wire.

Torch's three dtype-mismatch messages are distinct, and name the operator
(measured on torch 2.13)::

    nn.Linear WITH bias  (addmm) -> mat1 and mat2 must have the same dtype,
                                    but got BFloat16 and Half
    bare matmul / no bias        -> expected m1 and m2 to have the same dtype
    conv                         -> Input type (c10::BFloat16) and bias type
                                    (c10::Half) should be the same

The mechanism under test is IDENTITY, not the dtype decision. The content-keyed
shared-component cache keys on `LoadedComponentKey`, whose `dtype` field is the
binding's DECLARED dtype; hub bindings declare none. A quantizer only rewrites
the DENOISER, so the VAE and text encoders of `X` and its quantized mirror are
byte-identical — same content digest, same empty declared dtype, therefore ONE
cache entry — while the two compositions compute at DIFFERENT dtypes. Whichever
pick loads first wins the entry and the other aliases a foreign-precision module
into its composition: non-deterministic by ARRIVAL ORDER, needing no override,
and landing in a LATER window because injection only happens once an entry
already exists.

Two properties are taped here: identity carries the EFFECTIVE compute dtype, and
a hard post-load invariant refuses a mixed composition at LOAD, naming the
component and the tensor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from safetensors.torch import save_file  # noqa: E402

from gen_worker.convert.writer import streaming_w8a8_cast  # noqa: E402
from gen_worker.models.loading import (  # noqa: E402
    MixedComputeDtypeError,
    assert_uniform_compute_dtype,
    composition_compute_dtype,
    detect_on_disk_dtype,
)
from gen_worker.models.residency import LoadedComponentKey  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: the two real trees that share byte-identical components.
# ---------------------------------------------------------------------------


def _quantizable(dtype: Any) -> dict:
    tensors = {
        # Eligible for the producer: 2-D .weight, 16-aligned, inside a `.N.`
        # block container.
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight":
            torch.randn(16, 16, dtype=dtype),
    }
    for i in range(6):
        tensors[f"passthrough.norm_{i}.weight"] = torch.randn(8, dtype=dtype)
    return tensors


def _plain_fp16_tree(tmp_path: Path) -> Path:
    """A plain fp16-stored mirror — what an unquantized SDXL fine-tune is."""
    root = tmp_path / "plain-fp16"
    (root / "unet").mkdir(parents=True)
    save_file(_quantizable(torch.float16), str(root / "unet" / "model.safetensors"))
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "StableDiffusionXLPipeline",
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }))
    return root


def _w8a8_tree(tmp_path: Path) -> Path:
    """The `#fp8-w8a8` flavor of the SAME model, built by the REAL producer."""
    src = tmp_path / "src-unet"
    src.mkdir()
    save_file(_quantizable(torch.float16), str(src / "model.safetensors"))
    root = tmp_path / "flavor-w8a8"
    out = streaming_w8a8_cast(src / "model.safetensors", root / "unet")
    assert int(out["converted_count"]) == 1
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "StableDiffusionXLPipeline",
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }))
    return root


#: A hub-resolved binding: NO declared dtype (the production shape). pgw#1148
#: deleted the flavor axis, so the two "lanes" this file contrasts are two
#: distinct REPOS/checkpoints, which is what §1.32(d) makes them.
def _binding(path: str) -> Any:
    from gen_worker.api.binding import ModelRef

    return ModelRef(source="tensorhub", path=path, tag="prod")


# ---------------------------------------------------------------------------
# 1. The identity hole: two lanes, one shared entry.
# ---------------------------------------------------------------------------


def test_the_two_flavors_really_do_compute_at_different_dtypes(
    tmp_path: Path,
) -> None:
    """Pin the premise: the same upstream bytes, mirrored two ways, answer two
    DIFFERENT compute dtypes — and neither binding declares one."""
    plain, w8a8 = _plain_fp16_tree(tmp_path), _w8a8_tree(tmp_path)
    assert detect_on_disk_dtype(plain) == "fp16"
    assert composition_compute_dtype(plain, "") == "fp16"
    # pgw#675's lane rule: a quantized-artifact tree computes bf16 whatever its
    # majority on-disk dtype says.
    assert detect_on_disk_dtype(w8a8) == "fp16"
    assert composition_compute_dtype(w8a8, "") == "bf16"


def test_effective_dtype_identity_separates_the_execution_lanes(tmp_path: Path) -> None:
    """GREEN: keyed on the EFFECTIVE compute dtype — what the executor passes
    now — the same bytes under two lanes are two entries, so a bf16
    composition can never be handed the fp16 lane's module."""
    plain, w8a8 = _plain_fp16_tree(tmp_path), _w8a8_tree(tmp_path)
    shared_digest = "same-text-encoder-bytes"
    keys = [
        LoadedComponentKey.for_component(
            content_digest=shared_digest, component="text_encoder",
            binding=_binding("tensorhub/wai-illustrious"),
            dtype=composition_compute_dtype(tree, ""),
        )
        for tree in (plain, w8a8)
    ]
    assert keys[0].dtype == "fp16" and keys[1].dtype == "bf16"
    assert keys[0].cache_id() != keys[1].cache_id()
    # Same lane, same bytes still aliases — sharing is not disabled, only made
    # precision-honest.
    again = LoadedComponentKey.for_component(
        content_digest=shared_digest, component="text_encoder",
        binding=_binding("tensorhub/wai-illustrious"), dtype=composition_compute_dtype(w8a8, ""),
    )
    assert again.cache_id() == keys[1].cache_id()


def test_the_share_plan_itself_keys_on_the_effective_dtype(
    tmp_path: Path,
) -> None:
    """The FIX SITE, through the real `Executor._component_share_plan`: two
    pipeline slots on the two flavors, byte-identical `text_encoder`, and the
    plan must mint two different cache ids. RED pre-fix — one id, and the
    second lane injects the first lane's foreign-precision module."""
    from gen_worker.executor import Executor
    from gen_worker.models import residency as residency_mod

    plain, w8a8 = _plain_fp16_tree(tmp_path), _w8a8_tree(tmp_path)

    class _Pipeline:
        @classmethod
        def from_pretrained(cls, path: str) -> "_Pipeline":  # pragma: no cover
            return cls()

    class _Store:
        def __init__(self) -> None:
            self.residency = residency_mod.Residency()

        def component_digests(self, ref: str, local_path: Path) -> dict:
            # Byte-identical non-denoiser components across flavors; the
            # denoiser's bytes differ (the quantizer rewrote it).
            return {"text_encoder": "same-te-bytes",
                    "unet": f"unet-{Path(local_path).name}"}

    class _Exec:
        _component_share_plan = Executor._component_share_plan

        def __init__(self) -> None:
            self.store = _Store()
            self._no_share_refs: set = set()

    class _Spec:
        name = "sdxl"
        slots = {"plain": object(), "quant": object()}
        models = {"plain": _binding("tensorhub/cyberrealistic-xl"),
                  "quant": _binding("tensorhub/wai-illustrious")}

    ex = _Exec()
    paths = {"plain": str(plain), "quant": str(w8a8)}
    hints = {"plain": _Pipeline, "quant": _Pipeline}
    plan = ex._component_share_plan(_Spec(), paths, hints)  # type: ignore[arg-type]
    assert plan is not None
    te_ids = {
        slot: plan[slot]["text_encoder"].cache_id()
        for slot in ("plain", "quant") if "text_encoder" in plan[slot]
    }
    assert set(te_ids) == {"plain", "quant"}
    assert te_ids["plain"] != te_ids["quant"], (
        "byte-identical components of an fp16 tree and a bf16 quant lane must "
        "NOT share one entry — that alias is the pgw#683 fatal"
    )


# ---------------------------------------------------------------------------
# 2. The invariant: refuse at LOAD, naming the tensor.
# ---------------------------------------------------------------------------


class _Pipe:
    """A composition in the diffusers shape (`.components` -> modules)."""

    def __init__(self, **parts: Any) -> None:
        self._parts = parts

    @property
    def components(self) -> dict:
        return dict(self._parts)


def _linear_stack(dtype: Any, *, bias: bool = True) -> Any:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(16, 16, bias=bias),
        torch.nn.LayerNorm(16),
        torch.nn.Linear(16, 16, bias=bias),
    ).to(dtype)


def test_invariant_refuses_a_foreign_precision_component() -> None:
    """The aliasing shape: a Half `text_encoder` inside a bf16 composition is
    refused AT LOAD, naming the component and the offending GEMM input —
    instead of torch's message, which names neither, arriving at warm unit
    4/18 and costing the function."""
    pipe = _Pipe(
        unet=_linear_stack(torch.bfloat16),
        text_encoder=_linear_stack(torch.float16),
    )
    with pytest.raises(MixedComputeDtypeError) as excinfo:
        assert_uniform_compute_dtype(pipe, "bf16", label="slot 'pipeline'")
    msg = str(excinfo.value)
    assert "text_encoder" in msg
    assert "float16" in msg and "bfloat16" in msg
    assert "0.weight=float16" in msg, f"the tensor must be named: {msg}"
    assert "slot 'pipeline'" in msg


def test_invariant_refuses_an_internally_mixed_component() -> None:
    """The other fatal verdict, and it needs no `expected`: one component
    carrying both fp16 and bf16 GEMM inputs cannot survive its own forward."""
    mixed = torch.nn.Sequential(
        torch.nn.Linear(16, 16).to(torch.bfloat16),
        torch.nn.Linear(16, 16).to(torch.float16),
    )
    with pytest.raises(MixedComputeDtypeError, match="internally mixed"):
        assert_uniform_compute_dtype(_Pipe(unet=mixed), "")


def test_invariant_admits_every_legal_composition() -> None:
    """It must not refuse what the platform does on purpose. A uniform
    composition, a pgw#667 fp32-WIDENED component, an fp8-storage lane (weights
    at rest in fp8, upcast per forward), and a bare module all pass."""
    assert_uniform_compute_dtype(
        _Pipe(unet=_linear_stack(torch.bfloat16),
              text_encoder=_linear_stack(torch.bfloat16)), "bf16")
    # The wider part is a DECLARED decision, never a collision.
    assert_uniform_compute_dtype(
        _Pipe(unet=_linear_stack(torch.bfloat16),
              vae=_linear_stack(torch.float32)), "bf16")
    fp8_execution_lane = _linear_stack(torch.bfloat16)
    for leaf in fp8_execution_lane:
        if isinstance(leaf, torch.nn.Linear):
            leaf.weight.data = leaf.weight.data.to(torch.float8_e4m3fn)
    assert_uniform_compute_dtype(_Pipe(unet=fp8_execution_lane), "bf16")
    assert_uniform_compute_dtype(_linear_stack(torch.bfloat16), "bf16")
    # No compute-dtype opinion (an fp32-defaulting composition) never refuses
    # on the second verdict.
    assert_uniform_compute_dtype(
        _Pipe(unet=_linear_stack(torch.float16)), "")


def test_invariant_never_fails_a_load_on_introspection() -> None:
    """Fail-loud on a proven collision, never on a shape it cannot read."""

    class _Hostile:
        @property
        def components(self) -> dict:
            raise RuntimeError("no")

    assert_uniform_compute_dtype(_Hostile(), "bf16")
    assert_uniform_compute_dtype(object(), "bf16")
