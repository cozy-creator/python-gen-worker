"""pgw#1020: pgw#683's mixed-compute-dtype guard was BLIND to every quantized
leaf, because its evidence collector selected leaves by
``isinstance(leaf, (nn.Linear, nn.Conv*))`` and all five quantized leaves
(``_Fp8ScaledLinear``, ``_W4A4Linear``, ``_SvdqLinear``, ``_SvdqFusedLinear``,
``_AwqPackedLinear``) subclass ``nn.Module`` directly.

Measured before the fix, cardless::

    _gemm_param_dtypes on a w8a8 fp16 denoiser: {}
    PASSED (blind)

i.e. a w8a8 denoiser built at fp16 and composed into a bf16 pipeline — the
exact cross-composition aliasing shape pgw#683 exists to refuse — passed
silently, and the fp16-vs-bf16 collision surfaced where it always did: inside
torch, mid-forward, naming neither the component nor the tensor.

The guard reads ``self.compute_dtype``, which every quantized leaf RECORDS —
without it a bias-free quantized leaf states its compute dtype nowhere.

Real classes throughout — the production module factories, not a stand-in.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from gen_worker.models.loading import (  # noqa: E402
    MixedComputeDtypeError,
    _gemm_param_dtypes,
    assert_uniform_compute_dtype,
)
from gen_worker.models.w4a4 import w4a4_linear_class  # noqa: E402
from gen_worker.models.w8a8 import fp8_scaled_linear_class  # noqa: E402


class _Pipe:
    """A composition in the diffusers shape (`.components` -> modules)."""

    def __init__(self, **parts: Any) -> None:
        self._parts = parts

    @property
    def components(self) -> dict:
        return dict(self._parts)


def _w8a8_denoiser(compute_dtype: Any, *, bias: bool = False) -> Any:
    """A denoiser whose projection is a REAL ``_Fp8ScaledLinear`` — the shape
    `load_w8a8_denoiser` builds, bias-free by default because that is the case
    that stated its compute dtype nowhere else."""
    import torch.nn as nn

    cls = fp8_scaled_linear_class()

    class _Denoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = cls(16, 16, bias=bias, compute_dtype=compute_dtype,
                            static_input_scale=False, gemm_mode="rowwise")
            self.norm = nn.LayerNorm(16)

    return _Denoiser()


def _w4a4_denoiser(compute_dtype: Any) -> Any:
    """The same shape on the nvfp4 lane — a second leaf class, so the fix is
    proven to be about the DECLARATION and not about one module."""
    import torch.nn as nn

    cls = w4a4_linear_class()

    class _Denoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = cls(32, 16, bias=False, compute_dtype=compute_dtype,
                            static_input_scale=False)

    return _Denoiser()


@pytest.mark.parametrize("build", [_w8a8_denoiser, _w4a4_denoiser],
                         ids=["w8a8", "w4a4"])
def test_the_guard_can_see_a_quantized_leaf_at_all(build: Any) -> None:
    """The blindness itself, at the collector: a bias-free quantized leaf used
    to contribute NOTHING, so the guard's evidence for the whole denoiser was
    ``{}`` and every verdict below it was vacuous."""
    dtypes = _gemm_param_dtypes(build(torch.float16))
    assert dtypes, "the quantized leaf is invisible to the pgw#683 collector"
    assert dtypes["proj.compute_dtype"] == "float16"


def test_fp16_quantized_denoiser_in_a_bf16_composition_refuses() -> None:
    """The measured case. The w8a8 lane's composition dtype is bf16
    (`composition_compute_dtype` returns the quant lane default), so an fp16
    denoiser here is the pgw#683 aliasing shape and must be refused at LOAD,
    naming the component."""
    pipe = _Pipe(unet=_w8a8_denoiser(torch.float16),
                 vae=torch.nn.Linear(16, 16).to(torch.bfloat16))
    with pytest.raises(MixedComputeDtypeError) as excinfo:
        assert_uniform_compute_dtype(pipe, "bf16", label="slot 'pipeline'")
    msg = str(excinfo.value)
    assert "unet" in msg, f"the component must be named: {msg}"
    assert "proj.compute_dtype=float16" in msg, msg
    assert "slot 'pipeline'" in msg


def test_an_internally_mixed_quantized_denoiser_refuses() -> None:
    """Verdict 1 needs no ``expected``: one component holding a bf16 leaf and
    an fp16 leaf cannot survive its own forward, quantized or not."""
    import torch.nn as nn

    cls = fp8_scaled_linear_class()

    class _Mixed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = cls(16, 16, bias=False, compute_dtype=torch.bfloat16,
                         static_input_scale=False, gemm_mode="rowwise")
            self.b = cls(16, 16, bias=False, compute_dtype=torch.float16,
                         static_input_scale=False, gemm_mode="rowwise")

    with pytest.raises(MixedComputeDtypeError, match="internally mixed"):
        assert_uniform_compute_dtype(_Pipe(unet=_Mixed()), "")


def test_the_storage_carve_out_survives() -> None:
    """pgw#683's whole point is that fp8/fp4 STORAGE is legal. The w8a8 leaf's
    weight is `float8_e4m3fn` and the w4a4 leaf's is packed `uint8`; neither
    may be counted as a compute dtype, or the guard refuses the lanes it was
    written to admit."""
    for build in (_w8a8_denoiser, _w4a4_denoiser):
        dtypes = _gemm_param_dtypes(build(torch.bfloat16))
        assert all(v in ("float16", "bfloat16", "float32", "float64")
                   for v in dtypes.values()), dtypes
        assert not any(k.endswith(".weight") for k in dtypes), dtypes


def test_a_uniform_quantized_composition_is_admitted() -> None:
    """No new refusal on the normal path: a bf16 quantized denoiser inside the
    bf16 composition the loader actually builds (`compute` and the guard's
    `expected` come from ONE declared dtype) passes, with and without bias."""
    for bias in (False, True):
        assert_uniform_compute_dtype(
            _Pipe(unet=_w8a8_denoiser(torch.bfloat16, bias=bias),
                  vae=torch.nn.Linear(16, 16).to(torch.bfloat16)),
            "bf16")
    assert_uniform_compute_dtype(_Pipe(unet=_w4a4_denoiser(torch.bfloat16)),
                                 "bf16")
    # pgw#667's declared widening is still not a collision.
    assert_uniform_compute_dtype(
        _Pipe(unet=_w8a8_denoiser(torch.bfloat16),
              vae=torch.nn.Linear(16, 16).to(torch.float32)), "bf16")


def test_an_embedding_keeps_its_exclusion() -> None:
    """The fp8-storage lane stamps `compute_dtype` on EVERY covered leaf,
    embeddings included (`restructure_fp8_storage`). Embeddings are excluded
    from this guard by kind — they carry their own legitimately wider
    precision — and reading a declaration must not smuggle them back in."""
    import torch.nn as nn

    emb = nn.Embedding(8, 16).to(torch.float16)
    emb.compute_dtype = torch.float16
    assert _gemm_param_dtypes(emb) == {}
    assert_uniform_compute_dtype(_Pipe(text_encoder=emb), "bf16")
