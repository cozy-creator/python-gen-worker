"""te#150 svdq load path: decode-on-device plumbing.

The 13 GB qwen fp4_r128 artifact decoded in 223 s on CPU (5090 bench,
2026-08-02) because every fragment unpack/repack ran single-threaded on host;
nunchaku loads the same bytes in ~8 s because its kernels consume the on-disk
layout verbatim. Ours cannot (torch._scaled_mm wants blocked scales + plain
packed nibbles), so the transforms now run ON the target device.

This proves the plumbing changes NOTHING about the decoded bytes: the cpu
path is bit-identical to a direct decode_linear/decode_awq_linear (the
historical loader body). The ``_Art``/``_write_multiunit`` builders are shared
with the pgw#1330 projected-tree suites.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from gen_worker.models import svdq_native as native          # noqa: E402
from gen_worker.models.nvfp4_quant import BLOCK, cast_e2m1   # noqa: E402
from gen_worker.models.svdq_awq import (                     # noqa: E402
    decode_awq_linear,
    encode_awq_linear,
)
from gen_worker.models.svdq_layout import (                  # noqa: E402
    convert_linear,
    decode_linear,
    pack_lowrank,
    pack_qweight,
    pack_vector,
    pack_wscales,
    split_decoded,
    to_buffers,
)

E2M1_MAX, FP8_MAX = 6.0, 448.0


def _tiny_qwen(dim_heads: int = 4, head_dim: int = 32, layers: int = 1):
    diffusers = pytest.importorskip("diffusers")
    return diffusers.QwenImageTransformer2DModel(
        patch_size=2, in_channels=16, out_channels=16, num_layers=layers,
        attention_head_dim=head_dim, num_attention_heads=dim_heads,
        joint_attention_dim=128, axes_dims_rope=(8, 12, 12))


def _w4a4_entry(out_f: int, in_f: int, *, rank: int, seed: int,
                second_key: str = "wcscales", smooth: bool = True) -> dict:
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(out_f, in_f, generator=gen)
    if second_key == "wcscales":
        second = (w.abs().amax(dim=1) / (E2M1_MAX * FP8_MAX)).clamp(min=1e-8)
        second_bcast = second.reshape(out_f, 1)
    else:
        second = (w.abs().amax() / (E2M1_MAX * FP8_MAX)).clamp(min=1e-8)
        second_bcast = second.reshape(1, 1)
    blocks = w.reshape(out_f, in_f // BLOCK, BLOCK)
    bs = (blocks.abs().amax(dim=-1) / (E2M1_MAX * second_bcast)).clamp(
        min=2.0 ** -9, max=FP8_MAX).to(torch.float8_e4m3fn)
    codes = cast_e2m1(
        (blocks / (bs.float().unsqueeze(-1) * second_bcast.unsqueeze(-1))
         ).reshape(out_f, in_f))
    entry: dict[str, Any] = {
        "qweight": pack_qweight(codes),
        "wscales": pack_wscales(bs, out_f, in_f),
    }
    entry[second_key] = (pack_vector(second.to(torch.bfloat16), out_f)
                         if second_key == "wcscales"
                         else second.to(torch.bfloat16))
    down = (torch.randn(in_f, rank, generator=gen) * in_f ** -0.5
            ).to(torch.bfloat16)
    up = (torch.randn(out_f, rank, generator=gen) * rank ** -0.5
          ).to(torch.bfloat16)
    entry["proj_down"] = pack_lowrank(down.t().contiguous(), down=True)
    entry["proj_up"] = pack_lowrank(up, down=False)
    if smooth:
        entry["smooth_factor"] = pack_vector(
            (torch.rand(in_f, generator=gen) + 0.5).to(torch.bfloat16), in_f)
    entry["bias"] = pack_vector(
        torch.randn(out_f, generator=gen).to(torch.bfloat16), out_f)
    return entry


class _Art:
    component = "transformer"

    def __init__(self, file) -> None:
        self.file = file


def _write_multiunit(tmp_path, *, dim_heads: int = 4, head_dim: int = 32,
                     layers: int = 1, rank: int = 128, name: str = "multi"):
    """A tiny-qwen single file with THREE unit kinds: a fused W4A4 to_qkv,
    a plain W4A4 to_out.0, and an AWQ img_mod.1 — everything else verbatim."""
    from safetensors.torch import save_file

    model = _tiny_qwen(dim_heads, head_dim, layers)
    dim = dim_heads * head_dim
    state: dict[str, Any] = {}
    replaced_prefixes: list[str] = []
    for key, val in model.state_dict().items():
        skip = False
        for blk in range(layers):
            p = f"transformer_blocks.{blk}"
            if (key.startswith((f"{p}.attn.to_q.", f"{p}.attn.to_k.",
                                f"{p}.attn.to_v.", f"{p}.attn.to_out.0.",
                                f"{p}.img_mod.1."))):
                skip = True
        if not skip:
            state[key] = (val.to(torch.bfloat16).contiguous()
                          if val.is_floating_point() else val.contiguous())
    for blk in range(layers):
        p = f"transformer_blocks.{blk}"
        for leaf, t in _w4a4_entry(3 * dim, dim, rank=rank,
                                   seed=10 + blk).items():
            state[f"{p}.attn.to_qkv.{leaf}"] = t.contiguous()
        for leaf, t in _w4a4_entry(dim, dim, rank=rank, seed=40 + blk,
                                   second_key="wtscale",
                                   smooth=False).items():
            state[f"{p}.attn.to_out.0.{leaf}"] = t.contiguous()
        gen = torch.Generator().manual_seed(70 + blk)
        mod_w = torch.randn(6 * dim, dim, generator=gen) * 0.02
        mod_b = torch.randn(6 * dim, generator=gen) * 0.02
        for leaf, t in encode_awq_linear(mod_w, mod_b,
                                         adanorm_splits=6).items():
            state[f"{p}.img_mod.1.{leaf}"] = t.contiguous()
        replaced_prefixes.append(p)
    cfg = {k: v for k, v in dict(model.config).items()
           if not k.startswith("_")}
    path = tmp_path / f"tiny-qwen-{name}.safetensors"
    save_file(state, str(path),
              metadata={"model_class": "QwenImageTransformer2DModel",
                        "config": json.dumps(cfg)})
    return path, state, dim


def _buffer_map(model) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, mod in model.named_modules():
        if getattr(mod, "_cozy_svdq_linear", False):
            for leaf in ("weight", "weight_scale", "weight_scale_2",
                         "proj_down", "proj_up", "smooth_factor", "bias"):
                t = getattr(mod, leaf, None)
                if t is not None:
                    out[f"{name}.{leaf}"] = t
    return out


def _bit_equal(a: Any, b: Any) -> bool:
    a, b = a.detach().cpu(), b.detach().cpu()
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(torch.equal(a.view(torch.uint8) if a.dtype.is_floating_point
                            else a, b.view(torch.uint8)
                            if b.dtype.is_floating_point else b))


def test_cpu_load_bytes_match_direct_decode(tmp_path) -> None:
    """The loader's device plumbing changes nothing: every SvdqLinear buffer
    equals a direct convert_linear of the same entry, and the AWQ modulation
    Linear equals a direct decode_awq_linear — bit for bit."""
    path, state, dim = _write_multiunit(tmp_path)
    model = native.load_svdq_native_denoiser(
        _Art(path), mode="blockwise", device="cpu")
    p = "transformer_blocks.0"

    fused = {k.rsplit(".", 1)[1]: v for k, v in state.items()
             if k.startswith(f"{p}.attn.to_qkv.")}
    dec = decode_linear(fused, 3 * dim, dim)
    parts = split_decoded(dec, (dim, dim, dim))
    for part, leafname in zip(parts, ("to_q", "to_k", "to_v")):
        want = to_buffers(part)
        mod = model.get_submodule(f"{p}.attn.{leafname}")
        assert _bit_equal(mod.weight, want.weight)
        assert _bit_equal(mod.weight_scale, want.weight_scale)
        assert _bit_equal(mod.weight_scale_2, want.weight_scale_2.reshape(
            mod.weight_scale_2.shape))
        assert _bit_equal(mod.proj_down, want.proj_down.to(torch.bfloat16))
        assert _bit_equal(mod.proj_up, want.proj_up.to(torch.bfloat16))
        assert _bit_equal(mod.smooth_factor,
                          want.smooth_factor.to(torch.bfloat16).reshape(-1))
        assert _bit_equal(mod.bias, want.bias.to(torch.bfloat16))

    plainu = {k.rsplit(".", 1)[1]: v for k, v in state.items()
              if k.startswith(f"{p}.attn.to_out.0.")}
    want = convert_linear(plainu, dim, dim)
    mod = model.get_submodule(f"{p}.attn.to_out.0")
    assert _bit_equal(mod.weight, want.weight)
    assert _bit_equal(mod.weight_scale, want.weight_scale)

    awq = {k.rsplit(".", 1)[1]: v for k, v in state.items()
           if k.startswith(f"{p}.img_mod.1.")}
    want_lin = decode_awq_linear(awq, 6 * dim, dim, adanorm_splits=6)
    got_lin = model.get_submodule(f"{p}.img_mod.1")
    assert _bit_equal(got_lin.weight, want_lin.weight)
    assert _bit_equal(got_lin.bias, want_lin.bias)

