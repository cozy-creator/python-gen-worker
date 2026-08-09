"""The pgw#1062 ESCAPE-HATCH variant: author-defined ops through the mint.

pgw#1059 amendment 7 names one author-hint tier to verify PROACTIVELY —
custom ops (``torch.library``) and hand-written Triton kernels preserved
end-to-end through trace → package → publish → adopt — because its absence is
silent until a launch depends on it. This module carries all three author
surfaces the SDK sanctions, each the smallest thing that still exercises its
whole path:

* ``micro_escape::rms_gate`` — a ``torch.library.custom_op`` WITH
  ``register_fake``. The fake kernel is what makes it traceable at all
  (export traces on FakeTensors; pgw#1056's fake-weight mint doubles down on
  that), and AOTI serves it as an opaque fallback call into the registered
  impl. Remove the ``register_fake`` and the mint child refuses at export —
  that is this variant's RED proof.
* ``micro_escape::silu_scale`` — a hand-written Triton kernel through
  ``torch.library.triton_op`` + ``wrap_triton``, the sanctioned authoring
  surface: the kernel is visible to inductor, compiled and baked into the
  cell like any generated kernel.
* a RAW ``@triton.jit`` call in the forward — the unsanctioned-but-real form
  authors actually write. Dynamo captures it as a triton HOP.

GPU-ONLY by construction, like w8a8: inductor lowers a Triton implementation
on every backend that declares one, and CPU declares none — measured on
2.13.0: ``RuntimeError: 0 compatible backends for target (cpu)`` even with a
``register_kernel("cpu")`` fallback present. A cardless run of this family
would not be a weaker test, it would be a different one.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from .model import MicroConfig, MicroDenoiser


# ---------------------------------------------------------------------------
# Surface 1: custom op with a fake kernel
# ---------------------------------------------------------------------------


@torch.library.custom_op("micro_escape::rms_gate", mutates_args=())
def rms_gate(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
    return x * rms * weight


@rms_gate.register_fake
def _rms_gate_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


# ---------------------------------------------------------------------------
# Surface 2: hand-written Triton kernel via triton_op (the sanctioned form)
# ---------------------------------------------------------------------------


@triton.jit
def _silu_scale_kernel(x_ptr, out_ptr, n_elem, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * tl.sigmoid(x) * scale, mask=mask)


@triton_op("micro_escape::silu_scale", mutates_args={})
def silu_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    wrap_triton(_silu_scale_kernel)[(triton.cdiv(n, 1024),)](
        x, out, n, scale, BLOCK=1024)
    return out


# ---------------------------------------------------------------------------
# Surface 3: a raw @triton.jit call in the forward (the form authors write)
# ---------------------------------------------------------------------------


@triton.jit
def _affine_kernel(x_ptr, out_ptr, n_elem, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(x_ptr + offs, mask=mask)
    # Dyadic constants: exact in fp32, so eager-vs-served parity is clean.
    tl.store(out_ptr + offs, x * 1.0625 + 0.03125, mask=mask)


def _affine(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    _affine_kernel[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024)
    return out


class MicroEscapeDenoiser(MicroDenoiser):
    """Same weights and blocks as :class:`MicroDenoiser`; the escape-hatch
    ops wrap its output, so every one of them is INSIDE the traced graph.

    ``escape_gain`` is a non-persistent buffer (the ``freqs`` pattern,
    pgw#857): derived from config, absent from the checkpoint, present in the
    module — so the standard generated tree loads strictly unchanged.
    """

    def __init__(self, config: MicroConfig) -> None:
        super().__init__(config)
        self.register_buffer(
            "escape_gain",
            torch.linspace(0.5, 1.5, config.in_channels),
            persistent=False)

    def forward(self, x, t, cond):  # type: ignore[override]
        h = super().forward(x, t, cond)
        h = torch.ops.micro_escape.rms_gate(h, self.escape_gain)
        h = torch.ops.micro_escape.silu_scale(h, 1.25)
        return _affine(h)


__all__ = ["MicroEscapeDenoiser", "rms_gate", "silu_scale"]
