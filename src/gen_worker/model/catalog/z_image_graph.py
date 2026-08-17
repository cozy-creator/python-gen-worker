"""The two upstream rewrites Z-Image's graph classes cannot be traced without.

**MINT-SIDE.** This module imports diffusers at module scope and the serve role
must never reach it; it is imported only by :mod:`gen_worker.model.catalog.
z_image`'s ``build`` callable.

Both rewrites already exist as z-image ENDPOINT code (``z_image/rope_buffers.py``
for ie#630, ``z_image/pad32.py`` for ie#637) and both are load-bearing for the
DECLARATION, not merely for the endpoint: measured on this box, a fake-tensor
``torch.export`` of ``ZImageTransformer2DModel`` with the latent extents
declared dynamic

* GRAPH-BREAKS without the rope rewrite — ``RopeEmbedder.__call__`` materializes
  its tables on first call ``with torch.device("cpu")``, which dynamo refuses;
* is REFUSED by the declared-range gate without the pad rewrite — the equality
  guard ``Eq(PythonMod(-s18*s57, 32), 0)`` pins the very symbols being declared.

With both installed the same export succeeds and the two latent symbols keep
finite ranges, which is what makes z-image's TWO declared classes (one per CFG
arity) honest rather than optimistic: ten preset rows collapse onto one program
per arm exactly as the endpoint's ``shape_strategy="dynamic-collapse"`` says.

**They live here rather than being imported from the endpoint** because the
declaration is what the mint traces and the endpoint's copies die with its
migration onto this catalog (pgw#1346 W2). The arithmetic is upstream's own in
both cases; ``tests/test_z_image_pgw1346.py`` pins each against the real
upstream implementation rather than trusting the transcription.

The two ``install`` steps rebind names in ``diffusers``' own module, which is
the only reachable seam: ``ZImageTransformer2DModel.__init__`` resolves
``RopeEmbedder`` as a module global, and ``patchify_and_embed`` is called as a
method on the instance. Both are idempotent and both run before any
construction.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
from diffusers.models.transformers import transformer_z_image as _upstream
from torch import nn

#: The dtype the rope real/imag halves are pinned to. Rope is computed in fp32
#: upstream and a cast is a silent precision loss, never a reconfiguration.
TABLE_DTYPE = torch.float32

#: Upstream's own sequence multiple, read rather than restated.
SEQ_MULTI_OF: int = _upstream.SEQ_MULTI_OF

#: The two upstream objects :func:`install` replaces, captured BEFORE they are
#: replaced so tests can difference against the real thing rather than against a
#: copy of it.
UPSTREAM_ROPE = _upstream.RopeEmbedder
UPSTREAM_PATCHIFY = _upstream.ZImageTransformer2DModel.patchify_and_embed

_INSTALLED = False


def table_names(count: int) -> tuple[str, ...]:
    """Buffer names for ``count`` rope axes, in axis order."""

    return tuple(
        f"freqs_cis_{index}_{half}" for index in range(count) for half in ("real", "imag")
    )


class BoundRopeEmbedder(nn.Module):
    """``RopeEmbedder`` as a module whose three rope tables are BUFFERS.

    Upstream's ``RopeEmbedder`` is not an ``nn.Module`` at all: it holds
    ``freqs_cis = None`` and materializes three complex64 tables on first call
    inside ``with torch.device("cpu")``. Three consequences, all measured
    (ie#630): 393,216 bytes are lifted into every compiled cell as anonymous
    constants instead of being rebound from resident tensors (DESIGN-RULINGS
    §1.30); the device pin escapes a meta/fake instantiation context; and the
    lazy build is why the endpoint had to declare ``warm_changes_key=True``.

    The tables are carried as TWO REAL float32 buffers each rather than one
    complex buffer, because ``nn.Module.to()``'s convert casts complex tensors
    too — a module-wide ``complex64 -> bfloat16`` silently discards the
    imaginary part. ``torch.complex`` recomposes them exactly.
    """

    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: Sequence[int] = (16, 56, 56),
        axes_lens: Sequence[int] = (64, 128, 128),
    ) -> None:
        super().__init__()
        if len(axes_dims) != len(axes_lens):
            raise ValueError(
                "axes_dims and axes_lens must have the same length: "
                f"{list(axes_dims)!r} vs {list(axes_lens)!r}"
            )
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        for name, table in self.build_tables().items():
            self.register_buffer(name, table, persistent=False)

    def build_tables(self) -> dict[str, torch.Tensor]:
        """The real/imag halves, derived exactly as upstream derives its list.

        Device-agnostic on purpose — upstream's ``with torch.device("cpu")`` is
        precisely what escapes a meta/fake instantiation context.
        """

        tables: dict[str, torch.Tensor] = {}
        for index, (dim, end) in enumerate(zip(self.axes_dims, self.axes_lens, strict=True)):
            freqs = 1.0 / (
                self.theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
            )
            timestep = torch.arange(end, dtype=torch.float64)
            outer = torch.outer(timestep, freqs).float()
            table = torch.polar(torch.ones_like(outer), outer).to(torch.complex64)
            tables[f"freqs_cis_{index}_real"] = table.real.contiguous()
            tables[f"freqs_cis_{index}_imag"] = table.imag.contiguous()
        return tables

    @property
    def freqs_cis(self) -> list[torch.Tensor]:
        """The three complex tables, recomposed from their halves."""

        return [
            torch.complex(
                getattr(self, f"freqs_cis_{index}_real"),
                getattr(self, f"freqs_cis_{index}_imag"),
            )
            for index in range(len(self.axes_dims))
        ]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2:
            raise ValueError(f"rope ids must be 2-D, got shape {tuple(ids.shape)}")
        if ids.shape[-1] != len(self.axes_dims):
            raise ValueError(
                f"rope ids last dim {ids.shape[-1]} != {len(self.axes_dims)} axes"
            )
        # Gather on the halves and recompose the gathered rows — identical
        # values to upstream's `self.freqs_cis[i][index]`, without building the
        # full complex table per call.
        rows: list[torch.Tensor] = [
            torch.complex(
                getattr(self, f"freqs_cis_{index}_real")[ids[:, index]],
                getattr(self, f"freqs_cis_{index}_imag")[ids[:, index]],
            )
            for index in range(len(self.axes_dims))
        ]
        return torch.cat(rows, dim=-1)

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> BoundRopeEmbedder:
        names = table_names(len(self.axes_dims))
        pristine = {name: self._buffers[name] for name in names}
        # `nn.Module._apply` carries no annotations in the pinned torch, so it
        # is reached through one typed alias rather than left as an untyped
        # call in a strict module.
        apply = cast("Callable[..., BoundRopeEmbedder]", super()._apply)
        out = apply(fn, recurse)
        for name, before in pristine.items():
            after = self._buffers[name]
            if before is None or after is None:
                continue
            if before.is_meta and not after.is_meta:
                # `to_empty()` off a meta instantiation hands back UNINITIALIZED
                # storage; these tables are derived, not loaded, so derive them
                # rather than serve garbage.
                self._buffers[name] = self.build_tables()[name].to(device=after.device)
            elif after.dtype != TABLE_DTYPE:
                # A module-wide dtype cast. Follow the device it targeted; never
                # let it round the table.
                self._buffers[name] = before.to(device=after.device)
        return out


def _padded_len(ori_len: Any) -> Any:
    """``ori_len`` rounded UP to a multiple of :data:`SEQ_MULTI_OF`.

    Value-identical to upstream's ``ori_len + (-ori_len) % SEQ_MULTI_OF`` for
    every non-negative length, and deliberately spelled as a MULTIPLE rather
    than as a modulus: upstream's ``forward`` asserts
    ``len(padded) % SEQ_MULTI_OF == 0`` on the result, and that assertion is
    free only if the tracer can fold it. ``Mod(32*FloorDiv(L + 31, 32), 32)``
    folds to 0 at trace time; ``Mod(L + PythonMod(-L, 32), 32)`` does not,
    because the inner ``PythonMod`` is torch's function and not sympy's ``Mod``.
    """

    return ((ori_len + SEQ_MULTI_OF - 1) // SEQ_MULTI_OF) * SEQ_MULTI_OF


def _pad_mask(total: Any, ori_len: Any, device: Any) -> torch.Tensor:
    """``[False] * ori_len + [True] * (total - ori_len)``, without a 1-D ``cat``.

    Upstream builds this as ``cat([zeros(ori_len), ones(pad)])``. ``cat`` drops
    empty 1-D operands, and deciding whether an operand IS empty is a guard on
    the pad — the thing that must stay symbolic.
    """

    mask: torch.Tensor = torch.arange(total, device=device) >= ori_len
    return mask


def patchify_and_embed(
    self: Any,
    all_image: list[torch.Tensor],
    all_cap_feats: list[torch.Tensor],
    patch_size: int,
    f_patch_size: int,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[tuple[Any, Any, Any]],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    """Upstream's ``patchify_and_embed`` with every pad decision ARITHMETIC.

    Upstream pads each sample's image token run up to a multiple of 32 and then
    asks ``if image_padding_len > 0`` three times. With the latent extents
    declared dynamic every one of those is a decision about a symbolic value,
    so the tracer records an equality guard that PINS the declared symbols — and
    the guard is right to be fatal: the pad is genuinely different per aspect
    row (1024x1024 needs 0, 1248x832 needs 8, 1152x864 needs 16), so a graph
    that decided it once at trace time serves one row and lies about the rest.

    Taking the padded path unconditionally is value-identical at every shape:
    with a zero pad the concatenated tail is empty and the expression is the
    same tensor, and ``arange(L + P) >= L`` is the same boolean run as the
    concatenated mask. Nothing else changes — the coordinate grids, the pad
    token row and the caption branch are upstream's own expressions.
    """

    height_patch = width_patch = patch_size
    frame_patch = f_patch_size
    device = all_image[0].device

    all_image_out: list[torch.Tensor] = []
    all_image_size: list[tuple[Any, Any, Any]] = []
    all_image_pos_ids: list[torch.Tensor] = []
    all_image_pad_mask: list[torch.Tensor] = []
    all_cap_pos_ids: list[torch.Tensor] = []
    all_cap_pad_mask: list[torch.Tensor] = []
    all_cap_feats_out: list[torch.Tensor] = []

    for image, cap_feat in zip(all_image, all_cap_feats, strict=True):
        # --- caption. Its pad is the constant 0 at the pinned 512 today; the
        # same treatment is applied anyway, because a uniform function is
        # cheaper to keep correct than one whose halves disagree about why they
        # are safe.
        cap_ori_len = len(cap_feat)
        cap_total_len = _padded_len(cap_ori_len)
        cap_padding_len = cap_total_len - cap_ori_len
        all_cap_pos_ids.append(
            self.create_coordinate_grid(
                size=(cap_total_len, 1, 1), start=(1, 0, 0), device=device
            ).flatten(0, 2)
        )
        all_cap_pad_mask.append(_pad_mask(cap_total_len, cap_ori_len, device))
        all_cap_feats_out.append(
            torch.cat([cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)], dim=0)
        )

        # --- image.
        channels, frames, height, width = image.size()
        all_image_size.append((frames, height, width))
        frame_tokens = frames // frame_patch
        height_tokens = height // height_patch
        width_tokens = width // width_patch

        image = image.view(
            channels,
            frame_tokens,
            frame_patch,
            height_tokens,
            height_patch,
            width_tokens,
            width_patch,
        )
        # "c f pf h ph w pw -> (f h w) (pf ph pw c)"
        image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(
            frame_tokens * height_tokens * width_tokens,
            frame_patch * height_patch * width_patch * channels,
        )

        image_ori_len = len(image)
        image_total_len = _padded_len(image_ori_len)
        image_padding_len = image_total_len - image_ori_len

        image_ori_pos_ids = self.create_coordinate_grid(
            size=(frame_tokens, height_tokens, width_tokens),
            start=(cap_total_len + 1, 0, 0),
            device=device,
        ).flatten(0, 2)
        all_image_pos_ids.append(
            torch.cat(
                [
                    image_ori_pos_ids,
                    self.create_coordinate_grid(
                        size=(1, 1, 1), start=(0, 0, 0), device=device
                    )
                    .flatten(0, 2)
                    .repeat(image_padding_len, 1),
                ],
                dim=0,
            )
        )
        all_image_pad_mask.append(_pad_mask(image_total_len, image_ori_len, device))
        all_image_out.append(
            torch.cat([image, image[-1:].repeat(image_padding_len, 1)], dim=0)
        )

    return (
        all_image_out,
        all_cap_feats_out,
        all_image_size,
        all_image_pos_ids,
        all_cap_pos_ids,
        all_image_pad_mask,
        all_cap_pad_mask,
    )


def install() -> None:
    """Rebind both upstream names. Idempotent, and it must run BEFORE a build.

    Called from the declaration's ``build`` callable rather than at import, so
    importing the catalog never mutates a third-party module as a side effect of
    reading a declaration.
    """

    global _INSTALLED
    if _INSTALLED:
        return
    # `setattr` rather than assignment: both targets are attributes of a
    # third-party module and one of them is a CLASS, which a type checker
    # rightly refuses to see rebound. The rebinding is the point.
    setattr(_upstream, "RopeEmbedder", BoundRopeEmbedder)
    setattr(_upstream.ZImageTransformer2DModel, "patchify_and_embed", patchify_and_embed)
    _INSTALLED = True


def installed() -> bool:
    """Whether :func:`install` has rebound the two upstream names."""

    return _INSTALLED


__all__ = [
    "SEQ_MULTI_OF",
    "TABLE_DTYPE",
    "UPSTREAM_PATCHIFY",
    "UPSTREAM_ROPE",
    "BoundRopeEmbedder",
    "install",
    "installed",
    "patchify_and_embed",
    "table_names",
]
