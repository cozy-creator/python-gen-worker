"""pgw#844 part B — the MINT refuses a cell whose entries cannot be told
apart at the DISPATCH boundary, and it asks that question the way dispatch
asks it: by ADMISSION, not by equality.

Rewritten in WHOLE-GRAPH terms for pgw#846 (regional cells are retired).  The
failure class survives the retirement: ``EntryDispatch.select`` refuses
``entry_ambiguous`` — per REQUEST, served eager, no refusal at mint — whenever
two entries of one dispatch group admit the same call.  Equality of contracts
cannot see the dangerous half of that (a static row shadowed by a dynamic
sibling over the same hull admits identically while digesting differently),
so the gate runs every entry's own declared call against every sibling's
contract through :func:`aot_serve.assert_ingress` itself.

Real ``torch.export`` programs, real ``aot_package.input_contract``, real
ingress assertion; CPU, no AOTInductor compile, no GPU.  The 36-entry case
mirrors the real sdxl whole-graph declaration (18 declared class rows x 2
pgw#790 adapter arms) and MUST be admitted — that is the pgw#846 revert
precondition (part B's gate must not refuse the whole-graph cell).
"""

from __future__ import annotations

from typing import Any, Tuple

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker import aot_mint, aot_serve  # noqa: E402

FAMILY = "sdxl"
IN_CH = 4
XATTN = 2048
TEXT_LEN = 77
RANK = 64

#: sdxl's nine declared aspect buckets as LATENT extents — the real class
#: rows, whose (H_lat, W_lat) pairs are distinct coordinates even where their
#: products collide (the collision that killed the regional shape is invisible
#: to a whole-graph entry, whose ingress carries H and W separately).
SDXL_BUCKETS: Tuple[Tuple[int, int], ...] = (
    (128, 128),
    (152, 104), (104, 152),
    (168, 96), (96, 168), (144, 112), (112, 144),
    (192, 80), (80, 192),
)


class _UNet(nn.Module):
    """The whole-graph ingress signature: `sample` carries B/C/H_lat/W_lat as
    four SEPARATE extents."""

    def forward(self, sample: Any, encoder_hidden_states: Any) -> Any:
        return sample.mean() + encoder_hidden_states.mean()


class _UNetLifted(nn.Module):
    def forward(self, sample: Any, encoder_hidden_states: Any,
                lora_a: Any, lora_b: Any) -> Any:
        return (sample.mean() + encoder_hidden_states.mean()
                + lora_a.mean() + lora_b.mean())


def _export(b: int, h: int, w: int, *, arm: bool = False,
            hull: Tuple[int, int] | None = None) -> Any:
    args: list = [torch.zeros(b, IN_CH, h, w), torch.zeros(b, TEXT_LEN, XATTN)]
    mod: nn.Module = _UNet()
    if arm:
        args += [torch.zeros(RANK, XATTN), torch.zeros(XATTN, RANK)]
        mod = _UNetLifted()
    dynamic = None
    if hull is not None:
        dim = torch.export.Dim("s_h", min=hull[0], max=hull[1])
        dynamic = tuple([{2: dim}] + [None] * (len(args) - 1))
    return torch.export.export(
        mod.eval(), tuple(args), dynamic_shapes=dynamic, strict=True)


def _entry(name: str, program: Any, *, arm: bool | None = None,
           cfg: bool = False) -> Any:
    names = ("sample", "encoder_hidden_states")
    fork: Tuple[Tuple[str, Any], ...] = (("cfg", cfg),)
    if arm is not None:
        fork = tuple(sorted(
            fork + ((aot_mint.ADAPTER_FORK, bool(arm)),),
            key=lambda kv: str(kv[0])))
        if arm:
            names = names + ("lora_a", "lora_b")
    return aot_mint._MintedEntry(
        name=name,
        spec=aot_mint.ExportSpec(
            family=FAMILY, target="unet", fork=fork,
            lora_bucket=RANK if arm else 0),
        module=None,
        owner=None,
        program=program,
        input_names=names,
        flat_names=names,
        files=[],
        timings={},
    )


def _row_name(b: int, h: int, w: int, *, arm: bool, cfg: bool) -> str:
    return (f"unet/adapter={str(arm).lower()},cfg={str(cfg).lower()}"
            f"/B={b},H_lat={h},T_txt={TEXT_LEN},W_lat={w}")


def test_the_whole_graph_sdxl_shape_is_ADMITTED():
    """The pgw#846 revert precondition: 18 class rows (9 aspect buckets x 2
    CFG arms) x 2 adapter arms = 36 entries, every one a distinct
    (B, H_lat, W_lat) ingress coordinate — the gate must admit the cell the
    fleet is going back to."""
    minted = []
    for cfg, b in ((False, 1), (True, 2)):
        for h, w in SDXL_BUCKETS:
            for arm in (True, False):
                minted.append(_entry(
                    _row_name(b, h, w, arm=arm, cfg=cfg),
                    _export(b, h, w, arm=arm), arm=arm, cfg=cfg))
    assert len(minted) == 36
    aot_mint._gate_dispatch_ambiguity(minted)  # must not raise


def test_two_entries_of_one_dispatch_group_admitting_one_call_are_refused():
    """Two identical ingress coordinates in ONE (target, adapter arm) group:
    dispatch would refuse every call `entry_ambiguous`, so the mint must."""
    minted = [
        _entry("unet/a", _export(1, 128, 128), arm=False),
        _entry("unet/b", _export(1, 128, 128), arm=False),
    ]
    with pytest.raises(aot_mint.MintRefused) as err:
        aot_mint._gate_dispatch_ambiguity(minted)
    assert "dispatch-ambiguity" in str(err.value)
    assert "entry_ambiguous" in str(err.value)
    assert "admitted by more than one entry" in str(err.value)


def test_a_static_row_shadowed_by_a_dynamic_sibling_is_refused():
    """The case an equality digest cannot see, and the reason this gate had to
    change (pgw#844): the two contracts are DIFFERENT and both admit the same
    call."""
    minted = [
        _entry("unet/static", _export(1, 128, 128), arm=False),
        _entry("unet/dynamic", _export(1, 128, 128, hull=(80, 192)),
               arm=False),
    ]
    with pytest.raises(aot_mint.MintRefused) as err:
        aot_mint._gate_dispatch_ambiguity(minted)
    assert "entry_ambiguous" in str(err.value)

    # ...and the digest the gate used BEFORE pgw#844 would have passed it:
    # the two programs do not declare the same placeholder shapes at all.
    assert (aot_mint._entry_ingress_declaration(minted[0])[0]
            != aot_mint._entry_ingress_declaration(minted[1])[0])


def test_adapter_arms_partition_the_dispatch():
    """pgw#790: the two arms of one coordinate are DIFFERENT dispatch groups
    — the lifted pair (positively declared on one side, excluded on the
    other) is what discriminates them, so the same (B, H, W) on both arms is
    correct, never ambiguous."""
    aot_mint._gate_dispatch_ambiguity([
        _entry(_row_name(1, 128, 128, arm=True, cfg=False),
               _export(1, 128, 128, arm=True), arm=True),
        _entry(_row_name(1, 128, 128, arm=False, cfg=False),
               _export(1, 128, 128), arm=False),
    ])


def test_distinct_coordinates_stay_admitted():
    """The gate must not refuse a healthy cell: distinct (H_lat, W_lat)
    coordinates — including a transposed aspect pair, whose token PRODUCTS
    collide but whose whole-graph ingress shapes do not — discriminate by
    ingress."""
    aot_mint._gate_dispatch_ambiguity([
        _entry("unet/H_lat=192,W_lat=80", _export(1, 192, 80), arm=False),
        _entry("unet/H_lat=80,W_lat=192", _export(1, 80, 192), arm=False),
    ])
