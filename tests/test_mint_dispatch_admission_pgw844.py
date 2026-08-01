"""pgw#844 part B — the MINT refuses a cell whose entries cannot be told
apart at the DISPATCH boundary, and it asks that question the way dispatch
asks it: by ADMISSION, not by equality.

pgw#829 landed a dispatch-ambiguity gate that compared a digest of each
entry's placeholder shapes.  That catches entries whose declared contracts are
IDENTICAL — sdxl's nine static aspect rows collapsing onto four token counts,
which is the case it was written for.  It cannot catch the case pgw#829's own
remedy introduces: a STATIC row and a DYNAMIC row over the same token hull
have different digests and both admit the same call, so
``EntryDispatch.select`` refuses ``entry_ambiguous`` on a cell the mint
published as clean.  A partially collapsed cell — one block class collapsed,
one still specialized, which is exactly what
``regional_shape_strategy='dynamic-collapse'`` produces on a mixed
conv-free/conv-bearing block population — is that shape.

The gate now runs every entry's own declared call against every sibling's
contract through :func:`aot_serve.assert_ingress` itself.  Real
``torch.export`` programs, real ``aot_package.input_contract``, real ingress
assertion; CPU, no AOTInductor compile, no GPU.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker import aot_mint, aot_serve  # noqa: E402

FAMILY = "sdxl"
CHANNELS = 64
TEXT_LEN = 77


class _Block(nn.Module):
    """A block as the regional mint sees one: it takes the FLATTENED token
    sequence, never the latent H and W it came from."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(CHANNELS, CHANNELS)

    def forward(self, hidden_states: Any, encoder_hidden_states: Any) -> Any:
        return self.proj(hidden_states) + encoder_hidden_states.mean(
            dim=1, keepdim=True)


def _export(tokens: int, hull: Tuple[int, int] | None = None) -> Any:
    block = _Block().eval()
    hidden = torch.zeros(1, tokens, CHANNELS)
    encoder = torch.zeros(1, TEXT_LEN, CHANNELS)
    dynamic = None
    if hull is not None:
        dim = torch.export.Dim("s_tok", min=hull[0], max=hull[1])
        dynamic = ({1: dim}, None)
    return torch.export.export(
        block, (hidden, encoder), dynamic_shapes=dynamic, strict=True)


def _entry(name: str, program: Any, *, block: str = "Block#0") -> Any:
    return aot_mint._MintedEntry(
        name=name,
        spec=aot_mint.ExportSpec(
            family=FAMILY, target="unet", regional=True,
            fork=(("block", block),)),
        module=None,
        owner=None,
        program=program,
        input_names=("hidden_states", "encoder_hidden_states"),
        flat_names=("hidden_states", "encoder_hidden_states"),
        files=[],
        timings={},
    )


def test_two_static_rows_on_one_token_count_are_refused_by_admission():
    """The measured sdxl case: 192x80 and 80x192 are the same 15360 tokens."""
    minted = [
        _entry("unet/block=Block#0/H_lat=192,W_lat=80", _export(15360)),
        _entry("unet/block=Block#0/H_lat=80,W_lat=192", _export(15360)),
    ]
    with pytest.raises(aot_mint.MintRefused) as err:
        aot_mint._gate_dispatch_ambiguity(minted)
    assert "dispatch-ambiguity" in str(err.value)
    assert "entry_ambiguous" in str(err.value)
    assert "admitted by more than one entry" in str(err.value)


def test_a_static_row_shadowed_by_a_COLLAPSED_sibling_is_refused():
    """The case an equality digest cannot see, and the reason this gate had to
    change: the two contracts are DIFFERENT and both admit the same call.

    A partially collapsed regional cell produces exactly this — the conv-free
    class collapses onto one dynamic entry while a conv-bearing sibling of the
    same block class keeps its static rows.  Dispatch would admit both and
    refuse ``entry_ambiguous`` on every call in the overlap.
    """
    minted = [
        _entry("unet/block=Block#0/H_lat=128,W_lat=128", _export(16384)),
        _entry("unet/block=Block#0/collapsed",
               _export(16384, hull=(15360, 16384))),
    ]
    with pytest.raises(aot_mint.MintRefused) as err:
        aot_mint._gate_dispatch_ambiguity(minted)
    assert "entry_ambiguous" in str(err.value)

    # …and the digest the gate used BEFORE pgw#844 would have passed it: the
    # two programs do not declare the same placeholder shapes at all.
    static, collapsed = (row.program for row in minted)
    assert (aot_mint._entry_ingress_declaration(minted[0])[0]
            != aot_mint._entry_ingress_declaration(minted[1])[0])
    assert static is not collapsed


def test_the_collapsed_cell_alone_dispatches_every_declared_bucket():
    """Part B's serving-side proof: after the collapse there is ONE entry over
    the token hull, and every declared sdxl aspect bucket admits exactly it.

    This is the ``Done when`` bullet stated as a test — one call per declared
    bucket, exactly one admitting entry — asked of the same
    :class:`aot_serve.EntryDispatch` a pod runs.
    """
    buckets = ((128, 128), (152, 104), (104, 152), (168, 96), (96, 168),
               (144, 112), (112, 144), (192, 80), (80, 192))
    tokens = sorted({h * w for h, w in buckets})
    program = _export(tokens[0], hull=(tokens[0], tokens[-1]))
    minted = [_entry("unet/block=Block#0/collapsed", program)]

    # The mint admits it: one entry cannot be ambiguous with itself.
    aot_mint._gate_dispatch_ambiguity(minted)

    contract, _calls = aot_mint._entry_ingress_declaration(minted[0])
    dispatch = aot_serve.EntryDispatch(((
        "unet/block=Block#0/collapsed",
        aot_serve.ArtifactRunner(
            package=lambda *feeds: feeds[0], contract=contract,
            constants=(), module_name="unet",
            entry="unet/block=Block#0/collapsed", bound=True, family=FAMILY),
    ),))
    for h, w in buckets:
        name, _runner = dispatch.select((
            torch.zeros(1, h * w, CHANNELS),
            torch.zeros(1, TEXT_LEN, CHANNELS),
        ), {})
        assert name == "unet/block=Block#0/collapsed"


def test_distinct_token_counts_and_distinct_block_classes_stay_admitted():
    """The gate must not refuse a healthy cell.

    Two genuinely different token counts discriminate by ingress; two
    different BLOCK CLASSES land in different ``BlockDispatch`` objects and
    are never asked to discriminate each other, so identical contracts there
    are correct rather than ambiguous.
    """
    aot_mint._gate_dispatch_ambiguity([
        _entry("unet/block=Block#0/H_lat=128,W_lat=128", _export(16384)),
        _entry("unet/block=Block#0/H_lat=192,W_lat=80", _export(15360)),
    ])
    aot_mint._gate_dispatch_ambiguity([
        _entry("unet/block=A#0/x", _export(16384), block="A#0"),
        _entry("unet/block=B#1/x", _export(16384), block="B#1"),
    ])
