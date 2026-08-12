"""th#1800: a self-mint decline must name the CARD CLASS that would admit it.

The wall this covers is real and measured (ie#655, wan-2.2-t2v-a14b on an
80 GiB H100)::

    self_mint_skipped reason=insufficient_vram headroom=37.68GiB
                      needed~=72.54GiB resident=40.65GiB activation=26.89GiB

Everything in that line is about the card the mint did NOT fit on. Nothing in
it answers the only question a §4.28 platform can act on — *which card does?*
— because the mint path §4.28 leaves is "boot an ordinary serving pod there",
and a pod is booked by SKU, not by shortfall. That number had to be
re-derived by hand (40.65 + 72.54 = 113.19 GiB, i.e. H200-class) before
anybody could say so, and the hand-derivation is exactly why the issue was
filed as "no path to a compile cell" rather than as a placement fact.

The arithmetic is deliberately the server's resident set PLUS the child's
whole ask: the server's weights are already allocated and therefore already
outside ``free_bytes``, so a card that carries only ``need_bytes`` carries the
child and evicts the tenant.
"""

from __future__ import annotations

from gen_worker import mint_budget

_GIB = 1 << 30


def _wan22_decline() -> mint_budget.MintBudget:
    """The ie#655 reading, verbatim, as a budget value."""
    return mint_budget.MintBudget(
        fits=False,
        probed=True,
        measured=True,
        free_bytes=int(37.68 * _GIB),
        need_bytes=int(72.54 * _GIB),
        resident_bytes=int(40.65 * _GIB),
        activation_bytes=int(26.89 * _GIB),
        cap_bytes=int(72.54 * _GIB),
    )


def test_card_bytes_is_the_placement_fact() -> None:
    """RED before th#1800: ``card_bytes`` did not exist, so the H200 verdict
    lived only in a tracker paragraph."""
    budget = _wan22_decline()
    assert budget.card_bytes == budget.resident_bytes + budget.need_bytes
    gib = budget.card_bytes / _GIB
    assert 113.0 < gib < 113.5, gib

    # The verdict the number carries, stated as the two catalog facts it
    # decides between (runpod-go-sdk gpu_catalog.go): an 80 GiB H100 cannot
    # admit this mint and a 141 GB H200 can. Both are sm_90, so the cell the
    # H200 mints is key-identical for the H100 fleet.
    assert budget.card_bytes > 80 * _GIB
    assert budget.card_bytes < 131 * _GIB


def test_decline_line_names_the_card() -> None:
    """The fact has to be ON THE WIRE, not merely computable: the decline is
    an activity event the hub stores, and a fact nobody transmits is a fact
    the next lane re-derives."""
    line = _wan22_decline().line("self_mint_skipped", "insufficient_vram")
    assert "card>=113.19GiB" in line, line
    # ...beside, not instead of, everything the line already carried.
    for token in ("reason=insufficient_vram", "headroom=", "needed~=",
                  "resident=", "activation=", "cap=", "cache_slack="):
        assert token in line, (token, line)


def test_unprobeable_states_nothing() -> None:
    """A card that cannot be read decides no placement. ``card_bytes`` is 0
    and the line is unchanged — this module never guesses a SKU."""
    blind = mint_budget.MintBudget(fits=True, probed=False)
    assert blind.card_bytes == 0
    assert blind.line("self_mint_skipped", "x") == (
        "self_mint_skipped reason=x headroom=unprobeable")


def test_a_fitting_budget_also_states_its_card() -> None:
    """Not a decline-only field. A mint that fits still records what it took,
    so the fleet can learn a family's mint card from a SUCCESS as well as
    from a refusal."""
    ok = mint_budget.MintBudget(
        fits=True, probed=True, measured=True,
        free_bytes=40 * _GIB, need_bytes=12 * _GIB,
        resident_bytes=8 * _GIB, activation_bytes=2 * _GIB,
        cap_bytes=38 * _GIB)
    assert ok.card_bytes == 20 * _GIB
    assert "card>=20.00GiB" in ok.line("self_mint", "ok")
