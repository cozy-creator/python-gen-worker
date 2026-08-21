from __future__ import annotations

import pytest

from gen_worker.convert.clone import _KNOWN_DTYPES
from gen_worker.convert.dtype_pins import DTYPE_BITS, dtype_bits, is_narrowing
from gen_worker.convert.ingest import _SAFETENSORS_DTYPE_NAMES

_NOT_A_DTYPE = {"source"}


def _emitted_dtype_tokens() -> set[str]:
    return ({str(v) for v in _SAFETENSORS_DTYPE_NAMES.values() if v}
            | {str(d) for d in _KNOWN_DTYPES}) - _NOT_A_DTYPE


def test_the_gate_is_armed_for_every_emitted_dtype() -> None:
    """A token a producer can emit but the table cannot price is a gate that silently agrees with whatever it is shown."""
    unpriced = sorted(d for d in _emitted_dtype_tokens() if dtype_bits(d) == 0)
    assert not unpriced, (
        f"{len(unpriced)} emitted dtype(s) score 0 bits, so `is_narrowing` "
        f"answers False for them and the component-pin gate is OFF: {unpriced}"
    )


@pytest.mark.parametrize("spelling", [
    "fp8:e4m3", "fp8_e4m3", "fp8-e4m3", "FP8_E4M3", "F8_E4M3", "  fp8_e4m3  ",
])
def test_a_flavor_answers_the_same_width_however_it_is_spelled(spelling: str) -> None:
    """Four vocabularies meet in this module — this one's `:`, tensorlayout's `_`, producer labels' `-`, and the safetensors header's upper case."""
    assert dtype_bits(spelling) == 8, spelling
    assert is_narrowing(spelling, "fp32") is True, spelling
    assert is_narrowing(spelling, "bf16") is True, spelling


def test_eight_bit_integer_storage_is_narrower_than_every_pin() -> None:
    """The module docstring's own claim, asserted."""
    for d in ("int8", "uint8", "i8", "u8"):
        assert dtype_bits(d) == 8, d
        assert is_narrowing(d, "fp32") is True, d


def test_an_unknown_dtype_still_scores_zero() -> None:
    """The fold must not turn the table into a prefix or fuzzy match: a token nobody emits is still unpriced, and the fence above is what keeps that honest rather than convenient."""
    assert dtype_bits("definitely-not-a-dtype") == 0
    assert dtype_bits("") == 0
    assert is_narrowing("definitely-not-a-dtype", "fp32") is False


def test_widening_and_equal_width_are_never_narrowing() -> None:
    """The comparison this module exists to make, in the direction that must NOT fire — a cast to fp32 of an fp32 pin is a no-op, not a violation."""
    assert is_narrowing("fp32", "fp32") is False
    assert is_narrowing("fp32", "bf16") is False
    assert is_narrowing("bf16", "fp32") is True


def test_the_table_has_no_two_entries_that_fold_together_with_different_widths() -> None:
    """Folding is only safe while it is injective on WIDTH."""
    from gen_worker.convert.dtype_pins import _fold_dtype

    by_key: dict[str, set[int]] = {}
    for spelling, bits in DTYPE_BITS.items():
        by_key.setdefault(_fold_dtype(spelling), set()).add(bits)
    clashes = {k: sorted(v) for k, v in by_key.items() if len(v) > 1}
    assert not clashes, f"spellings fold together but disagree on width: {clashes}"
