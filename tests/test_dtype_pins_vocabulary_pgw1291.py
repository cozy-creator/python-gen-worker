"""pgw#1291 — the component-dtype pin gate was SILENTLY OFF for four live
spellings, and nothing could have told us.

`dtype_pins.dtype_bits` answers 0 for a spelling its table does not carry, and
`is_narrowing` reads 0 as "not narrower". So an unknown spelling does not make
the gate refuse or complain — it makes the gate AGREE. That is the exact shape
of pgw#1133, the invisible-truncation bug this module was built to catch.

Measured on master before this fix:

    dtype_bits("int8")     == 0   is_narrowing("int8", "fp32")     is False
    dtype_bits("uint8")    == 0   is_narrowing("uint8", "fp32")    is False
    dtype_bits("fp8_e4m3") == 0   is_narrowing("fp8_e4m3", "fp32") is False
    dtype_bits("fp8_e5m2") == 0   is_narrowing("fp8_e5m2", "fp32") is False

`int8` and `uint8` are contradicted by the module's own docstring ("Quant
dtypes are all narrower than every pin, which is the answer that matters").
`fp8_e4m3`/`fp8_e5m2` is `tensorlayout`'s canonical vocabulary — what tensorhub
derives from bytes and what th#1994 recorded the catalog as owing a move to.

The fix folds separators and case, so one fact cannot hide behind four
spellings. THIS FILE IS THE FENCE: every dtype token a producer in this
repository can EMIT must have a width, so the next vocabulary addition fails
here instead of disarming a publish gate on a rented pod.

RED-VERIFY, each arm reverted alone and each naming a DIFFERENT test — which is
the point, because the two halves of this fix fail differently:

  * drop `int8`/`uint8` from DTYPE_BITS
    -> `eight_bit_integer_storage_is_narrower_than_every_pin` (NOT the fence:
       neither token is in an emitted set today, which is exactly why the
       docstring's claim about them went unchecked for so long)
  * drop `nvfp4`, which `clone._KNOWN_DTYPES` DOES emit
    -> `the_gate_is_armed_for_every_emitted_dtype`: "1 emitted dtype(s) score
       0 bits ... ['nvfp4']"
  * revert `dtype_bits` to the plain `.get(lower())`
    -> `a_flavor_answers_the_same_width_however_it_is_spelled` on five of its
       six spellings

Run: uv run pytest tests/test_dtype_pins_vocabulary_pgw1291.py -v
"""

from __future__ import annotations

import pytest

from gen_worker.convert.clone import _KNOWN_DTYPES
from gen_worker.convert.dtype_pins import DTYPE_BITS, dtype_bits, is_narrowing
from gen_worker.convert.ingest import _SAFETENSORS_DTYPE_NAMES

#: `"source"` is a MODE ("publish the source's own weights untouched"), not a
#: dtype — it never reaches a width comparison. Named here so its absence is a
#: decision rather than an oversight.
_NOT_A_DTYPE = {"source"}


def _emitted_dtype_tokens() -> set[str]:
    """Every dtype token this repository can put on an artifact or a request.

    Two producers, read rather than restated: the safetensors header projection
    (`ingest._SAFETENSORS_DTYPE_NAMES`, which is what `detect_snapshot_dtype`
    reports and therefore what `verify_produced_tree` compares) and the
    conversion request vocabulary (`clone._KNOWN_DTYPES`, which is what
    `cast_exempt_components` and `check_explicit_pin_conflict` are handed).
    """
    return ({str(v) for v in _SAFETENSORS_DTYPE_NAMES.values() if v}
            | {str(d) for d in _KNOWN_DTYPES}) - _NOT_A_DTYPE


def test_the_gate_is_armed_for_every_emitted_dtype() -> None:
    """A token a producer can emit but the table cannot price is a gate that
    silently agrees with whatever it is shown."""
    unpriced = sorted(d for d in _emitted_dtype_tokens() if dtype_bits(d) == 0)
    assert not unpriced, (
        f"{len(unpriced)} emitted dtype(s) score 0 bits, so `is_narrowing` "
        f"answers False for them and the component-pin gate is OFF: {unpriced}"
    )


@pytest.mark.parametrize("spelling", [
    "fp8:e4m3", "fp8_e4m3", "fp8-e4m3", "FP8_E4M3", "F8_E4M3", "  fp8_e4m3  ",
])
def test_a_flavor_answers_the_same_width_however_it_is_spelled(spelling: str) -> None:
    """Four vocabularies meet in this module — this one's `:`, tensorlayout's
    `_`, producer labels' `-`, and the safetensors header's upper case. One
    fact must not be reachable by some of them and not others."""
    assert dtype_bits(spelling) == 8, spelling
    assert is_narrowing(spelling, "fp32") is True, spelling
    assert is_narrowing(spelling, "bf16") is True, spelling


def test_eight_bit_integer_storage_is_narrower_than_every_pin() -> None:
    """The module docstring's own claim, asserted. It was FALSE for `int8` and
    `uint8` — the two spellings a bitsandbytes / LLM.int8() tree carries."""
    for d in ("int8", "uint8", "i8", "u8"):
        assert dtype_bits(d) == 8, d
        assert is_narrowing(d, "fp32") is True, d


def test_an_unknown_dtype_still_scores_zero() -> None:
    """The fold must not turn the table into a prefix or fuzzy match: a token
    nobody emits is still unpriced, and the fence above is what keeps that
    honest rather than convenient."""
    assert dtype_bits("definitely-not-a-dtype") == 0
    assert dtype_bits("") == 0
    assert is_narrowing("definitely-not-a-dtype", "fp32") is False


def test_widening_and_equal_width_are_never_narrowing() -> None:
    """The comparison this module exists to make, in the direction that must
    NOT fire — a cast to fp32 of an fp32 pin is a no-op, not a violation."""
    assert is_narrowing("fp32", "fp32") is False
    assert is_narrowing("fp32", "bf16") is False
    assert is_narrowing("bf16", "fp32") is True


def test_the_table_has_no_two_entries_that_fold_together_with_different_widths() -> None:
    """Folding is only safe while it is injective on WIDTH. Two spellings of
    one fact must not disagree about how wide that fact is."""
    from gen_worker.convert.dtype_pins import _fold_dtype

    by_key: dict[str, set[int]] = {}
    for spelling, bits in DTYPE_BITS.items():
        by_key.setdefault(_fold_dtype(spelling), set()).add(bits)
    clashes = {k: sorted(v) for k, v in by_key.items() if len(v) > 1}
    assert not clashes, f"spellings fold together but disagree on width: {clashes}"
