"""The shared verdict ladder and the compiled-compiled graph (assembled-vs-eager)
calibration — the family-general survivors of the pgw#846 regional
retirement, moved here from the deleted regional test scenarios.

Every numeric assertion is pinned to a MEASURED coordinate from pgw#812 /
pgw#814 on the production toolchain (torch 2.13.0+cu130, L4/sm_89).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import pytest

torch = pytest.importorskip("torch")

from gen_worker import numerics_ladder  # noqa: E402
from gen_worker.api.decorators import Compile  # noqa: E402

MEASURED: Tuple[Tuple[str, float, float, str], ...] = (
    ("bf16 control, whole-graph", 0.99979, 0.997, "healthy"),
    ("sdxl w8a8, whole-graph", 0.99984, 0.998, "healthy"),
    ("flux2 w8a8 regional T_img=4096", 0.9890, 0.99, "degraded"),
    ("flux2 w8a8 regional T_img=8160", 0.9926, 0.99, "degraded"),
    ("flux2 w8a8 rowwise whole-graph", 0.97300, 0.902, "destroyed"),
    ("flux2 w8a8 pertensor whole-graph", 0.93094, 0.905, "destroyed"),
)


def _pair_at(cosine: float, retention: float, n: int = 4096) -> Tuple[Any, Any]:
    """Two real tensors whose aggregate cosine and norm ratio are EXACTLY the
    requested pair, built from an orthogonal decomposition rather than a
    random search — the gate must be tested on the numbers it will meet."""
    generator = torch.Generator().manual_seed(17)
    a = torch.randn(n, generator=generator, dtype=torch.float64)
    a = a / a.norm()
    o = torch.randn(n, generator=generator, dtype=torch.float64)
    o = o - (o @ a) * a
    o = o / o.norm()
    b = retention * (cosine * a + math.sqrt(max(0.0, 1.0 - cosine ** 2)) * o)
    return a.to(torch.float32), b.to(torch.float32)


def _decl(**kwargs: Any) -> Compile:
    base: Dict[str, Any] = dict(
        family="scenario", shapes=((1024, 1024),), targets=("unet",),
        text_len=77, shape_strategy="static-rows", warm_changes_key=False)
    base.update(kwargs)
    return Compile(**base)


def test_the_ladder_calls_every_MEASURED_configuration_correctly() -> None:
    """The whole calibration, pinned. Moving `NUMERICS_FLOOR` or
    `NUMERICS_WARN` fails here with the evidence attached.

    RED-VERIFY, in line: pgw#800's ADAPTER thresholds (0.80 / 0.99) — which
    pgw#814 explicitly warns must not be inherited — call the flux2 w8a8
    whole-graph artifact DEGRADED and would have SERVED it. pgw#814's own
    ruling on that artifact is 'do not adopt a flux2 w8a8 compiled graph until this
    closes'."""
    from gen_worker.models import adapter_fidelity

    for label, cosine, retention, want in MEASURED:
        got = numerics_ladder.DEFAULT_THRESHOLDS.verdict(cosine, retention)
        assert got == want, f"{label}: cos={cosine} ret={retention} -> {got}"

    # The red half: the adapter calibration serves what this one refuses.
    for label, cosine, retention, want in MEASURED:
        adapter_call = adapter_fidelity.ADAPTER_THRESHOLDS.verdict(
            cosine, retention)
        if want == "destroyed":
            assert adapter_call == "degraded", (
                f"{label}: the adapter ladder would have SERVED this")


def test_the_floor_and_warn_bracket_the_measured_band() -> None:
    """Both constants are DERIVED, and the derivation is checkable."""
    worst_accepted = 0.9890   # flux2 w8a8, pgw#812/#814
    best_refused = 0.97300    # flux2 w8a8 rowwise whole-graph, pgw#814
    assert best_refused < numerics_ladder.NUMERICS_FLOOR < worst_accepted
    assert abs(numerics_ladder.NUMERICS_FLOOR
               - math.sqrt(worst_accepted * best_refused)) < 0.001
    ret_accepted, ret_refused = 0.997, 0.905
    assert abs(numerics_ladder.NUMERICS_RETENTION_FLOOR
               - math.sqrt(ret_accepted * ret_refused)) < 0.001


def test_a_perfect_cosine_at_the_wrong_MAGNITUDE_is_not_healthy() -> None:
    """Cosine is scale-invariant. An artifact that reproduces eager's
    direction exactly at 0.9x the magnitude serves a systematically dimmer
    image and pgw#800's ladder could not see it, because an adapter's
    retention is evidence rather than a bound (a destroyed one measures
    15.3)."""
    assert numerics_ladder.DEFAULT_THRESHOLDS.verdict(1.0, 0.90) == "degraded"
    assert numerics_ladder.DEFAULT_THRESHOLDS.verdict(1.0, 1.0) == "healthy"


def test_declared_numerics_tolerance_is_validated_and_defaults_are_measured(
) -> None:
    decl = _decl(numerics_floor=0.995, numerics_warn=0.9999)
    thresholds = numerics_ladder.declared_thresholds(decl)
    assert (thresholds.floor, thresholds.warn) == (0.995, 0.9999)
    # Undeclared falls back to the SDK band derived from pgw#814.
    assert numerics_ladder.declared_thresholds(_decl()) is \
        numerics_ladder.DEFAULT_THRESHOLDS
    with pytest.raises(ValueError, match="must not exceed numerics_warn"):
        _decl(numerics_floor=0.999, numerics_warn=0.98)
    with pytest.raises(ValueError, match="COSINE bound"):
        _decl(numerics_floor=1.5)


def test_compare_outputs_is_norm_weighted_never_a_per_row_median() -> None:
    """pgw#800's rule, carried across populations: a handful of destroyed
    high-norm outputs must not hide behind many intact low-norm ones."""
    good_a, good_b = _pair_at(1.0, 1.0, n=64)
    bad_a, bad_b = _pair_at(0.0, 1.0, n=64)
    # Three intact tiny rows, one destroyed row carrying 100x the norm.
    reference = [good_a * 0.01, good_a * 0.01, good_a * 0.01, bad_a * 1.0]
    subject = [good_b * 0.01, good_b * 0.01, good_b * 0.01, bad_b * 1.0]
    cmp_ = numerics_ladder.compare_outputs(
        reference, subject, thresholds=numerics_ladder.DEFAULT_THRESHOLDS)
    median = sorted(r.cosine for r in cmp_.rows)[len(cmp_.rows) // 2]
    assert median > 0.99          # a median would call this healthy
    assert cmp_.cosine < 0.1      # the norm-weighted aggregate does not
    assert cmp_.verdict == "destroyed"


def test_compare_outputs_refuses_a_STRUCTURAL_mismatch() -> None:
    """A silently-dropped output is the failure this gate exists to catch,
    not a row to average over."""
    a, b = _pair_at(1.0, 1.0, n=16)
    with pytest.raises(ValueError, match="output structure differs"):
        numerics_ladder.compare_outputs(
            [a, a], [b], thresholds=numerics_ladder.DEFAULT_THRESHOLDS)
    with pytest.raises(ValueError, match="shape differs"):
        numerics_ladder.compare_outputs(
            a, b.reshape(4, 4), thresholds=numerics_ladder.DEFAULT_THRESHOLDS)
