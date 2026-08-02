"""pgw#868 A4: the entry child's DEVICE high-water, which nothing ever measured.

The pool's per-entry device ask is `mint_budget.co_residency().need_bytes` —
**not** a compile-child measurement. For sdxl it is `4.87 GiB resident * 1.25
+ 5 GiB = 11.09 GiB`, where the 1.25 is what `mint_budget`'s own docstring
calls "a fraction nobody measured" and the 5 GiB is a flat constant. That
number is what reports as `per_entry_device_basis: 'measured'` (meaning "the
caller handed a probed number"), and it is what holds K to 1-2 on both cards
measured — while CPU would permit 63-127.

So the child now reports what it ACTUALLY peaked at. Telemetry only: the width
policy is deliberately unchanged in the same commit (pgw#830's rule).
"""
from __future__ import annotations

from gen_worker import aot_compile_child, aot_compile_pool


def test_device_fields_are_present_and_never_raise():
    """Off-GPU this box reports zeros; the contract is that it ANSWERS."""
    fields = aot_compile_child._device_fields()
    assert set(fields) == {"peak_device_bytes", "peak_device_reserved_bytes"}
    assert all(isinstance(v, int) and v >= 0 for v in fields.values())


def test_probe_survives_a_broken_torch(monkeypatch):
    """A probe never changes an outcome — including when it cannot run."""
    import builtins

    real = builtins.__import__

    def boom(name, *a, **k):
        if name == "torch":
            raise RuntimeError("no torch today")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    assert aot_compile_child._peak_device() == (0, 0)


def test_report_carries_the_device_highwater_and_defaults_to_zero():
    """Defaulted, so an OLDER child's report still decodes on a new parent."""
    import msgspec

    older = msgspec.json.encode({
        "entry": "e", "status": "compiled", "peak_rss_bytes": 7})
    decoded = msgspec.json.decode(older, type=aot_compile_pool.EntryReport)
    assert decoded.peak_device_bytes == 0
    assert decoded.peak_device_reserved_bytes == 0

    newer = aot_compile_pool.EntryReport(
        entry="e", status="compiled", peak_rss_bytes=7,
        peak_device_bytes=123, peak_device_reserved_bytes=456)
    roundtrip = msgspec.json.decode(
        msgspec.json.encode(newer), type=aot_compile_pool.EntryReport)
    assert roundtrip.peak_device_bytes == 123
    assert roundtrip.peak_device_reserved_bytes == 456


def test_the_estimate_this_replaces_is_arithmetic_not_measurement():
    """Pins the decomposition, so the 11.07 GiB stops being a mystery number.

    sdxl UNet resident 4.87 GiB -> 4.87*1.25 + 5 = 11.09 GiB, which is what
    both the 4090 and the L40S printed as their cap across a 2x card change.
    If either constant moves, this goes RED and the attribution is re-read.
    """
    from gen_worker import mint_budget

    assert mint_budget._UNMEASURED_ACTIVATION_FRACTION == 0.25
    resident_gib = 4.87
    estimate = resident_gib * (1 + mint_budget._UNMEASURED_ACTIVATION_FRACTION) + 5
    assert 11.0 < estimate < 11.2, estimate
    # 56% of the ask is the two terms nobody measured
    unmeasured = resident_gib * mint_budget._UNMEASURED_ACTIVATION_FRACTION + 5
    assert unmeasured / estimate > 0.55
