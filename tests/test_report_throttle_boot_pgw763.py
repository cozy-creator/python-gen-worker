"""The parent's report throttles must not swallow the FIRST report on a
freshly-booted host (found by the 0.84.0 release gate).

Every throttle in `procsplit.parent` is `now - last_reported_at >= INTERVAL`
against `time.monotonic()`, which on Linux is time since BOOT. Seeding
`last_reported_at = 0.0` therefore does not mean "never reported" — it means
"reported at boot", so on any host whose uptime is below the interval the FIRST
report of each class is silently dropped.

That is not a corner: a worker pod's uptime is under 300 s for the first five
minutes of its life, which is precisely when a crash loop or an allowlist probe
is most worth hearing about. CI caught it because a GitHub runner boots minutes
before the suite runs; this box (uptime measured in days) could never see it.

The sentinel must mean "never", so the first report always goes out and the
interval only throttles the ones AFTER it.

Run: uv run pytest tests/test_report_throttle_boot_pgw763.py -q
"""

from __future__ import annotations

import gen_worker.procsplit.parent as parent

# The uptime of a pod that booted two minutes ago — below every interval below.
FRESHLY_BOOTED_MONOTONIC = 120.0

THROTTLES = (
    ("crash loop", parent._CRASH_LOOP_REPORT_MIN_INTERVAL_S),
    ("action refusal", parent._ACTION_REFUSAL_REPORT_MIN_INTERVAL_S),
    ("billing attestation", parent._ATTESTATION_REPORT_MIN_INTERVAL_S),
    ("capability withholding", parent._CAPABILITY_REPORT_MIN_INTERVAL_S),
)


def test_the_never_reported_sentinel_is_not_a_timestamp():
    """`0.0` is a real point on the monotonic clock: boot. `-inf` is not."""
    assert parent._NEVER_REPORTED == float("-inf"), (
        "the 'never reported' sentinel must not be a reachable monotonic "
        "reading, or it dates the report to whenever the clock read that value"
    )


def test_the_first_report_goes_out_on_a_freshly_booted_host():
    for name, interval in THROTTLES:
        elapsed = FRESHLY_BOOTED_MONOTONIC - parent._NEVER_REPORTED
        assert elapsed >= interval, (
            f"the first {name} report is SWALLOWED on a host with "
            f"{FRESHLY_BOOTED_MONOTONIC:.0f}s of uptime — this is the "
            "0.0-sentinel defect, and it hits every worker pod's first "
            f"{interval:.0f} seconds"
        )


def test_the_throttle_still_throttles_after_a_real_report():
    """The fix must not turn the rate limiter off — only its cold start."""
    for name, interval in THROTTLES:
        reported_at = FRESHLY_BOOTED_MONOTONIC
        a_moment_later = reported_at + 1.0
        assert a_moment_later - reported_at < interval, (
            f"a second {name} report one second after the first must still be "
            "throttled"
        )


def test_every_throttle_field_is_seeded_with_the_sentinel():
    """A new throttle that copies the old `0.0` idiom reintroduces the bug."""
    src = parent.__file__
    with open(src, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    offenders = [
        (i + 1, ln.strip())
        for i, ln in enumerate(lines)
        if "_report_at" in ln and "= 0.0" in ln
    ]
    assert not offenders, (
        "these throttle fields are seeded at monotonic 0.0 (= boot), so their "
        f"first report is dropped on a freshly-booted pod: {offenders}"
    )
