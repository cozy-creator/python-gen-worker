"""The residency ledger records what loads and requests actually cost."""

from __future__ import annotations

import json
from pathlib import Path

from gen_worker.models.residency_ledger import (
    MIN_SAMPLES_FOR_PERCENTILE,
    WINDOW,
    ResidencyLedger,
    shape_key,
)

_MIB = 1 << 20


def _ledger(
    tmp_path: Path, endpoint: str = "sdxl", ckpt: str = "abc123"
) -> ResidencyLedger:
    return ResidencyLedger(endpoint, ckpt, root=tmp_path)


def test_a_small_window_refuses_to_offer_a_percentile(tmp_path: Path) -> None:
    """THE GUARD THAT MATTERS."""
    led = _ledger(tmp_path)
    k = shape_key(width=1024, height=1024, batch=2)
    for i in range(MIN_SAMPLES_FOR_PERCENTILE - 1):
        led.observe_request(k, activation_bytes=(i + 1) * _MIB)
    assert led.stats(k).activation_percentile() is None, (
        "a percentile was offered from fewer samples than can support one"
    )
    led.observe_request(k, activation_bytes=99 * _MIB)
    assert led.stats(k).activation_percentile() is not None, (
        "the percentile never becomes available once the window is big enough"
    )


def test_the_summary_says_NOT_TRUSTED_rather_than_quoting_a_thin_number(
    tmp_path: Path,
) -> None:
    led = _ledger(tmp_path)
    k = shape_key(width=512, height=512, batch=1)
    led.observe_request(k, activation_bytes=100 * _MIB)
    s = led.summary(k)
    assert "NOT TRUSTED" in s and "GiB(p99" not in s


def test_a_cold_key_says_cold(tmp_path: Path) -> None:
    led = _ledger(tmp_path)
    assert led.summary(shape_key(width=1, height=1, batch=1)) == (
        f"ledger[{shape_key(width=1, height=1, batch=1)}]=cold"
    )


def test_one_outlier_cannot_inflate_the_reserve_forever(tmp_path: Path) -> None:
    led = _ledger(tmp_path)
    k = shape_key(width=1024, height=1024, batch=2)
    led.observe_request(k, activation_bytes=8000 * _MIB)
    for _ in range(WINDOW):
        led.observe_request(k, activation_bytes=100 * _MIB)
    p99 = led.stats(k).activation_percentile()
    assert p99 is not None and p99 <= 200 * _MIB, (
        "the outlier is still setting the number after a full window of "
        "ordinary samples"
    )


def test_the_key_separates_shape_and_extras_but_not_steps(tmp_path: Path) -> None:
    assert shape_key(width=1024, height=1024, batch=2) == "1024x1024x2:none:eager"
    assert shape_key(width=1024, height=1024, batch=2, extras="controlnet") != (
        shape_key(width=1024, height=1024, batch=2)
    )
    assert shape_key(width=1024, height=1024, batch=4) != (
        shape_key(width=1024, height=1024, batch=2)
    )
    assert shape_key(width=None, height=1024, batch=2) == "?x1024x2:none:eager"


def test_the_checkpoint_is_in_the_path_so_new_weights_start_cold(
    tmp_path: Path,
) -> None:
    """Staleness handled by construction: weights change -> key changes -> cold ledger -> safe default."""
    a = _ledger(tmp_path, ckpt="digest-aaa")
    b = _ledger(tmp_path, ckpt="digest-bbb")
    assert a.path != b.path
    k = shape_key(width=1024, height=1024, batch=2)
    a.observe_request(k, activation_bytes=500 * _MIB)
    a.flush()
    assert _ledger(tmp_path, ckpt="digest-bbb").summary(k).endswith("=cold")


def test_it_round_trips_and_a_corrupt_ledger_is_a_COLD_ledger(
    tmp_path: Path,
) -> None:
    led = _ledger(tmp_path)
    k = shape_key(width=1024, height=1024, batch=2)
    for _ in range(MIN_SAMPLES_FOR_PERCENTILE):
        led.observe_request(k, activation_bytes=400 * _MIB, retries=3)
    led.observe_placement(k, {"attr_cache_bytes": 123})
    led.close_boot()
    assert led.flush()

    again = _ledger(tmp_path)
    assert again.stats(k).activation_percentile() == 400 * _MIB
    assert again.stats(k).placement_bytes == {"attr_cache_bytes": 123}
    assert again.stats(k).requests_per_boot[-1] == MIN_SAMPLES_FOR_PERCENTILE

    led.path.write_text("{ not json")
    assert _ledger(tmp_path).summary(k).endswith("=cold"), (
        "a corrupt ledger must read as COLD, never raise into a placement"
    )


def test_an_unwritable_root_is_not_an_error(tmp_path: Path) -> None:
    """A ledger that cannot be written is not a failure — it is a cold ledger next time."""
    led = ResidencyLedger("sdxl", "abc", root=tmp_path / "f" / "g")
    led.observe_request(shape_key(width=8, height=8, batch=1), activation_bytes=1)
    (tmp_path / "f").write_text("i am a file, not a directory")
    assert led.flush() is False


def test_flush_is_atomic_and_leaves_no_partial_file(tmp_path: Path) -> None:
    led = _ledger(tmp_path)
    k = shape_key(width=64, height=64, batch=1)
    led.observe_request(k, activation_bytes=7)
    led.flush()
    assert json.loads(led.path.read_text())["keys"][k]["activation_bytes"] == [7]
    assert not list(tmp_path.glob("*.tmp")), "a temp file survived the flush"


def test_phase_one_records_and_decides_nothing(tmp_path: Path) -> None:
    """The safety property of phase 1, asserted rather than asserted-in-prose: no placement code path consults the ledger yet, so landing it cannot change a rung."""
    import subprocess

    src = Path(__file__).resolve().parents[1] / "src" / "gen_worker"
    hits = subprocess.run(
        ["grep", "-rl", "--include=*.py", "residency_ledger", str(src)],
        capture_output=True, text=True,
    ).stdout.split()
    others = [h for h in hits if Path(h).name != "residency_ledger.py"]
    assert others == [], (
        f"these already reference the ledger: {others} — phase 1 is supposed to "
        "record and decide nothing, so landing it cannot change a rung"
    )


def test_eager_and_compiled_samples_never_pool(tmp_path: Path) -> None:
    led = _ledger(tmp_path)
    eager = shape_key(width=1024, height=1024, batch=2, regime="eager")
    compiled = shape_key(width=1024, height=1024, batch=2, regime="compiled")
    assert eager != compiled

    for _ in range(MIN_SAMPLES_FOR_PERCENTILE):
        led.observe_request(eager, activation_bytes=764 * _MIB)

    assert led.stats(eager).activation_percentile() == 764 * _MIB
    assert led.stats(compiled).activation_percentile() is None, (
        "eager samples leaked into the compiled regime — this is the pgw#1548 "
        "daemon death, delivered by a number that looks measured"
    )
    assert led.summary(compiled).endswith("=cold")


def test_an_unrecognised_regime_pools_with_NOTHING_and_stays_cold(
    tmp_path: Path,
) -> None:
    """CAUGHT WHILE JUSTIFYING THE OPPOSITE."""
    unknown = shape_key(width=8, height=8, batch=1, regime="inductor")
    assert unknown != shape_key(width=8, height=8, batch=1, regime="eager")
    assert unknown != shape_key(width=8, height=8, batch=1, regime="compiled")

    led = _ledger(tmp_path)
    for _ in range(MIN_SAMPLES_FOR_PERCENTILE):
        led.observe_request(
            shape_key(width=8, height=8, batch=1, regime="eager"),
            activation_bytes=764 * _MIB,
        )
    assert led.stats(unknown).activation_percentile() is None, (
        "an unrecognised regime inherited another regime's samples"
    )
