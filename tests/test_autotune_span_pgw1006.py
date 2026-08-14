"""pgw#1006: the compile-time autotune benchmark is a NAMED overlay.

Why this test exists rather than a shared autotune cache: Triton autotuning is a
SMALL share of a mint. On a real w8a8 SDXL UNet AOTI entry (L4 sm_89,
`aoti_compile_s = 390.3`), `CachingAutotuner.benchmark_all_configs` was **12.1 s
over 96 calls, 3.1 %** — so the whole ceiling of a perfect cross-mint autotune
cache is ~3 % of a mint. Unnamed, that term gets argued about as a large share.

It is named now, in both the pooled ledger (`torch_compiled_graphs.spans`) and
the serial mint's own table (`aot_mint._phase_delta`), as an OVERLAY: the
autotune block runs during `GraphLowering.codegen`
(`autotune_at_compile_time` resolves True for AOTI), so summing it with
`codegen_s` double-counts.

The second reason it is named is the byte-identity finding — the selected
config is baked into the generated wrapper (`grid_0` block size and the
`launchKernel` `num_warps` argument), so a mint whose autotune picks
differently emits different bytes for the same key. `autotune_s` moving between
two mints of one key is the cheapest visible signal that it happened.
"""

from __future__ import annotations

from typing import Dict

from torch_compiled_graphs import spans

from gen_worker import aot_mint


AUTOTUNE_METRIC = "CachingAutotuner.benchmark_all_configs"


def _snapshot(**values: float) -> Dict[str, float]:
    return dict(values)


def test_autotune_is_an_overlay_in_both_ledgers() -> None:
    assert "autotune_s" in spans.OVERLAY_KEYS
    assert AUTOTUNE_METRIC in spans.OVERLAY_KEYS["autotune_s"]
    assert AUTOTUNE_METRIC in aot_mint._AUTOTUNE_KEYS
    # An overlay, never a partition member: the autotune block runs inside
    # codegen, so a reader summing the partition must not see it.
    assert "autotune_s" not in spans.PARTITION_KEYS
    assert "autotune_s" not in aot_mint._PHASE_KEYS
    for members in spans.PARTITIONS.values():
        assert "autotune_s" not in members
    assert "autotune_s" in spans.SUBSPANS


def test_pooled_ledger_reports_autotune_without_inflating_the_partition() -> None:
    before = _snapshot()
    after = _snapshot(**{
        "GraphLowering.run": 10.0,
        "GraphLowering.codegen": 100.0,
        AUTOTUNE_METRIC: 12.1,
        "AotCodeCompiler.compile": 180.0,
    })
    partition, overlays, raw = spans.phase_delta(before, after)

    assert overlays["autotune_s"] == 12.1
    assert partition["codegen_s"] == 100.0, (
        "autotune nests inside codegen — naming it must not shrink the member "
        "it nests in")
    assert sum(partition.values()) == 290.0, (
        "the overlay leaked into the partition and double-counted 12.1 s")
    assert raw[AUTOTUNE_METRIC] == 12.1


def test_serial_mint_table_reports_autotune_separately_from_triton() -> None:
    """The two GPU-touching overlays are different questions.

    `triton_s` is kernel COMPILATION (already fanned out across worker
    processes, 1.5 % of the banked entry); `autotune_s` is BENCHMARKING on the
    live card. Neither autotune key contains "triton" or "async_compile", so
    the two overlays cannot absorb each other — asserted, because the
    substring rule that builds `triton_s` would silently swallow a future
    autotune key that happened to be named for triton.
    """
    out = aot_mint._phase_delta(_snapshot(), _snapshot(**{
        "GraphLowering.codegen": 100.0,
        AUTOTUNE_METRIC: 12.1,
        "async_compile.wait": 5.8,
    }))
    assert out["autotune_s"] == 12.1
    assert out["triton_s"] == 5.8
    assert out["codegen_s"] == 100.0
    for key in aot_mint._AUTOTUNE_KEYS:
        assert "async_compile" not in key and "triton" not in key.lower()


def test_absent_autotune_is_omitted_rather_than_reported_as_zero() -> None:
    """A CPU entry never autotunes, and a recorded 0.0 would read as a
    measurement rather than as an absence."""
    partition, overlays, _ = spans.phase_delta(
        _snapshot(), _snapshot(**{"GraphLowering.codegen": 3.0}))
    assert "autotune_s" not in overlays
    out = aot_mint._phase_delta(
        _snapshot(), _snapshot(**{"GraphLowering.codegen": 3.0}))
    assert "autotune_s" not in out
