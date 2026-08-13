"""pgw#830: every second of a pooled compile has a name, and stays named.

Attempt nine recorded ``compile_s=1331.72`` with five named phases summing to
742.6 s. The other **44 % had no name** — not because a metric was missing,
but because ``compile_s`` measured a CHILD PROCESS while ``phases`` measured
the inside of ``aot_compile``, and nothing ever reconciled the two.

The regression this file exists to prevent is attribution ROT: a phase that
stops being measured, a span that stops being recorded, or a new step added
to the child that nobody names. All three look identical from the outside —
the table still renders, the numbers still add up to something, and the gap
grows silently. So the assertion is not "these labels exist"; it is the
INVARIANT that each partition equals the sum of its members, run against a
real pool driving real ``aot_compile`` children.

Driven for real, deliberately: the pool's process boundary IS the thing being
measured, so a mocked child would test nothing. Small CPU graphs keep it cheap
(the same shape pgw#809's own scenario test uses).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import pytest

from gen_worker import aot_compile_pool as pool
from gen_worker import aot_compile_spans as spans

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

_HIDDEN = 96


def _program(seed: int) -> Any:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Linear(_HIDDEN, _HIDDEN)
            self.b = torch.nn.Linear(_HIDDEN, _HIDDEN)

        def forward(self, x: Any) -> Any:
            y = torch.relu(self.a(x)) * (1.0 + seed)
            return torch.tanh(self.b(y)) + y

    return torch.export.export(Tiny(), (torch.randn(4, _HIDDEN),))


def _compiled_graphs(n: int) -> List[Tuple[str, Any]]:
    return [(f"unet/adapter=true/dim={i}", _program(i)) for i in range(n)]


# ---------------------------------------------------------------------------
# The invariant, on a real pool
# ---------------------------------------------------------------------------


def test_every_second_of_a_pooled_compile_is_attributed(tmp_path: Path) -> None:
    """Three nested partitions reconcile, per compiled graph, on a real pool run.

    This is the test that makes the 44 % un-re-openable. If someone adds a
    step to the compiled graph child and does not name it, ``child_other_s`` grows and
    the residual assertion fires; if someone removes a phase, the partition
    stops summing and :func:`aot_compile_spans.check` fires.
    """
    compiled_graphs = _compiled_graphs(4)
    width = pool.compiled_graph_workers(
        len(compiled_graphs), limit=2, vcpus=16, available_bytes=64 * 1024**3,
        device_lock=True)
    assert width.workers == 2
    box = pool.CompiledGraphCompilePool(
        tmp_path / "pool", width=width,
        inductor_configs={"compile_threads": 2},
        cache_dir=str(tmp_path / "cache"))

    out = box.compile(compiled_graphs)
    assert set(out) == {name for name, _ in compiled_graphs}

    for name, _ in compiled_graphs:
        table = box.compiled_graph_phases[name]
        violations = spans.check(table)
        assert not violations, (
            f"compiled_graph {name!r}: the compile attribution no longer partitions "
            f"its own total — {violations}. Every interval between the "
            f"recorded marks must have a name (pgw#830); a table that does "
            f"not add up is how 44 % went dark on attempt nine.\n"
            f"table={table}")

        # The recorded compile_s must equal the parent's own measurement,
        # not something the child computed about itself.
        assert table["compile_s"] == pytest.approx(
            box.compiled_graph_seconds[name], abs=0.05)

        # Named, not merely present: the spans that were literally invisible
        # before this issue. Each is real work paid once PER COMPILED_GRAPH because the
        # pool's unit of parallelism is a process that exits — 72 times on
        # attempt nine — so a zero here means the measurement broke.
        for label in ("child_boot_s", "child_program_load_s", "child_seal_s",
                      "compile_wall_s"):
            assert table.get(label, 0.0) > 0.0, (
                f"compiled_graph {name!r}: {label} is not being measured. It is inside "
                f"the recorded compile_s whether or not anyone names it")

        # The single largest named non-compile span on attempt nine's shape,
        # and the reason this issue found something: `env_seal.establish()`
        # re-hashes every toolchain .so in the image, per compiled graph child. It must
        # stay broken out — folded back into an anonymous `child_seal_s` it
        # would be indistinguishable from unavoidable process setup.
        seal = box.compiled_graph_overlays[name]
        assert "seal_libhash_s" in seal and "seal_config_s" in seal, (
            f"compiled_graph {name!r}: the seal's own split is gone ({seal}) — "
            f"`child_seal_s` on its own cannot tell a 4 GB SHA-256 pass "
            f"apart from an unavoidable setup cost")
        assert sum(
            v for k, v in seal.items()
            if k.startswith("seal_") and k != "seal_libhash_s"
        ) == pytest.approx(table["child_seal_s"], abs=0.25), (
            f"compiled_graph {name!r}: the seal split no longer covers child_seal_s "
            f"({seal} vs {table['child_seal_s']})")

        # The residuals are what the whole issue is about: they are allowed to
        # exist (an honest table has a remainder) but not to be where the time
        # is. Anything above this is a phase that needs naming, not a rounding
        # error.
        dark = spans.dark_fraction(table)
        assert dark <= 0.15, (
            f"compiled_graph {name!r}: {dark:.1%} of compile_s is in an unnamed "
            f"residual (child_other_s={table.get('child_other_s')}, "
            f"compile_other_s={table.get('compile_other_s')}). pgw#830's "
            f"whole point is that this number stays small and named — go "
            f"look at metrics_raw and give the phase a label.\n"
            f"table={table}")

    # The overlays must NOT be summed into the partition. Asserted rather than
    # commented, because the double-count is invisible in a rendered table.
    for name, _ in compiled_graphs:
        table = box.compiled_graph_phases[name]
        members = spans.PARTITIONS["compile_wall_s"]
        assert sum(table[m] for m in members) == pytest.approx(
            table["compile_wall_s"], abs=0.05)
        assert not (set(table) & set(spans.OVERLAY_KEYS)), (
            "an overlay leaked into the partition dict — overlays nest "
            "inside partition members and adding them double-counts")


# ---------------------------------------------------------------------------
# Pool idle is a SEPARATE line item from serial dark time
# ---------------------------------------------------------------------------


def test_pool_idle_is_accounted_separately_from_compile_seconds(
    tmp_path: Path,
) -> None:
    """``capacity = busy + idle``, and idle splits into named causes.

    These two numbers have OPPOSITE fixes. Serial dark time inside
    ``compile_s`` is compile work on the critical path — the target of
    instrumentation and then optimization. Pool idle is workers with nothing
    to run; it shrinks when the compiled graph count or the straggler spread changes
    (pgw#829) and not at all when the compiler gets faster. A table that adds
    them into one "dark" figure aims the next lane at the wrong term.
    """
    compiled_graphs = _compiled_graphs(4)
    width = pool.compiled_graph_workers(
        len(compiled_graphs), limit=2, vcpus=16, available_bytes=64 * 1024**3,
        device_lock=True)
    box = pool.CompiledGraphCompilePool(
        tmp_path / "pool", width=width,
        inductor_configs={"compile_threads": 2},
        cache_dir=str(tmp_path / "cache"))
    box.compile(compiled_graphs)

    led = box.ledger
    assert led.workers == 2 and led.compiled_graphs == 4
    assert led.capacity_s == pytest.approx(led.wall_s * led.workers, abs=0.01)
    # The identity that makes "75 % pool efficiency" a measured statement
    # rather than a ratio someone divided out afterwards.
    assert led.busy_s + led.idle_s == pytest.approx(led.capacity_s, rel=0.05), (
        f"pool capacity {led.capacity_s:.2f}s != busy {led.busy_s:.2f}s + "
        f"idle {led.idle_s:.2f}s — the idle split no longer covers the run, "
        f"so 'pool efficiency' would be an unexplained remainder")
    assert 0.0 < led.efficiency <= 1.05

    facts = led.facts()
    assert facts["pool_idle_s"] == pytest.approx(
        facts["idle_staging_s"] + facts["idle_drain_s"]
        + facts["idle_spawn_s"] + facts["idle_other_s"], abs=0.01)
    # Staging is parent-serial and really does hold a freed slot: with 4
    # compiled graphs at K=2 there is always a stage between reaps.
    assert facts["stage_total_s"] > 0.0

    # And the crossing claim: pool idle is NOT inside any compiled graph's compile_s.
    busy_from_compiled_graphs = sum(box.compiled_graph_seconds.values())
    assert busy_from_compiled_graphs == pytest.approx(led.busy_s, abs=0.01)


# ---------------------------------------------------------------------------
# The checker itself
# ---------------------------------------------------------------------------


def test_check_names_the_partition_that_broke() -> None:
    """A silent gap must be impossible: the checker says WHICH total stopped
    adding up and by how much, because "attribution is off" with no name is
    the same defect this issue is closing."""
    good = {
        "compile_s": 10.0, "child_boot_s": 1.0, "child_wall_s": 8.5,
        # pgw#1099: the outer partition has its own residual now, so a
        # measured span never doubles as the catch-all. Zero here because the
        # named members already close the level — which is the normal case.
        "reap_lag_s": 0.5, "parent_other_s": 0.0,
        "child_seal_s": 0.1, "child_torch_import_s": 1.5,
        "child_devlock_s": 0.0, "child_program_load_s": 0.4,
        "compile_wall_s": 6.0, "child_other_s": 0.5,
        "lowering_s": 0.5, "codegen_s": 1.0, "graph_passes_s": 0.3,
        "host_compile_s": 3.7, "compile_other_s": 0.5,
    }
    assert spans.check(good) == []
    assert spans.dark_fraction(good) == pytest.approx(0.10)

    broken = dict(good, host_compile_s=1.0)
    problems = spans.check(broken)
    assert problems and "compile_wall_s" in problems[0]
    assert "-2.700" in problems[0] or "+" in problems[0] or "-2.7" in problems[0]

    # A dropped span must not be absorbed in silence.
    dropped = {k: v for k, v in good.items() if k != "child_torch_import_s"}
    problems = spans.check(dropped)
    assert any("child_torch_import_s" in p for p in problems), problems

    # An overlay summed as a member drives the residual negative, and that is
    # reported as double-counting rather than as a small anomaly.
    doubled = dict(good, compile_other_s=-1.0)
    assert any("double-counted" in p for p in spans.check(doubled))
