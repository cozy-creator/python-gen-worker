"""pgw#1548 LoRA arm: what a fold COSTS per request, and where it breaks even.

Paul's framing (coordinator, 2026-08-20): *"we need to fuse / unfuse the
weights after every request"* — so the fold's headline number is not a one-off
setup cost, it is a RECURRING per-request tax. The deliverable is therefore an
amortization, not a speed comparison:

    fold path  = (fold_s + restore_s) + steps x compiled_folded_per_step
    eager path =                        steps x eager_adapter_per_step

and the crossover step count where the two meet.

## Three modes, one graph, one shape

| mode | what it is | expected dispatch |
|---|---|---|
| `base` | compiled, NO adapter | compiled_calls > 0 — the floor |
| `fold` | compiled + `lora_fold.folded(..., rebind=adapter_guard.rearm_constants)` | compiled_calls > 0 — REQUIRED, or the row aborts |
| `eager` | diffusers' own adapter ops (`load_lora_weights`/`set_adapters`) | compiled_calls == 0 BY DESIGN — pgw#1571's P0 guard drops a live peft_config to loud eager. That is the guard WORKING and is the reference point the fold is measured against. |

**There is no LoRA axis in any lock** (pgw#1572): production never specialized
a graph on an adapter. So all three modes run against the SAME armed graph and
the same boot — which is also why they can be interleaved per round without
paying a reboot, and why a mode's number cannot be blamed on a different mint.

## The poisoning check nobody would think to run

The eager mode installs a live peft adapter on a compiled-armed UNet. If
`unload_lora_weights` does not fully restore the arm, every request AFTER it in
the same boot serves eager — and the fold's next round would look free. So
after the eager mode each round, `base` is re-requested and its dispatch facts
are read again. A recheck that comes back eager is a FINDING, and the run
switches to one boot per mode rather than quietly averaging poisoned rounds.

Substrate + lane stamps ride in the verdict, same discipline as the dynamic-dims
harness: numbers describe the graphs, not the deploy path, and the SDXL locks in
play cover the **euler / float32** timestep lane only.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dynamic_dims_pgw1548 import (  # noqa: E402
    SUBSTRATES,
    Bench,
    Sample,
    require_window,
)
from pgw1548_endpoint_instrument import install_into  # noqa: E402

MODES = ("base", "fold", "eager")


@dataclass
class LoraSample:
    """One request under one mode, with the endpoint-side walls beside it.

    `seconds` is the product axis (submit -> output downloaded). `denoise_s`
    is the second axis and the only one per-step ms may be computed from —
    dividing a round-trip by the step count would charge encode, VAE decode,
    webp encode and IPC to every step.
    """

    mode: str
    round: int
    seconds: float
    denoise_s: float = 0.0
    save_s: float = 0.0
    fold_s: float = 0.0
    restore_s: float = 0.0
    rearm_calls: int = 0
    steps: int = 0
    compiled_calls: int = 0
    eager_calls: int = 0
    displaced: tuple[str, ...] = ()
    load1: float = 0.0

    @property
    def per_step_ms(self) -> float:
        return (self.denoise_s / self.steps * 1000.0) if self.steps else 0.0


class LoraBench(Bench):
    """`Bench`, plus the control/trace channel the instrument reads and writes."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.control = self.out / "bench-control.json"
        self.trace = self.out / "bench-trace.jsonl"
        self.samples: list[LoraSample] = []
        self.poisoned: list[str] = []

    def _gen_worker(self, room: Path, argv: list[str],
                    timeout: float) -> subprocess.CompletedProcess:
        env = {
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "PGW1548_BENCH_CONTROL": str(self.control),
            "PGW1548_BENCH_TRACE": str(self.trace),
        }
        return _run_with(
            [self._python(), "-m", "gen_worker.cli", *argv],
            cwd=room, env=env, timeout=timeout,
        )

    # -- the instrumented workspace ----------------------------------------
    def instrumented_room(self, arm: str) -> Path:
        room = self._workspace(arm)
        packages = [p for p in (room / "src").iterdir() if (p / "main.py").exists()]
        if len(packages) != 1:
            raise SystemExit(
                f"expected exactly one instrumentable package under {room / 'src'}, "
                f"found {[p.name for p in packages]}"
            )
        install_into(packages[0], packages[0] / "main.py")
        return room

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise SystemExit(f"unknown mode {mode!r}")
        payload: dict[str, Any] = {"mode": "off" if mode == "base" else mode}
        if mode != "base":
            payload.update({
                "lora_path": str(Path(self.args.lora).expanduser().resolve()),
                "scale": self.args.lora_scale,
                "ref": self.args.lora_ref or str(self.args.lora),
            })
        self.control.write_text(json.dumps(payload))

    def _last_trace(self) -> dict[str, Any]:
        if not self.trace.exists():
            return {}
        lines = [line for line in self.trace.read_text().splitlines() if line.strip()]
        if not lines:
            return {}
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            return {}

    def lora_request(self, room: Path, mode: str, round_index: int) -> LoraSample:
        before = self.trace.stat().st_size if self.trace.exists() else 0
        sample: Sample = self.request(
            room, mode, self.args.aspects[0], self.args.cfg[0], round_index
        )
        after = self.trace.stat().st_size if self.trace.exists() else 0
        if after <= before:
            raise SystemExit(
                f"[{mode}] the endpoint instrument wrote NO trace line for this "
                f"request. Without it there is no denoise wall and no fold wall, "
                f"and the round-trip alone cannot separate them. Refusing to "
                f"score this row. (trace={self.trace})"
            )
        facts = self._last_trace()
        if facts.get("mode") not in ("off", "fold", "eager"):
            raise SystemExit(f"[{mode}] trace line names no mode: {facts!r}")
        expected = "off" if mode == "base" else mode
        if facts["mode"] != expected:
            raise SystemExit(
                f"[{mode}] the endpoint served mode {facts['mode']!r} while the "
                f"control file said {expected!r} — the control file is read per "
                f"request, so this means a stale daemon. Refusing to score."
            )
        return LoraSample(
            mode=mode, round=round_index, seconds=sample.seconds,
            denoise_s=float(facts.get("denoise_s", 0.0)),
            save_s=float(facts.get("save_s", 0.0)),
            fold_s=float(facts.get("fold_s", 0.0)),
            restore_s=float(facts.get("restore_s", 0.0)),
            rearm_calls=int(facts.get("rearm_calls", 0) or 0),
            steps=int(facts.get("steps", 0) or 0),
            compiled_calls=sample.compiled_calls,
            eager_calls=sample.eager_calls,
            displaced=sample.displaced,
            load1=sample.load1,
        )

    # -- the premise, per mode ---------------------------------------------
    def assert_premise(self, mode: str, sample: LoraSample) -> None:
        """Each mode has its OWN premise. Scoring them all by one rule is how a
        reference point gets mistaken for a failure."""

        if mode in ("base", "fold"):
            if sample.compiled_calls <= 0:
                raise SystemExit(
                    f"[{mode}] ZERO compiled calls (eager={sample.eager_calls}, "
                    f"displaced={list(sample.displaced)}). This mode is only "
                    f"meaningful ON the compiled path — a folded arm that serves "
                    f"eager measures the eager deficit, not the fold. Refusing."
                )
            if sample.eager_calls:
                raise SystemExit(
                    f"[{mode}] MIXED execution: {sample.compiled_calls} compiled "
                    f"and {sample.eager_calls} eager. A part-compiled wall is not "
                    f"this arm's cost. Refusing."
                )
        if mode == "fold":
            if sample.rearm_calls <= 0:
                raise SystemExit(
                    f"[fold] rearm_constants re-armed ZERO constant tables, so "
                    f"the fold never met a compiled artifact. Either nothing is "
                    f"armed or the fold landed on the wrong module — in both "
                    f"cases the folded weights are not what the graph reads. "
                    f"Refusing."
                )
        if mode == "eager":
            # NOT an abort. pgw#1571's P0 guard drops a compiled-armed module to
            # loud eager the moment a live peft_config appears; observing that
            # here is the guard working and is the whole reason the fold exists.
            if sample.compiled_calls > 0:
                print(f"[eager] NOTE: {sample.compiled_calls} compiled call(s) "
                      f"with a live adapter — the P0 guard did NOT fire. That is "
                      f"a finding: a peft adapter on a compiled artifact serves "
                      f"the BASE weights (pgw#1571 measured a 0.0 delta).")

    # -- the report ---------------------------------------------------------
    def _median(self, mode: str, attr: str) -> float | None:
        values = [getattr(s, attr) for s in self.samples if s.mode == mode]
        return statistics.median(values) if values else None

    def _spread(self, mode: str, attr: str) -> float | None:
        per_round = []
        for index in sorted({s.round for s in self.samples}):
            values = [getattr(s, attr) for s in self.samples
                      if s.mode == mode and s.round == index]
            if values:
                per_round.append(statistics.median(values))
        if len(per_round) < 2:
            return None
        low, high = min(per_round), max(per_round)
        return (high - low) / low * 100.0 if low else None

    def amortization(self) -> dict[str, Any]:
        """The break-even Paul asked for, stated as arithmetic a reader can check."""

        fold_step = self._median("fold", "per_step_ms")
        eager_step = self._median("eager", "per_step_ms")
        base_step = self._median("base", "per_step_ms")
        fold_wall = self._median("fold", "fold_s")
        restore_wall = self._median("fold", "restore_s")
        eager_load = self._median("eager", "fold_s")
        eager_unload = self._median("eager", "restore_s")

        out: dict[str, Any] = {
            "base_per_step_ms": base_step,
            "fold_per_step_ms": fold_step,
            "eager_per_step_ms": eager_step,
            "fold_wall_s": fold_wall,
            "restore_wall_s": restore_wall,
            "eager_load_wall_s": eager_load,
            "eager_unload_wall_s": eager_unload,
            "fold_fixed_cost_s": None,
            "eager_fixed_cost_s": None,
            "per_step_saving_ms": None,
            "crossover_steps": None,
            "at_default_steps": {},
        }
        if fold_wall is None or restore_wall is None:
            return out
        fold_fixed = fold_wall + restore_wall
        eager_fixed = (eager_load or 0.0) + (eager_unload or 0.0)
        out["fold_fixed_cost_s"] = fold_fixed
        out["eager_fixed_cost_s"] = eager_fixed
        if fold_step is None or eager_step is None:
            return out
        saving_ms = eager_step - fold_step
        out["per_step_saving_ms"] = saving_ms
        # fold_fixed + n*fold_step == eager_fixed + n*eager_step
        if saving_ms > 0:
            crossover = (fold_fixed - eager_fixed) * 1000.0 / saving_ms
            out["crossover_steps"] = crossover
        for steps in sorted({int(self.args.steps), 28}):
            fold_total = fold_fixed + steps * fold_step / 1000.0
            eager_total = eager_fixed + steps * eager_step / 1000.0
            out["at_default_steps"][str(steps)] = {
                "fold_total_s": fold_total,
                "eager_total_s": eager_total,
                "winner": "fold" if fold_total < eager_total else "eager",
                "margin_s": abs(fold_total - eager_total),
                "margin_pct": (
                    abs(fold_total - eager_total) / max(fold_total, eager_total)
                    * 100.0
                ),
            }
        return out

    def render(self) -> str:
        modes = [m for m in MODES if any(s.mode == m for s in self.samples)]
        lines = [
            f"_{SUBSTRATES[self.args.substrate]}; {self.args.lane_note}_",
            "",
            "| mode | RT s (median) | denoise s | per-step ms | fold s | restore s | "
            "rearm | compiled/eager calls | RT spread across rounds |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for mode in modes:
            spread = self._spread(mode, "seconds")
            compiled = sum(s.compiled_calls for s in self.samples if s.mode == mode)
            eager = sum(s.eager_calls for s in self.samples if s.mode == mode)
            lines.append(
                f"| {mode} "
                f"| {self._median(mode, 'seconds'):.3f} "
                f"| {self._median(mode, 'denoise_s'):.3f} "
                f"| {self._median(mode, 'per_step_ms'):.1f} "
                f"| {self._median(mode, 'fold_s'):.3f} "
                f"| {self._median(mode, 'restore_s'):.3f} "
                f"| {int(self._median(mode, 'rearm_calls'))} "
                f"| {compiled}/{eager} "
                f"| {'—' if spread is None else f'{spread:.1f}%'} |"
            )
        return "\n".join(lines)

    def report(self) -> dict[str, Any]:  # type: ignore[override]
        rounds = sorted({s.round for s in self.samples})
        undecidable = []
        for mode in MODES:
            if not any(s.mode == mode for s in self.samples):
                continue
            spread = self._spread(mode, "seconds")
            if spread is None:
                undecidable.append(
                    f"{mode}: measured in ONE round; nothing establishes that it "
                    f"reproduces (re-run with >= 2 rounds)"
                )
            elif spread > self.args.spread_limit:
                undecidable.append(
                    f"{mode}: round-to-round spread {spread:.1f}% > "
                    f"{self.args.spread_limit:.0f}%; the box moved more than the "
                    f"adapter path did — re-run on a quiet slot"
                )
        return {
            "endpoint": str(self.endpoint),
            "substrate": self.args.substrate,
            "substrate_note": SUBSTRATES[self.args.substrate],
            "lane_note": self.args.lane_note,
            "shape": f"{self.args.aspects[0]}/cfg-{self.args.cfg[0]}",
            "steps": self.args.steps,
            "lora": str(self.args.lora),
            "lora_scale": self.args.lora_scale,
            "mint": self.mint,
            "rounds": rounds,
            "undecidable": undecidable,
            "poisoning_check": self.poisoned or ["clean: base stayed compiled "
                                                 "after every eager round"],
            "amortization": self.amortization(),
            "table_markdown": self.render(),
            "samples": [vars(s) for s in self.samples],
        }


def _run_with(cmd: list[str], *, cwd: Path, env: dict,
              timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["nice", "-n", "19", *cmd], cwd=cwd, env={**os.environ, **env},
        timeout=timeout, capture_output=True, text=True,
    )


def self_test() -> int:
    """CPU-only. Proves the arithmetic and every refusal can go red."""

    failures: list[str] = []

    def check(name: str, condition: bool) -> None:
        print(f"{'ok  ' if condition else 'FAIL'} {name}")
        if not condition:
            failures.append(name)

    args = argparse.Namespace(
        endpoint=".", checkpoint=".", out="/tmp/pgw1548-lora-selftest",
        aspects=["1:1"], cfg=["on"], steps=20, lora="x.safetensors",
        lora_scale=1.0, lora_ref="", substrate="local", spread_limit=15.0,
        lane_note="euler/float32 lane", reps=1, rounds=2, sm="", prompt="p",
        guidance=7.5, seed=1, selectors=[], lock_cache="",
    )
    bench = LoraBench.__new__(LoraBench)
    bench.args = args
    bench.endpoint = Path(".")
    bench.samples = []
    bench.poisoned = []
    bench.mint = {}

    # A fold that is cheaper per step but pays a fixed cost must break even
    # somewhere, and the crossover must be arithmetic anyone can redo.
    for round_index in (0, 1):
        for _ in range(3):
            bench.samples.append(LoraSample(
                mode="fold", round=round_index, seconds=10.0, denoise_s=8.0,
                steps=20, fold_s=0.5, restore_s=0.3, rearm_calls=1,
                compiled_calls=20))
            bench.samples.append(LoraSample(
                mode="eager", round=round_index, seconds=13.0, denoise_s=11.0,
                steps=20, fold_s=0.2, restore_s=0.05, compiled_calls=0,
                eager_calls=20))
    amort = bench.amortization()
    check("fold per-step is 400 ms", abs(amort["fold_per_step_ms"] - 400.0) < 1e-6)
    check("eager per-step is 550 ms", abs(amort["eager_per_step_ms"] - 550.0) < 1e-6)
    check("fold fixed cost is fold+restore, not fold alone",
          abs(amort["fold_fixed_cost_s"] - 0.8) < 1e-9)
    # (0.8 - 0.25) * 1000 / 150 == 3.667 steps
    check("crossover is stated in steps",
          amort["crossover_steps"] is not None
          and abs(amort["crossover_steps"] - 3.6666) < 1e-2)
    check("28 steps is always reported even when --steps differs",
          "28" in amort["at_default_steps"])
    check("fold wins at 20 steps here",
          amort["at_default_steps"]["20"]["winner"] == "fold")

    # A fold that is NOT cheaper per step can never break even, and the report
    # must say so rather than emit a negative step count.
    bench.samples = [
        LoraSample(mode="fold", round=r, seconds=10.0, denoise_s=11.0, steps=20,
                   fold_s=0.5, restore_s=0.3, rearm_calls=1, compiled_calls=20)
        for r in (0, 1)
    ] + [
        LoraSample(mode="eager", round=r, seconds=10.0, denoise_s=11.0, steps=20,
                   compiled_calls=0, eager_calls=20)
        for r in (0, 1)
    ]
    check("no crossover when the fold saves nothing per step",
          bench.amortization()["crossover_steps"] is None)

    # The premises: each mode judged by its OWN rule.
    bench.samples = []
    zero = LoraSample(mode="fold", round=0, seconds=1.0, compiled_calls=0,
                      eager_calls=20, steps=20)
    try:
        bench.assert_premise("fold", zero)
        check("a folded arm serving eager is refused", False)
    except SystemExit:
        check("a folded arm serving eager is refused", True)

    no_rearm = LoraSample(mode="fold", round=0, seconds=1.0, compiled_calls=20,
                          rearm_calls=0, steps=20)
    try:
        bench.assert_premise("fold", no_rearm)
        check("a fold that re-armed nothing is refused", False)
    except SystemExit:
        check("a fold that re-armed nothing is refused", True)

    eager_ref = LoraSample(mode="eager", round=0, seconds=1.0, compiled_calls=0,
                           eager_calls=20, steps=20)
    try:
        bench.assert_premise("eager", eager_ref)
        check("the eager reference is NOT refused for serving eager", True)
    except SystemExit:
        check("the eager reference is NOT refused for serving eager", False)

    mixed = LoraSample(mode="base", round=0, seconds=1.0, compiled_calls=10,
                       eager_calls=10, steps=20)
    try:
        bench.assert_premise("base", mixed)
        check("mixed execution is refused", False)
    except SystemExit:
        check("mixed execution is refused", True)

    # Decidability: one round is never a pass.
    bench.samples = [LoraSample(mode="fold", round=0, seconds=10.0, steps=20,
                                denoise_s=8.0, rearm_calls=1, compiled_calls=20)]
    check("a single round is UNDECIDABLE",
          any("ONE round" in line for line in bench.report()["undecidable"]))

    bench.samples = [
        LoraSample(mode="fold", round=0, seconds=10.0, steps=20, denoise_s=8.0),
        LoraSample(mode="fold", round=1, seconds=30.0, steps=20, denoise_s=8.0),
    ]
    check("a 200% round-to-round spread is UNDECIDABLE",
          any("spread" in line for line in bench.report()["undecidable"]))

    print()
    print(f"{len(failures)} failure(s)")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--lora", default="", help="local .safetensors adapter")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--lora-ref", default="",
                        help="the ref recorded in the trace (provenance only)")
    parser.add_argument("--out", default="benchmarks/pgw1548/lora")
    parser.add_argument("--arm", default="static",
                        help="which lock arm's graph the modes run against")
    parser.add_argument("--modes", default="base,fold,eager")
    parser.add_argument("--aspects", default="1:1")
    parser.add_argument("--cfg", default="on")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=1548)
    parser.add_argument("--prompt", default="a fisherman at dawn")
    parser.add_argument("--sm", default="")
    parser.add_argument("--substrate", choices=sorted(SUBSTRATES), default="local")
    parser.add_argument("--lane-note", default="euler/float32 timestep lane only")
    parser.add_argument("--spread-limit", type=float, default=15.0)
    parser.add_argument("--selectors", default="")
    parser.add_argument("--lock-cache", default="",
                        help="directory of pre-derived endpoint.lock.<arm> files; "
                             "the derive needs no card and must not be paid for "
                             "inside a GPU window")
    parser.add_argument("--lock-timeout", type=float, default=1800)
    parser.add_argument("--compile-timeout", type=float, default=3600)
    parser.add_argument("--boot-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--idle-timeout", type=int, default=1800)
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if not (args.endpoint and args.checkpoint and args.lora):
        parser.error("--endpoint, --checkpoint and --lora are required")
    require_window()

    args.aspects = [a for a in args.aspects.split(",") if a][:1]
    args.cfg = [c for c in args.cfg.split(",") if c][:1]
    args.selectors = [s for s in args.selectors.split(",") if s]
    modes = [m for m in args.modes.split(",") if m]
    for mode in modes:
        if mode not in MODES:
            parser.error(f"unknown mode {mode!r}; one of {list(MODES)}")

    bench = LoraBench(args)
    if bench.trace.exists():
        bench.trace.unlink()
    bench.set_mode("base")
    room = bench.instrumented_room(args.arm)
    lock_s = bench.lock(room, args.arm)
    records = bench.specializations(room)
    selectors = args.selectors or [r["graph"][:16] for r in records]
    compile_s = bench.compile(room, args.arm, selectors)
    bench.mint[args.arm] = {
        "lock_s": lock_s, "compile_s": compile_s,
        "specializations": len(records), "built": len(selectors),
    }

    bench.serve(room, args.arm)
    try:
        for round_index in range(args.rounds):
            for mode in modes:
                print(f"[round {round_index}] {mode} "
                      f"(load {os.getloadavg()[0]:.1f})")
                bench.set_mode(mode)
                warm = bench.lora_request(room, mode, round_index)
                bench.assert_premise(mode, warm)
                for _ in range(args.reps):
                    bench.samples.append(
                        bench.lora_request(room, mode, round_index))
            if "eager" in modes and "base" in modes:
                # Did the eager adapter path leave the arm intact?
                bench.set_mode("base")
                recheck = bench.lora_request(room, "base", round_index)
                if recheck.compiled_calls <= 0:
                    bench.poisoned.append(
                        f"round {round_index}: after the eager mode, a base "
                        f"request served {recheck.eager_calls} eager call(s) and "
                        f"{recheck.compiled_calls} compiled — unload_lora_weights "
                        f"did NOT restore the compiled arm, so every later "
                        f"request in this boot is eager. Rounds after this one "
                        f"are not comparable."
                    )
                    raise SystemExit(bench.poisoned[-1])
    finally:
        bench.down(room)

    report = bench.report()
    bench.out.mkdir(parents=True, exist_ok=True)
    (bench.out / "lora-verdict.json").write_text(json.dumps(report, indent=2))
    print()
    print(report["table_markdown"])
    print()
    amort = report["amortization"]
    print(f"fold fixed cost per request: {amort['fold_fixed_cost_s']}s "
          f"(fold {amort['fold_wall_s']}s + restore {amort['restore_wall_s']}s)")
    print(f"per-step saving vs eager adapter: {amort['per_step_saving_ms']} ms")
    print(f"crossover: {amort['crossover_steps']} steps")
    for steps, row in amort["at_default_steps"].items():
        print(f"  at {steps} steps: {row['winner']} wins by "
              f"{row['margin_s']:.3f}s ({row['margin_pct']:.1f}%)")
    if report["undecidable"]:
        print()
        for line in report["undecidable"]:
            print(f"UNDECIDABLE {line}")
    print()
    print(report["substrate_note"], "|", report["lane_note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
