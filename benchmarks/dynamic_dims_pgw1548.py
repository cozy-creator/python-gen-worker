"""pgw#1548 acceptance: dynamic dims vs the static bucket fan, per shape.

Paul's gate, verbatim in shape: *"For SDXL, Anima, and SD1.5, do 'aspect ratio
is dynamic dim' versus 'static dim, with many graph specializations' and then
compile them and test inference time"*, plus a SEPARATE cost for the CFG axis
(*"I doubt CFG can be a dynamic dim, but it's worth investigating how much time
we lose if it is"*). Adoption is per axis and per model, and only where the
regression is within a few percent.

**The production serve path, not a rig.** Every number is a round-trip through
`gen-worker up` + `gen-worker run` — the same two commands a cozy-local user
types — because a hand-built pipeline measures a codepath nobody serves. The
harness never imports diffusers itself.

**Reported per shape, never averaged.** An average over aspect buckets is the
one summary that can hide the exact thing this gate is looking for: dynamic
dims cost differently at different sizes, and a mean of a win and a loss reads
as "no change".

## The arms

| arm | `--dynamic-axes` | what it is |
|---|---|---|
| `static` | `off` | the control — one graph per observed shape, today's fleet |
| `aspect` | `aspect` | Paul's headline question |
| `batch` | `batch` | the CFG axis, costed on its own |
| `all` | `all` | both, the best case for mint cost |

## Discipline

* `VARENA_GPU_WINDOW=1` gates every card-touching step. Set it ONLY on a
  granted window.
* `--self-test` is CPU-ONLY, needs no window and no card. It red-arms the
  window gate, the per-shape table (a table that cannot show a regression is
  not an instrument) and the arm plan.
* Compile is bounded: the static arm builds ONLY the specializations this run
  actually benchmarks (`--first <selector> --fill none`), never all 14/18.
  Compiling buckets nobody measures spends the card on nothing.
* `nice -n 19` on every child; one heavy child at a time.

    .venv/bin/python benchmarks/dynamic_dims_pgw1548.py --self-test
    VARENA_GPU_WINDOW=1 .venv/bin/python benchmarks/dynamic_dims_pgw1548.py \
        --endpoint ~/cozy/serverless-endpoints/sd15 --checkpoint <tree> \
        --arms static,aspect --reps 5 --out benchmarks/pgw1548/sd15
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: What substrate produced these numbers, stamped into the verdict and the
#: table so a row CANNOT be published without it (coordinator, 2026-08-20).
#:
#: A raw pod runs `gen-worker lock/compile/up/run` — the endpoint's own serving
#: code, which is what the measurement law asks for — but it does NOT go
#: through release packaging and deploy. That is the right subject for a
#: graph-shape A/B (packaging does not touch a per-step wall) and it is a real
#: limit on what the number describes, so the limit travels WITH the number
#: rather than in a paragraph someone has to remember to quote.
SUBSTRATES = {
    "raw-pod": (
        "raw-pod substrate; numbers describe the graphs, not the deploy path"
    ),
    "local": (
        "local-card substrate; numbers describe the graphs, not the deploy path"
    ),
    "release": "release-built substrate; the full deploy path",
}

ARMS = ("static", "aspect", "batch", "all")
AXES = {"static": "off", "aspect": "aspect", "batch": "batch", "all": "all"}


class WindowRequired(SystemExit):
    """The card is gated by a granted window, not by an env var we set."""


def require_window() -> None:
    if os.environ.get("VARENA_GPU_WINDOW") != "1":
        raise WindowRequired(
            "REFUSING: VARENA_GPU_WINDOW=1 is not set. This box runs several "
            "agent sessions and one GPU lane at a time; the window is GRANTED "
            "by the coordinator and set only then. `--self-test` runs without "
            "one."
        )


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None,
         timeout: float | None = None) -> subprocess.CompletedProcess:
    full = ["nice", "-n", "19", *cmd]
    return subprocess.run(
        full, cwd=cwd, env={**os.environ, **(env or {})}, timeout=timeout,
        capture_output=True, text=True,
    )


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    """One request's round-trip, stamped with WHEN and under WHAT load.

    The load stamp is not decoration: this box runs several agent sessions and
    a peer's CPU job moves a wall by more than a dynamic dim ever will. A
    number without the load it was taken under cannot be compared to one taken
    an hour later, and `round` is what makes two arms comparable at all.
    """

    arm: str
    aspect: str
    cfg: str
    seconds: float
    round: int = 0
    load1: float = 0.0
    #: THE MEASURED dispatch facts for THIS request, read off the response
    #: envelope (`gen-worker run --json` carries `dispatch`). Not inferred from
    #: a log string, and not a proxy: pgw#1591 was an instrument that INFERRED
    #: "all N ran eager" from a flag instead of reading the counter beside it.
    compiled_calls: int = 0
    eager_calls: int = 0
    displaced: tuple[str, ...] = ()


@dataclass
class Table:
    """Per-shape round-trips, and the comparison the gate is decided on.

    The unit of judgement is one (aspect, cfg) CELL: `static` is the control
    and every other arm is a percentage against it, IN THAT CELL. Nothing here
    ever averages across cells — see the module docstring.
    """

    samples: list[Sample] = field(default_factory=list)

    def add(self, sample: Sample) -> None:
        self.samples.append(sample)

    def cell(self, arm: str, aspect: str, cfg: str) -> list[float]:
        return [
            s.seconds for s in self.samples
            if s.arm == arm and s.aspect == aspect and s.cfg == cfg
        ]

    def median(self, arm: str, aspect: str, cfg: str) -> float | None:
        values = self.cell(arm, aspect, cfg)
        return statistics.median(values) if values else None

    def shapes(self) -> list[tuple[str, str]]:
        seen: list[tuple[str, str]] = []
        for sample in self.samples:
            key = (sample.aspect, sample.cfg)
            if key not in seen:
                seen.append(key)
        return seen

    def arms(self) -> list[str]:
        return [arm for arm in ARMS if any(s.arm == arm for s in self.samples)]

    def rounds(self) -> list[int]:
        return sorted({s.round for s in self.samples})

    def spread(self, arm: str, aspect: str, cfg: str) -> float | None:
        """The CONTROL's own round-to-round spread, as a percentage.

        The decidability test the coordinator set (2026-08-20): a cell whose
        reference arm cannot reproduce itself across rounds within 15% is
        measuring the box, not the graph, and no verdict may be read from it.
        """

        per_round = []
        for index in self.rounds():
            values = [
                s.seconds for s in self.samples
                if s.arm == arm and s.aspect == aspect and s.cfg == cfg
                and s.round == index
            ]
            if values:
                per_round.append(statistics.median(values))
        if len(per_round) < 2:
            return None
        low, high = min(per_round), max(per_round)
        return (high - low) / low * 100.0 if low else None

    def undecidable(self, aspect: str, cfg: str, limit: float = 15.0) -> bool:
        """Is this cell's verdict unreadable — for EITHER reason?

        Two ways, and the second one used to read as decidable, which is the
        silent-vacuity shape this file exists to avoid: a cell measured in
        only ONE round has no reproducibility evidence at all, so `spread`
        answers None, and "no evidence" must never score as "no problem". A
        run cut short to fit a window is exactly when that happens, so the
        absence is treated as undecidable rather than as a pass.
        """

        spread = self.spread("static", aspect, cfg)
        if spread is None:
            return True
        return spread > limit

    def regression(self, arm: str, aspect: str, cfg: str) -> float | None:
        """Percent SLOWER than the static control in this cell. Negative = faster."""

        control = self.median("static", aspect, cfg)
        measured = self.median(arm, aspect, cfg)
        if not control or measured is None:
            return None
        return (measured - control) / control * 100.0

    def render(self, substrate: str = "") -> str:
        arms = self.arms()
        lines = [f"_{SUBSTRATES[substrate]}_", ""] if substrate else []
        lines += [
            "| shape | cfg | " + " | ".join(
                f"{arm} (s)" + ("" if arm == "static" else " / vs static")
                for arm in arms
            ) + " |",
            "|" + "---|" * (2 + len(arms)),
        ]
        for aspect, cfg in self.shapes():
            cells = []
            for arm in arms:
                median = self.median(arm, aspect, cfg)
                if median is None:
                    cells.append("—")
                    continue
                if arm == "static":
                    cells.append(f"{median:.3f}")
                    continue
                delta = self.regression(arm, aspect, cfg)
                cells.append(
                    f"{median:.3f} / {delta:+.1f}%" if delta is not None
                    else f"{median:.3f}"
                )
            lines.append(f"| {aspect} | {cfg} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def verdict(self, arm: str, tolerance: float) -> tuple[bool, list[str]]:
        """Adopt this axis? Only if EVERY decidable cell is inside tolerance.

        One bad bucket is a reason not to adopt an axis, not a number to
        average away — a request that lands in it pays the regression every
        time, forever. An UNDECIDABLE cell blocks adoption too: it is not a
        pass, it is a measurement that has to be re-run on a quiet box.
        """

        offenders = []
        for aspect, cfg in self.shapes():
            if self.undecidable(aspect, cfg):
                spread = self.spread("static", aspect, cfg)
                why = (
                    f"control spread {spread:.1f}% > 15%; re-run on a quiet slot"
                    if spread is not None
                    else "measured in ONE round, so nothing establishes that "
                         "the control reproduces; re-run with >= 2 rounds"
                )
                offenders.append(f"{aspect}/{cfg}: UNDECIDABLE ({why})")
                continue
            delta = self.regression(arm, aspect, cfg)
            if delta is None:
                offenders.append(f"{aspect}/{cfg}: NOT MEASURED")
            elif delta > tolerance:
                offenders.append(f"{aspect}/{cfg}: {delta:+.1f}%")
        return (not offenders), offenders

    def as_dict(self) -> dict[str, Any]:
        return {
            "samples": [vars(s) for s in self.samples],
            # The premise, carried BESIDE the timings so a reader cannot take
            # the numbers without the evidence that they measured compiled
            # serving at all.
            "dispatch": {
                f"{arm}": {
                    "compiled_calls": sum(
                        s.compiled_calls for s in self.samples if s.arm == arm
                    ),
                    "eager_calls": sum(
                        s.eager_calls for s in self.samples if s.arm == arm
                    ),
                }
                for arm in self.arms()
            },
            "rounds": self.rounds(),
            "control_spread_pct": {
                f"{aspect}/{cfg}": self.spread("static", aspect, cfg)
                for aspect, cfg in self.shapes()
            },
            "undecidable": [
                f"{aspect}/{cfg}" for aspect, cfg in self.shapes()
                if self.undecidable(aspect, cfg)
            ],
            "cells": {
                f"{aspect}/{cfg}": {
                    arm: {
                        "median_s": self.median(arm, aspect, cfg),
                        "runs_s": self.cell(arm, aspect, cfg),
                        "vs_static_pct": self.regression(arm, aspect, cfg),
                    }
                    for arm in self.arms()
                }
                for aspect, cfg in self.shapes()
            },
        }


# ---------------------------------------------------------------------------
# The arms, on the card
# ---------------------------------------------------------------------------


class Bench:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.endpoint = Path(args.endpoint).expanduser().resolve()
        self.out = Path(args.out).expanduser().resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.table = Table()
        self.mint: dict[str, dict[str, Any]] = {}
        #: arm -> the SERVING daemon's log (not the launcher's output).
        self._daemon_log: dict[str, Path | None] = {}

    # -- the endpoint copy: never mutate a shared checkout ------------------
    def _workspace(self, arm: str) -> Path:
        """A per-arm COPY of the endpoint.

        `gen-worker up` reads the endpoint's own `endpoint.lock`, so an arm has
        to write one — and `~/cozy/serverless-endpoints` is a shared checkout
        with a sibling lane working in it. The copy is the isolation; the
        `.venv` is symlinked because it is the environment, not the source.
        """

        room = self.out / f"arm-{arm}"
        if room.exists():
            shutil.rmtree(room)
        room.mkdir(parents=True)
        for name in ("src", "endpoint.toml", "pyproject.toml", "uv.lock",
                     "endpoint.lock", "cozy.toml"):
            source = self.endpoint / name
            if not source.exists():
                continue
            if source.is_dir():
                shutil.copytree(source, room / name)
            else:
                shutil.copy2(source, room / name)
        venv = self.endpoint / ".venv"
        if venv.exists():
            (room / ".venv").symlink_to(venv)
        return room

    def _python(self) -> str:
        venv = self.endpoint / ".venv" / "bin" / "python"
        return str(venv) if venv.exists() else sys.executable

    def _gen_worker(self, room: Path, argv: list[str], timeout: float) -> subprocess.CompletedProcess:
        """The CLI, running THIS branch's source (PYTHONPATH wins over the pin)."""

        return _run(
            [self._python(), "-m", "gen_worker.cli", *argv],
            cwd=room,
            env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
            timeout=timeout,
        )

    # -- one arm ------------------------------------------------------------
    def lock(self, room: Path, arm: str) -> float:
        started = time.monotonic()
        result = self._gen_worker(
            room,
            ["lock", str(room), "--force", "--dynamic-axes", AXES[arm],
             "--checkpoint", str(self.args.checkpoint)],
            timeout=self.args.lock_timeout,
        )
        if result.returncode != 0:
            raise SystemExit(f"[{arm}] lock failed:\n{result.stdout}\n{result.stderr}")
        return time.monotonic() - started

    def specializations(self, room: Path) -> list[dict[str, Any]]:
        """Every graph specialization this arm's lock declares.

        `read_lock` answers a plain dict (NOT an object with attributes) and
        the derive document sits under `["derive"]["document"]` as a JSON
        STRING. Both facts are asserted here rather than assumed: an attribute
        read that works on nothing would have raised on the pod, minutes into
        a paid window, after the compile it was supposed to plan.
        """

        from gen_worker.cli import endpoint_lock as el

        block = el.read_lock(room / el.LOCK_FILENAME)
        derive = block.get("derive") if isinstance(block, dict) else None
        if not isinstance(derive, dict) or "document" not in derive:
            raise SystemExit(
                f"{room / el.LOCK_FILENAME}: no [derive] document — a lock "
                f"written with --discovery-only has no graphs to benchmark"
            )
        raw = derive["document"]
        document = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        records = [
            record
            for lane in document["graphs"]["lanes"]
            for record in lane["graphs"]
        ]
        if not records:
            raise SystemExit(f"{room}: the lock declares zero specializations")
        return records

    def compile(self, room: Path, arm: str, selectors: list[str]) -> float:
        """Build ONLY what this run benchmarks. One selector per child."""

        require_window()
        started = time.monotonic()
        for selector in selectors:
            result = self._gen_worker(
                room,
                ["compile", str(room), "--first", selector, "--fill", "none"],
                timeout=self.args.compile_timeout,
            )
            if result.returncode != 0:
                raise SystemExit(
                    f"[{arm}] compile {selector!r} failed:\n"
                    f"{result.stdout}\n{result.stderr}"
                )
        return time.monotonic() - started

    def serve(self, room: Path, arm: str) -> None:
        require_window()
        # --sm is NOT optional: `gen-worker up --help` says it is "required to
        # adopt compiled graphs". Without it the boot serves EAGER, both arms
        # measure eager, and the table reports a beautiful 0% delta — the same
        # vacuous green the preflight exists to stop, arriving from the boot
        # side. The whole run would be worthless and would look fine.
        argv = ["up", str(room), "-d", "--checkpoint", str(self.args.checkpoint),
                "--idle-timeout", str(self.args.idle_timeout)]
        if self.args.sm:
            argv += ["--sm", self.args.sm]
        up = self._gen_worker(room, argv, timeout=self.args.boot_timeout)
        if up.returncode != 0:
            raise SystemExit(f"[{arm}] up failed:\n{up.stdout}\n{up.stderr}")
        banner = (up.stdout or "") + (up.stderr or "")
        (self.out / f"up-{arm}.log").write_text(banner)
        # `up -d` DETACHES: its own output is a three-line launcher banner and
        # says nothing about adoption. The serving daemon's log is a different
        # file, and the launcher prints where — so read THAT. Checking the
        # launcher's output for the word "adopt" is how this harness spent a
        # window measuring eager against eager (pgw#1591).
        self._daemon_log[arm] = None
        for line in banner.splitlines():
            if "logs:" in line:
                candidate = Path(line.split("logs:", 1)[1].strip())
                if candidate.name:
                    self._daemon_log[arm] = candidate
        if self._daemon_log[arm] is None:
            raise SystemExit(
                f"[{arm}] the boot did not name its daemon log, so nothing can "
                f"verify that this arm serves COMPILED. Refusing to measure."
            )

    def down(self, room: Path) -> None:
        self._gen_worker(room, ["down"], timeout=120)

    def request(self, room: Path, arm: str, aspect: str, cfg: str,
                round_index: int = 0) -> Sample:
        payload = {
            "prompt": self.args.prompt,
            "aspect_ratio": aspect,
            "num_inference_steps": self.args.steps,
            "seed": self.args.seed,
        }
        if cfg == "on":
            payload["guidance_scale"] = self.args.guidance
        started = time.monotonic()
        result = self._gen_worker(
            room,
            ["run", "-C", str(room), "--payload", json.dumps(payload), "--json"],
            timeout=self.args.request_timeout,
        )
        elapsed = time.monotonic() - started
        if result.returncode != 0:
            raise SystemExit(
                f"[{arm}] request {aspect}/{cfg} failed:\n"
                f"{result.stdout}\n{result.stderr}"
            )
        facts = self._dispatch_facts(arm, result.stdout)
        return Sample(
            arm=arm, aspect=aspect, cfg=cfg, seconds=elapsed,
            round=round_index, load1=os.getloadavg()[0],
            compiled_calls=int(facts.get("compiled_graph_calls", 0) or 0),
            eager_calls=int(facts.get("eager_calls", 0) or 0),
            displaced=tuple(facts.get("displaced_modules") or ()),
        )

    def _dispatch_facts(self, arm: str, stdout: str) -> dict[str, Any]:
        """The `dispatch` block of the response envelope, or a typed refusal.

        `gen-worker run --json` prints the raw envelope and the serving path
        puts `DispatchCounts.facts()` in it, so every request carries its own
        compiled/eager counts. An envelope without them is not a request this
        harness can score: it would leave the premise unmeasured, which is
        exactly how pgw#1591 nearly shipped a table of eager timings.
        """

        envelope: dict[str, Any] = {}
        for line in reversed((stdout or "").splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    envelope = json.loads(line)
                except json.JSONDecodeError:
                    continue
                break
        facts = envelope.get("dispatch")
        if not isinstance(facts, dict):
            raise SystemExit(
                f"[{arm}] the response envelope carries no `dispatch` facts, so "
                f"nothing measures whether this request served COMPILED. "
                f"Refusing to score it."
            )
        return facts

    def prepare(self, name: str) -> Path:
        """Lock + compile ONE arm. The card cost, paid once per arm."""

        room = self._workspace(name)
        lock_s = self.lock(room, name)
        records = self.specializations(room)
        # `--first` matches a facet by EQUALITY or the graph identity by
        # PREFIX (>= 8 chars) — `compile.Spec.short` is `graph[:16]`, scheme
        # included. A SUFFIX matches neither, so every compile would have been
        # refused with "names no specialization this endpoint has" — on the
        # pod, after the lock, inside the paid window.
        selectors = self.args.selectors or [r["graph"][:16] for r in records]
        compile_s = self.compile(room, name, selectors)
        self.mint[name] = {
            "lock_s": lock_s,
            "compile_s": compile_s,
            "specializations": len(records),
            "built": len(selectors),
            "load1_at_compile": os.getloadavg()[0],
        }
        return room

    def measure(self, room: Path, name: str, round_index: int) -> None:
        """One arm's turn in ONE round.

        Boot, one warm-up per cell (the first call through a fresh graph pays
        load and allocator costs no later request pays — reporting it would
        measure the boot), `--reps` measured calls, tear down. Rounds are the
        interleave: a peer's CPU job that lands mid-run then falls on every
        arm rather than on whichever one happened to be running, and the
        control's round-to-round spread is what says whether the cell is
        decidable at all.
        """

        self.serve(room, name)
        try:
            first = True
            for aspect in self.args.aspects:
                for cfg in self.args.cfg:
                    warmup = self.request(room, name, aspect, cfg, round_index)
                    if first:
                        # The warm-up call has now exercised the serve path
                        # once and CARRIES ITS OWN COUNTS. Check the premise
                        # here, at the cheapest possible point, before paying
                        # for the rest of the arm — let alone the other arm.
                        self.assert_compiled(name, warmup)
                        first = False
                    for _ in range(self.args.reps):
                        self.table.add(
                            self.request(room, name, aspect, cfg, round_index)
                        )
        finally:
            self.down(room)

    def assert_compiled(self, arm: str, sample: Sample) -> None:
        """Did THIS request actually serve compiled? Typed abort if not.

        Reads the MEASURED counts off the request's own envelope. It used to
        prove compiled serving by grepping the daemon log for `wrapper.tcg` —
        a string that exists only because of the alignment defect pgw#1593 is
        fixing. That check would have started aborting every HEALTHY arm the
        moment someone fixed that defect: an instrument whose green depends on
        another bug staying unfixed. The counter is the real evidence and it
        survives both fixes.

        Displacement is no longer read as "everything ran eager" either
        (pgw#1591): it is a separate fact, and a displaced module can still
        have served compiled calls. So the bar is what it always should have
        been — compiled calls happened, and none fell through.
        """

        if sample.compiled_calls <= 0:
            raise SystemExit(
                f"[{arm}] ZERO compiled calls on the warm-up request "
                f"(eager_calls={sample.eager_calls}, "
                f"displaced={list(sample.displaced)}). This arm serves EAGER; "
                f"two such arms compare to ~0% and read as 'the axis is free'. "
                f"Refusing to measure. See {self._daemon_log.get(arm)}"
            )
        if sample.eager_calls:
            raise SystemExit(
                f"[{arm}] MIXED execution on the warm-up request: "
                f"{sample.compiled_calls} compiled and {sample.eager_calls} "
                f"eager call(s). A wall that is part compiled and part eager "
                f"is not this axis's cost. Refusing to measure."
            )
        if sample.displaced:
            print(f"[{arm}] note: displaced={list(sample.displaced)} while "
                  f"{sample.compiled_calls} call(s) still served compiled")

    def report(self) -> dict[str, Any]:
        adoption = {}
        for arm in self.table.arms():
            if arm == "static":
                continue
            ok, offenders = self.table.verdict(arm, self.args.tolerance)
            adoption[arm] = {"adopt": ok, "outside_tolerance": offenders}
        return {
            "endpoint": str(self.endpoint),
            "substrate": self.args.substrate,
            # The attribution rides in the verdict itself, not only in the
            # rendered table, so a consumer reading the JSON cannot get the
            # numbers without the sentence that bounds them.
            "substrate_note": SUBSTRATES[self.args.substrate],
            "tolerance_pct": self.args.tolerance,
            "mint": self.mint,
            "table_markdown": self.table.render(self.args.substrate),
            "adoption": adoption,
            **self.table.as_dict(),
        }


# ---------------------------------------------------------------------------
# The CPU-only self-test: every instrument proven able to go red
# ---------------------------------------------------------------------------


def self_test() -> int:
    failures: list[str] = []

    def check(name: str, condition: bool) -> None:
        print(f"  {'ok  ' if condition else 'RED '} {name}")
        if not condition:
            failures.append(name)

    print("[self-test] the window gate")
    saved = os.environ.pop("VARENA_GPU_WINDOW", None)
    try:
        require_window()
        check("an ungranted window REFUSES", False)
    except WindowRequired:
        check("an ungranted window REFUSES", True)
    os.environ["VARENA_GPU_WINDOW"] = "1"
    try:
        require_window()
        check("a granted window proceeds", True)
    except WindowRequired:
        check("a granted window proceeds", False)
    finally:
        os.environ.pop("VARENA_GPU_WINDOW")
        if saved is not None:
            os.environ["VARENA_GPU_WINDOW"] = saved

    print("[self-test] the per-shape table CAN show a regression")
    table = Table()
    # TWO rounds: a single-round cell is undecidable by construction now, so a
    # fixture that wants to exercise the regression arithmetic has to supply
    # the reproducibility evidence a real run would.
    for round_index in range(2):
        for _ in range(3):
            table.add(Sample("static", "1:1", "on", 1.000, round=round_index))
            table.add(Sample("aspect", "1:1", "on", 1.010, round=round_index))
            table.add(Sample("static", "3:4", "on", 2.000, round=round_index))
            table.add(Sample("aspect", "3:4", "on", 2.600, round=round_index))
    check("a 1% cell reads +1.0%", round(table.regression("aspect", "1:1", "on"), 1) == 1.0)
    check("a 30% cell reads +30.0%", round(table.regression("aspect", "3:4", "on"), 1) == 30.0)
    ok, offenders = table.verdict("aspect", tolerance=5.0)
    check("ONE bad bucket refuses the whole axis", not ok and len(offenders) == 1)
    check("the offender is NAMED", offenders and offenders[0].startswith("3:4/on"))
    ok, _ = table.verdict("aspect", tolerance=50.0)
    check("a tolerance that covers it adopts", ok)

    print("[self-test] the >15% control spread makes a cell UNDECIDABLE")
    noisy = Table()
    for r, (control, measured) in enumerate(((1.00, 1.00), (1.40, 1.40))):
        noisy.add(Sample("static", "1:1", "on", control, round=r))
        noisy.add(Sample("aspect", "1:1", "on", measured, round=r))
    check("a 40% control spread is measured", round(noisy.spread("static", "1:1", "on")) == 40)
    check("and the cell is UNDECIDABLE", noisy.undecidable("1:1", "on"))
    ok, offenders = noisy.verdict("aspect", tolerance=5.0)
    check("an undecidable cell blocks adoption", not ok)
    check("and says re-run, not fail", any("UNDECIDABLE" in o for o in offenders))
    steady = Table()
    for r, value in enumerate((1.00, 1.05)):
        steady.add(Sample("static", "1:1", "on", value, round=r))
        steady.add(Sample("aspect", "1:1", "on", value, round=r))
    check("a 5% control spread stays decidable", not steady.undecidable("1:1", "on"))
    check("and adopts", steady.verdict("aspect", tolerance=3.0)[0])

    print("[self-test] an unmeasured cell is NOT silently adopted")
    thin = Table()
    for round_index in range(2):
        thin.add(Sample("static", "1:1", "on", 1.0, round=round_index))
        thin.add(Sample("aspect", "1:1", "on", 1.0, round=round_index))
        thin.add(Sample("static", "16:9", "on", 1.0, round=round_index))
    ok, offenders = thin.verdict("aspect", tolerance=5.0)
    check("a missing cell refuses", not ok and any("NOT MEASURED" in o for o in offenders))

    print("[self-test] ONE round is UNDECIDABLE, never a quiet pass")
    single = Table()
    single.add(Sample("static", "1:1", "on", 1.0, round=0))
    single.add(Sample("aspect", "1:1", "on", 1.0, round=0))
    check("a single-round cell has no spread", single.spread("static", "1:1", "on") is None)
    check("and is UNDECIDABLE", single.undecidable("1:1", "on"))
    ok, offenders = single.verdict("aspect", tolerance=3.0)
    check("so it cannot adopt", not ok)
    check("and the reason names the ROUND count, not a spread",
          any("ONE round" in o for o in offenders))

    print("[self-test] the table never averages across shapes")
    rendered = table.render()
    # header + separator + one row per shape, and render() joins without a
    # trailing newline, so the count is one less than the line count.
    check(
        "every shape has its own row",
        len(rendered.splitlines()) == 2 + len(table.shapes()),
    )
    check("the control column carries no percentage", "static (s) / vs static" not in rendered)

    print("[self-test] the substrate attribution cannot be dropped")
    stamped = table.render("raw-pod")
    check("the table leads with the substrate note",
          stamped.splitlines()[0].strip("_") == SUBSTRATES["raw-pod"])
    check("and it names the deploy-path limit",
          "not the deploy path" in stamped)
    check("every substrate carries a note", all(SUBSTRATES.values()))
    check("an unstamped render is only reachable deliberately",
          "substrate" not in table.render())

    print("[self-test] the arm plan is exhaustive over the axes")
    check("every arm names an axis policy", set(ARMS) == set(AXES))
    check("static is the OFF control", AXES["static"] == "off")

    print()
    if failures:
        print(f"SELF-TEST RED: {len(failures)} check(s) failed: {failures}")
        return 1
    print("SELF-TEST GREEN — every instrument above was shown able to go red.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true",
                        help="CPU-only; needs no card and no window")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--out", default="benchmarks/pgw1548")
    parser.add_argument("--arms", default="static,aspect")
    parser.add_argument("--aspects", default="1:1,3:4,16:9")
    parser.add_argument("--cfg", default="on")
    parser.add_argument("--reps", type=int, default=5,
                        help="measured requests per cell per round")
    parser.add_argument("--rounds", type=int, default=3,
                        help="interleave: every arm is measured once per "
                             "round, so box load falls on all of them. The "
                             "control's round-to-round spread decides whether "
                             "a cell is decidable (>15%% is not).")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=1548)
    parser.add_argument("--prompt", default="a fisherman at dawn")
    parser.add_argument("--sm", default="",
                        help="this GPU's sm (e.g. sm_89). REQUIRED to adopt "
                             "compiled graphs — without it every arm serves "
                             "eager and the table reads 0%% for the wrong reason")
    parser.add_argument("--substrate", choices=sorted(SUBSTRATES), default="raw-pod",
                        help="what produced these numbers; stamped into the "
                             "verdict and the table so the bound on what they "
                             "describe travels with them")
    parser.add_argument("--tolerance", type=float, default=3.0,
                        help="percent slower than static that still adopts")
    parser.add_argument("--selectors", default="",
                        help="comma-separated `gen-worker compile --first` "
                             "selectors; default: every specialization in the "
                             "arm's lock")
    parser.add_argument("--lock-timeout", type=float, default=1800)
    parser.add_argument("--compile-timeout", type=float, default=3600)
    parser.add_argument("--boot-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--idle-timeout", type=int, default=1800)
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.endpoint or not args.checkpoint:
        parser.error("--endpoint and --checkpoint are required for a real run")
    require_window()

    args.aspects = [a for a in args.aspects.split(",") if a]
    args.cfg = [c for c in args.cfg.split(",") if c]
    args.selectors = [s for s in args.selectors.split(",") if s]

    arms = args.arms.split(",")
    for arm in arms:
        if arm not in ARMS:
            parser.error(f"unknown arm {arm!r}; one of {list(ARMS)}")
    if "static" not in arms:
        parser.error("the static arm is the control; every run needs it")

    bench = Bench(args)
    rooms = {}
    for arm in arms:
        print(f"[prepare] {arm} (--dynamic-axes {AXES[arm]})")
        rooms[arm] = bench.prepare(arm)
    for round_index in range(args.rounds):
        for arm in arms:
            print(f"[round {round_index}] {arm} (load {os.getloadavg()[0]:.1f})")
            bench.measure(rooms[arm], arm, round_index)

    report = bench.report()
    (bench.out / "verdict.json").write_text(json.dumps(report, indent=2))
    print()
    print(report["table_markdown"])
    print()
    for arm, verdict in report["adoption"].items():
        state = "ADOPT" if verdict["adopt"] else "KEEP STATIC"
        print(f"{arm}: {state} {verdict['outside_tolerance'] or ''}")
    print()
    print(report["substrate_note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
