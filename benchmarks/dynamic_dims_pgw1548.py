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


_SRC = Path(__file__).resolve().parents[1] / "src"


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
        venv = self._venv()
        if venv.exists():
            (room / ".venv").symlink_to(venv)
        self._declare_axis(room, arm)
        return room

    #: The endpoint's own shape declaration, which is now the ONLY thing that
    #: decides an arm. `STATIC`/`DYNAMIC` are plain strings ("static" /
    #: "dynamic") in `serving.lane_spec`, so the arm is expressed by rewriting
    #: the VALUE and the endpoint's imports are left alone.
    _DECL = 'shapes={"aspect": STATIC}'

    def _declare_axis(self, room: Path, arm: str) -> None:
        """Write this arm's shape choice into the room's own endpoint source.

        **pgw#1599 DELETED `--dynamic-axes`.** The axis choice is no longer a
        flag on `gen-worker lock`; it is DECLARED on the model class as
        `shapes={"aspect": STATIC|DYNAMIC}` and travels in the release
        document's `fork_axes`. The flag is not merely ignored -- the CLI
        REJECTS it (`gen-worker: error: unrecognized arguments: --dynamic-axes
        off`, measured), so the old call could only ever have worked by
        accident, via a lock-cache hit that skipped the CLI entirely.

        That accident is exactly why this must REFUSE rather than warn. If the
        declaration cannot be found and rewritten, both arms compile from the
        same source, the ABBA table compares an arm against ITSELF, and every
        number in it looks perfectly reasonable. A silent no-op here does not
        produce a wrong row; it produces a wrong TABLE that reads as right.
        """

        target = {"static": "STATIC", "aspect": '"dynamic"'}.get(arm)
        if target is None:
            raise SystemExit(
                f"[{arm}] no shape declaration is defined for this arm. "
                f"pgw#1599 makes `batch` PERMANENTLY STATIC "
                f"(PERMANENTLY_STATIC_SHAPE_AXES), so the batch/all arms are "
                f"not expressible and CFG is not re-litigated here.")
        main = room / "src" / "sdxl" / "main.py"
        if not main.exists():
            raise SystemExit(f"[{arm}] {main} absent — cannot declare the axis")
        source = main.read_text()
        if source.count(self._DECL) != 1:
            raise SystemExit(
                f"[{arm}] REFUSING: expected exactly one {self._DECL!r} in "
                f"{main}, found {source.count(self._DECL)}. The arm cannot be "
                f"expressed, and running anyway would compare an arm against "
                f"itself while every number still looked plausible.")
        main.write_text(source.replace(
            self._DECL, f'shapes={{"aspect": {target}}}'))
        print(f"[{arm}] declared shapes={{'aspect': {target}}} in the room's "
              f"own source (pgw#1599: the declaration IS the arm)")

    def _venv(self) -> Path:
        """The environment the endpoint's own code runs in.

        NOT always `<endpoint>/.venv`. On this box that one carries
        `torch 2.13.0+cpu` — an endpoint booted in it would refuse the card,
        or worse, serve on CPU and produce a table of walls that compare two
        CPU runs. `--venv` names the fleet-line environment instead, and the
        boot below verifies it can actually see a GPU before anything is
        compiled.
        """

        if getattr(self.args, "venv", ""):
            return Path(self.args.venv).expanduser().resolve()
        return self.endpoint / ".venv"

    def _python(self) -> str:
        venv = self._venv() / "bin" / "python"
        return str(venv) if venv.exists() else sys.executable

    def assert_card(self) -> dict[str, Any]:
        """Refuse before the first compile if this environment has no card.

        A CPU-only torch is not a slow run, it is a different measurement
        wearing the same output shape: every arm would still produce walls,
        the table would still render, and every number in it would be wrong
        about the thing being asked. Cheaper to refuse here than to discover
        it in the verdict.
        """

        probe = _run(
            [self._python(), "-c",
             "import json,torch;print(json.dumps({'torch': torch.__version__,"
             "'cuda': torch.version.cuda, 'available': torch.cuda.is_available(),"
             "'name': torch.cuda.get_device_name(0) if torch.cuda.is_available()"
             " else '', 'capability': list(torch.cuda.get_device_capability(0))"
             " if torch.cuda.is_available() else []}))"],
            timeout=180,
        )
        line = (probe.stdout or "").strip().splitlines()[-1] if probe.stdout else ""
        try:
            facts = json.loads(line)
        except json.JSONDecodeError:
            raise SystemExit(
                f"could not read a card probe out of {self._python()}:\n"
                f"{probe.stdout}\n{probe.stderr}"
            ) from None
        if not facts.get("available"):
            raise SystemExit(
                f"{self._python()} reports torch {facts.get('torch')} "
                f"(cuda={facts.get('cuda')}) with NO card available. Every arm "
                f"would produce walls and the table would render — and it would "
                f"be a CPU-vs-CPU comparison labelled as a graph A/B. Refusing. "
                f"Pass --venv <fleet-line env>."
            )
        major, minor = (facts.get("capability") or [0, 0])[:2]
        facts["sm"] = f"sm_{major}{minor}"
        if self.args.sm and self.args.sm != facts["sm"]:
            raise SystemExit(
                f"--sm {self.args.sm!r} but the card is {facts['sm']} "
                f"({facts['name']}). An sm mismatch does not fail loudly at "
                f"adopt time — it silently matches no artifact and every arm "
                f"serves eager."
            )
        print(f"[card] {facts['name']} {facts['sm']} torch {facts['torch']} "
              f"cu{facts['cuda']}")
        return facts

    def _gen_worker(self, room: Path, argv: list[str], timeout: float) -> subprocess.CompletedProcess:
        """The CLI, running THIS branch's source (PYTHONPATH wins over the pin)."""

        return _run(
            [self._python(), "-m", "gen_worker.cli", *argv],
            cwd=room,
            env={
                # BOTH entries, and the second one is not optional. sdxl/main.py
                # imports `tensorfs` at top level; tensorfs is NOT on PyPI, and
                # pgw satisfies it by VENDORING the package at
                # src/gen_worker/_vendor, which is importable under its own
                # top-level name only when that directory is on PYTHONPATH.
                # The bootstrap exports exactly that pair -- but `_run` merges
                # os.environ and then this explicit key OVERRIDES it, so
                # naming only `src` here silently dropped the vendor half and
                # every lock died with `ModuleNotFoundError: No module named
                # 'tensorfs'`. It stayed hidden for the whole campaign because
                # the lock was always a CACHE HIT and never actually ran; the
                # first real derive found it immediately.
                "PYTHONPATH": os.pathsep.join((
                    str(_SRC), str(_SRC / "gen_worker" / "_vendor"))),
                # Fragmentation is a real term on a card this small: an SDXL
                # boot leaves the allocator holding segments the request's
                # activations cannot reuse. Held CONSTANT across every arm, so
                # it changes the headroom without changing the comparison.
                "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
                    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
            },
            timeout=timeout,
        )

    # -- one arm ------------------------------------------------------------
    def lock(self, room: Path, arm: str) -> float:
        """Derive this arm's graph set. **Cacheable, and it should be cached.**

        The derive touches no card — it is `torch.export` tracing on CPU — but
        it is not cheap: SDXL measured 865.6 s static / 201.4 s aspect. Paying
        that inside a granted GPU window spends the one scarce resource on the
        one step that does not need it, so `--lock-cache` takes a lock derived
        earlier on the box. The cached wall is carried forward from the sidecar
        rather than reported as zero: a cached arm must not read as a free
        derive in the mint-cost arithmetic, which is half of the ROI question.
        """

        cached = self._cached_lock(arm)
        if cached is not None:
            document, wall = cached
            shutil.copy2(document, room / "endpoint.lock")
            print(f"[{arm}] lock: reused {document} (derived off-window in "
                  f"{wall:.1f}s)")
            return wall
        started = time.monotonic()
        result = self._gen_worker(
            room,
            # NO `--dynamic-axes`: pgw#1599 deleted it and the CLI REJECTS it.
            # The arm now rides the room's own declaration (`_declare_axis`).
            ["lock", str(room), "--force",
             "--checkpoint", str(self.args.checkpoint)],
            timeout=self.args.lock_timeout,
        )
        if result.returncode != 0:
            raise SystemExit(f"[{arm}] lock failed:\n{result.stdout}\n{result.stderr}")
        return time.monotonic() - started

    def _cached_lock(self, arm: str) -> tuple[Path, float] | None:
        """A pre-derived `endpoint.lock.<arm>` plus the wall it actually cost.

        A cache HIT with no recorded wall is refused rather than scored as 0 s:
        the derive wall is an input to the specialization-ROI arithmetic, and a
        silent zero there would make every dynamic arm look free to author.
        """

        if not getattr(self.args, "lock_cache", ""):
            return None
        directory = Path(self.args.lock_cache).expanduser()
        document = directory / f"endpoint.lock.{arm}"
        if not document.exists():
            return None
        meta = directory / f"endpoint.lock.{arm}.meta.json"
        if not meta.exists():
            raise SystemExit(
                f"[{arm}] {document} has no {meta.name} beside it, so the derive "
                f"wall it cost is unknown. That wall is half of the ROI this "
                f"lane reports; scoring it as zero would make a dynamic arm read "
                f"as free to derive. Re-derive with the sidecar, or drop "
                f"--lock-cache."
            )
        wall = float(json.loads(meta.read_text())["lock_s"])
        return document, wall

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

    def covering_selectors(self, arm: str, records: list[dict[str, Any]]) -> list[str]:
        """Exactly the specializations this run's SHAPES enter — no more, no fewer.

        A single global `--selectors` list cannot serve a multi-arm run: the
        static arm needs one specialization per shape, the aspect arm needs one
        for all of them, and handing either the other's list is a refusal
        ("names no specialization this endpoint has") — arriving after the lock,
        inside the paid window. And building EVERY record instead would spend
        the card on 15 buckets nobody measures.

        So the selection is DERIVED from the same (aspect, cfg) cells the table
        will report, against each record's own declared `sample` ingress:
        concrete dims must match exactly; a symbolic dim (`8*s3`) matches when
        the value is inside the symbol's observed range AND on its stride —
        which is the dispatcher's own rule, not a looser one invented here.

        A cell no record covers is a REFUSAL, not a skipped row. Serving it
        would fall through to eager, and an eager wall averaged into a compiled
        column is the exact vacuous green this file exists to prevent.
        """

        wanted: list[tuple[str, str, int, int, int]] = []
        for aspect in self.args.aspects:
            latent = self.args.latents.get(aspect)
            if latent is None:
                raise SystemExit(
                    f"--latents has no entry for aspect {aspect!r}. The harness "
                    f"will not guess a bucket: a wrong latent silently selects "
                    f"the wrong specialization and the arm then serves eager."
                )
            for cfg in self.args.cfg:
                wanted.append((aspect, cfg, 2 if cfg == "on" else 1, *latent))

        chosen: list[str] = []
        for aspect, cfg, batch, height, width in wanted:
            match = None
            for record in records:
                if self._covers(record, batch, height, width):
                    match = record
                    break
            if match is None:
                raise SystemExit(
                    f"[{arm}] no specialization in this arm's lock covers "
                    f"{aspect}/{cfg} (batch {batch}, latent {height}x{width}). "
                    f"That cell would serve EAGER and its wall would be averaged "
                    f"into a compiled column. Refusing to run the arm."
                )
            short = match["graph"][:16]
            if short not in chosen:
                chosen.append(short)
        print(f"[{arm}] building {len(chosen)} of {len(records)} "
              f"specialization(s) for {len(wanted)} cell(s): {chosen}")
        return chosen

    @staticmethod
    def _covers(record: dict[str, Any], batch: int, height: int, width: int) -> bool:
        inputs = ((record.get("ingress") or {}).get("inputs") or [])
        sample = next((i for i in inputs if i.get("name") == "sample"), None)
        if sample is None:
            return False
        shape = sample.get("shape") or []
        if len(shape) != 4:
            return False
        symbols = (record.get("ingress") or {}).get("symbols") or {}
        for declared, value in zip(shape, (batch, None, height, width)):
            if value is None:  # the channel dim, never an axis
                continue
            if isinstance(declared, int):
                if declared != value:
                    return False
                continue
            bounds = symbols.get(declared)
            if not bounds:
                return False
            low, high = int(bounds[0]), int(bounds[-1])
            if not (low <= value <= high):
                return False
            # `8*s3` is a DERIVED dim: the stride rides in the symbol name and
            # a value off it satisfies no guard (tcg#77). Checking the range
            # alone would select a record the dispatcher then refuses.
            stride = 1
            if "*" in str(declared):
                head = str(declared).split("*", 1)[0]
                if head.isdigit():
                    stride = int(head)
            if stride > 1 and value % stride:
                return False
        return True

    def compile(self, room: Path, arm: str, selectors: list[str]) -> float:
        """Build ONLY what this run benchmarks. One selector per child."""

        require_window()
        started = time.monotonic()
        for selector in selectors:
            result = self._gen_worker(
                room,
                ["compile", str(room), "--first", selector, "--fill", "none"]
                + (["--graph-store", str(self.args.graph_store)]
                   if getattr(self.args, "graph_store", "") else []),
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
        # `--compile off` is NOT an optimization, it is a correctness
        # requirement for a benchmark, and it cost this lane a leg to learn.
        # The default policy is `auto`, whose help says it fills this boot's
        # holes in the background, "never blocking a request". It does not
        # block a request — it COMPETES with one. Measured on the campaign card
        # (2026-08-20): booting the static arm with 17 unbuilt specializations
        # started `gen_worker.serving.mint_child`, which took 5300 MiB of an
        # 8188 MiB card, and the very first measured request died with 6.6 MB
        # free. On a card that does NOT OOM the outcome is worse, because the
        # arm still produces walls — walls taken while a compiler had the SMs.
        # Every artifact this harness measures was built by the explicit
        # `gen-worker compile` above; a hole is a shape this run is not
        # measuring, and filling it during the measurement is pure contamination.
        argv = ["up", str(room), "-d", "--checkpoint", str(self.args.checkpoint),
                "--compile", "off",
                "--idle-timeout", str(self.args.idle_timeout)]
        # The CONFOUND leg (pgw#1586 rider) boots compiled-mode against an EMPTY
        # store so ZERO graphs are armed. If the VRAM step appears there too,
        # the cost is the mode PATH and not AOTI -- which is the attribution the
        # residency lane's +1176 MiB single allocation currently sits on.
        if getattr(self.args, "graph_store", ""):
            argv += ["--graph-store", str(self.args.graph_store)]
        # The CFG/batch axis is a CHECKPOINT-DEFAULTS flag, not a request field
        # (sd15 main.py:367-373, and sdxl reads `config.cfg` the same way): the
        # request's `guidance_scale` is warned-and-ignored on a cfg-off serving.
        # So batch-1 vs batch-2 cannot vary WITHIN one boot — it is a property
        # of the boot, and `--defaults` is where it is stated. Passing
        # guidance_scale and hoping for batch 1 would have measured batch 2
        # twice and reported the CFG axis as free.
        if self.args.defaults:
            argv += ["--defaults", self.args.defaults]
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
        selectors = self.args.selectors or self.covering_selectors(name, records)
        if getattr(self.args, "skip_compile", False):
            # The CONFOUND leg: compiled MODE, but nothing built, so ZERO graphs
            # are armed. Building into the "empty" store would arm one and
            # measure the opposite of what the leg asks.
            print(f"[{name}] SKIPPING compile — the confound leg wants zero "
                  f"armed graphs, not a freshly built one")
            compile_s = 0.0
            selectors = []
        else:
            compile_s = self.compile(room, name, selectors)
        self.mint[name] = {
            "lock_s": lock_s,
            "lock_cached": self._cached_lock(name) is not None,
            "compile_s": compile_s,
            "compile_s_per_spec": compile_s / len(selectors) if selectors else None,
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

        if getattr(self.args, "expect_eager", False):
            # Inverted premise, and it must be just as strict: this leg is only
            # meaningful if NOTHING was armed. A compiled call here means the
            # store was not actually empty and the confound was not tested.
            if sample.compiled_calls > 0:
                raise SystemExit(
                    f"[{arm}] --expect-eager but {sample.compiled_calls} "
                    f"compiled call(s): the graph store was NOT empty, so this "
                    f"measures an armed pod and answers nothing about the "
                    f"mode path. Refusing.")
            print(f"[{arm}] confound leg: {sample.eager_calls} eager call(s), "
                  f"0 compiled — zero graphs armed, as intended")
            return
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

    def roi(self) -> dict[str, Any]:
        """Paul's question, as arithmetic: *is the extra specialization worth it?*

        One axis, two sides, both measured here and neither guessed:

        * **BOUGHT** — the worst decidable cell's regression against static.
          The WORST, not the mean: a request that lands in the bad bucket pays
          it every time, so averaging a win against a loss answers a question
          nobody asked.
        * **PAID** — (N_static − N_dynamic) extra specializations, each costing
          one measured `compile_s_per_spec` on the card plus one export inside
          every `gen-worker lock` anyone ever runs, plus its bytes in the lock.

        The floor is not 1. SDXL's schedulers fork the timestep dtype (5 int64 /
        3 float32, measured $0 by the coordinator 2026-08-20), and a dtype fork
        is STRUCTURAL, not a shape — so the minimum honest spec count for a
        fully dynamic SDXL is 2, and the static ceiling is 2x the shape
        enumeration. `dtype_lanes` states that assumption in the output instead
        of leaving a reader to infer a floor of 1 from a table that only ever
        exercised one lane.
        """

        static = self.mint.get("static") or {}
        # A `gen-worker compile` that finds its artifact already in the box CAS
        # returns in under a second. That is a CACHE HIT, not a mint, and using
        # it to price a specialization would value the removed graphs at ~0 —
        # the ROI would then say every axis is free to keep, which is the
        # opposite of the truth. So the price comes from an arm that ACTUALLY
        # BUILT, and if no arm did, the paid side is reported as unpriced
        # rather than as cheap.
        CACHE_HIT_S = 5.0
        built: list[tuple[str, float]] = [
            (arm, row["compile_s_per_spec"])
            for arm, row in self.mint.items()
            if (row.get("compile_s_per_spec") or 0.0) > CACHE_HIT_S
        ]
        per_spec = max((w for _a, w in built), default=None)
        priced_from = max(built, key=lambda pair: pair[1])[0] if built else None
        out: dict[str, Any] = {
            "dtype_lanes": self.args.dtype_lanes,
            "measured_compile_s_per_spec": per_spec,
            "compile_price_from_arm": priced_from,
            "compile_price_note": (
                f"priced from the {priced_from!r} arm's own build"
                if priced_from else
                "UNPRICED: every arm's artifacts were already in the box CAS, "
                "so no mint wall was measured this run — the specializations "
                "removed are counted but not costed"
            ),
            "static_compile_s_per_spec_raw": static.get("compile_s_per_spec"),
            "axes": {},
        }
        for arm in self.table.arms():
            if arm == "static":
                continue
            mint = self.mint.get(arm) or {}
            worst: tuple[str, float] | None = None
            for aspect, cfg in self.table.shapes():
                if self.table.undecidable(aspect, cfg):
                    continue
                delta = self.table.regression(arm, aspect, cfg)
                if delta is None:
                    continue
                if worst is None or delta > worst[1]:
                    worst = (f"{aspect}/{cfg}", delta)
            saved = (static.get("specializations") or 0) - (
                mint.get("specializations") or 0)
            out["axes"][arm] = {
                "worst_cell": worst[0] if worst else None,
                "bought_pct_per_step": -worst[1] if worst else None,
                "specializations_static": static.get("specializations"),
                "specializations_dynamic": mint.get("specializations"),
                "specializations_saved": saved,
                "compile_s_saved": (saved * per_spec) if per_spec else None,
                "derive_s_static": static.get("lock_s"),
                "derive_s_dynamic": mint.get("lock_s"),
                "derive_s_saved": (
                    (static.get("lock_s") or 0.0) - (mint.get("lock_s") or 0.0)
                ),
            }
        return out

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
            # Which SCHEDULER lane these graphs cover. Same discipline as the
            # substrate stamp: the SDXL locks in play were derived under the
            # checkpoint's own EulerDiscrete (float32 timesteps), so a reader
            # must not take the table as an all-scheduler result.
            "lane_note": self.args.lane_note,
            "tolerance_pct": self.args.tolerance,
            "mint": self.mint,
            "table_markdown": self.table.render(self.args.substrate),
            "adoption": adoption,
            "roi": self.roi(),
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

    print("[self-test] a cached lock cannot smuggle in a free derive")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        (cache / "endpoint.lock.aspect").write_text("{}")
        bench = Bench.__new__(Bench)
        bench.args = argparse.Namespace(lock_cache=str(cache))
        try:
            bench._cached_lock("aspect")
            check("a cache hit with no derive wall is refused", False)
        except SystemExit as exc:
            check("a cache hit with no derive wall is refused",
                  "meta.json" in str(exc))
        (cache / "endpoint.lock.aspect.meta.json").write_text('{"lock_s": 201.4}')
        hit = bench._cached_lock("aspect")
        check("with the sidecar, the REAL derive wall is carried forward",
              hit is not None and abs(hit[1] - 201.4) < 1e-9)
        check("a miss is a miss, not a refusal", bench._cached_lock("batch") is None)
        bench.args = argparse.Namespace(lock_cache="")
        check("no cache configured means no cache", bench._cached_lock("aspect") is None)

    print("[self-test] the ROI reports the WORST cell, never the mean")
    roi_table = Table()
    for round_index in (0, 1):
        for _ in range(2):
            # 1:1 barely moves; 3:2 costs 12%. A mean would read as +6% and
            # invite adoption; the worst cell is what a request actually pays.
            roi_table.add(Sample(arm="static", aspect="1:1", cfg="on",
                                 seconds=10.0, round=round_index))
            roi_table.add(Sample(arm="aspect", aspect="1:1", cfg="on",
                                 seconds=10.0, round=round_index))
            roi_table.add(Sample(arm="static", aspect="3:2", cfg="on",
                                 seconds=10.0, round=round_index))
            roi_table.add(Sample(arm="aspect", aspect="3:2", cfg="on",
                                 seconds=11.2, round=round_index))
    roi_bench = Bench.__new__(Bench)
    roi_bench.table = roi_table
    roi_bench.args = argparse.Namespace(dtype_lanes=2)
    roi_bench.mint = {
        "static": {"specializations": 18, "compile_s_per_spec": 218.0,
                   "lock_s": 865.6},
        "aspect": {"specializations": 2, "lock_s": 201.4},
    }
    roi = roi_bench.roi()["axes"]["aspect"]
    check("the worst cell is the one reported", roi["worst_cell"] == "3:2/on")
    check("and it is stated as what the extra specializations BUY",
          roi["bought_pct_per_step"] is not None
          and abs(roi["bought_pct_per_step"] - -12.0) < 0.01)
    check("the paid side counts the specializations removed",
          roi["specializations_saved"] == 16)
    check("priced at the MEASURED compile wall, not a remembered one",
          abs(roi["compile_s_saved"] - 16 * 218.0) < 1e-6)
    check("the derive saving is reported too (it is author time, every run)",
          abs(roi["derive_s_saved"] - (865.6 - 201.4)) < 1e-6)
    check("the dtype-lane floor travels with the arithmetic",
          roi_bench.roi()["dtype_lanes"] == 2)

    print("[self-test] the arm order ALTERNATES, so a drifting box cancels")
    plans = [
        (["static", "aspect"] if index % 2 == 0
         else list(reversed(["static", "aspect"])))
        for index in range(4)
    ]
    check("odd rounds reverse the order", plans[1] == ["aspect", "static"])
    check("even rounds keep it", plans[0] == ["static", "aspect"])
    first_halves = [plan[0] for plan in plans]
    check("each arm goes first equally often over an even round count",
          first_halves.count("static") == first_halves.count("aspect"))
    # The property that matters: under a monotonic drift, a fixed order gives
    # the later arm a systematic penalty and ABBA does not.
    drift = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
    fixed = {"a": [], "b": []}
    abba = {"a": [], "b": []}
    tick = iter(drift)
    for index in range(4):
        for arm in (["a", "b"] if index % 2 == 0 else ["b", "a"]):
            abba[arm].append(next(tick))
    tick = iter(drift)
    for _index in range(4):
        for arm in ("a", "b"):
            fixed[arm].append(next(tick))
    check("a fixed order charges the drift to the second arm",
          statistics.mean(fixed["b"]) > statistics.mean(fixed["a"]))
    check("ABBA cancels it exactly",
          abs(statistics.mean(abba["a"]) - statistics.mean(abba["b"])) < 1e-9)

    print("[self-test] a CACHE HIT is never mistaken for a mint wall")
    roi_bench.mint = {
        "static": {"specializations": 18, "compile_s_per_spec": 0.54,
                   "lock_s": 865.6},
        "aspect": {"specializations": 2, "compile_s_per_spec": 0.51,
                   "lock_s": 201.4},
    }
    cached = roi_bench.roi()
    check("sub-second per-spec walls do not price a specialization",
          cached["measured_compile_s_per_spec"] is None)
    check("and the report SAYS the paid side is unpriced",
          "UNPRICED" in cached["compile_price_note"])
    check("the removed specializations are still COUNTED",
          cached["axes"]["aspect"]["specializations_saved"] == 16)
    check("but not costed from a cache hit",
          cached["axes"]["aspect"]["compile_s_saved"] is None)
    roi_bench.mint["aspect"]["compile_s_per_spec"] = 111.0
    priced = roi_bench.roi()
    check("one arm that really built prices the axis",
          abs(priced["measured_compile_s_per_spec"] - 111.0) < 1e-9)
    check("and the report names WHICH arm the price came from",
          priced["compile_price_from_arm"] == "aspect")

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
    parser.add_argument("--latents", default="",
                        help="aspect=HxW LATENT dims, comma separated "
                             "(sd15 512-native: `1:1=64x64,3:2=56x80,2:3=80x56`; "
                             "SDXL: `1:1=128x128,3:2=104x152,2:3=152x104`). "
                             "Stated rather than derived: the endpoint's bucket "
                             "table is its own code, and a harness that guessed "
                             "one would silently select the wrong specialization "
                             "and measure eager")
    parser.add_argument("--defaults", default="",
                        help='checkpoint-defaults JSON for `up --defaults`, '
                             'e.g. \'{"cfg": false}\' for the batch-1 half of '
                             'the CFG axis. CFG is a per-checkpoint flag, so it '
                             'is a property of the BOOT, not of a request')
    parser.add_argument("--skip-compile", action="store_true",
                        help="lock but do NOT build — the confound leg needs "
                             "compiled mode with zero graphs armed")
    parser.add_argument("--expect-eager", action="store_true",
                        help="invert the premise: this leg REQUIRES zero "
                             "compiled calls, and refuses if any appear")
    parser.add_argument("--graph-store", default="",
                        help="graph CAS root; point at an EMPTY dir to boot "
                             "compiled-mode with ZERO graphs armed (the pgw#1586 confound leg)")
    parser.add_argument("--venv", default="",
                        help="the environment the endpoint runs in. Default "
                             "<endpoint>/.venv — which on this box is CPU-only "
                             "torch, so a real leg must name the fleet-line env")
    parser.add_argument("--lock-cache", default="",
                        help="directory holding pre-derived "
                             "`endpoint.lock.<arm>` files, each with a "
                             "`.meta.json` carrying the derive wall it cost. "
                             "The derive needs no card; paying it inside a "
                             "granted GPU window spends the scarce resource on "
                             "the step that does not need it.")
    parser.add_argument("--lane-note", default="euler/float32 timestep lane only",
                        help="which scheduler/dtype lane these graphs cover; "
                             "stamped into the verdict so the table is not read "
                             "as an all-scheduler result")
    parser.add_argument("--dtype-lanes", type=int, default=2,
                        help="structural timestep-dtype lanes this model forks "
                             "into (SDXL: 2 — 5 int64 / 3 float32 schedulers). "
                             "This is the FLOOR on specialization count even "
                             "with every shape axis dynamic.")
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
    latents: dict[str, tuple[int, int]] = {}
    for entry in (e for e in args.latents.split(",") if e):
        aspect, _, dims = entry.partition("=")
        height, _, width = dims.partition("x")
        latents[aspect] = (int(height), int(width))
    args.latents = latents
    if not args.selectors and not latents:
        parser.error(
            "--latents is required (or pass explicit --selectors): without it "
            "the harness cannot know which specialization each shape enters, "
            "and building every record would spend the card on buckets this "
            "run does not measure")

    arms = args.arms.split(",")
    for arm in arms:
        if arm not in ARMS:
            parser.error(f"unknown arm {arm!r}; one of {list(ARMS)}")
    if "static" not in arms:
        parser.error("the static arm is the control; every run needs it")

    bench = Bench(args)
    bench.card = bench.assert_card()
    rooms = {}
    for arm in arms:
        print(f"[prepare] {arm} (--dynamic-axes {AXES[arm]})")
        rooms[arm] = bench.prepare(arm)
    for round_index in range(args.rounds):
        # ABBA, not AAA-BBB. Measured 2026-08-20 on the campaign card: this box
        # DRIFTS MONOTONICALLY SLOWER within a run — sd15 static 1:1 went
        # 3.104 -> 3.603 -> 3.737 s across three rounds while load rose 7.2 ->
        # 9.0 and the laptop GPU sat at 74-80 C. With a FIXED arm order the
        # second arm is always measured later inside every round, so a
        # monotonic drift is charged entirely to it — a systematic bias
        # favouring whichever arm goes first, in the exact direction that
        # would have made the dynamic arm look slower. Reversing on odd rounds
        # cancels a linear drift by construction; the interleave was already
        # here, but interleaving with a fixed order is not enough.
        order = arms if round_index % 2 == 0 else list(reversed(arms))
        for arm in order:
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
    print("SPECIALIZATION ROI — what the extra graphs buy, and what they cost")
    roi = report["roi"]
    for arm, row in roi["axes"].items():
        bought = row["bought_pct_per_step"]
        print(
            f"  {arm}: static buys "
            + ("(not decidable)" if bought is None
               else f"{bought:+.1f}% round-trip in its WORST cell "
                    f"({row['worst_cell']})")
            + f" and costs {row['specializations_saved']} extra "
              f"specialization(s) = "
            + (f"{row['compile_s_saved']:.0f}s of mint"
               if row["compile_s_saved"] else "an unmeasured mint")
            + f" + {row['derive_s_saved']:.0f}s of derive on EVERY lock run"
        )
    print(f"  floor: {roi['dtype_lanes']} specialization(s) even fully dynamic "
          f"(structural timestep-dtype lanes)")
    print()
    print(report["substrate_note"], "|", report["lane_note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
