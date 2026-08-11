"""``python -m gen_worker.author_ci`` — the AUTHOR's compile-vs-eager proof, in
one command (ie#664 tier 2, DESIGN-RULINGS §4.32).

An endpoint that declares ``compile=`` claims two things about its own code:
that the compiled output is CORRECT and that it is FASTER. §4.32 makes both the
AUTHOR's claims to prove, because every parity failure ever caught (a baked
``conv_out.bias``, timestep dtype scars) was an endpoint code / model config
defect that no consumer runtime can fix. ie#664 wrote that down as a two-tier
standard: tier 1 is the torch-free lint in ``inference-endpoints``
(``scripts/lint_author_ci.py``, every PR), and tier 2 is a pod runbook —
``AUTHOR-CI.md``'s checklist, executed by hand. This module is that checklist as
ONE COMMAND, run on a pod in the endpoint's own image::

    python -m gen_worker.author_ci --payload '{"prompt": "a cat"}'

**IT MEASURES; IT NEVER GATES PUBLISH.** The author's run is advisory and always
will be: it is self-reported (unverifiable card, wheel, honesty). Promotion is
the PLATFORM's, on trusted hardware, against the published code — th#1811's
validation session and its promotability gate, per ie#664 §6. Nothing here
returns a value anything downstream consults, and its exit code is a signal to
the person who ran it.

WHAT IT ASSEMBLES — every piece already ships; none of them were wired together:

1. ``rigcheck.assert_fleet_line`` first (pgw#1114/#1120). A rig off the fleet's
   torch/CUDA line has produced no evidence. Exits 90/91 in rigcheck's own
   vocabulary, so a wrapper script reads one exit table and not two.
2. The ARM — the endpoint's own ``setup()`` mints (or adopts this machine's
   cell) through the production path. The parity verdict is the MINT-PARENT
   GATE's (pgw#1141: ``adopt_delegated_mint`` -> ``arm_aot(verify_numerics=
   True)`` -> ``provision.gate_cell_numerics``, strict), never a comparison
   re-implemented here. A family with open ``Compile.blockers`` is a LEGAL
   state — ``blocked-by-declaration`` — and the run continues eager-only.
3. SPEED, steady-vs-steady, ONE process, compiled arm first. The eager arm is
   pgw#1142's operator order engaged through its IN-PROCESS api
   (``serve_posture.apply_command``) and released in a ``finally`` — never the
   hub route, because the author's pod may have no hub. N >= 5 per arm with the
   first request of each arm discarded, median AND p95, read off
   ``stage_ms.<stage>`` and never a round trip (th#1795: the "10.9x" whose
   corrected figure was 1.3x).
4. The RECORD — the ``[proof]`` block of ``<endpoint>/author-ci.toml``, in
   exactly the schema ``scripts/lint_author_ci.py`` validates. The harness owns
   ``[proof]`` and nothing else: ``[parity]`` and ``[speed]`` are DECLARATIONS
   the author made and this command judges itself against, so ``--write``
   splices and never rewrites them.

A below-bar speedup is recorded as a FAILURE (``status = "failed"``, evidence
kept), never as a proof. A number that only ever reached a pod's stdout is a
number we do not have.
"""

from __future__ import annotations

import argparse
import datetime
import math
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import rigcheck, serve_posture, stage_timing
from .api.export_contract import blocker_refusal, open_blockers
from .discovery.project import load_project_config

#: Exit codes. 90/91 are rigcheck's, deliberately unchanged: an author who
#: wraps this command must not have to learn a second table for the same
#: preflight (0 on the line, 90 off it, 91 when the host cannot run it).
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_RIG_OFF_LINE = 90
EXIT_HOST_UNUSABLE = 91

#: ie#664's fleet default, with its stated rationale: 1.0 ("not slower") is
#: unfalsifiable on a rented pod because denoise variance sits in the low
#: single digits, so a bar there is met by noise. A family raises it once it
#: has a proof to raise it to.
FLEET_MIN_SPEEDUP = 1.10

#: The standing measurement bar (AUTHOR-CI.md, and `lint_author_ci.py` refuses
#: a record below it): N >= 5 per arm, steady state, first request discarded.
MIN_SAMPLES = 5

#: `stage_ms.<stage>` is the only shape a metric may take here. `slot_held_ms`
#: and `total_round_trip_ms` are hub-side round-trip terms that do not exist in
#: this process at all, and the second is the one th#1795 named.
METRIC_PREFIX = "stage_ms."

ARM_MINTED = "minted"
ARM_ADOPTED = "adopted"
ARM_BLOCKED = "blocked-by-declaration"
ARM_EAGER = "eager"

#: `[proof] status` — `lint_author_ci.py`'s closed vocabulary. `failed` is a
#: FINISHED measurement that did not pass; the lint refuses it (CI stays red)
#: and keeps the evidence, which is the difference between a failure and the
#: `never-run` gap it must never be relabelled as.
STATUS_PROVEN = "proven"
STATUS_FAILED = "failed"
STATUS_NEVER_RUN = "never-run"

ARM_COMPILED = "compiled"
ARM_EAGER_ARM = "eager"


class HarnessError(Exception):
    """A usage / declaration problem, reported to the author by name."""


# ---------------------------------------------------------------------------
# the declared bar
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Bar:
    """The speed bar this run is judged against, and where it came from."""

    metric: str
    min_speedup: float
    source: str

    @property
    def stage(self) -> str:
        return self.metric[len(METRIC_PREFIX):]


def resolve_bar(declaration: Any, record: Dict[str, Any]) -> Bar:
    """The DECLARATION first, then the record, then the fleet default.

    pgw#1149 moves the bar onto ``Compile.speed_metric`` / ``min_speedup``
    (the hub cannot read a repo-side toml at discovery). Those fields do not
    exist yet, so they are read by ``getattr``: the day that lane lands this
    resolver follows it with no edit here, and until then there is still only
    one reader per source.
    """
    metric = str(getattr(declaration, "speed_metric", "") or "").strip()
    declared_speedup = getattr(declaration, "min_speedup", None)
    if metric and declared_speedup is not None:
        return Bar(_metric(metric), float(declared_speedup), "declaration")

    speed = record.get("speed")
    speed = speed if isinstance(speed, dict) else {}
    metric = str(speed.get("metric") or "").strip()
    if not metric:
        raise HarnessError(
            "no speed metric is declared. `[speed] metric` in author-ci.toml "
            "names the compute stage this family reads (`stage_ms.denoise`); "
            "a harness may not guess it, and a bar chosen after seeing the "
            "number is not a bar. See AUTHOR-CI.md")
    raw = speed.get("min_speedup")
    if raw is None:
        return Bar(_metric(metric), FLEET_MIN_SPEEDUP, "fleet-default")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise HarnessError(f"[speed] min_speedup = {raw!r} is not a number")
    if float(raw) < 1.0:
        raise HarnessError(
            f"[speed] min_speedup = {raw} is below 1.0 — a cell that is not "
            f"faster than eager has no reason to exist")
    return Bar(_metric(metric), float(raw), "author-ci.toml")


def _metric(metric: str) -> str:
    """Refuse a round-trip metric BY NAME (th#1795)."""
    if not metric.startswith(METRIC_PREFIX) or not metric[len(METRIC_PREFIX):]:
        raise HarnessError(
            f"[speed] metric = {metric!r} is not readable here. This harness "
            f"measures IN PROCESS, so the only thing it can honestly report "
            f"is a compute stage: `{METRIC_PREFIX}<stage>`. A round trip "
            f"(`total_round_trip_ms`) does not exist in this process and is "
            f"the term th#1795 named — a sibling lane's '10.9x' corrected to "
            f"1.3x when it was read off the stage instead")
    return metric


# ---------------------------------------------------------------------------
# what one arm measured
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Leg:
    """One arm's steady-state samples, in milliseconds, first ALREADY dropped."""

    name: str
    samples: Tuple[float, ...]
    discarded: float

    @property
    def median(self) -> float:
        rows = sorted(self.samples)
        mid = len(rows) // 2
        if len(rows) % 2:
            return rows[mid]
        return (rows[mid - 1] + rows[mid]) / 2.0

    @property
    def p95(self) -> float:
        """Nearest-rank p95 — at N=5 that is the max, which is the point: the
        tail a guard miss produces is exactly what a median hides."""
        rows = sorted(self.samples)
        rank = max(1, math.ceil(0.95 * len(rows)))
        return rows[rank - 1]

    def line(self) -> str:
        return (f"{self.name:8} n={len(self.samples)} "
                f"median={self.median:.1f}ms p95={self.p95:.1f}ms "
                f"(discarded first: {self.discarded:.1f}ms)")


@dataclass(frozen=True)
class Parity:
    """The MINT-PARENT gate's verdict, carried whole."""

    passed: bool
    cosine: Optional[float]
    floor_source: str
    detail: str


@dataclass(frozen=True)
class Report:
    """Everything the record and the human verdict are written from."""

    family: str
    arm: str
    #: The short citation a `never-run` record's `blocker` carries — a blocker
    #: id, or the issue that owns the gap. Prose belongs in `arm_detail`.
    blocker: str
    arm_detail: str
    cell_key: str
    bar: Bar
    pod: str
    sm: str
    commit: str
    date: str
    parity: Optional[Parity] = None
    compiled: Optional[Leg] = None
    eager: Optional[Leg] = None

    @property
    def speedup(self) -> Optional[float]:
        if self.compiled is None or self.eager is None:
            return None
        if self.compiled.median <= 0:
            return None
        return self.eager.median / self.compiled.median

    @property
    def status(self) -> str:
        """`never-run` when no cell served, else proven/failed on the legs.

        A blocked family and a family whose mint refused are the same record
        state — nothing was measured against a compiled arm — and they differ
        only in the blocker they cite.
        """
        if self.arm in (ARM_BLOCKED, ARM_EAGER):
            return STATUS_NEVER_RUN
        speedup = self.speedup
        if self.parity is None or not self.parity.passed:
            return STATUS_FAILED
        if speedup is None or speedup < self.bar.min_speedup:
            return STATUS_FAILED
        return STATUS_PROVEN

    # -- the record ---------------------------------------------------------

    def proof_block(self) -> str:
        """The ``[proof]`` table, in `lint_author_ci.py`'s exact schema."""
        out = ["[proof]", f'status = "{self.status}"']
        if self.status == STATUS_NEVER_RUN:
            out.append(f'blocker = "{_toml_str(self.blocker)}"')
            out.append(f'note = "{_toml_str(self._never_run_note())}"')
            return "\n".join(out) + "\n"
        assert self.compiled is not None and self.eager is not None
        cosine = self.parity.cosine if self.parity else None
        out += [
            f'date = "{self.date}"',
            f'commit = "{_toml_str(self.commit)}"',
            f'pod = "{_toml_str(self.pod)}"',
            f'sm = "{_toml_str(self.sm)}"',
            f'cell = "{_toml_str(self.cell_key)}"',
            f"n = {len(self.compiled.samples)}",
            f"cosine = {cosine if cosine is not None else 0.0:.6f}",
            f"eager_median_ms = {self.eager.median:.1f}",
            f"compiled_median_ms = {self.compiled.median:.1f}",
            f"eager_p95_ms = {self.eager.p95:.1f}",
            f"compiled_p95_ms = {self.compiled.p95:.1f}",
            f'note = "{_toml_str(self._proof_note())}"',
        ]
        return "\n".join(out) + "\n"

    def _never_run_note(self) -> str:
        return (f"python -m gen_worker.author_ci on {self.pod or 'this pod'} "
                f"({self.sm or 'sm unknown'}): no compiled arm to compare — "
                f"{self.arm}, {self.arm_detail}. The eager arm ran and proves "
                f"nothing on its own, so no evidence is recorded here")

    def _proof_note(self) -> str:
        speedup = self.speedup or 0.0
        head = (f"python -m gen_worker.author_ci, arm={self.arm}, "
                f"bar={self.bar.min_speedup:g}x from {self.bar.source} on "
                f"{self.bar.metric}, measured {speedup:.3f}x")
        if self.status == STATUS_PROVEN:
            return head
        if self.parity is not None and not self.parity.passed:
            return f"{head}. PARITY FAILED: {self.parity.detail}"
        return (f"{head}. SPEED BELOW THE DECLARED BAR — fix the code or the "
                f"declaration; do not lower the bar to fit the number")

    # -- the human ----------------------------------------------------------

    def human(self) -> str:
        verdict = {
            STATUS_PROVEN: "PROVEN",
            STATUS_FAILED: "FAILED",
            STATUS_NEVER_RUN: "NO PROOF",
        }[self.status]
        rows = [
            f"author-ci: {self.family} — {verdict}",
            f"  rig     {self.pod or '?'} / {self.sm or '?'} @ "
            f"{self.commit or '?'}",
            f"  mint    {self.arm}: {self.arm_detail}",
        ]
        if self.parity is not None:
            cosine = ("?" if self.parity.cosine is None
                      else f"{self.parity.cosine:.5f}")
            rows.append(
                f"  parity  {'HEALTHY' if self.parity.passed else 'REFUSED'} "
                f"cos={cosine} floor={self.parity.floor_source} "
                f"({self.parity.detail})")
        if self.compiled is not None:
            rows.append(f"  speed   {self.compiled.line()}")
        if self.eager is not None:
            rows.append(f"  speed   {self.eager.line()}")
        speedup = self.speedup
        if speedup is not None:
            rows.append(
                f"  speedup {speedup:.3f}x against a declared bar of "
                f"{self.bar.min_speedup:g}x ({self.bar.source}, "
                f"{self.bar.metric})")
        rows.append(
            "  NOTE    this run is ADVISORY. Publish promotion is the "
            "platform's own run on trusted hardware (th#1811, ie#664 §6).")
        return "\n".join(rows)


def _toml_str(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', "'").replace("\n", " ")


# ---------------------------------------------------------------------------
# splicing the record — the harness owns [proof] and nothing else
# ---------------------------------------------------------------------------

def splice_proof(existing: str, block: str) -> str:
    """Replace the ``[proof]`` table, keeping everything above it VERBATIM.

    ``[parity]`` and ``[speed]`` are the author's declarations and the bar this
    run was judged against; re-serializing them from a parse would silently
    drop their comments and their stated reasons.
    """
    lines = existing.splitlines(keepends=True)
    head: List[str] = []
    for index, line in enumerate(lines):
        if line.strip() == "[proof]":
            trailing = [
                row for row in lines[index + 1:]
                if row.strip().startswith("[") and row.strip().endswith("]")
            ]
            if trailing:
                raise HarnessError(
                    f"author-ci.toml has table(s) after [proof] "
                    f"({[t.strip() for t in trailing]}); this command replaces "
                    f"[proof] in place and will not reorder an author's file. "
                    f"Move [proof] last, or drop --write and paste the block")
            return "".join(head) + block
        head.append(line)
    tail = "" if existing.endswith("\n") or not existing else "\n"
    return existing + tail + "\n" + block


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

class _Subject:
    """The endpoint under measurement: loaded once, invoked many times.

    Everything here routes through ``cli.run`` — the same module loading,
    function selection, model resolution, ``setup()`` and dispatch the local
    worker uses. A harness with its own loader would be measuring a program
    the author does not ship.
    """

    def __init__(self, args: argparse.Namespace, root: Path) -> None:
        from .cli import run as run_mod

        self._run = run_mod
        self.root = root
        self._args = args
        self._loaded: Dict[str, Any] = {}
        self._setup_done = False
        self.selected: Any = None
        self.instance: Any = None
        self.raw_bytes = run_mod._read_payload_bytes(
            inline=args.payload, path=args.payload_file)

    def load(self) -> None:
        run_mod = self._run
        _root, module = run_mod.load_endpoint_module(
            config_path=self._args.config_path, module=self._args.module)
        self.selected = run_mod._select_function(
            run_mod.discover_candidates(module),
            cls_name=self._args.cls_name,
            method_name=self._args.method_name,
            default_name=getattr(module, "__name__", "").split(".", 1)[0])
        if self._args.execution_lane:
            self.selected.bindings = run_mod._apply_execution_lane_to_bindings(
                self.selected.bindings, self._args.execution_lane)
        self.raw_bytes = run_mod._apply_field_tokens(
            self.raw_bytes, self._args.fields, self.selected.payload_type)
        self.instance = run_mod.instantiate_class(self.selected.cls)

    @property
    def declaration(self) -> Any:
        """The endpoint's ``Compile``, or None when it declares no compile."""
        decl = getattr(type(self.instance), "__gen_worker_endpoint__", None)
        return getattr(decl, "compile", None) if decl is not None else None

    def invoke(self, stage: str) -> float:
        """One request through the real dispatch; the stage's own ms.

        ``setup()`` runs on the first invocation exactly as ``gen-worker run``
        does it — which is also where the mint (or the adopt) happens, and it
        is why the first request of the compiled arm is the discarded one.
        """
        run_mod = self._run
        ctx = run_mod.build_local_context(
            kind=self.selected.kind, allow_publish=False)
        ctx._set_execution_lane(run_mod._local_executing_execution_lane(
            self.selected.bindings, self._args.execution_lane,
            self.selected.handles))
        timer = getattr(ctx, "_stages", None)
        started = time.monotonic()
        if timer is not None:
            timer.handler_open()
        run_mod.dispatch_request(
            selected=self.selected, instance=self.instance, ctx=ctx,
            raw_bytes=self.raw_bytes, offline=bool(self._args.offline),
            emit=run_mod._stderr_emitter,
            write_event=lambda kind, value: None,
            on_resolved=self._on_resolved,
        )
        if timer is not None:
            timer.handler_close()
        runtime_ms = int((time.monotonic() - started) * 1000)
        stages = stage_timing.stage_ms_for_metrics(timer, runtime_ms)
        if stage not in stages:
            raise HarnessError(
                f"the declared metric names stage {stage!r}, which this "
                f"handler does not bracket. It recorded "
                f"{sorted(k for k in stages if '.' not in k)!r} — declare one "
                f"of those in [speed] metric, or bracket the compute stage "
                f"with `with ctx.stage({stage!r}):` in the endpoint")
        return float(stages[stage])

    def _on_resolved(self, resolved: Dict[str, str]) -> None:
        if self._setup_done:
            return
        self._setup_done = True
        self._loaded = self._run.run_setup(
            self.instance, resolved,
            device=str(self._args.device or ""),
            selected=self.selected, return_loaded=True) or {}

    def armed_pipeline(self) -> Optional[Any]:
        """The loaded slot currently serving a compiled cell, if any."""
        from . import aot_serve

        for obj in self._loaded.values():
            if aot_serve.armed_targets(obj):
                return obj
        return None

    def shutdown(self) -> None:
        fn = getattr(self.instance, "shutdown", None)
        if callable(fn):
            try:
                fn()
            except Exception:  # noqa: BLE001 — teardown never decides a verdict
                pass


def measure_leg(subject: _Subject, name: str, stage: str, n: int) -> Leg:
    """``n + 1`` requests; the first is DISCARDED and reported as discarded."""
    first = subject.invoke(stage)
    samples = tuple(subject.invoke(stage) for _ in range(n))
    return Leg(name=name, samples=samples, discarded=first)


def read_parity(subject: _Subject, declaration: Any,
                minted: Optional[Any]) -> Parity:
    """The mint-parent gate's verdict — read, or (on an ADOPT) taken.

    A cell this process MINTED was already gated strictly on the way to
    publication (pgw#1141), so its report (``minted``) is simply read back — an
    armed cell that had failed that gate would not be armed. A cell that came
    out of this machine's store was gated at ITS mint and adoption runs no
    quality gate (§4.32 item 3), so there is no verdict to read and this run
    takes one through the SAME function the mint uses. Never a comparison
    re-implemented here.
    """
    from . import numerics_probe
    from .models import provision

    pipe = subject.armed_pipeline()
    if pipe is None:
        return Parity(False, None, "?", "nothing armed")
    report = minted
    if report is None:
        passed = provision.gate_cell_numerics(pipe, declaration, strict=True)
        report = numerics_probe.last_report()
        detail = "taken by this run through the mint's own gate (adopted cell)"
    else:
        passed = True
        detail = "the mint-parent gate's own verdict (pgw#1141)"
    if report is None:
        return Parity(False, None, "?",
                      "the gate ran and produced no report — refusing to call "
                      "an unmeasurable cell healthy (pgw#868)")
    comparison = report.comparison()
    return Parity(
        passed=bool(passed),
        cosine=None if comparison is None else float(comparison.cosine),
        floor_source=report.threshold_source,
        detail=f"{detail}; {report.context()[:400]}")


def preflight(root: Path) -> Dict[str, Any]:
    """``assert_fleet_line`` first, always. Raises rigcheck's own exceptions."""
    return rigcheck.assert_fleet_line(
        "author-ci", start=root, stream=sys.stderr)


def _sm(env: Dict[str, Any]) -> str:
    raw = str(env.get("sm") or "").strip()
    return f"sm_{raw.replace('.', '')}" if raw else ""


def _commit(root: Path, override: str) -> str:
    if override:
        return override
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10.0, check=False)
    except (OSError, subprocess.SubprocessError):
        out = None
    commit = (out.stdout.strip() if out is not None and out.returncode == 0
              else "")
    if not commit:
        raise HarnessError(
            "cannot read the endpoint commit under measurement (no git here?) "
            "— pass --commit <sha>. A proof that cannot name WHICH code it "
            "measured is not re-derivable and the record refuses it")
    return commit


def run(args: argparse.Namespace) -> Tuple[int, Report]:
    """The whole checklist. Returns ``(exit code, report)``."""
    # The fleet line is asserted before the endpoint's own code is imported,
    # let alone loaded: a rig off the line has produced no evidence, so there
    # is nothing to gain by getting further first.
    root = (Path.cwd().resolve() if args.module
            else load_project_config(args.config_path).root)
    env = preflight(root)
    subject = _Subject(args, root)
    record_path = (Path(args.record) if args.record
                   else subject.root / "author-ci.toml")
    if not record_path.is_file():
        raise HarnessError(
            f"{record_path} does not exist. The DECLARATION half of the record "
            f"— [parity] floor and [speed] metric/min_speedup — is the "
            f"author's to write and is what this run is judged against; a "
            f"harness that invented its own bar would be marking its own "
            f"homework. Copy the schema from AUTHOR-CI.md")
    record = tomllib.loads(record_path.read_text(encoding="utf-8"))

    subject.load()
    declaration = subject.declaration
    if declaration is None:
        raise HarnessError(
            "this endpoint declares no `compile=`, so there is no compiled "
            "arm to compare against eager and nothing for author CI to prove")
    bar = resolve_bar(declaration, record)
    n = max(MIN_SAMPLES, int(args.n))
    # Asked BEFORE the legs: a pod hour spent producing a proof that cannot
    # name the code it measured is a pod hour thrown away.
    commit = _commit(subject.root, args.commit)

    family = str(getattr(declaration, "family", "") or "")
    blocked = open_blockers(declaration)
    arm = ARM_EAGER
    blocker = "ie#664"
    arm_detail = "no compiled arm"
    if blocked:
        # §4.32 / ie#664 §6: a declared block is a LEGAL state, not a failure.
        # Engaging the order makes the eager-only run STRUCTURAL rather than
        # hopeful — nothing adopts, mints or arms behind our back.
        arm = ARM_BLOCKED
        blocker = ", ".join(b.id for b in blocked)
        arm_detail = f"{len(blocked)} unresolved Compile.blockers: {blocker}"
        serve_posture.apply_command(
            True, actor="author-ci",
            reason=f"{family} declares open mint blockers: {blocker}")
        print(blocker_refusal(family, blocked), file=sys.stderr)

    parity: Optional[Parity] = None
    compiled: Optional[Leg] = None
    eager: Optional[Leg] = None
    cell_key = ""
    try:
        if arm == ARM_BLOCKED:
            eager = measure_leg(subject, ARM_EAGER_ARM, bar.stage, n)
        else:
            from . import aot_serve, numerics_probe

            # The compiled arm runs FIRST: its discarded request is the one
            # that runs setup(), and setup() is where the mint happens.
            # Identity, not presence: the gate's report is a process-wide
            # record, and reading a PREVIOUS run's would attribute somebody
            # else's cosine to this cell.
            before = numerics_probe.last_report()
            compiled = measure_leg(subject, ARM_COMPILED, bar.stage, n)
            pipe = subject.armed_pipeline()
            if pipe is None:
                arm, compiled = ARM_EAGER, None
                arm_detail = (
                    "no cell armed on this pod, so there is nothing to "
                    "compare — the reason is on the activity wire "
                    "(self_mint_skipped / self_mint_abort)")
            else:
                cell_key = str(aot_serve.armed_metadata(pipe).get("cell_key")
                               or "")
                fresh = numerics_probe.last_report()
                minted = fresh if fresh is not before else None
                arm = ARM_MINTED if minted is not None else ARM_ADOPTED
                arm_detail = f"cell {cell_key or '?'}"
                parity = read_parity(subject, declaration, minted)
            # pgw#1142 through its IN-PROCESS api: the author's pod may have
            # no hub, and the order is what makes this the SAME process, the
            # same weights and the same pipeline answer eager.
            serve_posture.apply_command(
                True, actor="author-ci",
                reason="the eager arm of the §4.32 speed leg")
            eager = measure_leg(subject, ARM_EAGER_ARM, bar.stage, n)
    finally:
        serve_posture.apply_command(
            False, actor="author-ci", reason="the speed leg is finished")
        subject.shutdown()

    report = Report(
        family=family, arm=arm, blocker=blocker, arm_detail=arm_detail,
        cell_key=cell_key, bar=bar, pod=str(env.get("device") or ""),
        sm=_sm(env), commit=commit,
        date=datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
        parity=parity, compiled=compiled, eager=eager)

    block = report.proof_block()
    if args.write:
        record_path.write_text(
            splice_proof(record_path.read_text(encoding="utf-8"), block),
            encoding="utf-8")
        print(f"author-ci: wrote [proof] into {record_path}", file=sys.stderr)
    else:
        sys.stdout.write(block)
    print(report.human(), file=sys.stderr)
    return (EXIT_OK if report.status == STATUS_PROVEN else EXIT_FAILED), report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m gen_worker.author_ci",
        description=(
            "Run ie#664's tier-2 author-CI checklist on THIS pod, in the "
            "endpoint's own image, and emit the author-ci.toml [proof] "
            "record. Measures; never gates publish (ie#664 §6)."),
    )
    # The selection flags are `gen-worker run`'s, spelled identically: an
    # author who can invoke their endpoint can measure it.
    p.add_argument("--class", dest="cls_name", default=None,
                   help="Class to invoke (inferred when only one exists).")
    p.add_argument("--method", dest="method_name", default=None,
                   help="Method to invoke (inferred when only one exists).")
    p.add_argument("--payload", dest="payload", default=None,
                   help="Inline JSON payload. One payload, both arms.")
    p.add_argument("--payload-file", dest="payload_file", default=None,
                   help="Read the JSON payload from this file.")
    p.add_argument("--config", dest="config_path", default=None,
                   help="Path to the endpoint's pyproject.toml.")
    p.add_argument("--module", dest="module", default=None,
                   help="Python module to import (overrides [tool.gen_worker]).")
    p.add_argument("--offline", action="store_true",
                   help="Use only the local CAS; never fetch weights.")
    p.add_argument("--device", dest="device", default=None,
                   help="Override the torch device (e.g. 'cuda:0').")
    p.add_argument("--lane", dest="execution_lane", default="",
                   help="Execution lane (th#913 dual-form), as `run` takes it.")
    p.add_argument("--n", dest="n", type=int, default=MIN_SAMPLES,
                   help=f"Steady-state requests per arm (>= {MIN_SAMPLES}). "
                        f"One more is run per arm and discarded.")
    p.add_argument("--record", dest="record", default=None,
                   help="The author-ci.toml to judge against (default: the "
                        "endpoint root's).")
    p.add_argument("--commit", dest="commit", default="",
                   help="The endpoint commit measured (default: git HEAD).")
    p.add_argument("--write", action="store_true",
                   help="Splice the [proof] block into the record in place. "
                        "Default: print it on stdout.")
    p.add_argument("fields", nargs="*", metavar="FIELD=VALUE",
                   help="Payload args, exactly as `gen-worker run` takes them.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """0 proven, 1 measured-and-failed, 2 usage, 90/91 rigcheck's own."""
    from .cli.run import _UsageError

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        code, _report = run(args)
        return code
    except rigcheck.CudaUnusable as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return EXIT_HOST_UNUSABLE
    except (rigcheck.FleetLineMismatch, rigcheck.FleetLineUnknown) as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return EXIT_RIG_OFF_LINE
    except (HarnessError, _UsageError, FileNotFoundError, ValueError) as exc:
        print(f"author-ci: {exc}", file=sys.stderr, flush=True)
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
