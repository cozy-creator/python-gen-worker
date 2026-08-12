#!/usr/bin/env python3
"""pgw#1152: a process-global fact about an ARMED cell is fed at a SEAM, never
by a caller who remembered — and a TEST may not feed one at all.

THE DEFECT CLASS. ``aot_serve._KNOWN_AOT_KEYS`` decides which failure detector
applies to a serving object: the DYNAMO lane's per-class cache-hit ledger (which
an AOTI artifact can never move, so it reads every honest adoption as a disproof)
or §4.31's serve-first rule. pgw#1033 wrote its feeding rule as a CONVENTION —
*"whoever reads a `cell_key` off an aot-inductor envelope registers it"* — and it
had exactly two feeders, ``fleet_cells.arm_from_local_store`` and
``adopt_delegated_mint``, **both SELF-PRODUCED routes**. ``arm_ordered`` (every
hub Plan arm and every §4.27 boot-adopt) fed it nothing. Measured on a real pod
(pgw#1141b, POD PROOF #4): a cell that resolved, materialized and armed was
scored on the dynamo ledger, disarmed, and the pod served eager for life.

That was the FOURTH gate in a row with the same shape (pgw#1108, pgw#1122,
pgw#1141, pgw#1141b): a rule written against the self-mint path, and an adopt
path that does not take it. So the rule is now structural — the registration
happens inside ``aot_serve.arm_entry``, the one function every arm route
passes — and this script keeps it that way.

pgw#1176 MOVED that seam and did not weaken it. ``load_and_wrap`` armed a whole
36-entry cell; ``arm_entry`` arms ONE graph class, and every arm route still
passes through it exactly once per class. The convention is unchanged: the
registration is a property of the function that arms, never of a caller
remembering to announce.

TWO SCOPES, TWO RULES.

**src/gen_worker** — every call to a listed feeder must be either the declared
SEAM itself, or a row in ``scripts/arm_state_feeders_allowlist.txt`` under one of
the classifications below. Unclassified is red; a stale row is red; and a SEAM
that stops calling its feeder is red (the registration is an asserted fact, not
a habit).

    SEAM      the one function every route passes; declared in FEEDERS below and
              exempt because it IS the structure. Never written in the allowlist.
    VERDICT   the site records a decision only this frame can know, and it sits
              AT that decision — no seam can derive it (a proof pass concluding
              a cell served its own graph; a disarm concluding one did not).
    OWNER     the module that owns the state, writing it from its own primary
              write path (a mint finalize putting its own bytes in its own
              store).

**There is deliberately NO ``CONVENTION`` classification, and no ``RELAY``.**
A feeder called because a docstring asked the caller to remember is exactly the
bug, twice. It has no label to write down: move the call to the seam, or delete
it and ask the OBJECT instead (``aot_serve.holds_exported_cell``, pgw#1141b).

**tests/** — a hand-feed is RED, and so is stubbing a lane ACCESSOR. pgw#1141
shipped thirteen green rows whose fleet-policy stand-ins called
``aot_serve.note_aot_key(key)`` BY HAND — production's single missing line — and
another suite stubbed ``is_aot_ref`` outright. Those fixtures answered the
question the gate was supposed to ask, so the tests entered one gate east of the
bug and RED-first prevented nothing. A test that needs an armed, boot-adopted
object drives ``tests/harness/adopt_rig.py``, which arms a real packed cell
through the real ordered path.

There is exactly ONE test classification, ``RECOGNIZER``, and **the lint checks
the claim instead of trusting it**: the row is accepted only when the enclosing
function DRIVES no arm and the module REPLACES no arm driver. A unit test of the
recognizer itself (no pipeline anywhere, the classifier is the subject) qualifies
structurally; a fixture standing in for an adoption cannot, whatever it writes in
the file. So a simulated arm still has no label to write down.

Same enforcement shape as ``lint_credential_identity.py`` (pgw#1122),
``lint_settings_writers.py`` (pgw#1049) and ``lint_config_reads.py`` (§1.18).
Sites are keyed ``<path>::<enclosing scope>::<name>``, never by line number: a
line is a fact other people change (pgw#931).
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]

CLASSIFICATIONS = {"VERDICT", "OWNER", "RECOGNIZER", "PROJECTION"}

#: Calling one of these means this frame ARMS something. A test that both arms
#: and hand-feeds is standing in for an adoption, which is the bug class.
ARM_DRIVERS = frozenset({
    "ensure_setup", "arm_aot", "arm_ordered", "enable_compiled",
    "_enable_compiled", "arm_entry", "adopt_delegated_mint",
    "arm_from_local_store", "self_mint",
})


@dataclass(frozen=True)
class Feeder:
    """One production-owned writer of arming/serving process state."""

    #: dotted spelling as other modules call it, e.g. ``aot_serve.note_aot_key``
    dotted: str
    #: the module that owns it, so a bare call inside that file counts too
    owner: str
    #: the state it feeds, for the message
    state: str
    #: ``(file, function)`` that is the structural seam, or None when the fact
    #: is a verdict no seam can derive
    seam: Optional[Tuple[str, str]] = None
    #: does a hand-feed in a TEST simulate an ARM? those are unwriteable
    arm_simulating: bool = True
    #: is the bare name distinctive enough to match on ANY receiver?
    distinctive: bool = True
    #: when it is not, the receiver spellings that count
    receivers: Tuple[str, ...] = ()


FEEDERS: Tuple[Feeder, ...] = (
    Feeder(
        dotted="aot_serve.note_aot_key", owner="aot_serve.py",
        state="aot_serve._KNOWN_AOT_KEYS (which lane a cell ref is on)",
        seam=("aot_serve.py", "arm_entry"),
    ),
    Feeder(
        dotted="compile_cache.record_cell_proven", owner="compile_cache.py",
        state="compile_cache._PROVEN_CELLS (this process served this cell)",
        seam=None,
    ),
    Feeder(
        dotted="compile_cache.record_cell_quarantined", owner="compile_cache.py",
        state="compile_cache._QUARANTINED_CELLS (this identity failed here)",
        seam=None, arm_simulating=False,
    ),
    Feeder(
        dotted="local_cell_store.store", owner="local_cell_store.py",
        state="this machine's own cell store (§4.28)",
        seam=None, arm_simulating=False,
        distinctive=False, receivers=("local_cell_store", "store"),
    ),
    Feeder(
        dotted="local_cell_store.note_memo", owner="local_cell_store.py",
        state="the pre-trace arm-token -> ck1 memo (§4.28)",
        seam=None, arm_simulating=False,
    ),
)

#: Reading these is how a frame asks WHICH LANE an object serves on. A test that
#: replaces one has answered the gate's question for it — the pgw#1141 fixture
#: sin, in its second form.
#: Scoped to the LANE-IDENTITY question — "is this the exported lane, and has
#: this process served this cell?" — not to the broader "is anything armed"
#: (``is_armed`` / ``is_compile_armed``), which is a different and much older
#: pattern this issue makes no claim about.
ACCESSORS: Dict[str, Tuple[str, ...]] = {
    "aot_serve": ("is_aot_ref", "holds_exported_cell", "proven_since"),
    "aot": ("is_aot_ref", "holds_exported_cell", "proven_since"),
    "compile_cache": ("cell_proven_in_process", "cell_quarantined_in_process"),
    "cc": ("cell_proven_in_process", "cell_quarantined_in_process"),
}

RIG = "tests/harness/adopt_rig.py"


@dataclass(frozen=True)
class OneConstructor:
    """A production object that must have exactly ONE map into it.

    THE THIRD VARIANT of the same bug class (pgw#1150, second pass). Two sites
    mapped a ``Compile`` onto a ``CompileCell`` — the registry's
    ``compile_cell()`` and ``cli.run``'s §4.28 desktop arm — and the
    ``numerics_floor``/``numerics_warn`` fields reached one and not the other,
    so a whole serving path was judged at an SDK default nobody chose while
    every record said ``declared``. The instance was a missing field; the CAUSE
    is that the same object had two independent constructors. Any place two
    constructors build one production object is a latent divergence of exactly
    that kind — and on the arming path the two constructors are usually "the
    self-mint/plan route" and "the adopt route", which is how four gates
    shipped.
    """

    #: the type name as it is CALLED
    name: str
    #: ``<file>::<scope>`` of the one constructor, which is exempt
    constructor: Tuple[str, str]
    #: only constructions passing one of these keywords are fenced; empty =
    #: every construction. (``_ArmOrder(backend=…)`` with no artifact is a
    #: complete answer for a dynamo/eager plan and maps nothing.)
    when_kwargs: Tuple[str, ...] = ()
    why: str = ""


ONE_CONSTRUCTOR: Tuple[OneConstructor, ...] = (
    OneConstructor(
        name="CompileCell",
        constructor=("registry.py", "CompileCell.from_declaration"),
        why="pgw#1150: a Compile field that reaches one map and not the other "
            "judges a whole path at a default nobody chose",
    ),
    OneConstructor(
        name="_ArmOrder",
        constructor=("executor.py", "_ArmOrder.for_artifact"),
        when_kwargs=("selection",),
        why="pgw#1152: the §4.27 boot-adopt order and the hub PLAN order were "
            "built independently and are field-for-field identical except "
            "`adopt` — the exact adopt-vs-plan asymmetry that produced "
            "pgw#1108/#1122/#1141/#1141b",
    ),
    OneConstructor(
        name="_CompileArtifactSelection",
        constructor=("executor.py", "_ArmOrder.for_artifact"),
        when_kwargs=(),
        why="pgw#1152: both order builders also built the selection from the "
            "same three fields; it is now built once, with the order",
    ),
    OneConstructor(
        name="ExpectedIdentity",
        constructor=("aot_identity.py", "ExpectedIdentity.named_by"),
        why="pgw#1152's last reported duplicate: this is the object EVERY arm "
            "gate compares a cell against, and its two expectation sources "
            "are the hub PLAN and the §4.27 pull-by-key resolve — the same "
            "plan-vs-adopt pair that produced pgw#1108/#1122/#1141/#1141b",
    ),
)


def _dotted(node: ast.AST) -> str:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


class _Calls(ast.NodeVisitor):
    """Every feeder call and every accessor stub in one module, scoped."""

    def __init__(self, owner_file: str) -> None:
        self.owner_file = owner_file
        self.feeds: List[Tuple[int, str, Feeder]] = []
        self.stubs: List[Tuple[int, str, str]] = []
        self.builds: List[Tuple[int, str, OneConstructor]] = []
        #: every function/class scope defined in this module, dotted
        #: enclosing scopes that call an arm driver
        self.arming_scopes: set = set()
        #: True when this module REPLACES an arm driver (a fixture standing in
        #: for the arm, which is how every one of these fixtures was built)
        self.replaces_arm = False
        self.defined_scopes: set = set()
        self._scope: List[str] = []

    def _where(self) -> str:
        return ".".join(self._scope) or "<module>"

    def _push(self, node: ast.AST, name: str) -> None:
        self._scope.append(name)
        self.defined_scopes.add(self._where())
        self.generic_visit(node)
        self._scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._push(node, node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._push(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._push(node, node.name)

    def visit_Call(self, node: ast.Call) -> None:
        path = _dotted(node.func)
        tail = path.rpartition(".")[2]
        for feeder in FEEDERS:
            bare = feeder.dotted.split(".")[-1]
            # Deliberately by NAME rather than by resolved receiver, except
            # where the name is too generic to be one: the receiver is
            # `aot_serve`, `aot`, `cc`, `fleet_cells.aot_serve` or a module
            # alias at different sites, and a fence that only catches the
            # spellings we thought of is not a fence (pgw#1122's rule).
            if tail != bare:
                continue
            receiver = path.rpartition(".")[0].rpartition(".")[2]
            if not receiver:
                if self.owner_file != feeder.owner:
                    continue
            elif not feeder.distinctive and receiver not in feeder.receivers:
                continue
            self.feeds.append((node.lineno, self._where(), feeder))
        if tail in ARM_DRIVERS:
            self.arming_scopes.add(self._where())
        for one in ONE_CONSTRUCTOR:
            if tail != one.name:
                continue
            if one.when_kwargs and not any(
                kw.arg in one.when_kwargs for kw in node.keywords
            ):
                continue
            self.builds.append((node.lineno, self._where(), one))
        self._note_stub(node, path)
        self.generic_visit(node)

    def _note_stub(self, node: ast.Call, path: str) -> None:
        if not path.endswith("setattr") or len(node.args) < 2:
            return
        owner = _dotted(node.args[0]).split(".")[-1]
        name = node.args[1]
        if not isinstance(name, ast.Constant) or not isinstance(name.value, str):
            return
        if name.value in ARM_DRIVERS:
            self.replaces_arm = True
        if name.value in ACCESSORS.get(owner, ()):
            self.stubs.append(
                (node.lineno, self._where(), f"{owner}.{name.value}"))


def _iter_modules(root: Path) -> List[Path]:
    return [p for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts]


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def scan_src(root: Path) -> Tuple[Dict[Tuple[str, str], int], List[str]]:
    """Feeder sites outside the declared seams, plus seam-integrity problems."""
    sites: Dict[Tuple[str, str], int] = {}
    problems: List[str] = []
    seams_seen = {f.seam: False for f in FEEDERS if f.seam}
    constructors_seen = {o.name: False for o in ONE_CONSTRUCTOR}
    for path in _iter_modules(root):
        calls = _Calls(path.name)
        calls.visit(ast.parse(path.read_text(encoding="utf-8")))
        rel = _rel(path)
        for lineno, scope, feeder in calls.feeds:
            if feeder.seam and (path.name, scope) == feeder.seam:
                seams_seen[feeder.seam] = True
                continue
            sites.setdefault((rel, f"{scope}::{feeder.dotted}"), lineno)
        for one in ONE_CONSTRUCTOR:
            if one.constructor[0] == path.name and (
                one.constructor[1] in calls.defined_scopes
            ):
                constructors_seen[one.name] = True
        for lineno, scope, one in calls.builds:
            if (path.name, scope) == one.constructor:
                continue
            sites.setdefault((rel, f"{scope}::{one.name}()"), lineno)
    for name, seen in constructors_seen.items():
        where = dict((o.name, o.constructor) for o in ONE_CONSTRUCTOR)[name]
        # Only demanded when the owning module is in the scanned tree at all,
        # so the rule can be exercised over a synthetic tree.
        if not seen and (root / where[0]).is_file():
            problems.append(
                f"ONE-CONSTRUCTOR MISSING: {where[0]}::{where[1]} — the one "
                f"map into {name} — does not exist. That map IS the fence; if "
                "it moved, move it in ONE_CONSTRUCTOR here too.")
    for seam, seen in seams_seen.items():
        if seen:
            continue
        feeder = next(f for f in FEEDERS if f.seam == seam)
        problems.append(
            f"SEAM BROKEN: {seam[0]}::{seam[1]} no longer calls "
            f"{feeder.dotted}. That call is what feeds {feeder.state} for "
            "EVERY arm route — with it gone, a new route is one forgotten "
            "convention away from pgw#1141b (a resolved, materialized, armed "
            "cell scored on the dynamo ledger and thrown away on a real pod). "
            "Put it back, or move the seam and update FEEDERS here.")
    return sites, problems


def scan_tests(
    root: Path, allowed: Dict[Tuple[str, str], str],
) -> Tuple[Dict[Tuple[str, str], int], List[str]]:
    """A test may not feed a production registry, nor stub a lane accessor.

    The one escape, ``RECOGNIZER``, is CHECKED rather than trusted: the row is
    accepted only when the enclosing function drives no arm and the module
    replaces no arm driver. A fixture standing in for an adoption fails that
    test whatever it writes in the allowlist.
    """
    sites: Dict[Tuple[str, str], int] = {}
    problems: List[str] = []
    for path in _iter_modules(root):
        if _rel(path) == RIG:
            # The rig is the sanctioned vehicle; it is fenced by
            # `test_adopt_rig_pgw1152.py::test_the_rig_itself_registers_nothing`,
            # which reads its source and refuses a registration.
            continue
        calls = _Calls(path.name)
        calls.visit(ast.parse(path.read_text(encoding="utf-8")))
        rel = _rel(path)

        def _recognizer_ok(scope: str) -> Optional[str]:
            """None when the RECOGNIZER claim holds, else why it does not."""
            if calls.replaces_arm:
                return ("this module REPLACES an arm driver, so it is standing "
                        "in for the arm — the claim is refused")
            if scope in calls.arming_scopes:
                return f"{scope} drives an arm itself — the claim is refused"
            return None

        def _judge(lineno: int, scope: str, name: str, complaint: str) -> None:
            key = (rel, f"{scope}::{name}")
            label = allowed.get(key)
            if label == "RECOGNIZER":
                refusal = _recognizer_ok(scope)
                if refusal is None:
                    sites[key] = lineno
                    return
                problems.append(
                    f"{rel}:{lineno}: RECOGNIZER claimed for {scope}::{name}, "
                    f"but {refusal}. Drive {RIG}.")
                return
            if label is not None:
                problems.append(
                    f"{rel}:{lineno}: {label} is not a TEST classification for "
                    f"{scope}::{name} — the only one is RECOGNIZER.")
                return
            problems.append(f"{rel}:{lineno}: {complaint}")

        for lineno, scope, feeder in calls.feeds:
            if not feeder.arm_simulating:
                continue
            _judge(
                lineno, scope, feeder.dotted,
                f"a TEST hand-feeds {feeder.dotted} in {scope} — that writes "
                f"{feeder.state}, which is exactly the fact production writes "
                "at its seam. pgw#1141 shipped 13 green rows whose stand-ins "
                "did this; they entered one gate east of the bug. Drive "
                f"{RIG} instead.")
        for lineno, scope, name in calls.stubs:
            _judge(
                lineno, scope, name,
                f"a TEST STUBBED the lane accessor {name} in {scope} — the "
                "gate under test then answers from the fixture rather than "
                f"from the object. Drive {RIG} and let a real arm make the "
                "answer true.")
    return sites, problems


def load_allowlist(path: Path) -> Tuple[Dict[Tuple[str, str], str], List[str]]:
    """Parse ``<path>::<scope>::<name>  <CLASSIFICATION>  <reason>`` lines."""
    allowed: Dict[Tuple[str, str], str] = {}
    errors: List[str] = []
    if not path.is_file():
        return allowed, [f"{path} is missing"]
    for num, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            errors.append(
                f"{path.name}:{num}: need '<path>::<scope>::<name> "
                f"<CLASSIFICATION> <reason>', got {line!r}")
            continue
        key, classification = parts[0], parts[1]
        if "::" not in key:
            errors.append(f"{path.name}:{num}: site key {key!r} lacks '::'")
            continue
        if classification not in CLASSIFICATIONS:
            errors.append(
                f"{path.name}:{num}: unknown classification "
                f"{classification!r} (want one of {sorted(CLASSIFICATIONS)}). "
                "There is no CONVENTION class and no RELAY class: a feeder "
                "called because the caller was asked to remember is the bug "
                "itself (pgw#1033 -> pgw#1141b). Move it to the seam, or "
                "delete it and ask the object.")
            continue
        file_part, name = key.split("::", 1)
        allowed[(file_part, name)] = classification
    return allowed, errors


def check(
    sites: Dict[Tuple[str, str], int],
    allowed: Dict[Tuple[str, str], str],
    seen: Optional[Dict[Tuple[str, str], int]] = None,
) -> List[str]:
    problems: List[str] = []
    for (rel, name), lineno in sorted(sites.items()):
        if allowed.get((rel, name)) == "RECOGNIZER":
            problems.append(
                f"{rel}:{lineno}: RECOGNIZER is a TEST classification and this "
                "is production code — a src feeder is SEAM, VERDICT or OWNER.")
            continue
        if name.endswith("()"):
            if (rel, name) in allowed:
                continue
            one = next(o for o in ONE_CONSTRUCTOR if o.name == name.rpartition("::")[2][:-2])
            problems.append(
                f"{rel}:{lineno}: SECOND MAP into {one.name} at {name} — "
                f"{one.constructor[0]}::{one.constructor[1]} is the one "
                f"constructor. {one.why}. Call it, or — if this builds the "
                "object from a DIFFERENT source shape and is therefore not a "
                "second copy of the same map — classify it PROJECTION in "
                "scripts/arm_state_feeders_allowlist.txt. There is no label "
                "for writing the same map twice.")
            continue
        if (rel, name) not in allowed:
            problems.append(
                f"{rel}:{lineno}: UNCLASSIFIED arm-state feed: {name} — this "
                "writes a process-global fact about an armed cell from "
                "somewhere that is not the seam every arm route passes. That "
                "is pgw#1033's convention, and the route that did not keep it "
                "cost four pods and four gates (pgw#1141b). Move it to the "
                "seam, delete it in favour of asking the object, or classify "
                "it in scripts/arm_state_feeders_allowlist.txt")
    matched = set(sites) | set(seen or {})
    for key in sorted(set(allowed) - matched):
        problems.append(
            f"stale allowlist row {key[0]}::{key[1]} matches no feed site — "
            "delete it (a row matching nothing is a boundary that lies)")
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=REPO / "src" / "gen_worker")
    ap.add_argument("--tests", type=Path, default=REPO / "tests")
    ap.add_argument(
        "--allowlist", type=Path,
        default=REPO / "scripts" / "arm_state_feeders_allowlist.txt")
    args = ap.parse_args(argv)

    sites, seam_problems = scan_src(args.src)
    allowed, errors = load_allowlist(args.allowlist)
    test_sites, test_problems = scan_tests(args.tests, allowed)
    problems = (seam_problems + errors
                + check(sites, allowed, seen=test_sites) + test_problems)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"arm-state fence: {len(sites)} classified production feed(s), "
          f"{len([f for f in FEEDERS if f.seam])} structural seam(s) intact, "
          f"{len(test_sites)} RECOGNIZER row(s); no test simulates an arm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
