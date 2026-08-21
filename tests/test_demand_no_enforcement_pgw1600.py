"""pgw#1600 acceptance (d): ZERO admission decisions consume the demand number.

**The falsifier ships before the enforcer.** A formula that has never been
falsified is a guess, and a guess wired into an admission predicate is a guess
that kills daemons — on the compiled path admission is the ONLY safety
mechanism (pgw#1255 leg 2: a mid-graph OOM inside a compiled artifact is
process death, uncatchable). So this issue ships the number, its serialization
and its falsifier, and NOTHING reads it to decide anything. pgw#1601 wires the
mint-time stamp into `headroom_admits`; pgw#1602 wires the grant.

The absence is asserted MECHANICALLY, over the AST, because prose does not go
red. The guard fails the moment an admission module imports the demand plane —
which is exactly what pgw#1601's first commit will do, and at that point this
file is edited in the same commit as the wiring, deliberately and visibly.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator

import gen_worker

SRC = Path(gen_worker.__file__).resolve().parent

#: The three modules that ARE the demand plane. Nothing else may reach them
#: except the two producers named below.
DEMAND_PLANE = {
    "gen_worker.demand",
    "gen_worker.demand_envelope",
    "gen_worker.demand_falsifier",
}

#: THE ADMISSION SURFACE — every module in this repository that decides
#: whether something may be placed, loaded, admitted or run. If the demand
#: number ever reaches a decision, it reaches it through one of these.
ADMISSION_SURFACE = (
    "models/partial_resident.py",   # headroom_admits / probe_plan
    "models/memory.py",             # the reserve and the oom ladder
    "models/grant.py",              # the arena grant
    "models/residency.py",          # leases and activation hints
    "serving/residency.py",         # admission_charge, NeverFits
    "serving/placement.py",         # warn_if_degraded
    "serving/lane_ladder.py",       # the owner-ranked (GPU, lane) ladder
    "serving/loader.py",            # boot-time lane selection
)

#: The ONLY modules allowed to import the demand plane while this issue's
#: posture holds: the two PRODUCERS (serialize it, measure against it) and the
#: package's own re-export surface.
PERMITTED_IMPORTERS = {
    "gen_worker.release.derive",       # serializes it into the document
    "gen_worker.serving.serve_loop",   # measures the request arena, banks nothing
    "gen_worker.worker",               # adds the regime and calls the falsifier
    "gen_worker.demand_envelope",
    "gen_worker.demand_falsifier",
    # The DECLARATION surface: `lane(request=...)` names the type so an author
    # cannot pass anything else. It reads no number and evaluates nothing.
    "gen_worker.serving.lane_spec",
}


def _modules() -> Iterator[tuple[str, ast.Module]]:
    for path in sorted(SRC.rglob("*.py")):
        if "_vendor" in path.parts:
            continue
        rel = path.relative_to(SRC)
        dotted = "gen_worker." + ".".join(rel.with_suffix("").parts)
        dotted = dotted.removesuffix(".__init__")
        yield dotted, ast.parse(path.read_text(encoding="utf-8"), str(path))


def _imported_demand_modules(dotted: str, tree: ast.Module) -> set[str]:
    package = dotted.rsplit(".", 1)[0] if "." in dotted else dotted
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in DEMAND_PLANE:
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package
                for _ in range(node.level - 1):
                    base = base.rsplit(".", 1)[0]
                target = f"{base}.{node.module}" if node.module else base
            else:
                target = node.module or ""
            if target in DEMAND_PLANE:
                found.add(target)
            for alias in node.names:
                if f"{target}.{alias.name}" in DEMAND_PLANE:
                    found.add(f"{target}.{alias.name}")
    return found


def test_no_admission_module_reaches_the_demand_plane_AT_ALL() -> None:
    offenders: dict[str, set[str]] = {}
    for dotted, tree in _modules():
        relative = dotted.removeprefix("gen_worker.").replace(".", "/") + ".py"
        if relative not in ADMISSION_SURFACE:
            continue
        found = _imported_demand_modules(dotted, tree)
        if found:
            offenders[relative] = found
    assert offenders == {}, (
        f"an ADMISSION module now imports the demand plane: {offenders}. "
        f"pgw#1600 ships the number with provably zero consumers — the "
        f"falsifier before the enforcer. If this is pgw#1601's stamp wiring, "
        f"edit this guard in the same commit as the wiring, deliberately."
    )


def test_the_admission_surface_this_guard_names_still_EXISTS() -> None:
    """A guard whose subjects were renamed away is a guard that cannot fail."""

    missing = [name for name in ADMISSION_SURFACE if not (SRC / name).is_file()]
    assert missing == [], (
        f"the demand no-enforcement guard names modules that no longer exist: "
        f"{missing} — re-point it at wherever the decision moved, or it is "
        f"proving nothing"
    )


def test_only_the_declared_producers_import_the_demand_plane() -> None:
    importers = {
        dotted for dotted, tree in _modules()
        if dotted not in DEMAND_PLANE and _imported_demand_modules(dotted, tree)
    }
    assert importers <= PERMITTED_IMPORTERS, (
        f"unexpected demand-plane importer(s): "
        f"{sorted(importers - PERMITTED_IMPORTERS)}. Every consumer of this "
        f"number is a design decision, not an import."
    )


def test_headroom_admits_still_takes_a_demand_it_is_never_GIVEN() -> None:
    """The socket exists and is INERT — pgw#1627 wired it, pgw#1601 fills it.

    Asserted positively so the day it stops being true is a red test rather
    than a silent behaviour change: every production call site passes either
    nothing or a literal zero.
    """

    from gen_worker.models import partial_resident

    tree = ast.parse(
        (SRC / "models" / "partial_resident.py").read_text(encoding="utf-8")
    )
    passed: list[ast.expr] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name != "headroom_admits":
            continue
        for keyword in node.keywords:
            if keyword.arg == "demand_bytes":
                passed.append(keyword.value)

    assert passed, "headroom_admits is no longer called with demand_bytes at all"
    for value in passed:
        assert isinstance(value, ast.Name) and value.id == "demand_bytes", (
            f"a production call site now computes demand_bytes "
            f"({ast.dump(value)}) — that is pgw#1601's wiring, not pgw#1600's"
        )

    # And the parameter's DEFAULT is what production actually gets, because
    # nothing upstream has a stamp to pass yet.
    assert partial_resident.headroom_admits(
        regime="compiled", free_bytes=1 << 30, cache_bytes=0, floor_bytes=0,
    ) == (True, "driver_free")


def test_the_release_derive_is_the_ONLY_place_the_document_is_built() -> None:
    """One producer of the serialized formula, so a second cannot drift."""

    callers = set()
    for dotted, tree in _modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = (
                    getattr(node.func, "id", None)
                    or getattr(node.func, "attr", None)
                )
                if name == "demand_document":
                    callers.add(dotted)
    assert callers == {"gen_worker.release.derive"}, (
        "the demand block is built in exactly one place; a second builder "
        "is a second answer to what a lane demands"
    )


def test_the_guard_ITSELF_can_go_red() -> None:
    """Falsify the instrument: point it at a module that DOES import the
    plane and prove it notices. A fence nobody has seen fail is a fence
    nobody knows is armed."""

    caught = {
        dotted for dotted, tree in _modules()
        if dotted == "gen_worker.serving.serve_loop"
        and _imported_demand_modules(dotted, tree)
    }
    assert caught == {"gen_worker.serving.serve_loop"}, (
        "the detector did not see a REAL relative import of the demand "
        "plane — every green above is meaningless"
    )

