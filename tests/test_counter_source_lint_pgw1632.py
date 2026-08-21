"""pgw#1632(b): a byte counter cannot exist without declaring its instrument.

`load:staged_bytes` measured `/proc/self/io` **read_bytes** plus anon-RSS
growth. Nothing about the NUMBER was wrong. The NAME said a write had happened,
th#2246's first mechanism lane believed it, and went looking for a per-child
write that does not exist anywhere in this repo.

The fence is two rules over one registry (`gen_worker.byte_sources`):

* **classification** — every byte counter created in `src/` resolves to a
  declared `Source`, and a registry entry nothing creates is deleted rather
  than left as decoration;
* **admissibility** — a counter's verb must suit the direction of its source.
  `staged`/`written`/`flushed`/`uploaded` need a write-side instrument;
  `read`/`ingested`/`pulled` need a read-side one; `freed`/`consumed` need the
  filesystem.

The scan is over the AST, not over a grep, because the names are built at the
call site (`f"download:{ref}"`) and a grep for the literal would miss exactly
the counters that are keyed per subject.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import gen_worker
from gen_worker import byte_sources
from gen_worker.byte_sources import (
    BYTE_COUNTERS,
    BYTE_POSITIONS,
    Direction,
    Source,
    misclassified,
    source_of,
    verbs_in,
)

#: The factories that mint a counter. Matched on the ATTRIBUTE/function name,
#: because every one of them is reached under several import spellings
#: (`activity.scoped_counter`, `activity_mod.scoped_counter`, `act.counter`,
#: `progress_mod.counter`).
COUNTER_FACTORIES = {"scoped_counter", "counter"}

#: How the unit argument spells "bytes".
BYTE_UNITS = {"bytes", "UNIT_BYTES"}

SRC = Path(gen_worker.__file__).parent


def _source_files() -> list[Path]:
    """Every module the worker actually ships, minus vendored trees.

    `_vendor` is other repos' code, snapshot-pinned; a lint that fails on a
    vendor bump is a lint nobody keeps.
    """

    return [
        path
        for path in sorted(SRC.rglob("*.py"))
        if "_vendor" not in path.parts
    ]


def _module_string_constants(module: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = "literal"`` bindings.

    Declaring the name once as a constant is the GOOD pattern — it is what
    `load_progress.COUNTER_NAME` does — so the lint has to follow it rather
    than punish it.
    """

    constants: dict[str, str] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                constants[target.id] = node.value.value
    return constants


def _all_string_constants() -> dict[str, str]:
    """Every module-level string constant in `src/`, by name.

    Cross-module references (`load_progress.COUNTER_NAME`) resolve through
    this. A name bound to two different strings resolves to neither, which
    reports as unclassifiable rather than silently picking one.
    """

    seen: dict[str, set[str]] = {}
    for path in _source_files():
        module = ast.parse(path.read_text(), filename=str(path))
        for name, value in _module_string_constants(module).items():
            seen.setdefault(name, set()).add(value)
    return {name: next(iter(v)) for name, v in seen.items() if len(v) == 1}


def _literal_prefix(node: ast.expr, constants: dict[str, str]) -> str | None:
    """The literal part of a counter-name expression.

    A plain string is its own name; a module constant resolves through
    ``constants``. An f-string contributes the literal head before the first
    interpolation, which is what the registry's prefix keys (`"download:"`)
    are for. Anything with no literal head at all cannot be classified and is
    reported as such.
    """

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.Attribute):
        return constants.get(node.attr)
    if isinstance(node, ast.JoinedStr):
        head = node.values[0] if node.values else None
        if isinstance(head, ast.Constant) and isinstance(head.value, str):
            return head.value
        return ""
    return None


def _unit_is_bytes(call: ast.Call) -> bool:
    args = list(call.args[1:2])
    args += [kw.value for kw in call.keywords if kw.arg == "unit"]
    for arg in args:
        if isinstance(arg, ast.Constant) and arg.value in BYTE_UNITS:
            return True
        if isinstance(arg, ast.Attribute) and arg.attr in BYTE_UNITS:
            return True
        if isinstance(arg, ast.Name) and arg.id in BYTE_UNITS:
            return True
    return False


def _factory_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _byte_counter_sites() -> list[tuple[Path, int, str | None]]:
    """Every byte-counter creation in `src/`, as (file, line, literal prefix)."""

    shared = _all_string_constants()
    found: list[tuple[Path, int, str | None]] = []
    for path in _source_files():
        module = ast.parse(path.read_text(), filename=str(path))
        constants = {**shared, **_module_string_constants(module)}
        # The module that DEFINES the factories is not a call site.
        if path.name in ("progress.py", "activity.py") and path.parent == SRC:
            defined = {
                node.name for node in ast.walk(module)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if COUNTER_FACTORIES & defined and path.name == "progress.py":
                continue
        for node in ast.walk(module):
            if not isinstance(node, ast.Call):
                continue
            if _factory_name(node) not in COUNTER_FACTORIES:
                continue
            if not node.args or not _unit_is_bytes(node):
                continue
            found.append(
                (path, node.lineno, _literal_prefix(node.args[0], constants))
            )
    return found


def test_the_registry_never_puts_a_verb_on_the_wrong_instrument() -> None:
    """THE TABLE TEST. Registry-driven, so it grows with the registry."""

    for name, source in BYTE_COUNTERS.items():
        problem = misclassified(name, source)
        assert not problem, problem


def test_every_byte_counter_in_src_is_classified() -> None:
    """A counter with no declared source cannot be created.

    The failure message names the file and line, because the fix is one line in
    `byte_sources.BYTE_COUNTERS` and the person adding the counter is the only
    one who knows which instrument they read.
    """

    sites = _byte_counter_sites()
    assert sites, "the AST scan found no byte counters at all — it is broken"

    unclassified: list[str] = []
    for path, line, prefix in sites:
        where = f"{path.relative_to(SRC.parent.parent)}:{line}"
        if prefix is None:
            unclassified.append(f"{where}: counter name is not a literal")
            continue
        if not prefix:
            unclassified.append(f"{where}: counter name has no literal prefix")
            continue
        if source_of(prefix) is None:
            unclassified.append(f"{where}: {prefix!r} is in no source registry")
    assert not unclassified, (
        "byte counters with no declared measurement source:\n  "
        + "\n  ".join(unclassified)
        + "\nAdd each to gen_worker.byte_sources.BYTE_COUNTERS with the "
          "instrument it is read off."
    )


def test_the_registry_has_no_dead_entries() -> None:
    """A registry entry nothing creates is decoration, and decoration rots.

    This is the arm that would have caught the rename going half-done: the old
    key stays in the table, the new counter is unclassified, and BOTH halves
    fail rather than neither.
    """

    prefixes = {p for _f, _l, p in _byte_counter_sites() if p}
    for key in BYTE_COUNTERS:
        assert any(
            name == key or (key.endswith(":") and name.startswith(key))
            for name in prefixes
        ), f"{key!r} is registered but no site in src/ creates it"


def test_every_byte_position_declares_the_same_source_it_is_registered_with() -> None:
    """A position's site declaration and the registry cannot drift apart."""

    import importlib

    for qualname, source in BYTE_POSITIONS.items():
        module_name, _, class_name = qualname.rpartition(".")
        cls = getattr(importlib.import_module(module_name), class_name)
        declared = getattr(cls, "SOURCE", None)
        assert declared is source, (
            f"{qualname}.SOURCE is {declared!r}, registered as {source!r}"
        )
        assert not misclassified(class_name, source)


def test_the_seed_fix_landed_and_the_read_meter_reads() -> None:
    """pgw#1632's seed: `load:staged_bytes` was a READ meter with a WRITE verb.

    Both halves are asserted, because renaming without reclassifying just moves
    the lie: the name must be read-side AND the declared instrument must be the
    `/proc` read counter the sampler genuinely uses.
    """

    from gen_worker.models import load_progress

    assert load_progress.COUNTER_NAME == "load:ingested_bytes"
    assert load_progress.COUNTER_SOURCE is Source.PROC_READ_IO
    assert BYTE_COUNTERS[load_progress.COUNTER_NAME] is Source.PROC_READ_IO
    assert "staged" not in load_progress.COUNTER_NAME

    # The sampler really does read `/proc/self/io` read_bytes — the fact the
    # classification asserts. A rename with no producer behind it is worse
    # than the wrong verb.
    source = ast.parse(Path(load_progress.__file__).read_text())
    reads = [
        node for node in ast.walk(source)
        if isinstance(node, ast.Constant) and node.value == "read_bytes:"
    ]
    assert reads, "COUNTER_SOURCE claims PROC_READ_IO with no /proc read behind it"


def test_a_write_verb_on_a_read_source_is_rejected() -> None:
    """RED ARM. The exact pre-fix pair must fail the rule that now exists."""

    assert misclassified("load:staged_bytes", Source.PROC_READ_IO), (
        "the original defect must be rejected by the rule written to catch it"
    )
    assert misclassified("cas:written_bytes", Source.RSS_GROWTH)
    assert misclassified("volume:freed_bytes", Source.NET_SOCKET_RECV)
    # ...and the honest pairings pass.
    assert not misclassified("load:ingested_bytes", Source.PROC_READ_IO)
    assert not misclassified("upload:bytes", Source.NET_SOCKET_SEND)
    assert not misclassified("disk:freed_bytes", Source.STATVFS_DELTA)


@pytest.mark.parametrize("source", list(Source))
def test_every_source_has_a_direction(source: Source) -> None:
    assert isinstance(source.direction, Direction)


def test_no_verb_is_classified_two_ways() -> None:
    """The table is a function, not a suggestion."""

    assert len(byte_sources.VERB_DIRECTIONS) == len(set(byte_sources.VERB_DIRECTIONS))
    assert verbs_in("load:staged_bytes") == {"staged"}
    assert verbs_in("download:acme/model@prod") == set()
