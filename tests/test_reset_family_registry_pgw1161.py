"""pgw#1161 — a RE-IMPORT of a family declaration is not a collision.

Cause A of ie#690: 66 of the 136 failures the endpoint declaration suites
surfaced once they stopped silently skipping. SDK-owned.

`@family` registration is an IMPORT SIDE EFFECT, and the duplicate guard
compared by IDENTITY (`existing is not cls`). Re-importing a declaration module
builds a NEW class object for the same declaration, so the second registration
raised `family 'X' kind 'checkpoint' already registered by ...` — the endpoint
suites re-import their declaration modules between tests and hit it on their
own declarations.

THE FIX THAT LOOKED OBVIOUS AND IS WRONG. The first attempt made
`reset_export_declarations()` also clear `families.base._REGISTRY`. CI proved
it wrong within one run: registration being an import side effect means a
cleared registry can only be refilled by a re-import, and a module already in
`sys.modules` re-imports as a NO-OP. So clearing permanently wipes every
module-level registration for all tests that run after any reset —
`test_family_wire_names_pgw692` imports `_example_family` exactly once, and
`family_for("example")` returned None thereafter. Three tests died that way.
That is pgw#1031's lesson one registry along, and it applies to the cure as
much as to the disease.

So the collision guard is fixed at its own end instead: a re-import (same
module, same qualname) REPLACES; anything else still refuses. Nothing needs
clearing for a re-import to succeed, and nothing that was registered once and
cannot re-register ever gets wiped.
"""

from __future__ import annotations

import pytest

from gen_worker.api.export_contract import reset_export_declarations
from gen_worker.families import base as families_base
from gen_worker.families.base import (
    GenerationDefaults, family, family_for, family_registry)

_DECL_SRC = '''
from gen_worker.families.base import GenerationDefaults, family


@family("pgw1161-probe")
class ProbeDefaults(GenerationDefaults):
    pass
'''


def _import_declaration_module() -> type:
    """Re-execute a declaration module, as a re-import would."""
    ns: dict = {}
    exec(compile(_DECL_SRC, "<pgw1161_decl>", "exec"), ns)
    return ns["ProbeDefaults"]


@pytest.fixture(autouse=True)
def _clean_registry():
    """Restore, never CLEAR.

    A blanket clear is the very hazard this module documents: it would wipe
    `test_family_wire_names_pgw692`'s module-level `_example_family`
    registration, which no re-import can restore. Snapshot and put back exactly
    what was there — the fixture must not be able to do the damage the code
    under test is being fixed to avoid.
    """
    before = dict(families_base._REGISTRY)
    yield
    families_base._REGISTRY.clear()
    families_base._REGISTRY.update(before)


def test_a_RE_IMPORT_of_the_same_declaration_replaces(
) -> None:
    """RED on master: raises `already registered`. This is the 66-test cause —
    a new class object for the same declaration read as a second family."""
    first = _import_declaration_module()
    assert family_for("pgw1161-probe") is first

    second = _import_declaration_module()
    assert second is not first, "the re-import must build a NEW class object"
    assert family_for("pgw1161-probe") is second, "the later import wins"


def test_a_GENUINE_collision_is_still_refused() -> None:
    """Two different declarations claiming one family is unadjudicable and
    stays a hard error. A fix that silenced this would trade a false positive
    for a family served by whichever module imported last."""
    _import_declaration_module()

    class Other(GenerationDefaults):
        pass

    with pytest.raises(ValueError, match="already registered"):
        family("pgw1161-probe")(Other)


def test_a_different_class_in_the_SAME_module_is_still_a_collision() -> None:
    """The narrow check is (module, qualname), not module alone — two classes
    in one file declaring one family is an authoring error, not a re-import."""
    first = _import_declaration_module()

    class Sibling(GenerationDefaults):
        pass

    Sibling.__module__ = first.__module__  # same file, different class

    with pytest.raises(ValueError, match="already registered"):
        family("pgw1161-probe")(Sibling)


def test_re_decorating_the_SAME_class_is_still_idempotent() -> None:
    """`existing is not cls` still short-circuits, untouched."""
    class P(GenerationDefaults):
        pass

    assert family("pgw1161-probe")(P) is family("pgw1161-probe")(P) is P


# ---------------------------------------------------------------------------
# The cure that was worse than the disease, pinned so it is not re-attempted
# ---------------------------------------------------------------------------


def test_the_RESET_does_not_wipe_the_family_registry() -> None:
    """Pinned deliberately, against the obvious "fix".

    Clearing `_REGISTRY` from `reset_export_declarations` looks right and
    breaks three tests, because a module-level registration cannot come back:
    the re-import is a `sys.modules` no-op. `test_family_wire_names_pgw692`
    imports `_example_family` once, at module scope, and every test after any
    reset then saw `family_for("example") is None`.
    """
    _import_declaration_module()
    assert "pgw1161-probe" in family_registry()

    reset_export_declarations()

    assert "pgw1161-probe" in family_registry(), (
        "the reset wiped a registration that only an import could restore — "
        "see this module's docstring before 'fixing' it again")


def test_the_fix_FIRES_on_the_sequence_master_could_not_do() -> None:
    """Severance experiment. If the re-import guard is ever narrowed back to a
    bare identity check, this raises and 66 endpoint tests go red with it."""
    _import_declaration_module()
    try:
        _import_declaration_module()
    except ValueError as exc:  # pragma: no cover - the severed path
        pytest.fail(f"a re-import is being read as a collision again: {exc}")
