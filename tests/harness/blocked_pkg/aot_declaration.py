"""A PACKAGE-resident declaration that refuses — the shape ltx/qwen/z-image
now ship. The endpoint walker imports every submodule of an endpoint package,
so this module is imported whether or not ``main`` asks for it.
"""

from __future__ import annotations

from gen_worker import Compile, register_export_declaration
from gen_worker.aot_mint import MintRefused

from ..blocked_declaration_parts import BLOCKER_TEXT, FAMILY, build_declaration

_FAMILY = FAMILY


def _refuse_if_blocked() -> None:
    raise MintRefused(BLOCKER_TEXT)


def _declaration() -> Compile:
    """Mirrors the endpoints' own shape exactly: build, then blocker-check."""
    decl = build_declaration()
    _refuse_if_blocked()
    return decl


register_export_declaration(_declaration, family=_FAMILY)
