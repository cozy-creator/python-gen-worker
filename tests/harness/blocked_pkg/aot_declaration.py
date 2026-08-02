"""A PACKAGE-resident declaration that refuses — the shape ltx/qwen/z-image
now ship. The endpoint walker imports every submodule of an endpoint package,
so this module is imported whether or not ``main`` asks for it.
"""

from __future__ import annotations

from gen_worker import Compile, register_export_declaration
from gen_worker.aot_mint import MintRefused

from ..blocked_declaration_parts import BLOCKER_TEXT, FAMILY, build_declaration

_FAMILY = FAMILY


def _declaration() -> Compile:
    decl = build_declaration()
    raise MintRefused(BLOCKER_TEXT)
    return decl  # noqa: unreachable — mirrors the endpoints' own shape


register_export_declaration(_declaration, family=_FAMILY)
