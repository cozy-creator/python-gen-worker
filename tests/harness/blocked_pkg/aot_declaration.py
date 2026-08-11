"""A PACKAGE-resident declaration that refuses to MINT.

The endpoint walker imports every submodule of an endpoint package, so this
module is imported whether or not ``main`` asks for it — which is why the
refusal must be data (``Compile.blockers``, pgw#1115) and not an exception.
"""

from __future__ import annotations

from gen_worker import register_export_declaration

from ..blocked_declaration_parts import BLOCKER, FAMILY, build_declaration

_FAMILY = FAMILY

DECLARATION = build_declaration(blockers=(BLOCKER,))

register_export_declaration(DECLARATION)
