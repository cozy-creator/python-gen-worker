"""An endpoint that carries a REFUSING export declaration and serves anyway.

pgw#853's invariant, as an endpoint module. Two mechanisms, both exercised:

- ``Compile(blockers=...)`` — the family declares WHY it may not mint yet, as
  data the mint gate reads (pgw#1115; it replaced pgw#853's thunk, which
  pgw#1107 retired because ``@endpoint(compile=)`` cannot carry a callable);
- ``import_export_declaration(...)`` — the backstop, for module-scope work in
  a declaration file that raises anyway (``harness.blocked_declaration`` does
  exactly that).

If either one leaked, importing this module would raise and the worker would
never collect an endpoint — no serving, for a compile feature.
"""

from __future__ import annotations

import msgspec

from gen_worker import (
    RequestContext, endpoint, import_export_declaration,
    register_export_declaration,
)
from gen_worker.families.base import GenerationDefaults, family as family_vocab

from .blocked_declaration_parts import BLOCKER, build_declaration

#: (2) the belt-and-braces backstop: this import RAISES, and must not escape.
DECLARATION_IMPORTED = import_export_declaration(
    "harness.blocked_declaration")

#: (1) the primary fix: a declaration whose blockers are VALUES.
BLOCKED_FAMILY = "harness-blocked-declared"
BLOCKED_DECLARATION = build_declaration(
    family=BLOCKED_FAMILY, blockers=(BLOCKER,))

register_export_declaration(BLOCKED_DECLARATION)

@family_vocab("harness-blocked-family")
class _BlockedDefaults(GenerationDefaults, frozen=True):
    steps: int = 4


class BlockedIn(msgspec.Struct):
    text: str = ""


class BlockedOut(msgspec.Struct):
    response: str


@endpoint
class BlockedDeclarationEndpoint:
    def echo(self, ctx: RequestContext, data: BlockedIn) -> BlockedOut:
        return BlockedOut(response=f"served:{data.text}")
