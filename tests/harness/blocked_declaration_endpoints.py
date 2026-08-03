"""An endpoint that carries a REFUSING export declaration and serves anyway.

pgw#853's invariant, as an endpoint module: this is what the five unwired
families must be able to look like. Two mechanisms, both exercised here:

- ``register_export_declaration(thunk, family=...)`` — the blockers evaluate
  when the MINT asks, not when python imports;
- ``import_export_declaration(...)`` — the backstop, for module-scope work
  that raises anyway (``harness.blocked_declaration`` does exactly that).

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

#: (3) the belt-and-braces backstop: this import RAISES, and must not escape.
DECLARATION_IMPORTED = import_export_declaration(
    "harness.blocked_declaration")

#: (1) the primary fix: a thunk whose blockers evaluate at mint time.
THUNK_FAMILY = "harness-thunk-family"


def _blocked_thunk():
    from gen_worker.aot_mint import MintRefused

    from .blocked_declaration_parts import BLOCKER_TEXT

    raise MintRefused(BLOCKER_TEXT)


register_export_declaration(_blocked_thunk, family=THUNK_FAMILY)


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
