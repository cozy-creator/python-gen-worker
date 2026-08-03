"""A declaration module that REFUSES AT IMPORT — the pgw#853 stimulus.

Modelled verbatim on the shape ltx-video-2.3, qwen-image and z-image ship
today: a blocker table, a ``_refuse_if_blocked()`` that raises ``MintRefused``
with the blocker text, and that call at MODULE SCOPE, above the
``register_export_declaration`` it guards.

Importing this module raises. That is the point: an endpoint that imports it
the naive way dies at boot, and a compile feature must never be able to do
that to serving.
"""

from __future__ import annotations

from gen_worker.aot_mint import MintRefused

from .blocked_declaration_parts import BLOCKER_TEXT, FAMILY, build_declaration


def _refuse_if_blocked() -> None:
    raise MintRefused(BLOCKER_TEXT)


DECLARATION = build_declaration()

_refuse_if_blocked()

# Never reached — exactly like z-image's own last line.
raise AssertionError(f"unreachable ({FAMILY})")
