"""pgw#1328: the ADOPT-ONLY serve role — the serve path behind dispatch.py's seam.

DESIGN-RULINGS §4.28/§4.29/§4.30, stated as a process boundary instead of a
code path: this role admits, adopts BY KEY, arms from the store (pgw#1329),
selects an ingress through tcg#37's published contract, serves — and on a miss
produces a typed refusal the hub can route on. It never serves eager on a miss
and it never mints, because it cannot: the mint lane is unimportable here
(:mod:`gen_worker.serve.guard` at runtime, ``scripts/lint_serve_role_closure.py``
in CI).

Read the modules in this order:

* :mod:`~gen_worker.serve.role` — the two roles, the two module sets, and why
  the role is declared rather than configured.
* :mod:`~gen_worker.serve.guard` — the runtime half: importing mint machinery
  raises.
* :mod:`~gen_worker.serve.refusal` — what this role says instead of eager+mint,
  and the ROUTE/REFUSE decision each miss kind carries.
* :mod:`~gen_worker.serve.boot_miss` — every ``boot_adopt`` reason's
  disposition, total and defaultless.
* :mod:`~gen_worker.serve.selection` — tcg#37's ``ingress_selection_v1``,
  adopted.
* :mod:`~gen_worker.serve.mint_seam` — mint supervision behind an interface,
  so the serving half's module scope names no mint module at all.

**This package's own module scope stays cheap and torch-free**: it is imported
by ``aot_serve``, and a role declaration that dragged in the world would be a
worse coupling than the one it removes. Submodules are imported by name, not
re-exported here.
"""

from __future__ import annotations

from .role import (
    MINT_MACHINERY, SERVE_ROLE_MODULES, ServeRole, adopt_only, current,
    declare)

__all__ = [
    "MINT_MACHINERY",
    "SERVE_ROLE_MODULES",
    "ServeRole",
    "adopt_only",
    "current",
    "declare",
]
