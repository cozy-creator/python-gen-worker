"""pgw#1328: WHICH serve role this process is, and the module sets that decide it.

DESIGN-RULINGS §4.28/§4.30 leave exactly two kinds of serving pod:

* **eager-capable** — the ordinary Python worker. It serves eager on a miss and
  mints as a SIDE EFFECT of serving (§4.28 verbatim: *"pre-warming a
  release/SKU = boot an ordinary serving pod there"*). This is the role every
  pod has had until now, and it is the default.
* **adopt-only** — a pod that PULLS BY KEY (§4.29), arms from the store
  (pgw#1329) and, on a miss, produces a typed refusal the hub can route on. It
  cannot compile, so it must not be able to *reach* the code that compiles.

The role is not configuration. §1.17 and Paul's standing rule (*an env may
carry a VALUE, never a DECISION*) rule out an env knob, and pgw#1327 already
refused one for exactly this question: *"an adopt-only role states its posture
by passing no deriver"*. It is DECLARED by the process entry point that knows
which process it is — the same shape as :mod:`gen_worker.process_role`, whose
wire role this one refines — and it is declared ONCE, before anything imports.

THE MODULE SETS, AND WHY THEY LIVE HERE
---------------------------------------
:data:`SERVE_ROLE_MODULES`, :data:`MINT_MACHINERY`, :data:`MODEL_FREE_MODULES`,
:data:`FORBIDDEN_LIBRARIES` and :data:`OPTIONAL_SERVE_IMPORTS` are read by THREE
consumers: the runtime blocker (:mod:`gen_worker.serve.guard`), the CI fence
(``scripts/lint_serve_role_closure.py``) and the tests. pgw#1176's measured
lesson is that a fence naming symbols in its own string literals rots silently,
and pgw#824's is that two lists of the same literals drift. So the role's own
module declares them and everything else — including the fence, which parses
this file — reads them from here. There is one list.

Two claims, two scopes, because they are not the same claim (pgw#1331). Every
serve-role module is asserted MINT-FREE. The family surface named in
``MODEL_FREE_MODULES`` is additionally asserted to reach no MODEL LIBRARY —
that subset, and not the whole role, for the reason recorded on the tuple.

``SERVE_ROLE_MODULES`` is a CLAIM, not a description: every name in it is a
module the adopt-only path executes, asserted to be unable to reach anything in
``MINT_MACHINERY``. Adding a name is how the guard grows; it is never how a
violation is silenced.

WHAT IS DELIBERATELY *NOT* IN THE SERVE-ROLE SET
------------------------------------------------
``gen_worker.fleet_cells`` and ``gen_worker.executor``. Both are the
EAGER-CAPABLE arming brain and the process host respectively, and both reach
mint machinery on purpose. The adopt-only path arms through
``aot_serve.arm_compiled_graph_from_store`` (pgw#1329) — the arm with no
``nn.Module`` and no diffusers on the path — which is precisely why that issue
was the prerequisite for this one. A future lane that carves the serving half
out of ``executor`` adds it here; until then the RUNTIME blocker covers the
process and the STATIC fence covers the role.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Tuple

logger = logging.getLogger(__name__)


class ServeRole(StrEnum):
    """The two roles a serving process can hold. Closed, on purpose."""

    #: Serves eager on a miss and mints as a side effect (§4.28). The default:
    #: a process that never declared is the role every pod has always had, so
    #: an undeclared interpreter can never silently become adopt-only.
    EAGER_CAPABLE = "eager_capable"

    #: Adopts by key, arms from the store, and REFUSES or ROUTES on a miss.
    #: Never eager-on-miss, never a mint.
    ADOPT_ONLY = "adopt_only"


#: The subset of :data:`SERVE_ROLE_MODULES` that must reach NO model library —
#: the typed family surface a request is actually served through (pgw#1331).
#:
#: **Why this is a subset and not the whole role, stated rather than hidden.**
#: Every ``gen_worker`` import executes ``gen_worker/__init__.py``, and that
#: package reaches ``models.loading`` / ``models.memory`` / ``view`` — the
#: EAGER-CAPABLE worker's own guts, which import diffusers inside functions and
#: legitimately need to, because an eager-capable pod serves eager on a miss
#: (§4.28). Those imports never EXECUTE on an adopt-only pod, which is what
#: :mod:`gen_worker.serve.guard` and pgw#1331's subprocess proof assert at run
#: time. What a static walk can prove today is the property for the surface
#: pgw#1331 built, and claiming more than that would be a fence describing a
#: tree nobody has cut. **Owed, and named here so it is not rediscovered:**
#: making the whole role statically model-free means making
#: ``gen_worker/__init__`` lazy the way this package's own ``__init__`` now is.
MODEL_FREE_MODULES: Tuple[str, ...] = (
    "gen_worker.model",
    "gen_worker.model.backing",
    "gen_worker.model.errors",
    "gen_worker.model.runtime",
    "gen_worker.model.scheduler",
    "gen_worker.model.snapshot",
    "gen_worker.model.spec",
    "gen_worker.model.tuned",
    "gen_worker.model.catalog",
    "gen_worker.model.catalog._generated",
    "gen_worker.model.catalog._generated.flux1_dev",
    "gen_worker.model.catalog._generated.sd2",
    "gen_worker.model.catalog._generated.sd15",
    "gen_worker.model.catalog._generated.sdxl",
    "gen_worker.model.catalog.flux1_dev_serve",
    "gen_worker.model.catalog.sd15_serve",
    "gen_worker.model.catalog.sdxl_serve",
)


#: Every module the ADOPT-ONLY serve path executes. The fence walks the
#: transitive ``gen_worker`` import closure of these — function-local imports
#: included, because a lazy ``from . import aot_mint`` inside a function is
#: exactly the shape a re-coupling would take — and refuses any reach into
#: :data:`MINT_MACHINERY`.
SERVE_ROLE_MODULES: Tuple[str, ...] = (
    # The role itself.
    "gen_worker.serve",
    "gen_worker.serve.guard",
    "gen_worker.serve.mint_seam",
    "gen_worker.serve.refusal",
    "gen_worker.serve.role",
    "gen_worker.serve.selection",
    # §4.27 steps 1-3: state the key set (as DATA since pgw#1327), ask this
    # machine, ask the hub, materialize.
    "gen_worker.boot_adopt",
    "gen_worker.cell_resolve",
    "gen_worker.local_cell_store",
    "gen_worker.keyset",
    "gen_worker.keyset.boot",
    "gen_worker.keyset.closure",
    "gen_worker.keyset.document",
    "gen_worker.keyset.fold",
    "gen_worker.keyset.identifiers",
    "gen_worker.keyset.store",
    # The arm (pgw#1329) and the dispatch it produces.
    "gen_worker.aot_constants",
    "gen_worker.aot_serve",
    "gen_worker.cell_adopt",
    # The neutral dispatch order the wire head projects into, and the wire
    # facts a refusal is reported as.
    "gen_worker.activity",
    "gen_worker.dispatch",
    "gen_worker.process_role",
    "gen_worker.serve_posture",
    "gen_worker.serving_mode",
    # pgw#1331: the typed family surface a request is actually SERVED through —
    # the bindings, the backings behind them, the bare-math schedulers, and the
    # catalog's serving halves. Roots, not incidental members: the claim this
    # issue makes is that a Flux request runs end to end from here, so it is
    # also asserted mint-free, and a claim that is not a root is a claim
    # nothing walks. They are ALSO the whole of MODEL_FREE_MODULES above,
    # spliced rather than retyped so the two sets cannot drift (pgw#824).
    *MODEL_FREE_MODULES,
)


#: Third-party packages :data:`MODEL_FREE_MODULES` may not import, at module
#: scope or inside a function (pgw#1331).
#:
#: ``diffusers`` is the one the issue names, and the reason is not tidiness: it
#: is 5-15 seconds of process start and 1-2 GB of host RAM (pgw#1326's measured
#: table) bought to run a handful of reshapes and one Euler step, and it is what
#: makes today's serve image un-shrinkable. ``transformers`` is here for the
#: same reason and by the same argument — the text encoders are its models, and
#: pgw#1331 makes them graph classes precisely so the serve path stops needing
#: the library that defines them.
#:
#: **This is not a claim that the modules cannot be installed.** The mint lane
#: needs both and has both; the fence is about what the SERVE closure REACHES.
#: The runtime half is :mod:`gen_worker.serve.guard`, which blocks these names
#: in an adopt-only process the same way it blocks the mint lane.
FORBIDDEN_LIBRARIES: Tuple[str, ...] = (
    "diffusers",
    "transformers",
)

#: The gen_worker modules a serve-role module may reach ONLY through an
#: ``ImportError``-guarded import, and the closed list of them.
#:
#: A generated family binding exposes ``SPEC`` by importing its own declaration
#: inside ``try: ... except ImportError: return None``. The declaration builds
#: diffusers modules, so on an adopt-only pod that import fails and the binding
#: serves without it — pgw#1339's ruling exactly: the absence of a serving fact
#: degrades loudly and serves, it does not refuse.
#:
#: The fence therefore does not FOLLOW a guarded edge, which would make every
#: binding drag its declaration into the closure and the whole guard vacuous.
#: What it does instead is require every guarded edge to name a module in this
#: tuple, and require every module in this tuple to actually be reached — so
#: the hatch is an enumerated list two people can read, never an open door that
#: any ``try: import`` can walk through.
OPTIONAL_SERVE_IMPORTS: Tuple[str, ...] = (
    "gen_worker.model.catalog.flux1_dev",
    # ONE declaration module for TWO bindings: `sd15.py` declares both `SD15`
    # and `SD2` (pgw#1346 B2 — same runner set, different graphs), so both
    # generated modules guard-import the same name.
    "gen_worker.model.catalog.sd15",
    "gen_worker.model.catalog.sdxl",
)

#: The mint lane. A serve-role module that reaches ANY of these is a pod that
#: can be made to compile — which is the one thing this role is defined by not
#: being able to do. Named exactly as pgw#1328 filed them, plus the three
#: siblings that reach the same entry points by another door
#: (``mint_child``/``mint_process`` spawn them, ``keyset.emit`` consumes a
#: tracer's output and is pgw#1327's own declared mint-side root).
MINT_MACHINERY: Tuple[str, ...] = (
    "gen_worker.aot_compile_child",
    "gen_worker.aot_compile_pool",
    "gen_worker.aot_mint",
    "gen_worker.boot_key",
    "gen_worker.boot_trace_child",
    # pgw#1331: a family declaration mints its OWN graph classes through this
    # bridge. It is the family surface's mint half, and the family surface is
    # on the serve path — which is exactly why it is named here: a serve-role
    # module that could reach it would be a pod that can compile.
    "gen_worker.model.mint",
    "gen_worker.keyset.emit",
    "gen_worker.mint_child",
    "gen_worker.mint_process",
    "gen_worker.mint_supervisor",
)

_role: ServeRole = ServeRole.EAGER_CAPABLE


def declare(role: ServeRole) -> None:
    """Declare this interpreter's serve role. Idempotent, once per process.

    Re-declaring the SAME role is a no-op (a re-connect re-declares). Changing
    it is refused: a process that started adopt-only installed an import
    blocker, and a process that started eager-capable already imported the
    machinery the blocker exists to keep out — neither is undone by a flag.
    """
    global _role
    if _role is not role and _role is not ServeRole.EAGER_CAPABLE:
        raise RuntimeError(
            f"serve role already declared as {_role.value}; it cannot become "
            f"{role.value} — the role decides what this process IMPORTED")
    _role = role
    logger.info("serve role declared: %s", role.value)


def current() -> ServeRole:
    return _role


def adopt_only() -> bool:
    """True when a miss must REFUSE or ROUTE rather than serve eager + mint."""
    return _role is ServeRole.ADOPT_ONLY


def _reset_for_test(role: ServeRole = ServeRole.EAGER_CAPABLE) -> None:
    """Test-only: put the module-level role back. Never called by the worker."""
    global _role
    _role = role


__all__ = [
    "FORBIDDEN_LIBRARIES",
    "MINT_MACHINERY",
    "OPTIONAL_SERVE_IMPORTS",
    "SERVE_ROLE_MODULES",
    "ServeRole",
    "adopt_only",
    "current",
    "declare",
]
