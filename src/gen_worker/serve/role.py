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
:data:`MODEL_BEARING_SERVE_MODULES`, :data:`FORBIDDEN_LIBRARIES` and
:data:`OPTIONAL_SERVE_IMPORTS` are read by THREE consumers: the runtime blocker
(:mod:`gen_worker.serve.guard`), the CI fence
(``scripts/lint_serve_role_closure.py``) and the tests. pgw#1176's measured
lesson is that a fence naming symbols in its own string literals rots silently,
and pgw#824's is that two lists of the same literals drift. So the role's own
module declares them and everything else — including the fence, which parses
this file — reads them from here. There is one list.

Two claims, two scopes, because they are not the same claim (pgw#1331). Every
serve-role module is asserted MINT-FREE. Each is ALSO either asserted to reach
no MODEL LIBRARY (``MODEL_FREE_MODULES``, 67 roots) or named as still reaching
one (``MODEL_BEARING_SERVE_MODULES``, 9) — and the second list is checked to be
TRUE, so a module that gets cut free must be moved rather than left sitting in
an exception list nobody re-reads. The role is the union, by splice.

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


#: Every serve-role module whose whole static closure reaches NO model library
#: (pgw#1331). 67 of the role's 76 roots; the other nine are named, with their
#: two causes, in :data:`MODEL_BEARING_SERVE_MODULES` directly below.
#:
#: This used to be 14 — the typed family surface alone — because two PACKAGE
#: ROOTS dragged the eager-capable worker's guts into every closure that named
#: them. ``gen_worker/__init__`` eagerly re-exported ``view`` and
#: ``models.provision``; ``gen_worker/models/__init__`` eagerly re-exported
#: ``residency`` and through it ``loading``. Both are PEP 562 lazy now, so
#: importing a package costs nothing and asking for a name costs exactly the
#: submodule that defines it. ``serving_mode`` separately stopped importing
#: ``compile_cache`` — 3,100 lines of arming brain — to read two facts, which
#: now live in :mod:`gen_worker.compile_facts`.
MODEL_FREE_MODULES: Tuple[str, ...] = (
    # The role itself.
    "gen_worker.serve",
    "gen_worker.serve.guard",
    "gen_worker.serve.mint_seam",
    "gen_worker.serve.refusal",
    "gen_worker.serve.role",
    "gen_worker.serve.selection",
    # §4.27 steps 1-3, the parts that state and materialize a key set.
    "gen_worker.local_cell_store",
    "gen_worker.keyset.document",
    "gen_worker.keyset.identifiers",
    # The arm's constants (pgw#1329) and the adopt outcome vocabulary.
    "gen_worker.aot_constants",
    "gen_worker.cell_adopt",
    # The compile facts a serving process READS — the arm marker and the
    # toolchain probe — with none of the machinery that writes them.
    "gen_worker.compile_facts",
    # The neutral dispatch order the wire head projects into, and the wire
    # facts a refusal is reported as.
    "gen_worker.activity",
    "gen_worker.dispatch",
    "gen_worker.process_role",
    "gen_worker.serve_posture",
    # pgw#1331: the typed family surface a request is actually SERVED through —
    # the bindings, the backings behind them, the bare-math schedulers, and the
    # catalog's serving halves. Roots, not incidental members: the claim this
    # issue makes is that a Flux request runs end to end from here.
    "gen_worker.model",
    "gen_worker.model.backing",
    "gen_worker.model.errors",
    "gen_worker.model.runtime",
    "gen_worker.model.scheduler",
    # pgw#1346 B3a: the two flow-match ladder parameters the scheduler above
    # does not read yet (Qwen-Image's `shift_terminal`), as bare typed math over
    # the same declared block.
    "gen_worker.model.flow_ladders",
    # HiDream-O1's flash ladder: its own module because it is NOT a member of
    # scheduler.py's closed declarable set — nothing can declare it while
    # GraphModelSpec.scheduler is single-valued (pgw#1346 K10).
    "gen_worker.model.scheduler_hidream",
    "gen_worker.model.snapshot",
    "gen_worker.model.spec",
    "gen_worker.model.tuned",
    "gen_worker.model.catalog",
    "gen_worker.model.catalog._generated",
    "gen_worker.model.catalog._generated.ernie",
    "gen_worker.model.catalog._generated.flux1_dev",
    "gen_worker.model.catalog._generated.flux1_schnell",
    "gen_worker.model.catalog._generated.flux2_klein_4b",
    "gen_worker.model.catalog._generated.flux2_klein_9b",
    "gen_worker.model.catalog._generated.qwen_image",
    "gen_worker.model.catalog._generated.sd2",
    "gen_worker.model.catalog._generated.sd15",
    "gen_worker.model.catalog._generated.sdxl",
    "gen_worker.model.catalog._generated.z_image",
    # K9's packed shape axis, shared by the families whose graph classes are a
    # SET of (width, height) pairs rather than a product of two axes.
    "gen_worker.model.catalog._packed_shape",
    "gen_worker.model.catalog._generated.ltx23",
    "gen_worker.model.catalog._generated.minimax_h3",
    "gen_worker.model.catalog._generated.wan22_i2v_a14b",
    "gen_worker.model.catalog._generated.wan22_t2v_a14b",
    "gen_worker.model.catalog._generated.wan22_ti2v_5b",
    # pgw#1346 B3b's two EAGER models. Both halves are model-free here, not
    # only the serving one: neither declaration builds a module, because
    # neither model's code (DiffSynth-Studio's) is reachable from this SDK at
    # all — which is one of the reasons they are eager. So they take no
    # guarded-import hatch and are held by the real walk.
    "gen_worker.model.catalog.anima",
    "gen_worker.model.catalog.anima_serve",
    "gen_worker.model.catalog.ernie_serve",
    "gen_worker.model.catalog.flux1_dev_serve",
    "gen_worker.model.catalog.flux1_schnell_serve",
    "gen_worker.model.catalog.flux2_klein_4b_serve",
    "gen_worker.model.catalog.flux2_klein_9b_serve",
    "gen_worker.model.catalog.hidream_o1",
    "gen_worker.model.catalog.hidream_o1_serve",
    "gen_worker.model.catalog.qwen_image_serve",
    "gen_worker.model.catalog.ltx23_serve",
    # pgw#1346 B4's EAGER model, held by the real walk for the same reason as
    # B3b's two: MiniMax-H3's architecture is vendored in the endpoint at a
    # pinned diffusers SHA and is not reachable from this SDK, so the
    # declaration builds nothing and needs no guarded-import hatch.
    "gen_worker.model.catalog.minimax_h3",
    "gen_worker.model.catalog.sd15_serve",
    "gen_worker.model.catalog.sdxl_serve",
    "gen_worker.model.catalog.z_image_serve",
    # pgw#1346 B5: the EAGER boundary models, DECLARATIONS INCLUDED. A graph
    # declaration is model-bearing and reaches its binding only through the
    # guarded hatch above; an eager one has no `build` callable, so it is
    # ordinary model-free Python and its binding imports it directly. Naming
    # both halves here is what turns "an eager declaration is import-cheap"
    # from a convention into something the fence checks — and the day one of
    # these grows a diffusers import, this list goes red instead of the claim
    # quietly becoming false.
    "gen_worker.model.catalog.boundary_3d",
    "gen_worker.model.catalog.boundary_audio",
    "gen_worker.model.catalog.boundary_llm",
    "gen_worker.model.catalog.flex2_preview",
    "gen_worker.model.catalog._generated.chatterbox",
    "gen_worker.model.catalog._generated.flex2_preview",
    "gen_worker.model.catalog._generated.foundation_1",
    "gen_worker.model.catalog._generated.hunyuan3d",
    "gen_worker.model.catalog._generated.internvl_u",
    "gen_worker.model.catalog._generated.joycaption",
    "gen_worker.model.catalog._generated.musicgen",
    "gen_worker.model.catalog._generated.qwen36_27b_mtp",
    "gen_worker.model.catalog._generated.qwen36_35b_a3b",
    "gen_worker.model.catalog._generated.stable_audio_open",
    "gen_worker.model.catalog._generated.trellis_3d",
    "gen_worker.model.catalog.wan22_serve",
)


#: The serve-role modules that still reach a model library, and this is a
#: LEDGER, not an allowlist: ``lint_serve_role_closure.py`` asserts every row
#: really does reach one TODAY, so a row that gets fixed goes red until it is
#: deleted. The list can only shrink. An owed cut written as prose rots
#: (pgw#1176); written here it is checked on every PR.
#:
#: **Nine rows, TWO causes, and neither is a design question — both are moves
#: too big to smuggle into the lane that measured them.**
#:
#: 1. **``compile_cache``, for seven of the nine.** It is the eager-capable
#:    arming brain: it imports ``models.loading`` / ``models.memory`` /
#:    ``models.w8a8_lora``, which construct diffusers and transformers objects
#:    inside their functions and are entitled to, because an eager-capable pod
#:    serves eager on a miss (§4.28). It is ALSO where the compile-cache
#:    IDENTITY functions live — ``toolchain_digest``, ``content_keys``,
#:    ``static_code_closure``, ``parse_cell_ref`` — and the key set folds a
#:    cell key from those, so ``keyset.fold`` and ``keyset.closure`` import the
#:    arming brain to do arithmetic and everything rooted at them inherits it
#:    (``keyset``, ``keyset.boot``, ``keyset.store``, ``boot_adopt``;
#:    ``cell_resolve`` reaches it through ``aot_delivery``). **The cut:** the
#:    identity half of ``compile_cache`` moves out, the way its READ half
#:    already did — :mod:`gen_worker.compile_facts` holds ``runtime_key`` and
#:    the marker readers and imports nothing above ``hostfacts``.
#: 2. **``aot_serve``'s three model-layer edges, for it and ``serving_mode``.**
#:    ``models.memory`` (``is_cuda_oom``, a pure exception predicate that needs
#:    no model library), ``compile_cache`` (four names, per cause 1) and
#:    ``models.lora_lifted`` — which is the only one of the three that is a
#:    real question rather than a move, because LoRA lifting genuinely belongs
#:    to the eager model layer and the adopt-only arm's use of it has to be
#:    read before it is relocated.
MODEL_BEARING_SERVE_MODULES: Tuple[str, ...] = (
    "gen_worker.boot_adopt",
    "gen_worker.cell_resolve",
    "gen_worker.keyset",
    "gen_worker.keyset.boot",
    "gen_worker.keyset.closure",
    "gen_worker.keyset.fold",
    "gen_worker.keyset.store",
    "gen_worker.aot_serve",
    "gen_worker.serving_mode",
)


#: Every module the ADOPT-ONLY serve path executes. The fence walks the
#: transitive ``gen_worker`` import closure of these — function-local imports
#: included, because a lazy ``from . import aot_mint`` inside a function is
#: exactly the shape a re-coupling would take — and refuses any reach into
#: :data:`MINT_MACHINERY`.
#:
#: Spliced from the two tuples above rather than retyped, so the model-free
#: claim and the role are TOTAL by construction: every serve-role module is
#: either asserted model-free or named as bearing one, and a module cannot fall
#: out of both by being edited into a third list (pgw#824's drift rule).
SERVE_ROLE_MODULES: Tuple[str, ...] = (
    *MODEL_FREE_MODULES,
    *MODEL_BEARING_SERVE_MODULES,
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
    # pgw#1346 B3a. ONE declaration module for TWO published checkpoints in
    # each case: ERNIE-Image and ERNIE-Image-Turbo differ only in weights and
    # recipe, and Z-Image's base and official DMD Turbo differ only in weights
    # and a scheduler shift.
    "gen_worker.model.catalog.ernie",
    "gen_worker.model.catalog.flux1_dev",
    # schnell DERIVES its architecture from dev's and its binding guard-imports
    # its own declaration module (pgw#1346 B1).
    "gen_worker.model.catalog.flux1_schnell",
    # ONE declaration for TWO checkpoints: Base and Turbo are instances of this
    # model, not two models, so the Turbo binding guard-imports the same name
    # (pgw#1346 B1 — byte-identical transformer configs, differing only in the
    # published recipe).
    "gen_worker.model.catalog.flux2_klein_4b",
    # Same shape at the second width: Base and Turbo are two instances of
    # ONE 9B model (their transformer configs differ only in
    # `_name_or_path`), so the 9B binding guard-imports one declaration too
    # (pgw#1346 B3b).
    "gen_worker.model.catalog.flux2_klein_9b",
    "gen_worker.model.catalog.qwen_image",
    "gen_worker.model.catalog.ltx23",
    # ONE declaration module for TWO bindings: `sd15.py` declares both `SD15`
    # and `SD2` (pgw#1346 B2 — same runner set, different graphs), so both
    # generated modules guard-import the same name.
    "gen_worker.model.catalog.sd15",
    "gen_worker.model.catalog.sdxl",
    "gen_worker.model.catalog.z_image",
    # ONE declaration module for THREE bindings: `wan22.py` declares
    # WAN22_T2V_A14B, WAN22_I2V_A14B and WAN22_TI2V_5B (pgw#1346 B4 — three
    # architectures, one endpoint repo), so all three generated modules
    # guard-import the same name.
    "gen_worker.model.catalog.wan22",
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
    "MODEL_BEARING_SERVE_MODULES",
    "MODEL_FREE_MODULES",
    "OPTIONAL_SERVE_IMPORTS",
    "SERVE_ROLE_MODULES",
    "ServeRole",
    "adopt_only",
    "current",
    "declare",
]
