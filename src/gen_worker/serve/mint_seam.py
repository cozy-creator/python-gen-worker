"""pgw#1328: mint supervision behind an interface the serve role cannot import.

Three modules used to name the mint lane directly:

* ``executor`` — ``from . import mint_supervisor`` at MODULE scope (:186 on
  ``7d33f3d0``), for exactly one region: ``_supervise_mint``'s
  ``supervise(MintTask(...))`` and its ``ABANDONED`` comparison.
* ``fleet_compiled_graphs`` — three function-local imports: ``mint_supervisor.
  delegation_refusal()``, ``aot_mint.declaration_module_gaps(...)`` and the
  ``aot_mint.ExportSpec`` construction.

The module-scope one is the load-bearing defect. pgw#1328's done-test is *"the
serve role boots, adopts, and serves a compiled class end-to-end in a process
where importing mint machinery raises"*, and a process that cannot import its
own executor never reaches the test. So the import moves behind this seam,
which resolves it LAZILY and only in the role that mints.

The function-local ones are already lazy, so the process boots either way —
but under :mod:`gen_worker.serve.guard` they would raise a bare
``MintMachineryUnavailable`` from inside somebody's ``except Exception``. Going
through the seam turns the same event into a
:class:`~gen_worker.serve.refusal.AdoptOnlyRefused` naming
``mint_forbidden``, which is a decision the fleet can count rather than an
import error it has to attribute.

WHY A PROTOCOL AND TWO IMPLEMENTATIONS, NOT AN ``if``
------------------------------------------------------
An ``if role.adopt_only(): ...`` at each call site is a second answer to "may
this pod compile" at every site that asks, which is exactly the drift channel
pgw#824's ``EagerPhase`` and pgw#1035's four literals were created to close.
One interface, two implementations, one dispatch function: a new mint call site
must pick an implementation, and the adopt-only one has no way to say yes.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Protocol

from . import role as role_mod
from .refusal import AdoptOnlyRefusal, AdoptOnlyRefused, MissKind, report

logger = logging.getLogger(__name__)


class MintSupervision(Protocol):
    """Everything the SERVING half asks of the mint lane. Nothing more.

    Deliberately narrow: this is not a facade over ``mint_supervisor``, it is
    the list of things a serving process actually calls. A future entry is a
    deliberate widening of what the serve role is allowed to be near.
    """

    def may_delegate(self) -> str:
        """"" when this worker may mint out of process, else the reason."""

    def make_task(self, **fields: Any) -> Any:
        """Build the supervisor's own task value for this obligation."""

    async def supervise(self, task: Any, *, act: Any, abandon: Any) -> Any:
        """Accrete one obligation's compiled graphs. Never raises for a miss."""

    def abandoned(self, status: str) -> bool:
        """Whether a supervised result carries the co-tenancy abandon signal."""

    def export_spec(self, pipe: Any, cfg: Any, **fields: Any) -> Any:
        """The export declaration a live serving pipeline describes."""

    def declaration_module_gaps(
        self, pipeline: Any, spec: Any, decl: Any,
    ) -> List[str]:
        """Declared input names the target module cannot take."""


class NoMint:
    """The adopt-only role. Every operation is a typed, reported refusal.

    It does not return an empty/neutral value for any of them. A ``may_delegate``
    that answered ``""`` would say "yes, mint out of process"; one that answered
    a plain reason string would let the caller degrade to an IN-PROCESS mint,
    which is the same capability by the other door. The only honest answer this
    role has is that it was asked to do something it exists not to do.
    """

    def _refuse(self, operation: str, *, function: str = "") -> AdoptOnlyRefused:
        return report(AdoptOnlyRefusal(
            kind=MissKind.MINT_FORBIDDEN, function=function,
            detail=f"{operation} was asked of the "
                   f"{role_mod.ServeRole.ADOPT_ONLY.value} serve role")).error()

    def may_delegate(self) -> str:
        raise self._refuse("may_delegate")

    def make_task(self, **fields: Any) -> Any:
        raise self._refuse("make_task", function=str(fields.get("function", "")))

    async def supervise(self, task: Any, *, act: Any, abandon: Any) -> Any:
        raise self._refuse(
            "supervise", function=str(getattr(task, "function", "") or ""))

    def abandoned(self, status: str) -> bool:
        raise self._refuse("abandoned")

    def export_spec(self, pipe: Any, cfg: Any, **fields: Any) -> Any:
        raise self._refuse("export_spec")

    def declaration_module_gaps(
        self, pipeline: Any, spec: Any, decl: Any,
    ) -> List[str]:
        raise self._refuse("declaration_module_gaps")


_NO_MINT: MintSupervision = NoMint()
_registered: Optional[MintSupervision] = None


class MintSupervisionUnregistered(RuntimeError):
    """An eager-capable process asked for a mint before one was plugged in."""


def register(impl: MintSupervision) -> None:
    """Plug the eager-capable implementation in. Called by the mint side.

    THE DEPENDENCY POINTS THE OTHER WAY, and it has to. A ``supervision()``
    that imported its own eager implementation — even lazily, even inside the
    function — puts the whole mint lane back in this module's static closure,
    which puts it back in the adopt-only role's. That is not a theory: it is
    what ``scripts/lint_serve_role_closure.py`` reported the first two times
    this seam was written, function-local import and all. So the serve side
    knows only the Protocol, and :mod:`gen_worker.mint_adapter` — imported by
    the eager-capable process host, never by this role — registers itself.
    """
    global _registered
    _registered = impl


def supervision() -> MintSupervision:
    """The mint implementation this process's declared role is entitled to.

    Adopt-only gets :class:`NoMint`, always, whether or not something
    registered: a process that imported the mint lane and THEN declared itself
    adopt-only must still refuse, because the role is about what it may do and
    not only about what it managed to import.
    """
    if role_mod.adopt_only():
        return _NO_MINT
    if _registered is None:
        raise MintSupervisionUnregistered(
            "this process holds the "
            f"{role_mod.ServeRole.EAGER_CAPABLE.value} serve role but no mint "
            "supervision is registered — import gen_worker.mint_adapter (the "
            "process host does) before serving")
    return _registered


def mint_forbidden(operation: str, *, function: str = "") -> AdoptOnlyRefused:
    """A reported ``mint_forbidden`` refusal for a caller outside this seam.

    Used where the mint is not a CALL into the supervisor but a decision — the
    executor's eager-first eligibility, which opens a background mint by
    building state rather than by invoking anything.
    """
    return report(AdoptOnlyRefusal(
        kind=MissKind.MINT_FORBIDDEN, function=function,
        detail=f"{operation} was asked of the "
               f"{role_mod.ServeRole.ADOPT_ONLY.value} serve role")).error()


__all__ = [
    "MintSupervision",
    "MintSupervisionUnregistered",
    "NoMint",
    "mint_forbidden",
    "register",
    "supervision",
]
