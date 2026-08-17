"""pgw#1328: the EAGER-CAPABLE side of the serve/mint seam.

:class:`gen_worker.serve.mint_seam.MintSupervision` is the interface the
serving half calls; this is §4.28's implementation of it — the ordinary Python
serving pod, which mints as a side effect of serving. It lives HERE, beside
``mint_supervisor`` and ``aot_mint``, rather than in ``gen_worker.serve``,
because ``gen_worker.serve`` is the adopt-only role's own module set: an
implementation that names the mint lane from inside that set would put the lane
back in the role's static import closure, and the CI fence says so.

Every import below is function-local, for the same reason one level down: this
module IS on the mint side, but ``gen_worker.serve.mint_seam`` imports it
lazily from :func:`~gen_worker.serve.mint_seam.supervision`, and a module-scope
import here would be dragged in by any static analysis that follows that edge.
"""

from __future__ import annotations

from typing import Any, List

from .serve import mint_seam


class EagerCapableMint:
    """§4.28's ordinary serving pod: it mints as a side effect of serving.

    Every import is function-local ON PURPOSE. The point of the seam is that
    the mint lane is not in the importing module's static closure, so a
    module-scope import here would move the coupling rather than remove it —
    and the CI fence, which counts function-local imports too, would say so.
    """

    def may_delegate(self) -> str:
        from . import mint_supervisor

        return mint_supervisor.delegation_refusal()

    def make_task(self, **fields: Any) -> Any:
        from . import mint_supervisor

        return mint_supervisor.MintTask(**fields)

    async def supervise(self, task: Any, *, act: Any, abandon: Any) -> Any:
        from . import mint_supervisor

        return await mint_supervisor.supervise(task, act=act, abandon=abandon)

    def abandoned(self, status: str) -> bool:
        from . import mint_supervisor

        return status == mint_supervisor.ABANDONED

    def export_spec(self, pipe: Any, cfg: Any, **fields: Any) -> Any:
        from . import aot_mint

        return aot_mint.ExportSpec(**fields)

    def declaration_module_gaps(
        self, pipeline: Any, spec: Any, decl: Any,
    ) -> List[str]:
        from . import aot_mint

        return aot_mint.declaration_module_gaps(pipeline, spec, decl)



#: The one instance. Constructed at import so the seam's lazy import is the
#: only lazy step, not a lazy import plus a per-call construction.
EAGER_CAPABLE = EagerCapableMint()

#: Registration is a SIDE EFFECT OF THE IMPORT, which is the point: the only
#: process that can mint is the one that imported the mint side, and it does
#: not have to remember to say so.
mint_seam.register(EAGER_CAPABLE)

__all__ = ["EAGER_CAPABLE", "EagerCapableMint"]
