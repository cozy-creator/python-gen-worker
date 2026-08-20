"""Per-request truth about WHERE a forward ran: compiled graph, or eager.

pgw#1491, and it is not a harness nicety. On 2026-08-19 a broken dispatch guard
served twelve perfect images entirely eager, at eager speed, with no complaint
from torch, from diffusers or from AOTI. "Compiled is about the same as eager"
was one counter away from being published as a measurement. Silent fall-through
must be COUNTABLE at the serving surface, permanently.

## Three numbers, and why it takes three

``module_calls`` — counted by a ``register_forward_pre_hook`` on the adopted
module. torch runs hooks inside ``Module._call_impl``, BEFORE it reads
``self.forward``, so this number survives anything that replaces ``forward``.

``compiled_graph_calls`` — counted by a wrapper around the compiled callable
that torchcg's ``_ForwardDispatcher`` holds, incremented at the moment the
compiled artifact is actually entered.

``eager_calls`` = ``module_calls - compiled_graph_calls``, DERIVED and never
counted separately, because the two counters answer different questions and a
third independent tally could only ever disagree with the arithmetic.

## The witness, and the bug it exists to catch

torchcg installs its dispatcher as the module's INSTANCE ``forward``
(``adopt._ForwardDispatcher``). An accelerate-managed offload pipeline
re-attaches its own wrapper to ``forward`` per call — so an instrument attached
by assignment is silently displaced, the pipeline keeps working, and the
instrument reads 0. That is the second bug of this species measured in one day.

The pre-hook is the INDEPENDENT witness: it counts calls that reach the module
no matter what owns ``forward``. So

* ``module_calls == compiled + eager`` and ``compiled > 0`` — compiled served;
* ``compiled == 0`` with ``module_calls > 0`` — everything fell through eager,
  and the count SAYS so instead of the timing implying it;
* ``forward is not the dispatcher`` — the dispatcher was DISPLACED, reported by
  name as :attr:`DispatchCounts.displaced_modules` rather than showing up as an
  unexplained zero.

Absence never renders as zero: a boot that adopted nothing reports
``armed_modules=0``, which is a different word from "armed and never entered".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

#: Marks a callable this module already wrapped, so re-arming after a late mint
#: never double-counts the artifacts it wrapped on the previous pass.
_WRAPPED = "_gen_worker_dispatch_counted"


@dataclass(frozen=True, slots=True)
class DispatchCounts:
    """One request's dispatch facts. Every field is a live read."""

    #: Calls that reached an adopted module (pre-hook; survives wrapper churn).
    module_calls: int
    #: Calls that entered a compiled artifact.
    compiled_graph_calls: int
    #: Modules carrying at least one armed graph at the time of the request.
    armed_modules: int
    #: Armed graphs across those modules.
    armed_graphs: int
    #: Modules whose ``forward`` is no longer torchcg's dispatcher — something
    #: displaced it, so nothing on them can dispatch compiled.
    displaced_modules: Tuple[str, ...]

    @property
    def eager_calls(self) -> int:
        """Derived, never counted: the calls that fell through to eager."""
        return max(0, self.module_calls - self.compiled_graph_calls)

    def facts(self) -> Dict[str, Any]:
        return {
            "module_calls": self.module_calls,
            "compiled_graph_calls": self.compiled_graph_calls,
            "eager_calls": self.eager_calls,
            "armed_modules": self.armed_modules,
            "armed_graphs": self.armed_graphs,
            "displaced_modules": list(self.displaced_modules),
        }

    def summary(self) -> str:
        """The line a human reads after every request.

        It names the state in WORDS, because a reader scanning a log for
        "did that run compiled" must not have to do the subtraction himself.
        """
        if self.armed_modules == 0:
            return "dispatch: no compiled graph armed — served eager (nothing adopted)"
        if self.displaced_modules:
            # THE COUNTS, NEVER AN INFERENCE FROM THE FLAG (pgw#1591). This
            # branch used to end "so all N call(s) ran eager" — a claim it did
            # not measure and, in the field, a false one: the sd15 benchmark
            # read it on 12/12 requests of both arms while 120 AOTI wrapper
            # invocations per arm were in the same log, and the lane could not
            # reconcile the two because one of them was not a measurement.
            # Displacement and dispatch are separate facts; both are stated.
            served = (
                f"{self.compiled_graph_calls} of {self.module_calls} call(s) "
                f"still served COMPILED"
                if self.compiled_graph_calls
                else f"all {self.module_calls} call(s) ran eager"
            )
            return (
                f"dispatch: DISPLACED on {', '.join(self.displaced_modules)} — "
                f"the compiled dispatcher is no longer reachable as this "
                f"module's forward; {served}"
            )
        if self.compiled_graph_calls == 0:
            return (
                f"dispatch: {self.armed_graphs} graph(s) armed on "
                f"{self.armed_modules} module(s) and NONE was entered — all "
                f"{self.module_calls} call(s) fell through to eager"
            )
        return (
            f"dispatch: compiled_graph_calls={self.compiled_graph_calls} "
            f"eager_calls={self.eager_calls} "
            f"(armed {self.armed_graphs} graph(s) on {self.armed_modules} module(s))"
        )


class DispatchCounter:
    """Counts compiled-vs-eager forwards for one booted endpoint.

    One instance per :class:`~gen_worker.serving.host.EndpointHost`. Attach with
    :meth:`install`, snapshot per request with :meth:`take`.
    """

    def __init__(self) -> None:
        self._module_calls = 0
        self._compiled_calls = 0
        #: (label, module, dispatcher) — the adopted modules, for the witness.
        self._targets: List[Tuple[str, Any, Any]] = []
        self._hooks: List[Any] = []

    # -- attachment ---------------------------------------------------------

    def install(self, host: Any) -> "DispatchCounter":
        """Attach to every module this boot adopted. Idempotent per module.

        Safe on a host that adopted nothing (the eager bridge): there is
        nothing to attach to, and the counts then report ``armed_modules=0``.
        """
        session = getattr(host, "adoption", None)
        if session is None:
            return self
        # torchcg exposes adopted GRAPHS publicly but not the modules they were
        # adopted onto; the dispatcher list is the only handle on the objects
        # that actually route. Read-only, and re-derived on every rearm so a
        # torchcg bump that renames it fails loudly here rather than quietly
        # zeroing the instrument.
        pairs = getattr(session, "_dispatchers", None)
        if not pairs:
            return self
        for module, dispatcher in list(pairs):
            label = type(module).__name__
            if any(existing is module for _, existing, _ in self._targets):
                continue
            self._targets.append((label, module, dispatcher))
            self._hooks.append(module.register_forward_pre_hook(self._count_module))
        self.rearm()
        return self

    def rearm(self) -> None:
        """Wrap any compiled callable armed since the last pass.

        The background mint arms artifacts onto the SAME session after boot, so
        a counter that wrapped only what boot adopted would under-count every
        graph the mint filled in — reading as fall-through exactly where the
        interesting thing just happened.
        """
        for _label, _module, dispatcher in self._targets:
            entries = getattr(dispatcher, "_entries", None)
            if entries is None:
                continue
            for index, (record, compiled) in enumerate(list(entries)):
                if getattr(compiled, _WRAPPED, False):
                    continue
                entries[index] = (record, self._wrap(compiled))

    def _wrap(self, compiled: Callable[..., Any]) -> Callable[..., Any]:
        def counted(*args: Any, **kwargs: Any) -> Any:
            self._compiled_calls += 1
            return compiled(*args, **kwargs)

        setattr(counted, _WRAPPED, True)
        return counted

    def _count_module(self, _module: Any, _args: Any) -> None:
        self._module_calls += 1

    # -- reading ------------------------------------------------------------

    def reset(self) -> None:
        self._module_calls = 0
        self._compiled_calls = 0

    def take(self) -> DispatchCounts:
        """Snapshot the counters and the live witness, then reset.

        The armed/displaced facts are read from the modules AT THIS MOMENT, not
        remembered from install time — a stamp echoed back from boot would say
        "armed" about a module something has since taken over.
        """
        armed_modules = 0
        armed_graphs = 0
        displaced: List[str] = []
        for label, module, dispatcher in self._targets:
            graphs = tuple(getattr(dispatcher, "armed_graphs", lambda: ())())
            if graphs:
                armed_modules += 1
                armed_graphs += len(graphs)
            # THROUGH any wrapper (pgw#1591). `module.forward is dispatcher`
            # calls every WRAPPED dispatcher displaced — and pgw#1573's
            # adapter guard is a legitimate wrapper installed on every adopted
            # module, so this read went false fleet-wide the moment it landed.
            # `adapter_guard.dispatcher_of` is the one resolver; a module it
            # cannot resolve is genuinely displaced (accelerate restoring
            # `_old_forward` after an offload rung is the real shape of that).
            from .adapter_guard import dispatcher_of

            if dispatcher_of(module) is not dispatcher:
                displaced.append(label)
        counts = DispatchCounts(
            module_calls=self._module_calls,
            compiled_graph_calls=self._compiled_calls,
            armed_modules=armed_modules,
            armed_graphs=armed_graphs,
            displaced_modules=tuple(displaced),
        )
        self.reset()
        return counts

    def close(self) -> None:
        for hook in self._hooks:
            try:
                hook.remove()
            except Exception:  # noqa: BLE001 — teardown must not raise
                pass
        self._hooks.clear()
        self._targets.clear()


__all__ = ["DispatchCounter", "DispatchCounts"]
