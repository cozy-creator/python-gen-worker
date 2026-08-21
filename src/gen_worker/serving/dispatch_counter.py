"""Per-request truth about WHERE a forward ran: compiled graph, or eager."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

_WRAPPED = "_gen_worker_dispatch_counted"


@dataclass(frozen=True, slots=True)
class DispatchCounts:
    """One request's dispatch facts."""

    module_calls: int
    compiled_graph_calls: int
    armed_modules: int
    armed_graphs: int
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
        """The line a human reads after every request."""
        if self.armed_modules == 0:
            return "dispatch: no compiled graph armed — served eager (nothing adopted)"
        if self.displaced_modules:
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
    """Counts compiled-vs-eager forwards for one booted endpoint."""

    def __init__(self) -> None:
        self._module_calls = 0
        self._compiled_calls = 0
        self._targets: List[Tuple[str, Any, Any]] = []
        self._hooks: List[Any] = []

    def install(self, host: Any) -> "DispatchCounter":
        """Attach to every module this boot adopted."""
        session = getattr(host, "adoption", None)
        if session is None:
            return self
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
        """Wrap any compiled callable armed since the last pass."""
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

    def reset(self) -> None:
        self._module_calls = 0
        self._compiled_calls = 0

    def take(self) -> DispatchCounts:
        """Snapshot the counters and the live witness, then reset."""
        armed_modules = 0
        armed_graphs = 0
        displaced: List[str] = []
        for label, module, dispatcher in self._targets:
            graphs = tuple(getattr(dispatcher, "armed_graphs", lambda: ())())
            if graphs:
                armed_modules += 1
                armed_graphs += len(graphs)
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
