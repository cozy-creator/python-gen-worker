"""Instrument: did the test enter at the PRODUCTION entrypoint, or at the unit
beneath it?

A unit test is structurally blind to wiring, because the unit test IS the
caller the production path is not. The only thing that catches that class is a
test that enters where production enters and lets production do the calling.

This module makes that mechanically checkable. Declare a LADDER — the ordered
production call chain for a path — wrap it, run a real scenario, and ask two
questions of the recording:

* was the deepest rung reached at all?
* when it was reached, was the whole chain above it ON THE CHAIN — i.e. did
  PRODUCTION call it, not the test?

The chain is carried in a ``ContextVar``, not read off the Python stack, so it
survives the two hops a stack walk loses: ``asyncio.create_task`` (the mint is
launched as a task) and ``asyncio.to_thread`` (the AOT arm runs there). Both
copy the caller's context. It does NOT survive a raw ``threading.Thread`` or an
OS process boundary; rungs past such a hop must be declared as a separate
ladder, which is honest rather than clever.
"""

from __future__ import annotations

import contextvars
import functools
import importlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

SRC = (Path(__file__).resolve().parents[1] / "src" / "gen_worker").resolve()
TESTS = Path(__file__).resolve().parent

_CHAIN: contextvars.ContextVar[Tuple[str, ...]] = contextvars.ContextVar(
    "pgw849_ladder_chain", default=())


@dataclass
class Entry:
    """One observed call of one rung."""
    chain: Tuple[str, ...]      # rungs already entered, outermost first
    caller: str                 # file:line of the immediate non-instrument frame

    @property
    def from_production(self) -> bool:
        try:
            Path(self.caller.split(":")[0]).resolve().relative_to(SRC)
        except ValueError:
            return False
        return True


@dataclass
class Recording:
    rungs: Tuple[str, ...]
    entries: Dict[str, List[Entry]] = field(default_factory=dict)

    def reached(self, rung: str) -> bool:
        return bool(self.entries.get(rung))

    def deepest_reached(self) -> str | None:
        for rung in reversed(self.rungs):
            if self.reached(rung):
                return rung
        return None

    def gap(self) -> str | None:
        """The first rung the run failed to traverse from production, or None.

        A rung passes when SOME call of it arrived with every rung above it
        already on the chain. The top rung is exempt from the chain test — it
        is the front door, so a test is allowed to be its caller — but it must
        still have been reached.
        """
        for idx, rung in enumerate(self.rungs):
            calls = self.entries.get(rung, [])
            if not calls:
                return rung
            if idx == 0:
                continue
            above = set(self.rungs[:idx])
            if not any(above <= set(c.chain) for c in calls):
                return rung
        return None

    def report(self) -> str:
        lines = []
        for idx, rung in enumerate(self.rungs):
            calls = self.entries.get(rung, [])
            if not calls:
                lines.append(f"  {idx}. {rung}  — NEVER REACHED")
                continue
            above = set(self.rungs[:idx])
            ok = idx == 0 or any(above <= set(c.chain) for c in calls)
            how = "from production" if any(c.from_production for c in calls) \
                else f"from the TEST ({calls[0].caller})"
            lines.append(
                f"  {idx}. {rung}  — {len(calls)} call(s) {how}"
                f"{'' if ok else '  <-- chain BROKEN above this rung'}")
        return "\n".join(lines)


def _resolve(dotted: str) -> Tuple[Any, str]:
    """``gen_worker.executor.Executor.handle_run_job`` -> (Executor, name).

    Raises on a stale ledger entry: a rename that silently drops a rung would
    turn this guard into decoration.
    """
    parts = dotted.split(".")
    for split in range(len(parts) - 1, 0, -1):
        mod_name = ".".join(parts[:split])
        try:
            obj: Any = importlib.import_module(mod_name)
        except ImportError:
            continue
        for attr in parts[split:-1]:
            obj = getattr(obj, attr)
        if not hasattr(obj, parts[-1]):
            raise AttributeError(
                f"pgw#849 ladder: {dotted!r} does not exist — the ledger is "
                f"stale (renamed or deleted rung)")
        return obj, parts[-1]
    raise ImportError(f"pgw#849 ladder: cannot import any prefix of {dotted!r}")


def _caller() -> str:
    frame = inspect.currentframe()
    this = str(Path(__file__).resolve())
    while frame is not None:
        fname = str(Path(frame.f_code.co_filename).resolve())
        if fname != this:
            return f"{fname}:{frame.f_lineno}"
        frame = frame.f_back
    return "<unknown>"


class ladder:
    """Wrap ``rungs`` for the duration of the ``with`` block and record how
    each one was entered.

        with ladder(*SERVE) as rec:
            ... drive a real job over the wire ...
        assert rec.gap() is None, rec.report()
    """

    def __init__(self, *rungs: str) -> None:
        self.rungs = tuple(rungs)
        self.rec = Recording(self.rungs)
        self._undo: List[Callable[[], None]] = []

    def __enter__(self) -> Recording:
        for rung in self.rungs:
            owner, attr = _resolve(rung)
            original = getattr(owner, attr)
            self._undo.append(
                functools.partial(setattr, owner, attr, original))
            setattr(owner, attr, self._wrap(rung, original))
        return self.rec

    def __exit__(self, *exc: Any) -> None:
        for undo in reversed(self._undo):
            undo()
        self._undo.clear()

    def _record(self, rung: str) -> contextvars.Token:
        chain = _CHAIN.get()
        self.rec.entries.setdefault(rung, []).append(
            Entry(chain=chain, caller=_caller()))
        return _CHAIN.set(chain + (rung,))

    def _wrap(self, rung: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def awrapper(*a: Any, **k: Any) -> Any:
                token = self._record(rung)
                try:
                    return await fn(*a, **k)
                finally:
                    _CHAIN.reset(token)
            return awrapper

        @functools.wraps(fn)
        def wrapper(*a: Any, **k: Any) -> Any:
            token = self._record(rung)
            try:
                return fn(*a, **k)
            finally:
                _CHAIN.reset(token)
        return wrapper


