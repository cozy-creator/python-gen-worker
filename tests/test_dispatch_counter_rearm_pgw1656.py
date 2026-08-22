"""The dispatch counter wraps a compiled callable AFTER it is armed.

`serving.dispatch_counter.rearm()` swaps each entry's compiled callable for a
counting wrapper by mutating `_ForwardDispatcher._entries` IN PLACE — that is
how the fleet learns a call served compiled at all.

tcg#90's rebuild routes dispatch through torchcg's `Dispatcher`, which is handed
the armed set at arm time. If it were handed the CALLABLES, this mutation would
be invisible to it and the failure would be silent AND INVERTED: the artifact
serves compiled while the counter reports zero, which reads exactly like "armed
and never entered" — the symptom that once cost a full night of counter-reading.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from gen_worker._vendor.torchcg.identity import CallIngress, CallInput
from gen_worker.graphs.adopt import _ForwardDispatcher
from gen_worker.graphs.document import GraphRecord



def _record() -> GraphRecord:
    return GraphRecord(
        graph="cg-graph-v1-" + "a" * 56,
        target="unet",
        ingress=CallIngress(
            parameters=("sample",),
            flat_arity=1,
            inputs=(CallInput("sample", 0, "sample", 0, (), "sample", "float32", (2, 4)),),
        ),
    )


class _Module(torch.nn.Module):
    def forward(self, *args: Any, **kwargs: Any) -> str:
        return "eager"


def test_a_callable_swapped_AFTER_arming_is_the_one_that_runs() -> None:
    module = _Module()
    dispatcher = _ForwardDispatcher(module)
    record = _record()
    dispatcher.arm(record, lambda *a, **k: "original")

    sample = torch.zeros(2, 4)
    assert dispatcher(sample) == "original"

    # What `rearm()` does: mutate the entry in place.
    seen: list[str] = []

    def wrapper(*a: Any, **k: Any) -> str:
        seen.append("counted")
        return "wrapped"

    entries = dispatcher._entries
    entries[0] = (entries[0][0], wrapper)

    assert dispatcher(sample) == "wrapped", (
        "the dispatcher called an arm-time snapshot, so the counter's wrapper "
        "never runs and compiled_graph_calls reports 0 while serving compiled"
    )
    assert seen == ["counted"]


def test_the_real_counter_sees_calls_through_the_rebuilt_dispatcher() -> None:
    """End to end through `dispatch_counter` itself, not a hand-rolled stand-in."""

    from gen_worker.serving.dispatch_counter import DispatchCounter

    module = _Module()
    dispatcher = _ForwardDispatcher(module)
    # What `AdoptSession._dispatcher_for` does: the instance attribute is what
    # `Module.__call__` reads, so the hook and the dispatcher see one call.
    module.forward = dispatcher  # type: ignore[method-assign,assignment]
    dispatcher.arm(_record(), lambda *a, **k: "compiled")

    class _Session:
        _dispatchers = [(module, dispatcher)]

    class _Host:
        adoption = _Session()

    counter = DispatchCounter().install(_Host())

    assert module(torch.zeros(2, 4)) == "compiled"
    counts = counter.take()
    assert counts.compiled_graph_calls == 1, counts.facts()
