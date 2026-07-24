"""pgw#637: dynamo's in-memory code cache is a legitimate serving surface.

Cell keys are checkpoint-free by design, so the 2nd checkpoint of an
already-minted family serves its warmup from dynamo's in-memory compiled
code with ZERO FX/AOT counter movement (torch 2.13, inlined nn-modules).
The finalize proof must credit that surface when the cell was already
proven in-process, and the disproof cleanup must never fire the global
``torch._dynamo.reset()`` while a healthy sibling's compiled code is live.
"""

from __future__ import annotations

import threading
from typing import Any, Iterator, List

import pytest

from gen_worker import compile_cache as cc


@pytest.fixture(autouse=True)
def _clean_process_registries() -> Iterator[None]:
    with cc._PROVEN_CELLS_LOCK:
        cc._PROVEN_CELLS.clear()
    armed = cc._armed_pipelines()
    for pipe in list(armed):
        armed.discard(pipe)
    yield
    with cc._PROVEN_CELLS_LOCK:
        cc._PROVEN_CELLS.clear()
    for pipe in list(armed):
        armed.discard(pipe)


def test_proven_cell_registry_roundtrip() -> None:
    ref = "cozy-fleet/cells/sdxl-rtx-4090-w8a8:abc123"
    assert cc.cell_proven_in_process(ref) is False
    cc.record_cell_proven(ref)
    assert cc.cell_proven_in_process(ref) is True
    # Whitespace/empty never register.
    cc.record_cell_proven("")
    cc.record_cell_proven("   ")
    assert cc.cell_proven_in_process("") is False


class _Mod:
    def forward(self, x: Any) -> Any:  # pragma: no cover - never called
        return x


class _Pipe:
    def __init__(self) -> None:
        self.transformer = _Mod()


def _armed_pipe() -> _Pipe:
    """A pipeline carrying a real apply()-shaped marker (manual arm: apply()
    itself refuses on CPU-only hosts, which is exactly why the unwrap
    scoping must be testable without CUDA)."""
    pipe = _Pipe()
    original = pipe.transformer.forward
    setattr(pipe, cc._MARKER_ATTR, {
        "targets": ["transformer"],
        "shapes": [(8, 8)],
        "cache": True,
        "originals": [(pipe.transformer, "forward", original)],
        "regional_mods": [],
        "failure_signal": {
            "callback": None,
            "lock": threading.Lock(),
            "successful_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "router": None,
        },
    })
    cc._armed_pipelines().add(pipe)
    return pipe


def test_inmemory_probe_reports_dynamo_truth_not_the_registry() -> None:
    """The in-memory credit needs DIRECT dynamo evidence — the proven-cell
    registry alone would let one object's hit certify another's silence
    (gw#603/gw#611). Nothing here is compiled, so the probe says no."""
    pytest.importorskip("torch")
    assert cc.has_inmemory_compiled_code(object()) is False
    pipe = _armed_pipe()
    cc.record_cell_proven("cozy-fleet/cells/whatever:abc")
    assert cc.has_inmemory_compiled_code(pipe) is False


def test_inmemory_probe_sees_a_live_dynamo_cache_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real mechanism: dynamo keys its code cache on the target's
    ``__code__``, which every checkpoint's pipeline instance shares."""
    pytest.importorskip("torch")
    from torch._dynamo import eval_frame

    pipe = _armed_pipe()
    seen: List[Any] = []

    def _entries(code: Any) -> List[Any]:
        seen.append(code)
        return [object()]

    monkeypatch.setattr(eval_frame, "_debug_get_cache_entry_list", _entries)
    assert cc.has_inmemory_compiled_code(pipe) is True
    assert seen and seen[0] is _Mod.forward.__code__


def test_unwrap_never_globally_resets_dynamo_while_siblings_are_armed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    import torch._dynamo

    resets: List[int] = []
    monkeypatch.setattr(torch._dynamo, "reset", lambda: resets.append(1))

    first = _armed_pipe()   # the healthy 1st checkpoint
    second = _armed_pipe()  # the failing 2nd checkpoint

    # Disproof cleanup of the 2nd checkpoint: the 1st's compiled code lives.
    assert cc.unwrap(second) is True
    assert resets == []
    assert getattr(second, cc._MARKER_ATTR, None) is None
    # The last armed pipeline leaving DOES reset (clean re-trace next apply).
    assert cc.unwrap(first) is True
    assert resets == [1]


def test_unwrap_still_restores_eager_callables_when_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    import torch._dynamo

    monkeypatch.setattr(torch._dynamo, "reset", lambda: None)
    keeper = _armed_pipe()
    victim = _armed_pipe()
    original = victim._cozy_compile["originals"][0][2]  # type: ignore[attr-defined]
    victim.transformer.forward = lambda x: "compiled"  # type: ignore[assignment]
    assert cc.unwrap(victim) is True
    assert victim.transformer.forward is original
    assert cc.unwrap(keeper) is True
