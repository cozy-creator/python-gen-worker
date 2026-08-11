"""pgw#1108: boot-adopt must not gate on the CHILD holding a worker JWT.

The executor runs in the compute child (pgw#783, the only execution model).
That child holds NO credential by construction — ``worker_jwt_provider()``
returns ``""`` (pgw#763 delta 1, ``child.py``'s ``current_worker_jwt``). The
resolve is a PARENT-mediated action (``cells.resolve``): the parent supplies the
credential and the base URL. So a ``not bearer`` gate in ``_boot_adopt`` refused
boot-adopt on every real serving pod — derive never ran, ``/v1/worker/cells/
resolve`` never fired, the pod fell straight through to self-mint, and the whole
compile-once-reuse-forever circle stayed open (measured on the 2026-08-11 2-pod
run: POD B re-minted, ZERO resolve calls, no ``trace_for_key``/``key_fold``).

The readiness that IS correct mirrors ``fleet_cells.CellPublisher``: a base URL
AND (a local bearer OR the control seam being up, ``broker.active()``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pytest

from gen_worker import executor as executor_mod
from gen_worker.procsplit import broker


@dataclass
class _Cfg:
    family: str = "micro-diffusion"


class _Spec:
    name = "generate"

    def compile_cell(self) -> _Cfg:
        return _Cfg()


def _executor(tmp_path: Path) -> Any:
    from gen_worker.executor import Executor, ModelStore

    async def _send(msg: Any) -> None:
        pass

    return Executor([], _send, store=ModelStore(_send, cache_dir=tmp_path / "cas"))


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """An executor whose ``_boot_adopt`` reaches the credential gate, with
    ``boot_adopt.attempt`` replaced by a recorder so the test observes ONLY
    whether the gate let the derive/resolve leg run — no trace child, no mint."""
    calls: list[Dict[str, Any]] = []

    def _attempt(**kw: Any) -> Any:
        calls.append(kw)
        return executor_mod.boot_adopt.BootAdoptOutcome(reason="miss")

    # Everything before the gate: a declaration exists and sizes one class.
    monkeypatch.setattr(executor_mod.aot_mint, "export_declaration",
                        lambda family: object())
    monkeypatch.setattr(executor_mod.aot_declaration, "cell_plans",
                        lambda decl: [object()])
    monkeypatch.setattr(executor_mod, "_mint_modules", lambda spec: ("m",))
    monkeypatch.setattr(executor_mod.fleet_cells, "declared_envelope_block",
                        lambda cfg: {})
    monkeypatch.setattr(executor_mod.boot_adopt, "attempt", _attempt)

    ex = _executor(tmp_path)
    ex.file_base_url = "http://hub.local"
    # The compute child's honest answer: it holds no credential.
    ex.worker_jwt_provider = lambda: ""

    # Isolate the process-global broker seam for this test.
    monkeypatch.setattr(broker, "_broker", None, raising=False)
    return ex, calls


def _run_boot_adopt(ex: Any) -> Any:
    return ex._boot_adopt(_Spec(), {})


def test_split_child_with_seam_up_resolves_though_it_holds_no_jwt(wired):
    """THE regression: bearer is "" (compute child) but the control seam is up,
    so the parent can answer. boot-adopt MUST proceed to derive+resolve."""
    ex, calls = wired
    broker.install(object())  # seam up: broker.active() is True
    try:
        _run_boot_adopt(ex)
    finally:
        broker.install(None)
    assert len(calls) == 1, (
        "boot-adopt refused to even attempt derive/resolve on a split pod whose "
        "seam is up — the pod would self-mint instead of adopting")
    assert calls[0]["cfg"].family == "micro-diffusion"


def test_no_bearer_and_no_seam_degrades_without_deriving(wired):
    """The gate must STILL protect the genuinely-no-hub case: no local bearer
    and no seam means nobody to ask, so no derive/resolve, no attempt.

    pgw#1116: it degrades by NAMING itself (`no_hub`), never by returning a
    bare None — a refusal that carries no reason is how three pods refused
    unattributably.
    """
    ex, calls = wired
    # broker._broker is None (fixture); seam is down.
    out = _run_boot_adopt(ex)
    assert out is not None and out.reason == "no_hub"
    assert not out.adopted
    assert calls == []


def test_single_process_bearer_still_resolves_with_no_seam(wired):
    """Embedded/single-process: broker is None but the executor holds a real
    JWT locally. Unchanged — boot-adopt proceeds."""
    ex, calls = wired
    ex.worker_jwt_provider = lambda: "real-jwt"
    out = _run_boot_adopt(ex)
    # NOT `x is not None or calls` — that disjunction could never fail on the
    # old code either, which is the tautology class pgw#1113 is fixing.
    assert len(calls) == 1, "boot-adopt did not attempt derive/resolve"
    assert out.reason == "miss"
