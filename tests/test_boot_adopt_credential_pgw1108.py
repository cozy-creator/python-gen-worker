"""pgw#1108: boot-adopt must not gate on the CHILD holding a worker JWT.

The executor runs in the compute child (pgw#783, the only execution model).
That child holds NO credential by construction — ``worker_jwt_provider()``
returns ``""`` (pgw#763 delta 1, ``child.py``'s ``current_worker_jwt``). The
resolve is a PARENT-mediated action (``compiled_graphs.resolve``): the parent supplies the
credential and the base URL. So a ``not bearer`` gate in ``_boot_adopt`` refused
boot-adopt on every real serving pod — derive never ran, ``/v1/worker/compiled-graphs/
resolve`` never fired, the pod fell straight through to self-mint, and the whole
compile-once-reuse-forever circle stayed open (measured on the 2026-08-11 2-pod
run: POD B re-minted, ZERO resolve calls, no ``trace_for_key``/``key_fold``).

The readiness that IS correct mirrors ``fleet_compiled_graphs.CompiledGraphPublisher``: a base URL
AND (a local bearer OR the control seam being up, ``broker.active()``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pytest

from gen_worker import executor as executor_mod
from gen_worker.api import export_contract as export_contract_mod
from gen_worker.procsplit import broker


@dataclass
class _Cfg:
    family: str = "micro-diffusion"


class _Spec:
    name = "generate"

    def compile_contract(self) -> _Cfg:
        return _Cfg()


def _executor(tmp_path: Path) -> Any:
    from gen_worker.executor import Executor
    from gen_worker.models.store import ModelStore

    async def _send(msg: Any) -> None:
        pass

    return Executor([], _send, store=ModelStore(_send, cache_dir=tmp_path / "cas"))


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """An executor whose ``_boot_adopt`` reaches the credential gate, with
    ``boot_adopt.attempt`` replaced by a recorder so the test observes ONLY
    whether the gate let the derive/resolve leg run — no trace child, no mint."""
    calls: list[Dict[str, Any]] = []

    # The pre-derive gate now asks whether ANYBODY could answer, and
    # this machine's own compiled graph store is one of the two answerers. Pin it to an
    # empty tmp root so the gate under test reads a fact about the test and not
    # about whatever the developer's `~/.cache/cozy/compiled graphs` happens to
    # hold — the ambient-input class of flake this repo has been bitten by.
    from gen_worker import local_compiled_graph_store

    monkeypatch.setenv(
        local_compiled_graph_store.ENV_STORE_DIR, str(tmp_path / "empty-compiled graphs"))

    def _attempt(**kw: Any) -> Any:
        calls.append(kw)
        # `attempt` returns ONE outcome per declared graph class.
        # `compiled_graph_plans` above sizes exactly one, so the double returns a
        # one-element TUPLE — a double still handing back a bare outcome would
        # let this suite pass against a contract production does not have.
        return (executor_mod.boot_adopt.BootAdoptOutcome(reason="miss"),)

    # Everything before the gate: a declaration exists and sizes one class.
    monkeypatch.setattr(export_contract_mod, "export_declaration",
                        lambda family: object())
    monkeypatch.setattr(executor_mod.aot_declaration, "compiled_graph_plans",
                        lambda decl: [object()])
    monkeypatch.setattr(executor_mod, "_mint_modules", lambda spec: ("m",))
    monkeypatch.setattr(executor_mod.boot_adopt, "attempt", _attempt)

    ex = _executor(tmp_path)
    ex.file_base_url = "http://hub.local"
    # The compute child's honest answer: it holds no credential.
    ex.worker_jwt_provider = lambda: ""

    # Isolate the process-global broker seam for this test.
    monkeypatch.setattr(broker, "_broker", None, raising=False)
    return ex, calls


def _run_boot_adopt(ex: Any) -> Any:
    # One outcome per declared class, and this suite declares one —
    # so the unpack ASSERTS that arity rather than indexing past a set nobody
    # checked. Every gate below is about the ONE declaration.
    (out,) = ex._boot_adopt(_Spec(), {})
    return out


