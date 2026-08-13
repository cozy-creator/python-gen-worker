"""pgw#1206 D: the Plan head is gone, and a RunAttempt is refused LOUDLY.

th#1457's RunAttempt producer landed hub-side (`2463f261`) and tensorhub
deleted it again in `ec978c68` (th#1842 Phase 2 Bucket A), keeping only the
proto and the `TestTH1457RunAttemptHasNoProducerBeforeTheCutover` fence. Two
heads read the wire and exactly one of them was ever driven; this file pins
which one died and what happens if the message ever appears anyway.

The refusal matters more than the deletion: a dropped `run_attempt` would be
tenant work silently discarded, and the hub's contract is accepted-or-result.
An unsupported command is fatal to the STREAM, which is the loud failure the
hub can see.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import inspect
from typing import cast

import pytest

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.executor import Executor
from gen_worker.lifecycle import Lifecycle
from gen_worker.transport import FatalTransportError


def test_the_plan_module_is_gone_with_no_shim() -> None:
    """Everything-is-v1: the module does not come back as a re-export."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("gen_worker.plan")


@pytest.mark.parametrize("name", [
    "handle_run_attempt", "_plan_order", "_plan_manifest_entry",
    "_grant_snapshots", "_materialize_arm", "_validate_plan_arm",
])
def test_no_plan_head_method_survives_on_the_executor(name: str) -> None:
    assert not hasattr(Executor, name)


def test_the_surviving_head_is_the_legacy_one() -> None:
    """One head, and it is the one the hub actually drives."""
    assert hasattr(Executor, "handle_run_job")
    assert hasattr(Executor, "_legacy_order")


def test_a_run_attempt_is_refused_as_an_unsupported_command() -> None:
    """The real dispatch path, not a stub of it.

    ``on_message``'s unknown-command arm touches nothing on ``self``, so this
    drives the production method itself with a bare object as the instance —
    no double, no monkeypatch. Reached the arm ⇒ the message is fatal.
    """
    msg = pb.SchedulerMessage(run_attempt=pb.RunAttempt())
    assert msg.WhichOneof("msg") == "run_attempt"

    bare = cast(Lifecycle, object())  # the arm reads nothing off `self`

    with pytest.raises(FatalTransportError) as err:
        asyncio.run(Lifecycle.on_message(bare, msg))
    assert "run_attempt" in str(err.value)


def test_the_dispatch_seam_reads_no_wire_message() -> None:
    """The reason deleting a head cost the driver nothing (pgw#904's seam):
    `dispatch` holds neutral orders and names no protobuf type."""
    mod = importlib.import_module("gen_worker.dispatch")
    assert not hasattr(mod, "pb")
    imported: list[str] = []
    for node in ast.walk(ast.parse(inspect.getsource(mod))):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not [m for m in imported if "pb" in m or "pb2" in m], imported
