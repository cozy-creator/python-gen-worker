"""pgw#1118 / th#1773: the pod's OWN warm pass must not fail as the caller's.

th#1771 removed two TRIGGERS of the incident (the one-of synthesizer, the
integrity gate judging warm frames). It did not touch the MECHANISM: any other
exception escaping the warm pass still left the setup boundary untyped, hit the
job path's generic FATAL tail, and became `error_type='fatal'` with a bare
`ValueError` on the request that happened to wake the pod.

A warm forward runs a payload the WORKER synthesized and calls the endpoint's
own handler with it. No caller participates, so the origin is not in doubt —
it just was not stated. These tests pin the statement and its three exclusions.
"""

from __future__ import annotations

import asyncio
from typing import List

import msgspec
import pytest

from gen_worker import activity as activity_mod
from gen_worker import Resources, endpoint
from gen_worker.api.errors import (
    EndpointSetupFailed,
    ModelSlotIdentityError,
    RetryableError,
)
from gen_worker.executor import Executor, _map_exception, _typed_setup_fault
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs


class _In(msgspec.Struct):
    prompt: str = "x"


class _Out(msgspec.Struct):
    y: str


# The incident's exception, verbatim.
_REFS_REFUSAL = (
    "each references[] compiled_graph carries exactly one of image / video / audio, got none"
)


def test_warm_forward_fault_leaves_setup_typed() -> None:
    """The integration: a real `ensure_setup`, a real synthesized warm pass."""
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    @endpoint(resources=Resources(vram_gb_hint=8))
    class Ep:
        def setup(self) -> None:
            return None

        def generate(self, ctx, payload: _In) -> _Out:
            # What the handler does on the WARM payload the worker built for
            # itself. The caller's own payload never reaches this run.
            raise ValueError(_REFS_REFUSAL)

    specs = extract_specs(Ep)
    ex = Executor(specs, _send)

    with pytest.raises(EndpointSetupFailed) as caught:
        asyncio.run(ex.ensure_setup(specs[0]))

    fault = caught.value
    assert fault.phase == activity_mod.PHASE_WARMUP_FORWARD, (
        f"the fault must name the pass it happened in, got phase={fault.phase!r}"
    )
    assert isinstance(fault.cause, ValueError)
    assert _REFS_REFUSAL in str(fault), "the worker's verbatim error must survive"

    # And this is what the hub reads off the wire: a label it can route on,
    # with the phase in the detail.
    status, message = _map_exception(fault)
    assert status == pb.JOB_STATUS_FATAL
    assert message.startswith("EndpointSetupFailed: phase=warmup_forward function=")
    assert _REFS_REFUSAL in message


def test_load_phase_fault_is_not_claimed_as_the_release_s() -> None:
    """LOAD is out of scope: a caller-ROUTED slot can fail to resolve there,
    and th#1259's rule is that nothing a payload participates in producing may
    be labelled release-owned."""
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    @endpoint(resources=Resources(vram_gb_hint=8))
    class Ep:
        def setup(self) -> None:
            raise ValueError("load blew up")

        def generate(self, ctx, payload: _In) -> _Out:
            return _Out(y="ok")

    specs = extract_specs(Ep)
    ex = Executor(specs, _send)

    with pytest.raises(ValueError) as caught:
        asyncio.run(ex.ensure_setup(specs[0]))
    assert not isinstance(caught.value, EndpointSetupFailed), (
        "a LOAD-phase failure was claimed as a warm/compile fault"
    )


def test_typed_setup_fault_exclusions() -> None:
    warm = activity_mod.PHASE_WARMUP_FORWARD

    # The case it exists for.
    typed = _typed_setup_fault("reference-to-video", warm, ValueError(_REFS_REFUSAL))
    assert isinstance(typed, EndpointSetupFailed)
    assert str(typed).startswith("phase=warmup_forward function=reference-to-video: ")

    # Compile passes are the worker's own forwards too.
    assert _typed_setup_fault(
        "f", activity_mod.PHASE_INDUCTOR_COMPILE, RuntimeError("x")
    ) is not None
    assert _typed_setup_fault(
        "f", activity_mod.PHASE_TRACE_GRAPH, RuntimeError("x")
    ) is not None

    # Exclusion 1 — a phase a payload can participate in.
    assert _typed_setup_fault("f", activity_mod.PHASE_LOAD, ValueError("x")) is None

    # Exclusion 2 — already non-FATAL. A warm-phase OOM is still an OOM, and a
    # bigger/idler card serves it; re-typing would fatal a retryable job.
    assert _typed_setup_fault(
        "f", warm, RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    ) is None
    assert _typed_setup_fault("f", warm, RetryableError("try again")) is None

    # Exclusion 3 — already typed. Those labels are the hub's routing keys and
    # wrapping them would erase the origin the worker already claimed.
    assert _typed_setup_fault(
        "f", warm,
        ModelSlotIdentityError("f", "pipeline", declared_ref="a", dispatched_ref="b"),
    ) is None
