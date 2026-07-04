"""ModelOp failure vocabulary: an OOM during LOAD must surface as
ModelEvent{FAILED, error="oom"} — the orchestrator's trigger to UNLOAD a
resident model for headroom. Everything else stays "load_failed"."""

from __future__ import annotations

import asyncio
from pathlib import Path

import msgspec
import pytest

from gen_worker.api.binding import HF
from gen_worker.executor import Executor, _model_op_error_vocab
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


class OutOfMemoryError(Exception):  # torch.cuda.OutOfMemoryError stand-in
    pass


class _In(msgspec.Struct):
    x: str


class _Out(msgspec.Struct):
    y: str


def test_model_op_error_vocab_classification() -> None:
    assert _model_op_error_vocab(OutOfMemoryError("CUDA out of memory")) == "oom"
    assert _model_op_error_vocab(RuntimeError("CUDA out of memory. Tried to allocate")) == "oom"
    assert _model_op_error_vocab(RuntimeError("shape mismatch")) == "load_failed"


@pytest.mark.parametrize(
    "exc, expected",
    [(OutOfMemoryError("CUDA out of memory"), "oom"),
     (RuntimeError("weights corrupt"), "load_failed")],
)
def test_model_op_load_failure_emits_vocab(tmp_path, exc, expected) -> None:
    binding = HF("acme/tiny")

    class Endpoint:
        def setup(self, model: str) -> None:
            raise exc

        def run(self, ctx, payload: _In) -> _Out:  # pragma: no cover
            return _Out(y=payload.x)

    spec = EndpointSpec(
        name="ep", method=Endpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint,
        attr_name="run", models={"model": binding},
    )

    sent: list[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    async def _run() -> None:
        ex = Executor([spec], _send)

        async def _fake_ensure_local(ref, snapshot=None, *, binding=None) -> Path:
            return tmp_path

        ex.store.ensure_local = _fake_ensure_local  # type: ignore[method-assign]
        await ex.handle_model_op(pb.ModelOp(op=pb.MODEL_OP_KIND_LOAD, ref="acme/tiny"))

    asyncio.run(_run())

    failed = [m for m in sent if m.WhichOneof("msg") == "model_event"
              and m.model_event.state == pb.MODEL_STATE_FAILED]
    assert failed, f"no FAILED ModelEvent emitted; sent={sent}"
    assert failed[-1].model_event.error == expected
    assert failed[-1].model_event.ref == "acme/tiny"
