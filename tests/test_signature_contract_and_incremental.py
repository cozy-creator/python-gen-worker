import json
import unittest
from typing import Annotated, Iterator

import msgspec

from gen_worker.injection import ModelRef, ModelRefSource as Src
from gen_worker.worker import ActionContext, Worker


class Input(msgspec.Struct):
    text: str


class Delta(msgspec.Struct):
    delta: str


class InputWithModel(msgspec.Struct):
    text: str
    model_key: str


class Output(msgspec.Struct):
    model_id: str


class FakeModel:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    @classmethod
    def from_pretrained(cls, model_id: str) -> "FakeModel":
        return cls(model_id)


def _make_worker() -> Worker:
    w = Worker.__new__(Worker)
    import threading

    w._gpu_busy_lock = threading.Lock()
    w._is_gpu_busy = False
    w._has_gpu = False
    w.max_output_bytes = 0
    w._model_manager = None
    w._runtime_loaders = {}
    w._custom_runtime_cache = {}
    w._custom_runtime_locks = {}
    w._release_model_id_by_key = {}
    w._model_id_by_key_by_function = {}
    w._release_allowed_model_ids = None
    w._active_tasks_lock = threading.Lock()
    w._active_tasks = {}
    w._active_function_counts = {}
    w._send_message = lambda msg: w._sent.append(msg)  # type: ignore[method-assign]
    w._sent = []
    w._stop_event = threading.Event()
    w._running = True
    w._materialize_assets = lambda ctx, obj: None  # type: ignore[method-assign]
    w._discovered_resources = {}
    w._model_cache = None
    return w


class TestContractAndIncremental(unittest.TestCase):
    def test_rejects_missing_return_annotation(self) -> None:
        def bad(ctx: ActionContext, payload: Input):  # type: ignore[no-untyped-def]
            return Delta(delta="x")

        w = _make_worker()
        with self.assertRaises(ValueError):
            w._inspect_task_spec(bad)  # type: ignore[arg-type]

    def test_incremental_output_emits_deltas_and_completed(self) -> None:
        def stream(ctx: ActionContext, payload: Input) -> Iterator[Delta]:
            yield Delta(delta=payload.text)
            yield Delta(delta="!")

        w = _make_worker()
        spec = w._inspect_task_spec(stream)  # type: ignore[arg-type]
        self.assertEqual(spec.output_mode, "incremental")

        ctx = ActionContext("run-1", emitter=lambda _e: None)
        payload = Input(text="hi")
        b = msgspec.msgpack.encode(msgspec.to_builtins(payload))
        w._execute_task(ctx, spec, b)

        # Capture worker_event messages.
        events = []
        for m in w._sent:
            if not hasattr(m, "WhichOneof") or m.WhichOneof("msg") != "worker_event":
                continue
            evt = m.worker_event
            # Ignore non-output events (e.g. metrics.*).
            if not str(evt.event_type or "").startswith("output."):
                continue
            events.append((evt.event_type, evt.payload_json))

        self.assertGreaterEqual(len(events), 3)
        self.assertEqual(events[0][0], "output.delta")
        self.assertEqual(json.loads(events[0][1].decode("utf-8"))["delta"], "hi")
        self.assertEqual(events[1][0], "output.delta")
        self.assertEqual(json.loads(events[1][1].decode("utf-8"))["delta"], "!")
        self.assertEqual(events[2][0], "output.completed")

    def test_payload_model_key_resolves_via_endpoint_map(self) -> None:
        def fn(
            ctx: ActionContext,
            model: Annotated[FakeModel, ModelRef(Src.PAYLOAD, "model_key")],
            payload: InputWithModel,
        ) -> Output:
            return Output(model_id=model.model_id)

        w = _make_worker()
        w._release_model_id_by_key = {"a": "google/foo"}
        w._release_allowed_model_ids = {"google/foo"}
        spec = w._inspect_task_spec(fn)  # type: ignore[arg-type]
        ctx = ActionContext("run-2", emitter=lambda _e: None)
        payload = InputWithModel(text="x", model_key="a")
        b = msgspec.msgpack.encode(msgspec.to_builtins(payload))
        w._execute_task(ctx, spec, b)

        run_results = [m.run_result for m in w._sent if hasattr(m, "WhichOneof") and m.WhichOneof("msg") == "run_result"]
        self.assertEqual(len(run_results), 1)
        rr = run_results[0]
        self.assertTrue(rr.success)
        out = msgspec.msgpack.decode(rr.output_payload, type=dict)
        self.assertEqual(out["model_id"], "google/foo")

    def test_payload_model_key_rejects_not_allowlisted(self) -> None:
        def fn(
            ctx: ActionContext,
            model: Annotated[FakeModel, ModelRef(Src.PAYLOAD, "model_key")],
            payload: InputWithModel,
        ) -> Output:
            return Output(model_id=model.model_id)

        w = _make_worker()
        w._release_model_id_by_key = {"a": "google/foo", "b": "google/bar"}
        w._release_allowed_model_ids = {"google/foo"}
        spec = w._inspect_task_spec(fn)  # type: ignore[arg-type]
        ctx = ActionContext("run-3", emitter=lambda _e: None)
        payload = InputWithModel(text="x", model_key="b")
        b = msgspec.msgpack.encode(msgspec.to_builtins(payload))
        w._execute_task(ctx, spec, b)

        run_results = [m.run_result for m in w._sent if hasattr(m, "WhichOneof") and m.WhichOneof("msg") == "run_result"]
        self.assertEqual(len(run_results), 1)
        rr = run_results[0]
        self.assertFalse(rr.success)
        self.assertEqual(rr.error_type, "validation")
