from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from gen_worker.trainer import (
    StepContext,
    StepControlHints,
    StepResult,
    TrainingCanceled,
    TrainingJobSpec,
    run_training_loop,
)


@dataclass
class FakeReporter:
    canceled: bool = False
    started_runs: list[str] = field(default_factory=list)
    metrics: list[tuple[str, float, int]] = field(default_factory=list)
    checkpoints: list[tuple[str, int]] = field(default_factory=list)
    samples: list[tuple[str, int]] = field(default_factory=list)
    completed_calls: list[tuple[int, str | None]] = field(default_factory=list)
    failed_calls: list[tuple[int, str]] = field(default_factory=list)

    def started(self, *, run_id: str) -> None:
        self.started_runs.append(run_id)

    def metric(self, *, name: str, value: float, step: int) -> None:
        self.metrics.append((name, value, step))

    def checkpoint(self, *, path: str, step: int) -> None:
        self.checkpoints.append((path, step))

    def sample(self, *, path: str, step: int) -> None:
        self.samples.append((path, step))

    def completed(self, *, step: int, final_checkpoint: str | None) -> None:
        self.completed_calls.append((step, final_checkpoint))

    def failed(self, *, step: int, error: str) -> None:
        self.failed_calls.append((step, error))

    def is_canceled(self) -> bool:
        return self.canceled


@dataclass
class FakeTrainer:
    fail_on_step: int | None = None
    invalid_on_step: int | None = None
    skip_cadence_on_step: int | None = None
    step_calls: int = 0

    def setup(self, ctx: StepContext) -> None:
        _ = ctx

    def configure(self, ctx: StepContext) -> dict[str, Any]:
        return {"configured": True, "job": ctx.job.run_id}

    def prepare_batch(self, raw_batch: Any, state: Any, ctx: StepContext) -> Any:
        _ = state
        _ = ctx
        return raw_batch

    def train_step(self, batch: Any, state: Any, ctx: StepContext) -> StepResult:
        _ = state
        _ = ctx
        self.step_calls += 1
        if self.fail_on_step == self.step_calls:
            raise RuntimeError(f"boom at step {self.step_calls}")
        if self.invalid_on_step == self.step_calls:
            return {"not": "stepresult"}  # type: ignore[return-value]
        if self.skip_cadence_on_step == self.step_calls:
            return StepResult(
                metrics={"train/loss": float(batch)},
                control=StepControlHints(skip_cadence_emit=True),
            )
        return StepResult(metrics={"train/loss": float(batch)})

    def state_dict(self, state: Any) -> dict[str, object]:
        if isinstance(state, dict):
            return {str(k): v for (k, v) in state.items()}
        return {"state": str(state)}

    def load_state_dict(self, state: Any, payload: dict[str, object], ctx: StepContext) -> None:
        _ = ctx
        if isinstance(state, dict):
            state.update(payload)


@dataclass
class FakeArtifactWriter:
    checkpoints: list[tuple[int, str]] = field(default_factory=list)
    samples: list[tuple[int, list[str]]] = field(default_factory=list)
    finalized: list[str | None] = field(default_factory=list)

    def write_checkpoint(
        self,
        *,
        step: int,
        state_payload: dict[str, object],
        trainer: object,
        state: Any,
        ctx: StepContext,
    ) -> str:
        _ = state_payload
        _ = trainer
        _ = state
        _ = ctx
        path = f"/tmp/ckpt-{step}.safetensors"
        self.checkpoints.append((step, path))
        return path

    def write_samples(self, *, step: int, state: Any, ctx: StepContext) -> list[str]:
        _ = state
        _ = ctx
        paths = [f"/tmp/sample-{step}.png"]
        self.samples.append((step, paths))
        return paths

    def finalize(
        self,
        *,
        state_payload: dict[str, object],
        trainer: object,
        state: Any,
        ctx: StepContext,
    ) -> str | None:
        _ = state_payload
        _ = trainer
        _ = state
        _ = ctx
        path = "/tmp/final.safetensors"
        self.finalized.append(path)
        return path


@dataclass
class FakeUploader:
    metrics: list[tuple[int, dict[str, float]]] = field(default_factory=list)
    checkpoints: list[tuple[int, str]] = field(default_factory=list)
    samples: list[tuple[int, str]] = field(default_factory=list)

    def upload_checkpoint(self, *, local_path: str, step: int) -> dict[str, Any]:
        self.checkpoints.append((step, local_path))
        return {"ok": True}

    def upload_sample(self, *, local_path: str, step: int) -> dict[str, Any]:
        self.samples.append((step, local_path))
        return {"ok": True}

    def upload_metrics(self, *, metrics: dict[str, float], step: int) -> dict[str, Any]:
        self.metrics.append((step, metrics))
        return {"ok": True}


def _job(**kwargs: Any) -> TrainingJobSpec:
    params = {
        "run_id": "run_159",
        "max_steps": 5,
        "metric_every": 2,
        "checkpoint_every": 3,
        "sample_every": 4,
    }
    params.update(kwargs)
    return TrainingJobSpec(**params)


def _ctx(job: TrainingJobSpec) -> StepContext:
    return StepContext(job=job, device="cuda:0", dtype="bf16")


def test_runtime_loop_cadence_and_uploads() -> None:
    job = _job()
    reporter = FakeReporter()
    trainer = FakeTrainer()
    writer = FakeArtifactWriter()
    uploader = FakeUploader()

    terminal_step = run_training_loop(
        job=job,
        ctx=_ctx(job),
        trainer=trainer,
        batches=[1, 2, 3, 4, 5, 6, 7],
        reporter=reporter,
        artifact_writer=writer,
        uploader=uploader,
    )

    assert terminal_step == 5
    assert trainer.step_calls == 5
    assert reporter.started_runs == ["run_159"]
    assert reporter.metrics == [("train/loss", 2.0, 2), ("train/loss", 4.0, 4)]
    assert reporter.checkpoints == [("/tmp/ckpt-3.safetensors", 3)]
    assert reporter.samples == [("/tmp/sample-4.png", 4)]
    assert reporter.completed_calls == [(5, "/tmp/final.safetensors")]
    assert reporter.failed_calls == []
    assert uploader.metrics == [(2, {"train/loss": 2.0}), (4, {"train/loss": 4.0})]
    assert uploader.checkpoints == [(3, "/tmp/ckpt-3.safetensors")]
    assert uploader.samples == [(4, "/tmp/sample-4.png")]


def test_runtime_loop_cancel_reports_failed_and_raises() -> None:
    job = _job(metric_every=1, checkpoint_every=0, sample_every=0)
    reporter = FakeReporter(canceled=True)
    trainer = FakeTrainer()
    writer = FakeArtifactWriter()

    with pytest.raises(TrainingCanceled, match="canceled"):
        run_training_loop(
            job=job,
            ctx=_ctx(job),
            trainer=trainer,
            batches=[1, 2, 3],
            reporter=reporter,
            artifact_writer=writer,
        )

    assert trainer.step_calls == 0
    assert reporter.completed_calls == []
    assert reporter.failed_calls == [(0, "canceled")]
    assert writer.finalized == []


def test_runtime_loop_step_error_reports_failed_and_re_raises() -> None:
    job = _job(metric_every=1, checkpoint_every=0, sample_every=0, max_steps=4)
    reporter = FakeReporter()
    trainer = FakeTrainer(fail_on_step=2)
    writer = FakeArtifactWriter()

    with pytest.raises(RuntimeError, match="boom at step 2"):
        run_training_loop(
            job=job,
            ctx=_ctx(job),
            trainer=trainer,
            batches=[1, 2, 3, 4],
            reporter=reporter,
            artifact_writer=writer,
        )

    assert trainer.step_calls == 2
    assert reporter.metrics == [("train/loss", 1.0, 1)]
    assert reporter.completed_calls == []
    assert reporter.failed_calls == [(2, "boom at step 2")]
    assert writer.finalized == []


def test_runtime_loop_invalid_result_reports_type_error() -> None:
    job = _job(metric_every=1, checkpoint_every=0, sample_every=0, max_steps=2)
    reporter = FakeReporter()
    trainer = FakeTrainer(invalid_on_step=1)
    writer = FakeArtifactWriter()

    with pytest.raises(TypeError, match="StepResult-like"):
        run_training_loop(
            job=job,
            ctx=_ctx(job),
            trainer=trainer,
            batches=[1, 2],
            reporter=reporter,
            artifact_writer=writer,
        )

    assert reporter.completed_calls == []
    assert reporter.failed_calls == [(1, "trainer.train_step must return StepResult-like object with metrics/debug fields")]


def test_runtime_loop_step_control_can_skip_cadence() -> None:
    job = _job(metric_every=1, checkpoint_every=1, sample_every=1, max_steps=3)
    reporter = FakeReporter()
    trainer = FakeTrainer(skip_cadence_on_step=2)
    writer = FakeArtifactWriter()

    terminal_step = run_training_loop(
        job=job,
        ctx=_ctx(job),
        trainer=trainer,
        batches=[1, 2, 3],
        reporter=reporter,
        artifact_writer=writer,
    )

    assert terminal_step == 3
    metric_steps = [step for (_name, _val, step) in reporter.metrics]
    assert metric_steps == [1, 3]


def test_runtime_loop_resume_start_step_continues_counters() -> None:
    job = _job(metric_every=1, checkpoint_every=2, sample_every=0, max_steps=5)
    reporter = FakeReporter()
    trainer = FakeTrainer()
    writer = FakeArtifactWriter()

    terminal_step = run_training_loop(
        job=job,
        ctx=_ctx(job),
        trainer=trainer,
        batches=[10, 11, 12, 13, 14],
        reporter=reporter,
        artifact_writer=writer,
        start_step=3,
    )

    assert terminal_step == 5
    assert trainer.step_calls == 2
    assert reporter.metrics[-1] == ("train/loss", 11.0, 5)
    assert reporter.checkpoints == [("/tmp/ckpt-4.safetensors", 4)]


def test_runtime_loop_normalizes_v1_metric_names_and_infers_lr() -> None:
    job = _job(metric_every=1, checkpoint_every=0, sample_every=0, max_steps=1)
    reporter = FakeReporter()
    writer = FakeArtifactWriter()

    class _Trainer:
        def setup(self, ctx: StepContext) -> None:
            _ = ctx

        def configure(self, ctx: StepContext) -> dict[str, Any]:
            return {"learning_rate": 0.001}

        def prepare_batch(self, raw_batch: Any, state: Any, ctx: StepContext) -> Any:
            _ = state
            _ = ctx
            return raw_batch

        def train_step(self, batch: Any, state: Any, ctx: StepContext) -> StepResult:
            _ = state
            _ = ctx
            return StepResult(metrics={"loss": 2.5})

        def state_dict(self, state: Any) -> dict[str, object]:
            return dict(state)

        def load_state_dict(self, state: Any, payload: dict[str, object], ctx: StepContext) -> None:
            _ = ctx
            if isinstance(state, dict):
                state.update(payload)

    terminal_step = run_training_loop(
        job=job,
        ctx=_ctx(job),
        trainer=_Trainer(),
        batches=[1],
        reporter=reporter,
        artifact_writer=writer,
    )

    assert terminal_step == 1
    assert reporter.metrics == [("train/loss", 2.5, 1), ("train/lr", 0.001, 1)]


def test_runtime_loop_restores_checkpoint_when_supported() -> None:
    job = _job(metric_every=1, checkpoint_every=0, sample_every=0, max_steps=2)
    reporter = FakeReporter()
    writer = FakeArtifactWriter()
    restored: list[dict[str, object]] = []

    class _Trainer:
        def setup(self, ctx: StepContext) -> None:
            _ = ctx
            return None

        def configure(self, ctx: StepContext) -> dict[str, Any]:
            _ = ctx
            return {"restored": False}

        def prepare_batch(self, raw_batch: Any, state: Any, ctx: StepContext) -> Any:
            _ = state
            _ = ctx
            return raw_batch

        def train_step(self, batch: Any, state: Any, ctx: StepContext) -> StepResult:
            _ = state
            _ = ctx
            return StepResult(metrics={"train/loss": float(batch)})

        def state_dict(self, state: Any) -> dict[str, object]:
            return dict(state)

        def load_state_dict(self, state: Any, payload: dict[str, object], ctx: StepContext) -> None:
            _ = ctx
            state.update(payload)
            restored.append(dict(payload))

    terminal_step = run_training_loop(
        job=job,
        ctx=_ctx(job),
        trainer=_Trainer(),
        batches=[9, 10],
        reporter=reporter,
        artifact_writer=writer,
        start_step=1,
        resume_state_payload={"restored": True, "step": 1},
    )

    assert terminal_step == 2
    assert restored == [{"restored": True, "step": 1}]
    assert reporter.metrics == [("train/loss", 9.0, 2)]
