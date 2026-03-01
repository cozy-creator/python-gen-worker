from __future__ import annotations

from collections.abc import Iterable, Iterator
import time
from typing import Any, Optional

from .artifacts import ArtifactWriter
from .contracts import StepContext, StepResult, TrainerPlugin, TrainingJobSpec, TrainingReporter
from .uploader import ArtifactUploadError, ArtifactUploader


class TrainingCanceled(RuntimeError):
    pass


def _iter_batches(batch_source: Any) -> Iterator[Any]:
    if hasattr(batch_source, "iter_batches"):
        return iter(batch_source.iter_batches())
    if isinstance(batch_source, Iterable):
        return iter(batch_source)
    raise TypeError("batch source must be iterable or implement iter_batches()")


def _terminal_step(step: int, max_steps: int) -> int:
    if max_steps <= 0:
        return 0
    return min(step, max_steps)


def _normalize_step_result(value: Any) -> StepResult:
    if isinstance(value, StepResult):
        return value
    if hasattr(value, "metrics"):
        metrics = getattr(value, "metrics", {}) or {}
        debug = getattr(value, "debug", {}) or {}
        control = getattr(value, "control", None)
        return StepResult(metrics=metrics, debug=debug, control=control)
    raise TypeError("trainer.train_step must return StepResult-like object with metrics/debug fields")


def _normalize_metric_name(name: str) -> str:
    n = name.strip().lower()
    if n in {"loss", "train_loss"}:
        return "train/loss"
    if n in {"lr", "learning_rate", "train_lr"}:
        return "train/lr"
    return name


def _extract_learning_rate(*, state: Any, ctx: StepContext, debug: dict[str, Any]) -> float | None:
    for key in ("train/lr", "learning_rate", "lr"):
        value = debug.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue

    if isinstance(state, dict):
        for key in ("train/lr", "learning_rate", "lr"):
            value = state.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except Exception:
                continue

    optimizer = ctx.optimizer
    if optimizer is None and isinstance(state, dict):
        optimizer = state.get("optimizer")
    if optimizer is not None:
        param_groups = getattr(optimizer, "param_groups", None)
        if isinstance(param_groups, list) and param_groups:
            group0 = param_groups[0]
            if isinstance(group0, dict) and "lr" in group0:
                try:
                    return float(group0["lr"])
                except Exception:
                    pass
    return None


def _normalized_metrics(*, result: StepResult, state: Any, ctx: StepContext) -> dict[str, float]:
    metrics: dict[str, float] = {}
    debug = dict(result.debug or {})
    for (raw_name, raw_value) in (result.metrics or {}).items():
        metrics[_normalize_metric_name(str(raw_name))] = float(raw_value)
    if "train/lr" not in metrics:
        lr = _extract_learning_rate(state=state, ctx=ctx, debug=debug)
        if lr is not None:
            metrics["train/lr"] = lr
    return metrics


def _serialize_state(*, trainer: TrainerPlugin, state: Any) -> dict[str, object]:
    payload = trainer.state_dict(state)
    if not isinstance(payload, dict):
        raise TypeError("trainer.state_dict(state) must return dict[str, object]")
    return payload


def _maybe_load_checkpoint_hook(
    *,
    trainer: TrainerPlugin,
    state: Any,
    checkpoint_path: str | None,
    payload: dict[str, object] | None,
    ctx: StepContext,
) -> None:
    if not checkpoint_path:
        return
    hook = getattr(trainer, "load_checkpoint", None)
    if not callable(hook):
        return
    checkpoint_dir = str(checkpoint_path)
    try:
        from pathlib import Path

        p = Path(checkpoint_path)
        if p.is_file():
            checkpoint_dir = str(p.parent)
        else:
            checkpoint_dir = str(p)
    except Exception:
        checkpoint_dir = checkpoint_path
    hook(state=state, checkpoint_dir=checkpoint_dir, payload=payload or {}, ctx=ctx)


def run_training_loop(
    *,
    job: TrainingJobSpec,
    ctx: StepContext,
    trainer: TrainerPlugin,
    batches: Any,
    reporter: TrainingReporter,
    artifact_writer: ArtifactWriter,
    uploader: Optional[ArtifactUploader] = None,
    start_step: int = 0,
    resume_state_payload: dict[str, object] | None = None,
    resume_checkpoint_path: str | None = None,
) -> int:
    """Run the runtime-owned training loop.

    Ownership boundaries:
    - Runtime owns lifecycle cadence, cancellation checks, artifact writing/upload,
      and reporter emissions.
    - Trainer class owns batch preparation and per-step training semantics.
    """

    completed_steps = max(0, int(start_step))
    attempted_step = completed_steps
    try:
        trainer.setup(ctx)
        state = trainer.configure(ctx)
        if resume_state_payload is not None:
            trainer.load_state_dict(state, resume_state_payload, ctx)
        _maybe_load_checkpoint_hook(
            trainer=trainer,
            state=state,
            checkpoint_path=resume_checkpoint_path,
            payload=resume_state_payload,
            ctx=ctx,
        )
        reporter.started(run_id=job.run_id)

        for raw_batch in _iter_batches(batches):
            if completed_steps >= job.max_steps:
                break

            if reporter.is_canceled():
                cancel_reason = str(getattr(reporter, "cancel_reason", "") or "canceled")
                raise TrainingCanceled(cancel_reason)

            attempted_step = completed_steps + 1
            prepared_batch = trainer.prepare_batch(raw_batch, state, ctx)
            result = _normalize_step_result(trainer.train_step(prepared_batch, state, ctx))
            completed_steps = attempted_step
            normalized_metrics = _normalized_metrics(result=result, state=state, ctx=ctx)

            skip_cadence = bool(result.control and result.control.skip_cadence_emit)
            if result.control and result.control.backoff_seconds:
                time.sleep(max(0.0, float(result.control.backoff_seconds)))

            if (not skip_cadence) and job.metric_every > 0 and completed_steps % job.metric_every == 0:
                for name, value in normalized_metrics.items():
                    reporter.metric(name=str(name), value=float(value), step=completed_steps)
                if uploader is not None and normalized_metrics:
                    uploader.upload_metrics(metrics=normalized_metrics, step=completed_steps)

            if (not skip_cadence) and job.checkpoint_every > 0 and completed_steps % job.checkpoint_every == 0:
                state_payload = _serialize_state(trainer=trainer, state=state)
                ckpt_path = artifact_writer.write_checkpoint(
                    step=completed_steps,
                    state_payload=state_payload,
                    trainer=trainer,
                    state=state,
                    ctx=ctx,
                )
                reporter.checkpoint(path=ckpt_path, step=completed_steps)
                if uploader is not None:
                    uploader.upload_checkpoint(local_path=ckpt_path, step=completed_steps)

            if (not skip_cadence) and job.sample_every > 0 and completed_steps % job.sample_every == 0:
                sample_paths = artifact_writer.write_samples(step=completed_steps, state=state, ctx=ctx)
                for sample_path in sample_paths:
                    reporter.sample(path=sample_path, step=completed_steps)
                    if uploader is not None:
                        uploader.upload_sample(local_path=sample_path, step=completed_steps)

        final_state_payload = _serialize_state(trainer=trainer, state=state)
        final_checkpoint = artifact_writer.finalize(
            state_payload=final_state_payload,
            trainer=trainer,
            state=state,
            ctx=ctx,
        )
        terminal_step = _terminal_step(completed_steps, job.max_steps)
        reporter.completed(step=terminal_step, final_checkpoint=final_checkpoint)
        return terminal_step
    except Exception as exc:
        failed_step = max(completed_steps, attempted_step)
        terminal_step = _terminal_step(failed_step, job.max_steps)
        if isinstance(exc, ArtifactUploadError):
            error_text = f"upload:{exc}"
        elif isinstance(exc, TrainingCanceled):
            error_text = str(exc)
        else:
            error_text = str(exc)
        try:
            reporter.failed(step=terminal_step, error=error_text)
        except Exception:
            # Preserve root cause if reporting itself fails.
            pass
        raise


__all__ = ["TrainingCanceled", "run_training_loop"]
