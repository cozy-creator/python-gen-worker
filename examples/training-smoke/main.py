from __future__ import annotations

from typing import Any

from gen_worker.trainer import StepContext, StepControlHints, StepResult


class TrainingSmokeTrainer:
    def setup(self, ctx: StepContext) -> None:
        _ = ctx

    def configure(self, ctx: StepContext) -> dict[str, object]:
        _ = ctx
        return {"step": 0}

    def prepare_batch(self, raw_batch: Any, state: dict[str, object], ctx: StepContext) -> Any:
        _ = state
        _ = ctx
        return raw_batch

    def train_step(self, batch: Any, state: dict[str, object], ctx: StepContext) -> StepResult:
        _ = state
        loss = float(batch) if isinstance(batch, (int, float)) else 0.0
        control = StepControlHints(skip_cadence_emit=False)
        return StepResult(
            metrics={
                "train/loss": loss,
                "train/lr": float(ctx.job.hyperparams.get("learning_rate", 0.0)),
            },
            control=control,
        )

    def state_dict(self, state: dict[str, object]) -> dict[str, object]:
        return dict(state)

    def load_state_dict(self, state: dict[str, object], payload: dict[str, object], ctx: StepContext) -> None:
        _ = ctx
        state.update(payload)
