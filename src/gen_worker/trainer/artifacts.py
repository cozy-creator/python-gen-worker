from __future__ import annotations

from typing import Any, Protocol

from .contracts import StepContext, TrainerPlugin


class ArtifactWriter(Protocol):
    """Runtime-owned local artifact writing contract."""

    def write_checkpoint(
        self,
        *,
        step: int,
        state_payload: dict[str, object],
        trainer: TrainerPlugin,
        state: Any,
        ctx: StepContext,
    ) -> str:
        ...

    def write_samples(self, *, step: int, state: Any, ctx: StepContext) -> list[str]:
        ...

    def finalize(
        self,
        *,
        state_payload: dict[str, object],
        trainer: TrainerPlugin,
        state: Any,
        ctx: StepContext,
    ) -> str | None:
        ...


__all__ = ["ArtifactWriter"]
