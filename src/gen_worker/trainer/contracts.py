from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol, TypeVar


@dataclass(frozen=True)
class TrainingJobSpec:
    """Immutable training job inputs owned by the runtime."""

    run_id: str
    max_steps: int
    trainer_api_version: str = "v1"
    metric_every: int = 10
    checkpoint_every: int = 0
    sample_every: int = 0
    owner: str = ""
    release_ref: str = ""
    hyperparams: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")
        if self.max_steps < 0:
            raise ValueError("max_steps must be >= 0")
        if self.metric_every < 0:
            raise ValueError("metric_every must be >= 0")
        if self.checkpoint_every < 0:
            raise ValueError("checkpoint_every must be >= 0")
        if self.sample_every < 0:
            raise ValueError("sample_every must be >= 0")
        if not self.trainer_api_version:
            raise ValueError("trainer_api_version is required")


@dataclass(frozen=True)
class StepContext:
    """Per-run context passed to trainer plugins.

    The runtime is responsible for lifecycle, IO, and artifact handling.
    Plugins use this context for optimizer/scheduler and step math only.
    """

    job: TrainingJobSpec
    model_handles: Mapping[str, Any] = field(default_factory=dict)
    dataset: Any | None = None
    optimizer: Any | None = None
    scheduler: Any | None = None
    device: str = "cuda:0"
    dtype: str = "bf16"
    is_canceled: Callable[[], bool] | None = None


@dataclass(frozen=True)
class StepControlHints:
    """Optional runtime hints emitted by a step plugin."""

    skip_cadence_emit: bool = False
    backoff_seconds: float | None = None


@dataclass(frozen=True)
class StepResult:
    """Per-step structured output returned by trainer plugins."""

    metrics: Mapping[str, float] = field(default_factory=dict)
    debug: Mapping[str, Any] = field(default_factory=dict)
    control: StepControlHints | None = None


StateT = TypeVar("StateT")
RawBatchT = TypeVar("RawBatchT")
PreparedBatchT = TypeVar("PreparedBatchT")


class TrainerEndpointContract(Protocol[StateT, RawBatchT, PreparedBatchT]):
    """Canonical class-only trainer contract for endpoint implementations."""

    def setup(self, ctx: StepContext) -> None:
        ...

    def configure(self, ctx: StepContext) -> StateT:
        ...

    def prepare_batch(self, raw_batch: RawBatchT, state: StateT, ctx: StepContext) -> PreparedBatchT:
        ...

    def train_step(self, batch: PreparedBatchT, state: StateT, ctx: StepContext) -> StepResult:
        ...

    def state_dict(self, state: StateT) -> dict[str, object]:
        ...

    def load_state_dict(self, state: StateT, payload: dict[str, object], ctx: StepContext) -> None:
        ...

    def save_checkpoint(
        self,
        *,
        state: StateT,
        step: int,
        output_dir: str,
        final: bool,
        ctx: StepContext,
    ) -> Mapping[str, object] | None:
        ...

    def load_checkpoint(
        self,
        *,
        state: StateT,
        checkpoint_dir: str,
        payload: Mapping[str, object],
        ctx: StepContext,
    ) -> None:
        ...


class TrainingReporter(Protocol):
    """Runtime reporter contract.

    Runtime owns reporter calls. Plugins should not emit lifecycle events directly.
    """

    def started(self, *, run_id: str) -> None:
        ...

    def metric(self, *, name: str, value: float, step: int) -> None:
        ...

    def checkpoint(self, *, path: str, step: int) -> None:
        ...

    def sample(self, *, path: str, step: int) -> None:
        ...

    def completed(self, *, step: int, final_checkpoint: Optional[str]) -> None:
        ...

    def failed(self, *, step: int, error: str) -> None:
        ...

    def is_canceled(self) -> bool:
        ...


class TrainerPlugin(Protocol):
    """Class-only trainer contract for endpoint-owned training semantics."""

    def setup(self, ctx: StepContext) -> None:
        ...

    def configure(self, ctx: StepContext) -> Any:
        ...

    def prepare_batch(self, raw_batch: Any, state: Any, ctx: StepContext) -> Any:
        ...

    def train_step(self, batch: Any, state: Any, ctx: StepContext) -> StepResult:
        ...

    def state_dict(self, state: Any) -> dict[str, object]:
        ...

    def load_state_dict(self, state: Any, payload: dict[str, object], ctx: StepContext) -> None:
        ...


class CheckpointingTrainerPlugin(Protocol):
    """Optional checkpoint IO hooks for real model artifacts.

    If present, the runtime will call these hooks in addition to JSON state_dict
    snapshots so plugins can persist model-weight artifacts (for example LoRA files).
    """

    def save_checkpoint(
        self,
        *,
        state: Any,
        step: int,
        output_dir: str,
        final: bool,
        ctx: StepContext,
    ) -> Mapping[str, Any] | None:
        ...

    def load_checkpoint(
        self,
        *,
        state: Any,
        checkpoint_dir: str,
        payload: Mapping[str, Any],
        ctx: StepContext,
    ) -> None:
        ...


class BatchProvider(Protocol):
    def iter_batches(self) -> Any:
        ...


__all__ = [
    "BatchProvider",
    "CheckpointingTrainerPlugin",
    "PreparedBatchT",
    "RawBatchT",
    "StateT",
    "StepControlHints",
    "StepContext",
    "StepResult",
    "TrainerEndpointContract",
    "TrainerPlugin",
    "TrainingJobSpec",
    "TrainingReporter",
]
