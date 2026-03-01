# Custom Trainer Authoring (Class-Only)

This is the canonical authoring contract for training endpoints in `python-gen-worker`.

## Required Trainer Class Hooks

Your trainer import target must resolve to a class (or class instance) with:

- `setup(ctx) -> None`
- `configure(ctx) -> state`
- `prepare_batch(raw_batch, state, ctx) -> prepared_batch`
- `train_step(prepared_batch, state, ctx) -> StepResult`
- `state_dict(state) -> dict[str, object]`
- `load_state_dict(state, payload, ctx) -> None`

Optional checkpoint hooks (recommended for real weight artifacts):

- `save_checkpoint(state, step, output_dir, final, ctx) -> Mapping[str, object] | None`
- `load_checkpoint(state, checkpoint_dir, payload, ctx) -> None`

Minimal skeleton:

```python
from gen_worker.trainer import StepContext, StepResult

class MyTrainer:
    def setup(self, ctx: StepContext) -> None:
        pass

    def configure(self, ctx: StepContext) -> dict[str, object]:
        return {"step": 0}

    def prepare_batch(self, raw_batch: object, state: dict[str, object], ctx: StepContext) -> object:
        return raw_batch

    def train_step(self, prepared_batch: object, state: dict[str, object], ctx: StepContext) -> StepResult:
        return StepResult(metrics={"train/loss": 0.0})

    def state_dict(self, state: dict[str, object]) -> dict[str, object]:
        return dict(state)

    def load_state_dict(self, state: dict[str, object], payload: dict[str, object], ctx: StepContext) -> None:
        state.update(payload)

    def save_checkpoint(
        self,
        *,
        state: dict[str, object],
        step: int,
        output_dir: str,
        final: bool,
        ctx: StepContext,
    ) -> dict[str, object] | None:
        return None

    def load_checkpoint(
        self,
        *,
        state: dict[str, object],
        checkpoint_dir: str,
        payload: dict[str, object],
        ctx: StepContext,
    ) -> None:
        return None
```

## Ownership Boundary

Runtime-owned (`python-gen-worker`):

- lifecycle control and cancellation checks
- cadence for metrics/checkpoints/samples
- artifact writing and upload flow
- terminal reporting (`completed` / `failed`)

Endpoint-owned (trainer class):

- dataset policy and batch shaping
- model forward/backward/update logic
- prompt/mask/curriculum logic
- trainer state serialization semantics
- optional model checkpoint save/load semantics (runtime still owns when uploads happen)

## Optional Low-Level Helpers (`gen_worker.trainer`)

Use these only if they fit your endpoint:

- `seed_everything(seed)` for reproducibility seeding (`random` + optional `torch`).
- `to_float_scalar(value)` to normalize tensor/scalar loss values.
- `build_default_adamw_bundle(model_or_params, hyperparams=...)` for a simple
  AdamW + cosine/warmup scheduler bundle.
- `save_trainable_module_checkpoint(...)` / `load_trainable_module_checkpoint(...)`
  for common LoRA-style module+optimizer checkpoint serialization.

## Local Validation Flow

1. Point trainer import in job spec to your class:
   - `"trainer": "my_pkg.main:MyTrainer"`
2. Run trainer mode:
   - `WORKER_MODE=trainer TRAINER_JOB_SPEC_PATH=/path/to/trainer_job.json python -m gen_worker.entrypoint`
3. Confirm:
   - metrics appear in `events.jsonl`
   - checkpoints include serialized `state` payload
   - resume path restores via `load_state_dict(...)`

## Cozy Creator Deployment Flow

1. Build tenant image from endpoint code + Dockerfile.
2. Submit training job referencing trainer import path.
3. Worker runtime loads trainer class and executes class-only hooks.
4. Runtime reports progress and publishes artifacts/checkpoints.
