# Issue 159: Trainer Runtime Contract (SDK-Native)

## Primary Decision

- The training integration contract is SDK-native Python imports under
  `gen_worker.trainer`.
- Primary path does not use stdout JSONL parsing between runtime and trainer
  plugin code.

## Typed Contract Objects

- `TrainingJobSpec`: immutable run identity + cadence policy + `trainer_api_version`.
- `StepContext`: runtime-owned context passed to trainer class hooks.
- `StepResult`: typed per-step output for scalar metrics/debug values.
- `TrainingReporter`: runtime reporter interface:
  `started`, `metric`, `checkpoint`, `sample`, `completed`, `failed`, `is_canceled`.

## Runtime Module Map (`gen_worker.trainer`)

- `job_lifecycle.py`: claim/heartbeat/cancel/retry/terminal lifecycle protocol.
- `authz.py`: auth/token scope context.
- `resolve.py`: refs/tags/digest resolution protocol.
- `inputs.py`: base weights + parquet dataset + resume checkpoint download protocol.
- `arrow_feed.py`: Arrow batch feed protocol/config.
- `model_runtime.py`: model-component loading + dtype/precision policy.
- `loop.py`: canonical runtime-owned training loop.
- `artifacts.py`: local checkpoint/sample writing protocol.
- `uploader.py`: artifact upload protocol.
- `reporting.py`: typed reporter events.

## Canonical Runtime Loop

1. Call trainer `setup(ctx)` then `configure(ctx) -> state`.
2. Emit `started(run_id=...)`.
3. Iterate batches while `completed_steps < max_steps`.
4. At each step: cancel check -> `prepare_batch(...)` -> `train_step(...)` -> validate `StepResult`.
5. Runtime cadence:
   - `metric_every`: emit metrics (+ optional metrics upload)
   - `checkpoint_every`: serialize `state_dict(state)`, write checkpoint locally, emit event, upload
   - `sample_every`: write sample(s) locally, emit event(s), upload
6. On success: serialize `state_dict(state)`, runtime `finalize(...)`, then `completed(step, final_checkpoint)`.
7. On any exception/cancel: `failed(step, error)` and re-raise root cause.

## Ownership Split

- Runtime side (`gen-worker`) owns:
  lifecycle, authz, ref resolution, input downloads, parquet->Arrow feeding,
  outer-loop cadence, model loading + dtype/precision policy, local checkpoint/sample
  writing, artifact uploads, terminal reporting.
- Endpoint trainer class owns:
  batch shaping, training semantics, and state serialization (`state_dict/load_state_dict`).
- Endpoint trainer class does not own:
  control-plane operations, network calls, artifact file ownership, upload behavior.
