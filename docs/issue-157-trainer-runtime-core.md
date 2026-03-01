# Issue 157: Trainer Runtime Core

This document records the runtime contract implemented in `python-gen-worker`
for trainer mode.

## Decisions

- Single-library architecture: `python-gen-worker` contains both inference and
  trainer runtimes.
- Runtime mode selection is explicit: `WORKER_MODE=inference|trainer`.
- Trainer authoring path is class-only:
  trainer class/instance with canonical hooks.
- Contract integration is SDK-native Python imports (`gen_worker.trainer`);
  primary path is not stdout text protocol parsing.
- Dataset ingestion path is parquet-first (Arrow batches via PyArrow scanner).
- Orchestrated training runtime uses `gen-worker` + `gen-trainer` plugins;
  no UI server is part of trainer runtime.

## Runtime Module Map

- `gen_worker.trainer.job_lifecycle`
- `gen_worker.trainer.authz`
- `gen_worker.trainer.resolve`
- `gen_worker.trainer.inputs`
- `gen_worker.trainer.arrow_feed`
- `gen_worker.trainer.model_runtime`
- `gen_worker.trainer.loop`
- `gen_worker.trainer.artifacts`
- `gen_worker.trainer.uploader`
- `gen_worker.trainer.reporting`
- `gen_worker.trainer.runtime` (entrypoint/orchestration glue)
- `gen_worker.trainer.api` (class-only plugin loader)

## Entry Contract (Trainer Mode)

`WORKER_MODE=trainer` routes `python -m gen_worker.entrypoint` to
`gen_worker.trainer.runtime.run_training_runtime_from_env()`.

Required runtime inputs:

- `TRAINER_JOB_SPEC_PATH` (JSON manifest, default `/app/.cozy/trainer_job.json`)
- `TRAINER_PLUGIN` (`module:symbol`) or `trainer` field in job JSON

Optional:

- `TRAINER_CHECKPOINTS_DIR`
- `TRAINER_SAMPLES_DIR`
- `TRAINER_EVENTS_PATH`

### Trainer Job JSON (v1)

```json
{
  "trainer_api_version": "v1",
  "run_id": "run_123",
  "trainer": "my_pkg.train:MyTrainer",
  "max_steps": 1000,
  "metric_every": 10,
  "checkpoint_every": 200,
  "sample_every": 200,
  "owner": "org_id",
  "release_ref": "org/repo:latest",
  "hyperparams": {"learning_rate": 0.0001},
  "dataset": {
    "parquet_paths": ["/data/train.parquet"],
    "batch_size": 32,
    "readahead": 2,
    "columns": ["image_ref", "caption"]
  }
}
```

`mock_batches` is supported for local smoke runs.

## Versioned Subprocess Contract

For process-boundary execution (runtime -> trainer subprocess), `gen-worker`
defines `TrainerSubprocessContractV1` in `gen_worker.trainer.subprocess_contract`.

Contract fields include:

- `contract_version` (`v1`)
- `trainer_plugin`
- `job_spec_path`
- `arrow_ipc_path` (deterministic Arrow IPC handoff path)
- `checkpoints_dir`
- `samples_dir`
- `events_path`

Compatibility rule:

- Any incompatible boundary shape change requires a new `contract_version`.
- `v1` readers reject unknown versions by default.

## Plugin Contract

- `setup(ctx) -> None`
- `configure(ctx) -> state`
- `prepare_batch(raw_batch, state, ctx) -> prepared_batch`
- `train_step(prepared_batch, state, ctx) -> StepResult`
- `state_dict(state) -> dict[str, object]`
- `load_state_dict(state, payload, ctx) -> None`

`StepResult` can emit:

- `metrics` (scalars)
- `debug` (structured optional)
- `control` (`skip_cadence_emit`, `backoff_seconds`)

## Ownership Split

`gen-worker` runtime owns:

- lifecycle/reporting cadence
- cancellation checks
- parquet->Arrow batch feeding
- checkpoint/sample local writes
- artifact upload bridge points
- terminal completed/failed state emission

trainer class owns only:

- dataset/batch shaping
- step math (forward/backward/update)
- trainer state serialization logic
