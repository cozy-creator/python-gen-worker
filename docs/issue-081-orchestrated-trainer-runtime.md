# Issue 81: Orchestrated Trainer Runtime (Tenant-Built Images)

This document defines the trainer-mode runtime contract for orchestrated jobs.

## Startup Contract

Required:

- `TRAINER_JOB_SPEC_PATH` (JSON job spec path)
- trainer import path via:
  - `TRAINER_PLUGIN`, or
  - `trainer` field in job spec

Artifact paths (default from `TRAINER_ARTIFACTS_DIR`):

- `TRAINER_CHECKPOINTS_DIR`
- `TRAINER_SAMPLES_DIR`
- `TRAINER_METRICS_DIR`
- `TRAINER_EVENTS_PATH`

Orchestrated mode:

- `TRAINER_ORCHESTRATED=1` enables strict startup checks
- capability token is required via:
  - `TRAINER_CAPABILITY_TOKEN`, or
  - `capability_token` in job spec

Deterministic startup errors:

- `startup.missing_job_spec_path`
- `startup.missing_trainer_import`
- `startup.invalid_artifact_paths`
- `startup.invalid_timeout`
- `startup.missing_capability_token`

## Input Materialization

Job-spec `inputs` block supports:

- `base_model_ref` / `base_model_url`
- `dataset_parquet_refs` / `dataset_parquet_urls`
- `resume_checkpoint_ref` / `resume_checkpoint_url`

Behavior:

- model refs are materialized with `ModelRefDownloader` (cozy/hf/local/url)
- dataset parquet refs are materialized to local parquet paths
- resume checkpoint ref is downloaded and applied through
  `trainer.load_state_dict(...)`

## Event Contract (v1)

`TRAINER_EVENTS_PATH` emits JSONL with stable shape:

- `schema_version`: `trainer_event.v1`
- `event`: `started|metric|checkpoint|sample|completed|failed`
- `run_id`
- `seq`
- `timestamp_ms`

Event payload fields remain event-specific (`name/value/step/path/error/...`).

## Artifact Upload Contract

Optional upload endpoints:

- `TRAINER_UPLOAD_METRICS_URL`
- `TRAINER_UPLOAD_CHECKPOINT_URL`
- `TRAINER_UPLOAD_SAMPLE_URL`
- `TRAINER_UPLOAD_TERMINAL_URL`

If configured, runtime uploads:

- periodic metrics
- periodic checkpoints
- periodic samples
- final checkpoint + terminal status payload

Upload auth:

- bearer token from `TRAINER_CAPABILITY_TOKEN` when provided

## Cancellation + Timeout

Cancellation checks run on each loop iteration:

- env cancel flag: `TRAINER_CANCELLED`
- optional cancel file: `TRAINER_CANCEL_FILE`
- max runtime: `TRAINER_MAX_RUNTIME_SECONDS`

Timeout is surfaced as deterministic cancel reason: `timeout`.

## Resume + Idempotency

- Resume source: latest valid checkpoint or explicit resume checkpoint path.
- Runtime applies serialized `state` payload with `load_state_dict(...)`.
- If `resume_from_latest=true` and a valid `final.json` already exists, runtime
  short-circuits as completed to avoid duplicate terminal publication.

## Failure Categories

Runtime maps failures to operator-facing categories:

- `startup`
- `input`
- `auth`
- `model-load`
- `train-step`
- `upload`

Safe failure text is emitted in `failed` events; debug detail stays in logs.
