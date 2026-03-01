# Issue 162: Trainer Artifacts, Resume, and Quality Signals

This records the v1 artifact and resume behaviors implemented in
`gen_worker.trainer`.

## Deterministic Local Artifact Layout

Runtime now resolves deterministic paths from `TRAINER_ARTIFACTS_DIR`
(default `/tmp/training`):

- checkpoints: `${TRAINER_ARTIFACTS_DIR}/checkpoints`
- samples: `${TRAINER_ARTIFACTS_DIR}/samples`
- metrics/events: `${TRAINER_ARTIFACTS_DIR}/metrics/events.jsonl`

Overrides remain supported:

- `TRAINER_CHECKPOINTS_DIR`
- `TRAINER_SAMPLES_DIR`
- `TRAINER_METRICS_DIR`
- `TRAINER_EVENTS_PATH`

## Checkpoint Writer Contract

- Intermediate checkpoints are written at `checkpoint_every` cadence as
  `step-%08d.json`.
- Final checkpoint marker is written at completion as `final.json`.
- Checkpoint payload includes serialized trainer state:
  `{"step": ..., "run_id": ..., "state": {...}}`.
- Checkpoint JSON writes are atomic (`tempfile + fsync + rename`) where possible.

## Sample Writer Contract

- Runtime accepts `sample_prompts` in job spec:
  - string prompt entries (`t2i` default)
  - object entries with task metadata (`task`, `prompt`, `instruction`,
    `source_image`, `seed`)
- Runtime accepts optional `sample_seed` fixed seed fallback.
- Runtime emits deterministic sample artifact files:
  `step-%08d-%02d.json`.
- `instruct-edit` requests are preserved as structured sample payloads.

## Resume Semantics

- Resume uses `resume_from_latest=true`.
- Runtime scans `step-*.json`, ignoring corrupt/non-JSON files.
- Optional `resume_checkpoint_path` is honored when valid.
- Runtime loads serialized checkpoint `state` payload and applies it through
  trainer `load_state_dict(state, payload, ctx)`.
- Step counters continue from resumed step (`start_step`).

## Minimal v1 Metric Set

Runtime normalizes stable metric names:

- `loss` / `train_loss` -> `train/loss`
- `lr` / `learning_rate` / `train_lr` -> `train/lr`

If `train/lr` is missing, runtime attempts extraction from:

- `StepResult.debug`
- state dict values
- optimizer param groups (`param_groups[0]['lr']`)

Engine-specific diagnostics are emitted as provided.
