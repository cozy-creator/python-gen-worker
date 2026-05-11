# training-smoke

Minimal trainer class showing the `gen_worker.trainer` contract. Reports a fake loss each step using whatever number the runtime hands it; serves as a smoke test for the trainer runtime + cadence emission + resume path.

## What it demonstrates
- The required trainer hooks: `setup`, `configure`, `prepare_batch`, `train_step`, `state_dict`, `load_state_dict`.
- Reading hyperparams from the job spec (`ctx.job.hyperparams`).
- Emitting metrics that get normalized: `train/loss`, `train/lr`.
- `StepControlHints` for tuning cadence (this trainer always emits).

## When to copy it
- Bootstrapping a new training endpoint — replace the no-op body with your forward/backward/optimizer loop.
- Local trainer smoke tests with `mock_batches` in the job spec — verify the cadence/checkpoint/resume plumbing works before you wire up real data.

## Running locally
```bash
WORKER_MODE=trainer \
TRAINER_JOB_SPEC_PATH=./trainer_job.example.json \
uv run python -m gen_worker.entrypoint
```
Confirm: metrics appear in `events.jsonl`, checkpoints land under `${TRAINER_ARTIFACTS_DIR}/checkpoints/`, and re-running with `resume_from_latest: true` continues step counters.

## Files
- `main.py` — the trainer class (~40 lines).
- `trainer_job.example.json` — runnable job spec for local smoke runs.
