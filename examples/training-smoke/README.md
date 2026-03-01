# training-smoke

Minimal demo of trainer mode using `gen_worker.trainer` class-only API.

## Files

- `main.py`: trainer class (`TrainingSmokeTrainer`) implementing class-only hooks.
- `trainer_job.example.json`: example trainer manifest for local runs.

## Required Env

- `WORKER_MODE=trainer`
- `TRAINER_JOB_SPEC_PATH` (path to trainer job JSON)
- `TRAINER_PLUGIN` (optional if `trainer` is set inside job spec)
- `TRAINER_CHECKPOINTS_DIR` (optional; default `/tmp/training/checkpoints`)
- `TRAINER_SAMPLES_DIR` (optional; default `/tmp/training/samples`)
- `TRAINER_EVENTS_PATH` (optional; writes line-delimited JSON events)

## Local Run

```bash
cd examples/training-smoke
cp trainer_job.example.json /tmp/trainer_job.json
PYTHONPATH="$(pwd)" \
WORKER_MODE=trainer \
TRAINER_JOB_SPEC_PATH=/tmp/trainer_job.json \
TRAINER_CHECKPOINTS_DIR=/tmp/training/checkpoints \
TRAINER_SAMPLES_DIR=/tmp/training/samples \
TRAINER_EVENTS_PATH=/tmp/training/events.jsonl \
python -m gen_worker.entrypoint
```

## Event Stream

`TRAINER_EVENTS_PATH` receives events emitted by the runtime:

- `started`
- `metric`
- `checkpoint`
- `sample`
- `completed`
- `failed`

Example lines:

```json
{"event":"started","run_id":"training-smoke-001"}
{"event":"metric","name":"train/loss","step":1,"value":1.0}
{"event":"checkpoint","path":"/tmp/training/checkpoints/step-00000002.json","step":2}
{"event":"completed","final_checkpoint":"/tmp/training/checkpoints/final.json","step":5}
```

The runtime writes local artifacts under checkpoint/sample dirs and can be wired
to uploaders in production integration.
