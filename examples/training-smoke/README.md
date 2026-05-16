# training-smoke

Runnable training endpoint skeleton for fine-tuning a text classifier with the `gen_worker.trainer` runtime.

This is intentionally a PyTorch-first example. The core step looks like a normal Hugging Face/PyTorch loop:

```python
output = model(**batch.inputs)
loss = output.loss
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
optimizer.step()
scheduler.step()
```

The worker runtime wraps that loop with TensorHub-owned lifecycle behavior: startup contract validation, cancellation, metric cadence, checkpoint cadence, sample cadence, resume detection, and artifact upload.

## Files

- `main.py` - endpoint-owned trainer hooks.
- `trainer_job.example.json` - runnable local job spec with mock text/label batches.

## Trainer Contract

`TextClassificationFineTuner` implements the class-only trainer contract:

- `setup(ctx)` seeds Python/Torch.
- `configure(ctx)` loads the model/tokenizer and creates optimizer/scheduler state.
- `prepare_batch(raw_batch, state, ctx)` validates `text` and `label` columns and tokenizes them.
- `train_step(batch, state, ctx)` runs the native PyTorch forward/backward/optimizer step.
- `state_dict(...)` and `load_state_dict(...)` persist lightweight resume state.
- `save_checkpoint(...)` and `load_checkpoint(...)` persist trainable model weights and optimizer state.

## Running Locally

```bash
cd examples/training-smoke
WORKER_MODE=trainer \
TRAINER_JOB_SPEC_PATH=./trainer_job.example.json \
uv run --with torch --with transformers python -m gen_worker.entrypoint
```

If your environment already has Torch and Transformers installed, `uv run python -m gen_worker.entrypoint` is enough.

The default job uses `model_name_or_path="hf-internal-testing/tiny-random-BertForSequenceClassification"`, a tiny public Hugging Face text-classification model. The example downloads the real tokenizer and model through Transformers, then performs real forward/backward/optimizer/checkpoint work.

Artifacts land under `/tmp/training`:

- `/tmp/training/metrics/events.jsonl`
- `/tmp/training/checkpoints/`
- `/tmp/training/samples/`

## Using A Hugging Face Model

Install the optional libraries in the endpoint image, then set `hyperparams.model_name_or_path` to any sequence-classification checkpoint:

```json
{
  "hyperparams": {
    "model_name_or_path": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "num_labels": 2,
    "learning_rate": 0.00002,
    "max_length": 256
  }
}
```

The trainer loads:

```python
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
)
```

For LoRA/PEFT, attach the adapter in `configure()` after loading the model and before collecting `trainable_params`. The important rule is the same one used by Hugging Face PEFT: freeze the base model, leave only adapter parameters trainable, and save only the trainable adapter/checkpoint artifacts.

## When To Use An External Training Engine

Use this class-hook pattern when the endpoint owns the PyTorch loop.

For systems like Ostris AI Toolkit, the endpoint should usually be a thin adapter instead: materialize TensorHub inputs to local files, write the toolkit YAML config, run the toolkit command, then expose its produced checkpoints/samples through `save_checkpoint`/sample hooks or a single-run backend object. Do not duplicate an external trainer's inner loop inside `train_step` unless TensorHub needs per-step control.
