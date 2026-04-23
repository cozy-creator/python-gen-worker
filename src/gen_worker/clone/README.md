# `gen_worker.clone` — clone model checkpoints from external upstreams

Platform capability for ingesting HuggingFace / Civitai model repos as
tensorhub checkpoints. Tenant endpoints (`@inference_function` /
`@training_function`) call into this module; library owns all
tensorhub-upload machinery.

## Six-method public surface

| Method | Purpose |
|--------|---------|
| `clone.from_huggingface(ctx, payload)` | Full HF clone pipeline (download + convert + upload + lineage). Returns `list[CheckpointRef]`. |
| `clone.from_civitai(ctx, payload)` | Same for Civitai. |
| `clone.fetch_huggingface_snapshot(source_ref, revision, dest_dir, …)` | Download only. Returns local snapshot path. |
| `clone.fetch_civitai_file(model_version_id, file_id, dest_dir)` | Download one Civitai file. |
| `clone.parse_huggingface_metadata(source_ref, revision)` | Parse repo metadata without downloading weights. |
| `clone.parse_civitai_metadata(model_version_id)` | Same for Civitai. |

Two stability tiers:

- **High-level** (`from_huggingface` / `from_civitai`): stable. Signature
  changes = major version bump. **99% of tenants use these.**
- **Intermediate** (4 remaining): stable, narrower use-cases. Changes
  flagged in release notes. Currently stubs pending refactor; use
  `from_huggingface` / `from_civitai` for the full flow.

## Three worked examples

### 1. Simplest one-liner tenant

```python
from gen_worker import RequestContext, clone, inference_function
from .shared import CloneHuggingFaceInput, ConversionOutput

@inference_function(concurrency="concurrent")
def clone_huggingface(ctx: RequestContext, payload: CloneHuggingFaceInput) -> ConversionOutput:
    return clone.from_huggingface(ctx, payload)
```

Tenant delegates entirely. Library opens session, downloads, converts,
uploads, emits lineage. Five lines of endpoint code.

### 2. Custom-metadata tenant

Uses `parse_huggingface_metadata` to inspect the upstream before
deciding how to route.

```python
@inference_function(concurrency="concurrent")
def clone_huggingface(ctx: RequestContext, payload: CloneHuggingFaceInput) -> ConversionOutput:
    meta = clone.parse_huggingface_metadata(payload.huggingface_repo, payload.source_revision)
    # Custom: if the model is a LoRA, set kind='lora' and skip conversion.
    if (meta.get("base_model") or "").strip():
        payload.outputs = [{"dtype": "bf16", "file_layout": "singlefile", "file_type": "safetensors"}]
    return clone.from_huggingface(ctx, payload)
```

### 3. Custom-transform tenant

Uses `fetch_huggingface_snapshot` to get local files, runs a custom
transform, then commits via `ctx.save_file`/`ctx.save_checkpoint` —
still zero lines touching upload contract.

```python
@training_function(kind='fine-tune')
def train_and_commit(ctx: TrainingContext, source: Source, payload: MyInput) -> list[ProducedVariant]:
    snapshot_path = clone.fetch_huggingface_snapshot(
        source_ref=payload.base_model,
        revision=payload.base_revision,
        dest_dir="/tmp/base",
    )
    # ... do custom training that produces new weights in /tmp/output/ ...
    return [ProducedVariant(
        dtype="fp16", file_layout="diffusers", file_type="safetensors",
        files={"unet.safetensors": "/tmp/output/unet.safetensors"},
        kind="model",
    )]
```

Library's `dispatch._finalize_produced_variants` handles session +
upload + finalize on the returned `ProducedVariant` list.

## Architectural principle

Tenant code — regardless of decorator — should **never touch
tensorhub's upload contract** (sessions, presigns, finalize, publish,
blob digests, manifests). All of that is library-internal.

The public surface on `ctx` for tenants is:

- `ctx.save_file` / `save_file_create` / `save_bytes`
- `ctx.save_checkpoint` / `save_checkpoint_bytes` / `open_output_stream`
- `ctx.publish_checkpoint(repo, id)` / `ctx.unpublish_checkpoint(repo, id)`

Plus `gen_worker.clone.*` for upstream ingest.

Everything else — `_UploadSessionManager`, `presigned_upload_file`,
`_open_upload_session`, `publish_repo_revision`, internal pipeline
helpers — is library-private. Tenant code importing any of those gets
flagged in code review as a layering violation.

## Library-internal files

- `pipeline.py` — the 2320-line orchestrator. Exposed via `run_clone()`.
- `_shared.py` — platform helpers (ConversionOutput / IngestResult / tensors_with).
- `_flashpack.py` — flashpack format conversion.

These are NOT part of the public API. Refactor freely.
