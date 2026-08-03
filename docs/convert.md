# Model conversion + transfer — measured floors and rulings

## Memory floors per operation (gw#395/#396)

Model size must never dictate converter hardware. What each operation actually needs:

| Operation | Peak anonymous RAM | Why |
|---|---|---|
| dtype cast (`streaming_dtype_cast` / `streaming_cast_snapshot`) | ≈ largest single tensor | two-pass streaming: header-only shard plan, then one tensor at a time |
| fp8-E4M3 storage flavor (`streaming_fp8_storage_cast` / `streaming_fp8_snapshot`) | ≈ largest single tensor | same engine; clamp ±448 + layerwise-cast skip patterns |
| byte-offset reshard / shard merge (`shard_safetensors_by_offset`) | O(1) (8 MB copy chunks) | raw byte-range copy, no tensor decode |
| bnb nf4/fp4 | full component | `from_pretrained` load, inherent to bnb |
| singlefile↔diffusers repackage | **full model** | `from_single_file` / whole-keyspace remap needs the full tensor set — the one legitimate big-RAM operation |
| GGUF | full model (llama.cpp converter) | external toolchain |

Casts and fp8 flavor production run on the standard 32 GB CPU class regardless of model size;
only repackage/GGUF of huge models still needs RAM sized to the model.

## Transfer rulings

- **Model uploads are grant-only. There is no presigned-multipart fallback.**
  A repo/model upload requires a Tensorhub-issued scoped `transfer_grant`
  (short-lived S3 credentials scoped per object). Non-model platform uploads
  (media, datasets) may still take presigned multipart URLs.
- The transfer fanouts — **file fanout 4, part fanout 4, process-wide
  presigned PUT budget 8** — are **implementation constants, not knobs**.
  They are chosen from `scripts/benchmark_model_transfer.py`, scored
  reliability first, then wall-clock, then RSS, then retries. Re-benchmark
  before changing them; do not expose them as configuration.
