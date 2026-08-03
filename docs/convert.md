# gen_worker.convert

Cozy Creator's model ETL: hub ingest (HF + Civitai), dtype cast / quantization, repackage, and Tensorhub publish.

- **Ingest**: HuggingFace (`HfApi.list_repo_files` + classifier + `snapshot_download(allow_patterns=…)`) and Civitai (bounded provider API).
- **Convert**: streaming dtype cast + fp8-E4M3 storage cast (`#fp8` flavor), bitsandbytes nf4/fp4, GGUF (llama.cpp toolchain), singlefile↔diffusers repackage.
- **Publish**: one commit call against Tensorhub's HF-shaped `/commits` write API. `mode` defaults to `replace` (th#1400 — a checkpoint is complete in itself); pass `mode="merge"` explicitly, and only when adding to an existing snapshot.
- **Tenant SDK**: `Source`, `Component`, `Dataset`, `ProducedFlavor`, streaming cast/fp8 writers, calibration policy — for `@endpoint(kind="conversion")` endpoints.

## Memory floors per operation (gw#395/#396)

Model size must never dictate converter hardware. What each operation actually needs:

| Operation | Peak anonymous RAM | Why |
|---|---|---|
| dtype cast (`streaming_dtype_cast` / `streaming_cast_snapshot`) | ≈ largest single tensor | two-pass streaming: header-only shard plan, then one tensor at a time |
| fp8-E4M3 storage flavor (`streaming_fp8_storage_cast` / `streaming_fp8_snapshot`) | ≈ largest single tensor | same engine; clamp ±448 + layerwise-cast skip patterns |
| byte-offset reshard / shard merge (`merge_safetensors_by_offset`) | O(1) (8 MB copy chunks) | raw byte-range copy, no tensor decode |
| bnb nf4/fp4 | full component | `from_pretrained` load, inherent to bnb |
| singlefile↔diffusers repackage | **full model** | `from_single_file` / whole-keyspace remap needs the full tensor set — the one legitimate big-RAM operation |
| GGUF | full model (llama.cpp converter) | external toolchain |

Casts and fp8 flavor production run on the standard 32 GB CPU class regardless of model size;
only repackage/GGUF of huge models still needs RAM sized to the model.

```python
from gen_worker.convert import clone

result = clone.from_huggingface(ctx, payload)   # download → convert → one commit per flavor
```
