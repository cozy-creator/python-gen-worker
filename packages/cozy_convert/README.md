# cozy-convert

Cozy Creator's model ETL, split out of `gen-worker` (issue #367).

- **Ingest**: HuggingFace (`HfApi.list_repo_files` + classifier + `snapshot_download(allow_patterns=…)`) and Civitai (bounded provider API).
- **Convert**: streaming dtype cast, weight-only quant (torchao / bitsandbytes), GGUF (llama.cpp toolchain), singlefile↔diffusers repackage.
- **Publish**: one commit call against Tensorhub's HF-shaped `/commits` write API (`mode: merge|replace`).
- **Tenant SDK**: `Source`, `Component`, `Dataset`, `ProducedFlavor`, `StreamingWriter`, calibration policy — for `@endpoint(kind="conversion")` endpoints.

```python
from cozy_convert import clone_huggingface

result = clone_huggingface(ctx, payload)   # download → convert → one commit per flavor
```

Lives in the `python-gen-worker` uv workspace; depends on `gen-worker` (contexts, identity), never the other way around.
