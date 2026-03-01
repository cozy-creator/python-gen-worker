# Issue 80: Migration Inventory (`gen-trainer` + `ostris-ai-toolkit`)

This inventory classifies candidate pieces into:

- runtime helper surface (`python-gen-worker`)
- endpoint template/example code (`~/cozy/training-endpoints`)
- drop (do not migrate)

## Source: `~/gen-trainer`

### Runtime Helper Surface

- `src/gen_trainer/data/parquet_reader.py`
  - keep in runtime/helper scope (parquet batch iteration)
  - mapped in `gen_worker.trainer.arrow_feed`
- `src/gen_trainer/data/materialize.py`
  - keep in runtime/helper scope (local materialization helpers)
  - no model-family logic
- `src/gen_trainer/optim/builders.py`
  - keep as optional low-level helper (generic AdamW/scheduler builder)
  - mapped in `gen_worker.trainer.helpers.optim`
- `src/gen_trainer/data/image_payloads.py`
  - keep as helper/template reference (image-ref local-path resolution strategy)

### Endpoint Template / Example Surface

- `src/gen_trainer/engines/common.py`
  - keep as reference pattern for endpoint-owned training logic
- `src/gen_trainer/engines/sdxl_lora.py`
- `src/gen_trainer/engines/flux1_lora.py`
- `src/gen_trainer/engines/flux2_lora.py`
- `src/gen_trainer/engines/qwen_image_lora.py`
- `src/gen_trainer/engines/z_image_lora.py`
  - all engine files remain endpoint-owned (model-family specific)

### Drop / Do Not Migrate

- `src/gen_trainer/entrypoint.py`
  - old outer runtime loop + old plugin contract; replaced by `gen_worker.trainer.runtime` + class-only contract
- `src/gen_trainer/types.py` old plugin protocol (`configure_optimizers_and_schedulers + step`)
  - superseded by class-only contract in `gen_worker.trainer.contracts`

## Source: `~/ostris-ai-toolkit`

### Endpoint Template / Example Surface

- `extensions_built_in/diffusion_models/*`
  - architecture-specific backend logic; endpoint-owned
- `extensions_built_in/sd_trainer/SDTrainer.py`
- `jobs/process/BaseSDTrainProcess.py`
  - training-step algorithm references only (not runtime ownership)

### Drop / Do Not Migrate

- `ui/*`
- sqlite/job wrapper orchestration layers
- extension manager/runtime server plumbing
  - all outside `python-gen-worker` runtime ownership

## Consolidation Rule

- Runtime (`python-gen-worker`) may expose only low-level, model-agnostic helpers.
- Model-family training semantics stay in endpoint code.
- No centralized runtime-owned model backend abstraction is introduced.
