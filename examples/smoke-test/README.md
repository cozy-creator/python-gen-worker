Simple worker functions for smoke testing the orchestrator.

Functions:
- `add_numbers` - adds two integers
- `multiply_numbers` - multiplies two integers
- `image_gen_action` - smoke image generation (returns real `Asset` outputs; no ML)
- `token_stream` - incremental output example (LLM-style token streaming)

Notes:
- `image_gen_action` returns a list of output `Asset`s. Each asset is written via `ctx.save_bytes(...)` under
  `runs/<run_id>/outputs/` so the orchestrator/file API can store it like a real run output.
