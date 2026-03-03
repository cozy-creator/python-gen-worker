Simple worker functions for smoke testing the orchestrator.

Functions:
- `add_numbers` - adds two integers
- `multiply_numbers` - multiplies two integers
- `image_gen_action` - smoke image generation (returns real `Asset` outputs; no ML)
- `token_stream` - incremental output example (LLM-style token streaming)
- `caption_prompts` - multi-item request example (`prompts[]`) for orchestrator split/combine + correlation tests

Notes:
- `image_gen_action` returns a list of output `Asset`s. Each asset is written via `ctx.save_bytes(...)` under
  `runs/<request_id>/outputs/` so the orchestrator/file API can store it like a real request output.
- `caption_prompts` keeps item order stable and emits request-scoped progress, making it a good test target for
  orchestrator split/combine aggregation flows.
