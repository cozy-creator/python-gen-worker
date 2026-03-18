# flux2-klein-4b

FLUX.2-klein 4B endpoint with separate base/turbo functions and dtype-specific variants.

Naming convention in this repo:

- Base model ref: `black-forest-labs/flux.2-klein-4b-base`
- Turbo model ref: `black-forest-labs/flux.2-klein-4b-turbo`

This avoids ambiguity with upstream naming where `flux.2-klein-4b` is commonly used for turbo variants.

Functions:

- `generate`: base bf16
- `generate_turbo`: turbo bf16
- `generate_fp8`: base fp8
- `generate_turbo_fp8`: turbo fp8
- `generate_nvfp4`: base nvfp4
- `generate_turbo_nvfp4`: turbo nvfp4

Notes:

- Fixed model selection is declared in code via `ModelRef(Src.FIXED, "<model-key>")`.
- Model refs/dtypes are declared in `endpoint.toml [models]`.
- `num_inference_steps` is accepted in the payload, but clamped to `[4, 8]`.
