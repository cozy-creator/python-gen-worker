This is a python package, called gen_worker, which provides the worker runtime SDK:

- Orchestrator gRPC client + job loop
- Function discovery via @worker_function
- ActionContext + errors + progress events
- Model downloading from the Cozy hub (async + retries + progress)
- Output saving via the Cozy hub file API (ctx.save_bytes/ctx.save_file -> Asset refs)

Torch-based model memory management is optional and installed via extras.

---

Files in src/gen_worker/pb must be auto-generated in the gen-orchestrator repo, using the proto files. Go in there and run `task proto`

Install modes:

- Core only: `gen-worker`
- Torch runtime add-on: `gen-worker[torch]` (torch + torchvision + torchaudio + safetensors + flashpack + numpy)

Example tenant projects live in `../worker-example-functions`. They use:

- `pyproject.toml` + `uv.lock` for dependencies (no requirements.txt)
- `[tool.cozy]` in `pyproject.toml` for deployment config (functions.modules, runtime.base_image, etc.)

Dependency policy:

- Require `pyproject.toml` and/or `uv.lock`
- Do not use `requirements.txt`
- Put Cozy deployment config in `pyproject.toml` under `[tool.cozy]`

Example:

```toml
[tool.cozy]
functions.modules = ["functions"]

[tool.cozy.runtime]
base_image = "ghcr.io/cozy/python-worker:cuda12.1-torch2.6"
```

Function signature:

```python
from gen_worker import worker_function, ResourceRequirements, ActionContext

@worker_function(ResourceRequirements(model_family="sdxl", requires_gpu=True))
def generate(ctx: ActionContext, payload: dict) -> dict:
    return {"ok": True}
```

Dynamic checkpoints:

- Use `ResourceRequirements(model_family=...)` to declare a family (e.g., "sdxl")
- Pass the exact checkpoint at runtime via request payload (e.g., `model_ref`)

Build contract (gen-builder):

- Tenant code + `pyproject.toml`/`uv.lock` are packaged together
- gen-builder layers tenant code + deps on top of a python-worker base image
- gen-orchestrator deploys the resulting worker image

---

Env hints:

- `SCHEDULER_ADDR` sets the primary scheduler address.
- `SCHEDULER_ADDRS` (comma-separated) provides seed addresses for leader discovery.
- `WORKER_JWT` is accepted as the auth token if `AUTH_TOKEN` is not set.
- `SCHEDULER_JWKS_URL` enables verification of `WORKER_JWT` before connecting.
- JWT verification uses RSA and requires PyJWT crypto support (installed by default via `PyJWT[crypto]`).
- `WORKER_MAX_INPUT_BYTES`, `WORKER_MAX_OUTPUT_BYTES`, `WORKER_MAX_UPLOAD_BYTES` cap payload sizes.
- `WORKER_MAX_CONCURRENCY` limits concurrent runs; `ResourceRequirements(max_concurrency=...)` limits per-function.
- `COZY_HUB_URL` base URL for Cozy hub downloads (used by core downloader).
- `COZY_HUB_TOKEN` optional bearer token for Cozy hub downloads.
- `MODEL_MANAGER_CLASS` optional ModelManager plugin (module:Class) loaded at startup.

Error hints:

- Use `gen_worker.errors.RetryableError` in worker functions to flag retryable failures.

API note:

- `output_format` is an orchestrator HTTP response preference (queue vs long-poll bytes/url) and does not change worker behavior; workers persist outputs as `Asset` refs via the Cozy hub file API.
