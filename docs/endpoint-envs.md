# Endpoint envs (tenant-defined configs/secrets)

Full API reference: tensorhub `docs/endpoint-envs-api.md`.

## How it works

1. The org attaches env entries to an endpoint via tensorhub
   (`/api/v1/endpoints/:owner/:endpoint_name/env`). Each entry has a name,
   value, optional `sensitive` flag, optional description, and optional
   `applies_to` (which releases receive it).
2. Values live in Vault; Postgres holds metadata only.
3. When the orchestrator launches a worker pod for a release, it resolves the
   filtered env map for that release and injects it into the pod's env
   alongside the worker's own envs (`WORKER_JWT`, ...).
4. Endpoint code reads them via plain `os.getenv("MY_KEY")`. No SDK wrapper,
   no context object.

Endpoint code does NOT declare expected envs anywhere — document them in your
endpoint's README; the tenant attaches values at runtime. This decoupling lets
a tenant add an env (e.g. a debug flag) to a deployed endpoint without a
rebuild.

```python
import os
from gen_worker import endpoint

@endpoint
class CivitaiProxy:
    def generate(self, ctx, payload):
        key = os.getenv("CIVITAI_API_KEY", "")
        if not key:
            raise RuntimeError("CIVITAI_API_KEY env is not configured for this endpoint")
        ...
```

## applies_to: targeting specific releases

Default is every release. Options (OR semantics — a release matches if any
condition holds):

```json
{"applies_to": {"tags": ["staging"]}}
{"applies_to": {"compatibility_versions": [">=2.0.0 <3.0.0"]}}
{"applies_to": {"release_ids": ["r_abc123"]}}
```

## Reserved namespace

Names set by gen-worker or the container runtime can't be bound:

- Exact: `PATH`, `HOME`, `USER`, `SHELL`, `PWD`, `TERM`, `HOSTNAME`, `LANG`
- Prefixes: `WORKER_*`, `HF_*`, `TENSORHUB_*`, `ORCHESTRATOR_*`, `TRAINER_*`,
  `RUNPOD_*`, `LC_*`, `CUDA_*`, `NVIDIA_*`, `PYTHON*`, `LD_*`

Exception: `HF_TOKEN` is bindable — the platform never injects a shared HF
token into tenant pods; authors attach their own per endpoint to pull
gated/private HF repos.

## Rotation

Editing a value takes effect on the **next pod spawn**. Existing pods keep
their env until recycled; there is no in-pod live reload.
