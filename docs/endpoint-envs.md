# Endpoint envs (tenant-defined configs/secrets)

Full API reference: tensorhub `docs/endpoint-envs-api.md`.

## How it works

1. The org attaches env compiled graphs to an endpoint via tensorhub
   (`/api/v1/endpoints/:owner/:endpoint_name/env`). Each compiled graph has a name,
   value, optional `sensitive` flag, optional description, and optional
   `applies_to` (which releases receive it).
2. Values live in Vault; Postgres holds metadata only.
3. When the orchestrator launches a worker pod for a release, it resolves the
   filtered env map for that release and injects it into the pod's env
   alongside the worker's own envs. Note that the worker's own credentials are
   then stripped before endpoint code runs: `procsplit/parent.py`
   `_CHILD_FORBIDDEN_ENVS` (`WORKER_JWT`, `RUNPOD_API_KEY`, `PUBLIC_KEY`) are
   popped from the compute child's env, so an endpoint never sees them in
   `os.environ`.
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
- Selected `GEN_WORKER_*` names only — `GEN_WORKER_C2PA_*` (th#714 platform
  signing material) and the pgw#763 process-split namespace
  (`GEN_WORKER_PROCESS_SPLIT`, `GEN_WORKER_COMPUTE_CHILD`,
  `GEN_WORKER_CHILD_*`)

Enforced in tensorhub, `internal/api/endpoint_env_reserved.go`
(`reservedEnvNames` / `reservedEnvPrefixes`) — that file is authoritative.

> **`GEN_WORKER_*` is NOT reserved as a namespace**, and it does not match the
> `WORKER_*` prefix. Only the specific names above are blocked. Everything
> else under `GEN_WORKER_*` is tenant-bindable today, including several safety
> gates read straight from `os.environ` in the same process tenant code runs
> in: `GEN_WORKER_INTERNAL_OBJECT_HOSTS` (SSRF-policy bypass for
> resolver-minted URLs), `GEN_WORKER_URL_FETCH_ALLOWED_HOSTS` (egress bound),
> `GEN_WORKER_HOST_MOVE_GUARD=0` (disables the host-RAM move guard),
> `GEN_WORKER_SUPERVISOR=0` (disables the OOM-reporting supervisor fork), and
> `GEN_WORKER_COMPUTE_UID` (privilege-drop target uid). This is a gap in the
> reserved list, recorded here rather than asserted as safe.

Exception: `HF_TOKEN` is bindable — the platform never injects a shared HF
token into tenant pods; authors attach their own per endpoint to pull
gated/private HF repos.

## Rotation

Editing a value takes effect on the **next pod spawn**. Existing pods keep
their env until recycled; there is no in-pod live reload.
