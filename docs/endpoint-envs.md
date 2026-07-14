# Endpoint envs (tenant-defined configs/secrets)

End-to-end design: progress.json issue #254 in `~/cozy/tensorhub`.

## How it works

1. The org owner attaches env entries to a specific endpoint via tensorhub's
   API (`/api/v1/endpoints/:owner/:endpoint_name/env`). Each entry has a
   name, value, optional `sensitive` flag, optional description, and
   optional `applies_to` (which releases of the endpoint receive it).
2. Values are stored in Vault at
   `tensorhub/orgs/<org_id>/endpoints/<endpoint_id>/env/<name>`. Postgres
   holds metadata only.
3. When gen-orchestrator launches a worker pod for a specific release, it
   makes one mTLS call to tensorhub's internal resolve route. tensorhub
   returns the filtered + dereferenced env map for that release. The
   orchestrator injects those into the pod's env alongside the worker's
   universal envs (HF_TOKEN, WORKER_JWT, etc.).
4. Endpoint code reads them via plain `os.getenv("MY_KEY")`. No SDK
   wrapper, no context object.

The endpoint code does NOT declare expected envs anywhere — the tenant
decides at runtime which envs to attach. Document expected envs in your
endpoint's README.

## Example: an endpoint that calls Civitai

```python
# my_endpoint/main.py
import os
import requests

from gen_worker import endpoint

CIVITAI_API_KEY = os.getenv("CIVITAI_API_KEY", "")

@endpoint
class CivitaiProxy:
    def setup(self) -> None:
        pass

    def generate(self, ctx, payload):
        if not CIVITAI_API_KEY:
            raise RuntimeError("CIVITAI_API_KEY env is not configured for this endpoint")
        resp = requests.get(
            f"https://civitai.com/api/v1/models/{payload.model_id}",
            headers={"Authorization": f"Bearer {CIVITAI_API_KEY}"},
        )
        ...
```

In the endpoint's README:

> ## Required env vars
>
> Bind these to your endpoint via `tensorhub`:
>
> - `CIVITAI_API_KEY` (sensitive) — your Civitai API token

The tenant then runs:

```bash
curl -X POST https://tensorhub.example.com/api/v1/endpoints/myorg/civitai-puller/env \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"name":"CIVITAI_API_KEY","value":"<token>","sensitive":true}'
```

Next pod spawn for that endpoint will have `CIVITAI_API_KEY` in its env.

## applies_to: targeting specific releases

By default an env applies to every release of the endpoint. Override with
`applies_to` to limit when it's exposed:

```json
{
  "name": "DEBUG_FLAG",
  "value": "1",
  "applies_to": {"tags": ["staging"]}
}
```

This entry only flows into pods spawned for releases tagged `staging`.
Other targeting options:

```json
{"applies_to": {"compatibility_versions": [">=2.0.0 <3.0.0"]}}
{"applies_to": {"release_ids": ["r_abc123"]}}
{"applies_to": {"tags": ["prod"], "compatibility_versions": [">=2.0.0"]}}  // OR
```

OR semantics: a release matches if **any** of the listed conditions hold.

## Reserved namespace

You cannot bind names matching any of these — they're set by gen-worker
or the container runtime, and shadowing them would break the worker:

- Exact: `PATH`, `HOME`, `USER`, `SHELL`, `PWD`, `TERM`, `HOSTNAME`, `LANG`
- Prefixes: `WORKER_*`, `HF_*`, `TENSORHUB_*`, `ORCHESTRATOR_*`,
  `TRAINER_*`, `RUNPOD_*`, `LC_*`, `CUDA_*`, `NVIDIA_*`, `PYTHON*`, `LD_*`

`HF_TOKEN` and `HF_HOME` specifically are worker-universal and read by
gen-worker itself from its `Settings` struct — every endpoint already has
them. Don't try to override.

## Rotation

Editing an env value via `PUT /api/v1/endpoints/:owner/:endpoint_name/env/:name`
takes effect on the **next pod spawn** for that endpoint. Existing pods
keep their stale env until they're recycled. There is no in-pod live
reload in v1.

## Why not declare envs in project config?

Considered and explicitly rejected. The endpoint author writes Python and
documents expected envs in their README; the tenant attaches values
through tensorhub at runtime. Decoupling the two lets the tenant add a
new env (e.g. a debug flag) to a deployed endpoint without rebuilding it,
and lets endpoint authors iterate without re-declaring config schemas.
