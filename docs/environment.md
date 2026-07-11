# Environment variables

All ambient worker config flows through the typed `Settings` struct
(`config/settings.py`), loaded by `config/loader.py` with precedence
env → `./.env` → `/run/secrets` → yaml → struct defaults. Call sites read
`Settings` (the startup instance, or the cached `get_settings()`), never
`os.getenv`. The guard test `tests/test_env_surface.py` fails the suite on
any raw env read outside `config/` that isn't on its plumbing allowlist.

Tenant *endpoint* code reads its own envs freely (`docs/endpoint-envs.md`);
this page covers the worker itself.

## Secrets (Settings fields)

| Env | Field | Why env |
|---|---|---|
| `HF_TOKEN` (alias `HUGGING_FACE_HUB_TOKEN`) | `hf_token` | HF pulls of gated/private repos |
| `WORKER_JWT` | `worker_jwt` | orchestrator-issued worker identity token |
| `GRPC_CA_BUNDLE` | `grpc_ca_bundle` | PEM CA path for orchestrator gRPC TLS |
| `TENSORHUB_TOKEN` | `tensorhub_token` | standalone-CLI private tensorhub pulls |
| `CIVITAI_API_KEY` (alias `CIVITAI_TOKEN`) | `civitai_api_key` | civitai provider downloads |

## Orchestrator-injected deployment config (Settings fields)

| Env | Field | Why env |
|---|---|---|
| `TENSORHUB_PUBLIC_URL` | `tensorhub_public_url` | injected per-cluster at pod launch |
| `ORCHESTRATOR_PUBLIC_ADDR` | `orchestrator_public_addr` | router address, injected at pod launch |
| `WORKER_ID` | `worker_id` | per-pod identity |
| `ENDPOINT_LOCK_PATH` | `endpoint_lock_path` | discovery manifest path (baked default in images) |
| `RUNPOD_POD_ID` | `runpod_pod_id` | set by the RunPod runtime |
| `WORKER_DISCONNECTED_TIMEOUT_S` | `worker_disconnected_timeout_s` | self-exit window when orchestrator unreachable |
| `WORKER_IMAGE_DIGEST` / `WORKER_GIT_COMMIT` | `worker_image_digest` / `worker_git_commit` | provenance stamped by the image build |

## Tuning knobs (Settings fields)

| Env | Field | Why env |
|---|---|---|
| `HF_HOME` | `hf_home` | HF cache root (also read by huggingface_hub itself) |
| `TENSORHUB_URL` | `tensorhub_url` | standalone-CLI resolve base URL |
| `TENSORHUB_CACHE_DIR` / `TENSORHUB_CAS_DIR` | `tensorhub_cache_dir` / `tensorhub_cas_dir` | move cache/CAS off `/tmp` (cozy local persistence) |
| `COZY_HF_DOWNLOAD_STALL_TIMEOUT_S` | `hf_download_stall_timeout_s` | HF download stall window (default 180s) |
| `COZY_HF_DOWNLOAD_MAX_SECONDS` | `hf_download_max_seconds` | HF download wall-clock cap (0 = off) |
| `COZY_HF_MAX_REPO_BYTES` | `hf_max_repo_bytes` | accidental-huge-repo guard (default 60 GB, 0 = off) |
| `GEN_WORKER_ATTACHED_LORA_MAX` / `_MAX_BYTES` | `attached_lora_max` / `attached_lora_max_bytes` | LoRA residency caps |
| `GEN_WORKER_COMPILE_CACHE` / `_CACHE_URL` | `compile_cache_path` / `compile_cache_url` | compile-cache artifact source (path / URL) |
| `GEN_WORKER_COMPILE_ALLOW_COLD` | `compile_allow_cold` | opt into cold compilation (needs a C toolchain) |

## CI-lane opt-ins (raw env, tests/CI only)

- `GEN_WORKER_GPU_SMOKE` / `GEN_WORKER_SMOKE_OUT` — nightly GPU CI lane
  selection + artifact dir (`gpu-ci.yml`); never read by worker runtime code.

## Internal plumbing (raw env, allowlisted in tests/test_env_surface.py)

- `PYTORCH_CUDA_ALLOC_CONF` — `entrypoint.py` setdefault before torch import.
- `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` — WRITTEN by
  `compile_cache.py` to latch inductor/triton onto seeded dirs (children
  inherit); `GEN_WORKER_COMPILE_ALLOW_COLD` is also written by the producer
  for its spawned compile workers.
- `GEN_WORKER_LOCAL_DEVICE` — set by `run/serve --device` for endpoint code;
  read back on the local path.
- `GEN_WORKER_LOCAL_OUTPUT_DIR`, `USER` — cozy-local app plumbing / login
  fallback (`cli/local_context.py`).
- `GEN_WORKER_NO_PRECISION_LADDER` — local-run ladder opt-out, read
  per-invocation (`cli/run.py`).
- `COZY_HTTP_CONNECT_TIMEOUT_S` / `COZY_HTTP_READ_TIMEOUT_S` — http timeout
  floors, per-call by design so tests can tune them (`net.py`, gw#456).
- `COZY_CIVITAI_DOWNLOAD_ATTEMPTS`, `COZY_CLONE_DOWNLOAD_ATTEMPTS` —
  per-call test-tunable retry counts (`models/download.py`,
  `convert/ingest.py`).
- `COZY_CONVERT_WORKDIR` / `_DISK_HEADROOM` / `_SCRATCH_TTL_S` /
  `_RETAIN_WORKDIR` — convert-job scratch knobs set by the invoking harness
  (`convert/clone.py`).
- `GEN_WORKER_FORBID_CPU_OFFLOAD` — dev-box kill-switch: raises at real
  pipeline placement time if weights would land on CPU (per-machine fact;
  kept deliberately after gw#139's veto-removal was superseded).
