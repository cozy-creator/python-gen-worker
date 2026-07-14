# Environment variables

Orchestrator-injected pod config and anything an operator needs to override
at deploy time flows through the typed `Settings` struct (`config/settings.py`),
loaded by `config/loader.py` with precedence env → `./.env` → `/run/secrets`
→ yaml → struct defaults. Call sites read `Settings` (the startup instance,
or the cached `get_settings()`).

A handful of modules that also work as standalone libraries/CLIs outside a
full worker bring-up (`net.py`, `convert/ingest.py`, `convert/clone.py`,
`cli/run.py`, `compile_cache.py`) read a few env vars directly instead —
see "Internal plumbing" below. This doc is the source of truth for which
knobs are which.

Tenant *endpoint* code reads its own envs freely (`docs/endpoint-envs.md`);
this page covers the worker itself.

## Secrets (Settings fields)

| Env | Field | Why env |
|---|---|---|
| `HF_TOKEN` | `hf_token` | HF pulls of gated/private repos |
| `WORKER_JWT` | `worker_jwt` | orchestrator-issued worker identity token |
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
| `WORKER_IMAGE_DIGEST` | `worker_image_digest` | provenance stamped by the image build — currently no launcher sets it (pgw#514/P4); kept for a future tensorhub stamp |

## Tuning knobs (Settings fields)

| Env | Field | Why env |
|---|---|---|
| `HF_HOME` | `hf_home` | HF cache root (also read by huggingface_hub itself) |
| `TENSORHUB_URL` | `tensorhub_url` | standalone-CLI resolve base URL |
| `TENSORHUB_CACHE_DIR` / `TENSORHUB_CAS_DIR` | `tensorhub_cache_dir` / `tensorhub_cas_dir` | move cache/CAS off `/tmp` (cozy local persistence); `CAS_DIR` also isolates the `cli/run.py` standalone CLI in tests |

## C2PA Content Credentials (Settings fields, th#714)

Every generated media asset (png/jpeg/webp/gif, mp4/mov, wav/mp3/flac/m4a)
gets a signed C2PA provenance manifest at `ctx.save_bytes`/`save_file` time --
the EU AI Act Art. 50 machine-readable AI-marking. ON iff the cert is
configured; unconfigured logs a loud startup warning and no-ops;
configured-but-broken refuses to start.

| Env | Field | Notes |
|---|---|---|
| `GEN_WORKER_C2PA_CERT_PATH` | `c2pa_cert_path` | PEM signing-cert chain, leaf first (the ON switch) |
| `GEN_WORKER_C2PA_KEY_PATH` | `c2pa_key_path` | PKCS#8 PEM private key for the leaf |
| `GEN_WORKER_C2PA_ALG` | `c2pa_alg` | COSE alg matching the cert key (default `es256`) |
| `GEN_WORKER_C2PA_TA_URL` | `c2pa_ta_url` | optional RFC3161 timestamp-authority URL |

Needs the `signing` extra (`pip install gen-worker[signing]`, c2pa-python).

## Removed in the pgw#514 dead-config sweep

These used to be Settings fields backed by env vars. No deployment
(gen-orchestrator / tensorhub / e2e) ever set the env var, so the values are
now plain module constants — change the source if you need a different
value, there is no env override anymore:

| Was | Now a constant in |
|---|---|
| `COZY_HF_DOWNLOAD_STALL_TIMEOUT_S` (180.0) | `models/download.py::_HF_DOWNLOAD_STALL_TIMEOUT_S` |
| `COZY_HF_DOWNLOAD_MAX_SECONDS` (0.0 = off) | `models/download.py::_HF_DOWNLOAD_MAX_SECONDS` |
| `COZY_HF_MAX_REPO_BYTES` (60 GB, 0 = off) | `models/download.py::_HF_MAX_REPO_BYTES` |
| `GEN_WORKER_ATTACHED_LORA_MAX` (8) | `utils/lora.py::MAX_ATTACHED_ADAPTERS` |
| `GEN_WORKER_ATTACHED_LORA_MAX_BYTES` (2 GiB) | `utils/lora.py::MAX_ATTACHED_ADAPTER_BYTES` |

Also removed outright (no consumer at all): `GRPC_CA_BUNDLE`,
`WORKER_GIT_COMMIT` (the `WorkerResources.git_commit` proto field stays but
gen-worker stopped populating it — see the proto comment), and the
`HUGGING_FACE_HUB_TOKEN` alias for `HF_TOKEN` (`HF_TOKEN` is the one name).

`GEN_WORKER_COMPILE_CACHE` / `_CACHE_URL` / `_ALLOW_COLD` moved from
Settings fields to raw reads in `compile_cache.py` (real, tested
manual-override / compile-cell-producer knobs — see below) since no
production launcher ever populated them through Settings either.

## CI-lane opt-ins (raw env, tests/CI only)

- `GEN_WORKER_GPU_SMOKE` — opts a GPU-only smoke test into a run (e.g. the
  llama-server CUDA smoke in `tests/test_llama_runtime.py`); never read by
  worker runtime code. Real-model GPU coverage now lives in the e2e repo's
  nightly `TestJ6` cloud journey, not a gen-worker-repo GPU lane.

## Internal plumbing (raw env)

- `PYTORCH_CUDA_ALLOC_CONF` — `entrypoint.py` setdefault before torch import.
- `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` — WRITTEN by
  `compile_cache.py` to latch inductor/triton onto seeded dirs (children
  inherit); `GEN_WORKER_COMPILE_ALLOW_COLD` is also written by the producer
  for its spawned compile workers.
- `GEN_WORKER_COMPILE_CACHE` / `_CACHE_URL` — compile-cache artifact source
  (local tar path / presigned URL), read raw in `compile_cache.py::prepare`.
  Tested (`tests/test_compile_cache.py`) but never set by a production
  launcher — the hub-attached-artifact path (tensorhub #569) is primary;
  these are the local-dev / compile-job manual override.
- `GEN_WORKER_COMPILE_ALLOW_COLD` — opt into cold `torch.compile` without a
  seeded artifact (needs a C toolchain), read raw in
  `compile_cache.py::apply`.
- `GEN_WORKER_LOCAL_OUTPUT_DIR`, `USER` — cozy-local app plumbing / login
  fallback (`cli/local_context.py`).
- `COZY_HTTP_CONNECT_TIMEOUT_S` / `COZY_HTTP_READ_TIMEOUT_S` — http timeout
  floors, per-call by design so tests can tune them (`net.py`, gw#456).
- `COZY_CIVITAI_DOWNLOAD_ATTEMPTS`, `COZY_CLONE_DOWNLOAD_ATTEMPTS` —
  per-call test-tunable retry counts (`models/download.py`,
  `convert/ingest.py`).
- `COZY_CONVERT_WORKDIR` / `_DISK_HEADROOM` / `_SCRATCH_TTL_S` /
  `_RETAIN_WORKDIR` — convert-job scratch knobs set by the invoking harness
  (`convert/clone.py`).
