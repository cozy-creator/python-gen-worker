# Environment variables

Orchestrator-injected pod config and anything an operator needs to override
at deploy time flows through the typed `Settings` struct (`config/settings.py`),
loaded by `config/loader.py` with precedence env → `./.env` → `/run/secrets`
→ yaml → struct defaults.

**Ruling §1.18: exactly one component in this process reads the environment.**
Each process entry performs one bootstrap-owned `load_settings()` and the
resulting `Settings` is **passed by parameter**. `get_settings()` — the
`lru_cache`d process-global — was DELETED in pgw#931; where a module is too
deep to be handed a parameter it reads what the entry published
(`config.current()`), which raises rather than silently loading if nothing was
installed.

**Raw-env reads ARE centrally registered now.** They are
`scripts/config_reads_allowlist.txt`, one classified line per accepted site,
enforced by `scripts/lint_config_reads.py` in CI. That file — not this page —
is the authoritative census; this page is orientation. The previous prose
version of that list named 5 files while the reads lived in 41.

An unrecognised key inside a gen-worker-owned namespace (`GEN_WORKER_`,
`TENSORHUB_`, `WORKER_`, `COZY_`) in `.env`, `/run/secrets` or yaml now
**raises** `UnknownSettingError` instead of being accepted and ignored. In the
process environment it is reported at boot (`unknown_owned_env`) but not
refused — the pod env is assembled by Tensorhub, which legitimately injects
owned-namespace names this worker has no reader for.

Tenant *endpoint* code reads its own envs freely (`docs/endpoint-envs.md`);
this page covers the worker itself.

## Secrets (Settings fields)

| Env | Field | Why env |
|---|---|---|
| `HF_TOKEN` | `hf_token` | HF pulls of gated/private repos |
| `WORKER_JWT` | `bootstrap_worker_jwt` | orchestrator-issued worker identity token. The FIELD was renamed (pgw#848) so no call site can mistake it for the live credential; the env name is hub-fixed. Stripped from the compute child's env (`procsplit/parent.py` `_CHILD_FORBIDDEN_ENVS`) before tenant code runs, so an endpoint never sees it in `os.environ` |
| `TENSORHUB_TOKEN` | `tensorhub_token` | private tensorhub pulls (standalone CLI) AND the mint/publish gate on real pods (`aot_mint.py`) |
| `CIVITAI_API_KEY` (alias `CIVITAI_TOKEN`) | `civitai_api_key` | civitai provider downloads |

## Orchestrator-injected deployment config (Settings fields)

| Env | Field | Why env |
|---|---|---|
| `TENSORHUB_PUBLIC_URL` | `tensorhub_public_url` | injected per-cluster at pod launch |
| `ORCHESTRATOR_PUBLIC_ADDR` | `orchestrator_public_addr` | router address, injected at pod launch |
| `WORKER_ID` | `worker_id` | per-pod identity |
| `ENDPOINT_LOCK_PATH` | `endpoint_lock_path` | discovery manifest path (baked default in images) |
| `RUNPOD_POD_ID` | `runpod_pod_id` | set by the RunPod runtime |
| `WORKER_IMAGE_DIGEST` | `worker_image_digest` | immutable provenance stamped by Tensorhub from the selected release image variant; also read raw by `compile_cache.py` to stamp `image_digest` into every compile cell |
| `WORKER_RELEASE_ID` | `worker_release_id` | release identity; re-exported into the compute child |
| `WORKER_CONFIG_GENERATION` | `boot_config_generation` | boot-config generation, for detecting hub-injected config |
| `WORKER_EXECUTION_TOPOLOGY` | — (raw, `topology.py`) | hub-delivered multi-GPU topology JSON (see [multi-gpu.md](multi-gpu.md)). Absent is legal and means one slot — so an unset value on a multi-GPU pod silently serves one card |
| `GEN_WORKER_CONFIG_SNAPSHOT_PATH` | `config_snapshot_path` | runtime-config snapshot path; also read raw and propagated to every subprocess |
| `GEN_WORKER_BOOT_RECORD` | `boot_record_path` | post-mortem boot-record carrier. Added in pgw#931: it had no field and was read raw in three places, so the parent and the child could disagree about where the record lived. Empty = `postmortem` picks a durable default |

## Tuning knobs (Settings fields)

| Env | Field | Why env |
|---|---|---|
| `HF_HOME` | `hf_home` | HF cache root (also read by huggingface_hub itself) |
| `TENSORHUB_URL` | `tensorhub_url` | standalone-CLI resolve base URL |
| `TENSORHUB_CACHE_DIR` | `tensorhub_cache_dir` | THE cache/CAS root knob — move cache/CAS off `/tmp` (cozy local persistence). `cache_paths.tensorhub_cas_dir()` derives the CAS from this and nothing else, so this is also what isolates the standalone `cli/run.py` in tests |
| `TENSORHUB_FILL_SOURCE_DIR` | `tensorhub_fill_source_dir` | th#850 managed-tier ruling: an endpoint-scoped datacenter-warm CAS mount (RunPod volume), checked before R2 on a blob miss and write-through warmed from R2. Never the CAS root — that always stays `TENSORHUB_CACHE_DIR`/local. tensorhub sets this only when a volume is attached; ismount-guarded, so a plain directory never gets mistaken for it |

## C2PA Content Credentials (Settings fields, th#714)

Every generated media asset (png/jpeg/webp/gif, mp4/mov, wav/mp3/flac/m4a)
gets a signed C2PA provenance manifest at `ctx.save_bytes`/`save_file` time --
the EU AI Act Art. 50 machine-readable AI-marking. ON iff the cert is
configured; unconfigured logs a loud startup warning and no-ops;
configured-but-broken refuses to start.

Only the PUBLIC half is pod config. The private key is **hub-side** (th#1307):
the hub holds it and signs claims over `POST /v1/worker/c2pa/sign`, armed at
HelloAck.

| Env | Field | Notes |
|---|---|---|
| `GEN_WORKER_C2PA_CERT_PEM` | `c2pa_cert_pem` | inline PEM signing-cert chain, leaf first. What the hub injects at launch (RunPod pods have no file mounts); takes precedence over `_CERT_PATH` |
| `GEN_WORKER_C2PA_CERT_PATH` | `c2pa_cert_path` | file-path variant for mounted deploys (the ON switch is either cert form) |
| `GEN_WORKER_C2PA_ALG` | `c2pa_alg` | COSE alg matching the cert key (default `es256`) |
| `GEN_WORKER_C2PA_TA_URL` | `c2pa_ta_url` | optional RFC3161 timestamp-authority URL |

**`GEN_WORKER_C2PA_KEY_PEM` and `GEN_WORKER_C2PA_KEY_PATH` are REFUSED, not
configured.** They are deliberately absent from `config/loader.py`'s env map,
and `content_credentials.configure()` RAISES at startup if either is present in
the pod environment (`_REFUSED_KEY_ENVS`). Tenant code runs in this process and
could read a key delivered here, so their presence is treated as a platform
regression that must kill the pod loudly. Do not set them.

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

The old `GEN_WORKER_COMPILE_CACHE`, `_CACHE_URL`, and `_ALLOW_COLD` knobs were
removed. Serving receives immutable compile artifacts through Tensorhub;
local-cell and producer tools use explicit library arguments. An inherited
environment can therefore never bypass scheduler selection or mandatory W8A8
compile evidence.

## CI-lane opt-ins (raw env, tests/CI only)

- `GEN_WORKER_GPU_SMOKE` — opts a GPU-only smoke test into a run (e.g. the
  llama-server CUDA smoke in `tests/test_llama_runtime.py`); never read by
  worker runtime code. Real-model GPU coverage now lives in the e2e repo's
  nightly `TestJ6` cloud journey, not a gen-worker-repo GPU lane.
- `GEN_WORKER_FORBID_CPU_OFFLOAD` — a DEV-BOX tripwire. **pgw#929 made the
  documented contract true.** It previously had exactly one read site,
  `benchmarks/swap_latency.py::check_on_pod()` (that benchmark was deleted by
  pgw#883), and affected nothing else — while the workspace `CLAUDE.md` told
  operators it "makes gen-worker raise on any CPU-touching placement". It now
  refuses at the real placement boundary, which is its only read site:
  `models/memory.py` raises `CpuOffloadForbidden` before
  `enable_model_cpu_offload` and `enable_sequential_cpu_offload`. Set it on a
  control-plane box and any CPU-offloading
  placement refuses. It is a tripwire, not configuration — it carries no
  behaviour of its own. `GEN_WORKER_HOST_MOVE_GUARD` (above) remains the
  separate, always-on `Module.to`/`.cpu` guard.

## Internal plumbing (raw env)

- `PYTORCH_CUDA_ALLOC_CONF` — `entrypoint.py` setdefaults it at import, but
  `env_seal.scrub_env()` then DELETES every `PYTORCH*` var during
  `establish()`, and `CANONICAL_CONFIG` does not re-impose it. Neither the
  default nor an operator override survives to the allocator. Treat this as
  vestigial, not as a knob.
- `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` — both READ and WRITTEN, and
  by more than one module: written by `compile_cache.py`, `entrypoint.py`
  (deliberately re-set after the env seal scrubs `TORCH*`),
  and `aot_compile_pool.py`; read back by `compile_cache.py` to latch
  inductor/triton onto verified seeded dirs (children inherit).
- `GEN_WORKER_HOST_MOVE_GUARD` — the actual CPU-placement guard
  (`host_move_guard.py`). Patches `torch.nn.Module.to`/`.cpu` and raises
  `HostRamMoveRefusedError` for moves ≥ 1 GiB that exceed the cgroup RAM
  budget. **ON by default; `=0` disables it** (inverted polarity — do not
  confuse this with `GEN_WORKER_FORBID_CPU_OFFLOAD`, below).
- `GEN_WORKER_URL_FETCH_ALLOWED_HOSTS` — deployment-wide egress bound
  (`url_fetch.py`). When set, no fetch may leave those hosts; empty means no
  bound.
- `GEN_WORKER_INTERNAL_OBJECT_HOSTS` — hosts whose resolver-minted URLs bypass
  the SSRF policy (`input_assets.py`).
- `GEN_WORKER_SUPERVISOR` (`=0` disables the OOM-reporting supervisor fork),
  `GEN_WORKER_COMPUTE_UID` (privilege-drop target uid),
  `GEN_WORKER_EAGER_FIRST_BOOT`, `GEN_WORKER_LOCAL_CELLS_DIR`,
  `GEN_WORKER_NATIVE_KERNELS`, `GEN_WORKER_AOT_*`,
  `GEN_WORKER_MINT_*`, `RUNPOD_PROVIDER` — further raw reads. **This list is no
  longer the source of truth**: `scripts/config_reads_allowlist.txt` is, and it
  is complete by construction because CI fails on any site missing from it.
- `NCCL_NVLS_ENABLE` — NOT a knob. `parallel/group.py` overwrites it to `0`
  unconditionally immediately before communicator creation (pgw#929): NVLS
  multicast cannot be bound in our containers and every Ulysses all-to-all dies
  with `ncclUnhandledCudaError` / CUDA 401. The image/operator override was
  deleted; a future collective that can use NVLS needs a measured capability
  and its own issue.
- `PODGUARD_STATE` — an **external watchdog adapter**, not config. Its producer
  is `podguard.arm()`, which runs before this process exists on pods podguard
  rents, so it can be neither argv nor a `Settings` field. pgw#929 owns its
  validation permanently: `aot_mint.podguard_status()` reports
  `armed` / `not_present` / `invalid`, so a set-but-unusable path is no longer
  the same silent no-op as a hub-created pod that legitimately has none.
- `GEN_WORKER_LOCAL_OUTPUT_DIR`, `USER` — cozy-local app plumbing / login
  fallback (`cli/local_context.py`).
- `COZY_HTTP_CONNECT_TIMEOUT_S` / `COZY_HTTP_READ_TIMEOUT_S` — http timeout
  floors, per-call by design so tests can tune them (`net.py`, gw#456).
- `COZY_CIVITAI_DOWNLOAD_ATTEMPTS`, `COZY_CLONE_DOWNLOAD_ATTEMPTS` —
  per-call test-tunable retry counts (`models/download.py`,
  `convert/ingest.py`).
- `COZY_CONVERT_WORKDIR` / `_SCRATCH_TTL_S` —
  convert-job scratch knobs set by the invoking harness (`convert/clone.py`).
  `COZY_CONVERT_RETAIN_WORKDIR` was **deleted** in pgw#929: retaining a failed
  job's scratch is a debugging action against one run, not a deployment mode,
  and as an env it could only ever be set fleet-wide and forgotten.
  Clone disk admission itself is derived from the resolved source and output
  operations and cannot be weakened by an environment override. The preflight
  covers plan-known files; repackage tools fail normally if they later fetch
  missing base components that exceed the remaining disk.
