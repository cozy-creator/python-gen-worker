# Environment variables

Exactly one component reads the process environment: `config/loader.py`,
which produces the typed `Settings` struct (`config/settings.py`) and passes
it around. **`settings.py` carries the env name in a comment on every field**
— that is the knob inventory; this page carries only what the code cannot
tell you.

## `TENSORHUB_FILL_SOURCE_DIR` (`tensorhub_fill_source_dir`)

th#850 managed-tier ruling: an endpoint-scoped datacenter-warm CAS mount
(a RunPod volume), checked before R2 on a blob miss and write-through warmed
from R2. **Never the CAS root** — that always stays `TENSORHUB_CACHE_DIR` /
local. tensorhub sets this only when a volume is actually attached, and the
worker **ismount-guards** it, so a plain directory can never be mistaken for
the mount.

## C2PA Content Credentials (th#714)

Every generated media asset (png/jpeg/webp/gif, mp4/mov, wav/mp3/flac/m4a)
gets a signed C2PA provenance manifest at `ctx.save_bytes`/`save_file` time —
the EU AI Act Art. 50 machine-readable AI-marking. ON iff the cert is
configured; unconfigured logs a loud startup warning and no-ops;
configured-but-broken refuses to start.

**The worker holds a cert, never a key.** Signing itself is hub-side: the
worker POSTs the claim to tensorhub's `/v1/worker/c2pa/sign` under its worker
identity, and the private key never leaves the hub.

| Env | Field | Notes |
|---|---|---|
| `GEN_WORKER_C2PA_CERT_PEM` | `c2pa_cert_pem` | inline PEM signing-cert chain, leaf first (the ON switch). **This is the mechanism actually used** — RunPod pods have no file mounts |
| `GEN_WORKER_C2PA_CERT_PATH` | `c2pa_cert_path` | file-path variant for mounted deploys. `_CERT_PEM` takes precedence when both are set |
| `GEN_WORKER_C2PA_ALG` | `c2pa_alg` | COSE alg matching the cert key (default `es256`) |
| `GEN_WORKER_C2PA_TA_URL` | `c2pa_ta_url` | optional RFC3161 timestamp-authority URL |

**There is no key field, and setting one bricks the pod.**
`GEN_WORKER_C2PA_KEY_PEM` and `GEN_WORKER_C2PA_KEY_PATH` are in
`content_credentials._REFUSED_KEY_ENVS`: if either is present in the pod
environment, `configure()` raises `C2paSigningError` at worker startup and
the worker refuses to boot. A private key in a tenant pod is the thing this
design exists to prevent, so it fails loudly rather than being ignored.

`c2pa-python` is a **core dependency**, not the `[signing]` extra — the
compliance default-ON posture must not depend on every endpoint image
remembering to ask for an extra.

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

