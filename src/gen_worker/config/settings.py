"""Worker `Settings` struct — the canonical typed config for the worker process.

Loaded exactly once at `entrypoint._run_main()` via `load_settings()`, then
passed by reference to every consumer. Orchestrator-injected pod config and
anything an operator needs to override at deploy time belongs here.

NOT everything reads through Settings: a handful of modules that also work
as standalone libraries/CLIs outside a full worker bring-up (`net.py`,
`convert/ingest.py`, `convert/clone.py`, `cli/run.py`, `compile_cache.py`)
read a few env vars directly with their own defaults — those are
self-contained and tested independently of the Settings loader (e.g.
`COZY_HTTP_*_TIMEOUT_S`, `COZY_CONVERT_*`, `GEN_WORKER_COMPILE_*`). New
pod-launch config should still go on Settings;
a raw read is the exception for library-standalone knobs, not the norm.

Built on msgspec.Struct (already a worker dep) instead of pydantic-settings to
avoid pulling in pydantic. The source-loader layering (env → .env → secrets dir
→ yaml → defaults) lives next to this in `loader.py`.
"""
import msgspec


class Settings(msgspec.Struct, frozen=True, kw_only=True):
    """Worker-process configuration. Loaded once at startup, passed by reference.

    All fields default to their zero value so the loader can omit any field
    that isn't explicitly set in any source — msgspec.convert applies struct
    defaults for missing keys. Required-ness is enforced by callers (e.g.
    `Worker.__init__` raises if `orchestrator_public_addr` is empty).
    """

    # HuggingFace credentials/cache. Universal across endpoints — read by the
    # huggingface_hub library and our HF download helpers.
    hf_token: str = ""
    hf_home: str = ""

    # Service endpoints injected by gen-orchestrator at pod launch.
    tensorhub_public_url: str = ""
    # Public address of the orchestrator router (ALPN-multiplexed HTTP + gRPC).
    # Single shared value across the cluster — the router resolves us to the
    # per-release lease owner. gen-orchestrator issue #317.
    orchestrator_public_addr: str = ""

    # Per-pod worker identity (orchestrator-injected).
    worker_id: str = ""
    # Path to the discovery manifest (endpoint.lock). Default is the baked
    # container location; non-container runs (e2e, bare-metal dev) override it.
    endpoint_lock_path: str = "/app/.tensorhub/endpoint.lock"
    worker_jwt: str = ""

    # Runtime introspection (set by the RunPod runtime; not configuration).
    runpod_pod_id: str = ""

    # Immutable image provenance stamped by Tensorhub from the release image
    # variant it selected for this pod.
    worker_image_digest: str = ""  # WORKER_IMAGE_DIGEST

    # tensorhub access for standalone clients (run/serve/prefetch). Production
    # workers get orchestrator-resolved manifests and never dial these.
    tensorhub_url: str = ""        # TENSORHUB_URL
    tensorhub_token: str = ""      # TENSORHUB_TOKEN
    # CAS/cache roots. cache_dir moves the whole tensorhub cache off /tmp
    # (cozy local persists weights across reboots); cas_dir points the
    # standalone CLI (`gen-worker run`) at an explicit CAS root — also the
    # test-isolation knob for the CLI resolver tests (real consumer:
    # models/provision.py::resolve_local_path).
    tensorhub_cache_dir: str = ""  # TENSORHUB_CACHE_DIR
    tensorhub_cas_dir: str = ""    # TENSORHUB_CAS_DIR
    # th#850 managed-tier ruling (gw#599): endpoint-scoped datacenter-warm
    # fill source (RunPod network volume mount), tried before R2. Never the
    # CAS root itself — see models/cache_paths.py::tensorhub_fill_source_dir.
    tensorhub_fill_source_dir: str = ""  # TENSORHUB_FILL_SOURCE_DIR

    # Civitai provider credential (CIVITAI_API_KEY, alias CIVITAI_TOKEN).
    civitai_api_key: str = ""

    # C2PA Content Credentials signing (th#714, EU AI Act Art. 50).
    # Signing is ON iff cert_path is set: every generated media asset gets a
    # signed provenance manifest at save time (content_credentials.py).
    # cert_path = PEM chain (leaf first), key_path = PKCS#8 PEM private key.
    c2pa_cert_path: str = ""  # GEN_WORKER_C2PA_CERT_PATH
    c2pa_key_path: str = ""   # GEN_WORKER_C2PA_KEY_PATH
    c2pa_alg: str = "es256"   # GEN_WORKER_C2PA_ALG (COSE alg matching the cert key)
    c2pa_ta_url: str = ""     # GEN_WORKER_C2PA_TA_URL (optional RFC3161 TSA)
