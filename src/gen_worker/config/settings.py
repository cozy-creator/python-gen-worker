"""Worker `Settings` struct — the canonical typed config for the worker process.

Loaded exactly once at `entrypoint._run_main()` via `load_settings()`, then
passed by reference to every consumer. Orchestrator-injected pod config and
anything an operator needs to override at deploy time belongs here.

NOT everything reads through Settings: a handful of modules that also work
as standalone libraries/CLIs outside a full worker bring-up (`net.py`,
`convert/ingest.py`, `convert/clone.py`, `cli/run.py`, `compile_cache.py`)
read a few env vars directly with their own defaults — those are
self-contained and tested independently of the Settings loader (e.g.
`COZY_HTTP_*_TIMEOUT_S`, `COZY_CONVERT_*`, `GEN_WORKER_FORBID_CPU_OFFLOAD`,
`GEN_WORKER_COMPILE_*`). New pod-launch config should still go on Settings;
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

    # Worker self-exit timeout when the orchestrator is unreachable.
    # The reconnect loop is otherwise infinite (correct for transient
    # network blips and single-replica restarts). After this many seconds
    # of NO successful connection, the worker exits cleanly so its
    # container is reaped by docker/runpod — handles stack-shutdown
    # ergonomics (compose down, machine power-off) without a separate
    # shutdown-coordination protocol. Default 600s = 10 min. gen-orchestrator
    # issue #317.
    worker_disconnected_timeout_s: int = 600

    # Provenance stamped into WorkerResources by the image build / launcher.
    # TODO(pgw#514/P4): nothing produces WORKER_IMAGE_DIGEST today (verified
    # against gen-orchestrator/tensorhub/e2e — no launcher sets it), so this
    # is always "" in production and the Go scaling-profile key that reads
    # it (profiles.go) always sees an empty digest. Kept because tensorhub
    # may start stamping it at pod-launch time; drop if that never lands.
    worker_image_digest: str = ""  # WORKER_IMAGE_DIGEST

    # tensorhub access for standalone clients (run/serve/prefetch). Production
    # workers get orchestrator-resolved manifests and never dial these.
    tensorhub_url: str = ""        # TENSORHUB_URL
    tensorhub_token: str = ""      # TENSORHUB_TOKEN
    # CAS/cache roots. cache_dir moves the whole tensorhub cache off /tmp
    # (cozy local persists weights across reboots); cas_dir points the
    # standalone CLI (`gen-worker run`) at an explicit CAS root — also the
    # test-isolation knob for tests/test_hub_resolve_and_variants.py,
    # tests/test_run_civitai.py, tests/test_cli_run.py (real consumer:
    # cli/run.py._resolve_local_path, part of the pgw#515 fork).
    tensorhub_cache_dir: str = ""  # TENSORHUB_CACHE_DIR
    tensorhub_cas_dir: str = ""    # TENSORHUB_CAS_DIR

    # Civitai provider credential (CIVITAI_API_KEY, alias CIVITAI_TOKEN).
    civitai_api_key: str = ""
