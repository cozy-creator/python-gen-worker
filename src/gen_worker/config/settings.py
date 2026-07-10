"""Worker `Settings` struct — the canonical typed config for the worker process.

Loaded exactly once at `entrypoint._run_main()` via `load_settings()`, then
passed by reference to every consumer. No module in `gen_worker` reads
`os.environ` or `os.getenv` directly — if a value isn't on Settings, add a
field here.

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
    # Optional PEM CA bundle for the orchestrator gRPC TLS connection.
    # Empty = system roots.
    grpc_ca_bundle: str = ""

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
