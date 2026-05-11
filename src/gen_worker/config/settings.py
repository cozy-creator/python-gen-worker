"""Worker `Settings` struct — the canonical typed config for the worker process.

Loaded exactly once at `entrypoint._run_main()` via `load_settings()`, then
passed by reference to every consumer. No module in `gen_worker` reads
`os.environ` or `os.getenv` directly — if a value isn't on Settings, add a
field here.

Built on msgspec.Struct (already a worker dep) instead of pydantic-settings to
avoid pulling in pydantic. The source-loader layering (env → .env → secrets dir
→ yaml → defaults) lives next to this in `loader.py`.
"""
from typing import Literal

import msgspec


class Settings(msgspec.Struct, frozen=True, kw_only=True):
    """Worker-process configuration. Loaded once at startup, passed by reference.

    All fields default to their zero value so the loader can omit any field
    that isn't explicitly set in any source — msgspec.convert applies struct
    defaults for missing keys. Required-ness is enforced by callers (e.g.
    `Worker.__init__` raises if `orchestrator_public_grpc_addr` is empty).
    """

    # HuggingFace credentials/cache. Universal across endpoints — read by the
    # huggingface_hub library and our HF download helpers.
    hf_token: str = ""
    hf_home: str = ""

    # Service endpoints injected by gen-orchestrator at pod launch.
    tensorhub_public_url: str = ""
    orchestrator_public_grpc_addr: str = ""

    # Per-pod worker identity (orchestrator-injected).
    worker_id: str = ""
    worker_mode: Literal["inference", "trainer"] = "inference"
    worker_jwt: str = ""

    # Trainer mode: path to the job spec JSON file the trainer pod was launched
    # with. Empty when worker_mode == "inference".
    trainer_job_spec_path: str = ""

    # Runtime introspection (set by the RunPod runtime; not configuration).
    runpod_pod_id: str = ""
