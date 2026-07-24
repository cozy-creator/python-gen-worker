"""
Worker entrypoint module.

This is the main entry point for running a Cozy worker. It loads the manifest,
discovers user functions, and starts the worker loop.

Usage:
    python -m gen_worker.entrypoint
"""

import os

# Reduce CUDA allocator fragmentation for tight-VRAM single-process workers.
# Set BEFORE any module imports torch (PyTorch reads this env var only at
# first cudaMalloc; setting it later is a no-op). gen-worker entrypoint is
# the first module loaded in every worker process, so putting it here gives
# library-wide coverage across conversion / inference / training workers.
#
# `setdefault` so an operator-set value (Dockerfile ENV or docker-compose env)
# overrides. The flag uses CUDA's virtual-memory APIs to grow allocator
# segments on demand, recovering hundreds of MiB of "reserved-but-unallocated"
# headroom that the default fixed-segment allocator wastes. See:
# https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# gw#608: compile-cell portability requires the (portable) FxGraphCache to be
# the lookup surface — the AOTAutogradCache key embeds a process memory
# address (ASLR) and can never hit across pods. TORCHINDUCTOR_AUTOGRAD_CACHE
# is an `env_name_force` config read ONCE at torch import, and runtime config
# assignments are thread-local (ContextVar) in torch>=2.13, so this must be
# set here, before any torch import, to bind every thread and every
# compile-worker subprocess. See compile_cache._disable_aot_autograd_cache.
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")

# gw#640: fork the supervisor BEFORE the heavy imports below. The parent stays
# a bare interpreter (so the OOM killer picks the fat child, not the reporter)
# and outlives the worker to report WTERMSIG / cgroup oom_kill over the wire.
# In the child this returns immediately; the parent never returns from it.
if __name__ == "__main__":
    from .supervisor import supervise  # noqa: E402

    supervise()

import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import msgspec

from .config import get_settings
from .cuda_probe import CUDA_PROBE_FAILED_MARKER, probe_cuda, should_probe_cuda
from .hardware_report import report_hardware_unsuitable
from .models.cache_paths import tensorhub_cas_dir
try:
    from .worker import Worker
except ImportError as e:
    print(f"Error importing Worker: {e}", file=sys.stderr)
    print("Please ensure the gen_worker package is installed.", file=sys.stderr)
    sys.exit(1)

# Default baked container location; overridden by Settings.endpoint_lock_path
# (env ENDPOINT_LOCK_PATH) for non-container runs.
MANIFEST_PATH = Path("/app/.tensorhub/endpoint.lock")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("WorkerEntrypoint")


def _startup_payload(phase: str, status: str = "ok", **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "phase": str(phase or "").strip(),
        "status": str(status or "ok"),
        "pid": int(os.getpid()),
        "uid": int(os.getuid()) if hasattr(os, "getuid") else None,
        "gid": int(os.getgid()) if hasattr(os, "getgid") else None,
        "cwd": str(os.getcwd()),
    }
    payload.update({k: v for k, v in extra.items() if v is not None})
    return payload


def _log_startup_phase(phase: str, *, status: str = "ok", level: int = logging.INFO, **extra: Any) -> None:
    payload = _startup_payload(phase, status=status, **extra)
    try:
        logger.log(level, "worker.startup.phase %s", json.dumps(payload, separators=(",", ":"), sort_keys=True))
    except Exception:
        logger.log(level, "worker.startup.phase phase=%s status=%s", phase, status)


def _log_worker_fatal(
    phase: str,
    exc: BaseException,
    *,
    exit_code: int,
    settings: Optional[Any] = None,
) -> None:
    """Record this process's cause of death to stdout AND to the hub.

    gw#640/th#1077: stdout alone is unreachable on RunPod (no container-logs
    API), so every cloud-only crash was un-debuggable. The wire report reuses
    the HardwareUnsuitable carrier and lands as a durable pod_events row.
    """
    try:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    except Exception:
        tb = traceback.format_exc()
    payload = _startup_payload(
        "worker_fatal",
        status="error",
        phase_context=str(phase or ""),
        exception_class=type(exc).__name__,
        exception_message=str(exc),
        traceback=tb,
        exit_code=int(exit_code),
    )
    try:
        logger.error("worker.fatal %s", json.dumps(payload, separators=(",", ":"), sort_keys=True))
    except Exception:
        logger.exception("worker.fatal: %s", exc)
    try:
        from .worker_fatal import report_worker_fatal

        delivered = report_worker_fatal(settings, phase, exc, exit_code=exit_code)
        _log_startup_phase(
            "worker_fatal_report",
            status="ok" if delivered else "error",
            level=logging.INFO if delivered else logging.WARNING,
            delivered=delivered,
            phase_context=str(phase or ""),
        )
    except Exception:
        logger.warning("worker-fatal wire report raised unexpectedly", exc_info=True)


def load_manifest(path: Path = MANIFEST_PATH) -> Optional[dict]:
    """Load the function manifest if it exists (baked in at build time)."""
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        manifest = msgspec.toml.decode(raw)
        if not isinstance(manifest, dict):
            raise ValueError("endpoint.lock must decode to a TOML table")
        return manifest
    except Exception as e:
        logger.warning("Failed to load manifest from %s: %s", path, e)
        return None


def get_modules_from_manifest(manifest: dict) -> List[str]:
    """Extract unique module names from the manifest."""
    modules = set()
    for func in manifest.get("functions", []):
        module = func.get("module")
        if module:
            modules.add(module)
    return sorted(modules)


def _probe_cache_path_writable(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / f".cozy-write-probe-{os.getpid()}"
    with open(probe, "wb") as f:
        f.write(b"ok")
        f.flush()
        os.fsync(f.fileno())
    probe.unlink(missing_ok=True)


def _check_cache_path(label: str, path_str: str) -> tuple[bool, Dict[str, Any]]:
    p = Path(path_str)
    details: Dict[str, Any] = {"label": label, "path": str(p)}
    try:
        _probe_cache_path_writable(p)
        _log_startup_phase("cache_preflight_ok", status="ok", path=str(p), label=label)
        return True, details
    except Exception as e:
        details["exception_class"] = type(e).__name__
        details["exception_message"] = str(e)
        _log_startup_phase(
            "cache_preflight_failed",
            status="error",
            level=logging.ERROR,
            path=str(p),
            label=label,
            exception_class=type(e).__name__,
            exception_message=str(e),
        )
        return False, details


def _preflight_cache_dirs() -> Dict[str, str]:
    """Validate model cache directory writeability before worker startup."""
    primary = str(tensorhub_cas_dir())

    _log_startup_phase(
        "cache_preflight_started",
        status="starting",
        primary_cache_dir=primary,
    )

    ok, details = _check_cache_path("TENSORHUB_CAS_DIR", primary)
    if not ok:
        raise RuntimeError(
            "worker cache preflight failed for tensorhub CAS path "
            f"{primary} ({details.get('exception_class')}: {details.get('exception_message')}). "
            "Fix volume permissions/ownership."
        )

    return {
        "model_cache_dir": primary,
        "local_model_cache_dir": "",
    }


def _run_main() -> int:
    _log_startup_phase("boot", status="starting")
    try:
        settings = get_settings()
    except Exception as e:
        logger.exception("Failed to load worker settings: %s", e)
        _log_worker_fatal("settings_load", e, exit_code=1)
        return 1
    manifest_path = Path(settings.endpoint_lock_path or MANIFEST_PATH)
    manifest = load_manifest(manifest_path)
    user_modules: List[str] = []
    if manifest:
        user_modules = get_modules_from_manifest(manifest)
        _log_startup_phase(
            "manifest_loaded",
            status="ok",
            manifest_path=str(manifest_path),
            function_count=len(manifest.get("functions", [])),
            module_count=len(user_modules),
        )
    else:
        _log_startup_phase(
            "manifest_loaded",
            status="error",
            level=logging.ERROR,
            manifest_path=str(manifest_path),
            reason="missing_or_invalid_manifest",
        )

    try:
        cache_cfg = _preflight_cache_dirs()
    except Exception as e:
        _log_worker_fatal("cache_preflight", e, exit_code=1, settings=settings)
        logger.error(str(e))
        return 1

    # Boot-time CUDA probe (gw#529): on a GPU-needing manifest, verify the
    # device actually works BEFORE we hello the orchestrator and accept a
    # job — a busy/unavailable GPU (RunPod bad-host fault) must kill this
    # pod now, not terminal-fail a real request at model load.
    if should_probe_cuda(manifest):
        probe = probe_cuda()
        if not probe.ok:
            logger.error("%s: %s", CUDA_PROBE_FAILED_MARKER, probe.reason)
            # gw#619/th#988: dial the hub with a typed hardware-unsuitable
            # report BEFORE exiting — closes the th#986 blindness where this
            # exit was previously silent pre-hello. Best-effort/bounded: the
            # exit below happens regardless of whether the hub is reachable.
            try:
                delivered = report_hardware_unsuitable(settings, probe)
                _log_startup_phase(
                    "cuda_probe_hardware_report",
                    status="ok" if delivered else "error",
                    level=logging.INFO if delivered else logging.WARNING,
                    delivered=delivered,
                )
            except Exception:
                logger.warning("hardware-unsuitable report raised unexpectedly", exc_info=True)
            # settings=None: this path ALREADY dialed the hub with the typed
            # HardwareUnsuitable report just above — a second wire dial would
            # only duplicate it (and double the pre-exit budget).
            _log_worker_fatal("cuda_probe", RuntimeError(probe.reason), exit_code=1)
            return 1
        _log_startup_phase("cuda_probe_ok", status="ok")

    if not settings.orchestrator_public_addr:
        logger.error("Settings.orchestrator_public_addr is empty (set ORCHESTRATOR_PUBLIC_ADDR env). Refusing to start worker.")
        return 1

    # C2PA content-credential signing (th#714): ON iff a cert is configured;
    # logs a loud warning when off, refuses to start when configured-but-broken
    # (a worker that believes it signs but doesn't is a compliance hole).
    try:
        from .content_credentials import configure as _c2pa_configure

        _c2pa_configure(settings)
    except Exception as e:
        _log_worker_fatal("c2pa_configure", e, exit_code=1, settings=settings)
        logger.error(str(e))
        return 1

    logger.info("Starting worker...")
    logger.info("  Orchestrator Public Address: %s", settings.orchestrator_public_addr)
    logger.info("  User Function Modules: %s", user_modules)
    logger.info("  Worker ID: %s", settings.worker_id or "(from JWT)")
    logger.info("  Model Cache Dir: %s", cache_cfg["model_cache_dir"])
    if cache_cfg["local_model_cache_dir"]:
        logger.info("  Local Model Cache Dir: %s", cache_cfg["local_model_cache_dir"])

    if not user_modules:
        logger.error(
            "No user function modules found. A baked manifest at %s is required.\n"
            "Your Dockerfile should run discovery at build time:\n"
            "  RUN mkdir -p /app/.tensorhub && python -m gen_worker.discovery > /app/.tensorhub/endpoint.lock\n"
            "(non-container runs: set ENDPOINT_LOCK_PATH to the generated file)",
            manifest_path,
        )
        return 1

    try:
        worker = Worker(
            settings=settings,
            user_module_names=user_modules,
            manifest=manifest,
        )
        code = worker.run()
        logger.info("Worker process finished gracefully (exit=%d).", code)
        return code
    except ImportError as e:
        logger.exception(
            "Failed to import user module(s) or dependencies: %s. "
            "Make sure modules '%s' and their requirements are installed.",
            e,
            user_modules,
        )
        _log_worker_fatal("import", e, exit_code=1, settings=settings)
        return 1
    except Exception as e:
        logger.exception("Worker failed unexpectedly: %s", e)
        _log_worker_fatal("runtime", e, exit_code=1, settings=settings)
        return 1


if __name__ == "__main__":
    sys.exit(_run_main())
