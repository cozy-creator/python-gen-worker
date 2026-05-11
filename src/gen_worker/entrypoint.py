"""
Worker entrypoint module.

This is the main entry point for running a Cozy worker. It loads the manifest,
discovers user functions, and starts the worker loop.

Usage:
    python -m gen_worker.entrypoint
"""

import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import msgspec

from .config import Settings, load_settings
from .models.cache_paths import tensorhub_cas_dir
try:
    from .worker import Worker
except ImportError as e:
    print(f"Error importing Worker: {e}", file=sys.stderr)
    print("Please ensure the gen_worker package is installed.", file=sys.stderr)
    sys.exit(1)

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


def _log_worker_fatal(phase: str, exc: BaseException, *, exit_code: int) -> None:
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


def load_manifest() -> Optional[dict]:
    """Load the function manifest if it exists (baked in at build time)."""
    if not MANIFEST_PATH.exists():
        return None
    try:
        raw = MANIFEST_PATH.read_text(encoding="utf-8")
        manifest = msgspec.toml.decode(raw)
        if not isinstance(manifest, dict):
            raise ValueError("endpoint.lock must decode to a TOML table")
        return manifest
    except Exception as e:
        logger.warning("Failed to load manifest from %s: %s", MANIFEST_PATH, e)
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
        settings = load_settings()
    except Exception as e:
        logger.exception("Failed to load worker settings: %s", e)
        _log_worker_fatal("settings_load", e, exit_code=1)
        return 1
    if settings.worker_mode == "trainer":
        _log_startup_phase("trainer_mode_selected", status="ok", worker_mode=settings.worker_mode)
        try:
            from .trainer.runtime import run_training_runtime

            return int(run_training_runtime(settings))
        except Exception as e:
            logger.exception("Trainer runtime failed unexpectedly: %s", e)
            _log_worker_fatal("trainer_runtime", e, exit_code=1)
            return 1

    manifest = load_manifest()
    user_modules: List[str] = []
    if manifest:
        user_modules = get_modules_from_manifest(manifest)
        _log_startup_phase(
            "manifest_loaded",
            status="ok",
            manifest_path=str(MANIFEST_PATH),
            function_count=len(manifest.get("functions", [])),
            module_count=len(user_modules),
        )
    else:
        _log_startup_phase(
            "manifest_loaded",
            status="error",
            level=logging.ERROR,
            manifest_path=str(MANIFEST_PATH),
            reason="missing_or_invalid_manifest",
        )

    try:
        cache_cfg = _preflight_cache_dirs()
    except Exception as e:
        _log_worker_fatal("cache_preflight", e, exit_code=1)
        logger.error(str(e))
        return 1

    reconnect_delay = 0.1
    max_reconnect_attempts = 0  # 0 = infinite
    lb_only_retries = True

    if not settings.orchestrator_public_grpc_addr:
        logger.error("Settings.orchestrator_public_grpc_addr is empty (set ORCHESTRATOR_PUBLIC_GRPC_ADDR env). Refusing to start worker.")
        return 1

    logger.info("Starting worker...")
    logger.info("  Scheduler Address: %s", settings.orchestrator_public_grpc_addr)
    logger.info("  User Function Modules: %s", user_modules)
    logger.info("  Worker ID: %s", settings.worker_id or "(from JWT)")
    logger.info("  Reconnect Delay (base): %.3fs", reconnect_delay)
    logger.info("  Max Reconnect Attempts: %s", max_reconnect_attempts or "Infinite")
    logger.info("  LB-only retries: %s", lb_only_retries)
    logger.info("  Model Cache Dir: %s", cache_cfg["model_cache_dir"])
    if cache_cfg["local_model_cache_dir"]:
        logger.info("  Local Model Cache Dir: %s", cache_cfg["local_model_cache_dir"])

    if not user_modules:
        logger.error(
            "No user function modules found. A baked manifest at /app/.tensorhub/endpoint.lock is required.\n"
            "Your Dockerfile should run discovery at build time:\n"
            "  RUN mkdir -p /app/.tensorhub && python -m gen_worker.discovery > /app/.tensorhub/endpoint.lock"
        )
        return 1

    try:
        worker = Worker(
            settings=settings,
            user_module_names=user_modules,
            reconnect_delay=reconnect_delay,
            max_reconnect_attempts=max_reconnect_attempts,
            lb_only_retries=lb_only_retries,
            manifest=manifest,
        )
        worker.run()
        logger.info("Worker process finished gracefully.")
        return 0
    except ImportError as e:
        logger.exception(
            "Failed to import user module(s) or dependencies: %s. "
            "Make sure modules '%s' and their requirements are installed.",
            e,
            user_modules,
        )
        _log_worker_fatal("import", e, exit_code=1)
        return 1
    except Exception as e:
        logger.exception("Worker failed unexpectedly: %s", e)
        _log_worker_fatal("runtime", e, exit_code=1)
        return 1


if __name__ == "__main__":
    sys.exit(_run_main())
