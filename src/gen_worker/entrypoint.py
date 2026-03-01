"""
Worker entrypoint module.

This is the main entry point for running a Cozy worker. It loads the manifest,
discovers user functions, and starts the worker loop.

Usage:
    python -m gen_worker.entrypoint
"""

import importlib.metadata as md
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cache_paths import worker_model_cache_dir
from .cozy_toml import constraint_satisfied, load_cozy_toml

try:
    from .worker import Worker
except ImportError as e:
    print(f"Error importing Worker: {e}", file=sys.stderr)
    print("Please ensure the gen_worker package is installed.", file=sys.stderr)
    sys.exit(1)

MANIFEST_PATH = Path("/app/.cozy/manifest.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("WorkerEntrypoint")


def _normalize_grpc_addr(addr: str) -> tuple[str, bool]:
    """Normalize scheduler address strings for grpc.{insecure,secure}_channel."""
    a = (addr or "").strip()
    if not a:
        return "", False
    lower = a.lower()
    if lower.startswith("grpcs://"):
        return a[len("grpcs://"):].strip(), True
    if lower.startswith("grpc://"):
        return a[len("grpc://"):].strip(), False
    if lower.startswith("https://"):
        return a[len("https://"):].strip(), True
    if lower.startswith("http://"):
        return a[len("http://"):].strip(), False
    tls = a.endswith(":443")
    return a, tls


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
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
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


def _dev_validate_gen_worker_version() -> None:
    """
    Dev-only guardrail.

    If COZY_MANIFEST_PATH points at a cozy.toml, verify the locally installed
    gen-worker version satisfies cozy.toml's gen_worker constraint.
    """
    manifest_path_str = os.getenv("COZY_MANIFEST_PATH", "").strip()
    if not manifest_path_str:
        return
    p = Path(manifest_path_str)
    if not p.exists():
        return
    try:
        cozy = load_cozy_toml(p)
        constraint = cozy.gen_worker
    except Exception as e:
        logger.warning("Failed to parse cozy.toml for dev runtime validation (%s): %s", p, e)
        return
    try:
        installed = md.version("gen-worker")
    except Exception:
        installed = ""
    if not installed:
        logger.warning("Dev validation skipped: could not determine installed gen-worker version")
        return
    if not constraint_satisfied(constraint, installed):
        logger.error(
            "Installed gen-worker version %s does not satisfy cozy.toml gen_worker constraint %r (%s).",
            installed,
            constraint,
            p,
        )
        sys.exit(2)


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
    """
    Validate model cache directory writeability before worker startup.

    - Primary: ${TENSORHUB_CACHE_DIR}/cas (default TENSORHUB_CACHE_DIR=~/.cache/tensorhub)
    - Optional local cache: WORKER_LOCAL_MODEL_CACHE_DIR (if set)
    """
    default_primary = str(worker_model_cache_dir())
    primary = default_primary
    local_cache = (os.getenv("WORKER_LOCAL_MODEL_CACHE_DIR", "") or "").strip()

    _log_startup_phase(
        "cache_preflight_started",
        status="starting",
        primary_cache_dir=primary,
        local_cache_dir=local_cache or None,
        tensorhub_cache_dir=os.getenv("TENSORHUB_CACHE_DIR", "~/.cache/tensorhub"),
    )

    ok, details = _check_cache_path("TENSORHUB_CAS_DIR", primary)
    effective_primary = primary
    if not ok:
        raise RuntimeError(
            "worker cache preflight failed for tensorhub CAS path "
            f"{primary} ({details.get('exception_class')}: {details.get('exception_message')}). "
            "Fix volume permissions/ownership or set TENSORHUB_CACHE_DIR to a writable tensorhub cache root."
        )

    effective_local = local_cache
    if local_cache:
        ok_local, local_details = _check_cache_path("WORKER_LOCAL_MODEL_CACHE_DIR", local_cache)
        if not ok_local:
            raise RuntimeError(
                "worker cache preflight failed for WORKER_LOCAL_MODEL_CACHE_DIR="
                f"{local_cache} ({local_details.get('exception_class')}: {local_details.get('exception_message')}). "
                "Fix path permissions or unset WORKER_LOCAL_MODEL_CACHE_DIR."
            )

    return {
        "model_cache_dir": effective_primary,
        "local_model_cache_dir": effective_local,
    }


def _run_main() -> int:
    _dev_validate_gen_worker_version()
    _log_startup_phase("boot", status="starting")
    worker_mode = (os.getenv("WORKER_MODE") or "inference").strip().lower()
    if worker_mode not in {"inference", "trainer"}:
        logger.error("Invalid WORKER_MODE=%r (expected 'inference' or 'trainer')", worker_mode)
        return 1
    if worker_mode == "trainer":
        _log_startup_phase("trainer_mode_selected", status="ok", worker_mode=worker_mode)
        try:
            from .trainer.runtime import run_training_runtime_from_env

            return int(run_training_runtime_from_env())
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

    scheduler_addr_raw = (os.getenv("SCHEDULER_PUBLIC_ADDR") or "").strip()
    scheduler_addrs_raw = os.getenv("SCHEDULER_ADDRS", "")
    worker_id = os.getenv("WORKER_ID", "").strip()
    worker_jwt = os.getenv("WORKER_JWT", "").strip()
    use_tls_env = os.getenv("USE_TLS")
    reconnect_delay = int(os.getenv("RECONNECT_DELAY", "5") or "5")
    max_reconnect_attempts = int(os.getenv("MAX_RECONNECT_ATTEMPTS", "0") or "0")

    if not scheduler_addr_raw:
        logger.error("SCHEDULER_PUBLIC_ADDR is required (scheduler dial address). Refusing to start worker.")
        return 1

    seed_addrs = [addr.strip() for addr in scheduler_addrs_raw.split(",") if addr.strip()]
    scheduler_addr, inferred_tls = _normalize_grpc_addr(scheduler_addr_raw)
    seed_addrs = [_normalize_grpc_addr(a)[0] for a in seed_addrs]
    if use_tls_env is None:
        use_tls = inferred_tls
    else:
        use_tls = use_tls_env.lower() in ("true", "1", "t")

    logger.info("Starting worker...")
    logger.info("  Scheduler Address: %s", scheduler_addr)
    if seed_addrs:
        logger.info("  Scheduler Seeds: %s", seed_addrs)
    logger.info("  User Function Modules: %s", user_modules)
    logger.info("  Worker ID: %s", worker_id or "(from JWT)")
    logger.info("  Use TLS: %s", use_tls)
    logger.info("  Reconnect Delay: %ss", reconnect_delay)
    logger.info("  Max Reconnect Attempts: %s", max_reconnect_attempts or "Infinite")
    logger.info("  Model Cache Dir: %s", cache_cfg["model_cache_dir"])
    if cache_cfg["local_model_cache_dir"]:
        logger.info("  Local Model Cache Dir: %s", cache_cfg["local_model_cache_dir"])

    if not user_modules:
        logger.error(
            "No user function modules found. A baked manifest at /app/.cozy/manifest.json is required.\n"
            "Your Dockerfile should run discovery at build time:\n"
            "  RUN mkdir -p /app/.cozy && python -m gen_worker.discover > /app/.cozy/manifest.json"
        )
        return 1

    if not worker_jwt:
        logger.error("WORKER_JWT is required (worker-connect JWT). Refusing to start unauthenticated worker.")
        return 1

    try:
        worker = Worker(
            scheduler_addr=scheduler_addr,
            scheduler_addrs=seed_addrs,
            user_module_names=user_modules,
            worker_id=worker_id or None,
            worker_jwt=worker_jwt,
            use_tls=use_tls,
            reconnect_delay=reconnect_delay,
            max_reconnect_attempts=max_reconnect_attempts,
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
