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
from pathlib import Path
from typing import List, Optional

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


def load_manifest() -> Optional[dict]:
    """Load the function manifest if it exists (baked in at build time)."""
    if not MANIFEST_PATH.exists():
        return None
    try:
        with open(MANIFEST_PATH, "r") as f:
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


# Configuration from environment
SCHEDULER_ADDR = os.getenv("SCHEDULER_ADDR", "localhost:8080")
SCHEDULER_ADDRS = os.getenv("SCHEDULER_ADDRS", "")
SEED_ADDRS = [addr.strip() for addr in SCHEDULER_ADDRS.split(",") if addr.strip()]

WORKER_ID = os.getenv("WORKER_ID", "worker-1")
AUTH_TOKEN = os.getenv("AUTH_TOKEN") or os.getenv("WORKER_JWT")
USE_TLS = os.getenv("USE_TLS", "false").lower() in ("true", "1", "t")
RECONNECT_DELAY = int(os.getenv("RECONNECT_DELAY", "5"))
MAX_RECONNECT_ATTEMPTS = int(os.getenv("MAX_RECONNECT_ATTEMPTS", "0"))


if __name__ == "__main__":
    # Load manifest if available (baked in at build time)
    manifest = load_manifest()

    # Determine user modules: from manifest (preferred) or USER_MODULES env var (fallback)
    if manifest:
        user_modules = get_modules_from_manifest(manifest)
        logger.info(
            "Loaded manifest from %s with %d functions",
            MANIFEST_PATH,
            len(manifest.get("functions", [])),
        )
    else:
        default_user_modules = "functions"
        user_modules_str = os.getenv("USER_MODULES", default_user_modules)
        user_modules = [mod.strip() for mod in user_modules_str.split(",") if mod.strip()]

    logger.info("Starting worker...")
    logger.info("  Scheduler Address: %s", SCHEDULER_ADDR)
    if SEED_ADDRS:
        logger.info("  Scheduler Seeds: %s", SEED_ADDRS)
    logger.info("  User Function Modules: %s", user_modules)
    logger.info("  Worker ID: %s", WORKER_ID or "(generated)")
    logger.info("  Use TLS: %s", USE_TLS)
    logger.info("  Reconnect Delay: %ss", RECONNECT_DELAY)
    logger.info("  Max Reconnect Attempts: %s", MAX_RECONNECT_ATTEMPTS or "Infinite")

    if not user_modules:
        logger.error(
            "No user function modules found. Either provide a manifest at "
            "/app/.cozy/manifest.json or set the USER_MODULES environment variable."
        )
        sys.exit(1)

    try:
        worker = Worker(
            scheduler_addr=SCHEDULER_ADDR,
            scheduler_addrs=SEED_ADDRS,
            user_module_names=user_modules,
            worker_id=WORKER_ID,
            auth_token=AUTH_TOKEN,
            use_tls=USE_TLS,
            reconnect_delay=RECONNECT_DELAY,
            max_reconnect_attempts=MAX_RECONNECT_ATTEMPTS,
            manifest=manifest,
        )
        worker.run()
        logger.info("Worker process finished gracefully.")
        sys.exit(0)
    except ImportError as e:
        logger.exception(
            "Failed to import user module(s) or dependencies: %s. "
            "Make sure modules '%s' and their requirements are installed.",
            e,
            user_modules,
        )
        sys.exit(1)
    except Exception as e:
        logger.exception("Worker failed unexpectedly: %s", e)
        sys.exit(1)
