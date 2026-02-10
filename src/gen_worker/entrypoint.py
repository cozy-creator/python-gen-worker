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
import importlib.metadata as md
from pathlib import Path
from typing import List, Optional

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


# Configuration from environment
SCHEDULER_ADDR = os.getenv("SCHEDULER_ADDR", "localhost:8080")
SCHEDULER_ADDRS = os.getenv("SCHEDULER_ADDRS", "")
SEED_ADDRS = [addr.strip() for addr in SCHEDULER_ADDRS.split(",") if addr.strip()]

WORKER_ID = os.getenv("WORKER_ID", "").strip()
WORKER_JWT = os.getenv("WORKER_JWT", "").strip()
USE_TLS_ENV = os.getenv("USE_TLS")
RECONNECT_DELAY = int(os.getenv("RECONNECT_DELAY", "5"))
MAX_RECONNECT_ATTEMPTS = int(os.getenv("MAX_RECONNECT_ATTEMPTS", "0"))

def _normalize_grpc_addr(addr: str) -> tuple[str, bool]:
    """Normalize scheduler address strings for grpc.{insecure,secure}_channel.

    Accepts:
      - host:port
      - grpc://host:port
      - grpcs://host:port
      - http(s)://host:port
    Returns:
      - normalized host:port
      - inferred tls bool (based on scheme or :443)
    """
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
    # No scheme: infer from port 443.
    tls = a.endswith(":443")
    return a, tls

SCHEDULER_ADDR, _ADDR_TLS = _normalize_grpc_addr(SCHEDULER_ADDR)
SEED_ADDRS = [(_normalize_grpc_addr(a)[0]) for a in SEED_ADDRS]

if USE_TLS_ENV is None:
    # Auto mode: use TLS when the primary scheduler address looks like a public TLS endpoint (:443 or grpcs://).
    USE_TLS = _ADDR_TLS
else:
    USE_TLS = USE_TLS_ENV.lower() in ("true", "1", "t")


if __name__ == "__main__":
    _dev_validate_gen_worker_version()

    # Load manifest if available (baked in at build time)
    manifest = load_manifest()

    # Determine user modules: baked discovery manifest is required in the Dockerfile-first contract.
    user_modules: list[str] = []
    if manifest:
        user_modules = get_modules_from_manifest(manifest)
        logger.info("Loaded manifest from %s with %d functions", MANIFEST_PATH, len(manifest.get("functions", [])))

    logger.info("Starting worker...")
    logger.info("  Scheduler Address: %s", SCHEDULER_ADDR)
    if SEED_ADDRS:
        logger.info("  Scheduler Seeds: %s", SEED_ADDRS)
    logger.info("  User Function Modules: %s", user_modules)
    logger.info("  Worker ID: %s", WORKER_ID or "(from JWT)")
    logger.info("  Use TLS: %s", USE_TLS)
    logger.info("  Reconnect Delay: %ss", RECONNECT_DELAY)
    logger.info("  Max Reconnect Attempts: %s", MAX_RECONNECT_ATTEMPTS or "Infinite")

    if not user_modules:
        logger.error(
            "No user function modules found. A baked manifest at /app/.cozy/manifest.json is required.\n"
            "Your Dockerfile should run discovery at build time:\n"
            "  RUN mkdir -p /app/.cozy && python -m gen_worker.discover > /app/.cozy/manifest.json"
        )
        sys.exit(1)

    if not WORKER_JWT:
        logger.error("WORKER_JWT is required (worker-connect JWT). Refusing to start unauthenticated worker.")
        sys.exit(1)

    try:
        worker = Worker(
            scheduler_addr=SCHEDULER_ADDR,
            scheduler_addrs=SEED_ADDRS,
            user_module_names=user_modules,
            worker_id=WORKER_ID or None,
            worker_jwt=WORKER_JWT,
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
