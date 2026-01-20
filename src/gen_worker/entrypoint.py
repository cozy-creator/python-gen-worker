import json
import logging
import os
import sys
from pathlib import Path
from typing import cast, List, Optional

try:
    from .worker import Worker
    from .model_interface import ModelManagementInterface
except ImportError as e:
    print(f"Error importing Worker: {e}", file=sys.stderr)
    print("Please ensure the gen_worker package is installed or accessible in PYTHONPATH.", file=sys.stderr)
    sys.exit(1)

MANIFEST_PATH = Path("/app/.cozy/manifest.json")


def load_manifest() -> Optional[dict]:
    """Load the function manifest if it exists (baked in at build time)."""
    if not MANIFEST_PATH.exists():
        return None
    try:
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.getLogger("WorkerEntrypoint").warning(
            "Failed to load manifest from %s: %s", MANIFEST_PATH, e
        )
        return None


def get_modules_from_manifest(manifest: dict) -> List[str]:
    """Extract unique module names from the manifest."""
    modules = set()
    for func in manifest.get("functions", []):
        module = func.get("module")
        if module:
            modules.add(module)
    return sorted(modules)

# Optional Default Model Management Components
DMM_AVAILABLE = False
DefaultModelManager_cls = None
dmm_load_config_func = None
dmm_set_config_func = None

try:
    from .torch_manager import (
        DefaultModelManager, # The class itself
        load_config,         # The config loading utility
        set_config           # The config setting utility
        # ModelManager as DefaultModelDownloader # If you export your downloader too
    )
    DefaultModelManager_cls = DefaultModelManager
    dmm_load_config_func = load_config
    dmm_set_config_func = set_config
    # DefaultModelDownloader_cls = DefaultModelDownloader
    DMM_AVAILABLE = True
except ImportError:
    # This is not necessarily an error if user doesn't intend to use default MMM
    pass # logging will happen in main

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('WorkerEntrypoint')

if DMM_AVAILABLE:
    logger.info("DefaultModelManager components are available in this gen-worker installation.")
else:
    logger.info("DefaultModelManager components not found. Built-in dynamic model management will be unavailable "
                "unless ENABLE_DEFAULT_MODEL_MANAGER is explicitly set to true.")

# --- Configuration ---
# Read from environment variables or set defaults
SCHEDULER_ADDR = os.getenv('SCHEDULER_ADDR', 'localhost:8080')
SCHEDULER_ADDRS = os.getenv('SCHEDULER_ADDRS', '')
SEED_ADDRS = [addr.strip() for addr in SCHEDULER_ADDRS.split(',') if addr.strip()]

# Load manifest if available (baked in at build time)
MANIFEST = load_manifest()

# Determine user modules: from manifest (preferred) or USER_MODULES env var (fallback)
if MANIFEST:
    USER_MODULES = get_modules_from_manifest(MANIFEST)
    logger.info("Loaded manifest from %s with %d functions", MANIFEST_PATH, len(MANIFEST.get("functions", [])))
else:
    default_user_modules = 'functions'
    user_modules_str = os.getenv('USER_MODULES', default_user_modules)
    USER_MODULES = [mod.strip() for mod in user_modules_str.split(',') if mod.strip()]

WORKER_ID = os.getenv('WORKER_ID', "worker-1") # Optional, will be generated if None
AUTH_TOKEN = os.getenv('AUTH_TOKEN') or os.getenv('WORKER_JWT') # Optional
USE_TLS = os.getenv('USE_TLS', 'false').lower() in ('true', '1', 't')
RECONNECT_DELAY = int(os.getenv('RECONNECT_DELAY', '5'))
MAX_RECONNECT_ATTEMPTS = int(os.getenv('MAX_RECONNECT_ATTEMPTS', '0'))
ENABLE_DEFAULT_MODEL_MANAGER = os.getenv('ENABLE_DEFAULT_MODEL_MANAGER', 'false').lower() in ('true', '1', 't')


if __name__ == '__main__':

    # if sys.platform == "win32":
    #     asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy()) # Or SelectorEventLoopPolicy

    logger.info(f'Starting worker...')
    logger.info(f'  Scheduler Address: {SCHEDULER_ADDR}')
    if SEED_ADDRS:
        logger.info(f'  Scheduler Seeds: {SEED_ADDRS}')
    logger.info(f'  User Function Modules: {USER_MODULES}')
    logger.info(f'  Worker ID: {WORKER_ID or "(generated)"}')
    logger.info(f'  Use TLS: {USE_TLS}')
    logger.info(f'  Reconnect Delay: {RECONNECT_DELAY}s')
    logger.info(f'  Max Reconnect Attempts: {MAX_RECONNECT_ATTEMPTS or "Infinite"}')
    logger.info(f'  Enable Default Model Manager: {ENABLE_DEFAULT_MODEL_MANAGER}')

    if not USER_MODULES:
        logger.error("No user function modules found. Either provide a manifest at /app/.cozy/manifest.json or set the USER_MODULES environment variable.")
        sys.exit(1)

    model_manager_instance_to_pass = None

    if ENABLE_DEFAULT_MODEL_MANAGER:
        if DMM_AVAILABLE and DefaultModelManager_cls and dmm_load_config_func and dmm_set_config_func:
            logger.info("DefaultModelManager is ENABLED and AVAILABLE. Initializing...")
            try:
                # Load config (e.g., from DB/YAML) needed by DefaultModelManager and its utils
                app_cfg = dmm_load_config_func()
                dmm_set_config_func(app_cfg) # Set it globally for utils used by DMM
                logger.info("Application configuration loaded for DefaultModelManager.")
                
                model_manager_instance_to_pass = cast(ModelManagementInterface, DefaultModelManager_cls())
                logger.info("DefaultModelManager instance created.")
            except Exception as e_dmm_init:
                logger.exception(f"Failed to initialize DefaultModelManager: {e_dmm_init}. "
                                 "Proceeding without dynamic model management.")
                model_manager_instance_to_pass = None
        else:
            logger.warning("ENABLE_DEFAULT_MODEL_MANAGER is true, but DefaultModelManager components "
                           "are not fully available/imported. Proceeding without dynamic model management.")
    else:
        logger.info("ENABLE_DEFAULT_MODEL_MANAGER is false. Worker will run without built-in dynamic model management.")

    try:
        worker = Worker(
            scheduler_addr=SCHEDULER_ADDR,
            scheduler_addrs=SEED_ADDRS,
            user_module_names=USER_MODULES,
            worker_id=WORKER_ID,
            auth_token=AUTH_TOKEN,
            use_tls=USE_TLS,
            reconnect_delay=RECONNECT_DELAY,
            max_reconnect_attempts=MAX_RECONNECT_ATTEMPTS,
            model_manager=model_manager_instance_to_pass
        )
        # This blocks until the worker stops
        worker.run()
        logger.info('Worker process finished gracefully.')
        sys.exit(0)
    except ImportError as e:
        logger.exception(f"Failed to import user module(s) or dependencies: {e}. Make sure modules '{USER_MODULES}' and their requirements are installed.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Worker failed unexpectedly: {e}")
        sys.exit(1) 
