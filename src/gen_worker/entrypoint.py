import logging
import os
import sys

# Ensure the package source is potentially discoverable if running locally
# In a proper install, this might not be strictly necessary
# but helps during development if the current dir is the repo root.
# script_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.dirname(script_dir))

try:
    from .worker import Worker
except ImportError as e:
    print(f"Error importing Worker: {e}", file=sys.stderr)
    print("Please ensure the gen_worker package is installed or accessible in PYTHONPATH.", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('WorkerEntrypoint')

# --- Configuration ---
# Read from environment variables or set defaults
SCHEDULER_ADDR = os.getenv('SCHEDULER_ADDR', 'localhost:8080')

# Default user module name, can be overridden by environment variable
default_user_modules = 'functions' # A sensible default
user_modules_str = os.getenv('USER_MODULES', default_user_modules)
USER_MODULES = [mod.strip() for mod in user_modules_str.split(',') if mod.strip()]

WORKER_ID = os.getenv('WORKER_ID', "worker-1") # Optional, will be generated if None
AUTH_TOKEN = os.getenv('AUTH_TOKEN') # Optional
USE_TLS = os.getenv('USE_TLS', 'false').lower() in ('true', '1', 't')
RECONNECT_DELAY = int(os.getenv('RECONNECT_DELAY', '5'))
MAX_RECONNECT_ATTEMPTS = int(os.getenv('MAX_RECONNECT_ATTEMPTS', '0'))

if __name__ == '__main__':
    logger.info(f'Starting worker...')
    logger.info(f'  Scheduler Address: {SCHEDULER_ADDR}')
    logger.info(f'  User Function Modules: {USER_MODULES}')
    logger.info(f'  Worker ID: {WORKER_ID or '(generated)'}')
    logger.info(f'  Use TLS: {USE_TLS}')
    logger.info(f'  Reconnect Delay: {RECONNECT_DELAY}s')
    logger.info(f'  Max Reconnect Attempts: {MAX_RECONNECT_ATTEMPTS or "Infinite"}')

    if not USER_MODULES:
        logger.error("No user function modules specified. Set the USER_MODULES environment variable.")
        sys.exit(1)

    try:
        worker = Worker(
            scheduler_addr=SCHEDULER_ADDR,
            user_module_names=USER_MODULES,
            worker_id=WORKER_ID,
            auth_token=AUTH_TOKEN,
            use_tls=USE_TLS,
            reconnect_delay=RECONNECT_DELAY,
            max_reconnect_attempts=MAX_RECONNECT_ATTEMPTS
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

