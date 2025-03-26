import os
import sys
import logging
import subprocess
import importlib
from gen_orchestrator import Worker, ActionContext

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pytorch-worker")

# def install_package(package_name: str) -> None:
#     """
#     Install a Python package using pip.
#     This is (or will be) useful when the worker needs to install packages dynamically.
#     """
#     logger.info(f"Installing package '{package_name}'...")
#     try:
#         subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
#         logger.info(f"Package '{package_name}' installed successfully.")
#     except subprocess.CalledProcessError as e:
#         logger.error(f"Failed to install package '{package_name}': {e}")
#         raise

def install_package(package_name: str) -> None:
    """
    Install a package based on the specification.
    If the package spec contains a colon and the second part starts with http,
    use that as the index URL; otherwise, install normally.
    """
    parts = package_name.split(":", 1)
    cmd = [sys.executable, "-m", "pip", "install"]
    
    if len(parts) == 2 and parts[1].strip().lower().startswith("http"):
        package_name = parts[0].strip()
        index_url = parts[1].strip()
        cmd.extend(["--index-url", index_url])
        print(f"Installing {package_name} from {index_url}")
        cmd.append(package_name)
    else:
        package_name = package_name.strip()
        print(f"Installing {package_name} from default index")
        cmd.append(package_name)
    
    subprocess.check_call(cmd)
    
def import_and_register(worker: Worker, package_name: str) -> None:
    """
    Import the given package and call its register_functions(worker) function if it exists.
    This allows the package to register one or more functions with the worker.
    """
    logger.info(f"Importing package '{package_name}' for function registration...")
    module = importlib.import_module(package_name)
    if hasattr(module, "register_functions"):
        logger.info(f"Registering functions from package '{package_name}'...")
        module.register_functions(worker)
    else:
        logger.warning(f"Package '{package_name}' does not expose a 'register_functions(worker)' function.")


def main():
    logger.info("Starting pytorch-worker...")

    # Read environment variables for configuration
    scheduler_addr = os.environ.get("SCHEDULER_ADDR", "localhost:8080")
    worker_id = os.environ.get("WORKER_ID", f"pytorch-worker-{os.getpid()}")
    function_packages = os.environ.get("FUNCTION_PACKAGES", "")

    package_list = [pkg.strip() for pkg in function_packages.split(",") if pkg.strip()]

    # Create a Worker instance from the gen-orchestrator library
    worker = Worker(
        scheduler_addr=scheduler_addr,
        worker_id=worker_id,
    )

    logger.info(f"Created Worker with ID '{worker_id}' connecting to '{scheduler_addr}'")

    # Install and register packages if specified
    for pkg in package_list:
        try:
            install_package(pkg)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install package '{pkg}': {e}")
            continue

        try:
            import_and_register(worker, pkg)
        except Exception as e:
            logger.error(f"Failed to import and register package '{pkg}': {e}")

    
    # Start the worker
    try:
        logger.info("Starting worker...")
        worker.run()
    except KeyboardInterrupt:
        logger.info("Worker interrupted, shutting down...")
    finally:
        worker.stop()
        logger.info("Worker stopped")
        
if __name__ == "__main__":
    main()
