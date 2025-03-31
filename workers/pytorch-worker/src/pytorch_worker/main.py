import os
import sys
import logging
import subprocess
import importlib
from gen_orchestrator import Worker
from pytorch_worker.model_memory_manager import ModelMemoryManager
from pytorch_worker.utils.config import set_config, load_config
from pytorch_worker.utils.parse_cli import parse_arguments
import dotenv

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pytorch-worker")


def install_package(package_spec: str) -> None:
    """
    Install a package based on the specification using the uv CLI.
    If the package spec contains a colon and the second part starts with http,
    use that as the index URL; otherwise, install normally.
    """
    parts = package_spec.split(":", 1)
    cmd = ["uv", "pip", "install"]
    
    if len(parts) == 2 and parts[1].strip().lower().startswith("http"):
        package_name = parts[0].strip()
        index_url = parts[1].strip()
        cmd.extend(["--index-url", index_url])
        logger.info(f"Installing {package_name} from {index_url} using uv")
        cmd.append(package_name)
    else:
        package_name = package_spec.strip()
        logger.info(f"Installing {package_name} from default index using uv")
        cmd.append(package_name)
    
    subprocess.check_call(cmd)
    
def import_and_register(worker: Worker, package_name: str, model_memory_manager: ModelMemoryManager) -> None:
    """
    Import the given package and call its register_functions(worker, model_memory_manager) function if it exists.
    This allows the package to register one or more functions with the worker and the model memory manager.
    """
    logger.info(f"Importing package '{package_name}' for function registration...")
    module = importlib.import_module(package_name)
    if hasattr(module, "register_functions"):
        logger.info(f"Registering functions from package '{package_name}'...")
        module.register_functions(worker, model_memory_manager)
    else:
        logger.warning(f"Package '{package_name}' does not expose a 'register_functions(worker, model_memory_manager)' function.")


def parse_function_packages(packages_str: str) -> list[str]:
    """
    Parse a comma-separated list of function package specifications.
    If a spec contains a colon (indicating a private index URL), return only the package name.
    """
    packages = []
    for pkg_spec in packages_str.split(","):
        pkg_spec = pkg_spec.strip()
        if not pkg_spec:
            continue

        if ":" in pkg_spec:
            pkg_name = pkg_spec.split(":", 1)[0].strip()
            packages.append(pkg_name)
        else:
            packages.append(pkg_spec)
    return packages


def main():
    logger.info("Starting pytorch-worker...")

    # Read environment variables for configuration
    scheduler_addr = os.environ.get("SCHEDULER_ADDR", "localhost:8080")
    worker_id = os.environ.get("WORKER_ID", f"pytorch-worker-{os.getpid()}")
    function_packages = os.environ.get("FUNCTION_PACKAGES", "") # TODO:consider passing function packages through cli too on startup
    

    package_list = parse_function_packages(function_packages)

    # Create a Worker instance from the gen-orchestrator library
    worker = Worker(
        scheduler_addr=scheduler_addr,
        worker_id=worker_id,
    )

    logger.info(f"Created Worker with ID '{worker_id}' connecting to '{scheduler_addr}'")

    config = load_config()
    print(f"config: {config}")

    set_config(config)

    # Instantiate the ModelMemoryManager (to handle dynamic model loading/unloading)
    model_manager = ModelMemoryManager()

    # Install and register packages if specified
    for pkg in package_list:
        try:
            install_package(pkg)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install package '{pkg}': {e}")
            continue

        try:
            import_and_register(worker, pkg, model_manager)
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


# TODO:
# - Upload to s3
