# populate_model_sizes.py
import asyncio
import gc
import logging
import os
import sys
import time
from typing import Optional, Dict, Any

import torch
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# --- Project Imports ---
try:
    from src.pytorch_worker.model_memory_manager import ModelMemoryManager
    from src.pytorch_worker.utils.model_downloader import ModelManager as ModelDownloader, ModelSource
    from src.pytorch_worker.utils.config import load_config, set_config, get_config, RuntimeConfig
    from src.pytorch_worker.utils.globals import set_available_torch_device
    from src.pytorch_worker.utils.db.database import get_db_connection, close_db_connection
    from src.pytorch_worker.utils.repository import get_pipeline_defs, PipelineDef
except ImportError as e:
    print(f"ImportError: {e}. Critical project modules could not be imported.")
    print("Please verify PYTHONPATH, script location, or package installation.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("PopulateModelSizes")

load_dotenv()
DB_DSN = os.getenv("DB_DSN")


async def ensure_model_and_components_downloaded(
    downloader: ModelDownloader, model_id: str, p_def: PipelineDef
):
    """
    Ensures the main model and all its specified components with unique sources are downloaded.
    """

    model_config_for_downloader = {
        "source": p_def.source,
        "class_name": p_def.class_name,
        "components": p_def.components,
        # Add other fields if your downloader.is_downloaded relies on them from a model_config dict
    }
    
    async with downloader: # Manages downloader's aiohttp session
        # 1. Download the main model source
        if p_def.source:
            logger.info(f"Ensuring main model source for '{model_id}' ({p_def.source}) is downloaded...")
            already_downloaded_main, _ = await downloader.is_downloaded(model_id, model_config=model_config_for_downloader)

            if not already_downloaded_main:
                logger.info(f"Main model source for '{model_id}' ({p_def.source}) not found or incomplete. Downloading...")
                main_model_src_obj = ModelSource(p_def.source)
                await downloader.download_model(model_id, main_model_src_obj)
                logger.info(f"Main model source for '{model_id}' download attempt complete.")
            else:
                logger.info(f"Main model source for '{model_id}' already downloaded.")
        else:
            logger.warning(f"Main model '{model_id}' has no source defined in its pipeline_def. Skipping main download.")

        # 2. Download components if they have their own sources
        if p_def.components and isinstance(p_def.components, dict):
            for comp_name, comp_details in p_def.components.items():
                comp_source_str = None
                if isinstance(comp_details, dict): # New structure where comp_details is a dict
                    comp_source_str = comp_details.get("source")
                elif isinstance(comp_details, str): # Old structure where comp_details might be the source string
                    comp_source_str = comp_details # This case might need review based on your actual PipelineDef.components structure
                
                if comp_source_str:
                    # Create a unique ID for the component if needed by downloader, or use comp_name
                    # For simplicity, let's assume component downloads can be keyed by their source or a derived ID
                    component_download_id = f"{model_id}::{comp_name}"
                    logger.info(f"Ensuring component '{comp_name}' for '{model_id}' ({comp_source_str}) is downloaded...")
                    comp_src_obj = ModelSource(comp_source_str)
                    try:
                        await downloader.download_model(component_download_id, comp_src_obj)
                        logger.info(f"Component '{comp_name}' for '{model_id}' download check/attempt complete.")
                    except Exception as comp_e:
                        logger.error(f"Failed to download component '{comp_name}' ({comp_source_str}) for model '{model_id}': {comp_e}")
                # else:
                #     logger.debug(f"Component '{comp_name}' for '{model_id}' has no distinct source. Assuming part of main model.")


async def get_model_vram_footprint_bytes(
    mmm: ModelMemoryManager, model_id: str, p_def_for_sizing: PipelineDef, downloader: ModelDownloader
) -> Optional[int]:
    """
    Downloads (model and components), loads a model to GPU, measures VRAM footprint, and unloads.
    """
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Cannot measure VRAM footprint.")
        return None

    target_device = torch.device("cuda:0")
    set_available_torch_device(target_device)

    pipeline_instance = None

    try:
        # 1. Ensure model and its components are downloaded
        await asyncio.run(ensure_model_and_components_downloaded(downloader, model_id, p_def_for_sizing))

        # 2. Measure VRAM footprint
        torch.cuda.empty_cache()
        gc.collect()
        vram_before_load_bytes = torch.cuda.memory_allocated(target_device)

        logger.info(f"Attempting to load '{model_id}' to {target_device} for VRAM sizing...")
        pipeline_instance = await mmm.load(model_id=model_id)

        if not pipeline_instance:
            logger.error(f"Failed to load '{model_id}' to GPU for VRAM sizing. MMM.load returned None.")
            return None

        vram_after_load_bytes = torch.cuda.memory_allocated(target_device)
        footprint_bytes = vram_after_load_bytes - vram_before_load_bytes

        if footprint_bytes <= 0:
            logger.warning(
                f"VRAM increase for '{model_id}' was {footprint_bytes}. "
                f"Using total VRAM allocated post-load ({vram_after_load_bytes} bytes) as estimate."
            )
            footprint_bytes = vram_after_load_bytes

        # Ensure a non-zero positive value if loaded successfully
        if pipeline_instance and footprint_bytes <= 0:
            logger.warning(f"Footprint for '{model_id}' calculated as <= 0 ({footprint_bytes}), but pipeline loaded. "
                           f"This is unusual. Defaulting to a minimal placeholder size if needed or re-evaluating logic.")
            # To avoid storing 0 for a loaded model, you might assign a minimal plausible size,
            # or simply trust vram_after_load_bytes in this case. The current logic already does this.

        logger.info(f"'{model_id}' VRAM footprint: {footprint_bytes / (1024**3):.2f} GB ({footprint_bytes} bytes)")
        return footprint_bytes

    except Exception as e:
        logger.exception(f"Error during VRAM sizing for model '{model_id}': {e}")
        return None
    finally:
        logger.info(f"Unloading '{model_id}' from GPU after VRAM sizing...")
        if pipeline_instance:
            try:
                await mmm.unload(model_id)
            except Exception as unload_err:
                logger.error(f"Error during explicit unload of '{model_id}': {unload_err}")
        else:
             try: # Attempt unload by ID even if pipeline_instance is None, mmm might have internal state
                  await mmm.unload(model_id)
             except Exception:
                  pass

        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Cleaned up GPU memory after '{model_id}'.")


async def main_sizing_logic():
    if not DB_DSN:
        logger.error("DB_DSN environment variable not set. Exiting.")
        return

    if not torch.cuda.is_available():
        logger.error("CUDA (GPU) not available. This script requires a GPU for VRAM sizing. Exiting.")
        return

    try:
        app_config = load_config()
        set_config(app_config)
        logger.info("Successfully loaded application configuration using load_config().")
        logger.info(f"  Using models_path: {get_config().models_path}")
        logger.info(f"  Using home_dir: {get_config().home_dir}")
    except Exception as e:
        logger.warning(
            f"Could not load standard application configuration via load_config(): {e}. "
            "Falling back to a minimal mock configuration. Paths might be default."
        )
        mock_config = RuntimeConfig(
            home_dir=os.path.expanduser("~/.cozy-creator"),
            models_path=os.path.join(os.path.expanduser("~/.cozy-creator"), "models"),
        )
        set_config(mock_config)
        logger.info("Initialized with a minimal RuntimeConfig for script operation.")
        logger.info(f"  Using models_path (mock): {get_config().models_path}")
        logger.info(f"  Using home_dir (mock): {get_config().home_dir}")

    db_conn = None
    model_vram_sizes_map = {}

    try:
        db_conn = get_db_connection()
        if not db_conn:
            logger.error("Failed to connect to the database.")
            return

        all_model_names_from_db = []
        with db_conn.cursor() as cur:
            cur.execute("SELECT name FROM pipeline_defs") # Fetch only names first
            rows = cur.fetchall()
            for row in rows:
                all_model_names_from_db.append(row['name'])

        if not all_model_names_from_db:
            logger.info("No models found in pipeline_defs table to process.")
            return

        logger.info(f"Found {len(all_model_names_from_db)} models to process for VRAM sizing.")

        # Fetch full PipelineDef objects for these names
        # This ensures we have p_def.source and p_def.components for downloader
        pipeline_defs_to_process = get_pipeline_defs(db_conn, all_model_names_from_db)

        mmm = ModelMemoryManager()
        downloader = ModelDownloader()

        for p_def in pipeline_defs_to_process: # Iterate through full PipelineDef objects
            model_id = p_def.name
            # model_source_str = p_def.source # This is now handled by ensure_model_and_components_downloaded

            logger.info(f"Processing model for VRAM sizing: {model_id} (Main Source: {p_def.source})")

            footprint_bytes = await get_model_vram_footprint_bytes(mmm, model_id, p_def, downloader) # Pass full p_def

            if footprint_bytes is not None and footprint_bytes > 0:
                model_vram_sizes_map[model_id] = footprint_bytes
            else:
                logger.warning(f"Could not determine VRAM footprint for '{model_id}'. "
                               "Skipping database update for this model.")
            
            time.sleep(2)

        if model_vram_sizes_map:
            logger.info("Updating database with calculated model VRAM sizes...")
            with db_conn.cursor() as cur:
                for model_id, size_bytes in model_vram_sizes_map.items():
                    logger.info(f"Updating '{model_id}' with VRAM size: {size_bytes} bytes")
                    cur.execute(
                        "UPDATE pipeline_defs SET estimated_size_bytes = %s WHERE name = %s",
                        (size_bytes, model_id)
                    )
                db_conn.commit()
            logger.info("Database update complete.")
        else:
            logger.info("No model VRAM sizes were calculated to update in the database.")

    except (Exception, psycopg2.Error) as error:
        logger.exception(f"Error during script execution: {error}")
        if db_conn:
            db_conn.rollback()
    finally:
        if db_conn:
            close_db_connection()
            logger.info("Database connection closed.")

def get_config_safe() -> Optional[RuntimeConfig]:
    try:
        return get_config()
    except ValueError:
        return None

if __name__ == "__main__":
    asyncio.run(main_sizing_logic())