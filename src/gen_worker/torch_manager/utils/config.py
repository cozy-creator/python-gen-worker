import os
import argparse
from typing import Optional, List, Callable, Dict, Any
from .base_types.config import RuntimeConfig, PipelineConfig
import yaml
import json
import logging

logger = logging.getLogger(__name__)

cozy_config: Optional[RuntimeConfig] = None
"""
Global configuration for the Cozy Gen-Server
"""


def config_loaded() -> bool:
    """
    Returns a boolean indicating whether the config has been loaded.
    This will return True if called within the cozy runtime, since the config is loaded at the start.
    """
    return cozy_config is not None


def load_pipeline_defs_from_db(enabled_models: List[str]) -> Dict[str, Any]:
    """
    Load pipeline definitions from the database for the enabled models.
    
    Args:
        enabled_models: List of model names to fetch from the database
        
    Returns:
        Dictionary of pipeline definitions keyed by model name
    """
    try:
        from .db.database import get_db_connection
        from .repository import get_pipeline_defs
        
        # if not enabled_models:
        #     return {}
            
        db_conn = get_db_connection()

        if not db_conn:
            logger.error("load_pipeline_defs_from_db: Could not get database connection.")
            return {} # Cannot proceed without DB connection

        names_to_query: List[str]
        if enabled_models is None or not enabled_models: # Check if None or empty list
            logger.info("load_pipeline_defs_from_db: No specific model names provided, fetching all model names from DB.")
            all_db_model_names = []
            with db_conn.cursor() as cursor: # Use a different variable name for cursor
                cursor.execute("SELECT name FROM pipeline_defs WHERE source IS NOT NULL AND source != ''") # Fetch only usable models
                rows = cursor.fetchall()
                for row in rows:
                    all_db_model_names.append(row['name'])
            
            if not all_db_model_names:
                logger.warning("load_pipeline_defs_from_db: No model names found in DB to load definitions for.")
                return {}
            names_to_query = all_db_model_names
        else:
            names_to_query = enabled_models
        
        logger.debug(f"load_pipeline_defs_from_db: Fetching definitions for models: {names_to_query}")
        
        # get_pipeline_defs returns List[PipelineDef objects]
        pipeline_def_objects = get_pipeline_defs(db_conn, names_to_query) 
        
        if not pipeline_def_objects:
            logger.warning(f"load_pipeline_defs_from_db: get_pipeline_defs returned no objects for names: {names_to_query}")
            return {}

        logger.info(f"load_pipeline_defs_from_db: Loaded {len(pipeline_def_objects)} PipelineDef objects from repository.")
        
        # Convert DB models to dictionary format
        db_pipeline_defs = {}
        for model in pipeline_def_objects:
            pipeline_def = {
                "source": model.source,
                "class_name": model.class_name,
                "custom_pipeline": model.custom_pipeline,
                "default_args": model.default_args,
                "metadata": model.metadata,
                "components": {}
            }
            
            # Convert components
            if model.components:
                for name, comp in model.components.items():
                    if isinstance(comp, dict):
                        pipeline_def["components"][name] = {
                            "class_name": comp.get("class_name", ""),
                            "source": comp.get("source", ""),
                            "kwargs": comp.get("kwargs", {})
                        }
            
            # Add prompt definitions if available
            if hasattr(model, "prompt_def") and model.prompt_def:
                if not pipeline_def.get("default_args"):
                    pipeline_def["default_args"] = {}
                pipeline_def["default_args"]["positive_prompt"] = model.prompt_def.positive_prompt
                pipeline_def["default_args"]["negative_prompt"] = model.prompt_def.negative_prompt
                
            db_pipeline_defs[model.name] = pipeline_def
            
        return db_pipeline_defs
    except Exception as e:
        logger.error(f"Error loading pipeline definitions from database: {e}")
        return {}
    

def merge_pipeline_defs(existing_defs: Dict[str, Any], incoming_defs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge pipeline definitions from different sources, similar to Go implementation.
    
    Args:
        existing_defs: Pipeline definitions from config.yaml
        incoming_defs: Pipeline definitions from database
        
    Returns:
        Merged pipeline definitions
    """
    merged_defs = existing_defs.copy()
    
    # Merge incoming defs into existing defs
    for model_id, model_def in incoming_defs.items():
        if model_id in merged_defs:
            # Update only empty fields in existing definition
            existing_def = merged_defs[model_id]
            
            if not existing_def.get("source"):
                existing_def["source"] = model_def.get("source", "")
                
            if not existing_def.get("class_name"):
                existing_def["class_name"] = model_def.get("class_name", "")
                
            if not existing_def.get("custom_pipeline"):
                existing_def["custom_pipeline"] = model_def.get("custom_pipeline", "")
                
            if not existing_def.get("default_args"):
                existing_def["default_args"] = model_def.get("default_args", {})
                
            if not existing_def.get("metadata"):
                existing_def["metadata"] = model_def.get("metadata", {})
                
            # Handle components
            if not existing_def.get("components"):
                existing_def["components"] = {}
                
            # Merge components
            for comp_name, comp_def in model_def.get("components", {}).items():
                if comp_name in existing_def["components"]:
                    # Update component fields if empty
                    existing_comp = existing_def["components"][comp_name]
                    
                    if not existing_comp.get("class_name"):
                        existing_comp["class_name"] = comp_def.get("class_name", "")
                        
                    if not existing_comp.get("source"):
                        existing_comp["source"] = comp_def.get("source", "")
                        
                    if not existing_comp.get("kwargs"):
                        existing_comp["kwargs"] = comp_def.get("kwargs", {})
                else:
                    # Add new component
                    existing_def["components"][comp_name] = comp_def
        else:
            # Add new model definition
            merged_defs[model_id] = model_def
    
    # Remove models without a source
    models_to_remove = [model_id for model_id, def_obj in merged_defs.items() 
                       if not def_obj.get("source")]
    for model_id in models_to_remove:
        del merged_defs[model_id]
        
    return merged_defs
    

def load_config() -> RuntimeConfig:
    """
    Load the configuration from a YAML file located at COZY_HOME/config.yaml.
    Merges it with default values and database pipeline definitions.
    """
    default_home = os.path.expanduser("~/.cozy-creator")

    cozy_mount_path = os.getenv("COZY_MOUNT_PATH")
    if cozy_mount_path:
        default_home = cozy_mount_path

    default_models_path = os.path.join(default_home, "models")
    
    default_config = {
        "home_dir": default_home,
        "environment": "dev",
        "host": "localhost",
        "port": 8882,
        "pipeline_defs": {},
        # "enabled_models": [],
        "models_path": default_models_path,
    }

    print(f"default_config: {default_config}")
    
    # Use COZY_HOME if set, else default.
    home_dir = os.environ.get("COZY_HOME", default_home)
    config_path = os.path.join(home_dir, "config.yaml")
    
    merged = default_config.copy()
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                yaml_config = yaml.safe_load(f) or {}
            
            # Merge basic config values
            # for key, value in yaml_config.items():
            #     if key != "pipeline_defs":
            #         merged[key] = value
            
            # Get pipeline defs from config
            config_pipeline_defs = yaml_config.get("pipeline_defs", {})
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            config_pipeline_defs = {}
    else:
        logger.warning(f"Config file {config_path} not found. Using default configuration.")
        config_pipeline_defs = {}
    
    # Get enabled models from environment if provided
    # enabled_models_env = os.environ.get("ENABLED_MODELS")
    # if enabled_models_env:
    #     try:
    #         # If provided as JSON.
    #         merged["enabled_models"] = json.loads(enabled_models_env)
    #     except Exception:
    #         # Otherwise, assume comma-separated.
    #         merged["enabled_models"] = [m.strip() for m in enabled_models_env.split(",") if m.strip()]
    
    # Load ALL pipeline definitions from the database
    # The modified load_pipeline_defs_from_db with no args should fetch all.
    logger.info("Loading all pipeline definitions from database...")
    db_pipeline_defs = load_pipeline_defs_from_db(enabled_models=None)
    print(f"db_pipeline_defs: {db_pipeline_defs}")
    
    # Merge pipeline definitions from config and database
    merged["pipeline_defs"] = merge_pipeline_defs(config_pipeline_defs, db_pipeline_defs)
    print(f"merged: {merged}")
    
    return RuntimeConfig(**merged)


def set_config(config: RuntimeConfig):
    """
    Sets the global configuration object .
    """
    global cozy_config
    cozy_config = config


def set_environment(environment: str):
    """
    Sets the global environment variable.
    """
    global ENVIRONMENT
    ENVIRONMENT = environment


def get_environment() -> str:
    """
    Returns the global environment variable.
    """
    if ENVIRONMENT is None:
        raise ValueError("Environment has not been set yet")

    return ENVIRONMENT


def get_config() -> RuntimeConfig:
    """
    Returns the global configuration object. This is only available if the config has been loaded, which happens at
    the start of the server, else it will raise an error.
    """
    if cozy_config is None:
        raise ValueError("Config has not been loaded yet")

    return cozy_config


ParseArgsMethod = Callable[
    [argparse.ArgumentParser, Optional[List[str]], Optional[argparse.Namespace]],
    Optional[argparse.Namespace],
]


def is_model_enabled(model_name: str) -> bool:
    """
    Returns a boolean indicating whether a model is enabled in the global configuration.
    """
    config = get_config()
    if config.pipeline_defs is None:
        return False

    return model_name in config.pipeline_defs.keys()


def get_mock_config() -> RuntimeConfig:
    """
    Returns a mock (or test) version of the global configuration object.
    This can be used outside of the cozy server environment.
    """

    environment = "test"
    # home_dir = DEFAULT_HOME_DIR

    return RuntimeConfig(
        port=8881,
        host="127.0.0.1",
        environment=environment,
    )
