import os
import argparse
from typing import Optional, List, Callable
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

def load_config() -> RuntimeConfig:
    """
    Load the configuration from a YAML file located at COZY_HOME/config.yaml.
    Merges it with default values.
    """
    default_home = os.path.expanduser("~/.cozy-creator")
    default_models_path = os.path.join(default_home, "models")
    default_config = {
        "home_dir": default_home,
        "environment": "dev",
        "host": "localhost",
        "port": 8882,
        "pipeline_defs": {},
        "enabled_models": [],
        "models_path": default_models_path,
    }
    
    # Use COZY_HOME if set, else default.
    home_dir = os.environ.get("COZY_HOME", default_home)
    config_path = os.path.join(home_dir, "config.yaml")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                yaml_config = yaml.safe_load(f) or {}
            # Merge loaded config with defaults (loaded values override defaults)
            merged = default_config.copy()
            # merged.update(yaml_config)

            raw_pipeline_defs = yaml_config.get("pipeline_defs", {})
            # pipeline_defs = {key: PipelineConfig(value) for key, value in raw_pipeline_defs.items()}
            pipeline_defs = raw_pipeline_defs
            merged["pipeline_defs"] = pipeline_defs
            
            enabled_models_env = os.environ.get("ENABLED_MODELS")
            if enabled_models_env:
                try:
                    # If provided as JSON.
                    merged["enabled_models"] = json.loads(enabled_models_env)
                except Exception:
                    # Otherwise, assume comma-separated.
                    merged["enabled_models"] = [m.strip() for m in enabled_models_env.split(",") if m.strip()]
            return RuntimeConfig(**merged)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return RuntimeConfig(**default_config)
    else:
        logger.warning(f"Config file {config_path} not found. Using default configuration.")
        return RuntimeConfig(**default_config)


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
