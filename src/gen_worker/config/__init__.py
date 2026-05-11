"""Worker config — single typed `Settings` struct loaded once at startup."""
from .loader import load_settings
from .settings import Settings

__all__ = ["Settings", "load_settings"]
