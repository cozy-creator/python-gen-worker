"""The one `Settings` this process was started with — PUBLISHED, never found."""

from __future__ import annotations

from typing import Optional

from .loader import load_settings
from .settings import Settings


class SettingsNotInstalled(RuntimeError):
    """`current()` before any process entry published its `Settings`."""


_SETTINGS: Optional[Settings] = None


def install(settings: Settings) -> Settings:
    """Publish the `Settings` this process entry loaded."""
    global _SETTINGS
    _SETTINGS = settings
    return settings


def installed() -> bool:
    """Whether a process entry has published its `Settings` yet."""
    return _SETTINGS is not None


def current() -> Settings:
    """The published `Settings`."""
    if _SETTINGS is None:
        raise SettingsNotInstalled(
            "no Settings have been installed in this process. A process entry "
            "must call config.install(load_settings()) before anything reads "
            "configuration — see gen_worker/config/__init__.py (ruling §1.18)."
        )
    return _SETTINGS


def current_or(default: Settings) -> Settings:
    """The published `Settings`, or `default` when nothing is installed."""
    return _SETTINGS if _SETTINGS is not None else default


def reset_for_test() -> None:
    """Drop the installed `Settings`."""
    global _SETTINGS
    _SETTINGS = None


def reload_for_test() -> Settings:
    """Re-run the loader over the CURRENT environment and publish the result."""
    return install(load_settings())
