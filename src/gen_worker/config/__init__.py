"""Worker config — the ONE component in this process that reads the environment.

`load_settings()` is called exactly once per process entry and the resulting
`Settings` is passed by parameter. There is deliberately no cached accessor: a
`lru_cache`d process-global is the same defect as a raw env read, only harder to
see.
"""
from .loader import (
    UnknownSettingError,
    load_settings,
    unrecognised_owned_env,
)
from .process import (
    SettingsNotInstalled,
    current,
    current_or,
    install,
    installed,
    reload_for_test,
    reset_for_test,
)
from .settings import Settings

__all__ = [
    "Settings",
    "SettingsNotInstalled",
    "UnknownSettingError",
    "current",
    "current_or",
    "install",
    "installed",
    "load_settings",
    "reload_for_test",
    "reset_for_test",
    "unrecognised_owned_env",
]
