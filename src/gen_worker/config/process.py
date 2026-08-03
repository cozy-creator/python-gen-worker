"""The one `Settings` this process was started with — PUBLISHED, never found.

pgw#931 deleted `get_settings()`, a `functools.lru_cache(maxsize=1)` wrapper
around `load_settings()`. The problem was never the caching; it was that any
module, at any depth, at any moment, could *materialise configuration out of
the environment* just by importing it. Three consequences, all measured:

* content depended on **which module imported first**, because `lru_cache`
  latches at first touch;
* a caller that wanted to pass different settings had no way to;
* nothing could tell whether config had been loaded at all, so a module running
  before bootstrap silently got a fresh read of a half-built environment
  instead of an error.

This module keeps exactly one property of the old accessor — a process-wide
answer for code too deep to hand a parameter to — and removes the rest. It
**cannot load**. `install()` is called by a process entry with the `Settings`
that entry loaded, and `current()` raises if that never happened.

Prefer a parameter. `Settings` is passed by parameter wherever the caller has
one, and this is for the residue: leaf helpers in `models/` with wide fan-in
from both the worker and the standalone CLI. Every remaining reader is listed,
classified, in `scripts/config_reads_allowlist.txt`.
"""

from __future__ import annotations

from typing import Optional

from .loader import load_settings
from .settings import Settings


class SettingsNotInstalled(RuntimeError):
    """`current()` before any process entry published its `Settings`.

    Loud on purpose. The old `get_settings()` answered this case by reading the
    environment there and then, which is how a module that ran before bootstrap
    got config nobody had validated — and why nothing in the tree could say
    what this process was actually configured as.
    """


_SETTINGS: Optional[Settings] = None


def install(settings: Settings) -> Settings:
    """Publish the `Settings` this process entry loaded. Returns them, so a
    bootstrap can write `settings = config.install(load_settings())`."""
    global _SETTINGS
    _SETTINGS = settings
    return settings


def installed() -> bool:
    """Whether a process entry has published its `Settings` yet."""
    return _SETTINGS is not None


def current() -> Settings:
    """The published `Settings`. Raises `SettingsNotInstalled` if there are none."""
    if _SETTINGS is None:
        raise SettingsNotInstalled(
            "no Settings have been installed in this process. A process entry "
            "must call config.install(load_settings()) before anything reads "
            "configuration — see gen_worker/config/__init__.py (ruling §1.18)."
        )
    return _SETTINGS


def current_or(default: Settings) -> Settings:
    """The published `Settings`, or `default` when nothing is installed.

    For library code that legitimately runs outside a worker bring-up and has a
    meaningful zero-config answer. It takes the default AS A VALUE so the
    fallback is visible at the call site, rather than being a silent env read.
    """
    return _SETTINGS if _SETTINGS is not None else default


def reset_for_test() -> None:
    """Drop the installed `Settings`. Test-only."""
    global _SETTINGS
    _SETTINGS = None


def reload_for_test() -> Settings:
    """Re-run the loader over the CURRENT environment and publish the result.

    Test-only, and named so. It replaces `get_settings.cache_clear()`, which
    tests called after monkeypatching env to force the next lazy read to see
    the change. The difference is not cosmetic: this states that a NEW
    configuration is being published to the process, where the old call stated
    only that a cache had been emptied and left the actual reload to whichever
    module happened to touch it first.
    """
    return install(load_settings())
