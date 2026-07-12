"""Pytest session guard — fail fast if a STALE gen_worker shadows the source.

A user-global / stale wheel install of ``gen-worker`` (e.g. in ``~/.local``)
silently shadows the working tree, so the suite would pass while testing old
code. We import ``gen_worker`` and assert it resolves under this repo's ``src/``.

The supported way to run the suite is ``uv run --extra dev pytest`` (pytest is
declared in the ``dev`` optional-dependency group, not the default deps). See
issue #345.
"""

from __future__ import annotations

import os
from pathlib import Path

import gen_worker

# Deterministic CPU encode wherever the suite runs (gw#476): never probe or
# engage NVENC from tests — CI has no GPU, and dev boxes that have one must
# not do GPU work from the unit suite. Selection tests override + refresh
# the cache explicitly.
os.environ.setdefault("GEN_WORKER_VIDEO_ENCODER", "x264")

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
_LOCATION = Path(gen_worker.__file__).resolve()

if _REPO_SRC not in _LOCATION.parents:
    raise RuntimeError(
        f"gen_worker is imported from {_LOCATION}, NOT this repo's src/ "
        f"({_REPO_SRC}). A stale global install is shadowing the working tree — "
        "tests would run against the wrong code. Fix: run via "
        "`uv run --extra dev pytest`, and remove any global install with "
        "`python3 -m pip uninstall --break-system-packages gen-worker`."
    )


import pytest  # noqa: E402

from gen_worker.config.loader import get_settings  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_settings_cache():
    """`get_settings()` caches process-wide; tests monkeypatch env vars, so
    every test starts (and ends) with a cleared cache."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


