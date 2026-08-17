"""Vendored upstream snapshots. See VENDORED.toml for revs and rationale.

Import them by their full path (``gen_worker._vendor.tensorfs``) so nothing can
confuse a vendored snapshot with a same-named distribution installed beside it —
which is exactly what the #1295 native-reader cutover will do with the current
``tensorfs``. Nothing is re-exported here.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

_MANIFEST = Path(__file__).with_name("VENDORED.toml")


@lru_cache(maxsize=None)
def vendored_rev(package: str) -> str:
    """The upstream commit a vendored package was taken at.

    The identity a caller used to read from installed distribution metadata.
    A vendored package HAS no distribution, and `importlib.metadata` answers
    that by raising — which every caller then swallowed into an empty string,
    silently dropping the component from whatever key it fed. The rev is
    strictly more exact than the release string it replaces.
    """
    packages = tomllib.loads(_MANIFEST.read_text())["packages"]
    if package not in packages:
        raise KeyError(f"{package} is not vendored; see {_MANIFEST}")
    return str(packages[package]["rev"])
