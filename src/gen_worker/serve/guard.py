"""pgw#1328/#1331: in the adopt-only role, importing what it forswore raises.

Two promises, one mechanism. **pgw#1328:** a pod that adopts by key cannot
compile, so importing :data:`~gen_worker.serve.role.MINT_MACHINERY` raises.
**pgw#1331:** a pod that serves through compiled graph classes and bare typed
math needs no model library, so importing
:data:`~gen_worker.serve.role.FORBIDDEN_LIBRARIES` raises too — and that one is
the promise with the measurable payoff, because ``import diffusers`` alone is
seconds of process start and hundreds of megabytes of resident host memory.

The CI fence (``scripts/lint_serve_role_closure.py``) proves no serve-role
module *can* reach the mint lane. This is the other half, and the two are not
redundant: the fence reads the tree, so it is blind to a name assembled at
runtime (``importlib.import_module(f"gen_worker.{part}")``) and to code that is
not in this repo at all — an endpoint's own module, which the worker imports
from the tenant's image. A pod that promises it cannot compile must keep that
promise against code the fence never saw.

So the adopt-only entry point installs a :mod:`sys.meta_path` finder ahead of
every other, and any import of a declared name (or a submodule of one) raises
an ``ImportError`` subclass, so a caller that already handles an optional
import degrades the way it does today rather than dying in a new place. That
degradation is not incidental: a generated family binding reaches its
declaration through ``try: ... except ImportError``, and on this pod that is
exactly the path it takes — pgw#1339's absent serving fact, degrading loudly
and serving.

**Why a finder and not a stub module.** A stub whose functions raise would let
the import succeed and move the failure to the first CALL, which may be an
hour into a pod's life and inside somebody's ``except Exception``. The import
is the earliest honest moment and the one a boot log can name.

**It is deliberately NOT installed in the eager-capable role.** That role mints
— §4.28 — and it serves eager on a miss, which is exactly what a model library
is for. Blocking either there would break the only path that produces the
artifacts the adopt-only role adopts.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import logging
import sys
from types import ModuleType
from typing import Optional, Sequence, Tuple

from . import role as role_mod

logger = logging.getLogger(__name__)


class MintMachineryUnavailable(ImportError):
    """A module of the mint lane was imported in an adopt-only serve process.

    ``module`` is the name that was asked for; ``blocked`` is the declared
    :data:`~gen_worker.serve.role.MINT_MACHINERY` entry it is (or is under).
    """

    def __init__(self, module: str, blocked: str) -> None:
        super().__init__(
            f"{module} is mint machinery and this process holds the "
            f"{role_mod.ServeRole.ADOPT_ONLY.value} serve role (pgw#1328): it "
            f"adopts by key and refuses or routes on a miss, so it must not be "
            f"able to compile. Blocked by the declared entry {blocked!r}.")
        self.module = module
        self.blocked = blocked


class ModelLibraryUnavailable(ImportError):
    """A model library was imported in an adopt-only serve process (pgw#1331).

    ``module`` is the name that was asked for; ``blocked`` is the declared
    :data:`~gen_worker.serve.role.FORBIDDEN_LIBRARIES` entry it is (or is
    under).
    """

    def __init__(self, module: str, blocked: str) -> None:
        super().__init__(
            f"{module} is a model library and this process holds the "
            f"{role_mod.ServeRole.ADOPT_ONLY.value} serve role (pgw#1331): it "
            f"serves through compiled graph classes and bare typed math, so it "
            f"has no model to construct. Blocked by the declared entry "
            f"{blocked!r}. If a family needs this, the need belongs in its "
            f"DECLARATION half, which runs on the mint lane.")
        self.module = module
        self.blocked = blocked


def _entry_for(name: str, declared: Sequence[str]) -> str:
    """The declared entry ``name`` is, or is a submodule of; else ""."""
    for entry in declared:
        if name == entry or name.startswith(f"{entry}."):
            return entry
    return ""


def _blocked_by(name: str) -> str:
    """The MINT_MACHINERY entry ``name`` is, or is a submodule of; else ""."""
    return _entry_for(name, role_mod.MINT_MACHINERY)


class _MintBlocker(importlib.abc.MetaPathFinder):
    """Refuses at FIND time, before any loader is consulted.

    ONE finder for both promises, rather than two: they are installed and
    removed together, and a second entry on ``sys.meta_path`` would let one be
    taken out while the other stayed, which is a half-kept promise nothing
    reports.
    """

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]] = None,
        target: Optional[ModuleType] = None,
    ) -> Optional[importlib.machinery.ModuleSpec]:
        blocked = _blocked_by(fullname)
        if blocked:
            raise MintMachineryUnavailable(fullname, blocked)
        library = _entry_for(fullname, role_mod.FORBIDDEN_LIBRARIES)
        if library:
            raise ModelLibraryUnavailable(fullname, library)
        return None


_installed: Optional[_MintBlocker] = None


def install() -> None:
    """Install the blocker at the front of ``sys.meta_path``. Idempotent.

    Refuses in any role but :attr:`~gen_worker.serve.role.ServeRole.ADOPT_ONLY`
    — an eager-capable process that installed this would fail its own first
    mint, and "the guard was on in the wrong role" must be loud where it is
    ordered, not silent until §4.28's side effect fires.

    Also refuses when a banned module is ALREADY in ``sys.modules``: the
    promise is that this process cannot reach the mint lane, and a blocker
    installed after the import has not kept it. Call it first.
    """
    global _installed
    if not role_mod.adopt_only():
        raise RuntimeError(
            f"the mint-import blocker is only for the "
            f"{role_mod.ServeRole.ADOPT_ONLY.value} role; this process holds "
            f"{role_mod.current().value}")
    already = (*present(), *libraries_present())
    if already:
        raise RuntimeError(
            f"forsworn modules are already imported in this process "
            f"({', '.join(already)}) — installing the blocker now would "
            f"promise something that is already false")
    if _installed is None:
        _installed = _MintBlocker()
        sys.meta_path.insert(0, _installed)
        logger.info(
            "pgw#1328/#1331: import blocker installed (%d mint modules, "
            "%d model libraries)",
            len(role_mod.MINT_MACHINERY), len(role_mod.FORBIDDEN_LIBRARIES))


def installed() -> bool:
    return _installed is not None and _installed in sys.meta_path


def present() -> Tuple[str, ...]:
    """Declared mint modules already in ``sys.modules``, in declared order."""
    return tuple(
        name for name in role_mod.MINT_MACHINERY if name in sys.modules)


def libraries_present() -> Tuple[str, ...]:
    """Declared model libraries already in ``sys.modules``, in declared order."""
    return tuple(
        name for name in role_mod.FORBIDDEN_LIBRARIES if name in sys.modules)


def _uninstall_for_test() -> None:
    """Test-only: take the blocker back out. Never called by the worker."""
    global _installed
    if _installed is not None and _installed in sys.meta_path:
        sys.meta_path.remove(_installed)
    _installed = None


__all__ = [
    "MintMachineryUnavailable",
    "ModelLibraryUnavailable",
    "install",
    "installed",
    "libraries_present",
    "present",
]
