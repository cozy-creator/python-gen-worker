"""Fail-loud lazy-import stubs for heavy deps during build-time discovery.

Discovery imports every endpoint module to read its ``@endpoint`` metadata
(live ``typing.get_type_hints`` on payload structs requires real imports).
That metadata is torch-free by design, but discovery may run in environments
where torch/CUDA isn't installed (manifest builds, CI). Instead of forcing
authors to defer ``import torch`` into handler bodies,
``stub_missing_heavy_deps`` wraps ``builtins.__import__``: an ``import``
statement targeting an allowlisted heavy root that is genuinely ABSENT from
the environment resolves to a stub whose EVERY attribute access raises
:class:`HeavyDepStubError` naming the real dependency and the fix.

* Root INSTALLED → nothing changes; the real module is used.
* Root MISSING → ``import torch`` / ``import torch.nn.functional`` /
  ``from torch import nn`` all succeed with stubs; schemas still build; any
  code that actually EXECUTES the dep at import time (``DTYPE =
  torch.bfloat16``, ``torch.cuda.is_available()``) fails fast, actionably.

Injection point rationale: a permanently-armed ``sys.meta_path`` finder
would make ``importlib.util.find_spec("torchvision")`` return a spec for a
missing package, fooling the availability probes third-party libraries gate
their optional surfaces on (e.g. transformers' ``is_torchvision_available``
is find_spec-only — a stubbed answer crashes its module-scope torchvision
use). Wrapping ``__import__`` serves only real ``import`` statements; the
helper finder is armed transiently inside the retry, so probe results stay
honest. ``HeavyDepStubError`` subclasses ``AttributeError`` so defaulted
``getattr(mod, "__version__", ...)`` probes on a stub degrade gracefully.

Extension point: the allowlist is ``DEFAULT_HEAVY_ROOTS`` plus per-project
``[tool.gen_worker] discovery_heavy_deps = ["my_heavy_lib"]`` entries
(merged, never replacing the defaults).
"""

from __future__ import annotations

import builtins
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Mapping, Sequence

# Known-heavy import roots: importing any of these pays seconds of load time
# (torch/CUDA init, kernel registration) that endpoint METADATA never needs.
# Deliberately small; diffusers/transformers are NOT here — they import
# cheaply without torch and their classes must be real for slot annotations.
DEFAULT_HEAVY_ROOTS: tuple[str, ...] = (
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "xformers",
    "flash_attn",
    "bitsandbytes",
)


class HeavyDepStubError(AttributeError):
    """A discovery stub for a missing heavy dependency was actually USED.

    Subclasses ``AttributeError`` so ``hasattr``/defaulted-``getattr`` probes
    and the ``from torch import nn`` submodule fallback in the import
    machinery keep working; any other attribute touch surfaces this loudly.
    """


class _HeavyDepStub(types.ModuleType):
    """Module whose every (missing-)attribute access raises HeavyDepStubError."""

    def __getattr__(self, attr: str) -> Any:
        root = self.__name__.split(".", 1)[0]
        raise HeavyDepStubError(
            f"'{self.__name__}.{attr}' was touched during discovery, but "
            f"{root!r} is not installed in this environment. Discovery stubs "
            f"missing heavy dependencies so a module-top `import {root}` is "
            f"free — but EXECUTING {root} code at import time (e.g. "
            f"`DTYPE = {root}.bfloat16` or `{root}.cuda.is_available()` at "
            f"module scope) is not. Move that code into setup() or the "
            f"handler body, or install {root!r} to run discovery against the "
            f"real module."
        )


class _HeavyDepStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves stub modules for missing heavy roots (and their submodules).

    Only ever armed transiently, inside the ``__import__`` retry — never left
    on ``sys.meta_path`` where ``find_spec`` probes would see it.
    """

    def __init__(self, missing_roots: frozenset[str]) -> None:
        self.missing_roots = missing_roots

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname.split(".", 1)[0] in self.missing_roots:
            # is_package=True so `import torch.nn.functional` resolves
            # through this same finder instead of "'torch' is not a package".
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        return None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType:
        return _HeavyDepStub(spec.name)

    def exec_module(self, module: types.ModuleType) -> None:
        pass


def _root_installed(root: str) -> bool:
    if root in sys.modules and not isinstance(sys.modules[root], _HeavyDepStub):
        return True
    try:
        return importlib.util.find_spec(root) is not None
    except (ImportError, ValueError):
        return False


@contextmanager
def stub_missing_heavy_deps(extra: Iterable[str] = ()) -> Iterator[frozenset[str]]:
    """Arm fail-loud import stubs for every allowlisted heavy root NOT installed.

    Yields the set of absent roots eligible for stubbing (empty when
    everything is installed — the normal in-image case, where imports behave
    exactly as without this context). On exit the ``__import__`` wrapper and
    every stub placed in ``sys.modules`` are removed; endpoint modules
    imported meanwhile keep their references, so later real USE of a stub
    still fails loudly.
    """
    roots = dict.fromkeys((*DEFAULT_HEAVY_ROOTS, *extra))
    missing = frozenset(r for r in roots if r and not _root_installed(r))
    if not missing:
        yield frozenset()
        return

    finder = _HeavyDepStubFinder(missing)
    original_import = builtins.__import__

    def _import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> types.ModuleType:
        # Arm the stub finder ONLY while importing an absent heavy root (or a
        # submodule of one) — this one __import__ call covers `import torch`,
        # `import torch.nn.functional`, and the `from torch import nn`
        # submodule fallback. Every other import (including a genuinely
        # missing non-heavy dep, a SyntaxError, a relative import) runs the
        # untouched machinery.
        root = name.split(".", 1)[0] if level == 0 else ""
        if root not in missing:
            return original_import(name, globals, locals, fromlist, level)
        sys.meta_path.append(finder)
        try:
            return original_import(name, globals, locals, fromlist, level)
        finally:
            try:
                sys.meta_path.remove(finder)
            except ValueError:
                pass

    builtins.__import__ = _import
    try:
        yield missing
    finally:
        if builtins.__import__ is _import:
            builtins.__import__ = original_import
        for name in [
            n for n, m in sys.modules.items()
            if isinstance(m, _HeavyDepStub) and n.split(".", 1)[0] in missing
        ]:
            del sys.modules[name]
