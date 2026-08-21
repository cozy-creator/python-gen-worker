"""Fail-loud lazy-import stubs for heavy deps during discovery: an allowlisted root genuinely absent from the environment imports as a stub whose every attribute access raises HeavyDepStubError. The stub finder is armed only transiently inside the __import__ retry — left on sys.meta_path it fools find_spec availability probes, and roots libraries probe via try/import must not be stubbed at all (NEVER_STUB). HeavyDepStubError must stay an AttributeError and cannot also subclass ImportError: CPython refuses the dual base, and the import machinery's submodule fallback catches AttributeError on torch.__path__ — an ImportError there breaks `from torch import nn`."""

from __future__ import annotations

import builtins
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Mapping, Sequence

DEFAULT_HEAVY_ROOTS: tuple[str, ...] = (
    "torch",
    "torchvision",
    "torchaudio",
)

NEVER_STUB: dict[str, str] = {
    "triton": "torch.utils._triton.has_triton_package() probes it by import; "
              "a stub makes torch._dynamo touch triton.language and die",
    "xformers": "probed by import across diffusers/transformers to select an "
                "attention backend",
    "flash_attn": "probed by import to select an attention backend",
    "bitsandbytes": "probed by import to gate quantized-linear surfaces",
}


class HeavyDepStubError(AttributeError):
    """A discovery stub for a missing heavy dependency was actually USED."""


class _HeavyDepStub(types.ModuleType):

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

    def __init__(self, missing_roots: frozenset[str]) -> None:
        self.missing_roots = missing_roots

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname.split(".", 1)[0] in self.missing_roots:
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
    """Arm fail-loud import stubs for every allowlisted heavy root NOT installed."""
    roots = dict.fromkeys((*DEFAULT_HEAVY_ROOTS, *extra))
    for root in [r for r in roots if r in NEVER_STUB]:
        del roots[root]
        print(
            f"gen-worker discovery: refusing to stub {root!r} — "
            f"{NEVER_STUB[root]}. A stub would make third-party availability "
            f"probes answer 'installed' for a package that is not.",
            file=sys.stderr,
        )
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
            for module_name in [
                n
                for n, module in sys.modules.items()
                if isinstance(module, _HeavyDepStub)
                and n.split(".", 1)[0] == root
            ]:
                del sys.modules[module_name]

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
