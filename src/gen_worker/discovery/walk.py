"""Unified endpoint walker shared by build-time and runtime discovery.

``find_endpoints`` takes a list of top-level module names, walks each one +
its submodules (when it's a package), and returns every ``@endpoint``-decorated
class or function found. Both discovery (endpoint.lock) and the worker
registry use this single walker so they can never drift apart.

Invariants:

* Objects whose ``__module__`` is outside the walked package tree are
  skipped (third-party re-exports stay out).
* Each object is yielded exactly once even when re-exported into multiple
  namespaces (dedup by ``id``).
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from dataclasses import dataclass
from typing import Any

from ..api.decorators import ATTR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoundEndpoint:
    """One ``@endpoint``-decorated class or function found by the walker."""

    obj: Any            # the live class or function object
    module: str         # declaring __module__
    qualname: str
    decl: Any           # the EndpointDecl the decorator attached
    walked_module: str  # top-level package the object was found under


def find_endpoints(module_names: list[str]) -> list[FoundEndpoint]:
    """Walk ``module_names`` (and their submodules); return decorated objects."""
    out: list[FoundEndpoint] = []
    seen_ids: set[int] = set()

    for top_name in module_names:
        try:
            top_module = importlib.import_module(top_name)
        except Exception as exc:
            logger.exception(
                "Could not import user module '%s': %s. "
                "Walker will return no endpoints from this module.",
                top_name, exc,
            )
            continue

        modules_to_scan: list[Any] = [top_module]
        if hasattr(top_module, "__path__"):
            prefix = top_module.__name__ + "."
            for sub in pkgutil.walk_packages(top_module.__path__, prefix=prefix):
                try:
                    modules_to_scan.append(importlib.import_module(sub.name))
                except Exception as exc:
                    logger.warning(
                        "Skipping submodule '%s' — import failed: %s.",
                        sub.name, exc,
                    )

        package_prefix = top_module.__name__ + "."
        for scan_module in modules_to_scan:
            out.extend(_scan_module(
                scan_module,
                walked_package_name=top_module.__name__,
                walked_package_prefix=package_prefix,
                seen_ids=seen_ids,
            ))

    out.sort(key=lambda f: (f.module, f.qualname))
    return out


def _scan_module(
    module: Any,
    *,
    walked_package_name: str,
    walked_package_prefix: str,
    seen_ids: set[int],
) -> list[FoundEndpoint]:
    found: list[FoundEndpoint] = []
    # Iterate __dict__ directly — inspect.getmembers triggers __getattr__,
    # which on some packages (transformers LazyModule) does expensive
    # lazy imports we don't want.
    for _attr_name, obj in module.__dict__.items():
        if not (inspect.isclass(obj) or inspect.isfunction(obj)):
            continue
        decl = getattr(obj, ATTR, None)
        if decl is None or id(obj) in seen_ids:
            continue
        obj_module = getattr(obj, "__module__", "") or ""
        if (
            obj_module != walked_package_name
            and not obj_module.startswith(walked_package_prefix)
        ):
            logger.info(
                "Skipping endpoint '%s.%s' — defined outside the walked "
                "package '%s'.",
                obj_module, getattr(obj, "__name__", "<unknown>"),
                walked_package_name,
            )
            continue
        seen_ids.add(id(obj))
        found.append(FoundEndpoint(
            obj=obj,
            module=obj_module,
            qualname=getattr(obj, "__qualname__", getattr(obj, "__name__", "")),
            decl=decl,
            walked_module=walked_package_name,
        ))
    return found
