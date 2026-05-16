"""Unified endpoint-class walker shared by build-time and runtime discovery.

Both ``gen_worker.discovery.discover.discover_functions`` (build-time, baked
into ``endpoint.lock``) and ``gen_worker.worker.Worker._discover_handlers``
(runtime, registered into the dispatch tables) need to find the same set of
``@inference`` / ``@training`` / ``@dataset`` / ``@conversion`` classes from
the same module list. When the two implementations drift apart, the worker
boots fine but rejects every request with ``Unknown function requested`` —
the outage pattern that hit the conversion endpoint on 2026-05-16.

This module is the single source of truth. ``find_endpoint_classes`` takes a
list of top-level module names, walks each one + its submodules (when it's a
package), and returns a list of :class:`FoundEndpointClass` entries. Each
entry carries the live ``cls``, its declaring ``__module__`` / ``__qualname__``,
the ``__gen_worker_endpoint_spec__`` value, and a list of
:class:`FoundFunctionMethod` (one per ``@inference.function`` method on the
class).

Both invariants the band-aid runtime walk was doing are enforced here:

* Classes whose ``__module__`` is outside the walked package tree are
  skipped (third-party re-exports stay out).
* Each class is yielded exactly once even when it's re-exported into
  multiple namespaces (dedup by ``id(cls)``).
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoundFunctionMethod:
    """One ``@inference.function`` / ``@<kind>.function`` method on a class."""

    attr_name: str
    """The Python attribute name on the class (e.g. ``"generate"``)."""

    method: Callable[..., Any]
    """The unbound function object as stored on the class."""

    fn_spec: Any
    """The ``_FunctionSpec`` attached by the decorator."""


@dataclass(frozen=True)
class FoundEndpointClass:
    """One ``@inference`` / ``@<kind>``-decorated class found by the walker."""

    cls: type
    """The live class object."""

    module: str
    """``cls.__module__`` — the module the class was actually *defined* in
    (not the module we happened to be walking when we found it)."""

    qualname: str
    """``cls.__qualname__``."""

    endpoint_spec: Any
    """The ``__gen_worker_endpoint_spec__`` value the decorator attached."""

    walked_module: str
    """The top-level module name from :func:`find_endpoint_classes`'s input
    list under which this class was first found. Build-time discovery uses
    this as the canonical ``module`` field in the lockfile (e.g. the
    package name ``"conversion"`` rather than the per-submodule name) so
    re-exports collapse to a stable identity."""

    function_methods: list[FoundFunctionMethod] = field(default_factory=list)
    """All ``@<kind>.function`` methods on the class. May be empty for
    @batched_inference-only classes that route through a different attribute."""


def find_endpoint_classes(module_names: list[str]) -> list[FoundEndpointClass]:
    """Walk ``module_names`` (and their submodules) and return endpoint classes.

    For each name in ``module_names``:

    * Import the module. Import failures are logged and that module is
      skipped; sibling modules still get walked.
    * Scan the module's namespace for classes that carry
      ``__gen_worker_endpoint_spec__``.
    * When the module is a package (has ``__path__``), also walk every
      submodule via :func:`pkgutil.walk_packages` and scan each one.

    A class is included only if its declaring ``__module__`` is the walked
    package itself or one of its submodules — third-party classes that
    happen to be re-exported into the walked namespace are silently skipped.

    Returned classes are deduplicated by ``id(cls)``, so a class re-exported
    into multiple namespaces is yielded exactly once. Each entry's
    ``function_methods`` list reflects the methods attached by the
    ``@<kind>.function`` decorator (in declaration order).
    """
    out: list[FoundEndpointClass] = []
    seen_ids: set[int] = set()

    for top_name in module_names:
        try:
            top_module = importlib.import_module(top_name)
        except Exception as exc:
            # Catch *every* import error (not just ImportError) — a broken
            # tenant module can raise NameError / AttributeError / etc.
            # Surface as ERROR so the operator can debug; sibling modules
            # still get walked.
            logger.exception(
                "Could not import user module '%s': %s. "
                "Walker will return no classes from this module.",
                top_name, exc,
            )
            continue

        modules_to_scan: list[Any] = [top_module]

        # If the top-level entry is a package, also walk every submodule.
        # This handles packages whose __init__.py doesn't re-export classes
        # from submodules — without the walk, discovery would silently miss
        # them (the original conversion-endpoint outage).
        if hasattr(top_module, "__path__"):
            prefix = top_module.__name__ + "."
            for sub in pkgutil.walk_packages(top_module.__path__, prefix=prefix):
                try:
                    submodule = importlib.import_module(sub.name)
                except Exception as exc:
                    logger.warning(
                        "Skipping submodule '%s' — import failed: %s. "
                        "Other submodules will still be walked.",
                        sub.name, exc,
                    )
                    continue
                modules_to_scan.append(submodule)

        package_prefix = top_module.__name__ + "."
        for scan_module in modules_to_scan:
            for found in _scan_module(
                scan_module,
                walked_package_name=top_module.__name__,
                walked_package_prefix=package_prefix,
                seen_ids=seen_ids,
            ):
                out.append(found)
        # NOTE: walked_module is stamped inside _scan_module rather than here
        # so per-class entries carry the correct top-level package name.

    # Stable order: classes returned in the order their declaring module
    # would sort alphabetically. Build-time discovery wants this for
    # reproducible lockfiles.
    out.sort(key=lambda f: (f.module, f.qualname))
    return out


def _scan_module(
    module: Any,
    *,
    walked_package_name: str,
    walked_package_prefix: str,
    seen_ids: set[int],
) -> list[FoundEndpointClass]:
    """Scan a single module's namespace for endpoint classes."""
    found: list[FoundEndpointClass] = []
    # Iterate __dict__ directly — inspect.getmembers triggers __getattr__,
    # which on some packages (transformers LazyModule) does expensive
    # lazy imports we don't want.
    for _attr_name, obj in module.__dict__.items():
        if not inspect.isclass(obj):
            continue
        spec = getattr(obj, "__gen_worker_endpoint_spec__", None)
        if spec is None:
            continue
        if id(obj) in seen_ids:
            # Already collected via another walked module (re-export).
            continue
        obj_module = getattr(obj, "__module__", "") or ""
        # Skip third-party re-exports: the class must be defined in the
        # walked package or one of its submodules.
        if (
            obj_module != walked_package_name
            and not obj_module.startswith(walked_package_prefix)
        ):
            logger.info(
                "Skipping endpoint class '%s.%s' — defined in module '%s' which is "
                "outside the walked package '%s'. If you intended this to be an "
                "endpoint of '%s', move the @inference class into the package or "
                "re-export it from a sibling submodule.",
                obj_module,
                getattr(obj, "__name__", "<unknown>"),
                obj_module,
                walked_package_name,
                walked_package_name,
            )
            continue

        function_methods_raw = (
            getattr(obj, "__gen_worker_function_methods__", None) or []
        )
        function_methods: list[FoundFunctionMethod] = [
            FoundFunctionMethod(attr_name=attr, method=method, fn_spec=fn_spec)
            for attr, method, fn_spec in function_methods_raw
        ]

        seen_ids.add(id(obj))
        found.append(
            FoundEndpointClass(
                cls=obj,
                module=obj_module,
                qualname=getattr(obj, "__qualname__", getattr(obj, "__name__", "")),
                endpoint_spec=spec,
                walked_module=walked_package_name,
                function_methods=function_methods,
            )
        )
    return found
