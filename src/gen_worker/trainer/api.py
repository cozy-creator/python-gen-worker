from __future__ import annotations

import importlib
from typing import Any

from .contracts import TrainerPlugin

_REQUIRED_HOOKS = (
    "setup",
    "configure",
    "prepare_batch",
    "train_step",
    "state_dict",
    "load_state_dict",
)


def _load_symbol(import_path: str) -> Any:
    import_path = str(import_path or "").strip()
    if import_path == "":
        raise ValueError("trainer import path is required")

    module_path, sep, symbol_name = import_path.partition(":")
    module_path = module_path.strip()
    if module_path == "":
        raise ValueError(f"invalid trainer import path: {import_path!r}")

    mod = importlib.import_module(module_path)
    if not sep:
        return _resolve_module_default_symbol(mod, module_path)

    symbol_name = symbol_name.strip()
    if symbol_name == "":
        raise ValueError(f"trainer symbol is required in import path: {import_path!r}")
    if not hasattr(mod, symbol_name):
        raise ValueError(f"module {module_path!r} has no symbol {symbol_name!r}")
    return getattr(mod, symbol_name)


def _resolve_module_default_symbol(mod: Any, module_path: str) -> Any:
    # Preferred explicit aliases when using module-only trainer import paths.
    for name in ("TRAINER", "Trainer", "trainer"):
        if not hasattr(mod, name):
            continue
        return getattr(mod, name)

    candidates: list[tuple[str, Any]] = []
    for name, value in vars(mod).items():
        if str(name).startswith("_"):
            continue
        if _missing_hooks(value):
            continue
        candidates.append((str(name), value))

    if len(candidates) == 1:
        return candidates[0][1]
    if len(candidates) == 0:
        raise ValueError(
            "trainer import path resolved module-only entrypoint but no class/instance "
            f"with required trainer hooks was found in module {module_path!r}"
        )
    names = ", ".join(sorted(name for name, _ in candidates))
    raise ValueError(
        "trainer import path resolved module-only entrypoint with multiple trainer candidates "
        f"in module {module_path!r}: {names}; use module:symbol"
    )


def _missing_hooks(value: Any) -> list[str]:
    missing: list[str] = []
    for hook in _REQUIRED_HOOKS:
        if not callable(getattr(value, hook, None)):
            missing.append(hook)
    return missing


def _validate_trainer_instance(instance: Any) -> TrainerPlugin:
    missing = _missing_hooks(instance)
    if missing:
        hooks = ", ".join(missing)
        raise TypeError(f"trainer class must implement class-only hooks: missing {hooks}")
    return instance


def load_trainer_plugin(import_path: str) -> TrainerPlugin:
    """Resolve an import path to a trainer class instance or trainer instance."""
    symbol = _load_symbol(import_path)

    if isinstance(symbol, type):
        try:
            instance = symbol()
        except TypeError as exc:
            raise TypeError(f"trainer class must be instantiable with no args: {symbol.__name__}") from exc
        return _validate_trainer_instance(instance)

    if not _missing_hooks(symbol):
        return _validate_trainer_instance(symbol)

    if callable(symbol):
        raise TypeError(
            "trainer import must resolve to a class/class-instance implementing class-only trainer hooks; "
            "plain callables are unsupported"
        )

    raise TypeError("trainer import must resolve to a class or class instance")


__all__ = ["load_trainer_plugin"]
