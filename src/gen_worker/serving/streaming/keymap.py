from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

logger = logging.getLogger(__name__)


class KeyMigrationError(RuntimeError):
    """The library's own key migration could not be obtained or trusted."""


class UnstreamableCheckpoint(KeyMigrationError):
    """A tensor needs a VALUE transform, not a rename, to reach this skeleton."""


def migration(module: "torch.nn.Module", names: Iterable[str]) -> Mapping[str, str]:
    """checkpoint tensor name -> the name ``module`` answers to."""

    keys = list(names)
    if not keys:
        return {}
    mapped = _transformers_migration(module, keys)
    if mapped is None:
        mapped = _diffusers_migration(module, keys)
    if mapped is None:
        return {}
    renames = {source: target for source, target in mapped.items() if source != target}
    if not renames:
        return {}
    _refuse_collisions(module, mapped)
    logger.info(
        "ctx.load: %s's own library migrates %d of %d checkpoint key(s) — e.g. "
        "%s (pgw#1453)",
        type(module).__name__,
        len(renames),
        len(keys),
        ", ".join(
            f"{source} -> {target}"
            for source, target in sorted(renames.items())[:2]
        ),
    )
    return renames


def _refuse_collisions(module: Any, mapped: Mapping[str, str]) -> None:

    landing: Dict[str, list[str]] = {}
    for source, target in mapped.items():
        landing.setdefault(target, []).append(source)
    clashes = {
        target: sorted(sources)
        for target, sources in landing.items()
        if len(sources) > 1
    }
    if clashes:
        raise KeyMigrationError(
            f"{type(module).__name__}: this library's key migration sends "
            f"{len(clashes)} skeleton name(s) more than one checkpoint tensor — "
            + "; ".join(
                f"{target} <- {sources}" for target, sources in sorted(clashes.items())
            )[:400]
            + ". One of each pair would be silently dropped, so the migration is "
            "refused rather than applied."
        )


def _transformers_migration(
    module: Any, keys: list[str]
) -> Mapping[str, str] | None:

    try:
        from transformers.modeling_utils import PreTrainedModel
    except ImportError:  # pragma: no cover - no transformers in this image
        return None
    if not isinstance(module, PreTrainedModel):
        return None
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import (
        WeightConverter,
        WeightRenaming,
        rename_source_key,
    )

    try:
        transforms = get_model_conversion_mapping(module)
    except Exception as exc:  # noqa: BLE001
        raise KeyMigrationError(
            f"{type(module).__name__}: transformers refused to state its own "
            f"key conversion mapping ({type(exc).__name__}: {exc}), so this "
            f"checkpoint's names cannot be migrated the way from_pretrained "
            f"migrates them"
        ) from exc
    renamings = [t for t in transforms if isinstance(t, WeightRenaming)]
    converters = [
        t for t in transforms
        if isinstance(t, WeightConverter) and not isinstance(t, WeightRenaming)
    ]
    unknown = [
        type(t).__name__ for t in transforms
        if not isinstance(t, (WeightRenaming, WeightConverter))
    ]
    if unknown:
        raise KeyMigrationError(
            f"{type(module).__name__}: transformers' conversion mapping "
            f"carries transform kind(s) {sorted(set(unknown))} that are "
            f"neither a WeightRenaming nor a WeightConverter. Ignoring one "
            f"would silently skip something from_pretrained applies, so the "
            f"migration is refused rather than half-applied."
        )
    meta_state_dict: Dict[str, None] = {
        name: None for name in _skeleton_names(module)
    }
    prefix = getattr(module, "base_model_prefix", None) or None
    mapped: Dict[str, str] = {}
    for key in keys:
        target, matched = rename_source_key(
            key, renamings, converters, prefix, meta_state_dict
        )
        if matched is not None:
            raise UnstreamableCheckpoint(
                f"{type(module).__name__}: checkpoint tensor {key!r} matches "
                f"transformers' converter {matched!r}, which FUSES, SPLITS or "
                f"PERMUTES bytes rather than renaming them. This engine "
                f"installs a container's bytes verbatim, so this checkpoint "
                f"cannot be streamed — it must be converted (loaded through "
                f"from_pretrained and re-saved) before publication."
            )
        mapped[key] = target
    return mapped


def _diffusers_migration(module: Any, keys: list[str]) -> Mapping[str, str] | None:

    fix = getattr(module, "_fix_state_dict_keys_on_load", None)
    if not callable(fix):
        return None
    sentinel: Dict[str, str] = {key: key for key in keys}
    try:
        returned = fix(sentinel)
    except Exception as exc:  # noqa: BLE001
        raise KeyMigrationError(
            f"{type(module).__name__}: diffusers' own key migration could not "
            f"be applied to names alone ({type(exc).__name__}: {exc}) — it "
            f"reads tensor VALUES in this version, so the engine cannot ask it "
            f"what a name becomes without loading the checkpoint"
        ) from exc
    result = sentinel if returned is None else returned
    if not isinstance(result, dict) or len(result) != len(keys):
        raise KeyMigrationError(
            f"{type(module).__name__}: diffusers' key migration returned "
            f"{len(result) if isinstance(result, dict) else type(result).__name__} "
            f"entries for {len(keys)} checkpoint tensor(s). It is documented to "
            f"rename keys and touch no value; anything else is a version this "
            f"engine cannot drive by name."
        )
    mapped: Dict[str, str] = {}
    for target, source in result.items():
        if not isinstance(source, str):
            raise KeyMigrationError(
                f"{type(module).__name__}: diffusers' key migration replaced a "
                f"VALUE (under {target!r}) rather than only renaming keys, so it "
                f"cannot be driven on names alone in this version"
            )
        mapped[source] = str(target)
    return mapped


def _skeleton_names(module: Any) -> Iterable[str]:
    for name, _ in module.named_parameters(remove_duplicate=False):
        yield name
    for name, _ in module.named_buffers(remove_duplicate=False):
        yield name


__all__ = ["KeyMigrationError", "UnstreamableCheckpoint", "migration"]
