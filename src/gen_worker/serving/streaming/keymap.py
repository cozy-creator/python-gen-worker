"""The key migration the LIBRARY applies on load, asked of the library (pgw#1453).

``from_pretrained`` does not install a checkpoint's tensors under the names the
checkpoint spells. Every model library carries its own history of renames and
applies them on the way in, which is why a stock sd1.5 mirror loads through the
eager bridge and produced a correct image — while this engine, which installs
tensors by EXACT name onto a meta skeleton, matched **0 of 197** tensors in
sd1.5's ``text_encoder``:

    skeleton (`CLIPTextModel`, transformers 5)  embeddings.token_embedding.weight
    checkpoint (published against transformers 4)
                                     text_model.embeddings.token_embedding.weight

The refusal itself was right — ``ctx.load`` never guesses, and installing 0 of
197 tensors silently would be far worse. What was missing is that **nothing
performed the rename**.

**The map is never hand-maintained here, and that is the whole design.** The
renames are SEMANTIC, not textual: transformers' flattening of the
``text_model.`` prefix is a prefix drop, but diffusers'
``query/key/value/proj_attn -> to_q/to_k/to_v/to_out.0`` is not expressible as
any string rule (``key`` is not a suffix of ``to_k``). Only the library knows
its own history, so this module ASKS each library for its own migration and
applies nothing of its own:

* **transformers** exports :func:`get_model_conversion_mapping` (the ordered
  ``WeightTransform`` list ``from_pretrained`` itself loads with) and
  :func:`rename_source_key`, a pure name->name function. Handing it the meta
  skeleton's own key set as ``meta_state_dict`` is what resolves the
  ``base_model_prefix`` add/strip, so the answer is the model's, not a guess.
* **diffusers** exports ``ModelMixin._fix_state_dict_keys_on_load``, which
  renames a state dict's KEYS in place and touches no value — so a dict of
  ``{name: name}`` comes back as ``{new_name: old_name}``, the library's own map
  with no tensor read and no byte moved.

It runs UNCONDITIONALLY, like pgw#1473's variant detection, because both
libraries answer "no change" for a checkpoint already spelled the way the
installed version spells it (verified: 0 renames over sd1.5's 686-tensor unet
and 248-tensor vae, and re-feeding the migrated names is a fixed point).

**Where this stops.** transformers distinguishes a RENAMING from a CONVERTER —
a fuse, split, transpose or RoPE permutation, which changes the bytes and not
only the name. This engine installs a container's bytes verbatim; a checkpoint
that needs a value transform is not streamable at all, and
:class:`UnstreamableCheckpoint` says so by name rather than installing raw bytes
under a converted name.
"""

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
    """checkpoint tensor name -> the name ``module`` answers to.

    Only the entries that actually CHANGE; a caller reads it as
    ``renames.get(name, name)``. Empty for a checkpoint whose names the
    installed library already spells, which is every converted artifact.
    """

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
    """Two checkpoint names must never migrate onto ONE skeleton name.

    A collision means one of the two tensors is dropped — the exact silent loss
    the ``NameMismatch`` refusal exists to prevent, so it is refused here rather
    than discovered as garbage on the first forward.
    """

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
    """transformers' own ``from_pretrained`` rename, or ``None`` if not one."""

    try:
        from transformers.modeling_utils import PreTrainedModel
    except ImportError:  # pragma: no cover - no transformers in this image
        return None
    if not isinstance(module, PreTrainedModel):
        return None
    # `conversion_mapping` is where this lives; `modeling_utils` re-exports it
    # without declaring it, which mypy is right to refuse — a re-export nobody
    # promised is a name that can move without a deprecation.
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
    # `WeightRenaming` and `WeightConverter` are SIBLINGS under
    # `WeightTransform`, so each arm is selected by what it IS rather than by
    # "not the other one" — and a third kind this code has never seen is
    # refused by name instead of being dropped into neither list, which would
    # silently skip a transform `from_pretrained` applies.
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
    # The skeleton's OWN key set is what decides whether `base_model_prefix`
    # is added or stripped — the model answering about itself.
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
    """diffusers' own ``_fix_state_dict_keys_on_load``, driven name-only."""

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
    # diffusers returns the dict it mutated; older spellings mutate in place
    # and return None.
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
