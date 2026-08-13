"""Component-tree derivation — slots are a TREE.

Bindings form a tree: the root slot is a checkpoint bundle, itself a
composition (unet, vae, text_encoder(s), scheduler, tokenizers); parts are
addressable by path (``pipeline.vae``). The tree is DERIVED from the
pipeline class — diffusers pipelines self-describe via
``_get_signature_keys`` / ``components`` / ``model_index.json``, and the
worker already enumerates exactly that at load time (``_register_lane``).
Endpoint authors never re-declare a subset by hand: sibling-as-part slot
declarations (``"vae": Slot(AutoencoderKL)``, ``"turbo_lora": Slot(str)``)
are deleted, not migrated.

Derived parts are classified WEIGHT-BEARING (residency-relevant: shareable,
evictable, a memory-pool entry — unet/vae/text encoders) vs CONFIG-ISH
(scheduler settings, tokenizers, processors). Only the former enter the
pool. The derived tree is published at BUILD time into the release manifest
(``functions[].slots[].components``): the hub needs the path vocabulary for
per-path policy (``pipeline`` open / ``pipeline.vae`` curated /
``pipeline.unet`` fixed) and for component-level
routing; pinning it at publish keeps diffusers version drift deterministic
(the cell key already carries the diffusers version).

Instance identity = root binding + override map: a swapped part changes
identity, so the compile cell DERIVES facts like "this cell must not claim
``vae.decode``" instead of relying on an author comment.

A part's LOAD DTYPE is part of that identity too. The tree therefore
carries a per-component dtype opinion (:func:`component_dtypes`), resolved from
the component's CLASS against the one facts table
(``gen_worker.families.facts``) — not declared per endpoint. It is published
with the tree at build time and applied at MATERIALIZE time by
``gen_worker.models.loading`` (a wide component cannot be recovered by
upcasting after a narrow load).
"""

from __future__ import annotations

import inspect
from pathlib import PurePath
from typing import Any, Dict, List, Mapping, Optional

from ..families.facts import ComponentDtype, component_dtypes_for_classes

# Part-name prefixes that are configuration, not weights. Everything else a
# pipeline class declares as a component is treated as weight-bearing.
_CONFIG_ISH_PREFIXES = (
    "scheduler", "tokenizer", "feature_extractor", "image_processor",
    "video_processor", "processor",
)

KIND_WEIGHTS = "weights"
KIND_CONFIG = "config"


def part_kind(name: str) -> str:
    """``"weights"`` | ``"config"`` classification for one derived part."""
    n = str(name or "").strip().lower()
    for prefix in _CONFIG_ISH_PREFIXES:
        if n == prefix or n.startswith(prefix + "_") or n.startswith(prefix + "2"):
            return KIND_CONFIG
    return KIND_WEIGHTS


def is_introspectable(pipeline_cls: type) -> bool:
    """True when the SDK can derive a component tree from ``pipeline_cls``
    (diffusers-style self-description). ``str``/``Path``/plain classes are
    the explicit-declaration escape hatch."""
    if not isinstance(pipeline_cls, type):
        return False
    if pipeline_cls in (str, bytes) or issubclass(pipeline_cls, (str, bytes)):
        return False
    return (
        callable(getattr(pipeline_cls, "_get_signature_keys", None))
        or isinstance(getattr(pipeline_cls, "components", None), property)
    )


def derive_components(pipeline_cls: type) -> Optional[Dict[str, str]]:
    """``{part_name: kind}`` for an introspectable pipeline class, or None.

    Uses diffusers' own ``_get_signature_keys`` (the same source
    ``pipe.components`` reads) when present, else the ``__init__``
    signature. Never instantiates the class.
    """
    if not is_introspectable(pipeline_cls):
        return None
    names: List[str] = []
    sig_keys = getattr(pipeline_cls, "_get_signature_keys", None)
    if callable(sig_keys):
        try:
            expected, _optional = sig_keys(pipeline_cls)
            names = sorted(str(n) for n in expected)
        except Exception:
            names = []
    if not names:
        try:
            sig = inspect.signature(inspect.getattr_static(pipeline_cls, "__init__"))
        except (TypeError, ValueError):
            return None
        names = [
            p.name for p in sig.parameters.values()
            if p.name not in ("self", "args", "kwargs")
            and p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
    if not names:
        return None
    return {n: part_kind(n) for n in names}


def component_classes(pipeline_cls: type) -> Dict[str, str]:
    """``{part_name: component class NAME}`` from the pipeline class's
    ``__init__`` annotations.

    diffusers annotates every component it composes (``vae:
    AutoencoderKLWan``), so the class vocabulary is derivable at BUILD time
    with no snapshot and no instantiation — which is what lets the per-component
    dtype opinion be published with the tree. Unannotated or non-class
    annotations are simply absent; the load path's ``model_index.json`` is the
    authoritative source when one exists.
    """
    if not isinstance(pipeline_cls, type):
        return {}
    try:
        init = inspect.getattr_static(pipeline_cls, "__init__")
    except AttributeError:
        return {}
    raw = getattr(init, "__annotations__", None) or {}
    out: Dict[str, str] = {}
    for name, ann in raw.items():
        if name in ("self", "return", "args", "kwargs"):
            continue
        if isinstance(ann, type):
            out[str(name)] = ann.__name__
        elif isinstance(ann, str) and ann.isidentifier():
            # PEP 563 / quoted annotation: the bare name is what the facts
            # table keys on, so no resolution (and no import) is needed.
            out[str(name)] = ann
    return out


def component_dtypes(
    pipeline_cls: Optional[type] = None,
    *,
    model_index_classes: Optional[Mapping[str, str]] = None,
) -> Dict[str, ComponentDtype]:
    """``{part_name: ComponentDtype}`` — the load-dtype opinions this
    composition's parts carry. Empty means "loads uniformly at the
    composition's compute dtype", which is the overwhelmingly common case.

    ``model_index_classes`` (``{part: class name}`` read from a snapshot's
    ``model_index.json``) is AUTHORITATIVE where present: a fine-tune may
    substitute a component class, and the bytes on disk decide. The pipeline
    class's ``__init__`` annotations fill in the rest (and are the only source
    at build time, before any snapshot exists).
    """
    classes: Dict[str, str] = {}
    if pipeline_cls is not None:
        classes.update(component_classes(pipeline_cls))
    for part, cls_name in (model_index_classes or {}).items():
        if str(cls_name or "").strip():
            classes[str(part)] = str(cls_name).strip()
    return component_dtypes_for_classes(classes)


def components_manifest(pipeline_cls: type) -> Optional[List[Dict[str, str]]]:
    """The manifest projection of the derived tree: ordered
    ``[{name, kind}, ...]`` rows, or None for non-introspectable classes.

    A part carrying a load-dtype fact also publishes ``dtype`` — the hub needs
    it in the path vocabulary for the same reason it needs ``kind``:
    a component override at ``pipeline.vae`` must be materialized at the same
    precision the base part requires.
    """
    tree = derive_components(pipeline_cls)
    if tree is None:
        return None
    dtypes = component_dtypes(pipeline_cls)
    rows: List[Dict[str, str]] = []
    for n, k in sorted(tree.items()):
        row: Dict[str, str] = {"name": n, "kind": k}
        fact = dtypes.get(n)
        if fact is not None:
            row["dtype"] = fact.dtype
        rows.append(row)
    return rows


def validate_no_sibling_parts(
    owner: str, slots: Dict[str, Any], models: Dict[str, Any],
) -> None:
    """Sibling-as-part declarations are refused, never migrated.

    Two shapes are rejected when the endpoint declares an introspectable
    root pipeline slot:

    1. A sibling slot whose NAME matches a component of the root's derived
       tree (the old ``"vae": Slot(AutoencoderKL)`` hand-wiring) — the part
       is already addressable as ``<root>.<name>``; the override is catalog
       data, not endpoint code.
    2. A sibling ``str``/``Path``/``PurePath`` slot next to a derived-tree
       pipeline slot (the old ``"turbo_lora": Slot(str)`` workaround) —
       modifiers on components are adapters riding the binding, never
       sibling slots.
    """
    trees: Dict[str, Dict[str, str]] = {}
    for name, slot in slots.items():
        cls = getattr(slot, "pipeline_cls", None)
        if isinstance(cls, type):
            tree = derive_components(cls)
            if tree:
                trees[name] = tree
    if not trees:
        return
    all_slot_names = set(slots) | set(models)
    for root, tree in trees.items():
        for sibling in sorted(all_slot_names - {root}):
            if sibling in tree:
                raise ValueError(
                    f"{owner}: slot {sibling!r} is a COMPONENT of "
                    f"{root!r}'s pipeline ({sibling!r} is in its derived "
                    f"component tree). SDK v2 derives the tree from the "
                    f"pipeline class — delete the sibling declaration; a "
                    f"component override is catalog data addressed as "
                    f"'{root}.{sibling}' (th#1116), never a sibling slot."
                )
    for name, slot in slots.items():
        if name in trees:
            continue
        cls = getattr(slot, "pipeline_cls", None)
        # Path counts too, not just str/bytes: a Slot(Path) is the same
        # deleted modifier shape.
        if isinstance(cls, type) and (
            cls is str or issubclass(cls, (str, bytes, PurePath))
        ):
            raise ValueError(
                f"{owner}: slot {name!r} is a self-loading str slot declared "
                f"as a SIBLING of derived-tree pipeline slot(s) "
                f"{sorted(trees)} — the deleted sibling-as-part shape "
                f"(z-image's turbo_lora). Modifiers on components are "
                f"adapters riding the model binding (loras=...), and "
                f"component swaps are catalog data; neither is a slot."
            )


__all__ = [
    "KIND_CONFIG",
    "KIND_WEIGHTS",
    "ComponentDtype",
    "component_classes",
    "component_dtypes",
    "components_manifest",
    "derive_components",
    "is_introspectable",
    "part_kind",
    "validate_no_sibling_parts",
]
