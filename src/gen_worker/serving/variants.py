"""Which weight VARIANT a checkpoint tree offers (pgw#1473).

Every fp16 mirror on the hub ships `*.fp16.safetensors` and diffusers reads
those only when told `variant="fp16"`. `ctx.load`'s eager bridge passed no
`variant=`, so a bare mirror refused at boot with

    OSError: Error no file named diffusion_pytorch_model.bin found in
    directory .../vae

which points at the wrong thing entirely: nothing is missing, the loader is
looking under the wrong name (`.bin` is merely the last candidate in diffusers'
own fallback ladder). **The eager bridge IS the cozy-local substrate** —
`engine_for` returns `None` for any tree with no chunk store behind it — so
this refused every bare download, and both trees this box was handed carried it.

**The variant is a property of the TREE, and the worker already resolves the
tree, so it is DETECTED rather than configured.** No environment variable and
no `ctx.load(variant=)` argument: an author stating a fact about bytes they did
not publish is a second source of truth, and the one that drifts.

**The naming rules are diffusers' own, and they are read from diffusers rather
than reimplemented** — `_add_variant` inserts the token before the final
extension (`model.safetensors` -> `model.fp16.safetensors`,
`x.safetensors.index.json` -> `x.safetensors.index.fp16.json`), and the sharded
spelling glues it to the shard suffix
(`diffusion_pytorch_model.fp16-00001-of-00003.safetensors`). Both are in the
wild; sdxl's own mirror carries BOTH index spellings at once.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

#: The shard suffix diffusers glues a variant to. Its own spelling.
_SHARD = r"\d{5}-of-\d{5}"

#: Weight file extensions that can carry a variant token.
_SUFFIXES = ("safetensors", "bin")

#: A variant token: a short alphanumeric tag, never a shard range and never a
#: known non-variant word. Kept deliberately narrow — a greedy pattern would
#: read `model.index.json` as variant `index`.
_TOKEN = r"[A-Za-z0-9][A-Za-z0-9_]{0,15}"

_VARIANT_WEIGHT = re.compile(
    rf"^(?P<stem>.+?)\.(?P<token>{_TOKEN})(?:-{_SHARD})?\.(?P<suffix>{'|'.join(_SUFFIXES)})$"
)
_PLAIN_WEIGHT = re.compile(
    rf"^(?P<stem>.+?)(?:-{_SHARD})?\.(?P<suffix>{'|'.join(_SUFFIXES)})$"
)


class VariantAmbiguous(RuntimeError):
    """A tree offers several weight variants and none is plainly named.

    Refused BY NAME rather than picked: choosing between `fp16` and `bf16`
    would be a guess about PRECISION, made by the worker, on bytes the
    publisher already labelled. The operator states which tree they meant.
    """


def _component_dirs(tree: Path) -> list[Path]:
    """The pipeline component directories — one level down, as diffusers lays
    a snapshot out. The root itself is included for single-component trees."""

    dirs = [tree]
    try:
        dirs += sorted(p for p in tree.iterdir() if p.is_dir())
    except OSError:
        return [tree]
    return dirs


def _classify(directory: Path) -> Tuple[set[str], bool]:
    """(variant tokens offered, whether a PLAIN weight file exists) here."""

    tokens: set[str] = set()
    plain = False
    try:
        names = sorted(p.name for p in directory.iterdir() if p.is_file())
    except OSError:
        return tokens, plain
    for name in names:
        variant = _VARIANT_WEIGHT.match(name)
        if variant is not None:
            tokens.add(variant.group("token"))
            continue
        if _PLAIN_WEIGHT.match(name) is not None:
            plain = True
    return tokens, plain


def detect_variant(tree: Path | str) -> Optional[str]:
    """The variant this tree must be loaded with, or ``None``.

    ``None`` means "load it the ordinary way" and is the answer for every
    published/converted checkpoint — which is why this is safe to run always.
    A variant is returned ONLY when the tree offers no plainly named weight
    file anywhere, so a tree carrying both names is untouched.

    Several variants and no plain file is a :class:`VariantAmbiguous` refusal.
    """

    root = Path(tree)
    if not root.is_dir():
        return None
    tokens: set[str] = set()
    for directory in _component_dirs(root):
        found, plain = _classify(directory)
        if plain:
            # SOMETHING here is plainly named, so diffusers' ordinary ladder
            # resolves. Whatever variants sit beside it are extras, not the
            # only way in.
            return None
        tokens |= found
    if not tokens:
        return None
    if len(tokens) > 1:
        raise VariantAmbiguous(
            f"{root} offers weight variants {sorted(tokens)!r} and no plainly "
            f"named weight file. Which one to serve is a statement about "
            f"PRECISION and it belongs to whoever published the tree, not to "
            f"the worker — project or convert the one you mean."
        )
    token = tokens.pop()
    logger.info(
        "ctx.load: %s carries only %r-variant weight files and no plainly "
        "named ones, so the eager bridge loads it with variant=%r (pgw#1473)",
        root, token, token,
    )
    return token


__all__ = ["VariantAmbiguous", "detect_variant"]
