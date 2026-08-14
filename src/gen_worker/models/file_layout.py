"""The ``file_layout`` vocabulary — ONE set of tokens, ruled at th#1937.

th#1932 inventory item 7 measured four live spellings of one axis across two
repositories, none of which was ever read back::

    tensorhub   "multi-file" / "single-file"   the coarse-tier map's keys, and
                                               the only place the value has
                                               ever MEANT anything
    tensorhub   "transformers"                 emitted by its metadata
                                               inferrer — a LIBRARY name spent
                                               as a layout value
    this repo   "singlefile" / "diffusers"     the ``FileLayout`` Literal, and
                                               what the classifier publishes
    this repo   "single-file"                  ``convert.py``'s gguf branch,
                                               drifting against its own Literal

Publish stored whatever a producer sent while the hub's layout gate re-inferred
its own value, so nothing ever disagreed out loud.  That is survivable for a
display string and fatal for a selector: under the th#1934 redesign the derived
contract IS the selector, and a spec asking for ``single-file`` against a
checkpoint stamped ``singlefile`` resolves to nothing, at deploy, with no bug
anywhere to find.

RULED: ``multi-file`` | ``single-file``.  They win because they are the only
spelling that ever carried meaning, and because ``diffusers`` spends a LIBRARY
name on a layout value — a second collision on top of the first.  There are no
aliases: tensorhub refuses a dead spelling at DECLARE with
``file_layout_unknown_token``, so this module is what keeps this repo's publishes
speaking the language the hub accepts.

It lives under ``models/`` rather than ``convert/`` (pgw#1252) because the LOAD
side now reads it too: ``tensor_layout_contract`` declares a ``file_layouts``
axis from these tokens. ``convert`` already imports ``models`` freely and the
reverse is a hard cycle — measured, not assumed: ``convert/__init__`` reaches
``api.slot`` and back into ``models``, so a ``models -> convert`` edge raises at
import. One home, imported by both, and no copy in either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

#: A component-directory tree, read through a multi-component loader entry
#: point (``DiffusionPipeline.from_pretrained``).
MULTI_FILE: Final[Literal["multi-file"]] = "multi-file"

#: One flat key namespace, read through a single-file entry point
#: (``from_single_file``).
SINGLE_FILE: Final[Literal["single-file"]] = "single-file"

#: The absent value: the declaring library's loader does not branch on this
#: axis.  Distinct from "unknown" only in that nothing may state it in a
#: contract-spec — an absent dimension already matches everything.
NOT_APPLICABLE: Final[Literal[""]] = ""

FileLayout = Literal["multi-file", "single-file"]

KNOWN_FILE_LAYOUTS: frozenset[str] = frozenset({MULTI_FILE, SINGLE_FILE})

#: Named in a refusal so a caller still sending one learns what happened.  They
#: are NOT accepted: naming a dead spelling in an error message is help;
#: accepting it is a compatibility shim, and this repo is pre-launch.
_DEAD_SPELLINGS = {
    "singlefile": f"this repo's pre-th#1937 Literal; write {SINGLE_FILE!r}",
    "diffusers": f"a LIBRARY name — it belongs in library_name, and this axis wants {MULTI_FILE!r}",
    "multifile": f"write {MULTI_FILE!r}",
    "transformers": "a LIBRARY name — a transformers tree declares no file_layout at all",
    "single_file": f"write {SINGLE_FILE!r}",
    "multi_file": f"write {MULTI_FILE!r}",
}


def validate_file_layout(token: str) -> str:
    """Return the token, or raise ``ValueError`` naming the vocabulary.

    The empty string is admitted: the axis does not apply to every library.
    """
    token = (token or "").strip()
    if token == NOT_APPLICABLE or token in KNOWN_FILE_LAYOUTS:
        return token
    why = _DEAD_SPELLINGS.get(token.lower())
    detail = f" ({why})" if why else ""
    raise ValueError(
        f"file_layout_unknown_token: {token!r} is not a file_layout token{detail}. "
        f"The vocabulary is {SINGLE_FILE} | {MULTI_FILE}, ruled at th#1937 with no aliases"
    )


def is_single_file_snapshot(path: Path) -> bool:
    """The single-file SHAPE, from names alone.

    The predicate half of ``loading._single_file_checkpoint``, split out of it
    so the load-side observation and the loader's own routing decide from ONE
    rule — and so the observation costs no shard reassembly.
    """
    if path.is_file():
        return path.suffix == ".safetensors"
    if not path.is_dir():
        return False
    if (path / "model_index.json").exists() or (path / "config.json").exists():
        return False
    singles = [p for p in path.glob("*.safetensors") if p.is_file()]
    if len(singles) == 1:
        return True
    return bool(singles) and len(list(path.glob("*.safetensors.index.json"))) == 1


def observed_file_layout(path: Path) -> str:
    """The layout a tree ON DISK is in, or :data:`NOT_APPLICABLE`.

    The same two shapes ``convert/classifier.py`` stamps at publish, read back
    off the tree at load: a component-directory tree (``model_index.json`` /
    ``config.json`` at the root of what is being read) is ``multi-file``;
    a loose checkpoint is ``single-file``. Anything else states NOTHING rather
    than guessing — an unclassifiable shape is not evaluated, which is the
    tri-state's UNDECLARED rung and not a fail-open, since the contract handle
    has already been checked by the time this is consulted.
    """
    p = Path(path)
    if is_single_file_snapshot(p):
        return SINGLE_FILE
    if p.is_dir() and (
            (p / "model_index.json").exists() or (p / "config.json").exists()):
        return MULTI_FILE
    return NOT_APPLICABLE


__all__ = [
    "FileLayout",
    "KNOWN_FILE_LAYOUTS",
    "MULTI_FILE",
    "NOT_APPLICABLE",
    "SINGLE_FILE",
    "is_single_file_snapshot",
    "observed_file_layout",
    "validate_file_layout",
]
