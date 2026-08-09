"""The ONE reader of the ``metadata.json`` packed at the root of a cell tarball.

Every artifact kind this worker handles — AOT (``aot_serve``), inductor-cache
(``compile_cache``), TRT (``trt_engine``) — packs its envelope as a
``metadata.json`` member at the tar root. Eight call sites had each grown their
own ``tarfile.open`` / member scan / ``json.loads`` loop, agreeing on the format
by convention and disagreeing on everything else: which members count, whether a
non-``dict`` payload is a value or an error, and what a missing member raises.

STDLIB ONLY, and deliberately so — that is what makes it usable from the two
modules that must not import the compile stack:

* ``receipts`` verifies a delivered artifact before anything imports torch;
* ``guard_closure`` sits inside the ``env_seal -> guard_closure -> compile_cache
  -> registry -> cell_key -> env_seal`` cycle, so its own metadata read had to be
  a function-local import.

Callers keep their own refusal vocabulary (``AdoptError``, ``ReceiptError``, a
store-verdict string); this module only answers "what does the envelope say", and
:class:`ArtifactMetadataError` subclasses :class:`ValueError` so the call sites
that already classify on ``ValueError`` are unchanged.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

#: The packed envelope's member name, at the tar root, for every artifact kind.
METADATA_NAME = "metadata.json"


class ArtifactMetadataError(ValueError):
    """An artifact carries no readable :data:`METADATA_NAME`."""


def read_metadata(artifact: Union[str, Path]) -> Dict[str, Any]:
    """The packed envelope of ``artifact``, WITHOUT unpacking the cell.

    Reads the one member and stops — kind sniffing and every metadata-only
    gate (host ISA, runtime key, store verdict) run off this, so none of them
    pays for the multi-GiB payload beside it.

    Raises :class:`ArtifactMetadataError` naming the artifact when the member is
    absent, the tar is unreadable, or the payload is not a JSON object.
    """
    path = Path(artifact)
    try:
        with tarfile.open(path, mode="r:*") as tar:
            for member in tar:
                if member.name != METADATA_NAME or not member.isfile():
                    continue
                src = tar.extractfile(member)
                if src is None:
                    break
                loaded = json.loads(src.read().decode("utf-8"))
                if not isinstance(loaded, dict):
                    raise ArtifactMetadataError(
                        f"artifact {path} carries a {METADATA_NAME} that is "
                        f"not an object ({type(loaded).__name__})")
                return dict(loaded)
    except ArtifactMetadataError:
        raise
    except (OSError, tarfile.TarError, ValueError, UnicodeDecodeError) as exc:
        raise ArtifactMetadataError(f"artifact {path} is unreadable: {exc}") from exc
    raise ArtifactMetadataError(f"artifact {path} has no {METADATA_NAME}")


def try_read_metadata(artifact: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """:func:`read_metadata`, or ``None`` when it cannot be read.

    For the best-effort readers whose contract is that they never fail: an
    adopt-event identity line, a lane verdict that falls back to a named
    default. They report the absence themselves; this only refuses to raise.
    """
    try:
        return read_metadata(artifact)
    except ArtifactMetadataError:
        return None


__all__ = [
    "METADATA_NAME",
    "ArtifactMetadataError",
    "read_metadata",
    "try_read_metadata",
]
