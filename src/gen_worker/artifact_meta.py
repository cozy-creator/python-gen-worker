"""The ONE reader of the ``metadata.json`` packed at the root of a cell tarball.

Every artifact kind this worker handles — AOT (``aot_serve``) and
inductor-cache (``compile_cache``) — packs its envelope as a
``metadata.json`` member at the tar root. Per-call-site ``tarfile.open`` /
member scan / ``json.loads`` loops agree on the format by convention and
disagree on everything else: which members count, whether a non-``dict``
payload is a value or an error, and what a missing member raises.

STDLIB ONLY, and deliberately so — that is what makes it usable from the two
modules that must not import the compile stack:

* ``receipts`` verifies a delivered artifact before anything imports torch;
* ``guard_closure`` sits inside the ``env_seal -> guard_closure -> compile_cache
  -> registry -> compiled_graph_key -> env_seal`` cycle, so its own metadata read had to be
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

#: THREAT: the tarball is gzipped, so a 50 GB zero-filled
#: ``metadata.json`` costs a few MB on the wire and OOMs the pod INSIDE
#: ``receipts.verify_delivered_artifact`` — this reader supplies the envelope
#: that gate is about to verify, so the read precedes the digest check and no
#: caller can bound it instead. Enforced ONCE, on the tar header's declared
#: size: ``tarfile`` stops the member reader there, so an under-declaring
#: header cannot yield a larger ``.read()``.
#:
#: WHY THIS IS NOT SIZED OFF THE DECLARE BOUND. "4x
#: ``fleet_cells.CELL_DECLARE_MAX_BYTES``" sizes an ARTIFACT-plane read off the
#: CONTROL-plane bound. Those are deliberately different planes:
#: ``_UNBOUNDED_ENVELOPE_BLOCKS``
#: (``entries``/``guard_manifest``/``composition``/``weight_contract``) are
#: STRIPPED from the declare precisely because they "belong in the artifact,
#: not in the declare" — so this member is by design the place the unbounded
#: blocks live, and bounding it at the declare's scale refuses the shape the
#: design demands. Measured, in this tree: a real published sdxl cell's
#: metadata is 13,377,167 bytes on a 69 MB artifact (see
#: ``fleet_cells._UNBOUNDED_ENVELOPE_BLOCKS``), and it grows with the
#: artifact — a 36-entry ~141 MB AOT cell's envelope did not fit 16 MiB, and
#: the 92-minute mint that produced it was discarded.
#:
#: This is a MEMORY-SAFETY bound and nothing else: what a pod can decode into
#: host RAM without OOMing. It is not a policy on how large a legitimate
#: envelope may be — the artifact's own digest is that.
#:
#: THE NUMBER, and its honest margin: 64 MiB is ~4.8x the largest envelope
#: anyone has measured (the 13,377,167-byte sdxl cell above), so this is a
#: margin, not a fit. Acceptable because exceeding it is not silent:
#: `fleet_cells.adopt_delegated_mint` refuses `compiled_graph_envelope_unreadable` naming
#: this constant and the byte count, so the next envelope that outgrows it
#: costs one typed event, not a mint. Raise it on that evidence; do not raise
#: it on a guess.
MAX_METADATA_BYTES = 64 << 20


class ArtifactMetadataError(ValueError):
    """An artifact carries no readable :data:`METADATA_NAME`."""


def read_metadata(artifact: Union[str, Path]) -> Dict[str, Any]:
    """The packed envelope of ``artifact``, WITHOUT unpacking the cell.

    Reads the one member and stops — kind sniffing and every metadata-only
    gate (host ISA, runtime key, store verdict) run off this, so none of them
    pays for the multi-GiB payload beside it.

    Raises :class:`ArtifactMetadataError` naming the artifact when the member is
    absent, the tar is unreadable, larger than :data:`MAX_METADATA_BYTES`, or
    the payload is not a JSON object.
    """
    path = Path(artifact)
    try:
        with tarfile.open(path, mode="r:*") as tar:
            for member in tar:
                if member.name != METADATA_NAME or not member.isfile():
                    continue
                if member.size > MAX_METADATA_BYTES:
                    raise ArtifactMetadataError(
                        f"artifact {path} declares a {member.size}-byte "
                        f"{METADATA_NAME}, over the {MAX_METADATA_BYTES}-byte "
                        f"bound; refused before decompressing it")
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
    "MAX_METADATA_BYTES",
    "METADATA_NAME",
    "ArtifactMetadataError",
    "read_metadata",
    "try_read_metadata",
]
