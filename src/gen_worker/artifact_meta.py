"""The ONE reader of the ``metadata.json`` packed at the root of a compiled graph tarball."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Union

METADATA_NAME = "metadata.json"

# Memory-safety bound, nothing else: the tarball is gzipped, so a 50 GB zero-filled metadata.json costs a few MB on the wire and OOMs the pod before the digest check can run. Enforced once on the tar header's declared size; 64 MiB is ~5x the largest measured envelope, and exceeding it is a typed refusal, not silence — raise it on evidence, not a guess.
MAX_METADATA_BYTES = 64 << 20


class ArtifactMetadataError(ValueError):
    """An artifact carries no readable :data:`METADATA_NAME`."""


def read_metadata(artifact: Union[str, Path]) -> Dict[str, Any]:
    """The packed envelope of ``artifact``, WITHOUT unpacking the compiled graph."""
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
    """:func:`read_metadata`, or ``None`` when it cannot be read."""
    try:
        return read_metadata(artifact)
    except ArtifactMetadataError:
        return None


def compiled_graph_metadata_fields() -> FrozenSet[str]:
    """TCG's CLOSED artifact-metadata vocabulary — what a compiled graph can state."""
    from gen_worker._vendor import torchcg as tcg

    fields = getattr(tcg, "ARTIFACT_METADATA_FIELDS", None)
    if not fields:
        raise ArtifactMetadataError(
            "the vendored torchcg states no artifact-metadata vocabulary "
            "(`ARTIFACT_METADATA_FIELDS`, public since tcg#40); nothing can "
            "decide which arm axes a compiled graph is able to state, and guessing would "
            "either refuse every mint or admit a comparison that can never "
            "succeed")
    return frozenset(str(name) for name in fields)


__all__ = [
    "MAX_METADATA_BYTES",
    "METADATA_NAME",
    "ArtifactMetadataError",
    "compiled_graph_metadata_fields",
    "read_metadata",
    "try_read_metadata",
]
