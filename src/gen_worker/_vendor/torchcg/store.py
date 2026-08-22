"""Where the artifact lives: content-addressed get/put over one tensorfs CAS.

An artifact is a compressed tarball, so it enters the store as one whole
unchunked blob named by the sha256 of its bytes, and an exact-key ref points
straight at it. There is no per-artifact manifest: a manifest holding exactly
one file entry is an indirection with nothing to describe.

**Mint-once, first-writer-wins. The KEY is the identity; the bytes are not.**
(tcg#84, ruled 2026-08-21.) A second write under an existing key is refused
OUTRIGHT -- the store never compares the incoming bytes with the stored ones,
because that comparison cannot mean what it looks like it means: AOTI does not
emit the same bytes twice for one graph on one host under one env and policy
(measured here, two mints, everything else held), so a byte difference is not
evidence of a key collision and a byte match is not evidence of its absence.
A verdict a check cannot prove is worse than no check, because it will be
believed.

Identity integrity therefore lives entirely in the key derivation -- graph
witness x env x layout x policy -- whose stability is what the checked-in hash
bank actually measures.

The artifact's content digest survives for exactly one job: TRANSPORT
INTEGRITY. It answers "did these bytes arrive intact", never "is this the right
artifact".
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..tensorfs import CASRef, DigestMismatch, LocalCAS, RefConflict

from .identity import is_artifact_key
from .refuse import KeyAlreadyMinted, StoreError

_REF_PREFIX = "torchcg/v1/graphs"
_BUFFER = 1 << 16
_MEMBERS = ("metadata.json", "model.pt2", "constants.safetensors")


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    """One verified artifact unpacked for a caller."""

    key: str
    directory: Path
    metadata: dict[str, Any]
    ref: CASRef

    @property
    def package(self) -> Path:
        return self.directory / "model.pt2"

    @property
    def literals(self) -> Path | None:
        path = self.directory / "constants.safetensors"
        return path if path.is_file() else None


def _require_key(key: object) -> str:
    value = str(key)
    if not is_artifact_key(value):
        raise StoreError(f"{value!r} is not an artifact key")
    return value


def _ref_name(key: str) -> str:
    return f"{_REF_PREFIX}/{key}"


def _digest_file(path: Path) -> CASRef:
    """Name a file by its bytes without writing them anywhere."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_BUFFER):
            digest.update(chunk)
    return CASRef(digest.hexdigest())


class Store:
    """Artifact naming and immutability over one ``LocalCAS``."""

    def __init__(self, cas: LocalCAS) -> None:
        self.cas = cas

    def _admit(self, artifact: Path) -> CASRef:
        """Admit one whole-file blob, writing nothing when the store holds it.

        TRANSPORT integrity, and only that. `verify_object`, not `contains`:
        presence is not integrity, and `contains` is one lstat that answers True
        for bytes corrupted IN PLACE. An artifact whose blob went bad is
        REPLACED rather than re-registered, so corruption cannot survive
        admission and resurface later as a serving failure.
        """

        ref = _digest_file(artifact)
        try:
            self.cas.verify_object(ref)
            return ref
        except (DigestMismatch, FileNotFoundError):
            return self.cas.put_file(artifact, expected=ref)

    def put(self, key: str | Any, artifact: Path) -> CASRef:
        """Store one artifact under its key. A key mints ONCE.

        A second write is refused whatever the bytes say, and the refusal is a
        statement about the KEY, not a comparison. `get` first: the stored
        artifact is servable and this one is redundant.
        """

        name = _require_key(key)
        if not artifact.is_file():
            raise StoreError(f"artifact {str(artifact)!r} is not a file")
        # PRESENCE, asked directly. Leaving this to `compare_and_swap_ref` alone
        # looked equivalent and was not: tensorfs treats a swap to the value
        # already stored as a no-op success, so the refusal would have fired
        # only when the bytes DIFFERED -- reintroducing the byte comparison the
        # ruling removed, one layer down and invisibly.
        if self.cas.read_ref(_ref_name(name)) is not None:
            raise KeyAlreadyMinted(self._minted(name))
        ref = self._admit(artifact)
        # ...and the swap for the race the presence check cannot close: two
        # minters can both read absent. Mint-once is a claim about CONCURRENT
        # minters, and a read-then-write alone cannot make it.
        try:
            self.cas.compare_and_swap_ref(_ref_name(name), ref, expected=None)
        except RefConflict as exc:
            raise KeyAlreadyMinted(self._minted(name)) from exc
        return ref

    @staticmethod
    def _minted(name: str) -> str:
        return (
            f"key {name} is already minted; the first artifact stands and nothing "
            f"was overwritten. The bytes offered here were NOT compared against "
            f"it, deliberately: AOTI does not emit the same bytes twice for one "
            f"graph on one host, so a comparison could not tell a key collision "
            f"from compiler nondeterminism. Identity is the key, and the key "
            f"already resolved -- call `get` and serve what is stored."
        )

    def get(self, key: str | Any, destination: Path) -> StoredArtifact | None:
        """Unpack and verify one artifact, or None when the key is unknown."""

        name = _require_key(key)
        ref = self.cas.read_ref(_ref_name(name))
        if ref is None:
            return None
        try:
            self.cas.verify_object(ref)
            path = self.cas.object_path(ref)
        except (DigestMismatch, FileNotFoundError) as exc:
            raise StoreError(f"key {name} points at unreadable bytes: {exc}") from exc
        directory = unpack(path, destination)
        stamped = read_metadata(directory)
        if stamped.get("key") != name:
            raise StoreError(
                f"artifact under key {name} states key {stamped.get('key')!r}; the "
                f"bytes and the address disagree"
            )
        return StoredArtifact(key=name, directory=directory, metadata=stamped, ref=ref)


def pack(directory: Path, destination: Path) -> Path:
    """Pack one artifact DETERMINISTICALLY.

    The envelope carries no time and no ownership: members are added in a fixed
    order with mtime 0, uid/gid 0 and no names, and gzip is given `mtime=0`.

    This USED to be load-bearing, and is no longer: it existed so that `put`'s
    byte comparison could not fire on a clock instead of on a collision, and
    that comparison is gone (tcg#84 ruling). What it still buys is smaller and
    honest -- the artifact's digest becomes a function of its CONTENTS alone, so
    the same contents cache and dedup as the same blob, and so the digest means
    "these bytes" rather than "these bytes, packed at that moment".

    It is deliberately NOT restored as evidence of identity. Today AOTI's own
    output varies between mints, so a stable envelope cannot make an artifact
    reproducible; if that ever changes, this is one of the things that would
    have to already be true, which is the cheapest reason to keep it.
    """

    import gzip

    destination.parent.mkdir(parents=True, exist_ok=True)
    # `filename=""` because gzip stores the OUTPUT PATH in its header, so
    # packing the same artifact to two scratch names would digest differently.
    with destination.open("wb") as sink, gzip.GzipFile(
        filename="", mode="wb", fileobj=sink, mtime=0
    ) as raw:
        with tarfile.open(fileobj=raw, mode="w") as tar:
            for member in _MEMBERS:
                path = directory / member
                if not path.is_file():
                    continue
                info = tar.gettarinfo(str(path), arcname=member)
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mode = 0o644
                with path.open("rb") as source:
                    tar.addfile(info, source)
    return destination


def unpack(artifact: Path, destination: Path) -> Path:
    """Extract one artifact tarball, refusing any member it does not name.

    A closed member list rather than a path check: an archive is untrusted
    input, and "reject what escapes the directory" is a filter somebody has to
    keep correct, while "accept exactly these three names" cannot be walked out
    of at all.
    """

    destination.mkdir(parents=True, exist_ok=True)
    # NAME THE SKEW. A bare AOTI `.pt2` is a ZIP, and it is the one wrong input
    # a caller actually produces -- it is what a mint hands over before
    # `pack` wraps it. Left to `tarfile`, it comes back as "not a gzip file",
    # which reads as CORRUPTION and sends the reader to scrub the disk. The
    # remedy is to re-publish, so the refusal has to say which mistake it is.
    with Path(artifact).open("rb") as probe:
        magic = probe.read(4)
    if magic[:2] == b"PK":
        raise StoreError(
            f"{artifact} is a bare AOTI .pt2 package, not a compiled-graph "
            f"envelope. A package carries no metadata, so nothing can be built "
            f"from it at load time -- the producer must PUBLISH an envelope "
            f"(metadata.json + model.pt2). Re-publish it; the bytes on disk are "
            f"not corrupt."
        )
    try:
        with tarfile.open(artifact, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name not in _MEMBERS or not member.isfile():
                    raise StoreError(
                        f"artifact carries member {member.name!r}; a v1 artifact is "
                        f"exactly {list(_MEMBERS)!r}"
                    )
                tar.extract(member, destination, filter="data")
    except tarfile.TarError as exc:
        raise StoreError(f"artifact could not be read: {exc}") from exc
    if not (destination / "model.pt2").is_file():
        raise StoreError("artifact carries no model.pt2")
    return destination


def open_artifact(envelope: Path, destination: Path) -> StoredArtifact:
    """Unpack one artifact envelope that came from somewhere else.

    The seam for a caller with its OWN transport -- a hub fetch, a baked image
    layer, a pre-staged volume -- which holds the bytes but never put them in
    this store. It gets the same unpack-and-verify `get` performs, so an
    artifact that arrived by another road is admitted on the same terms.

    The key is READ from the bytes rather than supplied: this path has no
    address to check against, so claiming one would be inventing it. Verify it
    against an expected key with `adopt.load(..., key=)`.
    """

    directory = unpack(Path(envelope), Path(destination))
    stamped = read_metadata(directory)
    return StoredArtifact(
        key=str(stamped.get("key") or ""),
        directory=directory,
        metadata=stamped,
        ref=_digest_file(Path(envelope)),
    )


def read_metadata(directory: Path) -> dict[str, Any]:
    try:
        stamped = json.loads((directory / "metadata.json").read_text())
    except (OSError, ValueError) as exc:
        raise StoreError(f"artifact metadata is unreadable: {exc}") from exc
    if not isinstance(stamped, dict):
        raise StoreError("artifact metadata is not an object")
    if stamped.get("kind") != "aot-inductor":
        raise StoreError(f"artifact states kind {stamped.get('kind')!r}")
    policy = stamped.get("compile_policy")
    if not isinstance(policy, dict) or not policy:
        raise StoreError(
            "artifact records no compile_policy; the options are half the "
            "compiler's input signature and a key without them collides"
        )
    if policy.get("aot_inductor.package_constants_in_so") is not False:
        raise StoreError(
            "package_constants_in_so must be false: an artifact is CODE, and its "
            "constants are raw pointers into the live module's weights"
        )
    if not (
        policy.get("always_keep_tensor_constants")
        or policy.get("aot_inductor.use_runtime_constant_folding")
    ):
        raise StoreError(
            "constant folding is unfenced: with neither always_keep_tensor_"
            "constants nor use_runtime_constant_folding, inductor inlines small "
            "lifted constants into generated code and the bindable table is short"
        )
    return stamped


__all__ = [
    "KeyAlreadyMinted",
    "Store",
    "StoreError",
    "StoredArtifact",
    "open_artifact",
    "pack",
    "read_metadata",
    "unpack",
]
