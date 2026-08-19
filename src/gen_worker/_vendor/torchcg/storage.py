"""Exact-key compiled-graph policy over one tensorfs content-addressed store.

A compiled-graph artifact is a compressed tarball, so it enters the store as one
whole unchunked blob named by the sha256 of its bytes. An exact-key ref points
straight at that blob: there is no per-artifact manifest, because a manifest
holding exactly one file entry is an indirection with nothing to describe.

Lookup therefore returns the blob's own immutable path. Reading an artifact
copies nothing -- ``resolve`` untars directly out of the store. Only
``export_artifact``, whose entire purpose is to hand a caller an independent
file on the caller's own device, writes the bytes a second time.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import secrets
import shutil
import stat
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tensorfs import CASRef, DigestMismatch, LocalCAS, RefConflict

from .artifact import ArtifactError, _fsync_dir, unpack_artifact
from .identity import CompiledGraphKey, is_compiled_graph_key

_COMPILED_GRAPH_PATH = "compiled_graph.tar.gz"
_REF_PREFIX = "torchcg/v1"
_READ_BUFFER = 1 << 20


class StorageError(RuntimeError):
    """A local compiled-graph record is malformed or unavailable."""


class QuarantinedArtifact(StorageError):
    """An exact-key record failed integrity or admission and cannot be reused."""


class StoreOutcome(StrEnum):
    STORED = "stored"
    PRESENT = "present"
    REPAIRED = "repaired"
    DIVERGENT = "divergent"


@dataclass(frozen=True, slots=True)
class StoreResult:
    outcome: StoreOutcome
    key: str
    artifact: CASRef


@dataclass(frozen=True, slots=True)
class StoredCompiledGraph:
    """One verified compiled graph unpacked for a caller."""

    key: str
    directory: Path
    metadata: dict[str, object]
    artifact: CASRef

    @property
    def package(self) -> Path:
        return self.directory / "model.pt2"

    @property
    def literals(self) -> Path | None:
        path = self.directory / "constants.safetensors"
        return path if path.is_file() else None


def _key_value(key: str | CompiledGraphKey) -> str:
    value = str(key)
    if not is_compiled_graph_key(value):
        raise StorageError("v1 storage requires a compiled-graph key")
    return value


def _graph_ref(key: str) -> str:
    return f"{_REF_PREFIX}/graphs/{key}"


def _quarantine_ref(key: str, artifact: CASRef) -> str:
    return f"{_REF_PREFIX}/quarantine/{key}/{artifact.digest}"


def _digest_file(path: Path) -> CASRef:
    """Name a file by its bytes without writing them anywhere."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_READ_BUFFER):
            digest.update(chunk)
    return CASRef(digest.hexdigest())


class _CompiledGraphStore:
    """Compiled-graph naming, immutability, and quarantine over one ``LocalCAS``."""

    def __init__(self, cas: LocalCAS) -> None:
        self.cas = cas

    def _admit(self, artifact: Path) -> CASRef:
        """Admit one whole-file blob, writing nothing when the store already holds it."""

        ref = _digest_file(artifact)
        try:
            if self.cas.contains(ref):
                return ref
        except DigestMismatch:
            # The named blob is present but unusable; put_file replaces it atomically.
            pass
        return self.cas.put_file(artifact, expected=ref)

    def _is_quarantined(self, key: str, artifact: CASRef) -> bool:
        return self.cas.read_ref(_quarantine_ref(key, artifact)) is not None

    def _clear_quarantine(self, key: str, artifact: CASRef, expected: CASRef) -> bool:
        """Retire exactly one observed marker, never a newer quarantine."""

        name = _quarantine_ref(key, artifact)
        try:
            self.cas.compare_and_swap_ref(name, None, expected=expected)
            return True
        except RefConflict:
            return self.cas.read_ref(name) is None

    def _retire_candidate_quarantine(
        self, key: str, artifact: CASRef, observed: CASRef | None
    ) -> None:
        if observed is not None and self._clear_quarantine(key, artifact, observed):
            return
        if observed is None and not self._is_quarantined(key, artifact):
            return
        raise QuarantinedArtifact(
            f"compiled graph {key} received a fresh quarantine during repair"
        )

    def quarantine(self, key: str | CompiledGraphKey, artifact: CASRef) -> None:
        value = _key_value(key)
        name = _quarantine_ref(value, artifact)
        marker = self.cas.put_bytes(
            json.dumps(
                {
                    "format": 1,
                    "key": value,
                    "artifact": str(artifact),
                    "marker": secrets.token_hex(16),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        )
        expected = self.cas.read_ref(name)
        while True:
            try:
                self.cas.compare_and_swap_ref(name, marker, expected=expected)
                return
            except RefConflict:
                expected = self.cas.read_ref(name)

    def store(self, key: str | CompiledGraphKey, artifact: str | Path) -> StoreResult:
        """Verify and keep the first bytes published under one exact graph key."""

        value = _key_value(key)
        source = Path(artifact)
        with tempfile.TemporaryDirectory(prefix="torchcg-import-") as raw:
            owned = Path(raw) / _COMPILED_GRAPH_PATH
            shutil.copyfile(source, owned)
            with owned.open("rb") as handle:
                os.fsync(handle.fileno())
            metadata = unpack_artifact(owned, Path(raw) / "verified")
            if metadata.get("compiled_graph_key") != value:
                raise StorageError(
                    f"artifact states key {metadata.get('compiled_graph_key')!r}, "
                    f"expected {value!r}"
                )
            candidate = self._admit(owned)
        candidate_quarantine = self.cas.read_ref(_quarantine_ref(value, candidate))
        ref_name = _graph_ref(value)
        current = self.cas.read_ref(ref_name)
        if current is None:
            try:
                self.cas.compare_and_swap_ref(ref_name, candidate, expected=None)
                self._retire_candidate_quarantine(
                    value, candidate, candidate_quarantine
                )
                outcome = (
                    StoreOutcome.REPAIRED
                    if candidate_quarantine is not None
                    else StoreOutcome.STORED
                )
                return StoreResult(outcome, value, candidate)
            except RefConflict:
                current = self.cas.read_ref(ref_name)
        if current is None:
            raise StorageError(f"exact-key ref {value} disappeared during publication")
        if current == candidate:
            if candidate_quarantine is not None:
                self._retire_candidate_quarantine(value, current, candidate_quarantine)
                return StoreResult(StoreOutcome.REPAIRED, value, current)
            if self._is_quarantined(value, current):
                raise QuarantinedArtifact(
                    f"compiled graph {value} received a fresh quarantine during publication"
                )
            return StoreResult(StoreOutcome.PRESENT, value, current)
        if self._is_quarantined(value, current):
            try:
                self.cas.compare_and_swap_ref(ref_name, candidate, expected=current)
                self._retire_candidate_quarantine(
                    value, candidate, candidate_quarantine
                )
                return StoreResult(StoreOutcome.REPAIRED, value, candidate)
            except RefConflict:
                current = self.cas.read_ref(ref_name)
                if current == candidate:
                    self._retire_candidate_quarantine(
                        value, candidate, candidate_quarantine
                    )
                    return StoreResult(StoreOutcome.PRESENT, value, candidate)
                if current is None:
                    raise StorageError(f"exact-key ref {value} disappeared during repair") from None
        self.quarantine(value, candidate)
        return StoreResult(StoreOutcome.DIVERGENT, value, candidate)

    @staticmethod
    def _verify_archive(artifact: Path, key: str) -> None:
        with tempfile.TemporaryDirectory(prefix="torchcg-export-verify-") as raw:
            metadata = unpack_artifact(artifact, Path(raw) / "artifact")
        if metadata.get("compiled_graph_key") != key:
            raise ArtifactError(
                f"artifact states key {metadata.get('compiled_graph_key')!r}, expected {key!r}"
            )

    @staticmethod
    def _require_exact_file(target: Path, expected: CASRef, size: int) -> None:
        """Accept an occupied destination only when it is the selected CAS blob."""

        try:
            if target.is_symlink():
                raise FileExistsError(f"destination {target} is a symbolic link")
            digest = hashlib.sha256()
            with target.open("rb") as source:
                before = os.fstat(source.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise FileExistsError(f"destination {target} is not a regular file")
                while chunk := source.read(_READ_BUFFER):
                    digest.update(chunk)
                after = os.fstat(source.fileno())
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise FileExistsError(f"cannot verify occupied destination {target}: {exc}") from exc
        stable = (
            before.st_dev == after.st_dev
            and before.st_ino == after.st_ino
            and before.st_size == after.st_size
            and before.st_mtime_ns == after.st_mtime_ns
        )
        if not stable or after.st_size != size or digest.hexdigest() != expected.digest:
            raise FileExistsError(
                f"destination {target} is not the selected compiled-graph artifact"
            )

    @staticmethod
    def _require_exact_directory(target: Path, expected: Path) -> None:
        """Accept an occupied directory only when every selected artifact byte won."""

        message = f"destination {target} is not the selected compiled-graph artifact"
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        target_fd: int | None = None
        expected_fd: int | None = None
        try:
            try:
                target_fd = os.open(target, directory_flags)
                expected_fd = os.open(expected, directory_flags)
            except OSError as exc:
                raise FileExistsError(f"{message}: {exc}") from exc
            if target_fd is None or expected_fd is None:  # pragma: no cover - os.open returns int
                raise FileExistsError(message)
            target_directory_before = os.fstat(target_fd)
            if not stat.S_ISDIR(target_directory_before.st_mode):
                raise FileExistsError(message)
            target_names = set(os.listdir(target_fd))
            expected_names = set(os.listdir(expected_fd))
            if target_names != expected_names:
                raise FileExistsError(message)
            for name in sorted(expected_names):
                target_member = os.open(name, file_flags, dir_fd=target_fd)
                try:
                    expected_member = os.open(name, file_flags, dir_fd=expected_fd)
                    try:
                        target_before = os.fstat(target_member)
                        expected_before = os.fstat(expected_member)
                        if (
                            not stat.S_ISREG(target_before.st_mode)
                            or not stat.S_ISREG(expected_before.st_mode)
                            or target_before.st_size != expected_before.st_size
                        ):
                            raise FileExistsError(message)
                        while True:
                            target_chunk = os.read(target_member, _READ_BUFFER)
                            expected_chunk = os.read(expected_member, _READ_BUFFER)
                            if target_chunk != expected_chunk:
                                raise FileExistsError(message)
                            if not target_chunk:
                                break
                        target_after = os.fstat(target_member)
                        expected_after = os.fstat(expected_member)
                        for before, after in (
                            (target_before, target_after),
                            (expected_before, expected_after),
                        ):
                            if (
                                before.st_dev != after.st_dev
                                or before.st_ino != after.st_ino
                                or before.st_size != after.st_size
                                or before.st_mtime_ns != after.st_mtime_ns
                            ):
                                raise FileExistsError(message)
                    finally:
                        os.close(expected_member)
                finally:
                    os.close(target_member)
            target_directory_after = os.fstat(target_fd)
            current_path = os.stat(target, follow_symlinks=False)
            if (
                target_directory_before.st_dev != target_directory_after.st_dev
                or target_directory_before.st_ino != target_directory_after.st_ino
                or target_directory_before.st_mtime_ns != target_directory_after.st_mtime_ns
                or current_path.st_dev != target_directory_after.st_dev
                or current_path.st_ino != target_directory_after.st_ino
            ):
                raise FileExistsError(message)
        except FileExistsError:
            raise
        except OSError as exc:
            raise FileExistsError(f"{message}: {exc}") from exc
        finally:
            if expected_fd is not None:
                os.close(expected_fd)
            if target_fd is not None:
                os.close(target_fd)

    def _require_selected(self, key: str, artifact: CASRef) -> None:
        """Fail closed if publication or quarantine moved during resolution."""

        if self.cas.read_ref(_graph_ref(key)) != artifact:
            raise StorageError(f"compiled graph {key} changed during resolution")
        if self._is_quarantined(key, artifact):
            raise QuarantinedArtifact(
                f"compiled graph {key} received a quarantine during resolution"
            )

    def export_artifact(self, key: str | CompiledGraphKey, destination: str | Path) -> Path:
        """Export one exact CAS blob as an independent, fully verified envelope.

        This is the one path that still writes the artifact's bytes twice, and it
        has to: the caller names a destination the store does not own, possibly on
        another device. Hardlinking the store's own inode out would alias canonical
        bytes into a tree that may outlive and rewrite them, so the export copies.
        """

        value = _key_value(key)
        artifact_ref = self.cas.read_ref(_graph_ref(value))
        if artifact_ref is None:
            raise StorageError(f"compiled graph {value} is not present")
        if self._is_quarantined(value, artifact_ref):
            raise QuarantinedArtifact(f"compiled graph {value} is quarantined")

        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            try:
                blob = self.cas.verify_object(artifact_ref)
                size = blob.stat().st_size
            except (DigestMismatch, FileNotFoundError, ValueError) as exc:
                self.quarantine(value, artifact_ref)
                raise QuarantinedArtifact(
                    f"compiled graph {value} failed export verification: {exc}"
                ) from exc
            # A destination write failing here is the caller's disk, not bad CAS
            # bytes, so it propagates as OSError and quarantines nothing.
            shutil.copyfile(blob, temporary)
            try:
                self._verify_archive(temporary, value)
            except (ArtifactError, StorageError, ValueError) as exc:
                self.quarantine(value, artifact_ref)
                raise QuarantinedArtifact(
                    f"compiled graph {value} failed export verification: {exc}"
                ) from exc
            try:
                os.link(temporary, target)
            except FileExistsError:
                self._require_exact_file(target, artifact_ref, size)
            _fsync_dir(target.parent)
            return target
        finally:
            temporary.unlink(missing_ok=True)

    def resolve(
        self, key: str | CompiledGraphKey, destination: str | Path
    ) -> StoredCompiledGraph | None:
        """Unpack and fully verify one exact key, or return a clean miss.

        The archive is untarred straight out of the store, so acquiring an artifact
        copies nothing. The only bytes written are the unpacked directory the
        caller asked for.
        """

        value = _key_value(key)
        artifact_ref = self.cas.read_ref(_graph_ref(value))
        if artifact_ref is None:
            return None
        if self._is_quarantined(value, artifact_ref):
            raise QuarantinedArtifact(f"compiled graph {value} is quarantined")

        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        candidate = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
        candidate.rmdir()
        try:
            try:
                blob = self.cas.verify_object(artifact_ref)
            except (DigestMismatch, FileNotFoundError, ValueError, StorageError) as exc:
                self.quarantine(value, artifact_ref)
                raise QuarantinedArtifact(
                    f"compiled graph {value} failed CAS verification: {exc}"
                ) from exc
            try:
                metadata = unpack_artifact(blob, candidate)
            except ArtifactError as exc:
                self.quarantine(value, artifact_ref)
                raise QuarantinedArtifact(
                    f"compiled graph {value} failed artifact verification: {exc}"
                ) from exc
            if metadata.get("compiled_graph_key") != value:
                self.quarantine(value, artifact_ref)
                raise QuarantinedArtifact(
                    f"stored artifact states key {metadata.get('compiled_graph_key')!r}, "
                    f"expected {value!r}"
                )
            try:
                os.rename(candidate, target)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not target.is_dir():
                    raise
                self._require_exact_directory(target, candidate)
                self._require_selected(value, artifact_ref)
                return StoredCompiledGraph(value, target, metadata, artifact_ref)
            _fsync_dir(target.parent)
            self._require_selected(value, artifact_ref)
            return StoredCompiledGraph(value, target, metadata, artifact_ref)
        except QuarantinedArtifact:
            raise
        except (OSError, ArtifactError):
            raise
        except ValueError as exc:
            self.quarantine(value, artifact_ref)
            raise QuarantinedArtifact(f"compiled graph {value} failed verification: {exc}") from exc
        finally:
            shutil.rmtree(candidate, ignore_errors=True)
