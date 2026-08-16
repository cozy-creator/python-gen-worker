"""Exact-key compiled-graph policy over HashRepo's generic local CAS."""

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
from enum import Enum
from pathlib import Path

from ..tensorfs import CASRef, DigestMismatch, FileEntry, LocalCAS, RefConflict, RepositoryManifest

from .artifact import ArtifactError, _fsync_dir, unpack_artifact
from .identity import CompiledGraphKey, is_compiled_graph_key

_COMPILED_GRAPH_PATH = "compiled_graph.tar.gz"
_REF_PREFIX = "torch-compiled-graphs/v1"


class StorageError(RuntimeError):
    """A local compiled-graph record is malformed or unavailable."""


class QuarantinedArtifact(StorageError):
    """An exact-key record failed integrity or admission and cannot be reused."""


class StoreOutcome(str, Enum):
    STORED = "stored"
    PRESENT = "present"
    REPAIRED = "repaired"
    DIVERGENT = "divergent"


@dataclass(frozen=True, slots=True)
class StoreResult:
    outcome: StoreOutcome
    key: str
    manifest: CASRef


@dataclass(frozen=True, slots=True)
class StoredCompiledGraph:
    """One verified compiled graph materialized for a caller."""

    key: str
    directory: Path
    metadata: dict[str, object]
    manifest: CASRef

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


def _quarantine_ref(key: str, manifest: CASRef) -> str:
    return f"{_REF_PREFIX}/quarantine/{key}/{manifest.digest}"


class _CompiledGraphStore:
    """Compiled-graph naming, immutability, and quarantine over one ``LocalCAS``."""

    def __init__(self, cas: LocalCAS) -> None:
        self.cas = cas

    @staticmethod
    def _file(manifest: RepositoryManifest) -> FileEntry:
        if len(manifest.files) != 1 or manifest.files[0].path != _COMPILED_GRAPH_PATH:
            raise StorageError("compiled-graph manifest must contain exactly its v1 artifact")
        return manifest.files[0]

    def _is_quarantined(self, key: str, manifest: CASRef) -> bool:
        return self.cas.read_ref(_quarantine_ref(key, manifest)) is not None

    def _clear_quarantine(self, key: str, manifest: CASRef, expected: CASRef) -> bool:
        """Retire exactly one observed marker, never a newer quarantine."""

        name = _quarantine_ref(key, manifest)
        try:
            self.cas.compare_and_swap_ref(name, None, expected=expected)
            return True
        except RefConflict:
            return self.cas.read_ref(name) is None

    def _retire_candidate_quarantine(
        self, key: str, manifest: CASRef, observed: CASRef | None
    ) -> None:
        if observed is not None and self._clear_quarantine(key, manifest, observed):
            return
        if observed is None and not self._is_quarantined(key, manifest):
            return
        raise QuarantinedArtifact(
            f"compiled graph {key} received a fresh quarantine during repair"
        )

    def quarantine(self, key: str | CompiledGraphKey, manifest: CASRef) -> None:
        value = _key_value(key)
        name = _quarantine_ref(value, manifest)
        marker = self.cas.put_bytes(
            json.dumps(
                {
                    "format": 1,
                    "key": value,
                    "manifest": str(manifest),
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
        with tempfile.TemporaryDirectory(prefix="torch-compiled-graphs-import-") as raw:
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
            file = self.cas.ingest_file(owned, manifest_path=_COMPILED_GRAPH_PATH)
        candidate = self.cas.store_manifest(RepositoryManifest((file,)))
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
        with tempfile.TemporaryDirectory(prefix="torch-compiled-graphs-export-verify-") as raw:
            metadata = unpack_artifact(artifact, Path(raw) / "artifact")
        if metadata.get("compiled_graph_key") != key:
            raise ArtifactError(
                f"artifact states key {metadata.get('compiled_graph_key')!r}, expected {key!r}"
            )

    @staticmethod
    def _require_exact_file(target: Path, expected: FileEntry) -> None:
        """Accept an occupied destination only when it is the selected CAS file."""

        try:
            if target.is_symlink():
                raise FileExistsError(f"destination {target} is a symbolic link")
            digest = hashlib.sha256()
            with target.open("rb") as source:
                before = os.fstat(source.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise FileExistsError(f"destination {target} is not a regular file")
                while chunk := source.read(1 << 20):
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
        if (
            not stable
            or after.st_size != expected.size_bytes
            or digest.hexdigest() != expected.digest.digest
        ):
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
                            target_chunk = os.read(target_member, 1 << 20)
                            expected_chunk = os.read(expected_member, 1 << 20)
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

    def _require_selected(self, key: str, manifest: CASRef) -> None:
        """Fail closed if publication or quarantine moved during resolution."""

        if self.cas.read_ref(_graph_ref(key)) != manifest:
            raise StorageError(f"compiled graph {key} changed during resolution")
        if self._is_quarantined(key, manifest):
            raise QuarantinedArtifact(
                f"compiled graph {key} received a quarantine during resolution"
            )

    def export_artifact(self, key: str | CompiledGraphKey, destination: str | Path) -> Path:
        """Export one exact CAS record as a fully verified artifact envelope."""

        value = _key_value(key)
        manifest_ref = self.cas.read_ref(_graph_ref(value))
        if manifest_ref is None:
            raise StorageError(f"compiled graph {value} is not present")
        if self._is_quarantined(value, manifest_ref):
            raise QuarantinedArtifact(f"compiled graph {value} is quarantined")

        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            try:
                manifest = self.cas.load_manifest(manifest_ref)
                file = self._file(manifest)
                self.cas.materialize(file, temporary)
                self._verify_archive(temporary, value)
            except (
                ArtifactError,
                DigestMismatch,
                FileNotFoundError,
                StorageError,
                ValueError,
            ) as exc:
                self.quarantine(value, manifest_ref)
                raise QuarantinedArtifact(
                    f"compiled graph {value} failed export verification: {exc}"
                ) from exc
            try:
                os.link(temporary, target)
            except FileExistsError:
                self._require_exact_file(target, file)
            _fsync_dir(target.parent)
            return target
        finally:
            temporary.unlink(missing_ok=True)

    def resolve(
        self, key: str | CompiledGraphKey, destination: str | Path
    ) -> StoredCompiledGraph | None:
        """Materialize and fully verify one exact key, or return a clean miss."""

        value = _key_value(key)
        manifest_ref = self.cas.read_ref(_graph_ref(value))
        if manifest_ref is None:
            return None
        if self._is_quarantined(value, manifest_ref):
            raise QuarantinedArtifact(f"compiled graph {value} is quarantined")

        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw_archive = tempfile.mkstemp(
            prefix="graph-", suffix=".tar.gz", dir=target.parent
        )
        os.close(descriptor)
        archive = Path(raw_archive)
        candidate = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
        candidate.rmdir()
        try:
            try:
                manifest = self.cas.load_manifest(manifest_ref)
                file = self._file(manifest)
                self.cas.materialize(file, archive)
            except (DigestMismatch, FileNotFoundError, ValueError, StorageError) as exc:
                self.quarantine(value, manifest_ref)
                raise QuarantinedArtifact(
                    f"compiled graph {value} failed CAS verification: {exc}"
                ) from exc
            try:
                metadata = unpack_artifact(archive, candidate)
            except ArtifactError as exc:
                self.quarantine(value, manifest_ref)
                raise QuarantinedArtifact(
                    f"compiled graph {value} failed artifact verification: {exc}"
                ) from exc
            if metadata.get("compiled_graph_key") != value:
                self.quarantine(value, manifest_ref)
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
                self._require_selected(value, manifest_ref)
                return StoredCompiledGraph(value, target, metadata, manifest_ref)
            _fsync_dir(target.parent)
            self._require_selected(value, manifest_ref)
            return StoredCompiledGraph(value, target, metadata, manifest_ref)
        except QuarantinedArtifact:
            raise
        except (OSError, ArtifactError):
            raise
        except ValueError as exc:
            self.quarantine(value, manifest_ref)
            raise QuarantinedArtifact(f"compiled graph {value} failed verification: {exc}") from exc
        finally:
            archive.unlink(missing_ok=True)
            shutil.rmtree(candidate, ignore_errors=True)
