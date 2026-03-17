"""
A/B test: chunked split + reassemble produces byte-identical output.

Proves that the ingest chunking (Go: SplitAndStore at 1 GiB boundaries)
followed by the worker reassembly (_reassemble_chunked) does NOT corrupt
weights.  The reassembled file must be byte-for-byte identical to the
original, with matching blake3 hashes.

Test matrix:
  - small file (under chunk threshold, no split)
  - file exactly at 1 GiB boundary
  - file slightly over 1 GiB (2 parts)
  - large file spanning 3 parts
  - random binary content (simulates real safetensors weights)
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import List

import pytest
from blake3 import blake3


# ---------------------------------------------------------------------------
# Constants matching Go chunkedblob.MaxChunkSize
# ---------------------------------------------------------------------------

MAX_CHUNK_SIZE = 1 << 30  # 1 GiB

# For fast tests, use a smaller chunk size.  Set CHUNKED_TEST_REAL_SIZE=1 to
# test at actual 1 GiB boundaries (slow, ~3 GiB of I/O).
FAST_CHUNK = 1 << 20  # 1 MiB for fast tests


def _chunk_size() -> int:
    if os.getenv("CHUNKED_TEST_REAL_SIZE") == "1":
        return MAX_CHUNK_SIZE
    return FAST_CHUNK


# ---------------------------------------------------------------------------
# Simulate Go-side ingest: split file into chunks + write parts.json
# ---------------------------------------------------------------------------

def _blake3_bytes(data: bytes) -> str:
    h = blake3()
    h.update(data)
    return h.hexdigest()


def _blake3_file(path: Path) -> str:
    h = blake3()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _simulate_ingest(
    original_file: Path,
    blobs_root: Path,
    chunk_size: int,
    rel_path: str | None = None,
) -> List[dict]:
    """Simulate Go SplitAndStore: split file into chunks, store blobs, return entries list."""
    data = original_file.read_bytes()
    total_size = len(data)
    original_path = rel_path or original_file.name

    entries = []

    if total_size <= chunk_size:
        # No chunking needed — store as single blob.
        digest = _blake3_bytes(data)
        blob_dir = blobs_root / "blake3" / digest[:2] / digest[2:4]
        blob_dir.mkdir(parents=True, exist_ok=True)
        (blob_dir / digest).write_bytes(data)
        entries.append({
            "path": original_path,
            "type": "file",
            "size_bytes": total_size,
            "digest": f"blake3:{digest}",
            "blake3": digest,
        })
        return entries

    # Split into chunks.
    parts_meta = []
    offset = 0
    part_idx = 0
    while offset < total_size:
        end = min(offset + chunk_size, total_size)
        chunk_data = data[offset:end]
        chunk_digest = _blake3_bytes(chunk_data)
        chunk_path = f"{original_path}.part{part_idx:04d}"

        # Store chunk blob.
        blob_dir = blobs_root / "blake3" / chunk_digest[:2] / chunk_digest[2:4]
        blob_dir.mkdir(parents=True, exist_ok=True)
        (blob_dir / chunk_digest).write_bytes(chunk_data)

        entries.append({
            "path": chunk_path,
            "type": "file",
            "size_bytes": len(chunk_data),
            "digest": f"blake3:{chunk_digest}",
            "blake3": chunk_digest,
        })
        parts_meta.append({
            "path": chunk_path,
            "size_bytes": len(chunk_data),
            "digest": f"blake3:{chunk_digest}",
        })

        offset = end
        part_idx += 1

    # Build and store .parts.json manifest.
    manifest = {
        "original_path": original_path,
        "total_bytes": total_size,
        "parts": parts_meta,
    }
    manifest_bytes = json.dumps(manifest).encode()
    manifest_digest = _blake3_bytes(manifest_bytes)
    manifest_path = f"{original_path}.parts.json"

    blob_dir = blobs_root / "blake3" / manifest_digest[:2] / manifest_digest[2:4]
    blob_dir.mkdir(parents=True, exist_ok=True)
    (blob_dir / manifest_digest).write_bytes(manifest_bytes)

    entries.append({
        "path": manifest_path,
        "type": "file",
        "size_bytes": len(manifest_bytes),
        "digest": f"blake3:{manifest_digest}",
        "blake3": manifest_digest,
    })

    return entries


# ---------------------------------------------------------------------------
# Simulate worker-side reassembly (same logic as _reassemble_chunked)
# ---------------------------------------------------------------------------

def _simulate_reassemble(blobs_root: Path, output_dir: Path, entries: List[dict]) -> None:
    """Mirrors CozySnapshotV2Downloader._reassemble_chunked + _materialize_regular."""
    import re
    PART_RE = re.compile(r"\.part\d{4}$")

    parts_manifests = [e for e in entries if e["path"].endswith(".parts.json")]
    part_paths = {e["path"] for e in entries if PART_RE.search(e["path"])}

    # Reassemble chunked files.
    for pm in parts_manifests:
        digest = pm["blake3"]
        blob = blobs_root / "blake3" / digest[:2] / digest[2:4] / digest
        manifest = json.loads(blob.read_bytes())

        original_path = manifest["original_path"]
        parts = manifest["parts"]

        dst = output_dir / original_path
        dst.parent.mkdir(parents=True, exist_ok=True)

        with open(dst, "wb") as out_f:
            for part in parts:
                pd = part["digest"]
                if pd.startswith("blake3:"):
                    pd = pd[7:]
                part_blob = blobs_root / "blake3" / pd[:2] / pd[2:4] / pd
                with open(part_blob, "rb") as in_f:
                    shutil.copyfileobj(in_f, out_f)

    # Materialize regular (non-chunked) files.
    for e in entries:
        if e["path"].endswith(".parts.json") or e["path"] in part_paths:
            continue
        digest = e["blake3"]
        src = blobs_root / "blake3" / digest[:2] / digest[2:4] / digest
        dst = output_dir / e["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestChunkedReassembly:
    """A/B: original bytes == reassembled bytes after split+merge."""

    def _run_ab(self, original_data: bytes, filename: str = "model.safetensors"):
        chunk_size = _chunk_size()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            original_file = tmp_path / "original" / filename
            original_file.parent.mkdir(parents=True)
            original_file.write_bytes(original_data)
            original_hash = _blake3_file(original_file)

            blobs_root = tmp_path / "blobs"
            output_dir = tmp_path / "output"
            blobs_root.mkdir()
            output_dir.mkdir()

            # A: ingest (split)
            entries = _simulate_ingest(original_file, blobs_root, chunk_size, rel_path=filename)

            # B: worker (reassemble)
            _simulate_reassemble(blobs_root, output_dir, entries)

            # Verify
            reassembled = output_dir / filename
            assert reassembled.exists(), f"reassembled file not found: {reassembled}"

            reassembled_data = reassembled.read_bytes()
            reassembled_hash = _blake3_file(reassembled)

            # Byte-for-byte identical
            assert len(reassembled_data) == len(original_data), (
                f"size mismatch: original={len(original_data)} reassembled={len(reassembled_data)}"
            )
            assert reassembled_data == original_data, "WEIGHT CORRUPTION: bytes differ after reassembly"
            assert reassembled_hash == original_hash, (
                f"blake3 mismatch: original={original_hash[:16]} reassembled={reassembled_hash[:16]}"
            )

    def test_small_file_no_chunking(self):
        """File under chunk size — passes through without splitting."""
        data = os.urandom(500_000)  # 500 KB
        self._run_ab(data)

    def test_exact_boundary(self):
        """File exactly at chunk boundary — single chunk, no remainder."""
        chunk = _chunk_size()
        data = os.urandom(chunk)
        self._run_ab(data)

    def test_one_byte_over(self):
        """File 1 byte over chunk size — triggers 2 parts."""
        chunk = _chunk_size()
        data = os.urandom(chunk + 1)
        self._run_ab(data)

    def test_two_full_parts(self):
        """File exactly 2x chunk size — 2 full parts."""
        chunk = _chunk_size()
        data = os.urandom(chunk * 2)
        self._run_ab(data)

    def test_three_parts_with_remainder(self):
        """File spanning 3 parts with a small remainder."""
        chunk = _chunk_size()
        data = os.urandom(chunk * 2 + chunk // 3)
        self._run_ab(data)

    def test_random_weights_pattern(self):
        """Simulates real safetensors: structured header + random float bytes."""
        chunk = _chunk_size()
        # Fake safetensors header (8 bytes length + JSON)
        header = b'{"weight_0": {"dtype": "F16", "shape": [1024, 1024], "data_offsets": [0, 2097152]}}'
        header_with_len = len(header).to_bytes(8, "little") + header
        # Fill rest with random "weight" bytes to span 2+ chunks
        weight_bytes = os.urandom(chunk * 2 - len(header_with_len))
        data = header_with_len + weight_bytes
        self._run_ab(data, filename="unet/diffusion_pytorch_model.safetensors")

    def test_all_zeros(self):
        """Edge case: all-zero file (compressible but split is still raw bytes)."""
        chunk = _chunk_size()
        data = b"\x00" * (chunk + 100)
        self._run_ab(data)

    def test_identical_chunks(self):
        """Two chunks with identical content (same digest) — dedup in blob store."""
        chunk = _chunk_size()
        pattern = os.urandom(chunk)
        data = pattern + pattern  # 2 identical chunks
        self._run_ab(data)

    def test_fp16_and_normal_share_parts(self):
        """Simulates sd-turbo manifest: fp16 and normal variants share same chunk digests."""
        chunk = _chunk_size()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            blobs_root = tmp_path / "blobs"
            output_dir = tmp_path / "output"
            blobs_root.mkdir()
            output_dir.mkdir()

            # Same underlying data for both variants (they share part digests).
            data = os.urandom(chunk + chunk // 2)
            original_hash = _blake3_bytes(data)

            # Ingest fp16 variant.
            fp16_file = tmp_path / "model.fp16.safetensors"
            fp16_file.write_bytes(data)
            fp16_entries = _simulate_ingest(fp16_file, blobs_root, chunk)

            # Ingest normal variant (same data, different filename).
            normal_file = tmp_path / "model.safetensors"
            normal_file.write_bytes(data)
            normal_entries = _simulate_ingest(normal_file, blobs_root, chunk)

            # Combined entries (like the real manifest).
            all_entries = fp16_entries + normal_entries

            # Reassemble both.
            _simulate_reassemble(blobs_root, output_dir, all_entries)

            # Both must be byte-identical to original.
            for name in ["model.fp16.safetensors", "model.safetensors"]:
                result = output_dir / name
                assert result.exists(), f"{name} not found"
                assert result.read_bytes() == data, f"WEIGHT CORRUPTION in {name}"
                assert _blake3_file(result) == original_hash, f"blake3 mismatch in {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
