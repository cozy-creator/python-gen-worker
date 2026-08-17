"""A real CAS, a real manifest, and a real projected tree — no mocks.

Everything under pgw#1330 is about what a consumer does when it meets a tree
whose tensor bytes are not at any file path. A fake tree proves nothing about
that, so this builds the article: bytes ingested into a real ``LocalCAS``, the
manifest pinned exactly as ``cozy_snapshot.ensure_snapshot`` pins it, and the
tree projected by the same ``project_snapshot`` the chokepoint will call.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple

from gen_worker._vendor.tensorfs import (
    LocalCAS,
    RepositoryManifest,
    project_snapshot,
    read_entry,
)
from gen_worker.models.projection import REF_PREFIX, SNAPSHOTS_DIR

# safetensors dtype -> (itemsize, struct format for one element)
_DTYPES: dict[str, int] = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1}


def safetensors_bytes(
    tensors: Mapping[str, Tuple[str, Sequence[int], bytes]]
) -> bytes:
    """One safetensors file, written by hand so the fixture owns the bytes.

    ``tensors`` maps name -> (dtype, shape, payload). The payload is real
    content, never a constant fill: a byte-exactness assertion over a constant
    cannot see a wrong offset (DONE-STANDARD, "fixture shape").
    """

    header: dict[str, object] = {}
    body = bytearray()
    for name, (dtype, shape, payload) in tensors.items():
        expected = _DTYPES[dtype]
        for dim in shape:
            expected *= dim
        assert len(payload) == expected, f"{name}: {len(payload)} bytes, want {expected}"
        start = len(body)
        body += payload
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(body)],
        }
    blob = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<Q", len(blob)) + blob + bytes(body)


def varied(nbytes: int, seed: int) -> bytes:
    """Non-constant, deterministic payload bytes."""

    return bytes(((seed * 31 + i * 17 + (i >> 3)) & 0xFF) for i in range(nbytes))


@dataclass(frozen=True)
class Fixture:
    """A projected snapshot and everything needed to check it."""

    base: Path
    source: Path
    tree: Path
    key: str
    cas: LocalCAS
    manifest: RepositoryManifest

    def snapshot_message(self) -> "_Snapshot":
        """The duck-typed ``pb.Snapshot`` the store's verifier consumes."""

        return _Snapshot(
            digest="sha256:" + self.key,
            files=tuple(
                _SnapshotFile(
                    path=entry.path,
                    digest=str(entry.digest),
                    size_bytes=entry.size_bytes,
                )
                for entry in self.manifest.files
            ),
        )

    def weight_paths(self) -> List[Path]:
        return sorted(self.tree.rglob("*.safetensors"))


@dataclass(frozen=True)
class _SnapshotFile:
    path: str
    digest: str
    size_bytes: int


@dataclass(frozen=True)
class _Snapshot:
    digest: str
    files: Tuple[_SnapshotFile, ...]


def write_model(source: Path, *, dtype: str = "BF16") -> None:
    """A three-component diffusers-shaped model: configs plus real shards."""

    source.mkdir(parents=True, exist_ok=True)
    (source / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "TinyPipeline",
                "unet": ["diffusers", "UNet2DConditionModel"],
                "text_encoder": ["transformers", "CLIPTextModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            },
            indent=2,
        )
    )
    item = _DTYPES[dtype]
    for component, shards in (
        ("unet", ("diffusion_pytorch_model.safetensors",)),
        ("text_encoder", ("model.safetensors",)),
        ("vae", ("diffusion_pytorch_model.safetensors",)),
    ):
        directory = source / component
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(
            json.dumps({"_class_name": component, "hidden": 8}, indent=2)
        )
        for index, shard in enumerate(shards):
            seed = abs(hash((component, shard))) % 251 + 1
            payload = {
                f"{component}.block.{index}.weight": (
                    dtype, (4, 8), varied(4 * 8 * item, seed)
                ),
                f"{component}.block.{index}.bias": (
                    dtype, (8,), varied(8 * item, seed + 1)
                ),
            }
            (directory / shard).write_bytes(safetensors_bytes(payload))


def write_fp8_model(source: Path) -> None:
    """A w8a8 artifact: an fp8 ``transformer`` with the weight/scale pairing.

    This is the shape ``_quantized_layers`` sniffs for. Under stubs its header
    read returned ``{}``, ``quantized`` came back empty, and the artifact
    stopped being detected as quantized at all — routed to the plain bf16 lane.
    """

    source.mkdir(parents=True, exist_ok=True)
    (source / "model_index.json").write_text(
        json.dumps({"_class_name": "TinyPipeline", "transformer": ["diffusers", "X"]})
    )
    directory = source / "transformer"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"_class_name": "X"}))
    payload: dict[str, Tuple[str, Sequence[int], bytes]] = {}
    for index in range(2):
        name = f"blocks.{index}.linear"
        payload[f"{name}.weight"] = ("F8_E4M3", (8, 8), varied(64, 3 + index))
        payload[f"{name}.weight_scale"] = ("F32", (8,), varied(32, 9 + index))
    (directory / "diffusion_pytorch_model.safetensors").write_bytes(
        safetensors_bytes(payload)
    )


def build(
    base: Path, *, dtype: str = "BF16", key: str = "a" * 64, fp8: bool = False
) -> Fixture:
    """Ingest a model, pin its manifest, project the tree. No materialization."""

    source = base / "source-model"
    if fp8:
        write_fp8_model(source)
    else:
        write_model(source, dtype=dtype)
    cas = LocalCAS(base)
    manifest = cas.ingest_repository(source)
    # Exactly what `cozy_snapshot._pin_manifest` does, so `resolve_projection`
    # is exercised against the production pinning and not a test convention.
    cas.compare_and_swap_ref(REF_PREFIX + key, cas.store_manifest(manifest), expected=None)
    tree = base / SNAPSHOTS_DIR / key
    project_snapshot(cas, manifest, tree)
    return Fixture(base=base, source=source, tree=tree, key=key, cas=cas, manifest=manifest)


def materialize(base: Path, fixture: Fixture, name: str = "materialized") -> Path:
    """The same manifest as a tree of REAL files, for the control arm.

    Built with ``read_entry``, NOT with the single-file materialization hatch:
    the hatch is the thing this whole program is retiring, and new code reaching
    for it in a test is the example that spreads. ``read_entry`` also verifies
    the reconstruction against the manifest's whole-file digest, so the control
    arm is known-good bytes rather than merely copied ones.
    """

    target = base / SNAPSHOTS_DIR / name
    target.mkdir(parents=True, exist_ok=True)
    for entry in fixture.manifest.files:
        destination = target / entry.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(read_entry(fixture.cas, entry))
    return target


def iter_stubs(tree: Path) -> Iterable[Path]:
    from gen_worker.models.projection import stub_at

    for path in sorted(tree.rglob("*")):
        if path.is_file() and not path.is_symlink() and stub_at(path) is not None:
            yield path
