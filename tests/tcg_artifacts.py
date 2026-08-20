"""Real ``torchcg`` artifacts, built without torch (pgw#1283).

The local compiled graph store used to hold OPAQUE bytes — it hashed whatever it was
handed and re-hashed it on the way out — so every test about it could feed it
``b"packed-compiled graph-bytes"``. pgw#1283 moved byte custody to TCG, which admits an
artifact only if it unpacks, restates its own key, and names a host ISA this
machine can run. Opaque bytes are now correctly refused, so the tests need the
real envelope.

Building one needs no torch and no GPU: the envelope is a gzip+USTAR tar of a
metadata block and a code-only AOTInductor ``.pt2`` zip, and TCG's introspection
reads the wrapper source and the ELF section table rather than loading either.
The ELF and wrapper shapes below are the minimum ``torchcg``
accepts; they are ported from TCG's own ``tests/test_artifact.py`` fixture,
which is the authority on that shape.

``host_isa`` is taken from :func:`torchcg.host_isa._host_requirement`
— a private symbol, deliberately: an artifact stamped with a hard-coded
``x86-64-v3`` would be refused on an arm64 runner, and a fixture whose
admissibility depends on which machine ran it is a flake, not a fence.
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from gen_worker._vendor.torchcg import CallIngress, CallInput, GraphSpecializationDeclaration
from gen_worker._vendor.torchcg.artifact import build_metadata, pack_artifact
from gen_worker._vendor.torchcg.host_isa import _host_requirement

#: The default compiler-content facts. Two artifacts that differ only here get
#: different ``ck1`` keys, which is how a test asks for a second compiled graph.
TOOLCHAIN: Dict[str, str] = {"torch": "record-digest", "triton": "compiler-digest"}

GRAPH_CLASS = "denoiser/h=64,w=64"
TARGET = "unet"


def host_isa() -> Dict[str, str]:
    """This machine's own ISA facts — the only ones TCG will admit here."""
    return _host_requirement().facts()


def aoti_package(path: Path, *, graph_specialization: str = GRAPH_CLASS,
                 filler: str = "") -> Path:
    """A code-only AOTInductor package TCG's introspection accepts.

    ``filler`` appends an inert comment to the wrapper source: it changes the
    package's BYTES without changing a single fact TCG derives from it, which
    is the only way to build two artifacts that state the same key over
    different content — i.e. TCG's ``DIVERGENT`` case.
    """
    names = b"\0.shstrtab\0.lrodata\0"
    section_offset = 64
    section_size = 64
    section_count = 3
    string_offset = section_offset + section_size * section_count
    payload_offset = string_offset + len(names)
    shared_object = bytearray(payload_offset)
    shared_object[:4] = b"\x7fELF"
    shared_object[4:7] = bytes((2, 1, 1))
    struct.pack_into("<Q", shared_object, 0x28, section_offset)
    struct.pack_into("<HHH", shared_object, 0x3A, section_size, section_count, 1)
    struct.pack_into("<II", shared_object, section_offset + section_size, 1, 3)
    struct.pack_into(
        "<QQ", shared_object, section_offset + section_size + 0x18,
        string_offset, len(names))
    struct.pack_into("<II", shared_object, section_offset + 2 * section_size, 11, 1)
    struct.pack_into(
        "<QQ", shared_object, section_offset + 2 * section_size + 0x18,
        payload_offset, 0)
    shared_object[string_offset:payload_offset] = names
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        root = f"data/aotinductor/{graph_specialization}"
        wrapper = (
            "AOTInductorModelBase(1, 1, 0, device_str, std::move(cubin_dir), false)")
        if filler:
            wrapper += f"\n// {filler}"
        archive.writestr(f"{root}/model.wrapper.cpp", wrapper)
        archive.writestr(f"{root}/model.so", bytes(shared_object))
    return path


def metadata(
    *,
    graph_specialization: str = GRAPH_CLASS,
    witness: str = "fedcba9876543210",
    sm: str = "sm_89",
    toolchain: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Metadata whose stamped ``compiled_graph_key`` TCG derives from itself."""
    ingress = CallIngress(
        parameters=("value",),
        flat_arity=1,
        inputs=(CallInput("value", 0, "value", 0, (), "value", "float32", (2,)),),
    )
    graph: Dict[str, Any] = {
        "v": 4,
        "constant_fqns": [],
        "ingress": ingress.as_dict(),
    }
    declaration = GraphSpecializationDeclaration(
        name=graph_specialization,
        target=TARGET,
        graph=graph,
        graph_witness=witness,
        range_digest=ingress.digest(),
        literal_values="",
    )
    return build_metadata(
        graph_specialization={
            "name": declaration.name,
            "target": declaration.target,
            "specialization_hash": declaration.specialization_hash,
            "graph": dict(declaration.graph),
            "graph_witness": declaration.graph_witness,
            "range_digest": declaration.range_digest,
            "fork": [],
            "specialization_dims": [],
            "strict": True,
            "lora_bucket": 0,
            "literal_values": declaration.literal_values,
            "literal_payload_values": "",
            "placement": list(declaration.placement),
            "constants": [],
        },
        sm=sm,
        toolchain=dict(toolchain or TOOLCHAIN),
        host_isa=host_isa(),
    )


def build(
    output: Path,
    *,
    graph_specialization: str = GRAPH_CLASS,
    witness: str = "fedcba9876543210",
    sm: str = "sm_89",
    toolchain: Optional[Mapping[str, str]] = None,
    filler: str = "",
) -> Path:
    """One real artifact at ``output``; its key is stamped in its metadata."""
    package = output.parent / f".{output.name}.pt2"
    output.parent.mkdir(parents=True, exist_ok=True)
    aoti_package(package, graph_specialization=graph_specialization, filler=filler)
    try:
        return pack_artifact(
            package, output,
            metadata(graph_specialization=graph_specialization, witness=witness, sm=sm,
                     toolchain=toolchain))
    finally:
        package.unlink(missing_ok=True)


def key_of(artifact: Path) -> str:
    """The ``ck1`` key the artifact states about itself."""
    from gen_worker._vendor.torchcg.artifact import read_metadata

    return str(read_metadata(artifact).get("compiled_graph_key") or "")


def unpacked(
    destination: Path,
    *,
    graph_specialization: str = GRAPH_CLASS,
    witness: str = "fedcba9876543210",
    sm: str = "sm_89",
    toolchain: Optional[Mapping[str, str]] = None,
) -> Path:
    """A real UNPACKED artifact directory at ``destination``.

    What ``Engine.compile`` leaves behind (``metadata.json`` + ``model.pt2``),
    and since pgw#1561 the only builder output ``publish_compiled`` accepts —
    the publish repacks it into the ENVELOPE the adopt loader reads, so a
    bare ``model.pt2`` in a directory with no metadata is now unrepresentable
    on the publish path rather than silently banked.

    ``destination`` must not exist yet: ``unpack_artifact`` materializes
    atomically and refuses an occupied target, exactly like the engine.
    """
    from gen_worker._vendor.torchcg.artifact import unpack_artifact

    envelope = destination.parent / f".{destination.name}.envelope.tar.gz"
    build(
        envelope,
        graph_specialization=graph_specialization,
        witness=witness,
        sm=sm,
        toolchain=toolchain,
    )
    try:
        unpack_artifact(envelope, destination)
    finally:
        envelope.unlink(missing_ok=True)
    return destination
