from __future__ import annotations

import struct
import zipfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from gen_worker._vendor.torchcg import CallIngress, CallInput
from gen_worker._vendor.torchcg.identity import artifact_key, contiguous_handle, host_facts
from gen_worker._vendor.torchcg.mint import compile_policy
from gen_worker._vendor.torchcg.store import pack, unpack

TOOLCHAIN: Dict[str, str] = {"torch": "record-digest", "triton": "compiler-digest"}

GRAPH_CLASS = "denoiser/h=64,w=64"
TARGET = "unet"


#: A graph hash the fixture's artifacts claim. Shaped like a real one because
#: `is_graph_hash` refuses anything else at the document boundary.
GRAPH = "cg-graph-v1-" + "a" * 56


def host_isa() -> Dict[str, str]:
    """This machine's own ISA facts — the only ones TCG will admit here."""
    return host_facts()


def aoti_package(path: Path, *, graph_specialization: str = GRAPH_CLASS,
                 filler: str = "") -> Path:
    """A code-only AOTInductor package TCG's introspection accepts."""
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
    """Metadata in the shape a real mint stamps, with a key derived the same way.

    tcg#90: the old `graph_specialization` block, `graph_witness`,
    `range_digest` and `host_isa` sections are gone — an artifact now carries
    ONE key over (graph x env x layout x policy x sm) and the ingress directly.
    `witness` survives as a parameter so a caller can make two artifacts differ;
    it rides in the graph hash, which is where "these are different graphs"
    belongs.
    """

    ingress = CallIngress(
        parameters=("value",),
        flat_arity=1,
        inputs=(CallInput("value", 0, "value", 0, (), "value", "float32", (2,)),),
    )
    env = {**dict(toolchain or TOOLCHAIN), **host_isa()}
    graph = "cg-graph-v1-" + (witness * 8)[:56]
    policy = compile_policy("cuda" if sm.startswith("sm_") else "cpu")
    key = artifact_key(
        graph, sm=sm, env=env, policy=policy, layout=contiguous_handle()
    ).value
    return {
        "kind": "aot-inductor",
        "key": key,
        "graph": graph,
        "name": graph_specialization,
        "target": TARGET,
        "sm": sm,
        "env": env,
        "compile_policy": policy,
        "declared_input_layout": contiguous_handle(),
        "placement": ["cuda:0"] if sm.startswith("sm_") else ["cpu"],
        "ingress": ingress.as_dict(),
        "passes": [],
        "constants": {"literal": [], "state": []},
    }


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
    import json
    import tempfile

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tcg-fixture-") as raw:
        staging = Path(raw)
        aoti_package(
            staging / "model.pt2",
            graph_specialization=graph_specialization,
            filler=filler,
        )
        (staging / "metadata.json").write_text(json.dumps(metadata(
            graph_specialization=graph_specialization, witness=witness, sm=sm,
            toolchain=toolchain,
        ), sort_keys=True, indent=2))
        return pack(staging, output)


def key_of(artifact: Path) -> str:
    """The key the artifact states about itself."""
    import json
    import tempfile

    with tempfile.TemporaryDirectory(prefix="tcg-key-") as raw:
        directory = unpack(artifact, Path(raw))
        stamped = json.loads((directory / "metadata.json").read_text())
    return str(stamped.get("key") or "")


def unpacked(
    destination: Path,
    *,
    graph_specialization: str = GRAPH_CLASS,
    witness: str = "fedcba9876543210",
    sm: str = "sm_89",
    toolchain: Optional[Mapping[str, str]] = None,
) -> Path:
    """A real UNPACKED artifact directory at ``destination``."""
    envelope = destination.parent / f".{destination.name}.envelope.tar.gz"
    build(
        envelope,
        graph_specialization=graph_specialization,
        witness=witness,
        sm=sm,
        toolchain=toolchain,
    )
    try:
        unpack(envelope, destination)
    finally:
        envelope.unlink(missing_ok=True)
    return destination
