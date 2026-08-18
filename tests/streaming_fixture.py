"""A real bf16 diffusers pipeline on disk — the article both pgw#1380 suites load.

Real classes, real ``save_pretrained``, real safetensors. Nothing here is
shaped to be convenient for the loader: ``text_encoder`` and ``text_encoder_2``
share every parameter name (which is what a pipeline checkpoint looks like and
what a globally-indexed reader collides on), and the weights are randomly
initialized rather than filled, so a byte-equality assertion can actually see a
wrong offset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


def tiny_pipeline_class() -> type:
    from diffusers import DiffusionPipeline

    class TinyPipeline(DiffusionPipeline):
        """A real ``DiffusionPipeline``: real components, real model_index."""

        def __init__(self, unet: Any, vae: Any, text_encoder: Any,
                     text_encoder_2: Any, scheduler: Any) -> None:
            super().__init__()
            self.register_modules(  # type: ignore[attr-defined]
                unet=unet, vae=vae, text_encoder=text_encoder,
                text_encoder_2=text_encoder_2, scheduler=scheduler,
            )

    return TinyPipeline


def build_source(target: Path) -> type:
    """Save a real bf16 pipeline to ``target`` and return its class."""
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel

    torch.manual_seed(1380)
    unet = UNet2DConditionModel(
        sample_size=16, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(32, 64), norm_num_groups=4, cross_attention_dim=32,
        attention_head_dim=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
    )
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=4, norm_num_groups=4,
        block_out_channels=(32,), down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
    )
    text_config = CLIPTextConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, vocab_size=256, max_position_embeddings=32,
        projection_dim=32,
    )
    pipeline_cls = tiny_pipeline_class()
    pipeline = pipeline_cls(
        unet=unet, vae=vae,
        text_encoder=CLIPTextModel(text_config),
        text_encoder_2=CLIPTextModel(text_config),
        scheduler=DDIMScheduler(),
    )
    # bf16 on disk, so "dtype passthrough" is a claim with something to pass
    # through: nothing in the engine names a dtype, and bf16 must come back
    # bf16 without a `torch_dtype=` anywhere.
    pipeline.to(torch.bfloat16)
    pipeline.save_pretrained(str(target), safe_serialization=True)
    for container in sorted(target.rglob("*.safetensors")):
        scramble_offsets(container)
    return pipeline_cls


def scramble_offsets(path: Path) -> None:
    """Rewrite a container so its HEADER order is not its OFFSET order.

    ``safetensors`` writes tensors in header-key order, so a checkpoint saved
    by the library has name order == offset order — and against such a file a
    loader that walked in NAME order would look perfectly sequential. That is
    the exact trap tensorfs#115 named ("assert order against a fixture whose
    header order differs from offset order"), and without this the file-order
    claim is untestable: a mutation that scrambles the walk stays green.

    The header keys keep their sorted order; the BODY is laid out in reverse.
    Both orders are legal safetensors — the format addresses by
    ``data_offsets``, not by position — so the library still reads it, which
    is what keeps the byte-equality control arm honest.
    """
    import struct

    raw = path.read_bytes()
    (size,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + size])
    body = raw[8 + size :]

    names = [key for key in header if key != "__metadata__"]
    if len(names) < 2:
        return
    payloads = {
        name: body[header[name]["data_offsets"][0] : header[name]["data_offsets"][1]]
        for name in names
    }
    rebuilt = bytearray()
    for name in reversed(names):
        start = len(rebuilt)
        rebuilt += payloads[name]
        header[name]["data_offsets"] = [start, len(rebuilt)]
    blob = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(rebuilt))


def header_order_differs_from_offset_order(path: Path) -> bool:
    """True when this container can witness a name-ordered walk."""
    import struct

    raw = path.read_bytes()
    (size,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + size])
    ordered = [key for key in header if key != "__metadata__"]
    offsets = [header[name]["data_offsets"][0] for name in ordered]
    return offsets != sorted(offsets)


def source_tensors(source: Path) -> Dict[str, Dict[str, Any]]:
    """component -> {name: tensor}, read from the ORIGINAL files."""
    from safetensors.torch import load_file

    index = json.loads((source / "model_index.json").read_text())
    found: Dict[str, Dict[str, Any]] = {}
    for component, spec in index.items():
        if component.startswith("_") or not isinstance(spec, list):
            continue
        directory = source / component
        shards = sorted(directory.glob("*.safetensors")) if directory.is_dir() else []
        if not shards:
            continue
        merged: Dict[str, Any] = {}
        for shard in shards:
            merged.update(load_file(str(shard)))
        found[component] = merged
    return found


def assert_byte_equal(pipeline: Any, source: Path) -> int:
    """Every parameter byte-equal to its source file. Returns the tensor count."""
    checked = 0
    for component, tensors in source_tensors(source).items():
        module = getattr(pipeline, component)
        live = dict(module.named_parameters(remove_duplicate=False))
        live.update(dict(module.named_buffers(remove_duplicate=False)))
        for name, want in tensors.items():
            got = live.get(name)
            assert got is not None, f"{component}/{name} never landed"
            assert got.dtype == want.dtype, (
                f"{component}/{name}: {want.dtype} on disk came back "
                f"{got.dtype} — the loader converted, which is the store's job"
            )
            assert got.shape == want.shape
            assert torch.equal(
                got.reshape(-1).view(torch.uint8),
                want.reshape(-1).view(torch.uint8),
            ), f"{component}/{name} is not byte-equal to the source"
            checked += 1
    return checked


class Lane:
    """The minimum of a lane contract the engine reads: its dtype."""

    dtype = torch.bfloat16


# -- read tracing -----------------------------------------------------------


class TracedStream:
    def __init__(self, inner: Any, container: str,
                 log: List[Tuple[str, int, int]]) -> None:
        self._inner = inner
        self._container = container
        self._log = log

    @property
    def tensors(self) -> Any:
        return self._inner.tensors

    @property
    def length(self) -> int:
        return self._inner.length

    def readinto(self, offset: int, length: int, buffer: Any) -> int:
        self._log.append((self._container, int(offset), int(length)))
        return self._inner.readinto(offset, length, buffer)


class TracedStore:
    """Records every byte range the engine asks the store for."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.reads: List[Tuple[str, int, int]] = []

    def containers(self) -> Any:
        return self._inner.containers()

    def open(self, container: str, *, direct: bool = False) -> Any:
        return TracedStream(
            self._inner.open(container, direct=direct), container, self.reads
        )

    def assert_file_order(self) -> int:
        """One forward pass per container, never a seek backwards.

        Returns how many windows were read, so a caller can insist the walk
        was actually multi-window — a single-window read is trivially ordered
        and proves nothing about the ordering.
        """
        assert self.reads, "the engine read nothing"
        per_container: Dict[str, List[Tuple[int, int]]] = {}
        for container, offset, length in self.reads:
            per_container.setdefault(container, []).append((offset, length))
        for container, reads in per_container.items():
            for (offset, length), (nxt, _) in zip(reads, reads[1:]):
                assert nxt >= offset + length, (
                    f"{container}: a read at {nxt} follows one ending at "
                    f"{offset + length} — the walk is not in file order"
                )
        return len(self.reads)


def write_bytes_now() -> int:
    for line in Path("/proc/self/io").read_text().splitlines():
        if line.startswith("write_bytes:"):
            return int(line.split()[1])
    raise AssertionError("/proc/self/io has no write_bytes")
