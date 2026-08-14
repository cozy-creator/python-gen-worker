"""Modular hydration loads every component at ITS OWN checkpoint dtype.

Deriving ONE dtype from a majority vote over every safetensors header in the
snapshot and applying it to every component breaks both ways:

* vote falls outside the sniff's vocabulary (a tree whose wide parts
  outnumber the stack, or unreadable headers) -> no dtype at all ->
  diffusers' fp32 default upcasts the narrow stack. On minimax-h3 that put a
  66.28 GB bf16 checkpoint at 74.9 GiB resident and OOM'd an H100.
* vote falls narrow -> a component the checkpoint stores WIDE is silently
  truncated. H3's VAEs carry the opposite instruction in their own source
  ("a pipeline-level torch_dtype=bfloat16 must not downcast the weights").

Real diffusers modular pipelines and real safetensors throughout, under
``HF_HUB_OFFLINE=1``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
from safetensors.torch import save_file

from gen_worker import activity as activity_mod
from gen_worker.models.loading import (
    _load_modular_pipeline,
    checkpoint_load_dtype,
    hydrate_modular_pipeline,
)

from harness.modular_endpoint import (
    TinyModularPipeline,
    build_base_tree,
    build_mixed_precision_tree,
    build_override_vae_tree,
    tiny_vae,
)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")


def _dtypes(module: Any) -> Dict[str, str]:
    return {n: str(t.dtype) for n, t in
            list(module.named_parameters()) + list(module.named_buffers())}


def _bytes_per_param(module: Any) -> float:
    total = sum(p.numel() for p in module.parameters())
    return sum(p.numel() * p.element_size() for p in module.parameters()) / total


def _checkpoint_dtypes(component: Path) -> Dict[str, str]:
    """Per-tensor dtypes as the component's own safetensors headers store
    them — the thing the load has to equal."""
    out: Dict[str, str] = {}
    for f in sorted(component.rglob("*.safetensors")):
        with open(f, "rb") as fh:
            import struct

            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n))
        for key, value in header.items():
            if isinstance(value, dict) and "dtype" in value:
                out[key] = str(value["dtype"])
    return out


_SAFETENSORS_TO_TORCH = {"BF16": "torch.bfloat16", "F16": "torch.float16",
                         "F32": "torch.float32"}


# ---------------------------------------------------------------------------
# the deliverable: per-tensor dtypes equal the checkpoint's
# ---------------------------------------------------------------------------


def test_a_mixed_tree_hydrates_at_the_checkpoints_own_per_tensor_dtypes(
    tmp_path,
) -> None:
    """The H3 shape end to end: bf16 stack + fp32 keep-in-fp32 heads in one
    component, a wide VAE beside it, no declared composition dtype."""
    tree = build_mixed_precision_tree(tmp_path / "base")

    pipe = _load_modular_pipeline(TinyModularPipeline, str(tree))

    for name in ("unet", "vae"):
        want = {k: _SAFETENSORS_TO_TORCH[v]
                for k, v in _checkpoint_dtypes(tree / name).items()}
        got = _dtypes(getattr(pipe, name))
        assert want, name
        assert {k: got[k] for k in want} == want, name


def test_the_bf16_stack_is_not_upcast_when_the_tree_vote_falls_wide(
    tmp_path,
) -> None:
    """ie#615's wall: with the wide parts outnumbering the stack, the
    tree-wide vote produced NO dtype and diffusers' fp32 default doubled the
    denoiser — 4 bytes/param against a 2-byte checkpoint."""
    tree = build_mixed_precision_tree(tmp_path / "base", extra_fp32_parts=3)

    pipe = _load_modular_pipeline(TinyModularPipeline, str(tree))

    stack = {n: d for n, d in _dtypes(pipe.unet).items() if "conv_in" not in n}
    assert set(stack.values()) == {"torch.bfloat16"}, stack
    # the ie#615 metric itself, not a proxy for it
    assert _bytes_per_param(pipe.unet) < 2.1


def test_a_wide_component_is_not_truncated_by_a_narrow_tree_vote(
    tmp_path,
) -> None:
    """The other half: an fp32 VAE inside a bf16-majority tree keeps its
    precision. Upcasting it after the fact recovers nothing."""
    tree = build_mixed_precision_tree(tmp_path / "base", extra_fp32_parts=0)

    pipe = _load_modular_pipeline(TinyModularPipeline, str(tree))

    assert set(_dtypes(pipe.vae).values()) == {"torch.float32"}


def test_the_keep_in_fp32_heads_survive_the_narrow_stack(tmp_path) -> None:
    """``_keep_in_fp32_modules`` is diffusers' business — and naming a dtype
    is what lets it act at all. Passing nothing (the old undeclared path)
    upcast the whole module instead."""
    tree = build_mixed_precision_tree(tmp_path / "base")

    pipe = _load_modular_pipeline(TinyModularPipeline, str(tree))

    assert set(_dtypes(pipe.unet.conv_in).values()) == {"torch.float32"}


# ---------------------------------------------------------------------------
# a DECLARATION still governs; a sniff never overrides one
# ---------------------------------------------------------------------------


def test_a_declared_composition_dtype_governs_every_component(
    tmp_path,
) -> None:
    tree = build_mixed_precision_tree(tmp_path / "base")

    pipe = _load_modular_pipeline(TinyModularPipeline, str(tree), dtype="fp16")

    assert set(_dtypes(pipe.vae).values()) == {"torch.float16"}
    stack = {n: d for n, d in _dtypes(pipe.unet).items() if "conv_in" not in n}
    assert set(stack.values()) == {"torch.float16"}


def test_a_declared_scalar_reaches_hydration_unchanged(tmp_path) -> None:
    tree = build_mixed_precision_tree(tmp_path / "base")
    pipe = TinyModularPipeline.from_pretrained(str(tree))

    hydrate_modular_pipeline(pipe, tree, torch_dtype=torch.bfloat16)

    assert set(_dtypes(pipe.vae).values()) == {"torch.bfloat16"}


def test_a_declared_map_routes_per_component_and_the_rest_read_disk(
    tmp_path,
) -> None:
    """diffusers' own ``{"part": ...}`` shape: what it names is declared,
    what it does not still comes off that component's own bytes."""
    tree = build_mixed_precision_tree(tmp_path / "base")
    pipe = TinyModularPipeline.from_pretrained(str(tree))

    hydrate_modular_pipeline(pipe, tree, torch_dtype={"vae": torch.float16})

    assert set(_dtypes(pipe.vae).values()) == {"torch.float16"}
    stack = {n: d for n, d in _dtypes(pipe.unet).items() if "conv_in" not in n}
    assert set(stack.values()) == {"torch.bfloat16"}


# ---------------------------------------------------------------------------
# the source of the dtype is the component's OWN dir
# ---------------------------------------------------------------------------


def test_an_override_tree_is_read_for_its_own_dtype(tmp_path) -> None:
    """A th#980/pgw#617 override is a different artifact from the base dir it
    replaces — the base tree's bytes say nothing about it."""
    tree = build_mixed_precision_tree(tmp_path / "base", vae_dtype="bf16")
    override = tmp_path / "ovr"
    tiny_vae(2.0).to(torch.float32).save_pretrained(override / "vae")
    pipe = TinyModularPipeline.from_pretrained(str(tree))

    hydrate_modular_pipeline(
        pipe, tree, component_trees={"vae": str(override)})

    assert set(_dtypes(pipe.vae).values()) == {"torch.float32"}


def test_the_hydration_event_carries_the_dtype_each_component_landed_at(
    tmp_path, monkeypatch,
) -> None:
    """The upcast was invisible for three deploy cycles. The dtype is
    hub-visible evidence now, not a log line."""
    seen: list = []
    monkeypatch.setattr(
        activity_mod, "emit_event",
        lambda kind, detail="", phase="", duration_ms=0: seen.append(
            (kind, detail)))
    tree = build_mixed_precision_tree(tmp_path / "base")
    pipe = TinyModularPipeline.from_pretrained(str(tree))

    hydrate_modular_pipeline(pipe, tree)

    detail = next(d for k, d in seen
                  if k == activity_mod.KIND_MODULAR_HYDRATION)
    assert "unet=bfloat16" in detail and "vae=float32" in detail


# ---------------------------------------------------------------------------
# the token map, including the one entry that is not identity
# ---------------------------------------------------------------------------


def test_fp8_storage_asks_for_its_compute_dtype_not_for_fp8(tmp_path) -> None:
    """An fp8-stored component is a quantized artifact: it carries its own
    config and bf16 is the compute dtype over it. Handing torch fp8 as a
    module dtype is not a thing the loaders do."""
    comp = tmp_path / "fp8"
    comp.mkdir()
    save_file({"w": torch.zeros(4, 4, dtype=torch.float8_e4m3fn)},
              str(comp / "model.safetensors"))

    assert checkpoint_load_dtype(comp) == "bf16"


def test_headers_that_say_nothing_declare_nothing(tmp_path) -> None:
    """No sniff, no dtype — the loader keeps its own default rather than
    this code inventing one."""
    (tmp_path / "empty").mkdir()

    assert checkpoint_load_dtype(tmp_path / "empty") == ""


def test_each_stored_precision_maps_to_itself(tmp_path) -> None:
    for token, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16),
                         ("fp32", torch.float32)):
        comp = tmp_path / token
        comp.mkdir()
        save_file({"w": torch.zeros(4, 4, dtype=dtype)},
                  str(comp / "model.safetensors"))
        assert checkpoint_load_dtype(comp) == token


# ---------------------------------------------------------------------------
# nothing changes on the uniform trees (the common case)
# ---------------------------------------------------------------------------


def test_a_uniform_tree_is_unaffected(tmp_path) -> None:
    tree = build_base_tree(tmp_path / "base")
    override = build_override_vae_tree(tmp_path / "ovr")

    pipe = _load_modular_pipeline(
        TinyModularPipeline, str(tree),
        component_trees={"vae": str(override)})

    assert set(_dtypes(pipe.unet).values()) == {"torch.float32"}
    assert set(_dtypes(pipe.vae).values()) == {"torch.float32"}
