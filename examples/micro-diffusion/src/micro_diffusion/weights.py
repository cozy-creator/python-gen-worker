"""Deterministic weight GENERATION — the micro family never downloads bytes.

The weights-locality rule says model weights never transit the developer's
box. A micro endpoint that shipped a checkpoint would either violate that or
need a hub round trip on every cycle, so it has neither: the checkpoint is a
pure function of ``(seed, config)`` and is materialized wherever it is needed —
in the image at ``docker build`` time, on the pod at boot, or in the local rig.

Two runs with the same seed produce BYTE-IDENTICAL files. That is not a nicety:
the compiled graph key and the snapshot digest are claims about which checkpoint was
traced, and they mean nothing if the same ref resolves to different bytes on
two machines.

    python -m micro_diffusion.weights --out /app/.micro-weights
    python -m micro_diffusion.weights --out /tmp/w --seed 997 --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import load_file, save_file

from .model import MicroConfig, MicroDecoder, MicroDenoiser

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "micro_diffusion.safetensors"

#: The state-dict LAYOUT. Part of the cached tree's identity, so a component
#: rename invalidates the cache instead of leaving bytes no loader can read.
#: 3 (pgw#1080): the flat tree also carries a `model_index.json` and a
#: config-only dir per component, so the SDK can resolve each compile
#: target's CLASS and build it from config alone. No weights move.
LAYOUT_VERSION = 3

#: The one seed the fleet's micro family is defined by. Changing it is a new
#: checkpoint, not a new build of the same one.
SEED = 997

#: Refuse to write a tree larger than this. The whole point of the family is
#: that a cycle costs minutes; a checkpoint that grew would take the cycle
#: with it, silently.
MAX_TREE_BYTES = 200 * 1000 * 1000


class WeightsRefused(RuntimeError):
    """A named precondition failure while materializing the checkpoint."""


def state_dict(config: MicroConfig, *, seed: int = SEED) -> Dict[str, torch.Tensor]:
    """Every tensor of both targets, derived from ``seed`` alone.

    One generator, seeded once, consumed in a fixed module-construction
    order — so the mapping (seed, config) -> bytes is total and stable.
    """
    torch.manual_seed(int(seed))
    transformer = MicroDenoiser(config)
    decoder = MicroDecoder(config)
    out: Dict[str, torch.Tensor] = {}
    for prefix, module in (("transformer", transformer), ("decoder", decoder)):
        for name, tensor in module.state_dict().items():
            out[f"{prefix}.{name}"] = tensor.contiguous()
    return out


def materialize(
    root: Path, *, seed: int = SEED, config: MicroConfig | None = None,
) -> Path:
    """Write the checkpoint tree under ``root``; return it.

    Idempotent by CONFIG, not by mtime: a tree whose ``config.json`` already
    matches is left alone, so repeated boots and repeated rig cycles do not
    re-serialize.
    """
    root = Path(root)
    cfg = config or MicroConfig(seed=seed)
    # pgw#999: LAYOUT is part of the cache identity. `materialize` is
    # idempotent by config, and the config does not mention the state-dict
    # prefixes — so renaming a component left a stale tree on disk that the
    # new loader could not read. Bump this whenever a key changes name.
    payload = dict(cfg.as_dict(), _class_name="MicroPipeline",
                   _format="safetensors", _layout=LAYOUT_VERSION,
                   seed=int(seed))
    cfg_path = root / CONFIG_NAME
    if cfg_path.is_file() and (root / WEIGHTS_NAME).is_file():
        try:
            if json.loads(cfg_path.read_text()) == payload:
                return root
        except ValueError:
            pass
    root.mkdir(parents=True, exist_ok=True)
    save_file(state_dict(cfg, seed=seed), str(root / WEIGHTS_NAME))
    cfg_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_component_index(root, cfg, seed=seed)
    size = tree_bytes(root)
    if size > MAX_TREE_BYTES:
        raise WeightsRefused(
            f"the generated tree is {size / 1e6:.1f} MB, over the "
            f"{MAX_TREE_BYTES / 1e6:.0f} MB ceiling this family exists to stay "
            f"under — shrink the config rather than raising the ceiling")
    return root


#: ``component -> [library, class]``, the map every diffusers-layout tree
#: carries and the ONLY thing that tells the SDK which class a compile target
#: is. Without it a structure-only build has nothing to instantiate
#: (pgw#1080); the weights themselves stay in the flat file.
COMPONENT_CLASSES = {
    "transformer": ["micro_diffusion.model", "MicroDenoiser"],
    "decoder": ["micro_diffusion.model", "MicroDecoder"],
}


def write_component_index(
    root: Path, config: MicroConfig, *, seed: int = SEED,
) -> None:
    """Name each component's CLASS and give it a config to be built from.

    Config only — no weights are written here and none move. A component dir
    holding just ``config.json`` is the SDK's recognised weightless shape
    (``loading._weightless_model_dir``), so the real-weight load still reads
    the flat checkpoint exactly as before.
    """
    root = Path(root)
    payload = dict(config.as_dict(), _layout=LAYOUT_VERSION, seed=int(seed))
    for component, (_library, class_name) in COMPONENT_CLASSES.items():
        directory = root / component
        directory.mkdir(parents=True, exist_ok=True)
        (directory / CONFIG_NAME).write_text(json.dumps(
            dict(payload, _class_name=class_name), indent=2, sort_keys=True))
    (root / "model_index.json").write_text(json.dumps(
        dict({"_class_name": "MicroPipeline"}, **COMPONENT_CLASSES),
        indent=2, sort_keys=True))


def tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


def load_config(root: Path) -> MicroConfig:
    raw = json.loads((Path(root) / CONFIG_NAME).read_text())
    fields = set(MicroConfig().as_dict())
    return MicroConfig(**{k: v for k, v in raw.items() if k in fields})


def load_state(root: Path) -> Dict[str, torch.Tensor]:
    return load_file(str(Path(root) / WEIGHTS_NAME))


def _verify(root: Path, seed: int) -> int:
    """Regenerate into memory and compare — the determinism claim, checked."""
    on_disk = load_state(root)
    fresh = state_dict(load_config(root), seed=seed)
    if sorted(on_disk) != sorted(fresh):
        print("MISMATCH: tensor names differ", file=sys.stderr)
        return 1
    for name, tensor in fresh.items():
        if not torch.equal(tensor, on_disk[name]):
            print(f"MISMATCH: {name} differs from its regeneration",
                  file=sys.stderr)
            return 1
    print(f"verified {len(fresh)} tensors reproduce from seed={seed}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="micro_diffusion.weights")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--verify", action="store_true",
                        help="regenerate and byte-compare after writing")
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = materialize(args.out, seed=args.seed)
    print(f"{root} — {tree_bytes(root) / 1e6:.2f} MB generated (no download)")
    return _verify(root, args.seed) if args.verify else 0


if __name__ == "__main__":
    raise SystemExit(main())


#: The denoiser component name. From `gen_worker.component_vocab`'s denoiser
#: vocabulary — `quantize_tree_w8a8` and `detect_w8a8_artifacts` both scan for
#: a directory named from it, so a family whose denoiser is called anything
#: else is invisible to the whole w8a8 producer/consumer path (pgw#1014).
DENOISER_COMPONENT = "transformer"


def materialize_diffusers(
    root: Path, *, seed: int = SEED, config: MicroConfig | None = None,
) -> Path:
    """Write the checkpoint as a DIFFUSERS TREE, not a flat file.

    The flat single-file layout is what `MicroPipeline` normally reads, and it
    is enough for every lane except one: `w8a8.quantize_tree_w8a8` refuses
    anything that is not a diffusers tree (`model_index.json` plus a
    denoiser component directory named from the SDK's own vocabulary), and
    `detect_w8a8_artifacts` scans component dirs to find what was quantized.

    So the w8a8 leg needs this layout to run the REAL producer path rather
    than a hand-rolled quantizer. Same weights, same seed, different shape on
    disk.
    """
    root = Path(root)
    cfg = config or MicroConfig(seed=seed)
    state = state_dict(cfg, seed=seed)
    comp = root / DENOISER_COMPONENT
    dec = root / "decoder"
    comp.mkdir(parents=True, exist_ok=True)
    dec.mkdir(parents=True, exist_ok=True)

    payload = dict(cfg.as_dict(), _class_name="MicroDenoiser",
                   _layout=LAYOUT_VERSION, seed=int(seed))
    save_file({k[len("transformer."):]: v for k, v in state.items()
               if k.startswith("transformer.")},
              str(comp / "diffusion_pytorch_model.safetensors"))
    (comp / "config.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    save_file({k[len("decoder."):]: v for k, v in state.items()
               if k.startswith("decoder.")},
              str(dec / "diffusion_pytorch_model.safetensors"))
    (dec / "config.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    # `_denoiser_class` imports the [library, class] pair named here and only
    # falls back to `diffusers` on ImportError — so a first-party module is a
    # legal denoiser library, which is what lets a non-diffusers family use
    # this path at all.
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "MicroPipeline",
        DENOISER_COMPONENT: ["micro_diffusion.model", "MicroDenoiser"],
        "decoder": ["micro_diffusion.model", "MicroDecoder"],
    }, indent=2, sort_keys=True))
    return root


def materialize_w8a8(
    root: Path, *, seed: int = SEED, config: MicroConfig | None = None,
) -> Path:
    """The fp8 tree, through the SDK's OWN producer (`quantize_tree_w8a8`).

    Two steps and neither is hand-rolled: write the diffusers tree, then let
    the platform's quantizer rewrite the eligible denoiser weights to
    fp8 + per-out-channel `weight_scale` per the gw#534 contract. Eligibility
    is `2D, both dims 16-aligned, name not matching ("embed", "norm")` — this
    family's channel widths are 16-aligned FOR THAT REASON, so the leg tests
    real quantized layers instead of silently getting a dequantized tree.
    """
    from gen_worker.models.w8a8 import quantize_tree_w8a8

    root = Path(root)
    bf16 = root.parent / (root.name + "-bf16")
    materialize_diffusers(bf16, seed=seed, config=config)
    return quantize_tree_w8a8(bf16, root)
