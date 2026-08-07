"""Deterministic weight GENERATION — the micro family never downloads bytes.

The weights-locality rule says model weights never transit the developer's
box. A micro endpoint that shipped a checkpoint would either violate that or
need a hub round trip on every cycle, so it has neither: the checkpoint is a
pure function of ``(seed, config)`` and is materialized wherever it is needed —
in the image at ``docker build`` time, on the pod at boot, or in the local rig.

Two runs with the same seed produce BYTE-IDENTICAL files. That is not a nicety:
the cell key and the snapshot digest are claims about which checkpoint was
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
    denoiser = MicroDenoiser(config)
    decoder = MicroDecoder(config)
    out: Dict[str, torch.Tensor] = {}
    for prefix, module in (("transformer", denoiser), ("decoder", decoder)):
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
    payload = dict(cfg.as_dict(), _class_name="MicroPipeline",
                   _format="safetensors", seed=int(seed))
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
    size = tree_bytes(root)
    if size > MAX_TREE_BYTES:
        raise WeightsRefused(
            f"the generated tree is {size / 1e6:.1f} MB, over the "
            f"{MAX_TREE_BYTES / 1e6:.0f} MB ceiling this family exists to stay "
            f"under — shrink the config rather than raising the ceiling")
    return root


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
