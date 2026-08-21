"""pgw#1548 pod leg: get a BF16 SDXL tree ONTO THE POD, from HF, never via the box.

Weights-locality (Paul, hard): multi-GB artifacts never transit the control box.
This script runs ON the rented pod. It pulls SDXL base from Hugging Face
directly, casts the four pipeline weight containers to BF16, and verifies the
result with the SAME reader the serving path uses — so a tree that would have
served loud-eager on dtype is caught here, at $0.03/min, instead of inside a
benchmark whose every arm would then measure eager.

## Why a cast at all

The endpoint declares `sdxl.diffusers-bf16@1` and the serve path is dtype
PASSTHROUGH (pgw#1567). A tree whose containers are F16 gives `armed 18,
entered 0` — silently — and both arms of any A/B then measure eager and report
a beautiful 0% delta. That is the exact failure the #1548 section warns about
for `pgw1460-serve/sdxl-tree`.

## Why FP32 is the preferred source, not the fp16 variant

`cast_bf16.py` on the box cast **F16 -> BF16** and carries a caveat in its own
docstring: fp16 -> bf16 drops 3 mantissa bits (10 -> 7) and is not recoverable,
so only STRUCTURAL facts (`compiled_graph_calls > 0`, placement, peak VRAM) are
quotable on it. Casting from the FP32 master weights instead has no such
caveat: bf16 shares fp32's exponent range and simply truncates mantissa, which
is what a bf16 lane is supposed to hold. On a pod the fp32 download is a few
extra GB of HF bandwidth and no box cost at all, so there is no reason to
inherit the caveat. `--allow-fp16-source` is available for a bandwidth-bound
pod and prints the caveat loudly if used.

    python pgw1548_pod_sdxl_tree.py --dest /workspace/sdxl-bf16
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

#: Exactly the components `model_index.json` names as pipeline members. A tree
#: that ships weights nothing loads is confusion, not safety.
WEIGHT_DIRS = ("unet", "vae", "text_encoder", "text_encoder_2")
COPY_DIRS = ("scheduler", "tokenizer", "tokenizer_2")
REPO = "stabilityai/stable-diffusion-xl-base-1.0"


def cast_file(src: Path, dst: Path) -> tuple[int, int]:
    """Cast one safetensors container to bf16. Returns (cast, carried)."""

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors: dict = {}
    cast = carried = 0
    with safe_open(str(src), framework="pt") as handle:
        metadata = handle.metadata() or {}
        for key in handle.keys():
            value = handle.get_tensor(key)
            if value.dtype in (torch.float32, torch.float16):
                tensors[key] = value.to(torch.bfloat16)
                cast += 1
            else:
                tensors[key] = value
                carried += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    staging = dst.with_suffix(dst.suffix + ".partial")
    save_file(tensors, str(staging), metadata=metadata)
    staging.rename(dst)
    return cast, carried


def verify(tree: Path) -> str:
    """Ask the PRODUCTION reader what dtype this tree is.

    Not a hand-rolled header peek: `serving/checkpoint_dtype.checkpoint_dtype()`
    is the function the serve path itself consults, so agreeing with it is the
    only check that means anything. A tree that reads float16 here would serve
    loud-eager and make every benchmark arm measure the same nothing.
    """

    from gen_worker.serving.checkpoint_dtype import checkpoint_dtype

    return str(checkpoint_dtype(tree))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--cache", default="")
    parser.add_argument("--allow-fp16-source", action="store_true",
                        help="download the fp16 variant instead of the fp32 "
                             "master weights. Cheaper to fetch, but fp16->bf16 "
                             "drops 3 mantissa bits irrecoverably and only "
                             "STRUCTURAL facts stay quotable on the result")
    args = parser.parse_args(argv)

    dest = Path(args.dest).expanduser().resolve()
    if dest.exists():
        print(f"{dest} exists; refusing to overwrite a tree")
        return 1

    from huggingface_hub import snapshot_download

    patterns = ["*.json", "*.txt", "*.model"]
    if args.allow_fp16_source:
        print("WARNING: fp16 source — fp16->bf16 loses 3 mantissa bits. Only "
              "structural facts (compiled_graph_calls, placement, peak VRAM) "
              "are quotable on this tree; image quality and any fidelity A/B "
              "are NOT.")
        patterns += ["*.fp16.safetensors"]
    else:
        # The fp32 masters, and ONLY the diffusers component tree.
        # `*.safetensors` alone is a trap: the SDXL base repo also ships the
        # SINGLE-FILE checkpoints `sd_xl_base_1.0.safetensors` and
        # `sd_xl_base_1.0_0.9vae.safetensors` (~7 GB each), which the diffusers
        # pipeline never loads. Measured on a rented pod: that pattern turns a
        # ~13 GB pull into ~27 GB of which half is discarded. Scoping to the
        # component directories keeps only what `model_index.json` names.
        patterns += [f"{name}/*.safetensors" for name in WEIGHT_DIRS]

    started = time.monotonic()
    source = Path(snapshot_download(
        args.repo, allow_patterns=patterns,
        cache_dir=args.cache or None,
        ignore_patterns=None if args.allow_fp16_source else [
            "*.fp16.safetensors", "sd_xl_base_1.0*.safetensors"],
    ))
    print(f"[fetch] {args.repo} -> {source} ({time.monotonic() - started:.1f}s)")

    staging = dest.with_name(dest.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copy2(source / "model_index.json", staging / "model_index.json")
    for name in COPY_DIRS:
        if (source / name).is_dir():
            shutil.copytree(source / name, staging / name)

    total_cast = total_carried = 0
    for name in WEIGHT_DIRS:
        directory = source / name
        if not directory.is_dir():
            print(f"[cast] {name}: ABSENT in the snapshot — refusing a partial tree")
            return 1
        for item in sorted(directory.iterdir()):
            if item.suffix == ".safetensors":
                cast, carried = cast_file(item, staging / name / item.name)
                total_cast += cast
                total_carried += carried
            elif item.is_file():
                (staging / name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, staging / name / item.name)
        print(f"[cast] {name}: done")

    staging.rename(dest)
    print(f"[cast] {total_cast} tensor(s) -> bfloat16, {total_carried} carried as-is")

    dtype = verify(dest)
    print(f"[verify] production reader says: {dtype}")
    if "bfloat16" not in dtype:
        print("REFUSING: the tree does not read as bfloat16 through the serve "
              "path's own reader. Serving it would be loud-eager on dtype and "
              "every benchmark arm would measure the same nothing.")
        return 1
    print(json.dumps({"tree": str(dest), "dtype": dtype,
                      "cast": total_cast, "carried": total_carried}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
