from __future__ import annotations

import json
import struct
import sys
import urllib.request

BASE = "stabilityai/stable-diffusion-xl-base-1.0"
CANDIDATES = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "playgroundai/playground-v2-1024px-aesthetic",
    "playgroundai/playground-v2.5-1024px-aesthetic",
    "RunDiffusion/Juggernaut-XL-v9",
    "RunDiffusion/Juggernaut-X-v10",
    "SG161222/RealVisXL_V4.0",
    "SG161222/RealVisXL_V5.0",
    "cagliostrolab/animagine-xl-3.1",
    "cagliostrolab/animagine-xl-4.0",
    "Lykon/dreamshaper-xl-1-0",
    "Lykon/dreamshaper-xl-v2-turbo",
    "stablediffusionapi/nightvision-xl",
    "misri/zavychromaxl_v80",
    "misri/epicrealismXL_v10",
    "GraydientPlatformAPI/albedobase2-xl",
    "fluently/Fluently-XL-v4",
    "recoilme/ColorfulXL-Lightning",
    "zenless-lab/sdxl-blue-pencil-xl-v7",
    "John6666/copycat-photo-sdxl-v1-sdxl",
    "dataautogpt3/ProteusV0.4",
    "dataautogpt3/OpenDalleV1.1",
    "segmind/SSD-1B",
]


def fetch_header(repo: str, path: str) -> dict | None:
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    req = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            (hlen,) = struct.unpack("<Q", r.read(8))
    except Exception as exc:
        print(f"  {repo}: no {path} ({type(exc).__name__})", file=sys.stderr)
        return None
    if hlen > 100 << 20:
        return None
    req = urllib.request.Request(url, headers={"Range": f"bytes=8-{7 + hlen}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read(hlen))


def unet_header(repo: str) -> tuple[dict | None, str]:
    for path in (
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "unet/diffusion_pytorch_model.safetensors",
    ):
        h = fetch_header(repo, path)
        if h is not None:
            return h, path
    return None, ""


def shape_map(header: dict) -> dict[str, tuple]:
    return {
        k: tuple(v["shape"])
        for k, v in header.items()
        if k != "__metadata__"
    }


def main() -> None:
    base_header, base_path = unet_header(BASE)
    assert base_header is not None, "base UNet header unreachable"
    base_shapes = shape_map(base_header)
    print(f"base: {BASE} ({base_path}), {len(base_shapes)} tensors")
    compatible = []
    for repo in CANDIDATES:
        header, path = unet_header(repo)
        if header is None:
            print(f"REFUSED {repo}: no unet safetensors in diffusers layout")
            continue
        shapes = shape_map(header)
        dtypes = {v["dtype"] for k, v in header.items() if k != "__metadata__"}
        total = sum(
            v["data_offsets"][1] - v["data_offsets"][0]
            for k, v in header.items() if k != "__metadata__"
        )
        if shapes == base_shapes:
            compatible.append(repo)
            print(f"OK      {repo}: {path} dtypes={sorted(dtypes)} "
                  f"unet={total / (1 << 30):.2f} GiB")
        else:
            missing = len(base_shapes.keys() - shapes.keys())
            extra = len(shapes.keys() - base_shapes.keys())
            diff = sum(
                1 for k in shapes.keys() & base_shapes.keys()
                if shapes[k] != base_shapes[k]
            )
            print(f"REFUSED {repo}: missing={missing} extra={extra} shape-diff={diff}")
    print(f"\ncompatible: {len(compatible)}")
    for r in compatible:
        print(f"  {r}")


if __name__ == "__main__":
    main()
