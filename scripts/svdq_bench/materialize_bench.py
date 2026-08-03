#!/usr/bin/env python3
"""te#137 bench: materialize ON THE POD — bf16 reference tree (R2 blobs) +
nunchaku svdq-fp4_r128 qwen-image checkpoint (public HF, pinned revision).
Reads refs_map.json next to this file. Weights never touch the control box."""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import sys
from pathlib import Path

ART = Path("/root/art")
REFS = ART / "refs"
NUN = ART / "nun"
FP8 = ART / "fp8"


def s3():
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3", endpoint_url=os.environ["TENSORHUB_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["TENSORHUB_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["TENSORHUB_S3_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("TENSORHUB_S3_REGION", "auto"),
        config=Config(max_pool_connections=16, retries={"max_attempts": 5}))


def get_blob(cli, bucket, blake3, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest, dest.stat().st_size, "cached"
    key = f"blobs/blake3/{blake3[:2]}/{blake3[2:4]}/{blake3}"
    for attempt in range(3):
        try:
            cli.download_file(bucket, key, str(dest))
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[mat] attempt{attempt} FAILED key={key} dest={dest}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            if attempt == 2:
                raise
    return dest, dest.stat().st_size, "get"


def refs_from_r2() -> None:
    m = json.loads((Path(__file__).parent / "refs_map.json").read_text())
    cli = s3()
    tasks = [(r["blake3"], REFS / r["path"]) for r in m["refs"]]
    print(f"[mat] {len(tasks)} R2 objects from {m['bucket']}", flush=True)
    total = 0
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(get_blob, cli, m["bucket"], b, d): d for b, d in tasks}
        for fu in cf.as_completed(futs):
            dest, sz, how = fu.result()
            total += sz
            print(f"[mat] {how:6} {sz/1e6:9.1f}MB {dest}", flush=True)
    print(f"[mat] R2 total {total/1e9:.2f} GB", flush=True)


def refs_from_hf(with_transformer: bool) -> None:
    """The chaos-R2 qwen-image blobs were GC'd 2026-08-01; our mirror IS
    Qwen/Qwen-Image HEAD 75e0b4be (eval_sets.py, verified 2026-07-26), so the
    public tree at that pinned revision is byte-identical for every component
    we load (VAE/TEs/tokenizer/scheduler/model_index; transformer is injected)."""
    import os as _os
    from huggingface_hub import snapshot_download
    ignore = ["*.png", "*.md"]
    if not with_transformer:
        ignore.append("transformer/*.safetensors")
    p = snapshot_download(
        repo_id="Qwen/Qwen-Image",
        revision="75e0b4be04f60ec59a75f475837eced720f823b6",
        local_dir=str(REFS), ignore_patterns=ignore,
        token=_os.environ.get("HF_TOKEN") or None)
    print(f"[mat] HF refs tree -> {p}", flush=True)


def fp8_from_manifest(manifest: Path) -> None:
    """The PUBLISHED prod #fp8-w8a8 flavor, straight from R2 through the
    presigned URLs the hub's /resolve handed us. Only the denoiser component
    and model_index.json: the pipeline's other components come from the same
    bf16 reference tree every other arm uses, so the arms differ in exactly
    one thing."""
    import urllib.request

    m = json.loads(manifest.read_text())
    want = [f for f in m["files"]
            if f["path"].startswith("transformer/")
            or f["path"] == "model_index.json"]
    print(f"[mat] fp8 flavor {m.get('checkpoint_id')} — {len(want)} files, "
          f"{sum(f['size_bytes'] for f in want) / 1e9:.2f} GB", flush=True)

    def fetch(f):
        dest = FP8 / f["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size == f["size_bytes"]:
            return dest, f["size_bytes"], "cached"
        # Big objects are chunk-CAS (th#1310): one presigned URL per chunk,
        # in order. Small ones carry a single "url".
        urls = f.get("chunk_urls") or ([f["url"]] if f.get("url") else [])
        if not urls:
            raise KeyError(f"{f['path']}: manifest entry has no url/chunk_urls")
        tmp = dest.with_suffix(dest.suffix + ".part")
        for attempt in range(3):
            try:
                with open(tmp, "wb") as fh:
                    for u in urls:
                        with urllib.request.urlopen(u, timeout=300) as r:
                            while True:
                                chunk = r.read(1 << 24)
                                if not chunk:
                                    break
                                fh.write(chunk)
                got = tmp.stat().st_size
                if got != f["size_bytes"]:
                    raise OSError(f"{f['path']}: got {got} of "
                                  f"{f['size_bytes']} bytes")
                want = str(f.get("digest") or "")
                if want.startswith("sha256:"):
                    h = hashlib.sha256()
                    with open(tmp, "rb") as fh:
                        for blk in iter(lambda: fh.read(1 << 24), b""):
                            h.update(blk)
                    if h.hexdigest() != want.split(":", 1)[1]:
                        raise OSError(f"{f['path']}: sha256 mismatch — "
                                      f"chunk order or content is wrong")
                tmp.rename(dest)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[mat] fp8 attempt{attempt} {f['path']}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                if attempt == 2:
                    raise
        return dest, dest.stat().st_size, f"get({len(urls)} chunks)"

    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for fu in cf.as_completed([ex.submit(fetch, f) for f in want]):
            dest, sz, how = fu.result()
            print(f"[mat] {how:6} {sz / 1e6:9.1f}MB {dest}", flush=True)
    print("FP8_TREE=" + str(FP8), flush=True)


def main() -> int:
    # chaos-R2 qwen refs blobs GC'd 2026-08-01 -> HF mirror is primary now.
    # --bf16 also pulls the transformer shards: the same-card bf16 arm.
    refs_from_hf("--bf16" in sys.argv)
    # Markers as soon as each artifact is real: the reference tree and the
    # svdq checkpoint are what every arm needs, and an optional extra
    # artifact must never be able to invalidate them.
    print("REFS_TREE=" + str(REFS), flush=True)

    NUN.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    fn = hf_hub_download(
        repo_id="nunchaku-ai/nunchaku-qwen-image",
        filename="svdq-fp4_r128-qwen-image.safetensors",
        revision="4d9f4f667ea571ab172e0ee29ac2c27b82a41a6b",
        local_dir=str(NUN), token=os.environ.get("HF_TOKEN") or None)
    print(f"[mat] nunchaku -> {fn} {Path(fn).stat().st_size/1e9:.2f} GB", flush=True)
    print("NUN_FILE=" + str(fn), flush=True)

    man = Path("/root/fp8_manifest.json")
    if "--fp8" in sys.argv and man.exists():
        try:
            fp8_from_manifest(man)
        except Exception as exc:  # noqa: BLE001
            print(f"[mat] fp8 flavor FAILED ({type(exc).__name__}: {exc}) — "
                  f"the fp8 arms will be skipped, every other arm stands",
                  flush=True)
    print("MAT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
