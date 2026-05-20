#!/usr/bin/env python3
"""Benchmark model transfer candidates with comparable resource metrics.

Examples:

  PYTHONPATH=src uv run python scripts/benchmark_model_transfer.py \
    --mode hash --file /models/flux/shard.safetensors

  PYTHONPATH=src uv run python scripts/benchmark_model_transfer.py \
    --mode hf-snapshot --hf-repo black-forest-labs/FLUX.2-klein-4B \
    --dest-dir /tmp/hf-bench

  PYTHONPATH=src uv run python scripts/benchmark_model_transfer.py \
    --mode presigned --file /models/flux/shard.safetensors \
    --base-url "$TENSORHUB_PUBLIC_URL" \
    --endpoint-path /api/v1/repos/root/flux/revisions/rev/uploads \
    --token "$TENSORHUB_TOKEN"

  PYTHONPATH=src uv run python scripts/benchmark_model_transfer.py \
    --mode boto3-upload --file /models/flux/shard.safetensors \
    --s3-endpoint-url "$R2_ENDPOINT_URL" --s3-bucket "$R2_BUCKET" \
    --s3-key bench/flux/shard.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import psutil

from gen_worker.presigned_upload import blake3_hash_file, presigned_upload_file


@dataclass
class TransferMetrics:
    mode: str
    success: bool
    wall_seconds: float
    cpu_user_seconds: float
    cpu_system_seconds: float
    peak_rss_bytes: int
    read_bytes: int
    write_bytes: int
    file_bytes: int
    blake3: str
    error_type: str = ""
    error_message: str = ""
    extra: dict[str, Any] | None = None


class _PeakSampler:
    def __init__(self, process: psutil.Process, interval_s: float = 0.05) -> None:
        self._process = process
        self._interval_s = float(interval_s)
        self._stop = threading.Event()
        self.peak_rss = 0
        self._thread = threading.Thread(target=self._run, name="transfer-bench-rss", daemon=True)

    def __enter__(self) -> "_PeakSampler":
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> bool:
        self._stop.set()
        self._thread.join(timeout=2.0)
        return False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rss = int(self._process.memory_info().rss)
            except Exception:
                rss = 0
            self.peak_rss = max(self.peak_rss, rss)
            self._stop.wait(self._interval_s)


def _io_bytes(process: psutil.Process) -> tuple[int, int]:
    try:
        counters = process.io_counters()
    except Exception:
        return (0, 0)
    return (int(getattr(counters, "read_bytes", 0)), int(getattr(counters, "write_bytes", 0)))


def _cpu_times(process: psutil.Process) -> tuple[float, float]:
    try:
        cpu = process.cpu_times()
    except Exception:
        return (0.0, 0.0)
    return (float(getattr(cpu, "user", 0.0)), float(getattr(cpu, "system", 0.0)))


def _write_fixture(path: Path, size_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    block = (b"tensorhub-transfer-benchmark\n" * 32768)[:1024 * 1024]
    remaining = int(size_bytes)
    with path.open("wb") as f:
        while remaining > 0:
            chunk = block[: min(len(block), remaining)]
            f.write(chunk)
            remaining -= len(chunk)


def _measure(mode: str, file_path: Path | None, fn: Callable[[], dict[str, Any]]) -> TransferMetrics:
    process = psutil.Process(os.getpid())
    read0, write0 = _io_bytes(process)
    user0, system0 = _cpu_times(process)
    started = time.monotonic()
    digest = ""
    file_bytes = 0
    if file_path is not None and file_path.exists():
        file_bytes = file_path.stat().st_size
        digest = blake3_hash_file(file_path)

    with _PeakSampler(process) as sampler:
        try:
            extra = fn()
            success = True
            error_type = ""
            error_message = ""
        except BaseException as exc:  # noqa: BLE001
            extra = {}
            success = False
            error_type = type(exc).__name__
            error_message = str(exc)

    wall = time.monotonic() - started
    read1, write1 = _io_bytes(process)
    user1, system1 = _cpu_times(process)
    return TransferMetrics(
        mode=mode,
        success=success,
        wall_seconds=wall,
        cpu_user_seconds=max(0.0, user1 - user0),
        cpu_system_seconds=max(0.0, system1 - system0),
        peak_rss_bytes=int(sampler.peak_rss),
        read_bytes=max(0, read1 - read0),
        write_bytes=max(0, write1 - write0),
        file_bytes=file_bytes,
        blake3=digest,
        error_type=error_type,
        error_message=error_message,
        extra=extra,
    )


def _mode_hash(file_path: Path) -> dict[str, Any]:
    return {"digest": blake3_hash_file(file_path)}


def _mode_local_copy(file_path: Path, dest_dir: Path) -> dict[str, Any]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file_path.name
    with file_path.open("rb") as src, dest.open("wb") as out:
        shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
    return {"dest": str(dest), "bytes": dest.stat().st_size}


def _mode_hf_snapshot(repo_id: str, dest_dir: Path, revision: str) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=repo_id,
        revision=revision or None,
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
    )
    total = 0
    files = 0
    for path in Path(local).rglob("*"):
        if path.is_file():
            files += 1
            total += path.stat().st_size
    return {"snapshot_dir": local, "files": files, "bytes": total}


def _mode_presigned(args: argparse.Namespace, file_path: Path) -> dict[str, Any]:
    result = presigned_upload_file(
        file_path=file_path,
        base_url=str(args.base_url or "").rstrip("/"),
        endpoint_path=str(args.endpoint_path or ""),
        headers={"Authorization": f"Bearer {args.token}"},
        create_payload={"path": args.remote_path or file_path.name},
        blake3_hex=blake3_hash_file(file_path),
        size_bytes=file_path.stat().st_size,
    )
    return {"dedup": result.dedup, "meta": result.meta}


def _mode_boto3_upload(args: argparse.Namespace, file_path: Path) -> dict[str, Any]:
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=args.s3_endpoint_url,
        aws_access_key_id=args.aws_access_key_id or None,
        aws_secret_access_key=args.aws_secret_access_key or None,
        config=Config(
            retries={"mode": "standard", "max_attempts": 5},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )
    config = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=8,
        use_threads=True,
    )
    client.upload_file(str(file_path), args.s3_bucket, args.s3_key or file_path.name, Config=config)
    return {"bucket": args.s3_bucket, "key": args.s3_key or file_path.name}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["hash", "local-copy", "hf-snapshot", "presigned", "boto3-upload"], required=True)
    parser.add_argument("--file", type=Path, help="Local file to upload/hash/copy")
    parser.add_argument("--make-file-size", type=int, default=0, help="Create --file with this many bytes before benchmarking")
    parser.add_argument("--dest-dir", type=Path, default=Path("/tmp/gen-worker-transfer-bench"))
    parser.add_argument("--json-out", type=Path, help="Optional path for the metrics JSON")
    parser.add_argument("--hf-repo", default="")
    parser.add_argument("--revision", default="")
    parser.add_argument("--base-url", default=os.getenv("TENSORHUB_PUBLIC_URL", ""))
    parser.add_argument("--endpoint-path", default="")
    parser.add_argument("--token", default=os.getenv("TENSORHUB_TOKEN", ""))
    parser.add_argument("--remote-path", default="")
    parser.add_argument("--s3-endpoint-url", default=os.getenv("R2_ENDPOINT_URL", ""))
    parser.add_argument("--s3-bucket", default=os.getenv("R2_BUCKET", ""))
    parser.add_argument("--s3-key", default="")
    parser.add_argument("--aws-access-key-id", default=os.getenv("AWS_ACCESS_KEY_ID", ""))
    parser.add_argument("--aws-secret-access-key", default=os.getenv("AWS_SECRET_ACCESS_KEY", ""))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    file_path = args.file
    if args.make_file_size:
        if file_path is None:
            raise SystemExit("--make-file-size requires --file")
        _write_fixture(file_path, int(args.make_file_size))

    if args.mode in {"hash", "local-copy", "presigned", "boto3-upload"} and file_path is None:
        raise SystemExit(f"--mode {args.mode} requires --file")
    if file_path is not None and args.mode != "hf-snapshot" and not file_path.exists():
        raise SystemExit(f"file does not exist: {file_path}")

    if args.mode == "hash":
        metrics = _measure(args.mode, file_path, lambda: _mode_hash(file_path))
    elif args.mode == "local-copy":
        metrics = _measure(args.mode, file_path, lambda: _mode_local_copy(file_path, args.dest_dir))
    elif args.mode == "hf-snapshot":
        if not args.hf_repo:
            raise SystemExit("--mode hf-snapshot requires --hf-repo")
        metrics = _measure(args.mode, None, lambda: _mode_hf_snapshot(args.hf_repo, args.dest_dir, args.revision))
    elif args.mode == "presigned":
        if not args.base_url or not args.endpoint_path or not args.token:
            raise SystemExit("--mode presigned requires --base-url, --endpoint-path, and --token")
        metrics = _measure(args.mode, file_path, lambda: _mode_presigned(args, file_path))
    else:
        if not args.s3_endpoint_url or not args.s3_bucket:
            raise SystemExit("--mode boto3-upload requires --s3-endpoint-url and --s3-bucket")
        metrics = _measure(args.mode, file_path, lambda: _mode_boto3_upload(args, file_path))

    payload = asdict(metrics)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0 if metrics.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
