"""``gen-worker publish`` — push this endpoint's source to the hub; it builds."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import endpoint_lock as el
from .login import LoginError, require_credential

MAX_SOURCE_BYTES = 100 * 1024 * 1024

HTTP_TIMEOUT_S = 600.0

_EXCLUDE_DIRS = frozenset({
    ".venv", "venv", "__pycache__", ".git", ".mypy_cache", ".pytest_cache",
    ".tox", ".ruff_cache", "node_modules",
    ".compiled-graphs", "outputs",
    ".aws", ".azure", ".gnupg", ".kube", ".secrets", ".ssh",
})
_EXCLUDE_SUFFIXES = (".safetensors", ".ckpt", ".bin", ".pt", ".pth", ".gguf",
                     ".onnx", ".so", ".pyc")
_EXCLUDE_NAMES = frozenset({".env"})


class PublishError(RuntimeError):
    """Publish could not proceed."""


def _tracked_files(root: Path) -> Optional[List[Path]]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True, check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    names = [n for n in completed.stdout.decode("utf-8").split("\0") if n]
    return [root / name for name in names]


def _excluded(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part in _EXCLUDE_DIRS for part in relative.parts):
        return True
    if path.name in _EXCLUDE_NAMES:
        return True
    return path.suffix.lower() in _EXCLUDE_SUFFIXES


def source_files(root: Path) -> List[Path]:
    tracked = _tracked_files(root)
    candidates = tracked if tracked is not None else [
        p for p in root.rglob("*") if p.is_file()
    ]
    files = [p for p in candidates if p.is_file() and not _excluded(p, root)]
    lock = root / el.LOCK_FILENAME
    if lock.is_file() and lock not in files:
        files.append(lock)
    return sorted(set(files))


def archive(root: Path, files: List[Path]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in files:
            tar.add(str(path), arcname=str(path.relative_to(root)))
    return buffer.getvalue()


def wheel_preflight(root: Path) -> None:
    """``uv build --wheel`` into a temp dir."""
    with tempfile.TemporaryDirectory(prefix="gen-worker-preflight-") as out:
        completed = subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", out, str(root)],
            capture_output=True, check=False,
        )
    if completed.returncode == 0:
        return
    stderr = completed.stderr.decode("utf-8", "replace").strip()
    raise PublishError(
        f"wheel preflight FAILED — the hub's image build would die on this "
        f"after paying for the whole CUDA layer, so it refuses here instead "
        f"(about five seconds in).\n\n{stderr}\n\n"
        f"  A source or editable install never runs the build backend's "
        f"validation, which is why a working dev env proves nothing about "
        f"this. Skip with --no-preflight if you know better."
    )


def _endpoint_identity(root: Path, org: str, name: str) -> Tuple[str, str]:
    resolved_name = name
    if not resolved_name:
        try:
            import tomllib

            document = tomllib.loads(
                (root / "pyproject.toml").read_text(encoding="utf-8")
            )
            resolved_name = str((document.get("project") or {}).get("name") or "")
        except (OSError, ValueError):
            resolved_name = ""
    if not resolved_name:
        raise PublishError(
            "cannot tell which endpoint this is: pass --name, or give "
            "pyproject.toml a [project].name"
        )
    if not org:
        raise PublishError("no org: pass --org (or log in, which records one)")
    return org, resolved_name


def upload(
    *, base_url: str, token: str, org: str, name: str, blob: bytes,
    promote: Optional[bool], dev: bool,
) -> Dict[str, Any]:
    query: List[str] = []
    if promote is not None:
        query.append(f"promote={'true' if promote else 'false'}")
    if dev:
        query.append("dev=true")
    url = (
        f"{base_url.rstrip('/')}/api/v1/endpoints/{org}/{name}/releases"
        + ("?" + "&".join(query) if query else "")
    )
    request = urllib.request.Request(url, data=blob, method="POST")
    request.add_header("Content-Type", "application/gzip")
    request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:600]
        raise PublishError(f"hub answered {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise PublishError(f"{url} unreachable: {exc.reason}") from exc
    try:
        return dict(json.loads(body))
    except json.JSONDecodeError:
        return {"raw": body}


def add_subparser(sub: "argparse._SubParsersAction[Any]") -> None:
    parser = sub.add_parser(
        "publish",
        help="Push this endpoint's source (+ endpoint.lock) to the hub to build.",
        description=(
            "Package what git tracks, run the wheel preflight, and upload to "
            "the hub, which builds the image. Releases are born only from "
            "source."
        ),
    )
    parser.add_argument("endpoint_dir", nargs="?", default=".",
                        help="directory holding endpoint.toml (default: .)")
    parser.add_argument("--org", default="", help="owning org (default: the "
                                                  "one the credential names)")
    parser.add_argument("--name", default="",
                        help="endpoint name (default: pyproject [project].name)")
    parser.add_argument("--hub-url", default="", help="hub base URL")
    parser.add_argument("--promote", dest="promote", action="store_true",
                        default=None,
                        help="promote the built release (hub default: yes)")
    parser.add_argument("--no-promote", dest="promote", action="store_false",
                        help="build without promoting")
    parser.add_argument("--dev", action="store_true",
                        help="a content-addressed dev release from the working "
                             "tree (packages untracked files too)")
    parser.add_argument("--no-preflight", action="store_true",
                        help="skip `uv build --wheel` (see the module docstring "
                             "for the 6h28m reason not to)")
    parser.add_argument("--dry-run", action="store_true",
                        help="package and preflight, upload nothing")
    parser.set_defaults(_handler=run_publish)


def run_publish(args: argparse.Namespace) -> int:
    root = Path(args.endpoint_dir).resolve()
    if not (root / "endpoint.toml").is_file():
        sys.stderr.write(
            f"gen-worker publish: {root} has no endpoint.toml — it is not an "
            f"endpoint directory\n"
        )
        return 2

    lock = root / el.LOCK_FILENAME
    if not lock.is_file():
        sys.stderr.write(
            f"gen-worker publish: WARNING — no {el.LOCK_FILENAME} beside the "
            f"endpoint, so this build will TRACE inside the image (~2 min for "
            f"an sd15-sized endpoint).\n"
            f"  `gen-worker lock` once, commit the result, and every later "
            f"build skips that step.\n"
        )

    try:
        if not args.no_preflight:
            sys.stderr.write("gen-worker publish: wheel preflight...\n")
            wheel_preflight(root)
            sys.stderr.write("gen-worker publish: wheel preflight ok\n")
        files = source_files(root)
        blob = archive(root, files)
    except PublishError as exc:
        sys.stderr.write(f"gen-worker publish: {exc}\n")
        return 1
    if len(blob) > MAX_SOURCE_BYTES:
        sys.stderr.write(
            f"gen-worker publish: the source archive is {len(blob) / 1e6:.1f} MB, "
            f"over the hub's {MAX_SOURCE_BYTES / 1e6:.0f} MB cap. Move large "
            f"artifacts into repos/datasets/media and reference them from "
            f"endpoint.toml.\n"
        )
        return 1

    sys.stderr.write(
        f"gen-worker publish: {len(files)} file(s), {len(blob) / 1024:.0f} KiB "
        + (f"(including {el.LOCK_FILENAME}, so the build skips the trace)\n"
           if lock.is_file() else "(no committed lock; the build traces)\n")
    )
    if args.dry_run:
        return 0

    try:
        credential = require_credential(args.hub_url)
        base_url = args.hub_url or credential.hub_url
        if not base_url:
            raise PublishError("no hub URL: pass --hub-url or set TENSORHUB_URL")
        org, name = _endpoint_identity(root, args.org or credential.org, args.name)
        answer = upload(
            base_url=base_url, token=credential.token, org=org, name=name,
            blob=blob, promote=args.promote, dev=args.dev,
        )
    except (PublishError, LoginError) as exc:
        sys.stderr.write(f"gen-worker publish: {exc}\n")
        return 1
    sys.stdout.write(json.dumps(answer) + "\n")
    build_id = answer.get("BuildID") or answer.get("build_id") or ""
    if build_id:
        sys.stderr.write(f"gen-worker publish: build {build_id} started\n")
    return 0


__all__ = [
    "MAX_SOURCE_BYTES",
    "PublishError",
    "add_subparser",
    "archive",
    "run_publish",
    "source_files",
    "wheel_preflight",
]
