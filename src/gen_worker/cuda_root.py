"""Compose the CUDA root AOTInductor's host compile needs."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import site
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

CUDA_ROOT = Path("/usr/local/cuda")

USER_CUDA_ROOT = Path.home() / ".cache" / "cozy" / "cuda-root"

CRT_HOST_DEFINES = Path("include/crt/host_defines.h")
NV_TARGET = Path("include/nv/target")


def default_root() -> Path:
    if CUDA_ROOT.exists():
        return CUDA_ROOT
    if os.access(CUDA_ROOT.parent, os.W_OK):
        return CUDA_ROOT
    return USER_CUDA_ROOT


@dataclass
class Composition:
    """What this run did, in the vocabulary the build log should carry."""

    root: str = ""
    crt: str = ""
    nv: str = ""
    path: str = ""
    notes: List[str] = field(default_factory=list)

    def lines(self) -> List[str]:
        out = [f"cuda_root: {self.root or 'none'}"]
        if self.path:
            out.append(f"cuda_home: {self.path}")
        if self.crt:
            out.append(f"cuda_crt: {self.crt}")
        if self.nv:
            out.append(f"cuda_nv: {self.nv}")
        out.extend(self.notes)
        return out


def wheel_cuda_root() -> str:
    """The single ``nvidia/*`` wheel directory that looks like a CUDA tree."""
    spec = importlib.util.find_spec("nvidia")
    roots = list(getattr(spec, "submodule_search_locations", None) or []) if spec else []
    found: List[str] = []
    for parent in roots:
        try:
            names = sorted(os.listdir(parent))
        except OSError:
            continue
        for name in names:
            cand = Path(parent) / name
            if not (cand / "include" / "cuda_runtime.h").is_file():
                continue
            if (cand / "lib64").is_dir() or (cand / "lib").is_dir():
                found.append(str(cand))
    return found[0] if len(found) == 1 else ""


def _donor_path(relative: Path) -> str:
    for base in site.getsitepackages():
        try:
            matches = sorted(Path(base).glob(f"*/**/{relative.as_posix()}"))
        except OSError:
            continue
        for match in matches:
            if match.exists():
                return str(match)
    return ""


def _install_cuda_cccl(target: Path) -> str:
    uv = shutil.which("uv")
    cmd = (
        [uv, "pip", "install", "--target", str(target), "--no-cache", "--no-deps", "cuda-cccl"]
        if uv
        else [sys.executable, "-m", "pip", "install", "--target", str(target),
              "--no-cache-dir", "--no-deps", "cuda-cccl"]
    )
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    except (subprocess.SubprocessError, OSError):
        return ""
    for match in sorted(target.glob(f"**/{NV_TARGET.as_posix()}")):
        return str(match)
    return ""


def compose(root_dir: Optional[Path] = None) -> Composition:
    """Assemble ``root_dir`` out of parts the image already ships."""
    root_dir = Path(root_dir) if root_dir is not None else default_root()
    out = Composition(path=str(root_dir))
    if root_dir.exists():
        out.root = "preexisting"
        return out
    if root_dir != CUDA_ROOT:
        out.notes.append(
            f"cuda_root: {CUDA_ROOT} is neither present nor this process's to "
            f"create, so the root is composed at {root_dir} instead — export "
            f"CUDA_HOME from `Composition.path`, never from the constant")

    wheel = wheel_cuda_root()
    if not wheel:
        out.notes.append(
            "cuda_root: no single nvidia/* wheel carries include/cuda_runtime.h "
            "with a lib64/lib — a CPU image has nothing to compose, and an "
            "ambiguous one is not guessed at")
        return out
    wheel_path = Path(wheel)

    include_dir = root_dir / "include"
    include_dir.mkdir(parents=True, exist_ok=True)
    for entry in sorted((wheel_path / "include").iterdir()):
        link = include_dir / entry.name
        if not link.exists():
            link.symlink_to(entry)
    for libdir in ("lib64", "lib"):
        src = wheel_path / libdir
        if src.is_dir() and not (root_dir / libdir).exists():
            (root_dir / libdir).symlink_to(src)
    out.root = wheel

    if not (root_dir / CRT_HOST_DEFINES).exists():
        donor = _donor_path(CRT_HOST_DEFINES)
        if donor:
            (root_dir / "include" / "crt").symlink_to(Path(donor).parent)
        out.crt = donor or "MISSING"

    if not (root_dir / NV_TARGET).exists():
        with tempfile.TemporaryDirectory(prefix="cuda-cccl-") as tmp:
            donor = _install_cuda_cccl(Path(tmp))
            if donor:
                shutil.copytree(Path(donor).parent, root_dir / "include" / "nv")
            out.nv = donor or "MISSING"
    return out


def missing_parts(root_dir: Optional[Path] = None) -> List[str]:
    """Which of the three measured facts this image still fails, if any."""
    root_dir = Path(root_dir) if root_dir is not None else default_root()
    if not root_dir.is_dir():
        return [f"the root itself ({root_dir} does not exist)"]
    gaps = []
    if not (root_dir / "include" / "cuda_runtime.h").exists():
        gaps.append("include/cuda_runtime.h")
    if not (root_dir / CRT_HOST_DEFINES).exists():
        gaps.append(f"{CRT_HOST_DEFINES.as_posix()} (the cu13 wheel flattens crt/ away)")
    if not (root_dir / NV_TARGET).exists():
        gaps.append(f"{NV_TARGET.as_posix()} (CCCL; no tree in the image ships it)")
    if not any((root_dir / d).exists() for d in ("lib64", "lib")):
        gaps.append("lib64/ or lib/")
    return gaps


def torch_cuda_home() -> str:
    """What torch itself would resolve, asked the way ``cpp_extension`` asks."""
    try:
        from torch.utils import cpp_extension

        return str(cpp_extension._find_cuda_home() or "")
    except Exception:  # noqa: BLE001 — no torch, or a private API that moved
        return ""


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m gen_worker.cuda_root",
        description=("Compose /usr/local/cuda for AOTInductor's host compile. "
                     "Run it in your Dockerfile when an endpoint in the image "
                     "declares an AOT export."))
    parser.add_argument("--root", default="",
                        help="where to compose the root (default: "
                             "/usr/local/cuda when it exists or /usr/local is "
                             "writable, else a per-user cache root)")
    parser.add_argument("--check", action="store_true",
                        help="report what is missing and exit nonzero; compose nothing")
    args = parser.parse_args(argv)
    root = Path(args.root) if args.root else default_root()

    if args.check:
        gaps = missing_parts(root)
        for line in (gaps or ["cuda_root: complete"]):
            print(line, file=sys.stderr)
        return 1 if gaps else 0

    for line in compose(root).lines():
        print(line, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
