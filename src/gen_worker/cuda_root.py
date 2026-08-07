"""Compose the CUDA root AOTInductor's host compile needs (pgw#1017, pgw#823).

``g++`` is necessary and not sufficient. AOTInductor emits a C++ wrapper and
links a real ``.so``, and torch's ``cpp_extension`` has to find a CUDA tree to
do it. On the pytorch runtime bases it finds none, for three separate measured
reasons — none of which a compiler install fixes:

1. **torch finds no CUDA at all.** ``_find_cuda_home`` reads ``CUDA_HOME`` /
   ``CUDA_PATH``, then ``which nvcc``, then ``/usr/local/cuda``. The runtime
   bases satisfy none: CUDA arrives as pip wheels, and ``/usr/local/cuda`` is
   on ``PATH`` but does not exist. AOTI dies with *"CUDA_HOME environment
   variable is not set"* before compiling a line.
2. **The cu13 wheel's ``include/`` is FLATTENED.** Its own
   ``cuda_runtime_api.h`` does ``#include "crt/host_defines.h"`` and no
   ``crt/`` directory ships. The top-level ``host_defines.h`` is itself a
   forwarder INTO ``crt/``, so it cannot stand in. Triton's bundled CUDA tree
   carries the real ``crt/`` headers.
3. **Several cu13 headers include ``<nv/target>``**, which is CCCL and ships in
   NO tree in the image. The ``cuda-cccl`` wheel has it.

Tensorhub's SYNTHESIZED Dockerfile has composed this root since pgw#823. A
family that ships its OWN Dockerfile got none of it, so it could install
``g++``, pass the ``cxx_toolchain`` precondition, boot a pod, load, export —
the expensive part — and only then die at ``CUDA_HOME``. That is a PAID
failure where the missing-compiler sibling was a free one.

The fix is this module, invoked by the author in one line:

    RUN python -m gen_worker.cuda_root

The author still owns invoking it — the platform never injects layers into
author-owned content (pgw#1017's settled contract). The SDK owns the recipe's
correctness, so there is ONE authority for it rather than twenty lines of
shell transcribed into every Dockerfile that needs it (the pgw#988 lesson).
``aot_preconditions.CHECK_CUDA_ROOT`` then VERIFIES the result at build time,
so an image that declares an AOT export and skipped this refuses for $0.00
instead of on a rented card.

Nothing here fails: a CPU image has no CUDA root to compose, and an image that
cannot AOT-compile is still a working eager-serving image. The refusal is the
precondition's job, and it is deliberately a separate one — this step reports,
the gate decides.

``/usr/local/cuda`` is COMPOSED (a real directory of symlinks) rather than
pointed at a wheel, so nothing is ever written inside a pip package.
"""

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

#: The header that proves the flattened-include problem is solved.
CRT_HOST_DEFINES = Path("include/crt/host_defines.h")
#: The CCCL header several cu13 headers include and no tree in the image ships.
NV_TARGET = Path("include/nv/target")


@dataclass
class Composition:
    """What this run did, in the vocabulary the build log should carry."""

    root: str = ""
    crt: str = ""
    nv: str = ""
    notes: List[str] = field(default_factory=list)

    def lines(self) -> List[str]:
        out = [f"cuda_root: {self.root or 'none'}"]
        if self.crt:
            out.append(f"cuda_crt: {self.crt}")
        if self.nv:
            out.append(f"cuda_nv: {self.nv}")
        out.extend(self.notes)
        return out


def wheel_cuda_root() -> str:
    """The single ``nvidia/*`` wheel directory that looks like a CUDA tree.

    A tree qualifies when it carries ``include/cuda_runtime.h`` AND a
    ``lib64``/``lib`` directory. Ambiguity is answered with "" rather than a
    guess: two candidates mean the image's CUDA story is not the one this
    recipe was measured against, and composing the wrong one is worse than
    composing nothing (the precondition will say so by name).
    """
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
    """A ``site-packages`` file matching ``*/<relative>``, or ""."""
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
    """Fetch ``cuda-cccl`` into a throwaway dir and return its ``nv/`` path.

    Into ``--target``, never into ``site-packages``: the image's installed
    environment is not this step's to modify, and only the ~56 KB ``nv/``
    subtree is kept. ``uv`` when the image has it (tensorhub's own images do),
    ``pip`` otherwise — the doc's minimum-viable Dockerfile installs with pip
    and must not be excluded from the AOT lane by a tool choice.
    """
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


def compose(root_dir: Path = CUDA_ROOT) -> Composition:
    """Assemble ``root_dir`` out of parts the image already ships.

    Idempotent by the same test the synthesized step uses: a pre-existing
    ``/usr/local/cuda`` is left alone. An image that already has a real CUDA
    install (a devel base, a distro package) is not one this recipe should
    second-guess.
    """
    out = Composition()
    if root_dir.exists():
        out.root = "preexisting"
        return out

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


def missing_parts(root_dir: Path = CUDA_ROOT) -> List[str]:
    """Which of the three measured facts this image still fails, if any.

    The one predicate ``aot_preconditions`` reads, so "is the CUDA root usable"
    has a single implementation and the gate cannot prove something the
    composer never did.
    """
    if not root_dir.is_dir():
        return ["the root itself (/usr/local/cuda does not exist)"]
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
    """What torch itself would resolve, asked the way ``cpp_extension`` asks.

    Same doctrine as ``compile_cache.cxx_compiler``: predict the failure by
    calling the thing that produces it, so the predicate cannot drift from what
    it predicts. ``_find_cuda_home`` already consults ``CUDA_HOME`` /
    ``CUDA_PATH`` / ``which nvcc`` / ``/usr/local/cuda``, in that order —
    reimplementing that ladder here would be a second spelling of the lookup
    whose divergence is the entire bug class this module exists for, so there
    is no fallback: no torch means no verdict, and the caller never asks.
    """
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
    parser.add_argument("--root", default=str(CUDA_ROOT),
                        help="where to compose the root (default /usr/local/cuda)")
    parser.add_argument("--check", action="store_true",
                        help="report what is missing and exit nonzero; compose nothing")
    args = parser.parse_args(argv)
    root = Path(args.root)

    if args.check:
        gaps = missing_parts(root)
        for line in (gaps or ["cuda_root: complete"]):
            print(line, file=sys.stderr)
        return 1 if gaps else 0

    for line in compose(root).lines():
        print(line, file=sys.stderr)
    # Deliberately 0 on every branch: a CPU image composing nothing is not a
    # broken build, and the AOT gate is `aot_preconditions`, not this step.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
