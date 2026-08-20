"""Compose the CUDA root AOTInductor's host compile needs.

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

Tensorhub's SYNTHESIZED Dockerfile composes this root. A family that ships its
OWN Dockerfile gets none of it, so without this module it can install ``g++``,
pass the ``cxx_toolchain`` precondition, boot a pod, load, export — the
expensive part — and only then die at ``CUDA_HOME``: a PAID failure where the
missing-compiler sibling is a free one. The author invokes it in one line:

    RUN python -m gen_worker.cuda_root

The author owns invoking it — the platform never injects layers into
author-owned content — while the SDK owns the recipe, so there is ONE authority
rather than twenty lines of shell transcribed into every Dockerfile.
``aot_preconditions.CHECK_CUDA_ROOT`` then VERIFIES the result at build time,
so an image that declares an AOT export and skipped this refuses for $0.00
instead of on a rented card.

Nothing here fails: a CPU image has no CUDA root to compose, and an image that
cannot AOT-compile is still a working eager-serving image. Refusing is the
precondition's job, deliberately separate — this step reports, the gate decides.

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

#: Where a CUDA root gets composed when ``/usr/local`` is not this process's to
#: write. Spelled the same way ``cli/workspace.DEFAULT_GRAPH_CAS`` spells the
#: box's graph CAS — one shape for "the box's cozy cache", and no environment
#: read: §1.18 is right that "wherever XDG happens to point" is not a config
#: value anyone can name, and the operator's own knob for this is CUDA_HOME,
#: which wins outright and is never second-guessed.
USER_CUDA_ROOT = Path.home() / ".cache" / "cozy" / "cuda-root"

#: The header that proves the flattened-include problem is solved.
CRT_HOST_DEFINES = Path("include/crt/host_defines.h")
#: The CCCL header several cu13 headers include and no tree in the image ships.
NV_TARGET = Path("include/nv/target")


def default_root() -> Path:
    """Where THIS process can actually compose a CUDA root (pgw#1533).

    ``/usr/local/cuda`` is the answer on a pod, and it stayed the only answer
    for too long. On a developer box ``/usr/local`` is root-owned: the compose
    ran as the user, ``mkdir`` raised ``PermissionError`` minutes into a mint,
    and every specialization failed for a reason that reads like a missing
    toolkit while the toolkit was sitting in the venv's own ``nvidia-*``
    wheels. Measured on this box: fourteen specializations, all
    ``FileNotFoundError: '/usr/local/cuda/include'``.

    An EXISTING ``/usr/local/cuda`` always wins — an image that composed one in
    its Dockerfile, or ships a devel base, is not to be second-guessed. Then a
    writable ``/usr/local``, which is the image case before the compose has
    run. Only when neither holds does this fall to a per-user cache, and it says
    so in the composition notes rather than silently relocating the toolkit.
    """
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
    #: WHERE the root was composed — the value a caller must export as
    #: ``CUDA_HOME``. Distinct from ``root``, which names the donor wheel: a
    #: caller that reads the module constant instead of this field points torch
    #: at a directory nobody wrote.
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


def compose(root_dir: Optional[Path] = None) -> Composition:
    """Assemble ``root_dir`` out of parts the image already ships.

    Idempotent by the same test the synthesized step uses: a pre-existing root
    is left alone. An image that already has a real CUDA install (a devel base,
    a distro package) is not one this recipe should second-guess.

    ``root_dir`` defaults to :func:`default_root`, which is what makes this
    work off a pod at all.
    """
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
    """Which of the three measured facts this image still fails, if any.

    The one predicate ``aot_preconditions`` reads, so "is the CUDA root usable"
    has a single implementation and the gate cannot prove something the
    composer never did — which is also why its default must be the composer's
    default and not a second spelling of it.
    """
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
    # Deliberately 0 on every branch: a CPU image composing nothing is not a
    # broken build, and the AOT gate is `aot_preconditions`, not this step.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
