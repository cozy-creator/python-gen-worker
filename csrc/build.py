#!/usr/bin/env python3
"""Build libcozy_kernels.so against THIS environment's torch (pgw#860).

Run inside the pinned devel toolchain image via
scripts/native/build_native_kernels.sh — never against a host toolkit.
Fatbin targets exactly sm_100a + sm_120a (fp4 instructions are arch-'a'
gated). Verifies arch coverage with cuobjdump and smokes op registration by
loading the built library (no GPU needed for either).

    python csrc/build.py <out_dir>
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ARCHS = ("100a", "120a")
HERE = Path(__file__).resolve().parent


def main() -> int:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "dist-native")
    out_dir.mkdir(parents=True, exist_ok=True)

    # "100a" -> "10.0a" (TORCH_CUDA_ARCH_LIST spelling for arch-'a' targets).
    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(
        f"{int(a[:-1]) // 10}.{int(a[:-1]) % 10}a" for a in ARCHS)
    import torch
    from torch.utils import cpp_extension

    arch_macro = '-DCOZY_FATBIN_ARCHS=\\"sm_' + "+sm_".join(ARCHS) + '\\"'
    build_dir = Path(tempfile.mkdtemp(prefix="cozy_kernels_build_"))
    cpp_extension.load(
        name="cozy_kernels",
        sources=[str(HERE / "cozy_kernels" / "ops.cpp"),
                 str(HERE / "cozy_kernels" / "probe.cu")],
        extra_cflags=["-O2", arch_macro],
        extra_cuda_cflags=["-O2", arch_macro],
        build_directory=str(build_dir),
        is_python_module=False,
        verbose=True,
    )
    so = build_dir / "cozy_kernels.so"
    assert so.exists(), f"build produced no {so}"

    # Arch census — both 'a' targets must be in the fatbin.
    census = subprocess.run(
        ["cuobjdump", "--list-elf", str(so)], check=True,
        capture_output=True, text=True).stdout
    missing = [a for a in ARCHS if f"sm_{a}" not in census]
    assert not missing, f"fatbin missing archs {missing}:\n{census}"

    # Registration smoke: load + schema + CPU-callable build_info.
    # TORCH_VERSION is the base semver (no +cuXXX local tag).
    info = torch.ops.cozy_kernels.build_info()
    base_ver = str(torch.__version__).split("+", 1)[0]
    assert f"torch={base_ver}" in info, (
        f"build_info {info!r} does not carry torch {base_ver}")
    assert hasattr(torch.ops.cozy_kernels, "probe_add_one")

    dest = out_dir / "libcozy_kernels.so"
    shutil.copy2(so, dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    print(f"[build] {dest} sha256={digest}")
    print(f"[build] build_info: {info}")
    print(f"[build] arch census:\n{census}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
