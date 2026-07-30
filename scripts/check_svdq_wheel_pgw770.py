#!/usr/bin/env python3
"""pgw#770 publish gate: the svdq layout fix must be in the ARTIFACT.

The conversion endpoint installs gen-worker as a PINNED WHEEL
(``uv export --locked --no-dev --no-sources``) with no source overlay, so a
symbol that only resolves from ``src/`` ships broken: 0.77.0's venv raised
``ImportError: cannot import name 'pack_lowrank' from
gen_worker.models.svdq_layout`` and no suite noticed, because the suite imports
the source tree.

Asserted here, against site-packages:

  A. ``gen_worker`` does NOT resolve from a checkout's ``src/``;
  B. the seven-tensor packer/unpacker pairs are importable by name;
  C. ``torch`` imports — without it
     ``tests/test_svdq_official_layout_pgw770.py`` (the only oracle that checks
     our inverses against deepcompressor's forward packers rather than against
     our own encode side) ``importorskip``s itself and the gate goes vacuously
     green.

Run it with the interpreter that has the wheel installed, from a directory
outside the checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

_FAILURES: list[str] = []


def _check(name: str, ok: bool, detail: str) -> None:
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        _FAILURES.append(f"{name}: {detail}")


def main() -> int:
    import gen_worker

    loc = Path(gen_worker.__file__).resolve()
    repo_src = (Path(__file__).resolve().parent.parent / "src").resolve()
    _check(
        "installed",
        repo_src not in loc.parents,
        f"gen_worker resolves from {loc} (must NOT be the checkout's src/)",
    )

    try:
        from gen_worker.models import svdq_layout
    except Exception as exc:  # noqa: BLE001 - the whole point of the gate
        _check("svdq_layout", False, f"import failed: {exc!r}")
        return _verdict()

    # pack_lowrank / pack_vector are the two the conversion probe died on; their
    # inverses are what decode_linear calls on every serve.
    for name in (
        "pack_lowrank",
        "unpack_lowrank",
        "pack_vector",
        "unpack_vector",
        "pack_wscales",
        "unpack_wscales",
        "pack_qweight",
        "unpack_qweight",
        "decode_linear",
    ):
        _check(
            "svdq_layout",
            hasattr(svdq_layout, name),
            f"{name} {'present' if hasattr(svdq_layout, name) else 'MISSING'}",
        )

    # svdq_native.fold_to_dense is the other half decode_linear's callers use;
    # the pgw#770 oracle imports it by name too.
    try:
        from gen_worker.models.svdq_native import fold_to_dense  # noqa: F401

        _check("svdq_native", True, "fold_to_dense present")
    except Exception as exc:  # noqa: BLE001
        _check("svdq_native", False, f"fold_to_dense import failed: {exc!r}")

    try:
        import torch

        _check("torch", True, f"importable ({torch.__version__}) — the oracle cannot skip")
    except Exception as exc:  # noqa: BLE001
        _check("torch", False, f"not importable ({exc!r}); the pgw#770 oracle would skip")

    return _verdict()


def _verdict() -> int:
    if _FAILURES:
        print("\nFAILED:", file=sys.stderr)
        for f in _FAILURES:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nsvdq wheel contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
