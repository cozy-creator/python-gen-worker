#!/usr/bin/env python3
"""Check or regenerate the temporary testdata projection of package data.

Tensorhub still consumes the historical tests/testdata paths.  Those files are
byte projections, not authorities, and disappear after th#1947 pins the wheel.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "gen_worker" / "contracts"
PROJECTION = ROOT / "tests" / "testdata"
NAMES = ("worker_value_contracts.json", "WORKER_VALUE_CONTRACTS_DIGEST")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="replace the removal-blocked projections with canonical bytes",
    )
    parser.add_argument(
        "--projection-dir",
        type=Path,
        default=PROJECTION,
        help="projection directory (a test seam; defaults to tests/testdata)",
    )
    args = parser.parse_args()

    mismatches: list[str] = []
    for name in NAMES:
        source = SOURCE / name
        projection = args.projection_dir / name
        payload = source.read_bytes()
        if args.write:
            projection.write_bytes(payload)
        elif not projection.is_file() or projection.read_bytes() != payload:
            mismatches.append(name)

    if mismatches:
        print(
            "worker-value-projection: FAIL — tests/testdata is not the "
            "byte-identical projection of gen_worker.contracts: "
            + ", ".join(mismatches)
        )
        return 1
    print("worker-value-projection: canonical package data and projection agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
