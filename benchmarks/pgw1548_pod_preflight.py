"""Pod preflight: does the MOUNTED tree satisfy the lane? (FIRST act)

Runs on the pod immediately after mount and before anything expensive — before
a derive, a compile, or a benchmarked request. A divergence is a typed abort
with the readings banked to the hub's durable rows, not a retry.

The layout stamp (quant/element axis) cannot answer this: two trees can both
stamp ``plain.bf16@1`` and differ on KEY CONVENTION (fused ``qkv_proj`` vs
split ``to_q/to_k/to_v``), so conformance is checked against the actual header
key histogram.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Exit codes: 90 is rigcheck's own fleet-line abort, 91 is this file's, so a caller can tell "wrong torch" from "wrong tree" without parsing prose.
EXIT_FLEET_LINE = 90
EXIT_NONCONFORMING = 91

DTYPE_SPELLINGS = {
    "bfloat16": "BF16",
    "float16": "F16",
    "float32": "F32",
    "float8_e4m3fn": "F8_E4M3",
}

SPLIT_MARKERS = (".to_q.weight", ".to_k.weight", ".to_v.weight")
FUSED_MARKERS = ("qkv_proj", "to_qkv")


class Nonconforming(RuntimeError):
    """The mounted tree does not satisfy the declared lane."""


def read_header(path: Path) -> dict[str, Any]:
    """The safetensors header of ONE file."""

    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise Nonconforming(f"{path}: too short to carry a safetensors header")
        length = struct.unpack("<Q", raw)[0]
        if length <= 0 or length > 200_000_000:
            raise Nonconforming(
                f"{path}: header length {length} is not credible — a 128-byte "
                f"TFSSTUB projection stub reads exactly this way, and it is not "
                f"a checkpoint"
            )
        body = handle.read(length)
    if len(body) != length:
        raise Nonconforming(f"{path}: header truncated ({len(body)} of {length})")
    return json.loads(body)


def survey(tree: Path) -> dict[str, dict[str, Any]]:
    """Per-component dtype histogram and attention-key census."""

    report: dict[str, dict[str, Any]] = {}
    for path in sorted(tree.glob("*/*.safetensors")):
        header = read_header(path)
        dtypes: Counter[str] = Counter()
        split = Counter[str]()
        fused = 0
        for key, entry in header.items():
            if key == "__metadata__" or not isinstance(entry, dict):
                continue
            if "dtype" in entry:
                dtypes[str(entry["dtype"]).upper()] += 1
            if any(marker in key for marker in FUSED_MARKERS):
                fused += 1
            for marker in SPLIT_MARKERS:
                if key.endswith(marker) and "attn" in key:
                    split[marker] += 1
        component = path.parent.name
        row = report.setdefault(
            component,
            {"dtypes": Counter(), "split": Counter(), "fused": 0, "files": []},
        )
        row["dtypes"].update(dtypes)
        row["split"].update(split)
        row["fused"] += fused
        row["files"].append(path.name)
    if not report:
        raise Nonconforming(f"{tree}: no safetensors containers found")
    return report


def assert_conforms(
    report: dict[str, dict[str, Any]], lane: str, dtype: str
) -> list[str]:
    """Both facts, or a typed abort naming exactly which one failed."""

    want = DTYPE_SPELLINGS.get(dtype)
    if want is None:
        raise Nonconforming(
            f"lane {lane!r} declares dtype {dtype!r}, which has no safetensors "
            f"spelling in this table — extend it deliberately, never guess"
        )

    lines = []
    offenders = []
    for component, row in sorted(report.items()):
        histogram = dict(row["dtypes"])
        lines.append(f"  {component:<16} {histogram}")
        stray = {name: count for name, count in histogram.items()
                 if name != want and not name.startswith(("I", "U", "BOOL"))}
        if stray:
            offenders.append(
                f"{component}: {stray} — lane {lane} declares {dtype} ({want})"
            )

    split_total = sum(sum(row["split"].values()) for row in report.values())
    fused_total = sum(row["fused"] for row in report.values())
    lines.append(f"  {'key convention':<16} split={split_total} fused={fused_total}")
    if fused_total:
        offenders.append(
            f"key convention: {fused_total} FUSED attention key(s) "
            f"(qkv_proj/to_qkv) — the lane wants the diffusers split set "
            f"(te#185: one key in common, 71 GB to find out)"
        )
    if not split_total:
        offenders.append(
            "key convention: ZERO split attention keys found — this tree "
            "carries neither convention where attention was looked for, so "
            "nothing here has actually been verified"
        )

    if offenders:
        raise Nonconforming(
            "the mounted tree does not satisfy the lane:\n  - "
            + "\n  - ".join(offenders)
        )
    return lines


def bank(kind: str, detail: str, phase: str) -> None:
    """Bank the reading in the hub's durable rows — best effort, never fatal."""

    try:
        from gen_worker.activity import emit_event

        emit_event(kind, detail, phase=phase)
    except Exception as exc:  # noqa: BLE001 — never mask the verdict
        print(f"[preflight] could not bank the reading ({exc}); the verdict stands",
              file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", required=True, help="the MOUNTED checkpoint tree")
    parser.add_argument("--lane", required=True, help="e.g. sdxl.diffusers-bf16@1")
    parser.add_argument("--dtype", default="bfloat16",
                        help="the dtype the lane declares (default: bfloat16)")
    parser.add_argument("--skip-fleet-line", action="store_true",
                        help="box-side dry run of the READER only; never on a pod")
    args = parser.parse_args(argv)

    if not args.skip_fleet_line:
        try:
            from gen_worker.rigcheck import assert_fleet_line

            assert_fleet_line("pgw1548-dynamic-dims", start=__file__)
        except Exception as exc:  # noqa: BLE001
            print(f"FLEET_LINE_ABORT: {exc}", file=sys.stderr)
            return EXIT_FLEET_LINE

    tree = Path(args.tree).resolve()
    print(f"[preflight] tree={tree}")
    print(f"[preflight] lane={args.lane} dtype={args.dtype}")
    try:
        report = survey(tree)
        lines = assert_conforms(report, args.lane, args.dtype)
    except Nonconforming as exc:
        detail = f"lane={args.lane} tree={tree}: {exc}"
        print(f"NONCONFORMING_TREE: {detail}", file=sys.stderr)
        bank("component_miss", detail[:1500], "preflight_nonconforming")
        return EXIT_NONCONFORMING

    print("[preflight] CONFORMS:")
    for line in lines:
        print(line)
    summary = "; ".join(
        f"{component}={dict(row['dtypes'])}" for component, row in sorted(report.items())
    )
    bank("applied_lane", f"pgw#1548 preflight: lane={args.lane} {summary}"[:1500],
         "preflight_conforms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
