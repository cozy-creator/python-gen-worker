#!/usr/bin/env python3
"""The GAUNTLET — every mint shape the fleet runs, on a model that compiles in
under a minute, in ONE command.

    task rig:gauntlet                 # every variant, cardless
    task rig:gauntlet -- --device cuda
    task rig:gauntlet -- --only micro,micro-lora

Each variant is a FULL production cycle — resolve, handoff, real child spawn,
`torch.export` + AOTInductor, seal, publish over the real wire, and a SECOND
process fetching the exact named compiled graph, arming and comparing every arm against eager. Not a smoke
test: the same machinery a pod runs, at ~15-60 s instead of ~95 minutes.

WHY A TABLE AND NOT A PASS/FAIL. Some variants exist to demonstrate a REFUSAL
(a bucket-bearing compiled graph offered to a plain-lane parent must be rejected). Those
are not failures, so each variant declares the outcome it EXPECTS and the
gauntlet reports agreement. A variant that flips in EITHER direction is news:
an expected-red going green means someone fixed something and did not say so.

The exit code is 0 when every variant matched its expectation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "src"))

RIG = REPO / "scripts" / "micro_mint_rig.py"

#: The order the table is printed in: cheapest and most fundamental first, so a
#: reader who stops after two rows has still seen the load-bearing ones.
ORDER = (
    "micro",
    "micro-lora",
    "micro-lora16",
    "micro-4d",
    "micro-pad32",
    "micro-pad32-branchy",
    "micro-conv",
    "micro-escape",
    "micro-rope",
    "micro-w8a8",
    "micro-w8a8-lora",
    "micro-lora-plain-parent",
)


def run_one(
    name: str, *, device: str, root: Path, python: str, timeout_s: float,
    gpu_only: bool = False, have_cuda: bool = False,
) -> Dict[str, Any]:
    if gpu_only and not have_cuda:
        # `scaled_mm` IS the w8a8 lane. Running it cardless would not be a
        # weaker test, it would be a DIFFERENT one, so it is NOT-RUN.
        return {"vehicle": name, "rc": 2, "wall_s": 0.0, "green": False,
                "refused": True, "tail": "",
                "failed_leg": "did-not-run",
                "failed_why": "gpu-only lane; this run is cardless"}
    out = root / f"{name}.json"
    # DELETE it first. A variant the rig REFUSED (the load gate, a
    # cardless `--device cuda`) exits before writing any json, and a stale file
    # from a previous run then gets read as THIS run's result — which is how a
    # CPU row was very nearly reported as a GPU one, complete with the previous
    # cycle's timing and peak. A variant that did not run must look like it did
    # not run.
    out.unlink(missing_ok=True)
    cmd = [
        python, str(RIG), "--vehicle", name, "--device", device,
        "--clean", "--root", str(root / name), "--json", str(out),
    ]
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
            cwd=str(REPO))
        rc, tail = proc.returncode, (proc.stderr or proc.stdout)[-1500:]
    except subprocess.TimeoutExpired:
        rc, tail = 124, f"timed out after {timeout_s:.0f}s"
    elapsed = time.monotonic() - started

    row: Dict[str, Any] = {
        "vehicle": name, "rc": rc, "wall_s": round(elapsed, 1),
        "green": rc == 0, "tail": tail,
        # rc 2 is the rig's REFUSED exit (a precondition, never a verdict on
        # the variant). It must not be scored as a red, because a red is a
        # claim about the mint path and a refusal is a claim about the box.
        "refused": rc == 2,
    }
    if not out.is_file():
        row["failed_leg"] = "did-not-run"
        row["failed_why"] = (
            "REFUSED before any leg (load gate / device precondition) — "
            "no result written" if row["refused"] else
            "the rig wrote no result")
    if out.is_file():
        try:
            data = json.loads(out.read_text())
        except ValueError:
            return row
        row["cycle_s"] = data.get("total_s")
        env = data.get("env") or {}
        row["device"] = env.get("device")
        row["sm"] = env.get("sm")
        row["torch"] = env.get("torch")
        row["covers"] = env.get("covers")
        for leg in data.get("legs") or ():
            facts = leg.get("facts") or {}
            if leg["name"] == "mint-child":
                row["entries"] = len(facts.get("packed_entries") or ())
                row["peak_gb"] = round(
                    float(facts.get("peak_vram_bytes") or 0) / 2 ** 30, 3)
                row["synthetic_sm"] = bool(facts.get("synthetic_runtime"))
            if leg["name"] == "adopt":
                parity = facts.get("parity_max_abs") or {}
                row["parity"] = (max(parity.values()) if parity else None)
                row["arm_reason"] = facts.get("arm_reason") or ""
            if not leg.get("ok") and "failed_leg" not in row:
                row["failed_leg"] = leg["name"]
                row["failed_why"] = _why(facts, leg)
    return row


def _why(facts: Dict[str, Any], leg: Dict[str, Any]) -> str:
    """The CLASSIFIED reason a leg failed, never a slice of raw stderr.

    A table compiled graph wide enough for a stack trace tail is a table compiled graph that says
    nothing; these refusals carry a class.
    """
    if facts.get("arm_reason"):
        return str(facts["arm_reason"])
    for blob in (facts.get("miss_log") or "", facts.get("detail") or "",
                 facts.get("stderr_tail") or ""):
        marker = "rejected by class:"
        if marker in blob:
            return blob.split(marker, 1)[1].strip().split("\n")[0][:60]
    detail = str(facts.get("detail") or leg.get("detail") or "").strip()
    return detail.split("\n")[0][:60] or "(unclassified)"


def _verdict(row: Dict[str, Any], expect: str) -> str:
    if row.get("refused"):
        return "NOT-RUN"
    actual = "green" if row["green"] else "red"
    return "OK" if actual == expect else ("REGRESSED" if expect == "green"
                                          else "NOW-GREEN")


def table(rows: List[Dict[str, Any]], vehicles: Dict[str, Any]) -> str:
    head = (f"{'variant':<26} {'exp':<5} {'got':<5} {'':<9} {'cycle':>7} "
            f"{'ent':>4} {'peak GB':>8} {'parity':>10}  note")
    lines = [head, "-" * len(head)]
    for row in rows:
        veh = vehicles[row["vehicle"]]
        got = ("n/a" if row.get("refused")
               else ("green" if row["green"] else "red"))
        verdict = _verdict(row, veh.expect)
        parity = row.get("parity")
        note = ""
        if not row["green"]:
            note = f"{row.get('failed_leg', '?')}: {row.get('failed_why', '')}"[:64]
        elif veh.expect == "red":
            note = "expected a refusal and did NOT get one"
        lines.append(
            f"{row['vehicle']:<26} {veh.expect:<5} {got:<5} {verdict:<9} "
            f"{(row.get('cycle_s') or row['wall_s']):>6.1f}s "
            f"{row.get('entries', 0):>4} {row.get('peak_gb', 0):>8.3f} "
            f"{(f'{parity:.2e}' if parity else '—'):>10}  {note}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="rig_gauntlet")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"),
                        default="auto")
    parser.add_argument("--only", default="",
                        help="comma-separated subset of the variants")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get(
        "PGW978_ROOT", str(Path.home() / ".cache" / "gen-worker" / "gauntlet"))))
    parser.add_argument("--python", default=os.environ.get(
        "GEN_WORKER_RIG_GPU_PYTHON", sys.executable),
        help="interpreter to run each cycle with (the GPU lane needs the "
             "cu126 venv — see scripts/rig_gpu_env.sh)")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    # carry the resolved environment, but do NOT assert the fleet line
    # here. This gauntlet's product is a plumbing verdict (did the machinery
    # agree with its expectation), never a number, and the box cannot run a
    # cu130 build against its card at all — `scripts/rig_gpu_env.sh` builds the
    # fleet's torch on cu126 deliberately. Any rig that produces a NUMBER calls
    # `assert_fleet_line()` instead and aborts; see research/RIG-ENV.md §3b.
    from gen_worker import rigcheck

    try:
        env = rigcheck.resolve_environment()
        print(rigcheck.format_report(env, rigcheck.resolve_fleet_line()),
              file=sys.stderr, flush=True)
    except rigcheck.FleetLineUnknown as exc:
        print(f"rig_gauntlet: fleet line unresolved ({exc})", file=sys.stderr)
    print("rig_gauntlet: PLUMBING VERDICT ONLY — these variants report machinery "
          "agreement, not measurements. No number from this rig is quotable.",
          file=sys.stderr, flush=True)

    from harness import rig_vehicles

    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(ORDER)
    unknown = [n for n in names if n not in rig_vehicles.VEHICLES]
    if unknown:
        print(f"unknown variant(s): {unknown!r} "
              f"(known: {sorted(rig_vehicles.VEHICLES)!r})", file=sys.stderr)
        return 2

    args.root.mkdir(parents=True, exist_ok=True)
    print(f"GAUNTLET — {len(names)} variant(s), device={args.device}, "
          f"python={Path(args.python).parent.parent.name}\n")

    have_cuda = args.device == "cuda" or (
        args.device == "auto" and "cu126" in str(args.python))
    rows: List[Dict[str, Any]] = []
    for name in names:
        veh = rig_vehicles.VEHICLES[name]
        print(f"  ... {name} ({veh.covers[:70]})", flush=True)
        rows.append(run_one(name, device=args.device, root=args.root,
                            python=args.python, timeout_s=args.timeout,
                            gpu_only=veh.gpu_only, have_cuda=have_cuda))

    print("\n" + table(rows, rig_vehicles.VEHICLES))

    env0 = next((r for r in rows if r.get("device")), {})
    print(f"\ncard={env0.get('device', '?')} {env0.get('sm', '')} "
          f"torch={env0.get('torch', '?')}")
    print(f"covers={env0.get('covers', '?')}")
    if any(r.get("synthetic_sm") for r in rows):
        print("⚠ synthetic `sm` supplied on at least one variant — those compiled graphs "
              "are PLUMBING artifacts and must not reach a shared namespace.")
    print("⚠ parity is the SDK's cosine ladder plus this rig's max|delta|; the "
          "GATE is cosine-based, so it is a directional-degradation "
          "instrument, not a max-abs bound.")

    for row in rows:
        veh = rig_vehicles.VEHICLES[row["vehicle"]]
        if veh.expect == "red" and not row["green"]:
            print(f"\n  {row['vehicle']}: red AS EXPECTED — {veh.expect_note}")

    if args.json:
        args.json.write_text(json.dumps(
            {"rows": rows,
             "expect": {n: rig_vehicles.VEHICLES[n].expect for n in names}},
            indent=2, sort_keys=True, default=str))

    mismatched = [r for r in rows
                  if _verdict(r, rig_vehicles.VEHICLES[r["vehicle"]].expect) != "OK"]
    if mismatched:
        print(f"\nGAUNTLET: {len(mismatched)} variant(s) did NOT match "
              f"expectation: {[r['vehicle'] for r in mismatched]!r}")
        return 1
    print(f"\nGAUNTLET: all {len(rows)} variant(s) matched expectation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
