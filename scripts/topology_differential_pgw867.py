#!/usr/bin/env python3
"""Differential fuzzer for the WORKER_EXECUTION_TOPOLOGY decoders.

OFF THE MERGE PATH by construction: this is a script, not a test. The merge gate
runs the checked-in vector fixture (`tests/testdata/topology_wire_vectors.json`,
byte-identical in tensorhub) as ordinary fast cases; this driver is what FINDS
new vectors to check in, and it needs a tensorhub checkout and a Go toolchain,
so it must never be a gate.

Why differential at all. `WORKER_EXECUTION_TOPOLOGY` is one grammar with two
decoders — tensorhub `internal/orchestrator/topology` writes it,
`gen_worker.topology` reads it. Each side's own tests only ever prove that side
self-consistent, which is how a rename hole ships: Go's zero value for the
degree field is illegal so Go refuses for the wrong reason, while Python's
default of 1 is legal so Python silently serves degree 1. The property that
catches that class is AGREEMENT: same bytes in, both accept with an identical
derived (gpu_count, gpus_per_execution_group, execution_groups, parallel), or
both refuse.

Usage:

    python scripts/topology_differential_pgw867.py --tensorhub ~/cozy/tensorhub
    python scripts/topology_differential_pgw867.py -n 20000 --seed 7

Disagreements print as a table and are written to --report as JSON, ready to be
turned into fixture vectors once ruled on.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import pathlib
import random
import subprocess
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from gen_worker.topology import (  # noqa: E402
    KEY_EXECUTION_GROUPS,
    KEY_GPU_COUNT,
    KEY_GPUS_PER_GROUP,
    KEY_PARALLEL,
    ExecutionTopology,
    TopologyError,
)

logging.getLogger("gen_worker.topology").setLevel(logging.ERROR)

# The pre-rename spellings. Neither side knows them any more, so they stay in
# the generated key space as the differential's retired-spelling case: both
# decoders must refuse them, and must refuse them the SAME way.
RETIRED_KEY_GPUS_PER_GROUP = "group_degree"
RETIRED_KEY_EXECUTION_GROUPS = "groups"

KEYS = [
    KEY_GPU_COUNT,
    KEY_GPUS_PER_GROUP,
    KEY_EXECUTION_GROUPS,
    KEY_PARALLEL,
    RETIRED_KEY_GPUS_PER_GROUP,
    RETIRED_KEY_EXECUTION_GROUPS,
]


def python_verdict(wire: str) -> dict[str, Any]:
    """The Python decoder reduced to the tuple both languages must agree on.

    Mirrors tensorhub's `topology.VerdictFor` field for field. An exception that
    is not a `TopologyError` is reported as ``UNTYPED:`` rather than swallowed —
    an untyped escape IS the refusal-property violation, not a harness problem.
    """
    if wire.strip() == "":
        return {"wire": wire, "accept": True, "code": "absent"}
    try:
        t = ExecutionTopology.decode(wire)
    except TopologyError as exc:
        return {"wire": wire, "accept": False, "code": exc.code}
    except Exception as exc:  # noqa: BLE001 - see docstring
        return {"wire": wire, "accept": False, "code": f"UNTYPED:{type(exc).__name__}: {exc}"}
    out: dict[str, Any] = {"wire": wire, "accept": True}
    if t.gpu_count:
        out["gpu_count"] = t.gpu_count
    if t.gpus_per_execution_group:
        out["gpus_per_execution_group"] = t.gpus_per_execution_group
    if t.execution_groups:
        out["execution_groups"] = t.execution_groups
    if t.parallel:
        out["parallel"] = t.parallel
    return out


def go_verdicts(tensorhub: pathlib.Path, wires: list[str]) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory() as tmp:
        src = pathlib.Path(tmp) / "in.json"
        dst = pathlib.Path(tmp) / "out.json"
        src.write_text(json.dumps(wires))
        env = dict(os.environ, TOPOLOGY_VERDICT_IN=str(src), TOPOLOGY_VERDICT_OUT=str(dst))
        proc = subprocess.run(
            ["go", "test", "-count=1", "-run", "TestEmitTopologyVerdicts",
             "./internal/orchestrator/topology"],
            cwd=tensorhub, env=env, capture_output=True, text=True,
        )
        if proc.returncode != 0 or not dst.exists():
            raise SystemExit(
                f"go verdict emitter failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
            )
        return json.loads(dst.read_text())


# --- generation ------------------------------------------------------------
#
# Structured first, noise second. A purely random byte generator spends its
# whole budget on strings that are not JSON, which both sides refuse
# identically and which therefore prove nothing; the interesting inputs are
# WELL-FORMED JSON objects over this key set whose VALUES sit on the edges the
# two decoders might read differently — zero (Go's "not written" sentinel vs
# Python's explicit-present), floats that are integral, non-string `parallel`,
# case, whitespace, and the retired/current spelling pairs in every combination.

INTERESTING_NUMBERS: list[Any] = [
    0, 1, 2, 3, 4, 8, -1, -2,
    1.0, 2.0, 2.5, 0.0,
    True, False, None,
    "1", "2", "",
    2 ** 31, 2 ** 63, 2 ** 63 - 1, 10 ** 30,
    1e309, -1e309,
]

INTERESTING_PARALLEL: list[Any] = [
    "", "sequence", "internal", "cfg", "SEQUENCE", "Sequence", " sequence ",
    "seq", "none", None, 0, False, [], {}, 1, "sequence\n",
]

UNKNOWN_KEYS = ["gpus_per_group", "group_size", "GPU_COUNT", "parallel_mode", "", "0"]


def structured(rng: random.Random) -> str:
    obj: dict[str, Any] = {}
    for key in (KEY_GPU_COUNT, KEY_GPUS_PER_GROUP, KEY_EXECUTION_GROUPS,
                RETIRED_KEY_GPUS_PER_GROUP, RETIRED_KEY_EXECUTION_GROUPS):
        if rng.random() < 0.55:
            obj[key] = rng.choice(INTERESTING_NUMBERS)
    if rng.random() < 0.6:
        obj[KEY_PARALLEL] = rng.choice(INTERESTING_PARALLEL)
    if rng.random() < 0.12:
        obj[rng.choice(UNKNOWN_KEYS)] = rng.choice(INTERESTING_NUMBERS)
    body = json.dumps(obj)
    r = rng.random()
    if r < 0.06:
        return body + " " + body            # trailing second value
    if r < 0.10:
        return body + "garbage"
    if r < 0.14:
        return "  " + body + "  "
    if r < 0.17:
        return body[:-1]                    # truncated
    return body


def coherent(rng: random.Random) -> str:
    """Payloads a real hub could emit, so agreement on the HAPPY path is
    exercised too — a differential that only ever compares refusals proves
    nothing about what production actually sends."""
    d = rng.choice([1, 1, 2, 4])
    g = rng.choice([1, 2, 4])
    parallel = "" if d == 1 else rng.choice(["sequence", "internal", "cfg"])
    obj: dict[str, Any] = {KEY_GPU_COUNT: d * g, KEY_GPUS_PER_GROUP: d,
                           KEY_EXECUTION_GROUPS: g}
    if rng.random() < 0.7:
        obj[RETIRED_KEY_GPUS_PER_GROUP] = d
        obj[RETIRED_KEY_EXECUTION_GROUPS] = g
    if parallel:
        obj[KEY_PARALLEL] = parallel
    return json.dumps(obj)


NOISE_ALPHABET = '{}[]":,0123456789abcdefgpu_countseqilnrN aI\\\x00\xff'


def noise(rng: random.Random) -> str:
    return "".join(rng.choice(NOISE_ALPHABET) for _ in range(rng.randrange(0, 40)))


def generate(n: int, rng: random.Random) -> list[str]:
    out: list[str] = []
    # Exhaustive small cross-product first: every legal-shaped payload whose
    # current and RETIRED degree spellings are written in every combination of
    # present / absent / zero / disagreeing. That compiled graph is where the rename hole
    # lived, and it is where a one-sided un-retirement would show up.
    for gc, d, legacy_d in itertools.product([0, 1, 2, 4], [None, 0, 1, 2], [None, 0, 1, 2]):
        obj: dict[str, Any] = {KEY_GPU_COUNT: gc}
        if d is not None:
            obj[KEY_GPUS_PER_GROUP] = d
        if legacy_d is not None:
            obj[RETIRED_KEY_GPUS_PER_GROUP] = legacy_d
        obj[KEY_PARALLEL] = "sequence" if max(d or 0, legacy_d or 0) > 1 else ""
        out.append(json.dumps(obj))
    while len(out) < n:
        r = rng.random()
        out.append(coherent(rng) if r < 0.2 else structured(rng) if r < 0.9 else noise(rng))
    return out[:n]



# The recorded divergence CLASSES — currently EMPTY.
#
# Suppression is by CLASS, not by exact string, because each class has thousands
# of concrete witnesses. A non-zero exit from this driver means "a disagreement
# nobody has ruled on" — with nothing suppressed, that is now simply "any
# disagreement". Add a class here ONLY alongside a fixture `divergent` entry and
# a tracker issue that owns the fix.
def known_divergence_class(go: dict[str, Any], py: dict[str, Any]) -> str:
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensorhub", default=os.environ.get("TENSORHUB_REPO", "~/cozy/tensorhub"))
    ap.add_argument("-n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    tensorhub = pathlib.Path(args.tensorhub).expanduser()
    if not (tensorhub / "go.mod").exists():
        raise SystemExit(f"not a tensorhub checkout: {tensorhub}")

    wires = generate(args.n, random.Random(args.seed))
    gos = go_verdicts(tensorhub, wires)
    disagreements, suppressed = [], 0
    for wire, go in zip(wires, gos):
        py = python_verdict(wire)
        # Codes are per-language prose; the CONTRACT is accept/refuse plus the
        # derived tuple. A code mismatch on a shared refusal is a documentation
        # defect, not a serving one.
        keys = ("accept", "gpu_count", "gpus_per_execution_group", "execution_groups", "parallel")
        if any(go.get(k) != py.get(k) for k in keys):
            cls = known_divergence_class(go, py)
            if cls:
                suppressed += 1
                continue
            disagreements.append({"wire": wire, "go": go, "python": py})

    print(f"{len(wires)} inputs, {len(disagreements)} NEW disagreements "
          f"({suppressed} suppressed as already-recorded)")
    seen: set[tuple[bool, bool]] = set()
    for d in disagreements[:40]:
        sig = (bool(d["go"].get("accept")), bool(d["python"].get("accept")))
        mark = "*" if sig not in seen else " "
        seen.add(sig)
        print(f"{mark} {d['wire']!r}\n    go={d['go']}\n    py={d['python']}")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(disagreements, indent=2))
        print(f"wrote {args.report}")
    return 1 if disagreements else 0


if __name__ == "__main__":
    raise SystemExit(main())
