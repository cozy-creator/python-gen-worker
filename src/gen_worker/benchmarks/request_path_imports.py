"""pgw#1331's measurement: what the request path costs BEFORE and AFTER.

pgw#1326's expectations table claims process start drops from 5-15 s to ~1 s and
that a worker sheds 1-2 GB of host RAM. Those are the numbers this program
exists to stop anyone from quoting without having run something. It measures
exactly two interpreters, each in a FRESH subprocess so no import is warm:

* **before** — a Flux endpoint as authored today: ``FluxPipeline`` off
  ``diffusers``, which is the object ``examples/flux2-klein-image`` takes as its
  ``setup()`` parameter and calls per request.
* **after** — the typed family surface: the generated ``Flux1Dev`` binding, the
  catalog's serving half, and the bare-math scheduler. Nothing else, because
  nothing else is on the path.

Both arms import ``torch`` first and report the DELTA over it. A serve process
holds torch either way — it is the runtime, not the model library — so charging
it to one arm would inflate the result by about a second and half a gigabyte,
and the honest question is what the model library costs ON TOP of the runtime.

**This is not a latency benchmark and does not claim to be.** Per-request
latency needs a card, real weights and armed artifacts; the graph-coverage tail
pgw#1331 estimates at 5-15% is measured by
:mod:`gen_worker.benchmarks.store_arm_parity`'s sibling on a pod, not here.
What this measures is the fixed cost every serve process pays at boot and holds
resident for its whole life, which is the part that is a property of the CODE
and can therefore be measured on any machine, in CI, with no card.

    python -m gen_worker.benchmarks.request_path_imports
    python -m gen_worker.benchmarks.request_path_imports --json out.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Sequence

#: The BEFORE arm: what an endpoint imports today to serve one Flux request.
BEFORE: tuple[str, ...] = ("diffusers:FluxPipeline",)

#: The AFTER arm: the typed family surface, and that is the whole list.
AFTER: tuple[str, ...] = (
    "gen_worker.model.catalog:Flux1Dev",
    "gen_worker.model.catalog.flux1_dev_serve:generate",
    "gen_worker.model.scheduler:FlowMatchEulerDiscrete",
)

#: Run inside a fresh interpreter. Imports torch, takes a baseline, imports the
#: arm's names, and reports the delta. ``ru_maxrss`` is a HIGH-WATER mark, so
#: the delta is the additional peak the arm forced — which is the number that
#: decides how much host RAM a pod must be bought with.
_PROBE = """
import json, resource, sys, time

def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

t0 = time.perf_counter()
import torch
torch_s = time.perf_counter() - t0
base_rss, base_modules = rss(), len(sys.modules)

t1 = time.perf_counter()
for target in json.loads(sys.argv[1]):
    module, _, attribute = target.partition(":")
    loaded = __import__(module, fromlist=["_"])
    if attribute:
        getattr(loaded, attribute)
arm_s = time.perf_counter() - t1

print(json.dumps({
    "torch_s": torch_s,
    "torch_rss": base_rss,
    "torch_modules": base_modules,
    "import_s": arm_s,
    "rss_bytes": rss() - base_rss,
    "modules": len(sys.modules) - base_modules,
    "diffusers_loaded": "diffusers" in sys.modules,
    "transformers_loaded": "transformers" in sys.modules,
}))
"""


@dataclass(frozen=True, slots=True)
class Arm:
    """One measured interpreter."""

    name: str
    targets: tuple[str, ...]
    import_s: float
    rss_bytes: int
    modules: int
    diffusers_loaded: bool
    transformers_loaded: bool
    torch_s: float
    torch_rss: int


def measure(name: str, targets: Sequence[str], *, repeat: int = 3) -> Arm:
    """Run one arm ``repeat`` times in fresh interpreters and keep the FASTEST.

    The fastest, not the mean: an import is deterministic work, so the spread is
    the shared box's scheduler and page cache, and the minimum is the closest
    estimate of the cost the code actually has. Reporting a mean here would
    publish this machine's contention as a property of the library.
    """

    best: Arm | None = None
    for _ in range(repeat):
        done = subprocess.run(
            [sys.executable, "-c", _PROBE, json.dumps(list(targets))],
            check=True,
            capture_output=True,
            text=True,
        )
        row: dict[str, Any] = json.loads(done.stdout.strip().splitlines()[-1])
        arm = Arm(
            name=name,
            targets=tuple(targets),
            import_s=float(row["import_s"]),
            rss_bytes=int(row["rss_bytes"]),
            modules=int(row["modules"]),
            diffusers_loaded=bool(row["diffusers_loaded"]),
            transformers_loaded=bool(row["transformers_loaded"]),
            torch_s=float(row["torch_s"]),
            torch_rss=int(row["torch_rss"]),
        )
        if best is None or arm.import_s < best.import_s:
            best = arm
    assert best is not None  # repeat >= 1 is enforced by the caller
    return best


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    before = measure("before", BEFORE, repeat=args.repeat)
    after = measure("after", AFTER, repeat=args.repeat)

    print(f"{'arm':<8} {'import':>9} {'+RSS':>10} {'+modules':>9}  loaded")
    for arm in (before, after):
        libraries = ", ".join(
            name
            for name, loaded in (
                ("diffusers", arm.diffusers_loaded),
                ("transformers", arm.transformers_loaded),
            )
            if loaded
        )
        print(
            f"{arm.name:<8} {arm.import_s:>8.2f}s {arm.rss_bytes / 1e6:>9.0f}MB "
            f"{arm.modules:>9}  {libraries or 'neither'}"
        )
    print(
        f"\nover torch alone ({before.torch_s:.2f}s, {before.torch_rss / 1e6:.0f}MB): "
        f"{before.import_s / max(after.import_s, 1e-9):.1f}x faster, "
        f"{(before.rss_bytes - after.rss_bytes) / 1e6:.0f}MB less resident, "
        f"{before.modules - after.modules} fewer modules"
    )
    if after.diffusers_loaded or after.transformers_loaded:
        print(
            "\nFAILED: the AFTER arm loaded a model library. That is the whole "
            "claim of pgw#1331, and this run refutes it.",
            file=sys.stderr,
        )
        return 1
    if args.json:
        with open(args.json, "w") as handle:
            json.dump([asdict(before), asdict(after)], handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - a CLI
    raise SystemExit(main())
