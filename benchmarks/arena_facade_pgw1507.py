"""pgw#1507: the four card-side legs of the varena facade.

Run inside a coordinator-arbitrated GPU window on the box's RTX 4070.

    .venv/bin/python benchmarks/arena_facade_pgw1507.py --arms ABCD

* **A — identity.** sd1.5 through the facade vs eager resident, same seed, same
  config: the latents must be BITWISE identical. Moving bytes must not change
  them, and a tolerance would hide exactly the class of bug worth finding.
* **B — cycles.** demote_to_host / promote_to_device BETWEEN requests, three
  times, signature-checked, with the output still bitwise identical after
  re-promotion.
* **C — the number varena exists for.** pgw#1497's pricing arms re-run with
  arena backing: resident-equivalent + 50/25/5 % of the same byte basis,
  against the software rung's measured 1.91x / 3.16x / 3.56x. One config,
  warmup, best of two.
* **D — cold load.** Meta-init tree + RefillEngine (disk -> arena, no torch
  allocation on the way) against `from_pretrained(...).to("cuda")`.

Discipline, stated so the numbers can be read: ONE config, a warmup before
every timed set, best-of-2, `uptime` recorded, and both the torch-allocator
peak and the DEVICE peak reported — the arena's bytes are invisible to
`torch.cuda.max_memory_allocated` by construction.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SNAPSHOT = Path(
    "/home/fidika/.cache/huggingface/hub/"
    "models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/"
    "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
)
PROMPT = "a photograph of an astronaut riding a horse"
STEPS = 25
SIZE = 512
GUIDANCE = 7.5
SEED = 1507
MIB = 1 << 20
GIB = 1 << 30


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------


class DevicePeak:
    """Device-level peak bytes, sampled. The arena is outside torch's allocator.

    `torch.cuda.max_memory_allocated` reports what TORCH allocated and is blind
    to `cuMemMap`-ed chunks by construction, so a facade run measured that way
    would report a peak that is missing the weights. This samples the driver's
    own free/total, which is the number pgw#1497's table reports.
    """

    def __init__(self, torch: Any, device: Any, hz: int = 50) -> None:
        self._torch = torch
        self._device = device
        self._interval = 1.0 / hz
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak = 0

    def _run(self) -> None:
        while not self._stop.is_set():
            free, total = self._torch.cuda.mem_get_info(self._device)
            self.peak = max(self.peak, total - free)
            self._stop.wait(self._interval)

    def __enter__(self) -> "DevicePeak":
        self.peak = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def load_line() -> str:
    return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()


def settle(torch: Any) -> None:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def build_pipeline(torch: Any) -> Any:
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        str(SNAPSHOT),
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")


def generate(torch: Any, pipe: Any) -> Any:
    """One request. Latent output, so the comparison is of the model's own bytes."""
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    with torch.no_grad():
        out = pipe(
            PROMPT,
            num_inference_steps=STEPS,
            height=SIZE,
            width=SIZE,
            guidance_scale=GUIDANCE,
            generator=generator,
            output_type="latent",
        )
    torch.cuda.synchronize()
    return out.images.detach().clone()


def timed(torch: Any, pipe: Any, runs: int = 2) -> Tuple[float, int, Any]:
    """Best-of-``runs`` wall time after one warmup, plus the device peak."""
    latent = generate(torch, pipe)  # warmup
    times: List[float] = []
    peak = 0
    for _ in range(runs):
        settle(torch)
        with DevicePeak(torch, 0) as probe:
            start = time.perf_counter()
            latent = generate(torch, pipe)
            times.append(time.perf_counter() - start)
        peak = max(peak, probe.peak)
    return min(times), peak, latent


# ---------------------------------------------------------------------------
# The facade
# ---------------------------------------------------------------------------


def arena_over(torch: Any, pipe: Any, budget_bytes: int, **kwargs: Any) -> Any:
    """The facade's own production entry point — the caller states a budget."""
    from gen_worker.models.arena_residency import ArenaResidency

    return ArenaResidency.arm(pipe, device="cuda", budget_bytes=budget_bytes, **kwargs)


def weight_basis(torch: Any, pipe: Any) -> Tuple[int, List[Tuple[str, Any]]]:
    """The byte basis the budget percentages are taken of.

    pgw#1497's arms are percentages of the RAW weight bytes of the hooked
    tree, so this lane takes its percentages of the same number — otherwise
    "50 %" would mean two different budgets in the two tables and the
    comparison would be of nothing.
    """
    from gen_worker.models.memory import _named_components, unhookable_components
    from gen_worker.models.stream_residency import discover_leaves

    excluded = set(unhookable_components(pipe))
    roots = [
        (name, module)
        for name, module in _named_components(pipe)
        if name not in excluded and hasattr(module, "named_modules")
    ]
    _leaves, costs, _adapters = discover_leaves(roots)
    return sum(c.resident_bytes for c in costs), roots


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def arm_a_and_b(torch: Any, report: Dict[str, Any], cycles: int = 3) -> None:
    pipe = build_pipeline(torch)
    basis, _roots = weight_basis(torch, pipe)
    report["weight_basis_bytes"] = basis
    print(f"[basis] hooked-tree weight bytes = {basis / GIB:.3f} GiB")

    settle(torch)
    eager_ms, eager_peak, eager_latent = timed(torch, pipe)
    report["eager"] = {
        "ms_per_step": 1000 * eager_ms / STEPS,
        "device_peak_bytes": eager_peak,
        "torch_peak_bytes": int(torch.cuda.max_memory_allocated()),
    }
    print(
        f"[A] eager resident: {1000 * eager_ms / STEPS:.1f} ms/step, "
        f"device peak {eager_peak / GIB:.2f} GiB"
    )

    settle(torch)
    residency = arena_over(torch, pipe, budget_bytes=1 << 62)
    plan = residency.engage()
    layout = residency.layout
    report["layout"] = {
        "granularity": layout.granularity,
        "regions": len(layout.regions),
        "core_leaves": len(layout.core_names),
        "virtual_bytes": layout.virtual_bytes,
        "weight_bytes": layout.weight_bytes,
        "tax_bytes": layout.tax_bytes,
        "tax_fraction": layout.tax_bytes / max(1, layout.weight_bytes),
    }
    print(
        f"[A] layout: {len(layout.regions)} regions, "
        f"{layout.weight_bytes / GIB:.3f} GiB of weights in "
        f"{layout.virtual_bytes / GIB:.3f} GiB of span "
        f"(granularity tax {layout.tax_bytes / MIB:.0f} MiB = "
        f"{100 * layout.tax_bytes / layout.weight_bytes:.1f} %)"
    )
    print(f"[A] resident-equivalent plan: streams {len(plan.streamed)} leaves, fits={plan.fits}")
    assert not plan.streamed, "resident-equivalent budget must stream nothing"

    arena_ms, arena_peak, arena_latent = timed(torch, pipe)
    identical = bool(torch.equal(eager_latent, arena_latent))
    report["arena_resident_equivalent"] = {
        "ms_per_step": 1000 * arena_ms / STEPS,
        "device_peak_bytes": arena_peak,
        "bitwise_identical_to_eager": identical,
        "max_abs_diff": float((eager_latent - arena_latent).abs().max()),
        "arena_stats": {k: int(v) for k, v in residency.stats().items()},
    }
    print(
        f"[A] arena resident-equivalent: {1000 * arena_ms / STEPS:.1f} ms/step, "
        f"device peak {arena_peak / GIB:.2f} GiB, "
        f"BITWISE IDENTICAL = {identical}"
    )

    # -- B: demote / promote between requests ------------------------------
    cycle_report: List[Dict[str, Any]] = []
    for index in range(cycles):
        before_sig = int(residency.reservation.signature())
        before_mapped = int(residency.arena.stats()["mapped_bytes"])
        freed = residency.demote_to_host()
        demoted_mapped = int(residency.arena.stats()["mapped_bytes"])
        demote_sig = int(residency.reservation.signature())
        claimed = residency.promote_to_device()
        after_mapped = int(residency.arena.stats()["mapped_bytes"])
        promote_sig = int(residency.reservation.signature())
        again = generate(torch, pipe)
        same = bool(torch.equal(eager_latent, again))
        # Every region the plan calls resident must really be mapped, asked of
        # the driver and not of our own bookkeeping.
        backed = all(
            residency.reservation.is_backed(r.offset, r.span)
            for r in residency.layout.regions
            if residency.is_resident(r.name)
        )
        cycle_report.append(
            {
                "freed_bytes": freed,
                "claimed_bytes": claimed,
                "mapped_before": before_mapped,
                "mapped_after_demote": demoted_mapped,
                "mapped_after_promote": after_mapped,
                "signature_before": before_sig,
                "signature_after_demote": demote_sig,
                "signature_after_promote": promote_sig,
                "signature_advanced_on_demote": demote_sig > before_sig,
                "signature_stable_across_promote": promote_sig == demote_sig,
                "every_resident_region_is_backed": backed,
                "bitwise_identical_to_eager": same,
            }
        )
        print(
            f"[B] cycle {index + 1}: freed {freed / GIB:.3f} GiB, mapped "
            f"{before_mapped / MIB:.0f} -> {demoted_mapped / MIB:.0f} -> "
            f"{after_mapped / MIB:.0f} MiB, signature {before_sig} -> "
            f"{demote_sig} -> {promote_sig}, backed={backed}, IDENTICAL={same}"
        )
    report["cycles"] = cycle_report
    return pipe, residency, eager_latent, report


def arm_c(
    torch: Any, pipe: Any, residency: Any, eager_latent: Any, report: Dict[str, Any]
) -> None:
    basis = report["weight_basis_bytes"]
    rows: List[Dict[str, Any]] = []
    for fraction in (0.5, 0.25, 0.05):
        budget = int(basis * fraction)
        settle(torch)
        plan = residency.rebudget(budget)
        ms, peak, latent = timed(torch, pipe)
        stats = {k: int(v) for k, v in residency.stats().items()}
        rows.append(
            {
                "fraction": fraction,
                "budget_bytes": budget,
                "ms_per_step": 1000 * ms / STEPS,
                "device_peak_bytes": peak,
                "streamed_leaves": len(plan.streamed),
                "resident_leaves": len(plan.all_resident),
                "plan_resident_bytes": plan.resident_bytes,
                "plan_window_bytes": plan.window_bytes,
                "plan_fits": plan.fits,
                "host_bytes": plan.host_bytes,
                "bitwise_identical_to_eager": bool(torch.equal(eager_latent, latent)),
                "max_abs_diff": float((eager_latent - latent).abs().max()),
                "arena_stats": stats,
            }
        )
        print(
            f"[C] {int(100 * fraction):>3}% budget ({budget / GIB:.3f} GiB): "
            f"{1000 * ms / STEPS:.1f} ms/step, device peak {peak / GIB:.2f} GiB, "
            f"{len(plan.streamed)} streamed, page-ins {stats['page_ins']}, "
            f"unbacks {stats['unbacks']}, identical="
            f"{rows[-1]['bitwise_identical_to_eager']}"
        )
    report["pricing"] = rows


def evict_page_cache(path: Path) -> bool:
    """Drop ``path`` from the page cache. No root needed, and it is the point.

    A "cold load" measured with the checkpoint already in RAM is measuring
    memcpy, not loading — and it is exactly the arm where a loader that mmaps
    the file looks fastest, for a reason that will not hold on a fresh pod.
    ``fadvise DONTNEED`` gives the honest arm without touching ``drop_caches``.
    """
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return False
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return True
    finally:
        os.close(fd)


def arm_d(torch: Any, report: Dict[str, Any]) -> None:
    """Cold load: RefillEngine (disk -> arena) vs `from_pretrained().to(cuda)`.

    Both arms twice: with the checkpoint in the page cache, and with it evicted.
    """
    from diffusers import UNet2DConditionModel

    from gen_worker.models.arena_residency import ArenaResidency, safetensors_triples

    unet_dir = SNAPSHOT / "unet"
    weight_file = (unet_dir / "diffusion_pytorch_model.fp16.safetensors").resolve()
    triples = safetensors_triples(unet_dir, variant="fp16")
    config = UNet2DConditionModel.load_config(str(unet_dir))
    reference: Dict[str, Any] = {}

    def loader_arm(evict: bool) -> float:
        settle(torch)
        if evict:
            evict_page_cache(weight_file)
        start = time.perf_counter()
        model = UNet2DConditionModel.from_pretrained(
            str(unet_dir), torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if not reference:
            reference.update({k: v.detach().to("cpu") for k, v in model.state_dict().items()})
        del model
        settle(torch)
        return elapsed

    def arena_arm(evict: bool) -> Tuple[float, int, List[str]]:
        settle(torch)
        if evict:
            evict_page_cache(weight_file)
        start = time.perf_counter()
        with torch.device("meta"):
            model = UNet2DConditionModel.from_config(config).to(torch.float16)
        residency = ArenaResidency(
            [("unet", model)], device="cuda", budget_bytes=1 << 62, triples=triples
        )
        residency.engage()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        bad = [
            key
            for key, value in model.state_dict().items()
            if key in reference and not torch.equal(value.detach().to("cpu"), reference[key])
        ]
        weight_bytes = residency.layout.weight_bytes
        residency.release()
        del model, residency
        settle(torch)
        return elapsed, weight_bytes, bad

    warm_loader = loader_arm(evict=False)
    warm_arena, weight_bytes, bad_warm = arena_arm(evict=False)
    cold_loader = loader_arm(evict=True)
    cold_arena, _bytes, bad_cold = arena_arm(evict=True)

    rows: Dict[str, Any] = {}
    for label, loader_s, arena_s in (
        ("page_cache_warm", warm_loader, warm_arena),
        ("page_cache_evicted", cold_loader, cold_arena),
    ):
        rows[label] = {
            "loader_seconds": loader_s,
            "arena_refill_seconds": arena_s,
            "loader_gbps": weight_bytes / loader_s / 1e9,
            "arena_gbps": weight_bytes / arena_s / 1e9,
            "loader_over_arena": loader_s / arena_s if arena_s else 0.0,
        }
        print(
            f"[D] {label}: loader {loader_s:.2f} s "
            f"({weight_bytes / loader_s / 1e9:.2f} GB/s) vs RefillEngine->arena "
            f"{arena_s:.2f} s ({weight_bytes / arena_s / 1e9:.2f} GB/s)"
        )
    rows["weight_bytes"] = weight_bytes
    rows["mismatched_tensors_warm"] = len(bad_warm)
    rows["mismatched_tensors_evicted"] = len(bad_cold)
    rows["mismatched_sample"] = (bad_warm + bad_cold)[:5]
    report["cold_load"] = rows
    print(
        f"[D] byte-exactness: {len(bad_warm)} mismatched warm, "
        f"{len(bad_cold)} mismatched evicted (of {len(reference)} tensors)"
    )


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="ABCD")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument(
        "--out", default=os.environ.get("PGW1507_OUT", "/tmp/pgw1507-arena-facade.json")
    )
    args = parser.parse_args()

    import torch

    report: Dict[str, Any] = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "load_at_start": load_line(),
        "config": {
            "prompt": PROMPT, "steps": STEPS, "size": SIZE,
            "guidance": GUIDANCE, "seed": SEED, "dtype": "fp16",
        },
    }
    import varena

    report["varena"] = varena.__version__
    print(f"[env] {report['device']}, torch {torch.__version__}, varena {varena.__version__}")
    print(f"[env] {report['load_at_start']}")

    pipe = residency = eager_latent = None
    try:
        if "A" in args.arms or "B" in args.arms or "C" in args.arms:
            pipe, residency, eager_latent, report = arm_a_and_b(
                torch, report, cycles=args.cycles if "B" in args.arms else 0
            )
        if "C" in args.arms:
            arm_c(torch, pipe, residency, eager_latent, report)
        if residency is not None:
            residency.release()
            del residency, pipe, eager_latent
            settle(torch)
        if "D" in args.arms:
            arm_d(torch, report)
    finally:
        report["load_at_end"] = load_line()
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
