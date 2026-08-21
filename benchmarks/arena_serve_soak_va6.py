"""Real-serve soak: sd1.5 served out of the arena while residency churns underneath.

Two verdicts per run, deliberately different instruments:

1. OUTPUT IDENTITY — every Nth request compared BITWISE against a banked eager
   reference for the same seed; any divergence is red, with the churn history
   since the last green banked for repro.
2. CONTENT INTEGRITY — per-region digests taken from the LIVE weights before
   adoption, re-verified on every re-promotion and at teardown; catches
   corruption that a byte-identical output would hide.

Residency may move only BETWEEN requests: the churn thread takes
``request_lock`` before acting, so it blocks while a generation is in flight.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

SNAPSHOT = Path(
    "/home/fidika/.cache/huggingface/hub/"
    "models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/"
    "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
)
PROMPT = "a photograph of an astronaut riding a horse"
STEPS = 25
SIZE = 512
GUIDANCE = 7.5
MIB = 1 << 20
GIB = 1 << 30
POKE_MODES = ("byte", "weight")


def load_digest(path: str):
    """Import varena's digest primitive."""
    directory = Path(os.environ.get("VARENA_TESTS", path)).expanduser()
    module = directory / "digest.py"
    if not module.exists():
        raise SystemExit(
            f"REFUSING: varena's digest primitive is not at {module}. This soak shares ONE\n"
            "digest definition with varena's tests/integrity_soak.py; reimplementing it here\n"
            "would let the two instruments disagree in silence. Pass --varena-tests <dir>."
        )
    sys.path.insert(0, str(directory))
    import digest as D

    return D


def require_window() -> None:
    if os.environ.get("VARENA_GPU_WINDOW") != "1":
        sys.stderr.write(
            "REFUSING: this run drives the shared GPU for hours and VARENA_GPU_WINDOW is not\n"
            "set. Ask the coordinator for a window, export it for the window's life, unset it\n"
            "at handback. (--self-test needs neither a window nor a card.)\n"
        )
        raise SystemExit(4)


def uptime_line() -> str:
    return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()


def nvsmi(query: str) -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<unavailable: {exc}>"


def substrate(config: dict[str, Any]) -> dict[str, Any]:
    """Card identity comes from nvidia-smi, never torch: torch.cuda.get_device_name() creates a ~128 MiB CUDA context as a side effect."""
    return {
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "uptime": uptime_line(),
        "gpu": nvsmi("name,driver_version,memory.total,memory.used,temperature.gpu,clocks.sm"),
        "python": sys.version.split()[0],
        "pgw_commit": subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip(),
        "pgw_branch": subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip(),
        "nice": os.nice(0),
        "config": config,
    }


def bank(path: str, payload: dict[str, Any]) -> str:
    if "substrate" not in payload:
        raise ValueError("refusing to bank a result with no substrate block")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


class RegionDigests:
    """One digest per region, folded over its SLOTS — never over its span."""

    def __init__(self, D, torch, residency) -> None:
        self.D = D
        self.torch = torch
        self.res = residency
        self.reference: dict[str, int] = {}
        self.last_checked = 0

    def _fold(self, byte_views) -> int:
        acc = 0
        consumed = 0
        for view in byte_views:
            acc = (acc + self.D.digest_tensor(view, base_lane=consumed // 8)) & self.D.MASK
            consumed += int(view.numel())
        return (acc ^ ((consumed * self.D.K3) & self.D.MASK)) & self.D.MASK

    def _as_bytes(self, t):
        return t.detach().contiguous().view(-1).view(self.torch.uint8)

    def take_reference(self) -> dict[str, int]:
        """From the LIVE tree, BEFORE the arena has touched a byte."""
        for region in self.res.layout.regions:
            views = [self._as_bytes(self.res._live(slot)) for slot in region.slots]
            self.reference[region.name] = self._fold(views)
        return self.reference

    def device_digest(self, region) -> int:
        views = [
            self.torch.from_dlpack(
                self.res.reservation.tensor(slot.offset, [slot.nbytes], 1, 8)
            )
            for slot in region.slots
        ]
        return self._fold(views)

    def verify(self, where: str) -> list[dict[str, Any]]:
        """Every region the facade calls resident, digested against the reference."""
        bad = []
        checked = 0
        for region in self.res.layout.regions:
            if not self.res.is_resident(region.name):
                continue
            checked += 1
            want = self.reference[region.name]
            got = self.device_digest(region)
            if got != want:
                bad.append(
                    {
                        "where": where,
                        "region": region.name,
                        "offset": region.offset,
                        "span": region.span,
                        "weight_bytes": region.weight_bytes,
                        "want": f"{want:#018x}",
                        "got": f"{got:#018x}",
                        "matches_other_region": self._whose(got),
                    }
                )
        self.last_checked = checked
        return bad

    def _whose(self, got: int) -> str | None:
        for name, value in self.reference.items():
            if value == got:
                return name
        return None


class ServeSoak:
    def __init__(self, args, D) -> None:
        self.args = args
        self.D = D
        self.history: deque = deque(maxlen=args.history)
        self.request_lock = threading.Lock()
        self.in_request = False
        self.stop = threading.Event()
        self.requests = 0
        self.comparisons = 0
        self.churn_actions = 0
        self.digest_passes = 0
        self.regions_digested = 0
        self.boundary_violations = 0
        self.divergences: list[dict[str, Any]] = []
        self.mismatches: list[dict[str, Any]] = []
        self.poke_record: dict[str, Any] | None = None
        self.poke_done = False
        self.poke_digest_caught: bool | None = None
        self.poke_output_caught: bool | None = None
        self.churn_error: str | None = None
        self.t0 = time.time()

    def record(self, kind: str, **kw) -> None:
        self.history.append(
            {"req": self.requests, "t": round(time.time() - self.t0, 3), "kind": kind, **kw}
        )

    def generate(self, torch, pipe, seed: int):
        generator = torch.Generator(device="cuda").manual_seed(seed)
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

    def churn_once(self, rng, residency, digests, basis: int) -> None:
        action = rng.choice(
            ["rebudget-low", "rebudget-full", "demote-promote", "partial-unload", "partial-load"]
        )
        before = int(residency.reservation.signature())
        detail: dict[str, Any] = {}
        if action == "rebudget-low":
            frac = rng.choice([0.05, 0.25, 0.5])
            plan = residency.rebudget(int(basis * frac))
            detail = {"fraction": frac, "streamed": len(plan.streamed), "fits": bool(plan.fits)}
        elif action == "rebudget-full":
            plan = residency.rebudget(basis)
            detail = {"streamed": len(plan.streamed), "fits": bool(plan.fits)}
        elif action == "demote-promote":
            freed = residency.demote_to_host()
            claimed = residency.promote_to_device()
            detail = {"freed": freed, "claimed": claimed}
        elif action == "partial-unload":
            detail = {"released": residency.partial_unload(rng.randint(1, 8) * 64 * MIB)}
        else:
            detail = {"claimed": residency.partial_load(rng.randint(1, 8) * 64 * MIB)}
        self.churn_actions += 1
        after = int(residency.reservation.signature())
        stats = {k: int(v) for k, v in residency.stats().items()}
        self.record("churn", action=action, sig_before=before, sig_after=after,
                    stats=stats, **detail)

        bad = digests.verify(f"after-{action}")
        self.digest_passes += 1
        self.regions_digested += digests.last_checked
        if bad:
            self.mismatches.extend(bad)
            raise AssertionError(
                f"CONTENT INTEGRITY: {len(bad)} region(s) hold the wrong bytes after "
                f"{action}: {json.dumps(bad[:3])}"
            )

    def churn_thread(self, rng, residency, digests, basis: int) -> None:
        while not self.stop.is_set():
            acquired = True
            if not self.args.prove_boundary_matters:
                acquired = self.request_lock.acquire(timeout=5.0)
                if not acquired:
                    continue
            try:
                if self.in_request:
                    self.boundary_violations += 1
                self.churn_once(rng, residency, digests, basis)
            except Exception as exc:  # noqa: BLE001
                self.churn_error = f"{type(exc).__name__}: {exc}"
                self.stop.set()
                return
            finally:
                if acquired and not self.args.prove_boundary_matters:
                    self.request_lock.release()
            time.sleep(self.args.churn_gap)

    def poke(self, torch, residency, digests, rng) -> None:
        live = [r for r in residency.layout.regions if residency.is_resident(r.name)
                and r.name != "__core__" and r.slots]
        if not live:
            return
        region = rng.choice(live)
        slot = max(region.slots, key=lambda s: s.nbytes)
        view = torch.from_dlpack(residency.reservation.tensor(slot.offset, [slot.nbytes], 1, 8))
        if self.args.poke == "byte":
            at = slot.nbytes // 2
            view[at] = view[at] ^ 1
            detail = {"byte_offset": at}
        else:
            view.fill_(0xA5)
            detail = {"bytes_destroyed": slot.nbytes}
        torch.cuda.synchronize()
        self.poke_done = True
        self.poke_record = {
            "mode": self.args.poke, "region": region.name, "slot": f"{slot.leaf}.{slot.attr}",
            "slot_bytes": slot.nbytes, "at_request": self.requests, **detail,
        }
        self.record("POKE", **self.poke_record)
        print(f"[poke] {self.args.poke} into {region.name} / {slot.leaf}.{slot.attr}")

    def run(self) -> int:
        args = self.args
        import torch

        require_window()
        rng = random.Random(args.seed)
        report: dict[str, Any] = {}

        from diffusers import StableDiffusionPipeline

        print("[boot] loading sd1.5")
        pipe = StableDiffusionPipeline.from_pretrained(
            str(args.snapshot), torch_dtype=torch.float16, variant="fp16",
            safety_checker=None, requires_safety_checker=False,
        )
        pipe.set_progress_bar_config(disable=True)
        pipe = pipe.to("cuda")

        armed = getattr(getattr(pipe.unet, "forward", None), "_entries", None)
        if armed:
            raise SystemExit(
                f"REFUSING: {len(armed)} compiled graph(s) are armed on the unet. This soak "
                "is the EAGER arm; compiled-under-churn is roadmap phase 3's."
            )

        seeds = [args.seed + i for i in range(args.seeds)]
        print(f"[refs] generating {len(seeds)} eager reference latents")
        references = {}
        for s in seeds:
            references[s] = self.generate(torch, pipe, s)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        from gen_worker.models.arena_residency import ArenaResidency, safetensors_triples

        triples = None
        if args.no_host_mirror:
            triples = {}
            for component in ("unet", "vae", "text_encoder"):
                part = safetensors_triples(args.snapshot / component, variant="fp16")
                clash = set(part) & set(triples)
                if clash:
                    raise SystemExit(
                        f"REFUSING: triple keys collide across components ({sorted(clash)[:3]}); "
                        "a disk refill would read the wrong component's bytes. Run the "
                        "host-mirror shape, or arm over one component."
                    )
                triples.update(part)

        residency = ArenaResidency.arm(
            pipe, device="cuda", budget_bytes=1 << 62,
            triples=triples, host_mirror=not args.no_host_mirror,
        )
        basis = sum(r.span for r in residency.layout.regions)

        digests = RegionDigests(self.D, torch, residency)
        print(f"[refs] digesting {len(residency.layout.regions)} regions from the LIVE tree")
        digests.take_reference()
        distinct = len(set(digests.reference.values()))
        if distinct != len(digests.reference):
            print(
                f"[warn] {len(digests.reference) - distinct} regions share a reference digest "
                "(identical weights); a cross-region DMA between them would be invisible"
            )

        residency.engage()
        bad = digests.verify("after-engage")
        if bad:
            self.mismatches.extend(bad)
            raise SystemExit(f"CONTENT INTEGRITY failed at adoption: {json.dumps(bad[:3])}")
        self.digest_passes += 1
        self.regions_digested += digests.last_checked
        print(f"[refs] adoption verified: {digests.last_checked} regions byte-correct")

        if args.verify_page_ins:
            self._wrap_page_in(residency, digests)

        thread = threading.Thread(
            target=self.churn_thread, args=(rng, residency, digests, basis), daemon=True
        )
        thread.start()

        deadline = time.time() + args.wall_secs
        failure = None
        try:
            while not self.stop.is_set() and time.time() < deadline:
                seed = seeds[self.requests % len(seeds)]
                with self.request_lock:
                    self.in_request = True
                    try:
                        latent = self.generate(torch, pipe, seed)
                    finally:
                        self.in_request = False
                self.requests += 1
                self.record("request", seed=seed)

                if args.poke and not self.poke_done and self.requests >= args.poke_at:
                    with self.request_lock:
                        self.poke(torch, residency, digests, rng)
                        found = digests.verify("post-poke")
                    self.poke_digest_caught = bool(found)
                    self.mismatches.extend(found)
                    after = self.generate(torch, pipe, seed)
                    self.poke_output_caught = not bool(torch.equal(references[seed], after))
                    break

                if self.requests % args.compare_every == 0:
                    self.comparisons += 1
                    if not bool(torch.equal(references[seed], latent)):
                        diff = float((references[seed].float() - latent.float()).abs().max())
                        self.divergences.append(
                            {"request": self.requests, "seed": seed, "max_abs_diff": diff}
                        )
                        failure = (
                            f"OUTPUT DIVERGED at request {self.requests} (seed {seed}), "
                            f"max abs diff {diff}"
                        )
                        break
                if self.requests % args.progress_every == 0:
                    self._progress(residency)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop.set()
            thread.join(timeout=30)

        if failure is None and self.churn_error:
            failure = self.churn_error

        if not args.poke:
            try:
                final = digests.verify("teardown")
                self.digest_passes += 1
                self.regions_digested += digests.last_checked
                if final:
                    self.mismatches.extend(final)
                    failure = failure or f"CONTENT INTEGRITY at teardown: {json.dumps(final[:3])}"
            except Exception as exc:  # noqa: BLE001
                failure = failure or f"teardown verify raised: {type(exc).__name__}: {exc}"

        stats = {k: int(v) for k, v in residency.stats().items()}
        residency.release()
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

        report = {
            "substrate": substrate(
                {
                    "arm": "real-serve-soak",
                    "model": "sd1.5",
                    "execution": "eager",
                    "steps": STEPS, "size": SIZE, "guidance": GUIDANCE,
                    "seeds": seeds, "compare_every": args.compare_every,
                    "churn_gap_secs": args.churn_gap,
                    "host_mirror": not args.no_host_mirror,
                    "verify_page_ins": args.verify_page_ins,
                    "wall_secs": args.wall_secs,
                    "poke": args.poke, "poke_at": args.poke_at,
                    "weight_basis_bytes": basis,
                    "regions": len(residency.layout.regions),
                }
            ),
            "requests": self.requests,
            "bitwise_comparisons": self.comparisons,
            "output_divergences": self.divergences,
            "churn_actions": self.churn_actions,
            "digest_passes": self.digest_passes,
            "region_digests": self.regions_digested,
            "content_mismatches": self.mismatches,
            "boundary_violations_observed": self.boundary_violations,
            "arena_stats_at_end": stats,
            "poke": self.poke_record,
            "poke_digest_caught": self.poke_digest_caught,
            "poke_output_caught": self.poke_output_caught,
            "failure": failure,
            "history_since_last_green": list(self.history),
            "elapsed_secs": time.time() - self.t0,
        }
        out = os.path.join(args.out, f"serve-soak-{args.poke or 'clean'}.json")
        bank(out, report)
        print(f"[bank] {out}")

        if args.poke:
            ok = bool(self.poke_digest_caught)
            print(
                f"POKE RUN ({args.poke}): digest {'CAUGHT' if ok else 'MISSED'} it; "
                f"bitwise output comparison "
                f"{'also caught it' if self.poke_output_caught else 'did NOT catch it'}"
            )
            if args.poke == "byte" and ok and not self.poke_output_caught:
                print(
                    "  ^ THIS IS THE POINT OF ARM 2's SECOND INSTRUMENT: one rotten byte in a\n"
                    "    served weight produced a BITWISE IDENTICAL image. Output identity alone\n"
                    "    would have reported this soak green."
                )
            if not ok:
                print("  THE DIGEST IS BLIND. Loudest possible result.")
            return 0 if ok else 1

        if failure:
            print(f"SERVE SOAK FAILED: {failure}")
            return 1
        print(
            f"serve soak GREEN: {self.requests} requests, {self.comparisons} bitwise "
            f"comparisons, {self.churn_actions} churn actions, {self.digest_passes} digest "
            f"passes over {self.regions_digested} regions, "
            f"{self.boundary_violations} boundary violations"
        )
        return 0

    def _wrap_page_in(self, residency, digests) -> None:
        original = residency._page_in
        soak = self

        def verified(region):
            original(region)
            want = digests.reference[region.name]
            got = digests.device_digest(region)
            soak.digest_passes += 1
            soak.regions_digested += 1
            if got != want:
                bad = {
                    "where": "page-in", "region": region.name,
                    "want": f"{want:#018x}", "got": f"{got:#018x}",
                    "matches_other_region": digests._whose(got),
                }
                soak.mismatches.append(bad)
                raise AssertionError(f"CONTENT INTEGRITY at page-in: {json.dumps(bad)}")

        residency._page_in = verified

    def _progress(self, residency) -> None:
        print(
            f"[{time.time() - self.t0:8.0f}s] requests={self.requests} "
            f"compares={self.comparisons} churn={self.churn_actions} "
            f"digests={self.regions_digested} stats={residency.stats()}",
            flush=True,
        )


def self_test(args, D) -> int:
    import torch

    fails: list[str] = []

    def check(name, cond, detail=""):
        if not cond:
            fails.append(f"{name}{': ' + detail if detail else ''}")

    check("varena digest selftest", D.selftest(verbose=False) == 0,
          "the shared primitive fails its own red-arm in the pgw venv")

    saved = os.environ.pop("VARENA_GPU_WINDOW", None)
    try:
        require_window()
        fails.append("require_window() let a run through with no window")
    except SystemExit as exc:
        check("window gate exit code", exc.code == 4, f"exited {exc.code}, wanted 4")
    finally:
        if saved is not None:
            os.environ["VARENA_GPU_WINDOW"] = saved

    rd = RegionDigests(D, torch, residency=None)
    tensors = [torch.randn(500, dtype=torch.float32), torch.randn(250, dtype=torch.float32)]
    views = [rd._as_bytes(t) for t in tensors]
    base = rd._fold(views)
    check("fold is stable", base == rd._fold([rd._as_bytes(t) for t in tensors]))
    for i, t in enumerate(tensors):
        flat = t.view(-1).view(torch.uint8)
        flat[len(flat) // 3] ^= 1
        check(f"fold sees a byte flip in slot {i}",
              rd._fold([rd._as_bytes(x) for x in tensors]) != base)
        flat[len(flat) // 3] ^= 1
    check("fold restored", rd._fold([rd._as_bytes(t) for t in tensors]) == base)
    check("fold is order sensitive", rd._fold(list(reversed(views))) != base)
    check("fold sees a missing slot", rd._fold(views[:1]) != base)
    shifted = rd._as_bytes(tensors[0])[8:]
    check("fold is position sensitive", rd._fold([shifted]) != rd._fold([views[0]]))

    ref = torch.randn(1, 4, 64, 64, dtype=torch.float16)
    check("comparator green on a clone", bool(torch.equal(ref, ref.clone())))
    doctored = ref.clone()
    doctored.view(-1)[7] = torch.tensor(
        float(doctored.view(-1)[7]) + 0.001, dtype=torch.float16
    )
    check("comparator red on one perturbed element", not bool(torch.equal(ref, doctored)))
    same_bits = ref.clone()
    same_bits.view(torch.int16).view(-1)[11] ^= 1
    check("comparator red on one flipped MANTISSA BIT", not bool(torch.equal(ref, same_bits)))

    soak = ServeSoak.__new__(ServeSoak)
    soak.history = deque(maxlen=200)
    soak.request_lock = threading.Lock()
    soak.in_request = False
    soak.boundary_violations = 0
    soak.requests = 0
    soak.t0 = time.time()
    stop = threading.Event()
    observed = {"violations": 0, "acts": 0}

    def fake_churn():
        while not stop.is_set():
            with soak.request_lock:
                if soak.in_request:
                    observed["violations"] += 1
                observed["acts"] += 1
                time.sleep(0.0005)

    t = threading.Thread(target=fake_churn, daemon=True)
    t.start()
    for _ in range(200):
        with soak.request_lock:
            soak.in_request = True
            time.sleep(0.0005)
            soak.in_request = False
    stop.set()
    t.join(timeout=5)
    check("churn actually ran", observed["acts"] > 10, f"only {observed['acts']} actions")
    check("boundary respected", observed["violations"] == 0,
          f"{observed['violations']} churn actions saw a live request")

    soak.in_request = False
    stop2 = threading.Event()
    seen = {"violations": 0}

    def unlocked_churn():
        while not stop2.is_set():
            if soak.in_request:
                seen["violations"] += 1
            time.sleep(0.0002)

    t2 = threading.Thread(target=unlocked_churn, daemon=True)
    t2.start()
    for _ in range(200):
        soak.in_request = True
        time.sleep(0.0005)
        soak.in_request = False
    stop2.set()
    t2.join(timeout=5)
    check("violation detector can go red", seen["violations"] > 0,
          "the unlocked variant saw no violation — the detector is blind")

    try:
        bank(os.path.join(args.out, "naked.json"), {"requests": 1})
        fails.append("bank() accepted a result with no substrate block")
    except ValueError:
        pass

    for f in fails:
        print("RED-ARM FAIL:", f)
    print(f"serve-soak self-test: {len(fails)} failures")
    return 1 if fails else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="benchmarks/va6")
    p.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    p.add_argument("--varena-tests", default="~/cozy/varena/tests")
    p.add_argument("--wall-secs", type=float, default=900.0)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=1507)
    p.add_argument("--compare-every", type=int, default=2)
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--churn-gap", type=float, default=0.05)
    p.add_argument("--history", type=int, default=4000)
    p.add_argument("--no-host-mirror", action="store_true",
                   help="RAM half of zero: every page-in comes off disk through RefillEngine")
    p.add_argument("--verify-page-ins", action="store_true",
                   help="digest EVERY streamed refill as it lands (short legs only)")
    p.add_argument("--prove-boundary-matters", action="store_true",
                   help="DESTRUCTIVE: churn without the request lock, to show it is load-bearing")
    p.add_argument("--poke", choices=POKE_MODES, default=None)
    p.add_argument("--poke-at", type=int, default=3)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    D = load_digest(args.varena_tests)
    if args.self_test:
        return self_test(args, D)
    return ServeSoak(args, D).run()


if __name__ == "__main__":
    sys.exit(main())
