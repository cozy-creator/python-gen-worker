"""pgw#1548 box-side pod driver: rent, BANK THE ID, poll the hub, tear down.

Renting and arming teardown are ONE act (`podguard.rent`), the guard is armed
BEFORE any wait, and the pod id is written to a durable file AND printed the
moment it exists — before the first wait loop — so a session that dies mid-leg
still leaves the coordinator something to kill from the provider API.

Progress is read from the HUB, never from elapsed time: the pod publishes a
verdict document per stage, and this driver polls the release's variant list.
That is the only channel a raw pod has (pgw#1568: SSH does not reach rented
pods and RunPod has no logs API).
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path.home() / "cozy/cozy-creator-tracker/scripts/podguard"))
import podguard  # noqa: E402

CRED = Path.home() / ".cache/cozy/pgw1548/hub-credentials.json"
PODS = Path.home() / ".cache/cozy/pgw1548/pods.jsonl"
HERE = Path(__file__).resolve().parent

#: A CUDA image even for the CPU leg: the derive imports torch, and a pod whose
#: torch cannot import is a rental that buys nothing.
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"


#: MEASURED CEILING, 2026-08-20: RunPod's REST create accepts a ~50 KB `env`
#: and returns HTTP 500 on ~150 KB -- and the 500 is its own backend failing
#: to parse an upstream error (`invalid character 'I'`), so it reads as a
#: server fault rather than as "your body is too big". Isolated with three
#: probe creates (tiny / 50 KB / 150 KB); the first two created and were
#: terminated immediately. The endpoint source therefore rides the hub.
ENV_BYTES_CEILING = 60_000


def endpoint_tarball(name: str) -> str:
    """The MINIMAL endpoint source as tar.gz+base64, from `origin/master`.

    `src` + `endpoint.toml` + `pyproject.toml` only. The full archive is
    141 KB (anima) / 108 KB (sdxl) and blows the env ceiling above; nearly
    all of that is `uv.lock`, which the pod does not use because it builds
    its own venv. Minimal is 27 KB / 17 KB.
    """

    repo = Path.home() / "cozy/serverless-endpoints"
    raw = subprocess.run(
        ["git", "archive", "--format=tar.gz", f"origin/master:{name}",
         "src", "endpoint.toml", "pyproject.toml"],
        cwd=repo, capture_output=True, check=True).stdout
    return base64.b64encode(raw).decode()


def bank(pod_id: str, lane: str, extra: dict) -> None:
    """Write the id where a DIFFERENT session can find it. Before any wait."""

    PODS.parent.mkdir(parents=True, exist_ok=True)
    row = {"pod_id": pod_id, "lane": lane, "at": time.strftime("%FT%T%z"), **extra}
    with PODS.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    print(f"\n*** POD ID BANKED: {pod_id}  ({lane}) -> {PODS}")
    print(f"*** kill with: python3 {podguard.__file__} release {pod_id}\n",
          flush=True)


def published_stages(cred: dict) -> list[str]:
    """Which stages have landed. The hub is the progress signal, not a clock.

    Read from `GET /repos/:org/:name/tree?release=…`, which lists real file
    PATHS. The release listing (`/releases/:release`) was the obvious place to
    look and is the WRONG one: it reports `checkpoint_id`, `size_bytes` and a
    file COUNT, but no names — so it can say a pod is progressing and never say
    at what. Verified against a live publish before this driver depended on it.
    """

    url = (f"{cred['hub_local']}/api/v1/repos/{cred['org']}/{cred['repo']}"
           f"/tree?release={cred['release']}")
    req = urllib.request.Request(url)
    req.add_header("Authorization", "Bearer " + cred["access_token"])
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        # A hub read that fails is UNKNOWN, never "no progress" — treating it
        # as absence is how a working pod gets killed for a network blip.
        print(f"[poll] hub read failed ({exc}); progress UNKNOWN this tick")
        return []
    stages: list[str] = []
    for entry in body.get("entries") or []:
        path = str(entry.get("path") or "")
        head = path.split("/", 1)[0] if "/" in path else path.removesuffix(".json")
        if head and head not in stages:
            stages.append(head)
    return stages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("anima-derive", "sdxl-matrix"))
    parser.add_argument("--endpoint", required=True, help="anima | sdxl")
    parser.add_argument("--gpu", default="", help="RunPod gpuTypeId; empty = CPU pod")
    parser.add_argument("--disk", type=int, default=120)
    parser.add_argument("--vcpu", type=int, default=8)
    parser.add_argument("--cpu-flavors", default="cpu5m,cpu3m",
                        help="RunPod CPU flavors, priority order; the anima derive is MEMORY-bound (weight-full loads), so the memory-optimised flavors lead")
    parser.add_argument("--ram", type=int, default=64)
    parser.add_argument("--budget-usd", type=float, default=4.0)
    parser.add_argument("--max-minutes", type=float, default=180.0)
    parser.add_argument("--expect", default="",
                        help="comma-separated stages that mean SUCCESS")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.cpu_flavors = [f for f in args.cpu_flavors.split(",") if f]

    cred = json.loads(CRED.read_text())
    sha = subprocess.run(["git", "rev-parse", "origin/master"],
                         cwd=Path.home() / "cozy/python-gen-worker",
                         capture_output=True, text=True, check=True).stdout.strip()

    env = {
        "PGW1548_MODE": args.mode,
        "PGW1548_HUB": cred["hub_public"],
        "PGW1548_TOKEN": cred["machine_token"],
        "PGW1548_REPO": f"{cred['org']}/{cred['repo']}",
        "PGW1548_RELEASE": cred["release"],
        "PGW1548_SHA": sha,
        "PGW1548_ENDPOINT_B64": endpoint_tarball(args.endpoint),
    }
    hf = ""
    for line in (Path.home() / "cozy/e2e/.env").read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            hf = line.split("=", 1)[1].strip()
    if hf:
        env["HF_TOKEN"] = hf

    boot = (HERE / "pgw1548_pod_bootstrap.sh").read_text()
    env["PGW1548_BOOT_B64"] = base64.b64encode(boot.encode()).decode()
    start = ["/bin/bash", "-lc",
             'printf %s "$PGW1548_BOOT_B64" | base64 -d > /pgw1548-boot.sh; '
             'bash /pgw1548-boot.sh; sleep infinity']

    body: dict = {
        "name": f"pgw1548-{args.mode}-{int(time.time())}",
        "imageName": IMAGE,
        "containerDiskInGb": args.disk,
        "cloudType": "SECURE",
        "supportPublicIp": False,
        "env": env,
        "dockerStartCmd": start,
    }
    if args.gpu:
        # `computeType` is stated explicitly rather than left to the default:
        # the schema's default is GPU, and a leg that relies on a default is a
        # leg that changes meaning when the provider changes one.
        body["computeType"] = "GPU"
        body["gpuTypeIds"] = [args.gpu]
        body["gpuCount"] = 1
        body["minRAMPerGPU"] = args.ram
        body["minVCPUPerGPU"] = args.vcpu
    else:
        # A CPU pod is NOT "a GPU pod with gpuCount 0" — the REST schema pins
        # `gpuCount` to minimum 1 and rejects 0 outright (measured: HTTP 400,
        # `At /pods/properties/gpuCount/minimum: got 0, want 1`). CPU sizing
        # rides `cpuFlavorIds` + `vcpuCount`, and `gpuCount` must be ABSENT.
        body["computeType"] = "CPU"
        body["cpuFlavorIds"] = list(args.cpu_flavors)
        body["vcpuCount"] = args.vcpu

    env_bytes = sum(len(k) + len(v) for k, v in env.items())
    if env_bytes > ENV_BYTES_CEILING:
        print(f"REFUSING: env is {env_bytes} bytes, over the measured "
              f"{ENV_BYTES_CEILING} ceiling. RunPod answers HTTP 500 with a "
              f"backend parse error, which reads as an outage rather than "
              f"as an oversized body — so this refuses HERE, where the "
              f"reason is legible.")
        return 1
    print(f"[plan] mode={args.mode} endpoint={args.endpoint} "
          f"gpu={args.gpu or 'CPU-only'} sha={sha[:12]} "
          f"env_bytes={sum(len(k) + len(v) for k, v in env.items())}")
    if args.dry_run:
        print("[dry-run] not renting")
        return 0

    api, _pub = podguard.creds()
    lease = podguard.rent(api, body, lane="pgw1548",
                          lease_seconds=900, orig_cmd=start)
    # BEFORE the first wait, per the coordinator's rider.
    bank(lease.pod_id, args.mode,
         {"rate_per_hr": lease.rate_per_hr, "image": IMAGE,
          "endpoint": args.endpoint, "gpu": args.gpu or "cpu"})

    expect = [s for s in args.expect.split(",") if s]
    deadline_cost = args.budget_usd
    started = time.time()
    seen: list[str] = []
    try:
        while True:
            elapsed_h = (time.time() - started) / 3600.0
            spent = elapsed_h * (lease.rate_per_hr or 0.0)
            stages = published_stages(cred)
            new = [s for s in stages if s not in seen]
            if new:
                seen = stages
                print(f"[stage] +{new}  (all: {stages})  "
                      f"${spent:.2f} / {elapsed_h * 60:.0f} min", flush=True)
            if expect and all(s in stages for s in expect):
                print(f"[done] every expected stage landed: {expect}")
                return 0
            if "done" in stages:
                print("[done] pod published its terminal stage")
                return 0
            if spent >= deadline_cost:
                print(f"[stop] budget ${deadline_cost} reached (${spent:.2f})")
                return 2
            if (time.time() - started) / 60.0 >= args.max_minutes:
                print(f"[stop] {args.max_minutes} min ceiling reached")
                return 2
            time.sleep(30)
    finally:
        print("[teardown] releasing")
        try:
            # 404 from GET /v1/pods/{id} is the ONLY teardown proof.
            dead = podguard.terminate_and_confirm(api, lease.pod_id)
            podguard.record_release(lease.pod_id, dead, reason="lane complete")
            print(f"[teardown] provider-confirmed dead: {dead}")
        except Exception as exc:  # noqa: BLE001
            print(f"[teardown] release raised {exc!r} — VERIFY MANUALLY: "
                  f"pod {lease.pod_id}")


if __name__ == "__main__":
    raise SystemExit(main())
