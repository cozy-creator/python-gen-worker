"""pgw#1607 phase 4: the box-side pod driver.

Rents ONE RunPod GPU pod through podguard (renting and arming teardown are
one act), bootstraps the fleet line + this branch + the post-va#12 varena
wheel, runs `checkpoint_juggle_pod_pgw1607.py`, ships the verdicts OFF the
pod before teardown, and terminates with provider-404 proof. One pod per
invocation; the matrix is sequential invocations, each gated on the last.

    nice -n 19 .venv/bin/python benchmarks/checkpoint_juggle_pod_driver_pgw1607.py \\
        --gpu "NVIDIA GeForce RTX 4090" --disk-gb 120 --repos 6 --arms SWCZ

Discipline (the va#5/#1548 shape, reused not reinvented): the pod id is
BANKED to the ledger file the moment it exists, before any wait loop; every
exit path runs terminate_and_confirm; results are scp'd out BEFORE teardown;
the driver refuses to start if a previous pod from this ledger is still
alive (one heavy thing at a time).
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

TRACKER_SCRIPTS = Path.home() / "cozy" / "cozy-creator-tracker" / "scripts" / "podguard"
sys.path.insert(0, str(TRACKER_SCRIPTS))

import podguard  # noqa: E402

LEDGER = Path.home() / ".cache" / "cozy" / "pgw1607" / "pods.jsonl"
OUT_ROOT = Path.home() / ".cache" / "cozy" / "pgw1607"
WHEEL = (Path.home() / "cozy" / "varena" / "target" / "wheels" /
         "varena-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl")
FLOORS = Path.home() / "cozy" / "serverless-endpoints" / "fleet-floors.toml"
BRANCH = "1607-checkpoint-juggle"
REPO_URL = "https://github.com/cozy-creator/python-gen-worker"

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=20", "-o", "LogLevel=ERROR",
    "-i", str(Path.home() / ".ssh" / "id_ed25519"),
]

BOOTSTRAP = r"""
set -euo pipefail
cd /workspace
if [ ! -d pgw ]; then
  git clone --depth 1 -b {branch} {repo} pgw
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)
if [ ! -d venv ]; then
  uv venv --python 3.11 venv
fi
. venv/bin/activate
python -c "import torch" 2>/dev/null || \
  uv pip install --python venv/bin/python torch==2.13.0 \
    --index-url https://download.pytorch.org/whl/cu130
uv pip install --python venv/bin/python \
  diffusers transformers accelerate huggingface_hub safetensors numpy msgspec
uv pip install --python venv/bin/python --force-reinstall --no-deps /workspace/{wheel}
python - <<'PY'
import varena
assert hasattr(varena.Reservation, "page_signatures"), "wheel predates va#12"
print("varena wheel OK (post-va#12)")
PY
"""


def log(msg: str) -> None:
    print(f"[driver {time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def bank_pod(pod_id: str, note: str, rate: float) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as fh:
        fh.write(json.dumps({
            "pod_id": pod_id, "note": note, "rate_per_hr": rate,
            "at": time.strftime("%FT%TZ", time.gmtime()),
            "kill": f"python3 {TRACKER_SCRIPTS}/podguard.py release {pod_id}",
        }) + "\n")
    log(f"BANKED pod {pod_id} rate=${rate}/hr -> {LEDGER}")


def any_prior_alive(api: str) -> str | None:
    if not LEDGER.exists():
        return None
    for line in LEDGER.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("released"):
            continue
        pod_id = rec["pod_id"]
        try:
            podguard.rest(api, "GET", f"/pods/{pod_id}")
            return pod_id
        except RuntimeError as exc:
            if "404" not in str(exc):
                return pod_id
    return None


def mark_released(pod_id: str) -> None:
    with LEDGER.open("a") as fh:
        fh.write(json.dumps({
            "pod_id": pod_id, "released": True,
            "at": time.strftime("%FT%TZ", time.gmtime()),
        }) + "\n")


def ssh_target(api: str, lease) -> tuple[str, int] | None:
    return lease._ssh_target(api)


def pod_sh(tgt: tuple[str, int], cmd: str, timeout: int = 1800) -> tuple[int, str]:
    return podguard._sh(
        ["ssh", *SSH_OPTS, "-p", str(tgt[1]), f"root@{tgt[0]}", cmd], timeout
    )


def scp_to(tgt: tuple[str, int], src: Path, dst: str) -> None:
    rc, out = podguard._sh(
        ["scp", *SSH_OPTS, "-P", str(tgt[1]), str(src), f"root@{tgt[0]}:{dst}"], 300
    )
    if rc != 0:
        raise RuntimeError(f"scp {src} failed: {out[-300:]}")


def scp_from(tgt: tuple[str, int], src: str, dst: Path) -> bool:
    dst.mkdir(parents=True, exist_ok=True)
    rc, out = podguard._sh(
        ["scp", *SSH_OPTS, "-P", str(tgt[1]), "-r", f"root@{tgt[0]}:{src}", str(dst)],
        600,
    )
    if rc != 0:
        log(f"scp-out failed: {out[-300:]}")
    return rc == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    parser.add_argument("--disk-gb", type=int, default=120)
    parser.add_argument("--cloud", default="COMMUNITY")
    parser.add_argument("--repos", type=int, default=6)
    parser.add_argument("--arms", default="SWCZ")
    parser.add_argument("--budget-gib", type=float, default=6.0)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--max-hours", type=float, default=2.0)
    parser.add_argument("--name", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    assert WHEEL.exists(), f"wheel missing: {WHEEL}"
    assert FLOORS.exists(), f"fleet floors missing: {FLOORS}"
    api, pub = podguard.creds()

    prior = any_prior_alive(api)
    if prior:
        raise SystemExit(
            f"a prior pgw1607 pod ({prior}) is still alive; release it first — "
            f"one pod at a time"
        )

    name = args.name or f"pgw1607-{args.gpu.split()[-1].lower()}-{int(time.time()) % 100000}"
    body = {
        "name": name,
        "imageName": "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
        "gpuTypeIds": [args.gpu],
        "gpuCount": 1,
        "containerDiskInGb": args.disk_gb,
        "volumeInGb": 0,
        "cloudType": args.cloud,
        "supportPublicIp": True,
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": pub},
    }
    if args.dry_run:
        print(json.dumps(body, indent=2))
        return 0

    log(f"renting {args.gpu} ({args.cloud}, {args.disk_gb} GB disk) as {name}")
    lease = podguard.rent(api, body, lane="pgw1607-juggle",
                          lease_seconds=900.0)
    pod_id = lease.pod_id
    rate = float(lease.rate_per_hr or 0.0)
    bank_pod(pod_id, f"{name} {args.gpu} arms={args.arms} repos={args.repos}", rate)

    out_dir = OUT_ROOT / f"pod-{pod_id}"
    started = time.time()
    ok = False
    try:
        log("waiting for SSH ...")
        tgt = None
        deadline = time.time() + 900
        while time.time() < deadline:
            tgt = ssh_target(api, lease)
            if tgt and pod_sh(tgt, "true", 45)[0] == 0:
                break
            tgt = None
            time.sleep(15)
        if tgt is None:
            raise RuntimeError("SSH never came up inside 15 min")
        log(f"ssh up at {tgt[0]}:{tgt[1]}")

        scp_to(tgt, WHEEL, f"/workspace/{WHEEL.name}")
        scp_to(tgt, FLOORS, "/workspace/fleet-floors.toml")
        rc, out = pod_sh(
            tgt, BOOTSTRAP.format(branch=BRANCH, repo=REPO_URL, wheel=WHEEL.name),
            timeout=1800,
        )
        log(f"bootstrap rc={rc}; tail: {out[-500:]}")
        if rc != 0:
            raise RuntimeError("bootstrap failed")

        # EVIDENCE EGRESS IS PART OF THE SMOKE GATE (coordinator rider,
        # 2026-08-20; pgw#1568 measured SSH egress dead on raw pods of a
        # different shape). A full round trip with content verification runs
        # BEFORE any download or arm — if this fails, abort at ~$0.05 and
        # wire the publishes channel rather than losing verdicts at the end.
        canary = f"pgw1607-egress-{pod_id}-{int(time.time())}"
        rc, _ = pod_sh(tgt, f"mkdir -p /workspace/pgw1607-out && "
                            f"echo -n {shlex.quote(canary)} "
                            f"> /workspace/pgw1607-out/egress-canary.txt", 60)
        if rc != 0 or not scp_from(tgt, "/workspace/pgw1607-out", out_dir):
            raise RuntimeError("EGRESS UNPROVEN: canary round trip failed — "
                               "no arm runs without a proven evidence path")
        got_txt = (out_dir / "pgw1607-out" / "egress-canary.txt")
        if not got_txt.exists() or got_txt.read_text() != canary:
            raise RuntimeError("EGRESS UNPROVEN: canary content mismatch")
        log("evidence egress PROVEN by round trip (canary content verified)")

        harness = (
            "cd /workspace && . venv/bin/activate && "
            "export GEN_WORKER_FLEET_LINE_FILE=/workspace/fleet-floors.toml && "
            "export PYTHONPATH=/workspace/pgw/src && "
            f"python pgw/benchmarks/checkpoint_juggle_pod_pgw1607.py "
            f"--arms {shlex.quote(args.arms)} --repos {args.repos} "
            f"--budget-gib {args.budget_gib} --steps {args.steps} "
            f"--requests {args.requests}"
        )
        budget_s = int(args.max_hours * 3600)
        log(f"running harness (wall cap {budget_s}s) ...")
        rc, out = pod_sh(tgt, f"timeout {budget_s} bash -lc {shlex.quote(harness)}",
                         timeout=budget_s + 300)
        log(f"harness rc={rc}")
        print(out[-4000:])
        got = scp_from(tgt, "/workspace/pgw1607-out", out_dir)
        ok = rc == 0 and got
    finally:
        elapsed_h = (time.time() - started) / 3600
        log(f"terminating {pod_id} (elapsed {elapsed_h:.2f} h, "
            f"~${rate * elapsed_h:.2f}) ...")
        dead = podguard.terminate_and_confirm(api, pod_id)
        podguard.record_release(pod_id, dead, note="pgw1607 driver finally")
        mark_released(pod_id)
        log(f"provider-404 confirmed: {dead}")
        if not dead:
            log(f"!! POD MAY BE ALIVE — run: python3 {TRACKER_SCRIPTS}/podguard.py "
                f"release {pod_id}")
    log(f"verdicts: {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
