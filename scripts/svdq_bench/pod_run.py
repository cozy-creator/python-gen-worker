#!/usr/bin/env python3
"""svdq lane bench driver — ONE watched pod, cells run in sequence (pgw#865).

Promoted from the one-shot te#137/pgw#862 drivers into the repeatable
instrument pgw#865 owns, and generalised over the card so pgw#863's sm_100
row is the same instrument as the banked sm_120 one.

Cells (``--cells``, run in the order given):

  mat    materialize pod-side: Qwen-Image component tree (+ bf16 transformer
         when the bf16 arm is asked for) and the official nunchaku fp4_r128
         checkpoint. Weights never touch the control box.
  corr   bench_b0 correctness — activation-quant bit-identity vs the pgw#685
         chain on REAL captured activations, fused-vs-fp32-ref error per unit.
  bb     bench_b0 bucket bench, BASELINE lane (synthetic frame, eager +
         compiled forward, top kernels, peak VRAM).
  bf     bench_b0 bucket bench, FUSED lane.
  e:NAME e2e arm in the pipeline frame. NAME is one of
         base | native | nativec | bf16 (compiled variants suffixed c).
  met    LPIPS/PSNR over whatever e2e arms have been banked this run.

Everything is banked off-pod after each cell, so a teardown at any point
keeps the evidence. Teardown is REST DELETE + GET-404 + podguard record.

  pod_run.py --gpu b200 --cells mat,corr,bb,bf,e:bf16,e:base,e:native,e:nativec,met
  pod_run.py --list / --terminate <id> / --pod <id> (attach to a live pod)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# NOTE: do NOT drop SSH_AUTH_SOCK. The runpod key on this box is
# passphrase-protected, so `-i` alone cannot authenticate — the agent holds
# the only usable copy. Dropping it produced 90 silent "Permission denied"
# retries that read exactly like a slow boot.

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MODELS = REPO / "src" / "gen_worker" / "models"

PODGUARD_DIR = Path(os.environ.get(
    "COZY_PODGUARD_DIR",
    Path.home() / "cozy" / "cozy-creator-tracker" / "scripts" / "podguard"))
sys.path.insert(0, str(PODGUARD_DIR))
import podguard  # noqa: E402

REST = "https://rest.runpod.io/v1"
ENVF = Path(os.environ.get("COZY_BENCH_ENV",
                           Path.home() / "cozy" / "e2e" / ".env"))
KEY = Path.home() / ".ssh" / "runpod"
SSH_OPTS = ["-i", str(KEY), "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=15",
            "-o", "LogLevel=ERROR"]

# cuda-13 profile tag of the deployed conversion image: the bare -linux-x86
# tag of the same build is CPU torch.
IMAGE = ("tensorhub/endpoints-chaos:"
         "tensorhub-conversion-697fdaa0d2f40e2f75dcec10-linux-cuda-13-x86")
CARDS = {
    "b200": (["NVIDIA B200"], 220),
    "h100": (["NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe",
              "NVIDIA H100 NVL"], 220),
    "5090": (["NVIDIA GeForce RTX 5090",
              "NVIDIA RTX PRO 6000 Blackwell Server Edition"], 220),
}
# gen-worker source overlaid onto the image's installed package: the image
# pins a published wheel, the lane under test is this worktree.
OVERLAY = ("svdq_fused.py", "svdq_awq_packed.py", "native_kernels.py",
           "svdq_native.py", "svdq_layout.py", "svdq_awq.py", "svdq.py",
           "nvfp4_quant.py", "w4a4.py")

SSHD = (
    'set -e; mkdir -p /root/.ssh && chmod 700 /root/.ssh; '
    'printf "%s\\n" "$PUBLIC_KEY" >> /root/.ssh/authorized_keys && '
    'chmod 600 /root/.ssh/authorized_keys; '
    'if [ ! -x /usr/sbin/sshd ] && ! command -v sshd >/dev/null 2>&1; then '
    'apt-get update -qq && DEBIAN_FRONTEND=noninteractive '
    'apt-get install -y -qq openssh-server; fi; '
    'ssh-keygen -A 2>/dev/null || true; '
    'mkdir -p /run/sshd; exec /usr/sbin/sshd -D -e')

# e2e arms: label -> (native lane env, extra bench_arm flags)
ARMS = {
    "bf16": ("0", "--arm bf16"),
    "bf16c": ("0", "--arm bf16 --compile"),
    "fp8": ("0", "--arm fp8 --fp8-tree /root/art/fp8"),
    "fp8c": ("0", "--arm fp8 --fp8-tree /root/art/fp8 --compile"),
    "base": ("0", "--arm ours"),
    "basec": ("0", "--arm ours --compile"),
    "native": ("1", "--arm ours"),
    "nativec": ("1", "--arm ours --compile"),
}
# Which arms need which materialized artifact.
NEEDS_BF16_TREE = ("bf16", "bf16c")
NEEDS_FP8_TREE = ("fp8", "fp8c")
NEEDS_SVDQ_CK = ("base", "basec", "native", "nativec")

HUB_BASE = os.environ.get("COZY_HUB_BASE", "http://localhost:30070")
FP8_REF = ("tensorhub", "qwen-image", "latest", "fp8-w8a8")


def dotenv(names):
    out = {}
    for line in ENVF.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() in names:
                out[k.strip()] = v.strip().strip('"').strip("'")
    for n in names:
        if not out.get(n) and os.environ.get(n):
            out[n] = os.environ[n]
    return out


def rest(api, method, path, body=None):
    if method == "POST" and path.rstrip("/") == "/pods":
        podguard.assert_armed(body)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{REST}{path}", data=data, method=method,
        headers={"Authorization": f"Bearer {api}",
                 "Content-Type": "application/json",
                 "User-Agent": "curl/8.5.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as fh:
            raw = fh.read()
        return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        sys.stderr.write(
            f"HTTP {exc.code} {method} {path}: {exc.read()[:600]!r}\n")
        raise


def sh(cmd, timeout=3600):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        parts = (exc.stdout or b"", exc.stderr or b"")
        txt = "".join(o.decode(errors="replace") if isinstance(o, bytes) else o
                      for o in parts)
        return 124, txt + f"\n[sh] timeout after {timeout}s"
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def ssh(ip, port, script, timeout=3600, env=None):
    pre = "".join(f"export {k}={json.dumps(v)}; "
                  for k, v in (env or {}).items())
    return sh(["ssh", *SSH_OPTS, "-p", str(port), f"root@{ip}", pre + script],
              timeout)


def ssh_endpoint(api, pod_id):
    d = rest(api, "GET", f"/pods/{pod_id}")
    ip = d.get("publicIp")
    port = (d.get("portMappings") or {}).get("22")
    if ip and port:
        return ip, int(port), float(d.get("costPerHr") or 0.0)
    return None


def confirm_404(api, pod_id):
    try:
        rest(api, "GET", f"/pods/{pod_id}")
        return False
    except urllib.error.HTTPError as exc:
        return exc.code == 404


def registry_auth_id(api):
    import base64
    creds = dotenv(("DOCKERHUB_USERNAME", "DOCKERHUB_TOKEN"))
    user = creds.get("DOCKERHUB_USERNAME", "")
    pw = creds.get("DOCKERHUB_TOKEN", "")
    if not (user and pw):
        d = json.load(open(Path.home() / ".docker" / "config.json"))
        a = d["auths"]["https://index.docker.io/v1/"]["auth"]
        user, pw = base64.b64decode(a).decode().split(":", 1)
    r = rest(api, "POST", "/containerregistryauth",
             {"name": f"svdqbench-{int(time.time())}",
              "username": user, "password": pw})
    print(f"[auth] registry auth id={r.get('id')}", flush=True)
    return r["id"]


def wait_for(ip, port, log, marker, deadline_s, tick=45, label=""):
    """Poll a pod-side log for DONE/Traceback. Returns (ok, tail)."""
    end = time.time() + deadline_s
    while time.time() < end:
        time.sleep(tick)
        _rc, out = ssh(ip, port,
                       f"grep -hE '{marker}|Traceback' {log} | tail -1", 60)
        if "Traceback" in out:
            _rc, tail = ssh(ip, port, f"tail -30 {log}", 60)
            return False, tail
        if marker.split("|")[0] in out:
            _rc, tail = ssh(ip, port, f"tail -30 {log}", 60)
            return True, tail
        _rc, prog = ssh(ip, port, f"tail -1 {log}", 60)
        print(f"[{label}] {prog.strip()[:160]}", flush=True)
    _rc, tail = ssh(ip, port, f"tail -30 {log}", 60)
    return False, tail + "\n[driver] TIMEOUT"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="b200", choices=sorted(CARDS))
    ap.add_argument("--cells", default="mat,corr,bb,bf,e:bf16,e:base,"
                                       "e:native,e:nativec,met")
    ap.add_argument("--rows", default="",
                    help="row-id subset for the e2e cells")
    ap.add_argument("--norm-rows", type=int, default=3)
    ap.add_argument("--norm-shape", default="1024x1024x30x4.0")
    ap.add_argument("--wall-min", type=float, default=300.0)
    ap.add_argument("--pod", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--terminate", default="")
    ap.add_argument("--keep", action="store_true",
                    help="skip teardown (a follow-on cell run will attach)")
    args = ap.parse_args()

    api = dotenv(("RUNPOD_API_KEY",))["RUNPOD_API_KEY"]
    if args.list:
        pods = rest(api, "GET", "/pods")
        for p in (pods if isinstance(pods, list)
                  else pods.get("data", [])) or []:
            print(p.get("id"), p.get("name"), p.get("desiredStatus"),
                  (p.get("machine") or {}).get("gpuTypeId"), p.get("costPerHr"))
        return 0
    if args.terminate:
        rest(api, "DELETE", f"/pods/{args.terminate}")
        print("terminated", args.terminate,
              "404=", confirm_404(api, args.terminate))
        return 0

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    e2e_arms = [c[2:] for c in cells if c.startswith("e:")]
    for a in e2e_arms:
        assert a in ARMS, f"unknown arm {a}; known: {sorted(ARMS)}"
    want_bf16 = any(a in NEEDS_BF16_TREE for a in e2e_arms)
    want_fp8 = any(a in NEEDS_FP8_TREE for a in e2e_arms)

    hf = dotenv(("HF_TOKEN",)).get("HF_TOKEN", "")
    started = time.time()
    res = Path(args.out) if args.out else (
        HERE / f"results-{args.gpu}-{int(started)}")
    res.mkdir(parents=True, exist_ok=True)
    print(f"[run] banking to {res}", flush=True)

    gpu_ids, disk = CARDS[args.gpu]
    pod_id = args.pod
    if not pod_id:
        body = {"name": f"svdq-bench-{args.gpu}-{int(started)}",
                "imageName": IMAGE, "gpuTypeIds": gpu_ids, "gpuCount": 1,
                "containerDiskInGb": disk, "cloudType": "SECURE",
                "supportPublicIp": True, "ports": ["22/tcp"],
                # cu130 image: a 12.8-driver host raises "driver too old"
                "allowedCudaVersions": ["13.0"],
                "containerRegistryAuthId": registry_auth_id(api),
                "env": {"PUBLIC_KEY": KEY.with_suffix(".pub")
                        .read_text().strip()}}

        def entrypoint_post(armed):
            # RunPod treats dockerEntrypoint=[] as unset; ship the armed
            # command AS the entrypoint so sshd (not the worker) runs.
            armed = dict(armed)
            sc = armed.pop("dockerStartCmd", None)
            if sc:
                armed["dockerEntrypoint"] = sc
            return rest(api, "POST", "/pods", armed)

        lease = podguard.rent(api, body, lane=f"svdq-bench-{args.gpu}",
                              lease_seconds=args.wall_min * 60,
                              orig_cmd=["/bin/bash", "-lc", SSHD],
                              post=entrypoint_post)
        pod_id = lease.pod_id
        print(f"[pod] created {pod_id} rate=${lease.rate_per_hr}/hr",
              flush=True)
    else:
        podguard.attend(api, pod_id, lane=f"svdq-bench-{args.gpu}")
        print(f"[pod] attached {pod_id}", flush=True)
    (res / "pod.txt").write_text(f"{pod_id} {int(started)} {args.gpu}\n")

    manifest = {"pod": pod_id, "gpu": args.gpu, "image": IMAGE,
                "cells": cells, "started": started, "results": {}}

    def bank():
        (res / "manifest.json").write_text(json.dumps(manifest, indent=1))

    try:
        ep = None
        while time.time() - started < 1800:
            ep = ssh_endpoint(api, pod_id)
            if ep:
                break
            time.sleep(10)
        if not ep:
            print("[pod] no ssh endpoint in 30min", flush=True)
            return 3
        ip, port, rate = ep
        manifest["rate_per_hr"] = rate
        print(f"[pod] ssh root@{ip}:{port} rate=${rate}/hr", flush=True)
        last = ""
        for attempt in range(60):
            rc, last = sh(["ssh", *SSH_OPTS, "-p", str(port), f"root@{ip}",
                           "true"], 30)
            if rc == 0:
                break
            # Print the reason, every time. A retry loop that hides ssh's
            # stderr cannot distinguish "still booting" from "wrong key".
            if attempt % 5 == 0:
                print(f"[pod] ssh not ready ({attempt}): "
                      f"{last.strip()[-160:]}", flush=True)
            time.sleep(10)
        else:
            print(f"[pod] ssh never came up: {last.strip()[-300:]}",
                  flush=True)
            return 3

        files = [str(HERE / f) for f in
                 ("bench_b0.py", "bench_arm.py", "materialize_bench.py",
                  "metrics_pod.py", "bench_prompts.json")]
        files += [str(MODELS / f) for f in OVERLAY]
        rc, out = sh(["scp", *SSH_OPTS, "-P", str(port), *files,
                      f"root@{ip}:/root/"], 300)
        print(f"[scp] rc={rc}", flush=True)
        if rc != 0:
            print(out[-1500:], flush=True)
            return 4
        rc, out = ssh(
            ip, port,
            'M=$(python3 -c "import gen_worker.models, pathlib; '
            'print(pathlib.Path(gen_worker.models.__file__).parent)"); '
            + "cp " + " ".join(f"/root/{f}" for f in OVERLAY)
            + ' "$M/" && echo overlay-ok', 180)
        print(f"[overlay] rc={rc} {out.strip()[-300:]}", flush=True)
        if rc != 0 or "overlay-ok" not in out:
            return 4
        # Environment census is EVIDENCE, never a gate: a probe that cannot
        # read a version must not be able to delete a rented pod.
        _rc, env_out = ssh(
            ip, port,
            'python3 -c "import torch, importlib.metadata as im; '
            'print(\'ENV\', torch.__version__, '
            'torch.cuda.get_device_name(0), '
            'torch.cuda.get_device_capability(), '
            'im.version(\'gen-worker\'))" 2>&1 | tail -3', 180)
        print(f"[env] {env_out.strip()[-300:]}", flush=True)
        manifest["env_line"] = env_out.strip().splitlines()[-3:]
        bank()

        ck = refs_tree = ""

        dropped: set = set()

        for cell in cells:
            if cell in dropped:
                print(f"[skip] {cell} — dropped upstream", flush=True)
                continue
            print(f"\n===== CELL {cell} "
                  f"(t+{(time.time() - started) / 60:.0f}min) =====",
                  flush=True)
            try:

                if cell == "mat":
                    flag = "--bf16" if want_bf16 else ""
                    if want_fp8:
                        org, name, tag, flavor = FP8_REF
                        url = (f"{HUB_BASE}/api/v1/repos/{org}/{name}/resolve"
                               f"?tag={tag}&flavor={flavor}")
                        with urllib.request.urlopen(url, timeout=120) as fh:
                            man = json.loads(fh.read())
                        manifest["fp8_checkpoint_id"] = man["checkpoint_id"]
                        manifest["fp8_size_bytes"] = man["size_bytes"]
                        mf = res / "fp8_manifest.json"
                        mf.write_text(json.dumps(man))
                        print(f"[mat] fp8 flavor {man['checkpoint_id']} "
                              f"{man['size_bytes'] / 1e9:.1f} GB resolved",
                              flush=True)
                        rc, _o = sh(["scp", *SSH_OPTS, "-P", str(port), str(mf),
                                     f"root@{ip}:/root/fp8_manifest.json"], 120)
                        assert rc == 0, "fp8 manifest scp failed"
                        flag += " --fp8"
                    ssh(ip, port,
                        f"setsid nohup python3 /root/materialize_bench.py {flag} "
                        f"> /root/mat.log 2>&1 & echo started", 60,
                        env={"HF_TOKEN": hf})
                    last, stalls = -1, 0
                    deadline = time.time() + 90 * 60
                    while time.time() < deadline:
                        time.sleep(60)
                        _rc, out = ssh(
                            ip, port,
                            "grep -hE 'NUN_FILE=|REFS_TREE=' /root/mat.log "
                            "2>/dev/null; du -sb /root/art 2>/dev/null | cut -f1",
                            60)
                        size = 0
                        for line in out.splitlines():
                            if line.startswith("NUN_FILE="):
                                ck = line.split("=", 1)[1].strip()
                            elif line.startswith("REFS_TREE="):
                                refs_tree = line.split("=", 1)[1].strip()
                            elif line.strip().isdigit():
                                size = int(line.strip())
                        if ck and refs_tree:
                            break
                        stalls = stalls + 1 if size == last else 0
                        last = size
                        print(f"[mat] {size / 2**30:.1f} GiB (stalls={stalls})",
                              flush=True)
                        if stalls >= 10:
                            print("[mat] stalled 10min — abort", flush=True)
                            break
                    print(f"[mat] ck={ck!r} refs={refs_tree!r}", flush=True)
                    if not (ck and refs_tree):
                        return 5
                    if want_fp8:
                        _rc, out = ssh(ip, port,
                                       "ls /root/art/fp8/transformer/"
                                       "*.safetensors 2>/dev/null | wc -l; "
                                       "du -sh /root/art/fp8 2>/dev/null", 60)
                        print(f"[mat] fp8 tree: {out.strip()}", flush=True)
                        # An OPTIONAL artifact that did not land drops its OWN
                        # arms. It does not fail the run, and it must never tear
                        # down a pod already carrying 70 GB of other artifacts.
                        if out.strip().split()[:1] == ["0"]:
                            dropped.update(f"e:{a}" for a in NEEDS_FP8_TREE)
                            print("[mat] fp8 tree absent — fp8 arms dropped; "
                                  "every other arm stands", flush=True)
                    manifest["ckpt"], manifest["refs_tree"] = ck, refs_tree
                    bank()
                    continue

                if not (ck and refs_tree):
                    raise RuntimeError("mat must run before any other cell")

                if cell in ("corr", "bb", "bf"):
                    mode = "correctness" if cell == "corr" else "bench"
                    env_val = "1" if cell in ("corr", "bf") else "0"
                    ssh(ip, port,
                        f"setsid nohup env GEN_WORKER_NATIVE_KERNELS={env_val} "
                        f"python3 /root/bench_b0.py --ckpt {ck} "
                        f"--out /root/b0_{cell} --mode {mode} "
                        f"> /root/{cell}.log 2>&1 & echo started", 60)
                    ok, tail = wait_for(ip, port, f"/root/{cell}.log",
                                        "DONE", 45 * 60, 45, cell)
                    (res / f"{cell}_log.txt").write_text(tail)
                    sh(["scp", *SSH_OPTS, "-P", str(port), "-r",
                        f"root@{ip}:/root/b0_{cell}", str(res) + "/"], 600)
                    print(f"[{cell}] ok={ok}\n{tail[-1200:]}", flush=True)
                    manifest["results"][cell] = "ok" if ok else "FAILED"
                    bank()
                    continue

                if cell.startswith("e:"):
                    name = cell[2:]
                    env_val, extra = ARMS[name]
                    rows = f"--rows {args.rows}" if args.rows else ""
                    rows += (f" --norm-rows {args.norm_rows} "
                             f"--norm-shape {args.norm_shape}")
                    ssh(ip, port,
                        f"setsid nohup env GEN_WORKER_NATIVE_KERNELS={env_val} "
                        f"python3 /root/bench_arm.py {extra} --ckpt {ck} "
                        f"--reference-tree {refs_tree} "
                        f"--prompts /root/bench_prompts.json {rows} "
                        f"--out /root/e2e/{name} > /root/arm_{name}.log 2>&1 & "
                        f"echo started", 60)
                    ok, tail = wait_for(ip, port, f"/root/arm_{name}.log",
                                        "DONE", 100 * 60, 90, f"arm {name}")
                    (res / f"arm_{name}_log.txt").write_text(tail)
                    sh(["scp", *SSH_OPTS, "-P", str(port), "-r",
                        f"root@{ip}:/root/e2e/{name}", str(res) + "/"], 900)
                    print(f"[arm {name}] ok={ok}\n{tail[-1200:]}", flush=True)
                    manifest["results"][cell] = "ok" if ok else "FAILED"
                    bank()
                    continue

                if cell == "met":
                    done = [a for a in e2e_arms
                            if manifest["results"].get(f"e:{a}") == "ok"]
                    if "bf16" not in done or len(done) < 2:
                        print("[met] need bf16 + >=1 candidate arm — skipped",
                              flush=True)
                        continue
                    cand = " ".join(f"--cand {a}=/root/e2e/{a}"
                                    for a in done if a != "bf16")
                    ssh(ip, port,
                        "python3 -c 'import torchmetrics' 2>/dev/null || "
                        "pip install -q torchmetrics lpips 2>&1 | tail -2", 900)
                    rc, out = ssh(
                        ip, port,
                        f"python3 /root/metrics_pod.py --ref /root/e2e/bf16 "
                        f"{cand} --out /root/metrics.json 2>&1 | tail -60", 1800)
                    (res / "metrics_log.txt").write_text(out)
                    sh(["scp", *SSH_OPTS, "-P", str(port),
                        f"root@{ip}:/root/metrics.json", str(res) + "/"], 300)
                    print(f"[met] rc={rc}\n{out[-2000:]}", flush=True)
                    manifest["results"]["met"] = "ok" if rc == 0 else "FAILED"
                    bank()
                    continue

                raise RuntimeError(f"unknown cell {cell}")

            except Exception as exc:  # noqa: BLE001
                # One cell's failure is one row missing from the table, not a
                # lost pod. Only `mat` is load-bearing for everything after.
                print(f"[{cell}] RAISED {type(exc).__name__}: {exc}",
                      flush=True)
                manifest["results"][cell] = f"RAISED {type(exc).__name__}"
                bank()
                if cell == "mat":
                    return 5
                continue
        return 0
    finally:
        manifest["elapsed_min"] = (time.time() - started) / 60
        manifest["est_cost_usd"] = (manifest.get("rate_per_hr", 0.0)
                                    * manifest["elapsed_min"] / 60)
        bank()
        if args.keep:
            print(f"[pod] KEPT {pod_id} — attach with --pod {pod_id}",
                  flush=True)
            return 0
        try:
            rest(api, "DELETE", f"/pods/{pod_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"[pod] DELETE failed ({exc}) — VERIFY {pod_id}", flush=True)
        gone = False
        for _ in range(30):
            gone = confirm_404(api, pod_id)
            if gone:
                break
            time.sleep(10)
        podguard.record_release(pod_id, gone, "svdq-bench teardown")
        manifest["teardown_404"] = gone
        bank()
        print(f"[pod] terminated {pod_id}; 404={gone}; "
              f"elapsed={manifest['elapsed_min']:.1f}min "
              f"est=${manifest['est_cost_usd']:.2f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
