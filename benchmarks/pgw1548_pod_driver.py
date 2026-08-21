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
import gzip
import json
import re
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path.home() / "cozy/cozy-creator-tracker/scripts/podguard"))
import podguard  # noqa: E402
import podlive  # noqa: E402  -- the uptime corroboration the boot guard needs

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


def lock_cache_tarball(directory: str) -> str:
    """The pre-derived locks + their `.meta.json` sidecars, tar.gz+base64.

    A GPU pod deriving SDXL would burn 865 s (static) + 201 s (aspect) of
    RENTED card on `torch.export` tracing that needs no card at all. The
    pair compresses to ~15 KB, which fits under the env cliff beside the
    17 KB endpoint archive — so the derive stays paid for exactly once, on
    the box, and its REAL wall still rides in the sidecars for the ROI.
    """

    if not directory:
        return ""
    raw = subprocess.run(["tar", "czf", "-", "-C", directory, "."],
                         capture_output=True, check=True).stdout
    return base64.b64encode(raw).decode()


def pod_facing_hub() -> str:
    """The POD-FACING front door, READ FRESH from `task stacks` at launch.

    **Never from config, and that is the whole point.** `hub-credentials.json`
    held `https://cozy-e2e-4.ngrok.app` for this lane's entire life. The stack
    restarted (th#2250 deploy, ~06:00Z), the tunnel names rotated to `-6`, and
    `-4` was handed to `ltxfp8` -- a DEAD scratch stack. Four pods then
    published into a stranger's tunnel and hung. A baked URL is a fact with an
    expiry date and no expiry check; `task stacks` is the authority.
    """

    out = subprocess.run(["task", "stacks"], cwd=Path.home() / "cozy/e2e",
                         capture_output=True, text=True, timeout=240).stdout
    for line in out.splitlines():
        if "AUTHORITATIVE" not in line:
            continue
        found = re.search(r"frontdoor=bound\s+http=(\S+)", line)
        if found:
            return found.group(1).strip()
    raise SystemExit(
        "REFUSING TO RENT: `task stacks` shows no AUTHORITATIVE standing stack "
        "with a bound frontdoor. There is nowhere for a pod to publish, so a "
        "rental would buy silence.")


def assert_pod_can_publish(hub: str, cred: dict, nonce: str) -> None:
    """Round-trip the POD'S OWN URL, with the POD'S OWN token, BEFORE renting.

    🔴 THIS IS THE CHECK THAT WOULD HAVE REFUSED ALL FOUR MUTE RENTALS, in
    under two seconds, for $0.

    The campaign's evidence channel had two halves that addressed the service
    by DIFFERENT NAMES: the box polled `hub_local` (127.0.0.1) and the pod
    published to `hub_public` (an ngrok tunnel). Nothing ever asserted they
    were the same hub -- so when the tunnel went stale, the watching half
    stayed perfectly healthy while the writing half published into nothing.
    Measured discriminator, same route and same token:

        cozy-e2e-6 (correct)  -> HTTP 403   route EXISTS, auth refused
        cozy-e2e-4 (stale)    -> HTTP 404   route ABSENT: not our hub
        127.0.0.1:31550 (box) -> HTTP 403

    **Why a raw bounded GET, and not a real publish.** Two reasons, both
    measured rather than assumed:

    1. `publish_v2` cannot tell you this quickly, because it treats 404 as
       RETRYABLE: `POST .../publishes retrying (attempt 6, status=404; no
       definite answer for 33s of 600s)`. THAT is the hang -- the client spends
       ten minutes politely re-asking a server that does not have the route.
       A raw GET sees in 0.19 s exactly the 404 the retry loop hides.
    2. A publish-based preflight is self-poisoning. Tried it, and the SECOND
       launch was refused: `release_variant_equal_contract -- release "v1"
       already has a variant with this exact tensor-layout contract`. A guard
       that works once and then blocks every later rental is worse than none,
       and it litters the release with junk variants the harvester must skip.

    So: one bounded GET, no side effects, and the 403/404 split is the signal.
    403 means the route EXISTS and merely refuses this token's scope -- which
    is exactly what a correct hub says. 404 means the front door belongs to
    somebody else.
    """

    route = (f"/api/v1/repos/{cred['org']}/{cred['repo']}"
             f"/releases/{cred['release']}")
    request = urllib.request.Request(hub.rstrip("/") + route)
    request.add_header("Authorization", "Bearer " + cred["machine_token"])
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except Exception as exc:  # noqa: BLE001 — DNS, TLS, timeout, refused
        raise SystemExit(
            f"REFUSING TO RENT: the pod-facing hub {hub} did not answer in "
            f"{time.time() - started:.1f}s ({exc!r}). The pod would publish "
            f"into nothing and read as a silent death.") from exc

    took = time.time() - started
    if status == 404:
        raise SystemExit(
            f"REFUSING TO RENT: {hub} answered 404 for {route} in {took:.2f}s "
            f"-- the tunnel is UP but IT IS NOT OUR HUB (a correct hub answers "
            f"403 for this token). This is the exact condition that cost four "
            f"mute pods and $3.53: a stale ngrok name fronting a dead stack.")
    if status >= 500:
        raise SystemExit(
            f"REFUSING TO RENT: {hub} answered {status} for {route} in "
            f"{took:.2f}s -- the hub is unhealthy, so a pod would publish "
            f"into an outage.")
    print(f"[preflight] pod-facing hub {hub} answered {status} for the "
          f"release route in {took:.2f}s (403/200 = route exists = it IS our "
          f"hub) -- proven before any card is billed", flush=True)
    return
    print(f"[preflight] pod-facing hub {hub} accepted a real publish in "
          f"{time.time() - started:.1f}s -- the evidence channel is PROVEN "
          f"before any card is billed", flush=True)


def bank(pod_id: str, lane: str, extra: dict) -> None:
    """Write the id where a DIFFERENT session can find it. Before any wait."""

    PODS.parent.mkdir(parents=True, exist_ok=True)
    row = {"pod_id": pod_id, "lane": lane, "at": time.strftime("%FT%T%z"), **extra}
    with PODS.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    print(f"\n*** POD ID BANKED: {pod_id}  ({lane}) -> {PODS}")
    print(f"*** kill with: python3 {podguard.__file__} release {pod_id}\n",
          flush=True)


def relogin(cred: dict) -> str:
    """Mint a FRESH user access token from the stored password.

    The hub's user tokens live **15 minutes**. A poll loop that reads the hub
    for an hour therefore starts answering 404/401 partway through — and this
    driver's own `published_stages` treats a failed read as UNKNOWN, so the
    symptom is not an error but a pod that silently stops appearing to make
    progress until the budget cap tears it down mid-derive. Measured on the
    anima leg: the token minted at 18:35 was dead by 18:49, while the pod was
    still working.

    So the credential is REFRESHED rather than trusted, and the refresh is by
    password login (`/api/v1/password/login`) because the stored refresh_token
    does not redeem on this deployment.
    """

    body = json.dumps({"login": cred["username"],
                       "password": cred["password"]}).encode()
    req = urllib.request.Request(cred["hub_local"] + "/api/v1/password/login",
                                 data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=45) as resp:
        out = json.loads(resp.read().decode())
    cred["access_token"] = out["access_token"]
    CRED.write_text(json.dumps(cred, indent=2))
    CRED.chmod(0o600)
    return cred["access_token"]


def _get(cred: dict, path: str, *, _retried: bool = False):
    """One authenticated hub GET, refreshing the 15-minute token on 401/403."""

    req = urllib.request.Request(cred["hub_local"] + path)
    req.add_header("Authorization", "Bearer " + cred["access_token"])
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403) and not _retried:
            relogin(cred)
            return _get(cred, path, _retried=True)
        raise


def published_stages(cred: dict, pod: str) -> tuple[list[str], list[str]]:
    """Which stages THIS POD has landed. The hub is the progress signal, not a clock.

    Returns ``(my_stages, other_pods)``.

    **Scoped to one pod, and that scoping is the campaign's central fix.**
    Publishes use ``mode="merge"``, i.e. a UNION over the release, so before
    pgw#1548's post-mortem every pod wrote the same ``<stage>/<file>`` paths
    and the last writer won. Five pods died with zero measurement stages run,
    and the harvest named this as the reason the deaths were invisible: a pod
    that had never started read as HEALTHY because a *different*, concurrent
    pod's stages were sitting on the paths this poller reads. Progress for pod
    A was being inferred from evidence produced by pod B.

    Now every publish lands under ``pods/<nonce>/<stage>/<file>`` and this
    function counts ONLY its own nonce. Rows from any other pod are returned
    separately, to be reported and never counted -- a neighbour's progress is
    not this rental's progress, and the whole point is that it can no longer
    masquerade as it.

    **Read PER VARIANT, and that detail is load-bearing.** Each per-stage
    publish creates its OWN variant, so once more than one has landed:

    * `GET …/tree?release=v1` returns **404** — with several checkpoints under
      one release there is no single tree to show; and
    * `gen-worker download org/repo@release` returns **HTTP 409** for the same
      reason.

    Both were measured live against this lane's own verdicts repo. A poller
    that reads only the release route therefore goes blind the moment the pod
    publishes its SECOND stage — i.e. exactly when it starts making progress —
    and the failure looks like "no progress", which is the worst possible
    disguise: the driver would tear a healthy pod down at its budget cap.

    So the release listing supplies the variants, and each variant's tree is
    fetched by `?checkpoint_id=`, which is the form that works.
    """

    try:
        release = _get(cred, f"/api/v1/repos/{cred['org']}/{cred['repo']}"
                             f"/releases/{cred['release']}")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        # UNKNOWN, never "no progress" — treating a blip as absence is how a
        # working pod gets killed.
        print(f"[poll] release read failed ({exc}); progress UNKNOWN this tick")
        return [], []
    stages: list[str] = []
    others: list[str] = []
    variants = sorted(release.get("variants") or [],
                      key=lambda v: v.get("added_seq", 0))
    # ONLY THE NEWEST VARIANT, VIA `resolve`. Both halves are load-bearing:
    #
    # * `resolve?digest=` returns FULL paths; `tree?checkpoint_id=` returns
    #   only TOP-LEVEL names (`boot`, `bootstrap`, `pods`, ...). That
    #   difference silently blinded this driver the moment stages moved under
    #   `pods/<nonce>/...`: every path collapsed to the single entry `pods`,
    #   no stage ever matched, and a pod that had FAILED at smoke two hours
    #   earlier still read as "boot heartbeat still absent -- HOLDING".
    #   The pod-unique-path fix and the progress reader were coupled, and
    #   changing one without the other cost ~176 min of billed silence.
    # * a merge publish is a UNION, so the newest variant already resolves to
    #   the cumulative set. Walking all 48 variants bought nothing and cost 48
    #   round trips per poll.
    if variants:
        cid = variants[-1].get("checkpoint_id")
        try:
            body = _get(cred, f"/api/v1/repos/{cred['org']}/{cred['repo']}"
                              f"/resolve?digest={cid}") if cid else {}
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            print(f"[poll] resolve failed ({exc}); progress UNKNOWN this tick")
            return [], []
        for entry in body.get("files") or []:
            parts = str(entry.get("path") or "").split("/")
            # pods/<nonce>/<stage>/<file> is the only shape this lane writes.
            if len(parts) < 4 or parts[0] != "pods":
                continue
            if parts[1] != pod:
                if parts[1] not in others:
                    others.append(parts[1])
                continue
            if parts[2] and parts[2] not in stages:
                stages.append(parts[2])
    return stages, others


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("anima-derive", "sdxl-matrix", "install-probe"))
    parser.add_argument("--endpoint", required=True, help="anima | sdxl")
    parser.add_argument("--gpu", default="", help="RunPod gpuTypeId; empty = CPU pod")
    # MEASURED CEILING: RunPod refuses a container disk over the flavor's
    # limit with `HTTP 500 ... Container Disk must be less than or equal
    # to 240 / ... to 160` — a 500 again, not a 4xx, so it reads as an
    # outage rather than as "too big". CPU flavors cap lower than GPU ones.
    parser.add_argument("--disk", type=int, default=120)
    parser.add_argument("--vcpu", type=int, default=8)
    parser.add_argument("--cpu-flavors", default="cpu5m,cpu3m",
                        help="RunPod CPU flavors, priority order; the anima derive is MEMORY-bound (weight-full loads), so the memory-optimised flavors lead")
    parser.add_argument("--ram", type=int, default=64)
    parser.add_argument("--budget-usd", type=float, default=4.0)
    parser.add_argument("--max-minutes", type=float, default=180.0)
    parser.add_argument("--boot-deadline", type=float, default=10.0,
                        help="minutes to wait for the boot heartbeat before "
                             "declaring the container never started (the "
                             "heartbeat lands ~17 s in on a healthy pod)")
    parser.add_argument("--expect", default="",
                        help="comma-separated stages that mean SUCCESS")
    parser.add_argument("--lock-cache", default="",
                        help="directory of pre-derived locks to ship "
                             "to the pod (saves rented-card derive time)")
    parser.add_argument("--ref", default="",
                        help="git ref the pod checks out "
                             "(default: this lane's pushed branch)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.cpu_flavors = [f for f in args.cpu_flavors.split(",") if f]

    cred = json.loads(CRED.read_text())
    # THE BRANCH, not master. The pod runs THIS lane's harness — `--latents`,
    # `--venv`, `--lock-cache`, the ABBA order, and `pgw1548_pod_sdxl_tree.py`
    # itself — and NONE of that is on `origin/master` yet. Checking out master
    # would clone a tree whose `benchmarks/` lacks the tree-builder outright and
    # whose `dynamic_dims_pgw1548.py` refuses the flags the bootstrap passes:
    # a rented card that boots, downloads 13 GB, and then dies on an unknown
    # argument. python-gen-worker is PUBLIC, so a pushed branch needs no
    # credential.
    ref = args.ref or "origin/1548-bench-arms"
    sha = subprocess.run(["git", "rev-parse", ref],
                         cwd=Path.home() / "cozy/python-gen-worker",
                         capture_output=True, text=True, check=True).stdout.strip()

    # THE PUBLISH NAMESPACE, minted before the rental exists. It has to be
    # known box-side up front, because it is what the poller below filters on;
    # the provider's own pod id is not available until after create, and env is
    # fixed at create. A nonce also survives the case a pod is recreated under
    # a reused id.
    pod_nonce = "pod-" + secrets.token_hex(4)

    # THE URL THE POD WILL USE, resolved fresh and then PROVEN. Both halves
    # matter: reading it from `task stacks` stops a rotated tunnel being baked
    # in, and round-tripping it stops a bound-but-wrong front door. Either one
    # alone would still have bought silence.
    hub_public = pod_facing_hub()
    if hub_public != cred.get("hub_public"):
        print(f"[preflight] pod-facing hub MOVED since config was written: "
              f"{cred.get('hub_public')} -> {hub_public} (config is stale; "
              f"`task stacks` wins)")
    assert_pod_can_publish(hub_public, cred, pod_nonce)

    env = {
        "PGW1548_POD": pod_nonce,
        "PGW1548_MODE": args.mode,
        "PGW1548_HUB": hub_public,
        "PGW1548_TOKEN": cred["machine_token"],
        "PGW1548_REPO": f"{cred['org']}/{cred['repo']}",
        "PGW1548_RELEASE": cred["release"],
        "PGW1548_SHA": sha,
        "PGW1548_ENDPOINT_B64": endpoint_tarball(args.endpoint),
        "PGW1548_LOCKS_B64": lock_cache_tarball(args.lock_cache),
    }
    hf = ""
    for line in (Path.home() / "cozy/e2e/.env").read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            hf = line.split("=", 1)[1].strip()
    if hf:
        env["HF_TOKEN"] = hf

    # GZIP then base64. The bootstrap is heavily commented on purpose -- every
    # launch fact this lane paid for is written where the next person will hit
    # it -- and that pushed the plain base64 to 25 KB, which combined with the
    # endpoint archive and the lock cache tripped the env ceiling below. The
    # guard refused the launch rather than letting RunPod answer HTTP 500, so
    # nothing was rented; the fix is to stop shipping the comments UNCOMPRESSED,
    # not to delete them. Measured: 19,128 raw -> 25,504 b64 -> 9,956 gz+b64.
    boot = (HERE / "pgw1548_pod_bootstrap.sh").read_bytes()
    env["PGW1548_BOOT_GZ_B64"] = base64.b64encode(gzip.compress(boot, 9)).decode()
    start = ["/bin/bash", "-lc",
             'printf %s "$PGW1548_BOOT_GZ_B64" | base64 -d | gunzip > /pgw1548-boot.sh; '
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
        # A LIST, in preference order: a single id turns a busy
        # datacenter into a failed rental for no reason.
        body["gpuTypeIds"] = [g.strip() for g in args.gpu.split(",") if g.strip()]
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
    # DECLARE the renewal channel (pgw#1617). A raw rented pod cannot be renewed
    # over SSH at all (pgw#1568), so saying so explicitly puts the caveat in the
    # reaper's own reason string instead of leaving it implicit in a default.
    lease.renewal_channel = "none"
    lease.save()
    # BEFORE the first wait, per the coordinator's rider.
    bank(lease.pod_id, args.mode,
         {"rate_per_hr": lease.rate_per_hr, "image": IMAGE,
          "endpoint": args.endpoint, "gpu": args.gpu or "cpu",
          "pod_nonce": pod_nonce})
    print(f"*** PUBLISH NAMESPACE: pods/{pod_nonce}/  "
          f"(harvest with --pod {pod_nonce})\n", flush=True)

    expect = [s for s in args.expect.split(",") if s]
    deadline_cost = args.budget_usd
    started = time.time()
    seen: list[str] = []
    warned_others = False
    try:
        while True:
            elapsed_h = (time.time() - started) / 3600.0
            spent = elapsed_h * (lease.rate_per_hr or 0.0)
            stages, others = published_stages(cred, pod_nonce)
            if others and not warned_others:
                warned_others = True
                print(f"[poll] NOTE other pods are publishing to this release "
                      f"({others}); their rows are NOT counted as this pod's "
                      f"progress.", flush=True)
            new = [s for s in stages if s not in seen]
            if new:
                seen = stages
                print(f"[stage] +{new}  (all: {stages})  "
                      f"${spent:.2f} / {elapsed_h * 60:.0f} min", flush=True)
            # ── THE BOOT GUARD IS DELETED. Not repaired — DELETED. ──────────
            # It was rewritten three times and killed a HEALTHY pod twice:
            # f7wllw4zj348p6 (fired on silence alone while uptime climbed
            # 35->215 s) and rzahnoq94y5zlt (fired on ONE spurious uptime==0
            # from the provider, 17 min after the driver had itself printed
            # `[stage] +['telemetry1']`). Between them my guard destroyed more
            # healthy pods than every real infrastructure fault in this
            # campaign combined.
            #
            # Each rewrite fixed the previous instance and invented a new way
            # to be wrong, because the guard's PREMISE is wrong: it tried to
            # INFER liveness in order to decide a kill, when two mechanisms
            # already bound the loss without inferring anything --
            # `--budget-usd` (money is genuinely linear in time) and podguard's
            # reaper (~30 min if this driver dies). Neither needs to guess
            # whether a container is alive, and neither can be fooled by a bad
            # sample.
            #
            # So the watch loop REPORTS and never decides. Liveness is printed
            # as an observation for a human to read; nothing here returns.
            if not stages and (time.time() - started) / 60.0 >= args.boot_deadline:
                try:
                    live = podlive.live(lease.pod_id)
                    print(f"[watch] nothing published yet at "
                          f"{(time.time() - started) / 60.0:.1f} min; provider "
                          f"says {live}. REPORT ONLY -- the money cap and the "
                          f"reaper are the bounds.", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[watch] liveness probe unreadable ({exc!r}) -- "
                          f"REPORT ONLY.", flush=True)
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
