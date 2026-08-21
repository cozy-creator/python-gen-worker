#!/usr/bin/env bash
# pgw#1548 pod-side bootstrap — BLIND BY DESIGN.
#
# pgw#1568 proved SSH does not reach rented pods (two rentals, two zero-byte
# captures, no RunPod logs API). So this script is the whole leg: it runs from
# `dockerStartCmd`, and the ONLY way anything it learns reaches the box is a
# per-stage publish to the hub. Every stage therefore publishes, with
# mode="merge", so a pod that dies at stage N leaves stages 1..N-1 behind.
#
# Env contract (all injected at pod-create, none baked into an image):
#   PGW1548_MODE          anima-derive | sdxl-matrix
#   PGW1548_HUB           public frontdoor URL
#   PGW1548_TOKEN         machine token, org:repo:write ONLY
#   PGW1548_REPO          org/name of the verdicts repo
#   PGW1548_RELEASE       release the verdicts attach to (MUST pre-exist)
#   PGW1548_ENDPOINT_B64  the MINIMAL endpoint source (src + the two tomls),
#                         tar.gz+base64 -- 27 KB, under the measured env cliff
#   PGW1548_SHA           pgw commit to check out (public repo, no auth)
#   HF_TOKEN              optional, for the SDXL snapshot download
set -u
exec > /workspace/pgw1548-boot.log 2>&1
set -x
export DEBIAN_FRONTEND=noninteractive
export PATH="$HOME/.local/bin:$PATH"
mkdir -p /workspace/out
cd /workspace

# The pod's own publish namespace. The driver mints this and injects it, so it
# is known box-side BEFORE the rental exists; the fallback exists only so a
# hand-run bootstrap still publishes somewhere legible rather than colliding
# on a shared path.
export PGW1548_POD="${PGW1548_POD:-nonce-missing-${RUNPOD_POD_ID:-$(hostname)}}"

STAGE_RC=0

# --- the evidence channel, defined before anything can fail ------------------
# A plain python function rather than a CLI: it reuses HubClient.publish_v2,
# which is what pgw#1568 ruled (reuse, not a second publish path).
cat > /workspace/publish.py <<'PYEOF'
import json, os, socket, sys, threading
from pathlib import Path
sys.path.insert(0, "/workspace/pgw/src")
from gen_worker.hubio.client import HubClient, CommitFile

# 🔴 A PUBLISH THAT CAN BLOCK FOREVER IS NOT AN EVIDENCE CHANNEL.
# Against a stale ngrok tunnel `publish_v2` did NOT fail -- it HUNG (measured:
# still running past 120 s). The bootstrap's note/fail_out calls are not
# wrapped in `timeout`, so the FIRST publish a pod attempted blocked forever
# and four pods died mute with the container still alive. A bare
# `except Exception` around an un-bounded request converts a transport outage
# into silence indistinguishable from a dead pod.
#
# Two bounds, because they catch different things: the socket default stops a
# hung read, and the watchdog stops anything that gets past it (a retry loop,
# a stall between chunks). The watchdog is a daemon thread that hard-exits the
# PROCESS -- not the leg -- so the stage moves on and the pod keeps working.
socket.setdefaulttimeout(120)
threading.Timer(300, lambda: os._exit(75)).start()

stage = sys.argv[1]
paths = [Path(p) for p in sys.argv[2:] if Path(p).exists()]
if not paths:
    print(f"[publish] {stage}: nothing to publish"); raise SystemExit(0)
c = HubClient(base_url=os.environ["PGW1548_HUB"], token=os.environ["PGW1548_TOKEN"])
# POD-UNIQUE PATH. This segment is the fix the failed campaign's harvest named.
# `mode="merge"` makes a publish a UNION over the whole release, so a bare
# `<stage>/<file>` path is LAST-WRITER-WINS ACROSS PODS: two pods alive at once
# silently overwrite each other's verdicts, and the campaign's fatal symptom
# was exactly that -- a DEAD pod read as healthy because a CONCURRENT pod's
# stages had landed on the paths the driver was polling. The segment is a
# driver-minted launch nonce (`PGW1548_POD`), not RUNPOD_POD_ID: the driver
# must know the value BEFORE the pod exists in order to poll only its own
# rows, and env is fixed at create time whereas the provider id is not.
pod = os.environ.get("PGW1548_POD") or "unknown-pod"
files = [CommitFile(path=f"pods/{pod}/{stage}/{p.name}", local_path=p) for p in paths]
try:
    res = c.publish_v2(
        destination_repo=os.environ["PGW1548_REPO"],
        files=files,
        release=os.environ["PGW1548_RELEASE"],
        mode="merge",      # UNION, so earlier stages survive this one
    )
    print(f"[publish] {stage}: ok {res}")
except Exception as exc:
    # A failed publish must never kill the leg: the measurement is the point,
    # and a later stage may still get through.
    print(f"[publish] {stage}: FAILED {exc!r}")
PYEOF

# --- THE TELEMETRY SIDECAR (pgw#1548 install probe) --------------------------
# Publishes `free -m`, `df -h` and `dmesg | tail` every 60 s, EACH AS ITS OWN
# PUBLISH, so a pod that dies mid-series still leaves every earlier sample.
#
# Why it exists: three GPU pods started and then went silent inside the torch
# install, and the `timeout ... || fail_out` guard NEVER SPOKE. That asymmetry
# is the clue -- a timeout kills with SIGTERM, whose handler runs, but the
# kernel OOM-killer sends SIGKILL, which no trap can catch. (Same mechanism
# this lane used deliberately to hand a pod between attendants without firing
# the outgoing process's `finally`.) So the `dmesg` tail is the point: it
# catches the oom-kill line directly, from outside the dying process.
telemetry() {
  local n=0
  while :; do
    n=$((n+1))
    {
      echo "=== sample $n  $(date -Is) ==="
      echo "--- free -m ---";        free -m 2>/dev/null
      echo "--- df -h /workspace /root ---"; df -h /workspace /root / 2>/dev/null
      echo "--- dmesg tail ---";     dmesg 2>/dev/null | tail -25
    } > "/workspace/out/telemetry-$n.txt" 2>&1
    "${PY:-python3}" /workspace/publish.py "telemetry$n" "/workspace/out/telemetry-$n.txt" || true
    sleep 60
  done
}

note() {  # note <stage> <json-string>
  printf '%s\n' "$2" > "/workspace/out/$1.json"
  "${PY:-python3}" /workspace/publish.py "$1" "/workspace/out/$1.json" || true
}

fail_out() {  # fail_out <stage> <message>
  printf '{"stage":"%s","ok":false,"error":%s,"log_tail":%s}\n' \
    "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" \
    "$(tail -c 4000 /workspace/pgw1548-boot.log | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')" \
    > "/workspace/out/$1.json"
  "${PY:-python3}" /workspace/publish.py "$1" "/workspace/out/$1.json" || true
}

# --- 1. toolchain -----------------------------------------------------------
timeout 600 apt-get update -y && timeout 600 apt-get install -y --no-install-recommends git curl ca-certificates

# --- 1a. THE EVIDENCE CHANNEL IS BUILT BEFORE THE STEP IT MUST OBSERVE -------
# 🔴 THIS ORDER IS THE WHOLE POINT, and getting it wrong cost three pods.
#
# Attempts 1-3 all died BEFORE their first publish and left NOTHING to read:
# container alive, uptime climbing 42 -> 14,382 s, zero variants on the release.
# The reason was structural, not bad luck. The publish machinery needs the
# CLONE (publish.py imports HubClient out of /workspace/pgw/src) and the pip
# set -- and both used to sit AFTER `curl astral.sh | sh`, the ONE UNBOUNDED
# step in the whole pre-heartbeat sequence. So the only step that can hang
# forever was also the one step no instrument could survive. A channel that
# comes after the thing it is supposed to observe is not a channel.
#
# So: clone and the publish deps now come FIRST, using the IMAGE's python (the
# venv does not exist yet and building it needs uv, which is what we are about
# to test). The suspect step below is then left verbatim, with a published
# marker on each side of it. Silence now LOCALISES instead of just happening.
#
# python-gen-worker is a PUBLIC repo, so the clone needs no credential. The
# PRIVATE sibling is never cloned at all -- its endpoint arrives as bytes in
# env (see 2b), which keeps a GitHub PAT off rented hardware entirely.
timeout 900 git clone --filter=blob:none https://github.com/cozy-creator/python-gen-worker /workspace/pgw || exit 90
git -C /workspace/pgw checkout "${PGW1548_SHA}" || exit 90
timeout 900 python3 -m pip install --no-cache-dir \
    requests msgspec psutil grpcio grpcio-tools grpcio-health-checking \
    grpcio-reflection protobuf PySocks typing_extensions certifi \
    charset-normalizer idna urllib3 > /workspace/out/syspip.log 2>&1 || true

# INSTRUMENT SEEN GREEN ONCE. Until this publishes, silence proves nothing --
# it cannot be told apart from a channel that never worked. Everything after
# this marker is interpretable; everything before it is not.
note probe-alive-1 '{"stage":"probe-alive-1","step":"apt + clone + system-pip DONE","next":"curl astral.sh | sh — the unbounded suspect"}'

# The series. Forked HERE, before the suspect, so a hang leaves a TAIL whose
# last sample names the step it stopped after. `${PY:-python3}` resolves to the
# image python at fork time, which is why this works before the venv exists.
telemetry &
TELE=$!

# ⚠️ THE PRE-REGISTERED SUSPECT, LEFT VERBATIM AND UNBOUNDED ON PURPOSE.
# Bounding it would hide the very behaviour this pod was rented to observe.
# PREDICTION: the telemetry series stops after probe-alive-1 and probe-alive-2
# never lands. FALSIFIED IF: probe-alive-2 lands (the hang is downstream), or
# the series never starts at all (the script never ran -- suspect the env
# payload / gunzip in dockerStartCmd instead).
curl -LsSf https://astral.sh/uv/install.sh | sh || true
export PATH="$HOME/.local/bin:$PATH"
note probe-alive-2 '{"stage":"probe-alive-2","step":"curl astral.sh | sh RETURNED — the suspect is NOT the hang"}'


# --- 2. venv ----------------------------------------------------------------
timeout 600 uv venv --python 3.12 /workspace/venv || {
  fail_out toolchain "uv venv failed or timed out"; exit 91; }
export VIRTUAL_ENV=/workspace/venv
PY=/workspace/venv/bin/python
note probe-alive-3 '{"stage":"probe-alive-3","step":"uv venv built"}'

# --- 2a. WHY THERE IS NO PRE-INSTALL HEARTBEAT HERE ---------------------------
# One was written, shipped, and RETRACTED. The idea was to publish a liveness
# stage before the multi-GB torch install, using only `requests` + `msgspec`.
# MEASURED, by blocking every third-party module except those two: importing
# `HubClient` actually needs
#   certifi chardet charset_normalizer google grpc grpc_health grpc_reflection
#   grpc_tools idna msgspec psutil requests socks typing_extensions urllib3
# -- grpcio-tools and psutil are not small wheels. So the heartbeat could never
# fire, and because it was guarded by `|| true` it failed SILENTLY: a fix that
# looks like observability and delivers none is worse than none at all.
#
# The blind window is closed the honest way instead: every long step below is
# TIMEOUT-BOUNDED, so a hang becomes a non-zero exit and a published verdict
# rather than infinite silence, and the first publish sits where it can
# actually run -- after the install.

# --- 2a. INSTALLS — restored after a deletion accident, see below -----------
# ⚠️ THESE LINES WERE ACCIDENTALLY DELETED and it cost a rental. Removing the
# inert heartbeat block took the pip installs with it, because the slice ran
# from the heartbeat comment to section 2b and the installs sat inside that
# range. The pod then created a venv, installed NOTHING, and died on the first
# `gen_worker` import -- while `sleep infinity` held the container up.
# MEASURED signature on pod lmkaw20mk0iodt: uptime 3452 s, cpu 0%, mem 0%,
# nothing published, 57 minutes. The anima pod launched one commit earlier
# still had these lines, which is exactly why anima worked and SDXL could not.
#
# The PUBLISH set goes FIRST and deliberately: importing `HubClient` needs
# certifi chardet charset_normalizer google grpc grpc_health grpc_reflection
# grpc_tools idna msgspec psutil requests socks typing_extensions urllib3
# (measured by blocking every other module). That set is tens of MB against
# torch's several GB, so paying it up front costs ~1 min and buys the one thing
# missing all night: an early failure that can PUBLISH ITS OWN LOG instead of
# dying mute.
timeout 900 uv pip install --python $PY requests msgspec psutil \
    grpcio grpcio-tools grpcio-health-checking grpcio-reflection \
    protobuf PySocks typing_extensions certifi charset-normalizer idna urllib3 \
    > /workspace/out/pubdeps.log 2>&1 || true
if PYTHONPATH=/workspace/pgw/src "$PY" -c "from gen_worker.hubio.client import HubClient" 2>/dev/null; then
  printf '{"stage":"boot","ok":true,"mode":"%s","sha":"%s","pod":"%s","runpod_pod_id":"%s","note":"publish deps in; torch install starting"}\n' \
    "$PGW1548_MODE" "$PGW1548_SHA" "$PGW1548_POD" "${RUNPOD_POD_ID:-}" \
    > /workspace/out/boot.json
  PYTHONPATH=/workspace/pgw/src "$PY" /workspace/publish.py boot /workspace/out/boot.json || true
fi

timeout 900 uv pip install --python $PY -r /workspace/pgw/requirements.txt 2>/dev/null || true
timeout 2700 uv pip install --python $PY -e /workspace/pgw || {
  fail_out install "the pgw editable install failed or timed out"; exit 91; }
timeout 1800 uv pip install --python $PY diffusers==0.39.0 transformers safetensors \
    accelerate huggingface_hub || {
  fail_out install "the endpoint dependency install failed"; exit 91; }
if [ "$PGW1548_MODE" = "anima-derive" ]; then
  timeout 1800 uv pip install --python $PY "diffsynth==2.0.17" torchvision || {
    fail_out install "diffsynth install failed"; exit 91; }
fi

# `import tensorfs` (sdxl/main.py, top level) is satisfied WITHOUT shipping or
# cloning anything: pgw VENDORS tensorfs at src/gen_worker/_vendor, and that
# directory on PYTHONPATH makes the vendored copy importable under its own
# top-level name. tensorfs is NOT on PyPI (404) and its source is 2.1 MB, far
# over the env cliff, so the vendor path is the only free answer.
export PYTHONPATH=/workspace/pgw/src:/workspace/pgw/src/gen_worker/_vendor
export TENSORHUB_URL="$PGW1548_HUB"
# The window gate exists to stop several agent sessions fighting over the ONE
# card on the shared box (`VARENA_GPU_WINDOW`, granted by the coordinator). A
# rented pod is a dedicated card with exactly one tenant -- this process -- so
# the rental IS the window, and the guard has nothing left to protect. Without
# this export the harness refuses on a pod that just spent 22 s building a
# verified bf16 tree, which is precisely what happened on pod iorr3zmp41mea9:
#   "REFUSING: VARENA_GPU_WINDOW=1 is not set."
export VARENA_GPU_WINDOW=1
$PY -c "import tensorfs; print('tensorfs ok', tensorfs.__file__)" || {
  fail_out toolchain "tensorfs import failed"; exit 91; }

# --- 2b. the endpoint source, from env --------------------------------------
# MEASURED CEILING, 2026-08-20: RunPod's REST create takes a ~50 KB `env` and
# returns HTTP 500 on ~150 KB -- and that 500 is its own backend failing to
# parse an upstream error ("invalid character 'I'"), which reads as an outage
# rather than as "your body is too big". Isolated with three probe creates
# (tiny / 50 KB / 150 KB); the first two created and were terminated at once.
#
# The FULL endpoint archive is 141 KB (anima) and 108 KB (sdxl) -- over the
# cliff, and almost all of it is `uv.lock`, which this pod does not use because
# it builds its own venv. The MINIMAL archive (src + endpoint.toml +
# pyproject.toml) is 27 KB and 17 KB, comfortably under. So the source rides
# env after all, and no GitHub PAT and no private clone ever touch rented
# hardware.
mkdir -p /workspace/endpoint
printf '%s' "$PGW1548_ENDPOINT_B64" | base64 -d | tar xz -C /workspace/endpoint
ls -la /workspace/endpoint

mkdir -p /workspace/locks
if [ -n "${PGW1548_LOCKS_B64:-}" ]; then
  printf '%s' "$PGW1548_LOCKS_B64" | base64 -d | tar xz -C /workspace/locks
  ls -la /workspace/locks
fi

# --- 2c. the endpoint's uv.lock ---------------------------------------------
# `gen-worker lock` REFUSES without one: "no uv.lock beside <dir>; a lock states
# the compile stack it traced under, and there is no second source for it"
# (measured, on a pod, after a successful 5.6 GB checkpoint download).
#
# The box's own uv.lock cannot ride here: with it the anima archive is 111 KB
# base64 and the SDXL one 76 KB, against a MEASURED env ceiling between 80 KB
# (creates) and 115 KB (HTTP 500). So it is GENERATED on the pod, which is also
# the more honest artifact -- a lock is a statement about the stack that
# actually traced, and that stack is this pod's venv, not the box's.
if [ ! -f /workspace/endpoint/uv.lock ]; then
  ( cd /workspace/endpoint && timeout 900 uv lock ) > /workspace/out/uvlock.log 2>&1
  if [ ! -f /workspace/endpoint/uv.lock ]; then
    # Fallback, recorded rather than silent: pgw's own lock. It states a real
    # compile stack -- the one this pod's venv was built from -- but it is NOT
    # the endpoint's own, and any artifact derived under it must say so.
    cp /workspace/pgw/uv.lock /workspace/endpoint/uv.lock
    echo "UVLOCK_FALLBACK=pgw" >> /workspace/out/uvlock.log
  fi
  "${PY:-python3}" /workspace/publish.py uvlock /workspace/out/uvlock.log || true
fi

note bootstrap "$(printf '{"stage":"bootstrap","ok":true,"mode":"%s","sha":"%s","python":"%s"}' \
  "$PGW1548_MODE" "$PGW1548_SHA" "$($PY -V 2>&1)")"

# --- 3. the leg -------------------------------------------------------------
case "$PGW1548_MODE" in

install-probe)
  # INSTALL ONLY. No tree, no smoke, no matrix -- the question is whether the
  # torch install survives, and every extra step is a way to lose the answer.
  # The sidecar is ALREADY RUNNING (forked in 1a, before the unbounded suspect)
  # -- starting a second one here would double every sample and race the same
  # publish paths for no gain.
  echo "PROBE: RAM requested $(free -m | awk '/^Mem:/{print $2}') MiB"
  start=$(date +%s)
  timeout 2700 uv pip install --python $PY -e /workspace/pgw \
      > /workspace/out/probe-install.log 2>&1
  rc=$?
  end=$(date +%s)
  sleep 2; kill $TELE 2>/dev/null
  printf '{"stage":"probe","install_rc":%s,"wall_s":%s,"ram_total_mib":%s}\n' \
    "$rc" "$((end-start))" "$(free -m | awk '/^Mem:/{print $2}')" \
    > /workspace/out/probe.json
  "${PY:-python3}" /workspace/publish.py probe \
      /workspace/out/probe.json /workspace/out/probe-install.log || true
  note probe-done "$(printf '{"stage":"probe-done","install_rc":%s}' "$rc")"
  ;;

anima-derive)
  # The checkpoint is PUBLIC on our hub and readable with NO credential
  # (verified from the box before renting), so nothing here carries a
  # checkpoint-read token.
  #
  # `gen-worker download` takes REFs positionally and has NO --dest/--release:
  # it materializes into the machine's weight store "through the same path a
  # pod's boot uses (integrity gate included)". So the tree's location is
  # DISCOVERED afterwards rather than dictated -- a hardcoded path would be a
  # guess, and a wrong guess here is a 3-hour derive against nothing.
  # RELEASE CHOICE IS NOT ARBITRARY, and `latest-cut` is the WRONG answer twice
  # over. It holds 2 variants, and a multi-variant release is a typed refusal:
  #   resolve latest-cut -> 409 release_ambiguous
  #     "holds more than one variant; pin one with ?digest="
  # (measured; it is what killed the first anima leg in 3 seconds). And the
  # right single-variant release is not just "one that downloads" — it is the
  # one the ENDPOINT'S OWN GLOBS put first: main.py documents `@composed-v3`'s
  # COMPONENT layout as the head of every glob tuple, and that release's notes
  # say "component-directory layout + model_index.json". `prod` and
  # `w8a8-staging` also resolve 200, but they are not the layout this code
  # reaches for first.
  ( cd /workspace/endpoint && timeout 3600 $PY -m gen_worker.cli download \
      "tensorhub/anima@${PGW1548_ANIMA_RELEASE:-composed-v3}" ) \
      > /workspace/out/download.log 2>&1
  rc=$?
  "${PY:-python3}" /workspace/publish.py download /workspace/out/download.log || true
  [ $rc -ne 0 ] && { fail_out download "gen-worker download exited $rc"; exit 92; }

  TREE=$($PY - <<'TREEEOF'
import os, sys
from pathlib import Path
# The DiT container names the tree, whatever layout it landed in (anima ships
# both a component layout and a flat split_files/ one -- main.py carries both
# glob families, so neither may be assumed).
roots = [Path.home() / ".cache/cozy", Path("/workspace"), Path("/root/.cache")]
best = ""
for root in roots:
    if not root.exists():
        continue
    for hit in root.rglob("anima-base-v1.0.safetensors"):
        # the tree ROOT is the parent of the component directory
        cand = hit.parent if hit.parent.name in ("", ".") else hit.parent.parent
        if (cand / "model_index.json").exists() or any(cand.iterdir()):
            best = str(cand)
            break
    if best:
        break
print(best)
TREEEOF
)
  if [ -z "$TREE" ]; then
    fail_out download "downloaded, but no anima-base-v1.0.safetensors found — the tree layout is not what the endpoint's globs expect"
    exit 92
  fi
  note download "$(printf '{"stage":"download","ok":true,"tree":"%s","du":"%s"}' \
      "$TREE" "$(du -sh "$TREE" 2>/dev/null | cut -f1)")"

  # The derive drives the entrypoint once per (payload variant x defaults
  # variant) — measured 16 on anima, each a WEIGHT-FULL load. It is the
  # expensive half and the reason this runs on a CPU pod at all.
  ( cd /workspace/endpoint && timeout 10800 $PY -m gen_worker.cli lock . --force \
      --checkpoint "$TREE" ) > /workspace/out/derive.log 2>&1
  rc=$?
  cp /workspace/endpoint/endpoint.lock /workspace/out/endpoint.lock 2>/dev/null || true
  if [ $rc -ne 0 ]; then
    # Publish the LOG even on failure: per the coordinator's rider, a derive
    # that dies at drive 9 must leave drives 1-8 diagnosable.
    "${PY:-python3}" /workspace/publish.py derive /workspace/out/derive.log || true
    fail_out derive "gen-worker lock exited $rc"
    exit 93
  fi
  "${PY:-python3}" /workspace/publish.py derive /workspace/out/derive.log /workspace/out/endpoint.lock || true
  note derive-ok '{"stage":"derive","ok":true,"rc":0}'
  ;;

sdxl-matrix)
  $PY /workspace/pgw/benchmarks/pgw1548_pod_sdxl_tree.py \
      --dest /workspace/sdxl-bf16 > /workspace/out/tree.log 2>&1 || {
        "${PY:-python3}" /workspace/publish.py tree /workspace/out/tree.log || true
        fail_out tree "bf16 tree build failed"; exit 92; }
  "${PY:-python3}" /workspace/publish.py tree /workspace/out/tree.log || true

  SM=$($PY -c "import torch;m,n=torch.cuda.get_device_capability(0);print(f'sm_{m}{n}')")
  FREE=$($PY -c "import torch;print(torch.cuda.mem_get_info(0)[0]//1048576)")
  note headroom "$(printf '{"stage":"headroom","sm":"%s","free_mib":%s,"needed_over_resident_mib":1198}' "$SM" "$FREE")"

  # SMOKE GATE — one arm, one shape, three requests, before any matrix spend.
  #
  # The pgw#1586 PROBE PAIR rides this stage. The instrument's own `vram` block
  # supplies the allocator half and a GIL-BLINDED in-process driver half; the
  # driver half that #1586 can actually use has to come from OUTSIDE the
  # process, because an AOTI `.so` holds the GIL across the compiled call (the
  # blind method was measured wrong by 1174 MiB). So the out-of-process sampler
  # runs across the whole smoke leg and its TSV is published WITH the stage --
  # publishing both is what lets the blinding be demonstrated instead of
  # asserted, which is the form #1586 asked for.
  bash /workspace/pgw/benchmarks/pgw1548_vram_sampler.sh \
      /workspace/out/vram-smoke.tsv 0.05 &
  SMOKESAMP=$!
  sleep 2   # baseline samples of the card before the first request
  ( cd /workspace/pgw && timeout 5400 $PY benchmarks/dynamic_dims_pgw1548.py \
      --endpoint /workspace/endpoint --checkpoint /workspace/sdxl-bf16 \
      --venv /workspace/venv --lock-cache /workspace/locks \
      --latents '1:1=128x128,3:2=104x152,2:3=152x104' \
      --arms static --aspects 1:1 --cfg on --reps 3 --rounds 1 \
      --sm "$SM" --substrate raw-pod --steps 20 --idle-timeout 1800 \
      --lane-note 'sdxl, euler/float32 timestep lane' --dtype-lanes 2 \
      --out /workspace/out/smoke ) > /workspace/out/smoke.log 2>&1
  rc=$?
  sleep 2; kill $SMOKESAMP 2>/dev/null
  $PY /workspace/pgw/benchmarks/pgw1548_analyze_folding.py \
      /workspace/out/vram-smoke.tsv 5222 > /workspace/out/smoke-vram-verdict.txt 2>&1 || true
  "${PY:-python3}" /workspace/publish.py smoke /workspace/out/smoke.log \
      /workspace/out/smoke/verdict.json /workspace/out/vram-smoke.tsv \
      /workspace/out/smoke-vram-verdict.txt || true
  [ $rc -ne 0 ] && { fail_out smoke "smoke gate exited $rc"; exit 94; }

  # MATRIX — ABBA arm order, per-shape, never averaged.
  ( cd /workspace/pgw && timeout 10800 $PY benchmarks/dynamic_dims_pgw1548.py \
      --endpoint /workspace/endpoint --checkpoint /workspace/sdxl-bf16 \
      --venv /workspace/venv --lock-cache /workspace/locks \
      --latents '1:1=128x128,3:2=104x152,2:3=152x104' \
      --arms static,aspect --aspects '1:1,3:2,2:3' --cfg on --reps 3 --rounds 4 \
      --sm "$SM" --substrate raw-pod --steps 20 --idle-timeout 1800 \
      --lane-note 'sdxl, euler/float32 timestep lane; ABBA arm order' --dtype-lanes 2 \
      --out /workspace/out/matrix ) > /workspace/out/matrix.log 2>&1
  rc=$?
  "${PY:-python3}" /workspace/publish.py matrix /workspace/out/matrix.log /workspace/out/matrix/verdict.json || true
  [ $rc -ne 0 ] && { fail_out matrix "matrix exited $rc"; exit 95; }
  note matrix-ok '{"stage":"matrix","ok":true}'

  # --- the LoRA amortization arm, on the SAME graph and one shape ------------
  # A small public SDXL LoRA, downloaded ON THE POD (never through the box).
  LORA=$($PY - <<'LORAEOF'
from huggingface_hub import hf_hub_download
print(hf_hub_download("nerijs/pixel-art-xl", "pixel-art-xl.safetensors"))
LORAEOF
)
  LORA2=$($PY - <<'LORAEOF'
try:
    from huggingface_hub import hf_hub_download
    print(hf_hub_download("ostris/ikea-instructions-lora-sdxl",
                          "ikea_instructions_xl_v1_5.safetensors"))
except Exception:
    print("")
LORAEOF
)
  ( cd /workspace/pgw && timeout 7200 $PY benchmarks/pgw1548_lora_amortization.py \
      --endpoint /workspace/endpoint --checkpoint /workspace/sdxl-bf16 \
      --venv /workspace/venv --lock-cache /workspace/locks \
      --latents '1:1=128x128' --arm static \
      --modes base,fold,eager,prefused,sticky,multi \
      --lora "$LORA" --lora-ref nerijs/pixel-art-xl \
      ${LORA2:+--lora2 "$LORA2"} --sticky-n 3 \
      --aspects 1:1 --cfg on --reps 3 --rounds 3 --steps 20 \
      --sm "$SM" --substrate raw-pod \
      --lane-note 'sdxl, euler/float32 timestep lane' \
      --out /workspace/out/lora ) > /workspace/out/lora.log 2>&1
  rc=$?
  "${PY:-python3}" /workspace/publish.py lora /workspace/out/lora.log /workspace/out/lora/lora-verdict.json || true
  [ $rc -ne 0 ] && { fail_out lora "lora arm exited $rc"; exit 96; }
  note lora-ok '{"stage":"lora","ok":true}'
  # ⚠️ STAGE ORDER: the PRIMARY deliverables (matrix, lora) run BEFORE the
  # pgw#1586 rider legs. Without the pre-#1599 lock cache every stage derives
  # on-pod, so the riders cost ~38 min of rented card AHEAD of the tables this
  # campaign exists to produce -- and those tables have never once run in six
  # attempts. The rider is not starved by this: pgw#1586's probe PAIR now rides
  # the SMOKE stage's out-of-process sampler above, which has already published
  # by this point. What follows the tables is the extra DEPTH (does the first
  # compiled excursion release or persist), which is the part that can be cut
  # by a budget cap without losing the rider itself.
  # --- FOLDING INSTRUMENTATION (pgw#1586 rider) -----------------------------
  # Answers Paul's "why does compiled want so much more VRAM" with a MECHANISM
  # rather than a number: does the first compiled request spike and RELEASE
  # (folding -- AOTI materializing a second copy of the constants, then freeing
  # it), or does it PERSIST across requests (a cudagraph pool that never
  # returns)? The two have the same peak and opposite consequences for how a
  # compiled ceiling should be priced.
  #
  # The sampler is OUT OF PROCESS on purpose: an in-process probe is
  # GIL-blinded and the residency lane proved that wrong by 1.17 GB.
  bash /workspace/pgw/benchmarks/pgw1548_vram_sampler.sh \
      /workspace/out/vram-folding.tsv 0.09 &
  SAMPLER=$!
  sleep 2   # a few samples of the EMPTY card first: the analyzer's baseline

  # TWO requests, deliberately: prediction (2) is first-excursion-releases vs
  # persists, which is unanswerable from a single request.
  ( cd /workspace/pgw && timeout 3600 $PY benchmarks/dynamic_dims_pgw1548.py \
      --endpoint /workspace/endpoint --checkpoint /workspace/sdxl-bf16 \
      --venv /workspace/venv --lock-cache /workspace/locks \
      --latents '1:1=128x128' --arms static --aspects 1:1 --cfg on \
      --reps 2 --rounds 1 --sm "$SM" --substrate raw-pod --steps 20 \
      --idle-timeout 1800 --lane-note 'sdxl folding probe' --dtype-lanes 2 \
      --out /workspace/out/folding ) > /workspace/out/folding.log 2>&1
  sleep 3; kill $SAMPLER 2>/dev/null
  $PY /workspace/pgw/benchmarks/pgw1548_analyze_folding.py \
      /workspace/out/vram-folding.tsv 5222 > /workspace/out/folding-verdict.txt 2>&1 || true
  "${PY:-python3}" /workspace/publish.py folding \
      /workspace/out/vram-folding.tsv /workspace/out/folding-verdict.txt \
      /workspace/out/folding.log || true

  # --- THE CONFOUND LEG: compiled mode, ZERO graphs armed -------------------
  # The residency lane measured a +1176 MiB single allocation AND a death in
  # this configuration locally, which sits confoundingly on the ">1218 MiB AOTI
  # demand" attribution: if an EMPTY store shows the same step, the cost is the
  # mode PATH, not AOTI. Same sampler, same shape, one request.
  mkdir -p /workspace/empty-store
  bash /workspace/pgw/benchmarks/pgw1548_vram_sampler.sh \
      /workspace/out/vram-nograph.tsv 0.09 &
  SAMPLER2=$!
  sleep 2
  ( cd /workspace/pgw && timeout 1800 $PY benchmarks/dynamic_dims_pgw1548.py \
      --endpoint /workspace/endpoint --checkpoint /workspace/sdxl-bf16 \
      --venv /workspace/venv --lock-cache /workspace/locks \
      --graph-store /workspace/empty-store --skip-compile --expect-eager \
      --latents '1:1=128x128' --arms static --aspects 1:1 --cfg on \
      --reps 1 --rounds 1 --sm "$SM" --substrate raw-pod --steps 20 \
      --idle-timeout 900 --lane-note 'sdxl no-graph confound' --dtype-lanes 2 \
      --out /workspace/out/nograph ) > /workspace/out/nograph.log 2>&1
  sleep 3; kill $SAMPLER2 2>/dev/null
  $PY /workspace/pgw/benchmarks/pgw1548_analyze_folding.py \
      /workspace/out/vram-nograph.tsv 5222 > /workspace/out/nograph-verdict.txt 2>&1 || true
  "${PY:-python3}" /workspace/publish.py nograph \
      /workspace/out/vram-nograph.tsv /workspace/out/nograph-verdict.txt \
      /workspace/out/nograph.log || true

  ;;

*)
  fail_out mode "unknown PGW1548_MODE"
  exit 96
  ;;
esac

note done "$(printf '{"stage":"done","ok":true,"mode":"%s"}' "$PGW1548_MODE")"
# Hand back to whatever the image's own CMD was (podguard's watchdog stub
# execs it), so the pod stays alive for its lease rather than exiting into a
# restart loop the box would misread as progress.
exit 0
