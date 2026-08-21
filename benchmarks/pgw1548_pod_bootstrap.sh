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
#   PGW1548_ENDPOINT_B64  the endpoint source, tar.gz+base64 (no GitHub PAT on
#                         rented hardware, and no private clone)
#   PGW1548_SHA           pgw commit to check out (public repo, no auth)
#   HF_TOKEN              optional, for the SDXL snapshot download
set -u
exec > /workspace/pgw1548-boot.log 2>&1
set -x
export DEBIAN_FRONTEND=noninteractive
export PATH="$HOME/.local/bin:$PATH"
mkdir -p /workspace/out
cd /workspace

STAGE_RC=0

# --- the evidence channel, defined before anything can fail ------------------
# A plain python function rather than a CLI: it reuses HubClient.publish_v2,
# which is what pgw#1568 ruled (reuse, not a second publish path).
cat > /workspace/publish.py <<'PYEOF'
import json, os, sys
from pathlib import Path
sys.path.insert(0, "/workspace/pgw/src")
from gen_worker.hubio.client import HubClient, CommitFile

stage = sys.argv[1]
paths = [Path(p) for p in sys.argv[2:] if Path(p).exists()]
if not paths:
    print(f"[publish] {stage}: nothing to publish"); raise SystemExit(0)
c = HubClient(base_url=os.environ["PGW1548_HUB"], token=os.environ["PGW1548_TOKEN"])
files = [CommitFile(path=f"{stage}/{p.name}", local_path=p) for p in paths]
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

note() {  # note <stage> <json-string>
  printf '%s\n' "$2" > "/workspace/out/$1.json"
  python3 /workspace/publish.py "$1" "/workspace/out/$1.json" || true
}

fail_out() {  # fail_out <stage> <message>
  printf '{"stage":"%s","ok":false,"error":%s,"log_tail":%s}\n' \
    "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" \
    "$(tail -c 4000 /workspace/pgw1548-boot.log | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')" \
    > "/workspace/out/$1.json"
  python3 /workspace/publish.py "$1" "/workspace/out/$1.json" || true
}

# --- 1. toolchain -----------------------------------------------------------
apt-get update -y && apt-get install -y --no-install-recommends git curl ca-certificates
curl -LsSf https://astral.sh/uv/install.sh | sh || true
export PATH="$HOME/.local/bin:$PATH"

# python-gen-worker is a PUBLIC repo, so this needs no credential. The private
# sibling is NOT cloned: its endpoint arrives as bytes in env, which keeps a
# GitHub PAT off rented hardware entirely.
git clone --filter=blob:none https://github.com/cozy-creator/python-gen-worker /workspace/pgw || exit 90
git -C /workspace/pgw checkout "${PGW1548_SHA}" || exit 90

mkdir -p /workspace/endpoint
printf '%s' "$PGW1548_ENDPOINT_B64" | base64 -d | tar xz -C /workspace/endpoint --strip-components=1

# --- 2. venv ----------------------------------------------------------------
uv venv --python 3.12 /workspace/venv || exit 91
export VIRTUAL_ENV=/workspace/venv
PY=/workspace/venv/bin/python
uv pip install --python $PY -r /workspace/pgw/requirements.txt 2>/dev/null || true
uv pip install --python $PY -e /workspace/pgw || exit 91
uv pip install --python $PY diffusers==0.39.0 transformers safetensors accelerate huggingface_hub || exit 91

export PYTHONPATH=/workspace/pgw/src
note bootstrap "$(printf '{"stage":"bootstrap","ok":true,"mode":"%s","sha":"%s","python":"%s"}' \
  "$PGW1548_MODE" "$PGW1548_SHA" "$($PY -V 2>&1)")"

# --- 3. the leg -------------------------------------------------------------
case "$PGW1548_MODE" in

anima-derive)
  # The checkpoint is PUBLIC on our hub and readable with NO credential
  # (verified from the box before renting), so nothing here carries a
  # checkpoint-read token.
  $PY -m gen_worker.cli download tensorhub/anima --release "${PGW1548_ANIMA_RELEASE:-latest-cut}" \
      --dest /workspace/anima-tree > /workspace/out/download.log 2>&1 || {
        fail_out download "anima checkpoint download failed"; exit 92; }
  note download "$(printf '{"stage":"download","ok":true,"tree":"/workspace/anima-tree","du":"%s"}' \
      "$(du -sh /workspace/anima-tree 2>/dev/null | cut -f1)")"

  # The derive drives the entrypoint once per (payload variant x defaults
  # variant) — measured 16 on anima, each a WEIGHT-FULL load. It is the
  # expensive half and the reason this runs on a CPU pod at all.
  ( cd /workspace/endpoint && timeout 10800 $PY -m gen_worker.cli lock . --force \
      --checkpoint /workspace/anima-tree ) > /workspace/out/derive.log 2>&1
  rc=$?
  cp /workspace/endpoint/endpoint.lock /workspace/out/endpoint.lock 2>/dev/null || true
  if [ $rc -ne 0 ]; then
    # Publish the LOG even on failure: per the coordinator's rider, a derive
    # that dies at drive 9 must leave drives 1-8 diagnosable.
    python3 /workspace/publish.py derive /workspace/out/derive.log || true
    fail_out derive "gen-worker lock exited $rc"
    exit 93
  fi
  python3 /workspace/publish.py derive /workspace/out/derive.log /workspace/out/endpoint.lock || true
  note derive-ok "$(printf '{"stage":"derive","ok":true,"rc":0}')"
  ;;

sdxl-matrix)
  $PY /workspace/pgw/benchmarks/pgw1548_pod_sdxl_tree.py \
      --dest /workspace/sdxl-bf16 > /workspace/out/tree.log 2>&1 || {
        python3 /workspace/publish.py tree /workspace/out/tree.log || true
        fail_out tree "bf16 tree build failed"; exit 92; }
  python3 /workspace/publish.py tree /workspace/out/tree.log || true

  SM=$($PY -c "import torch;m,n=torch.cuda.get_device_capability(0);print(f'sm_{m}{n}')")
  FREE=$($PY -c "import torch;print(torch.cuda.mem_get_info(0)[0]//1048576)")
  note headroom "$(printf '{"stage":"headroom","sm":"%s","free_mib":%s,"needed_over_resident_mib":1198}' "$SM" "$FREE")"

  # SMOKE GATE — one arm, one shape, three requests, before any matrix spend.
  ( cd /workspace/pgw && timeout 5400 $PY benchmarks/dynamic_dims_pgw1548.py \
      --endpoint /workspace/endpoint --checkpoint /workspace/sdxl-bf16 \
      --venv /workspace/venv --lock-cache /workspace/locks \
      --latents '1:1=128x128,3:2=104x152,2:3=152x104' \
      --arms static --aspects 1:1 --cfg on --reps 3 --rounds 1 \
      --sm "$SM" --substrate raw-pod --steps 20 --idle-timeout 1800 \
      --lane-note 'sdxl, euler/float32 timestep lane' --dtype-lanes 2 \
      --out /workspace/out/smoke ) > /workspace/out/smoke.log 2>&1
  rc=$?
  python3 /workspace/publish.py smoke /workspace/out/smoke.log /workspace/out/smoke/verdict.json || true
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
  python3 /workspace/publish.py matrix /workspace/out/matrix.log /workspace/out/matrix/verdict.json || true
  [ $rc -ne 0 ] && { fail_out matrix "matrix exited $rc"; exit 95; }
  note matrix-ok '{"stage":"matrix","ok":true}'
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
