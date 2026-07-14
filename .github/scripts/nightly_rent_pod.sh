#!/usr/bin/env bash
# gw#429: rent ONE RunPod GPU pod for the nightly tier, wait until it's SSH-reachable.
#
# Requires: RUNPOD_API_KEY, POD_NAME, SSH_PUBKEY (env). GPU_TYPE optional
# (default RTX 4090). Prints "POD_ID=... POD_IP=... POD_PORT=..." lines to
# stdout on success (source-able); everything else goes to stderr.
#
# Used by both .github/workflows/nightly.yml and manual dry runs — this is
# the one place that knows how to create + wait for a pod, so both paths
# exercise identical logic.
set -euo pipefail

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is required}"
: "${POD_NAME:?POD_NAME is required}"
: "${SSH_PUBKEY:?SSH_PUBKEY is required}"
GPU_TYPE="${GPU_TYPE:-NVIDIA GeForce RTX 4090}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-60}"

# Leak guard: a crashed/cancelled prior run's teardown step may never have
# executed. Sweep any stale gw-nightly-* pod older than 3h before renting a
# new one — the required teardown+verify below is still the primary defense.
cutoff=$(date -u -d '3 hours ago' '+%Y-%m-%d %H:%M:%S')
stale=$(curl -sf -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods" \
  | jq -r --arg cutoff "$cutoff" \
    '.[] | select((.name | test("^gw-nightly-[0-9]+$")) and .createdAt < $cutoff) | .id' 2>/dev/null || true)
for id in $stale; do
  echo "leak guard: terminating stale pod $id" >&2
  curl -sf -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$id" >/dev/null || true
done

# Plain CUDA-devel base + the documented custom-image SSH recipe
# (docs.runpod.io/pods/configuration/use-ssh) — no dependency on RunPod's
# official-template auto-SSH behavior, so this works on any image we pick.
IMAGE="${IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu22.04}"
START_CMD='apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server git curl ca-certificates >/dev/null && mkdir -p /root/.ssh && chmod 700 /root/.ssh && echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys && service ssh start && sleep infinity'

body=$(jq -n \
  --arg name "$POD_NAME" \
  --arg image "$IMAGE" \
  --arg gpu "$GPU_TYPE" \
  --arg key "$SSH_PUBKEY" \
  --arg startcmd "$START_CMD" \
  --argjson diskgb "$CONTAINER_DISK_GB" \
  '{
    name: $name,
    imageName: $image,
    gpuTypeIds: [$gpu],
    gpuCount: 1,
    cloudType: "COMMUNITY",
    interruptible: false,
    containerDiskInGb: $diskgb,
    volumeInGb: 0,
    ports: ["22/tcp"],
    dockerStartCmd: ["bash", "-c", $startcmd],
    env: { PUBLIC_KEY: $key }
  }')

pod=""
for attempt in 1 2 3; do
  for cloud in COMMUNITY SECURE; do
    try=$(echo "$body" | jq --arg c "$cloud" '.cloudType = $c')
    resp=$(curl -s -w '\n%{http_code}' -X POST "https://rest.runpod.io/v1/pods" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -d "$try")
    code=${resp##*$'\n'}; json=${resp%$'\n'*}
    if [ "$code" = "200" ] || [ "$code" = "201" ]; then pod="$json"; break 2; fi
    echo "create attempt $attempt ($cloud) -> HTTP $code: $(echo "$json" | head -c 300)" >&2
    sleep 20
  done
done
[ -n "$pod" ] || { echo "no GPU capacity after 6 attempts (COMMUNITY+SECURE x3)" >&2; exit 1; }

pod_id=$(echo "$pod" | jq -r .id)
echo "created pod $pod_id" >&2

ip=""
port=""
for i in $(seq 1 60); do
  cur=$(curl -sf -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$pod_id")
  ip=$(echo "$cur" | jq -r '.publicIp // empty')
  port=$(echo "$cur" | jq -r '.portMappings["22"] // empty')
  status=$(echo "$cur" | jq -r '.desiredStatus // empty')
  if [ "$status" = "TERMINATED" ] || [ "$status" = "EXITED" ]; then
    echo "pod $pod_id entered $status while waiting for SSH" >&2
    exit 1
  fi
  if [ -n "$ip" ] && [ -n "$port" ]; then
    if timeout 5 bash -c "cat < /dev/null > /dev/tcp/$ip/$port" 2>/dev/null; then
      echo "pod $pod_id reachable at $ip:$port" >&2
      echo "POD_ID=$pod_id"
      echo "POD_IP=$ip"
      echo "POD_PORT=$port"
      exit 0
    fi
  fi
  sleep 10
done

echo "pod $pod_id never became SSH-reachable (id=$pod_id ip=$ip port=$port) — leaving it for teardown" >&2
echo "POD_ID=$pod_id"
exit 1
