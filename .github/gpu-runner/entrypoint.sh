#!/usr/bin/env bash
# One-shot GitHub Actions runner. The pod receives ONLY a single-use
# registration token — never the RunPod API key. Pod termination is the CI
# sweep job's duty; the TTL below just bounds a hung/idle runner.
set -euo pipefail

: "${REPO_URL:?REPO_URL is required}"
: "${RUNNER_TOKEN:?RUNNER_TOKEN is required}"
RUNNER_NAME="${RUNNER_NAME:-gw-gpu-runner-$(hostname)}"
RUNNER_LABELS="${RUNNER_LABELS:-gpu-4090}"
TTL_MINUTES="${TTL_MINUTES:-45}"

cd /runner

timeout 5m ./config.sh --unattended --ephemeral --disableupdate \
    --url "$REPO_URL" --token "$RUNNER_TOKEN" \
    --name "$RUNNER_NAME" --labels "$RUNNER_LABELS"

# --ephemeral: runs exactly one job, then deregisters.
timeout --signal=INT --kill-after=30s "${TTL_MINUTES}m" ./run.sh || true

echo "runner done; exiting so the pod stops burning GPU time"
