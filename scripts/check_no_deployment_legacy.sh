#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPOS=(
  "$ROOT/python-gen-worker"
  "$ROOT/gen-orchestrator"
  "$ROOT/cozy-hub"
)
PATTERN='deployment_id|DeploymentID|DEPLOYMENT_ID|/v1/deployments|/api/v1/deployments|/deployments'

FAILED=0
for repo in "${REPOS[@]}"; do
  if [[ ! -d "$repo" ]]; then
    continue
  fi
  echo "[check] $repo"
  if (
    cd "$repo" && \
      rg -n --hidden "$PATTERN" . \
        --glob '!agents/**' \
        --glob '!scripts/check_no_deployment_legacy.sh' \
        --glob '!**/*.pb.go' \
        --glob '!**/frontend/tsconfig.tsbuildinfo' \
        --glob '!.git/**'
  ); then
    FAILED=1
  fi
done

if [[ "$FAILED" -ne 0 ]]; then
  echo "legacy deployment references detected"
  exit 1
fi

echo "no legacy deployment-id routes/fields detected"
