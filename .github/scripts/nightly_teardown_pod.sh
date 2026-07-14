#!/usr/bin/env bash
# gw#429: destroy a rented nightly pod and hard-fail if it doesn't actually
# go away — a pod that survives is silent GPU spend, so this must be loud.
set -euo pipefail

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is required}"
: "${POD_ID:?POD_ID is required}"

curl -sf -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
  "https://rest.runpod.io/v1/pods/$POD_ID" >/dev/null || true

for i in 1 2 3 4 5 6; do
  code=$(curl -s -o /dev/null -w '%{http_code}' \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "https://rest.runpod.io/v1/pods/$POD_ID")
  if [ "$code" = "404" ]; then
    echo "pod $POD_ID destroyed, confirmed 404"
    exit 0
  fi
  sleep 5
done

echo "::error::pod $POD_ID still present after DELETE (last HTTP $code) — leaking GPU spend!" >&2
exit 1
