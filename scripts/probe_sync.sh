#!/usr/bin/env bash
# pgw#980 — push this working tree's `src/gen_worker` onto a held probe pod and
# respawn its compute child, so one iteration costs seconds instead of an image
# build and a pod spawn.
#
#   scripts/probe_sync.sh <ssh-target>            # sync + respawn
#   scripts/probe_sync.sh <ssh-target> --dry-run  # show what would move
#   scripts/probe_sync.sh <ssh-target> --no-respawn
#
# <ssh-target> is anything ssh accepts (`root@1.2.3.4 -p 12345` works if you
# quote it, or use an ~/.ssh/config alias — preferred).
#
# WHAT MOVES: code only. Weights stay pod-side; the weights-locality rule is
# untouched by design, and this script has no path that would move a checkpoint.
#
# WHAT DOES NOT REFRESH: the control PARENT. A compute-child respawn re-imports
# `gen_worker` in a fresh interpreter, which is what picks up your edit — but the
# parent process (procsplit/parent.py, transport.py, config/, worker_credential.py)
# keeps running the code it booted with. Editing those needs a pod restart, and
# this script says so rather than letting you chase a change that never loaded.
set -euo pipefail

TARGET="${1:-}"
if [ -z "$TARGET" ]; then
  echo "usage: $0 <ssh-target> [--dry-run] [--no-respawn]" >&2
  exit 2
fi
shift || true

DRY_RUN=0
RESPAWN=1
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --no-respawn) RESPAWN=0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/src/gen_worker"
[ -d "$SRC" ] || { echo "no $SRC — run this from the python-gen-worker tree" >&2; exit 2; }

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }

# --- 1. Refuse a pod that is not marked a probe ------------------------------
# GEN_WORKER_PROBE is what disarms cell publish in the parent's action
# allowlist (procsplit/actions.py). Syncing onto a pod without it would put
# unreleased code on a worker whose mints can still reach the shared family
# namespace — the exact poisoning pgw#980 exists to make impossible. The check
# reads the PARENT's environment, because that is the process the guard runs in.
say "checking $TARGET is marked a probe"
PARENT_PID="$(ssh $TARGET "pgrep -f 'gen_worker.entrypoint' | head -1" || true)"
if [ -z "$PARENT_PID" ]; then
  echo "no gen_worker process on $TARGET — is the worker running?" >&2
  exit 3
fi
if ! ssh $TARGET "tr '\0' '\n' < /proc/$PARENT_PID/environ | grep -qx 'GEN_WORKER_PROBE=1'"; then
  cat >&2 <<'EOF'
REFUSED: this pod is not marked a live-edit probe.

Its worker was not started with GEN_WORKER_PROBE=1, so the parent's cell-publish
disarm (pgw#980) is not in force and a mint from rsync'd code could reach the
shared family namespace. Bring the pod up with:

    GEN_WORKER_PROBE=1 WORKER_MODE=forge   # forge => no tenant dispatch either

and re-run. Do not "just this once" a sync onto a serving pod.
EOF
  exit 3
fi

# --- 2. Discover the install path, pod-side ----------------------------------
# Never assumed. The image contract says gen-worker is importable; WHERE it
# landed is the interpreter's business, and an endpoint image that used an
# editable install puts it somewhere /usr/lib guesswork would miss.
say "discovering the pod-side gen_worker path"
DEST="$(ssh $TARGET "python3 -c 'import gen_worker, os; print(os.path.dirname(gen_worker.__file__))'")"
[ -n "$DEST" ] || { echo "could not resolve gen_worker on the pod" >&2; exit 3; }
say "  -> $DEST"

# --- 3. Sync -----------------------------------------------------------------
# --checksum, NOT mtime: rsync preserving OUR mtimes can hand the pod a file
# whose timestamp is OLDER than the .pyc already cached for it, and the
# interpreter would then keep serving the stale bytecode. Content decides.
RSYNC_ARGS=(-rlv --checksum --delete
            --exclude '__pycache__/' --exclude '*.pyc' --exclude '*.so')
[ "$DRY_RUN" = "1" ] && RSYNC_ARGS+=(--dry-run)
say "syncing src/gen_worker/ -> $DEST/"
rsync "${RSYNC_ARGS[@]}" -e "ssh" "$SRC/" "$TARGET:$DEST/"

if [ "$DRY_RUN" = "1" ]; then
  say "dry run — nothing changed, no respawn"
  exit 0
fi

# Drop any bytecode the pod cached under PYTHONPYCACHEPREFIX. Cheap, and it
# removes the one class of "my edit did not take" that is not the parent.
ssh $TARGET "rm -rf /var/lib/gen-worker/compute/pycache 2>/dev/null || true"

if [ "$RESPAWN" = "0" ]; then
  say "synced; respawn skipped (--no-respawn)"
  exit 0
fi

# --- 4. Respawn the compute child -------------------------------------------
# SIGTERM IS THE WRONG SIGNAL and it is a trap worth naming: the compute child
# installs SIGTERM as `lifecycle.start_drain`, which flushes and takes the WHOLE
# worker down — you would lose the pod you are paying to hold. A non-zero exit
# is what the parent's supervision loop treats as a death to recover from, so
# SIGKILL is the sanctioned respawn trigger. In-flight jobs on that group get a
# typed FATAL; a probe has none.
say "respawning the compute child (SIGKILL — SIGTERM would drain the pod)"
CHILD_PIDS="$(ssh $TARGET "pgrep -f 'gen_worker.entrypoint' | grep -v '^$PARENT_PID\$'" || true)"
if [ -z "$CHILD_PIDS" ]; then
  echo "no compute child found beside parent $PARENT_PID; nothing to respawn" >&2
  exit 4
fi
# shellcheck disable=SC2086
ssh $TARGET "kill -9 $CHILD_PIDS"
say "killed child pid(s): $(echo $CHILD_PIDS | tr '\n' ' ')"

# --- 5. Wait for the new child, by OBSERVING it, not by sleeping -------------
# gw#666: no fixed-duration wait decides anything here. The loop watches for a
# child pid that is not one we just killed, and gives up only when the parent
# itself is gone (which would mean the death was not recoverable).
say "waiting for the parent to respawn a child"
while :; do
  if ! ssh $TARGET "kill -0 $PARENT_PID" 2>/dev/null; then
    echo "the control parent exited — the child death was not recoverable." >&2
    echo "Check the pod's logs; a pre-Hello death loop exits the worker by design." >&2
    exit 5
  fi
  NEW="$(ssh $TARGET "pgrep -f 'gen_worker.entrypoint' | grep -v '^$PARENT_PID\$'" || true)"
  FRESH=""
  for pid in $NEW; do
    case " $(echo $CHILD_PIDS | tr '\n' ' ') " in
      *" $pid "*) ;;
      *) FRESH="$pid" ;;
    esac
  done
  if [ -n "$FRESH" ]; then
    say "new compute child: pid $FRESH — running your tree"
    break
  fi
done

say "done. Parent still runs its BOOT-TIME code: edits to procsplit/parent.py,"
say "transport.py, config/ or worker_credential.py need a pod restart to load."
