# The live-edit probe worker (pgw#980)

Buy **one** pod, hold it, iterate on it live, terminate it deliberately.

One iteration is an `rsync` and a child respawn — seconds — instead of a PyPI
release, an image build and a pod spawn. Weights never move; only code does.

## When to reach for it

Tier by tier, cheapest first:

| Tier | Cost per iteration | Covers |
|---|---|---|
| `task rig:mint` (pgw#978) | seconds, free | resolve, handoff, spawn, load, warm, export, compile, seal, publish, adopt — on a toy model, ONE export entry |
| `task rig:micro` (pgw#997) | ~15 s, free | all of the above on a REAL org-worker package — 3 entries, two fork arms, a second target, container inputs, a derived dynamic range, plus a PARITY check that arms the adopted cell and compares every arm to eager |
| experimental image (pgw#979) | one image build | the real image, the real deps, a real pod, a real family |
| **probe pod (this doc)** | seconds, on a pod you already hold | everything above, on real weights and a real card, iterating |

Reach for a probe when the rig has run out of things it can tell you: when the
question is about *this* GPU, *these* weights, or a wall the toy model cannot
reach. Do not reach for it first — it costs $/hr for as long as you hold it, and
almost every plumbing defect dies to the rig.

## Bringing one up

One environment variable, required, and the sync script refuses a pod without
it:

```bash
GEN_WORKER_PROBE=1      # marks the pod a probe; DISARMS cell publish
```

There is no longer a mode that keeps the hub from dispatching to a probe:
§4.28 deleted the forge and `WORKER_MODE` with it (pgw#1092). Buy the probe pod
against a release nothing is routing traffic to.

`GEN_WORKER_PROBE=1` is not advisory. It is read by the **control parent**, in
`procsplit/actions.py`, and it removes `cells.publish_intent` /
`cells.publish_complete` from the set of hub calls the parent will make on the
compute child's behalf. The compute child holds no credential and can reach the
hub only through that allowlist, so **nothing you rsync into the child can
re-arm publishing.** That is the guarantee: it is structural, not procedural.

Why it matters: a probe runs code that is, by construction, not any released
version and not any built image. Its mints are stamped with a `gen_worker`
version read from dist-info — which rsync does not move, so it *lies* — and a
`code_closure` metadata memo no other pod can reproduce (pgw#990 took the
closure out of the key; the honesty problem is unchanged). A cell published
from a probe into the shared family namespace is a cell that every later pod
may adopt and none can explain.

Arming it is a second, separate decision:

```bash
GEN_WORKER_PROBE_PUBLISH_ARMED=1
```

Two names rather than one truthy flag: "this is a probe" and "this probe may
write to the store" are different decisions, and the second must never be
reachable by forgetting the first.

## The loop

```bash
task probe-sync -- my-probe-pod              # sync + respawn
task probe-sync -- my-probe-pod --dry-run    # what would move
task probe-sync -- my-probe-pod --no-respawn # sync only
```

`my-probe-pod` is best kept as an `~/.ssh/config` alias.

What the script does, in order:

1. **Refuses a pod that is not marked a probe** — it reads `GEN_WORKER_PROBE`
   out of the running parent's `/proc/<pid>/environ`, because the guard runs in
   that process and nowhere else.
2. **Discovers the install path pod-side** with
   `python3 -c 'import gen_worker, os; print(os.path.dirname(gen_worker.__file__))'`.
   Never assumed: the image contract says gen-worker is *importable*; where it
   landed is the interpreter's business.
3. **Syncs with `--checksum`, not mtime.** rsync preserving your local mtimes can
   hand the pod a file older than the `.pyc` already cached for it, and the
   interpreter would keep serving stale bytecode. It also clears
   `/var/lib/gen-worker/compute/pycache`.
4. **Respawns the compute child with `SIGKILL`.**
5. **Waits by observing** for a child pid that is not one it killed — no fixed
   sleep decides anything (gw#666).

### SIGTERM is the wrong signal

The compute child installs `SIGTERM` as `lifecycle.start_drain`. A drain is
terminal for the **whole pod**: it flushes and the worker exits. You would lose
the pod you are paying to hold.

A non-zero exit is what the parent's supervision loop treats as a death to
recover from, so `kill -9` is the sanctioned respawn trigger. The parent
respawns after ~1 s of backoff, reports the death as a typed
`ComputeProcessDied` for any in-flight job (a probe has none), and at one
execution group cycles the hub connection so residency is re-driven.

### What a respawn does NOT reload

Only the **compute child** re-imports your code. The control parent keeps
running what it booted with. So edits to any of:

- `procsplit/parent.py`, `procsplit/transport.py`, `procsplit/actions.py`
- `config/`, `worker_credential.py`

need a **pod restart**, not a sync. The script says so on the way out rather
than letting you chase a change that never loaded.

One more consequence worth knowing: the `code_closure` memo is `lru_cache`d
over source content, so within a live process your swap does not move the
recorded closure — after the respawn it does. (Since pgw#990/pgw#1059 the
closure is a memo, never identity: a code edit changes the cell key only
through the traced graph.)

## Terminating

Deliberately, and by hand. A probe has no idle timeout; that is the trade for
holding it. When the session is done, terminate the pod and say so in the
tracker issue. A forgotten probe is the most expensive thing in this document.
