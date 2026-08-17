# The pod mint-rig (pgw#1347)

Rent a pod, run **one** named command on it, bank a machine-readable row, tear
it down and prove it is gone.

```bash
task rig:pod -- mint --gpu sm89 --rail 1.10 --lane pgw1331-clip \
  --target gen_worker.model.catalog.flux1_dev:FLUX1_DEV --runner clip \
  --fleet-line ~/cozy/serverless-endpoints/fleet-floors.toml

task rig:pod -- run --gpu sm89 --rail 0.50 --lane adopt --name adopt \
  --command 'python3 adopt_and_infer.py && echo RIG_DONE' \
  --upload ./adopt_and_infer.py --artifact /root/rig/out

task rig:pod -- sweep                    # live pods vs every record
task rig:pod -- terminate --pod <id>     # or --name <pod name>
task rig:pod -- cards
```

The code is `scripts/mint_rig/`. It is **not** in `src/gen_worker` on purpose:
no pod needs the thing that rents pods, and a control plane inside the worker
wheel puts an account credential on rented hardware for no reason.

## Where it sits

`docs/probe-worker.md` tiers the cheaper rigs. This one is the tier above them:
the first that runs a **real compile on a card you pay for**, and the only one
that does it without a human holding the pod.

| Tier | Cost | Covers |
|---|---|---|
| `task rig:mint` / `rig:micro` / `rig:gauntlet` | free | the whole mint cycle on a toy model |
| **`task rig:pod`** (this) | one rental, capped | a REAL AOTI compile of a real family, on a real card, unattended |
| probe pod | $/hr while you hold it | iterating live on real weights |

## The four rules

**A spend rail is mandatory.** `--rail` is in dollars and there is no default.
It becomes a wall using the rate the create call actually returned, so the same
`--rail 2.00` is five hours on an A40 and twenty minutes on a B200.

**The kill command is written before the pod exists.**
`~/.cache/cozy-mint-rig/killset/<pod-name>.json` is fsynced *before* the POST
leaves, keyed by the name we are about to ask for, and it carries its own
literal kill command. podguard's record is built from the create *response* and
has the same blind spot; the rig arms both.

**No timeouts.** Every wait is a progress gate; the only negative verdict is
`stuck`, meaning the progress token stopped changing. The one exception is pod
BRING-UP, which has no progress signal at all (a pod reports
`desiredStatus: RUNNING` from the instant it is rented and exposes its port only
when the container starts, so an image pull and a wedged host are
byte-identical) — that phase is bounded by `--boot-budget`, a declared fraction
of your own rail.

**Teardown is proved three ways**: DELETE, `GET` → 404, and absent from the
account listing. All three land in the row.

## The row

One JSON per rental under `--out` (default `rig-runs/`):

```
verdict          green | red | stuck | railed | reroll | refused
asked_gpu        the card SET you asked for
observed_gpu     what nvidia-smi says arrived, and observed_sm beside it
cuda_path        native | compat   (RIG-ENV §3c — a wall measured through
                 forward-compat libcuda is a fact about the report)
rate_per_hr      what the provider charged; est_cost_usd = runtime x that
workload_digest  same digest => same thing ran
uploads/artifacts with sha256
teardown         {delete_issued, get_404, absent_from_list}
```

This **prices** a rental. It never reconciles a bill — the ledger's settlement
rows are the charge (see `e2e/privatedeploy/cost.go`).

## Things that cost a pod to learn

Recorded here so the next lane does not pay again.

- **There is no capacity query.** `rest.runpod.io/v1` has no gpu-type or
  availability path at all. Availability is discovered by asking for a card SET
  and reading the create call: HTTP 500 *"this machine does not have the
  resources"* is the out-of-capacity signal, and it is free.
- **sm_86 can be entirely empty.** On 2026-08-17 five Ampere SKUs in one call
  were refused on SECURE and with the cloud type omitted; an Ada card came up on
  the first try. Hence the `sm86` / `sm89` sets and `--cloud`.
- **The fleet base image's interpreter is EXTERNALLY-MANAGED.** PEP 668 refuses
  `pip install` and recommends a venv — which would shadow or re-resolve the
  fleet's own torch. The rig passes `--break-system-packages`; the pod is
  disposable and single-purpose.
- **The fleet-line authority has to be shipped.** `rigcheck` reads
  `endpoint.toml` / `fleet-floors.toml` / ENDPOINT dist metadata and refuses
  gen-worker's own requirement (an SDK certifying its own floor passes every
  rig). This repo has none of those files, so `--fleet-line` is required for
  `mint` unless the image already carries one. The rig refuses **before**
  renting.
- **A mint downloads no checkpoint.** Cell identity is checkpoint-free (§4.27):
  the declaration builds its architecture under fake tensors and the constants
  arrive at arm time from the store. A per-family mint pod costs a card and a
  compile, never 24 GB.

## Adding a lane

Write a `Workload` — a value — and pass it to `Rig.run`. Do not write another
pod driver; that is the mechanism `research/RIG-ENV.md` §5 blames for two false
verdicts and about $11 of wasted rental.
