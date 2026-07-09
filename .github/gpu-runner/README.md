# GPU CI lane (gw#421)

Runs the torch suite on a real RTX 4090 (SM89, the consumer target class). GitHub
Actions orchestrates; the tests execute on an ephemeral RunPod pod registered as a
one-shot self-hosted runner.

## How it works

`gpu-ci.yml`, three jobs:

1. **bootstrap** (ubuntu-latest): mints a single-use runner registration token via the
   GitHub API, creates ONE on-demand RunPod 4090 pod named `gw-gpu-ci-<run_id>` running
   the image built from this directory, waits until the runner shows `online`.
2. **test** (`runs-on: [self-hosted, gpu-4090]`): checkout, `uv sync --extra dev --extra
   torch --extra images` (torch comes from the cu128 lock index, never default PyPI),
   pytest, junit artifact.
3. **sweep** (`always()`): terminates pods named exactly `gw-gpu-ci-<run_id>`, plus any
   `gw-gpu-ci-*` pod older than 2h (leak guard). Never touches other pods.

The pod entrypoint registers with `--ephemeral --labels gpu-4090`, runs exactly one job,
deregisters, and exits its container. A hard TTL (`TTL_MINUTES`, default 45) inside the
entrypoint bounds a hung job; the sweep job does the actual pod termination.

## Security model

Threat: a fork PR edits the workflow or tests, runs attacker code on our runner, steals
credentials or GPU time. Mitigations:

- Triggers are ONLY: nightly cron, push to master, maintainer-applied `gpu` label on a
  PR, and `workflow_dispatch`. Never bare `pull_request`, never `pull_request_target`.
- The label path additionally requires the PR head to live in this repo. Fork PRs never
  reach the runner (and GitHub withholds secrets from fork `pull_request` runs anyway);
  maintainers push the branch to this repo to run the lane.
- `RUNPOD_API_KEY` and `GH_RUNNER_ADMIN_TOKEN` exist only in bootstrap/sweep on
  GitHub-hosted runners. The pod receives exactly one thing: the single-use, 1h-expiry,
  repo-scoped runner registration token. Nothing else to steal.
- Repo setting (Settings → Actions → General): "Require approval for all outside
  collaborators" on fork PR workflows — belt-and-suspenders for the hosted lanes too.
- The runner is `--ephemeral`: one job, then gone; no state survives between runs.

## Cost

Community-cloud on-demand 4090 is ~$0.35/hr; a run is ~10-15 min ≈ $0.10. Nightly +
occasional label runs ≈ single-digit dollars/month. The TTL caps any single run at 45
GPU-minutes even if everything else fails; the sweep + 2h leak guard cap the tail.

## Running it on a PR

Apply the `gpu` label (maintainers only). Re-apply to re-run. This lane is non-blocking
(not a required check).

## Runner image

`ghcr.io/cozy-creator/gw-gpu-runner` — CUDA 12.8 runtime base + actions runner + uv +
git + a baked python 3.11. `gpu-runner-image.yml` rebuilds and pushes on any change to
`.github/gpu-runner/**` (`:latest` moves only on master). The ghcr package must be
public (or the pod create must pass `containerRegistryAuthId`) for RunPod to pull it.

## Setup (one-time)

- Secret `RUNPOD_API_KEY`: RunPod account API key.
- Secret `GH_RUNNER_ADMIN_TOKEN`: fine-grained PAT, this repo only, permission
  "Administration: read & write" (mints runner registration tokens; `GITHUB_TOKEN`
  cannot). Set expiry + rotate.
- Label `gpu` created on the repo.
- ghcr package `gw-gpu-runner` set public after first image push.
