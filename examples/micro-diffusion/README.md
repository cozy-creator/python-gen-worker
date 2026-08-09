# micro-diffusion — the fleet's smallest REAL endpoint family

**Why it exists.** AOT-mint iteration has been using sdxl as its test vehicle:
6.9 GB of weights, **36 export entries**, ~95 minutes per pod cycle. Nothing
about the mint MACHINERY needs any of that. This family runs the identical
machinery over a 1.1 MB generated checkpoint with **3 export entries**, so a
change to the mint path can be proven in minutes instead of hours.

It is a real org worker, not a fixture: its own `pyproject.toml`,
`endpoint.toml`, Dockerfile-first build contract, catalog slot, registered
export declaration, and two served arms. The only thing about it that is small
is the model.

```
src/micro_diffusion/
  model.py            MicroDenoiser (tiny DiT) + MicroDecoder — conv-free, register_buffer table
  weights.py          deterministic seed -> checkpoint; `python -m micro_diffusion.weights`
  pipeline.py         MicroPipeline: .transformer / .decoder, from_pretrained
  aot_declaration.py  the Compile declaration — 3 entries, container inputs
  main.py             @endpoint Generate: generate (cfg on) / generate_turbo (cfg off)
```

## The three entries, and why exactly three

```
transformer/cfg=true     the guided arm; container arity 2
transformer/cfg=false    the turbo arm; container arity 1
decoder               a SECOND target, its own dims, no fork
```

That set is the smallest one that still exercises every decision the mint path
makes: plan selection, a fork coordinate, a **derived dynamic range** (two
latent rows collapse into one artifact per arm), a second target, entry naming,
the seal, the publish wire, and the cross-process adopt filter.

Two declared facts are there specifically to keep a known defect class under
test on every cycle:

* **A container input with a plain input immediately after it.** `x` is a
  python `list[Tensor]` of arity N, `t` is a plain `(N,)` tensor, `cond` is a
  second list. When `x` expands to N leaves, every contract position after it
  shifts — which is exactly the divergence pgw#994 fixed (`input_contract`
  records the FLATTENED position; `bind_call_inputs` was matching it against
  the caller's PRE-flattening args). Any regression binds the whole list to
  element 0 and the parity leg goes red in seconds.
* **One dim carried by both a container element axis and a plain tensor axis.**
  `H_lat` is `("x", 1)` for the denoiser and `("latent", 2)` for the decoder.
  That is pgw#993's seam: `carried_by` has to resolve through the same
  expansion `dynamic_shapes` mirrors.

## Weights: generated, never fetched

There is no checkpoint in git, none on a developer's box, and no download at
boot. `micro_diffusion.weights` maps `(seed, config) -> bytes` deterministically
and materializes the tree wherever it is needed — into the image at
`docker build` time, on the pod at boot if the binding resolved to an empty
tree, or into the local rig's scratch root.

```bash
python -m micro_diffusion.weights --out /tmp/w --verify   # 1.1 MB, reproduces from seed 997
```

`--verify` regenerates every tensor and byte-compares. The Dockerfile runs it
with `--verify`, so an image whose weights did not reproduce fails the build
rather than shipping a checkpoint nobody else can rebuild.

## Local: the full cycle, in the rig

The pgw#978 micro-mint rig drives the whole production mint path against this
package on this box — no pod, no RunPod, no hub.

```bash
task rig:micro                      # the full cycle against micro-diffusion
task rig:mint -- --vehicle micro    # same thing, long form
task rig:mint                       # pgw#978's original one-entry plumbing toy
```

Legs: gates → weights → handoff → mint-child (a REAL spawned interpreter doing
`torch.export` + AOTInductor) → publish (the real `CellPublisher` wire) →
adopt+parity (a SECOND OS process discovering the cell, arming it, and
comparing every arm against eager).

---

# ⛔ POD RUNBOOK — **DO NOT RUN WITHOUT PAUL'S EXPLICIT GO**

Nothing below has been executed. It is written so that when the go comes, the
run is a transcription rather than a design session. Every step is the same
step attempt 26's sdxl runbook uses; only the sizes differ. **Timings marked
ESTIMATE are derived, not measured** — measuring them is the first cycle's job.

Preconditions (all verified before step 1, none assumed):

- `task inflight STACK=dev` clean, 0 pods live, `max_concurrent_boots` is 1
  fleet-wide so one `booting` row anywhere blocks this.
- The hub's `platform_discretionary_budget` has headroom (a cell mint is
  platform-paid work).
- The wheel is a version that has completed a local `task rig:micro` cycle.
  A wheel nothing has proven locally is what this whole family exists to stop.
- **pgw#1017's `cuda_root` gap is closed** (pgw#1068). A family that ships its
  own Dockerfile gets no composed `/usr/local/cuda` — Dockerfile-LESS families
  inherit it from the hub's synthesized Dockerfile (`cudaRootLine`) — so this
  one asks for it explicitly: `RUN python -m gen_worker.cuda_root` after the
  app install. Without that line the 0.96.x discovery precondition refuses the
  build (`aot precondition cuda_root: … Missing: the root itself`).
- **Step 0 below has been run against the target stack.** The `pipeline` slot
  is a CATALOG slot with no code default, so a first release needs a binding
  naming a repo that already exists AND resolves (th#1087 + th#980).

### 0. Catalog bootstrap — `tensorhub/micro-diffusion` must EXIST and RESOLVE

Do this once per stack, before anything else. **Skipping it costs a build**:
`main.py` declares `Slot(MicroPipeline, selected_by="model")` — a CATALOG slot
with no `default_checkpoint=`, deliberately, because that is sdxl's shape and
the shape the delegation boundary must survive. Since th#1087 a first release
must carry a binding, and th#980 then checks that binding against the live
catalog. On a stack with no such repo the deploy answers:

```
binding_repo_not_found — repo "tensorhub/micro-diffusion" does not exist   (422)
```

**An EMPTY repo does not satisfy it.** The gate
(`internal/bindingcheck/check.go`, `checkTensorhubBinding`) does three things
in order: resolve the owner org, `GetRepo`, then
`ResolveCheckpointSelector(tag)` — and a repo with no checkpoint fails the
third with the SAME `binding_repo_not_found` code. So the README's older
"the pod regenerates the weights if the binding resolved to an empty tree"
idea is not reachable through the deploy gate as it stands; the bootstrap
publishes the tree.

Which is cheap, because the tree is generated, not downloaded — the whole
1.12 MB is a pure function of seed 997:

```
python -m micro_diffusion.weights --out /tmp/micro-weights --seed 997 --verify
```

Then the ordinary 3-step CAS publish v2 flow, as an operator
(`POST /api/v1/password/login` → `access_token`):

```
POST /api/v1/repos/tensorhub/micro-diffusion/publishes
     {"mode":"replace","files":[{path,size_bytes,digest:"sha256:<hex>"}...],
      "tags":[{"tag":"prod"}]}
PUT  <each grant.put_url>            # grant.headers VERBATIM — never rebuild
POST /api/v1/repos/tensorhub/micro-diffusion/publishes/<id>/complete
```

Two things that are easy to get wrong:

- **The repo row must be in the `tensorhub` org.** `POST /api/v1/repos` with a
  user JWT always creates in the caller's PERSONAL org (`createRepo` →
  `authorizePersonalOrgPermissionGin`), so a plain operator cannot create a
  root-org repo through the API at all. Use an org-bound service token, or the
  same direct `INSERT INTO repos` the e2e harness uses for root-org fixtures
  (`e2e/scenarios/publish_v2_test.go`, `newThrowawayRepo`).
- **Stamp `model_family = micro-diffusion`.** The byte classifier answers
  `family_reason: "no architecture signature matched"` for a toy checkpoint, so
  the family comes from the repo row (it fills blanks) or from
  `PATCH /api/v1/repos/tensorhub/micro-diffusion/checkpoints/<id>/classification`.
  Without it the gate fails closed on `binding_incompatible`: *"slot declares
  family … but the artifact's family is undeterminable"*.

One checkpoint serves all three families: `micro-4d` and `micro-escape` both
root to `micro-diffusion` (they `load_state_dict(base.transformer.state_dict(),
strict=True)`), so the seed-997 tree satisfies every slot the package declares.

Verify before spending anything — the tag must resolve:

```
GET /api/v1/repos/tensorhub/micro-diffusion/tags/resolve?tag=prod
GET /api/v1/repos/tensorhub/micro-diffusion/checkpoints     # model_family stamped
```

### 1. Wheel — pin the endpoint at the version under test

`gen-worker==<version>` in `pyproject.toml`, `uv lock`, tar the directory.
`endpoint.toml` is KEPT (it carries the real build profile).
**ESTIMATE: < 1 min.** No commit to `inference-endpoints` is wanted — this is a
proof artifact, not a fleet pin.

> ⚠️ **The pin must carry pgw#994 (`2165c2d5`) — 0.93.4 or a master build.**
> This family declares a container input with a plain input after it, which is
> exactly the shape whose contract positions pgw#994 fixed. On 0.93.3 the cell
> mints and seals and then **refuses at ingress on its first served call**, so
> the run would burn a pod to rediscover a fixed defect. The local rig runs
> against master and its parity leg is a green proof of that fix.

### 2. Publish the release

There is no endpoint-creation call to make first. `POST /api/v1/endpoints` does
not exist; publishing a release auto-creates the endpoint. A runbook step that
posts it has nothing to hit.

```
POST /api/v1/endpoints/tensorhub/micro-diffusion/releases?dev=true&skip_profiling=true
Content-Type: application/gzip     (raw tarball body)
```

Auth: `POST /api/v1/password/login` → **`access_token`** (not `access`),
15-minute expiry — refresh it around the build.

202 → `{build_id, proposed_release_id}`; `200 {"status":"noop"}` is also
success. Poll `GET /api/v1/endpoint-builds/{id}` to `succeeded`.
**ESTIMATE: 2-3 min** (sdxl's measured build is 4m30s; this image installs no
diffusers/transformers/accelerate and its weight-generation layer is ~1 s, so
the delta is the dependency install).

### 3. Tag `prod` — BEFORE the buy

```
PUT /api/v1/endpoints/tensorhub/micro-diffusion/tags/prod  {"release_id": …}
```

`compile-cells` reads the orchestrator's in-memory `LoadRelease` cache and
answers `409 no_compile_declaration` for a release the runtime has not loaded.
That 409 has twice been misread as a missing declaration; it is a cold cache.
**ESTIMATE: seconds.**

### 4. Arm mint boots

`TENSORHUB_COMPILE_OBLIGATION_MINT_BOOTS=true`, restart the hub **process
alone**. Confirm from the LOG, not the API:
`[cell-obligation] loop started mint_boots=true`. The `"armed"` field in the
`compile-cells` 202 reads current config, not the running loop's snapshot, and
will lie to you. **ESTIMATE: < 1 min.**

### 5. Buy

```
POST /v1/admin/compile-cells  {"release_id":…, "gpu_model":"L40S", "coverage":"warmup"}
```

Required, not optional: micro-diffusion is not a flagship family, so a
publish-seeded obligation is `tier=lazy` and is never bought. The same call
un-parks (its `ON CONFLICT (release_id, sku) DO UPDATE` resets `status`,
`attempts`, `discharged_at`, `not_before`). **Buy it as FORGE** — 12 h worker
JWT instead of 30 min, plus the activity-backstop and idle-turnover exemptions.

### 6. Watch

```
GET /v1/admin/mints?release=…
GET /v1/admin/worker-activity-events?kind=aot_mint_phases&release=…
GET /v1/admin/fleet-status | .compile_cells
```

**ESTIMATE: 3-5 min pod-side**, decomposed rather than guessed:

| phase | sdxl (measured) | micro (ESTIMATE) | why |
|---|---|---|---|
| pod boot + image pull | ~2 min | ~1-2 min | much smaller image |
| weights download | minutes (6.9 GB) | **0 s** | generated into the image |
| load + warm | ~1 min | seconds | 1.1 MB, 2 steps |
| export + AOTI compile | ~90 min / 36 entries | ~2-3 min / 3 entries | 12x fewer entries, tiny graphs |
| seal + publish | ~1 min | seconds | the cell is small |

### 7. Adopt

Drive demand so a SECOND pod boots cold and adopts the published cell from the
hub. Bank the eager arm on a pod that is NOT concurrently minting.

### 8. Safety half

The mint is sm_89 (L40S). Boot an `a100-sxm4-80gb` (sm_80) and record the TYPED
refusal — an untested refusal is not a guarantee.

**Whole-cycle ESTIMATE: 5-8 minutes**, against sdxl's measured ~95. That
number is the deliverable; the first real run replaces every ESTIMATE above
with a measurement.
