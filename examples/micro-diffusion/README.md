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
  *_4d.py             micro-4d: the pgw#998 nonlinear-extent (z-image) shape
  *_escape.py         micro-escape: author-defined ops (pgw#1062), GPU-only
  *_conv.py           micro-conv: STATIC-ROWS (sdxl's strategy) — conv-bearing,
                      4 static entries, int64 timestep, persistent named buffer (pgw#1073)
  *_pad32.py          micro-pad32: ie#637's PAD-TO-32 shape — the token extent is
                      `32*FloorDiv(H*W+31, 32)`, and one collapsed artifact serves
                      three L values whose pads are 0, 16 and 28
  *_pad32_branchy.py  micro-pad32-branchy: the RED twin — branches on the pad, so
                      the declared-range gate must REFUSE it (pgw#1079)
  FAMILIES            the seven families discovery finds here — half of the
                      cross-repo fence below
```

## Adding a member — FOUR steps, and three of them are outside this file

**Discovery walks the WHOLE package** (`top_level = main.split(".")[0]`), so a
member whose family is unregistered hub-side fails **every build that carries
this example**, not just its own function. That has happened three times —
`micro-escape` (pgw#1068), `micro-conv` (pgw#1073), `micro-pad32` +
`micro-pad32-branchy` (pgw#1079, found mid-campaign by pgw#1084 §8.4.1, which
could only proceed by excluding `*pad32*` from the proof tarball). The typed
refusal is not the problem; it fires against a live hub with the tarball already
uploaded, which is too late and costs a lane its leg.

1. **The example** — `main_<x>.py` + `aot_declaration_<x>.py` + a pipeline class,
   with its own `FAMILY = "micro-<x>"`.
2. **`FAMILIES`** — add the name, sorted. `tests/test_micro_family_registration_pgw1084.py`
   runs the REAL discovery scan and goes RED until you do; it also goes red on a
   line here whose member is gone.
3. **The tensorhub registration** — `internal/modelfamily/modelfamily.go`:
   `canonicalFamilies`, plus a `rootOverrides` entry to `micro-diffusion` when
   the member loads the base transformer's `state_dict` (all of them so far do,
   which is why ONE seeded checkpoint satisfies every slot). Update the vendored
   `internal/modelfamily/MICRO_EXAMPLE_FAMILIES` in the same commit — its
   offline test is in the required `gates` check, and `scripts/micro-family-drift.sh`
   diffs the vendored copy against THIS file on `python-gen-worker@master`.
   There is no runtime registration path; it takes effect at the next hub deploy.
   **pgw lands first** — the hub gate reads pgw's default branch.
4. **The stack binding** — step 0 below. Registered-but-unbound is the *second*
   failure, one layer deeper, and it is what still blocks `micro-conv`:
   `declared slot has no binding: function "generate-conv" slot "pipeline" has no
   code default and current prod release … has no matching family "micro-conv"`
   (pgw#1084 §8.4.2). Bindings are stack state, not repo state — nothing in any
   repository seeds them, so a fresh stack owes step 0 for every family here.

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
`torch.export` + AOTInductor) → publish (the real `CompiledGraphPublisher` wire) →
adopt+parity (a SECOND OS process discovering the compiled graph, arming it, and
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
- The hub's `platform_discretionary_budget` has headroom (a compiled graph mint is
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

Create the repo first, then the ordinary 3-step CAS publish v2 flow, all as an
operator (`POST /api/v1/password/login` → `access_token`):

```
POST /api/v1/repos
     {"org":"tensorhub","name":"micro-diffusion"}
POST /api/v1/repos/tensorhub/micro-diffusion/publishes
     {"mode":"replace","files":[{path,size_bytes,digest:"sha256:<hex>"}...],
      "tags":[{"tag":"prod"}]}
PUT  <each grant.put_url>            # grant.headers VERBATIM — never rebuild
POST /api/v1/repos/tensorhub/micro-diffusion/publishes/<id>/complete
```

Two things that are easy to get wrong:

- **NAME the org — `"org":"tensorhub"` — or the repo lands in your personal
  one.** Omitting it is not an error: a user token defaults to the caller's
  personal org, which is how the first attempt at this runbook created
  `cozy/micro-diffusion`. The operator's authority over a non-personal org is
  the platform-admin role (th#1730, tensorhub `authorizeNamedOrgRepoWrite`); a
  caller holding neither that nor `org:repo:write` in `tensorhub` is refused
  with `403 org_repo_write_forbidden`. This step used to require the direct
  `INSERT INTO repos` the e2e harness uses for root-org fixtures — it does not
  any more, and a re-seed that still hand-INSERTs is working around a hub older
  than th#1730.
- **Stamp `model_family = micro-diffusion`.** The byte classifier answers
  `family_reason: "no architecture signature matched"` for a toy checkpoint, so
  the family comes from the repo row (it fills blanks) or from
  `PATCH /api/v1/repos/tensorhub/micro-diffusion/checkpoints/<id>/classification`.
  Without it the gate fails closed on `binding_incompatible`: *"slot declares
  family … but the artifact's family is undeterminable"*.

One checkpoint serves **all seven** families: every variant either
`load_state_dict(base.transformer.state_dict(), strict=True)` or (micro-conv)
derives its weights from the same tree's declared seed, and all of them root to
`micro-diffusion` in tensorhub's `rootOverrides` — so the seed-997 tree
satisfies every slot the package declares.

### 0b. One binding PER FUNCTION — the second failure, and what blocked micro-conv

A checkpoint that resolves is not a binding. Every function here declares
`Slot(..., selected_by="model")` with **no code default**, so each one needs its
own entry, and the build refuses per function:

```
declared slot has no binding: function "generate-conv" slot "pipeline" has no
code default and current prod release … has no matching family "micro-conv"
```

That is pgw#1084 §8.4.2 — `micro-conv` was registered in tensorhub and still
unbuildable, so the campaign's gauntlet was 3 families, not 5. A new function
inherits nothing, because there is no prior prod release declaring it; supply the
binding **on the publish that first declares it**, with `?bindings=` (the tarball
form of th#1087's payload — the JSON deploy form takes the same map as
`{"bindings": …}`):

```
POST /api/v1/endpoints/tensorhub/micro-diffusion/releases?dev=true&skip_profiling=true
     &bindings={"generate-conv":{"pipeline":{"ref":"tensorhub/micro-diffusion","tag":"prod"}}}
```

Function names are the NORMALIZED spellings (`generate_conv` → `generate-conv`).
Thereafter the binding is config and is patchable without a rebuild:

```
PATCH /api/v1/endpoints/tensorhub/micro-diffusion/config?tag=prod
      {"bindings":{"generate-conv":{"pipeline":{"ref":"tensorhub/micro-diffusion"}}}}
```

**Bindings are STACK state.** No repository holds them — grep for
`micro-diffusion` across tensorhub, e2e and the gitops repo and you find the
family registration and nothing else — so every fresh stack owes this step for
all nine functions, and a family added to the package is unbuildable on an
existing stack until its function is bound here.

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

> ⚠️ **Pin at 0.111.0 (current), never below it on THIS family.** Below
> 0.111.0 the reuse circle cannot close: pgw#1141 (`0dbf68e5`) deletes the setup
> warmup barrier that DISARMED every boot-adopted compiled graph — measured twice,
> identically, in POD PROOF #3 — and pgw#1132 (`8867baac`) arms the lifted
> forward in the boot-key loop so a `lora_bucket`-bearing family can derive a key
> at all. Below
> 0.106.0 the boot-key derivation refuses outright: `models/structure_only`
> imported `accelerate`, which this image deliberately does not ship, so no key
> exists, no `/v1/worker/compiled-graphs/resolve` is ever issued and the pod self-mints
> forever — measured on `ykwoaiqub6ktt3` and `3o09rf9ehnc4ym`, and it is what
> silenced the first paid reuse-circle proof (pgw#1123). Older floors still
> apply and are all below it: pgw#994 (`2165c2d5`, 0.93.4) for the container
> input followed by a plain input — below it the compiled graph mints and seals and then
> **refuses at ingress on its first served call**; 0.97.0 for pgw#1084's
> four-axis `ck1`; 0.100.0 for the AOT re-key (pgw#1089/1090), the folding fence
> (pgw#1097) and AOT-local mint (pgw#1096). Keep this pin on the SAME wheel the
> fleet serves (the example's `pyproject.toml` pin is the source of truth) so
> the gauntlet stays a usable probe.

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

### 3. Point semver-major 0 at the release — BEFORE the buy

```
PUT /api/v1/endpoints/tensorhub/micro-diffusion/serving/0  {"release_id": …}
```

**Endpoint tags are dead (th#2044): serving is one hidden pointer per
`(endpoint, semver_major)`, moved only by an explicit author act, and demand
and every other read resolve through it (th#2046).** `0` is the common case —
the package's version is 0.x, and the invoke grammar in step 5 says so. A
release no pointer names is a release nothing can invoke. The `"tag"` inside a
binding (step 0b) is the REPO axis — a model release — and is untouched; the
`config?tag=` query is the endpoint axis and dies with th#2048.
**ESTIMATE: seconds.**

### 4. Buy a SERVING pod

**There is no mint-only pod class and no `POST /v1/admin/compiled graphs` route.**
DESIGN-RULINGS §4.28 retired both: a serving pod boots, tries to adopt a compiled graph
by its own derived `ck1` key, and self-mints in the BACKGROUND on a miss while
it serves eager. Nothing has to be armed, un-parked or bought as forge.

```
POST /v1/admin/releases/<release_id>/workers?count=1&compute_class=gpu
```

**ESTIMATE: < 1 min to `running`.**

### 5. Drive one request

```
POST /tensorhub/micro-diffusion/v0/generate
{"prompt": "a tiny test", "size": "256x256", "steps": 2}
```

The `vN` segment is REQUIRED and has no default (th#2045); it names the
semver-major whose pointer step 3 wrote.

`size` is an **ENUM** (`"256x256"` / `"384x384"`), not a `{"width":…,"height":…}`
object — the object form is a `400`, and it cost a pod leg. The rows are an enum
on purpose: a free width/height int would be a shape-affecting payload field,
which is the thing every fleet endpoint is linted against.

The request is served EAGER (the pod missed and is minting). The mint runs
behind it; watch for the mint-parent parity verdict and the publish.

### 6. Watch

```
GET /v1/admin/mints?release=…
GET /v1/admin/worker-activity-events?kind=aot_mint_phases&release=…
GET /v1/admin/fleet-status | .compile_contracts
```

**ESTIMATE: 3-5 min pod-side**, decomposed rather than guessed:

| phase | sdxl (measured) | micro (ESTIMATE) | why |
|---|---|---|---|
| pod boot + image pull | ~2 min | ~1-2 min | much smaller image |
| weights download | minutes (6.9 GB) | **0 s** | generated into the image |
| load + warm | ~1 min | seconds | 1.1 MB, 2 steps |
| export + AOTI compile | ~90 min / 36 entries | ~2-3 min / 3 entries | 12x fewer entries, tiny graphs |
| seal + publish | ~1 min | seconds | the compiled graph is small |

### 7. Adopt

Retire the minting pod, buy a SECOND one the same way, and drive the same
request. It must boot cold and adopt the compiled graph the first one published. Read the
verdict off the pod's own typed rows, never off a log:

```
GET /v1/admin/worker-activity-events?release=…&kind=boot_adopt
GET /v1/admin/worker-activity-events?release=…&kind=compiled_graph_numerics
GET /v1/admin/worker-activity-events?release=…&kind=serve_eager_posture
GET /v1/admin/worker-activity-events?release=…&kind=serve_degrade
```

GREEN is `boot_adopt=hit`, `compiled_graph_numerics=armed_undispatched`, `lane=…+compiled`
/ `serving_mode=aot_cell`, and **no `serve_degrade` row at all**. A
`target_applicability_incomplete` or `armed_target_unresolved` row is pgw#1141b
(POD PROOF #4) recurring — that boot adopted a compiled graph and then threw it away.

Bank the eager arm on a pod that is NOT concurrently minting.

### 8. Safety half

The mint is sm_89 (L40S). Boot an `a100-sxm4-80gb` (sm_80) and record the TYPED
refusal — an untested refusal is not a guarantee.

**Whole-cycle ESTIMATE: 5-8 minutes**, against sdxl's measured ~95. That
number is the deliverable; the first real run replaces every ESTIMATE above
with a measurement.
