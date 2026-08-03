# Local development with `gen-worker run`

`gen-worker run` executes one method of your endpoint in the **local
Python interpreter**, against a JSON payload you supply. No docker-compose,
no tensorhub round-trip, no orchestrator. Inner-loop dev for endpoint
authors.

```bash
pip install -e .          # or: uv sync
gen-worker run --payload '{"text": "marco"}'
# {"event":"result","value":{"response":"polo"}}
```

## The two-input model

`gen-worker run` takes **two** inputs. That's the whole interface.

1. **Which function to call** — by class + method name. Both are inferred
   when the endpoint has exactly one routable function (or one named after the package).
2. **What payload to send** — JSON, inline (`--payload '...'`) or from a
   file (`--payload-file ./fixture.json`). Validated against the function's
   `msgspec.Struct` input type.

Everything else derives from the code:

- Model bindings come from the `@endpoint(models=…)` decorator argument.
- Model weights resolve from the local CAS first (see *Model fetch + the
  local CAS* below).
- On cache miss the CLI auto-fetches from the upstream registry
  (HuggingFace for `HF` bindings; cozy refs require an
  orchestrator-warmed cache today — see *Cozy ref cache miss* below).

There is **no** `--models` flag and **no** stub mode. The code declares which
model to load. A `Slot` with `selected_by=` is a hub-resolved choice, so
hub-less `run` only ever loads the slot's `default_checkpoint`; naming a model
in the payload is a typed usage error rather than a silent fallback. If you
want to test pre/post-processing without loading a model, write a pytest
against the helpers — that's what pytest is for.

## Output channels

- **stdout** is for **results**. One JSON line per yielded item (generator
  methods) plus a final `{"event":"result", ...}` line. Use `| jq` to
  filter.
- **stderr** is for **events** from `ctx.emit / progress / log`, model-
  fetch progress lines, and tracebacks. One JSON line per event.

This split keeps the result on stdout pipeable while the inner loop's
diagnostics stay visible.

```bash
gen-worker run --payload '{"prompt":"x"}' | jq .value.image
```

## Exit codes

| Code | Meaning                                                                   |
|------|---------------------------------------------------------------------------|
| 0    | success                                                                   |
| 1    | user-code exception (traceback to stderr)                                 |
| 2    | CLI usage / payload validation error                                      |
| 3    | model resolution failure (cache miss + `--offline`, or registry error)    |
| 130  | SIGINT — Ctrl-C (standard shell convention)                               |

## Selecting which method to run

```bash
# Single-class, single-method endpoint — both can be inferred.
gen-worker run --payload '{"prompt":"x"}'

# Pick by class + method explicitly.
gen-worker run --class MyEndpoint --method generate --payload '...'

# `--method` accepts either the Python attribute name or the registered
# function name (the method / function name).
gen-worker run --method marco_polo --payload '{"text":"marco"}'
```

Filtering with `--class` and/or `--method`:

- Exactly one match → that's the one. No flags needed when only one is
  registered.
- Zero matches → exit 2, with the available `Class.method` list printed
  on stderr.
- More than one match → exit 2, "ambiguous; specify --class and/or
  --method".

## Model fetch + the local CAS

The first invocation against a fresh checkout fetches model weights from
the upstream registry. You'll see a stderr line like:

```json
{"kind":"model_fetch.started","ref":"Qwen/Qwen2.5-1.5B-Instruct","provider":"hf"}
{"kind":"model_fetch.completed","ref":"Qwen/Qwen2.5-1.5B-Instruct","provider":"hf","local_dir":"/home/me/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/…"}
```

Subsequent invocations reuse the local cache. The cache locations:

- `HF` bindings → `$HF_HOME` (default `~/.cache/huggingface`).
- `Hub` (tensorhub) bindings → `<$TENSORHUB_CACHE_DIR>/cas`, default
  `/tmp/tensorhub-cache/cas`.

**`TENSORHUB_CACHE_DIR` is the ONE knob for where the CAS lives**
(`models/cache_paths.py`); `tensorhub_cas_dir()` is always
`tensorhub_cache_dir()/cas`. Set it to somewhere durable or every run
re-downloads — `/tmp` gets wiped.

`TENSORHUB_CAS_DIR` is a **different, narrower** setting: it is read only by
the standalone CLI resolver (`models/provision.py`), and the real worker path
(`executor.py`) never looks at it. Setting `TENSORHUB_CAS_DIR` to stop
re-downloading has no effect on a worker run.

Bindings are `HF`, `Hub`, `Civitai`, `ModelScope`, `ModelRef` (`gen_worker`'s
exports). There is no `Repo` class.

### `--offline`

Pass `--offline` to forbid network fetches. If a binding misses the local
cache the CLI exits 3 with the missing ref printed on stderr — useful for
CI and air-gapped iteration once the cache is warm.

```bash
gen-worker run --offline --payload '{"prompt":"x"}'
# stderr: gen-worker run: model resolution failed: --offline: huggingface ref hf:owner/repo not in local cache; warm the cache by running without --offline first.
# exit 3
```

### Cozy ref cache miss

Cozy / tensorhub refs (`Hub("owner/repo")` — no `hf:` / `civitai:`
prefix) use the worker's CAS at `<$TENSORHUB_CACHE_DIR>/cas`. If the requested
snapshot isn't there, the CLI exits 3 with a pointer to invoke the
endpoint via the orchestrator once to populate the cache. This is the
one production-equivalence gap: cozy refs require an orchestrator-
resolved presigned manifest, which is owned by the orchestrator and not
yet wired into the local CLI. HuggingFace refs are fully self-contained.

## SIGINT (Ctrl-C)

`gen-worker run` installs a two-stage SIGINT handler:

- **First Ctrl-C** — trips the request's cancel flag so user code observes
  via `ctx.cancelled` / `ctx.raise_if_cancelled()`. Long-running loops
  inside the function body exit cleanly with `CanceledError`, which the
  CLI translates to exit 130.
- **Second Ctrl-C within 2s** — hard-exits 130 immediately. Useful when
  the function isn't checking for cancellation.

Authors who want to test their cancellation path can press Ctrl-C once
and watch their `raise_if_cancelled()` fire.

## `ctx.save_*` and the local output dir

`ctx.save_bytes(ref, data)` and `ctx.save_file(ref, path)` write under
`./.gen-worker-run/outputs/<ref>` in the cwd. The returned `Asset` has
`local_path` set, so downstream code that reads `asset.local_path` sees
the on-disk path directly — no tensorhub upload.

The `.gen-worker-run/` directory is throwaway. Add it to `.gitignore` /
`.dockerignore`.

## Conversion / dataset endpoints

Checkpoint publishing goes through `gen_worker.convert.publish_flavors(ctx,
flavors)`, which talks to tensorhub directly using the worker capability
token — with no token configured (plain local runs) it fails loudly
instead of pretending to publish. `ConversionContext.materialize_blob`
is stubbed against the local CAS by default; pass `--allow-publish` to
call the real tensorhub API (useful for round-tripping against a dev
tensorhub).

```bash
gen-worker run --payload '{"source":{"ref":"..."},"specs":[...]}' --allow-publish
```

## Persistent dev server — `gen-worker serve` + `gen-worker invoke`

`gen-worker run` reloads the model on **every** invocation — a fresh cold
start per poke (minutes for a real model). `gen-worker serve` boots the
endpoint **once** and serves many requests warm; `gen-worker invoke <fn>`
is the client. One endpoint per `serve` process (matches prod: one worker =
one release). The wire protocol, sidecar, and transports are the versioned
host contract — see [host-integration.md](host-integration.md).

**Transport-fidelity caveat.** Production dispatch is gRPC-from-the-orchestrator.
`serve` mirrors setup, context wiring, memory management, and GPU serialization
faithfully (shared code with `run`), but the **transport** differs (NDJSON over
stdin/UDS locally vs gRPC in prod). That's the right trade for warm-model fast
iteration; byte-for-byte prod fidelity would need the real gRPC Worker against a
local stub-scheduler.

## Ergonomic payload args (#350)

Instead of hand-writing JSON, `run` and `invoke` accept httpie-style
tokens that are **coerced against the function's `msgspec.Struct`** so types
and bounds match the real decode path:

- **`field=value`** — set `field`; value coerced to the field's declared type
  (`seed=42` → int, `hires=true` → bool, `prompt=hi` → str).
- **`field:=<json>`** — raw JSON value, for lists / objects / explicit types:
  `tags:='["a","b"]'`, `size:=1024`.
- **`field@path`** — load the field's value from a file (long prompts, etc.).
- **bare positional** — fills the payload's *primary* field (the first required
  `str` field), so you don't have to name the prompt.
- **`a.b=value`** — dotted key sets a nested object (best-effort coercion).

```bash
gen-worker run "a cat" seed=42 hires=true
gen-worker invoke generate "a cat" seed=42
```

`--payload '<json>'` still works as the escape hatch; ergonomic tokens **merge
over** it.

## When `gen-worker run` is the wrong tool

- **Resource gating.** The CLI doesn't enforce fit-ladder placement: on a
  card too small for the model it happily tries to load and fails inside
  torch instead of degrading through the production offload rungs.
- **Multi-tenant scheduling.** No request queuing, no fairness, no
  micro-batching. One request, sequential dispatch.
- **Cross-machine repro.** Captures local Python + local cache state.
  For shareable repros, ship via the orchestrator.

For all three, run the endpoint through the orchestrator instead.

## See also

- [endpoint-authoring.md](endpoint-authoring.md) — full decorator + binding reference.
