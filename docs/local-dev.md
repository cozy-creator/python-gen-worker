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
   when the endpoint has exactly one `@inference.function` method.
2. **What payload to send** — JSON, inline (`--payload '...'`) or from a
   file (`--payload-file ./fixture.json`). Validated against the function's
   `msgspec.Struct` input type.

Everything else derives from the code:

- Model bindings come from the `@inference(models=…)` decorator argument.
- The payload's `_models` field can override any binding that declared
  `.allow_override(...)`, exactly the way it does in production.
- Model weights resolve from the local CAS first
  (`$TENSORHUB_CAS_DIR`, default `/tmp/tensorhub-cache/cas`).
- On cache miss the CLI auto-fetches from the upstream registry
  (HuggingFace for `HFRepo` bindings; cozy refs require an
  orchestrator-warmed cache today — see *Cozy ref cache miss* below).

There is **no** `--models` flag and **no** stub mode. The code declares
which model to load; the payload can override it. If you want to test
pre/post-processing without loading a model, write a pytest against the
helpers — that's what pytest is for.

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
# function name (the @inference.function(name=...) value).
gen-worker run --method marco_polo --payload '{"text":"marco"}'
```

Filtering with `--class` and/or `--method`:

- Exactly one match → that's the one. No flags needed when only one is
  registered.
- Zero matches → exit 2, with the available `Class.method` list printed
  on stderr.
- More than one match → exit 2, "ambiguous; specify --class and/or
  --method".

## Payload overrides via `_models`

The reserved `_models` field on the payload lets you swap any binding
that declared `.allow_override(...)` for an arbitrary ref. Same shape as
production:

```bash
# String shorthand: owner/repo[:tag][#flavor]
gen-worker run --payload '{
  "prompt": "x",
  "_models": {"pipe": "other/repo:canary#bf16"}
}'

# Structured form.
gen-worker run --payload '{
  "prompt": "x",
  "_models": {"pipe": {"ref": "other/repo", "tag": "prod", "flavor": "bf16"}}
}'
```

If the binding has no `.allow_override(...)`, the CLI rejects the
override with exit 2 — matches production's `model_override_not_allowed`
error.

## Model fetch + the local CAS

The first invocation against a fresh checkout fetches model weights from
the upstream registry. You'll see a stderr line like:

```json
{"kind":"model_fetch.started","ref":"Qwen/Qwen2.5-1.5B-Instruct","provider":"hf"}
{"kind":"model_fetch.completed","ref":"Qwen/Qwen2.5-1.5B-Instruct","provider":"hf","local_dir":"/home/me/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/…"}
```

Subsequent invocations reuse the local cache. The cache locations:

- `HFRepo` bindings → `$HF_HOME` (default `~/.cache/huggingface`).
- `Repo` (tensorhub) bindings → `$TENSORHUB_CAS_DIR` (default
  `/tmp/tensorhub-cache/cas`).

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

Cozy / tensorhub refs (`Repo("owner/repo")` — no `hf:` / `civitai:`
prefix) use the worker's CAS at `$TENSORHUB_CAS_DIR`. If the requested
snapshot isn't there, the CLI exits 3 with a pointer to invoke the
endpoint via the orchestrator once to populate the cache. This is the
one production-equivalence gap: cozy refs require an orchestrator-
resolved presigned manifest, which is owned by the orchestrator and not
yet wired into the local CLI. HuggingFace refs are fully self-contained.

## SIGINT (Ctrl-C)

`gen-worker run` installs a two-stage SIGINT handler:

- **First Ctrl-C** — flips `ctx._canceled` so user code observes via
  `ctx.is_canceled() / ctx.raise_if_canceled()`. Long-running loops
  inside the function body exit cleanly with `CanceledError`, which the
  CLI translates to exit 130.
- **Second Ctrl-C within 2s** — hard-exits 130 immediately. Useful when
  the function isn't checking for cancellation.

Authors who want to test their cancellation path can press Ctrl-C once
and watch their `raise_if_canceled()` fire.

## `ctx.save_*` and the local output dir

`ctx.save_bytes(ref, data)` and `ctx.save_file(ref, path)` write under
`./.gen-worker-run/outputs/<ref>` in the cwd. The returned `Asset` has
`local_path` set, so downstream code that reads `asset.local_path` sees
the on-disk path directly — no tensorhub upload.

The `.gen-worker-run/` directory is throwaway. Add it to `.gitignore` /
`.dockerignore`.

## Conversion / dataset endpoints

`ConversionContext.publish_repo_revision` and `materialize_blob` are
stubbed by default — they print the would-be call to stderr and return
a fake response. Pass `--allow-publish` to call the real tensorhub APIs
(useful for round-tripping against a dev tensorhub).

```bash
gen-worker run --payload '{"source":{"ref":"..."},"specs":[...]}' --allow-publish
```

## Worked example — marco-polo

```bash
$ cd examples/marco-polo
$ gen-worker run --payload '{"text":"marco"}'
{"event":"result","value":{"response":"polo"}}

$ gen-worker run --payload '{"text":"hello"}'
{"event":"result","value":{"response":"Bro you're supposed to say 'marco'!"}}

$ gen-worker run --payload '{"text":42}'
gen-worker run: payload validation failed: Expected `str`, got `int` - at `$.text`
# exit 2

$ gen-worker run --payload '{"text":"marco"}' | jq -r .value.response
polo
```

## Worked example — streaming generator

```python
@inference.function
def stream(self, ctx, data: Input) -> Iterator[Delta]:
    for word in data.text.split():
        yield Delta(chunk=word)
```

```bash
$ gen-worker run --payload '{"text":"alpha beta gamma"}'
{"event":"yield","value":{"chunk":"alpha"}}
{"event":"yield","value":{"chunk":"beta"}}
{"event":"yield","value":{"chunk":"gamma"}}
{"event":"result","value":{"yielded":3}}
```

## Worked example — `--module` override

If `endpoint.toml` is missing or you want to invoke a sibling module:

```bash
gen-worker run --module my_pkg.alt_main --payload '{"prompt":"x"}'
```

## Persistent dev server — `gen-worker serve` + `gen-worker invoke`

`gen-worker run` reloads the model on **every** invocation — a fresh cold
start per poke (minutes for a real model). For tight local iteration use
`gen-worker serve`: it boots the endpoint **once** (imports `main`, runs
`setup()` per class, holds the instances + loaded models VRAM-resident), then
serves many requests warm. Ctrl-C tears down (`shutdown()` if present, socket
removed, exit 0).

One endpoint per `serve` process (matches prod: one worker = one release).
`--config PATH` serves an endpoint outside the cwd; run several serves with
distinct `--socket` paths to host multiple endpoints at once.

Two transports, **one shared dispatch handler** (the same code path `run`
uses):

- **Unix domain socket (always on):** `serve` listens on `./.gen-worker.sock`
  (override `--socket PATH`). Fire requests from another shell with
  `gen-worker invoke`:

  ```bash
  # terminal 1
  $ cd examples/marco-polo
  $ gen-worker serve
  gen-worker serve: listening on .../.gen-worker.sock (functions: marco_polo)
  gen-worker serve: ready

  # terminal 2 — address by FUNCTION NAME (no --class/--method)
  $ gen-worker invoke marco_polo '{"text":"marco"}'
  {"response":"polo"}
  $ gen-worker invoke marco_polo @req.json          # curl-style @file
  $ echo '{"text":"marco"}' | gen-worker invoke marco_polo -   # stdin
  ```

  If launching `serve` in the background, pass `--no-stdin` so it doesn't
  consume the parent shell's stdin.

- **stdin/stdout NDJSON (default, single process):** pipe a batch of
  newline-delimited JSON requests in; get one NDJSON response line each. Logs
  go to stderr. The process exits when stdin closes.

  ```bash
  $ printf '{"function":"marco_polo","payload":{"text":"marco"}}\n' \
      | gen-worker serve
  {"ok":true,"events":[{"event":"result","value":{"response":"polo"}}]}
  ```

**Wire format** (symmetric between the two transports):

- request:  `{"function": "<fn_name>", "payload": <json>}`
- response: `{"ok": true, "events": [{"event":"result","value":...}, ...]}`
            or `{"ok": false, "error": {"kind":"...","message":"..."}}`

**Transport-fidelity caveat.** Production dispatch is gRPC-from-the-orchestrator.
`serve` mirrors setup, context wiring, memory management, and GPU serialization
faithfully (shared code with `run`), but the **transport** differs (NDJSON over
stdin/UDS locally vs gRPC in prod). That's the right trade for warm-model fast
iteration; byte-for-byte prod fidelity would need the real gRPC Worker against a
local stub-scheduler.

## The three shapes

`gen-worker` gives you three ways to run an endpoint locally, differing in
process model and how often the model loads:

| Shape | Process model | When to use |
|---|---|---|
| `gen-worker run` | One-shot: load → run → exit (one process). | Scripting, CI, one-off pokes, piping to `jq`. Cold-loads each time (unless a warm `serve` is already up on the default socket — then `run` attaches to it). |
| `gen-worker serve` + `gen-worker invoke` | Two processes / terminals: a warm worker + a thin client. | Tight iteration; this is the production/Docker topology (long-running worker, requests fired at it). |
| `gen-worker repl` | One process: load once, then interactive. | Interactive exploration with the model held resident; type many requests at a prompt. |

## Ergonomic payload args (#350)

Instead of hand-writing JSON, `run`, `invoke`, and `repl` accept httpie-style
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
over** it. In `repl`, tokens are split shell-style, so a multi-word value must
be quoted just like in a shell (`"a cat" steps=5`).

## `gen-worker repl`

A load-once interactive session: it boots the endpoint (eager `setup()`, model
resident) and loops over typed requests, reusing the same engine as `serve`.

```bash
$ cd examples/marco-polo
$ gen-worker repl
gen-worker repl — marco_polo.main (1 function(s): marco_polo)
type ':help' for commands, ':quit' to exit
active function: marco_polo
marco_polo> "marco"
{"response":"polo"}
```

Each line is either ergonomic tokens / raw JSON (a request to the active
function) or a meta-command:

| Command | Action |
|---|---|
| `:use <fn>` | switch the active function |
| `:functions` | list functions |
| `:schema` | print the active function's input JSON Schema |
| `:help` | show help |
| `:quit` (or `:q`) | exit |

**Ctrl-C** cancels the in-flight request and returns to the prompt; **Ctrl-D**
exits (and runs `shutdown()`).

## Deployment shapes (Docker / k8s)

> In real production the worker container's entrypoint is the **gRPC worker**
> that dials the orchestrator. `serve` and `run` are the **local / self-hosted**
> shapes — handy for one-off jobs and self-managed deployments.

### `run` = one-off Job

A single load-run-exit invocation. Mount the CAS volume so weights are reused,
and the exit code propagates out of the container:

```bash
docker run --rm \
  -v ~/.cache/cozy:/cache -e TENSORHUB_CAS_DIR=/cache \
  <img> gen-worker run generate "a cat" seed=42
```

Maps cleanly to a Kubernetes **Job** / **CronJob** (exit 0 = success).

### `serve` = long-running service

A warm, always-on worker. Run detached, listen on TCP for cross-container
access, and submit work either via `docker exec` or over a published port:

```bash
docker run -d -p 8731:8731 \
  -v ~/.cache/cozy:/cache -e TENSORHUB_CAS_DIR=/cache \
  <img> gen-worker serve --listen tcp://0.0.0.0:8731

# submit via exec (shares the container's unix socket)…
docker exec <ctr> gen-worker invoke generate "a cat" seed=42
# …or over the published TCP port from the host:
gen-worker invoke generate "a cat" --socket tcp://localhost:8731
```

Maps to a Kubernetes **Deployment** (one endpoint per Pod, mirroring prod's
one-worker-per-release model).

## Cancellation & signals (#352 / #353)

Cancelling a request and stopping the worker both funnel through the same
`ctx.cancel()` — so tenant code observes them identically, in prod and locally.

- **`run` / `invoke` client, Ctrl-C:** cancels **that request** on the server;
  the warm `serve` stays running. A **second** Ctrl-C within 2s detaches the
  client (exit 130) — the request may still finish server-side.
- **`serve` terminal, Ctrl-C / SIGTERM:** cancels **all** in-flight requests,
  drains them cooperatively, then shuts down (runs `shutdown()`, removes the
  socket). SIGTERM (k8s/orchestrator graceful stop) takes the identical path.
  `SIGKILL` is uncatchable — it bypasses all cleanup.
- **`repl`, Ctrl-C:** cancels the in-flight request and returns to the prompt;
  Ctrl-D exits.

Tenant code should observe cancellation by calling `ctx.raise_if_canceled()`
inside loops, or by waiting on `ctx.cancel_event` — the **same idiom**
production uses for an orchestrator interrupt.

## When `gen-worker run` is the wrong tool

- **Resource gating.** The CLI doesn't enforce VRAM / compute-capability
  gates. If your endpoint declares `min_vram_gb=80` and you're on a 24GB
  card, the CLI happily tries to load the model and fails inside torch.
- **Multi-tenant scheduling.** No request queuing, no fairness, no
  micro-batching. One request, sequential dispatch.
- **Cross-machine repro.** Captures local Python + local cache state.
  For shareable repros, ship via the orchestrator.

For all three, run the endpoint through the orchestrator instead.

## See also

- [endpoint-authoring.md](endpoint-authoring.md) — full decorator + binding reference.
- [endpoint-toml.md](endpoint-toml.md) — `endpoint.toml` schema.
- [cookbook-image-diffusion.md](cookbook-image-diffusion.md) and friends — per-modality recipes.
