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
