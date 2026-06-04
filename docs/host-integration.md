# Host integration contract

This is the stable contract a **host orchestrator** (cozy-local, a Go CLI)
integrates against when it drives `gen-worker` — over the CLI and over the
`serve` socket. Everything here is machine-readable and versioned. A host
should **never** scrape `--help` or guess wire shapes; it keys behavior off
the `protocol_version` and `capabilities` carried in `describe --json` and the
serve sidecar.

## Versioning

Every machine-readable surface (`describe --json`, the serve ready-sidecar)
carries two version markers:

- **`protocol_version`** — an integer, currently **`1`**. Bumped on any
  incompatible change to the request/response/cancel frame shapes, the
  `describe` document, or the serve sidecar.
- **`capabilities`** — a list of feature tokens the build actually ships. A
  host that wants to use an optional feature must check for its token here, not
  infer support from a version number or `--help` text.

Current capabilities:

| Token | Meaning |
|---|---|
| `describe` | `gen-worker describe --json` |
| `list_functions` | `gen-worker serve --list-functions [--json]` |
| `prefetch` | `gen-worker prefetch` |
| `cancel` | per-request `{"cancel":{"request_id"}}` control frame |
| `streaming` | multi-frame streamed responses (`{"stream":true}`) |
| `tcp_listen` | `serve --listen tcp://host:port` |
| `serve_sidecar` | machine-readable `.gen-worker.serve.json` handle |

Rule of thumb: **gate every behavior off `capabilities`.** New tokens are
added when a feature ships; absence means "not available in this build."

## `gen-worker describe --json`

Introspect the endpoint and emit a stable JSON document **without loading any
model**. This is how a host learns an endpoint's functions, schemas, and model
bindings before booting anything.

```bash
gen-worker describe --json            # compact, single line
gen-worker describe --pretty          # newlines + 2-space indent
gen-worker describe --module my.main  # override endpoint.toml `main`
```

Document shape:

```json
{
  "protocol_version": 1,
  "gen_worker_version": "0.8.0",
  "capabilities": ["describe", "list_functions", "prefetch", "cancel",
                   "streaming", "tcp_listen", "serve_sidecar"],
  "endpoint": {
    "main_module": "myendpoint.main",
    "kind": "inference",
    "classes": ["MyEndpoint"]
  },
  "functions": [
    {
      "name": "generate",
      "class": "MyEndpoint",
      "method": "generate",
      "kind": "inference",
      "is_generator": false,
      "input_schema": { "type": "object", "properties": { "...": {} } },
      "output": "Output",
      "models": { "pipe": { "type": "HFRepo", "provider": "hf", "ref": "..." } }
    }
  ]
}
```

Notes:

- `endpoint.kind` is a single string when all functions share one kind,
  otherwise a list.
- **`input_schema`** is JSON Schema derived from the function's
  `msgspec.Struct` payload type. msgspec emits a top-level
  `{"$ref": "#/$defs/X", "$defs": {...}}`; gen-worker **inlines that top-level
  ref** so `input_schema["properties"]` is directly available (a host builds
  field prompts off it). Nested struct references stay under `$defs`.
- **`models`** maps each model param to its binding descriptor — `Repo` /
  `HFRepo` / `CivitaiRepo` carry `provider`/`ref`/`tag`/`flavor`/
  `allow_override`; a `Dispatch` binding carries `field` + a `table` of
  per-key bindings.

`serve --list-functions --json` is a **thin alias** — it emits
`{"functions": [...]}` using the identical per-function builder, so the array
matches `describe`'s `functions` field exactly:

```bash
gen-worker serve --list-functions --json   # {"functions":[...]} — no model load
gen-worker serve --list-functions          # text listing: name (Class)
```

## NDJSON IPC protocol

A running `serve` speaks newline-delimited JSON (NDJSON) over its listener
(Unix socket or TCP) — and over its own stdin/stdout when run interactively.
One frame per line.

### Request frame

```json
{"request_id": "abc123", "function": "generate", "payload": { }, "stream": false}
```

- `function` (string, required) — the routable `@inference.function(name=...)`.
- `payload` (object) — the decoded JSON payload; the reserved `_models` field
  inside it overrides bindings exactly as in production.
- `request_id` (string) — required for cancellation to work; echoed back on
  responses.
- `stream` (bool, optional) — request streamed frames (see below).

### Cancel control frame

Send on **any** connection — it does not need to be the connection the request
came in on:

```json
{"cancel": {"request_id": "abc123"}}
```

The server looks the id up in its in-flight registry and trips `ctx.cancel()`
for that request. The server keeps running; only that one request is canceled.
The cancel frame is acked with `{"ok": true, "canceled": <bool>, "request_id": ...}`
(`canceled` is `false` if no matching in-flight request was found).

### Non-streaming response

The default. The server buffers all events and returns one terminal envelope:

```json
{"ok": true, "events": [{"event": "result", "value": {}}], "request_id": "abc123"}
```

Generator functions produce one `{"event":"yield","value":...}` per item
followed by a final `{"event":"result","value":{"yielded":N}}`. On failure:

```json
{"ok": false, "error": {"kind": "user_exception", "message": "..."}, "request_id": "abc123"}
```

### Streaming response (`stream: true`)

The server writes one frame per event **as produced**:

```json
{"event": "yield",   "value": {}, "request_id": "abc123"}
{"event": "result",  "value": {}, "request_id": "abc123"}
```

then a single terminal frame:

```json
{"ok": true, "done": true, "request_id": "abc123"}
```

A streaming error still comes back as `{"ok": false, "error": {...}}`.

### Error kinds

| `kind` | Cause |
|---|---|
| `usage` | bad request / payload validation / unparseable frame |
| `not_found` | no function with that name |
| `model_resolution` | a model binding could not be resolved/fetched |
| `canceled` | the request was canceled (cancel frame or client disconnect) |
| `user_exception` | the tenant handler raised |

## Serve sidecar — `.gen-worker.serve.json`

On ready, `serve` writes a machine-readable handle so a host reads pid / socket
/ ready-state instead of guessing. It is removed on teardown.

Location:

- **Unix socket:** next to the socket file, at `<socket-path>.json`.
- **TCP:** in the current working directory, at `.gen-worker.serve.json`.

Shape:

```json
{
  "protocol_version": 1,
  "gen_worker_version": "0.8.0",
  "pid": 12345,
  "listen": "/abs/path/.gen-worker.sock",
  "ready_at": 1733356800.0,
  "idle_timeout": 0.0,
  "functions": ["generate", "describe_thing"]
}
```

`listen` is the display form (a Unix path, or `tcp://host:port`). A host should
read this file to discover where to connect and confirm the worker is ready —
its presence is the readiness signal.

## Transports

The default transport is a **Unix domain socket** (same host / same container).
For cross-process / cross-container topologies, switch to **TCP** — the same
NDJSON protocol rides over it.

```bash
# server
gen-worker serve                              # unix: ./.gen-worker.sock
gen-worker serve --socket /run/gw.sock        # unix: custom path
gen-worker serve --listen tcp://0.0.0.0:8731  # TCP

# client
gen-worker invoke generate "a cat"                          # default unix sock
gen-worker invoke generate "a cat" --socket /run/gw.sock    # custom unix path
gen-worker invoke generate "a cat" --socket tcp://host:8731 # TCP
```

Address forms accepted everywhere: `tcp://host:port`, `unix:///abs/path`, or a
bare path (treated as a Unix socket).

## Cancellation

Two paths, both funneling through the canonical `ctx.cancel()`:

- **Explicit:** a `{"cancel":{"request_id"}}` frame on any connection trips
  `ctx.cancel()` for that in-flight request. The server stays up.
- **Client disconnect (backstop):** if a streaming client's connection drops
  mid-response, the server cancels that request automatically.

Tenant code observes cancellation cooperatively — via
`ctx.raise_if_canceled()` in loops, or by waiting on `ctx.cancel_event`. This
is the **same idiom** in production (orchestrator interrupt) and locally.

## Exit codes

The CLI client commands (`run`, `invoke`) propagate a stable exit-code matrix;
a host can branch on it directly.

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | user-code exception |
| 2 | usage / payload validation error |
| 3 | model resolution failure |
| 130 | SIGINT (Ctrl-C) |

`invoke` maps the server-side `error.kind` onto this same matrix (`usage` /
`not_found` → 2, `model_resolution` → 3, `canceled` → 130, everything else → 1).

## See also

- [local-dev.md](local-dev.md) — the three CLI shapes (`run` / `serve` +
  `invoke` / `repl`), ergonomic payload args, and deployment topologies.
- [endpoint-authoring.md](endpoint-authoring.md) — decorator + binding reference.
