# Cookbook: `async def` Handlers — High-Concurrency I/O Endpoints

One page on when to write a `def` (sync) handler vs an `async def` (async)
handler on the **SerialWorker** archetype, and the pitfalls of getting it
wrong. Added by gen-orchestrator #345 Improvement B.

Cross-links:
- [endpoint-authoring.md](endpoint-authoring.md) — base SDK reference.
- [cookbook-batched-llm.md](cookbook-batched-llm.md) — BatchedWorker
  (continuous-batching engine via `runtime="sglang"|"vllm"`), a *different*
  thing from an async SerialWorker.

---

## TL;DR

```python
@inference()
class LookupEndpoint:
    async def setup(self):
        self.client = httpx.AsyncClient()

    @inference.function(name="lookup")
    async def lookup(self, ctx: RequestContext, payload: LookupInput) -> LookupOutput:
        resp = await self.client.get(f"https://api.example/v1/items/{payload.id}")
        return LookupOutput(value=resp.json()["value"])
```

- An `async def` `@inference.function` **without** `runtime=` is a first-class
  SerialWorker async endpoint. It is **not** a BatchedWorker and **not** a new
  archetype — the SDK detects `async def` via `inspect.iscoroutinefunction` /
  `inspect.isasyncgenfunction` (the function signature is the declaration; there
  is no tenant-facing flag).
- Async handlers run on the worker's shared asyncio loop
  (`_batched_loop`) via `asyncio.run_coroutine_threadsafe`. Thousands of
  coroutines can be in flight at once on a single worker.
- Sync (`def`) handlers continue to run on the `ThreadPoolExecutor` (32 threads
  by default). One request occupies one thread for its whole runtime.

## Which one do I write?

| Workload | Write | Why |
|---|---|---|
| HTTP fan-out, DB reads, file fetches, RPC chains (**I/O-bound**) | `async def` | Coroutine-bound; scales to 10K+ concurrent. A sync handler blocks a thread for the whole I/O wait — caps at ~`threadpool_size / wait_time`. |
| Pure-Python compute, CPU image processing, no GPU (**CPU-bound**) | `def` | The GIL serializes Python compute either way. Async adds no throughput and complicates the code. |
| GPU inference (diffusion, etc.) | `def` | The GPU semaphore + handler runtime dominate; async doesn't help. (`async def` with `await` only inside I/O sections is fine — just no gain.) |

## Streaming (incremental) async handlers

An async handler that streams deltas must be an **async generator** returning
`AsyncIterator[Delta]`:

```python
@inference.function(name="stream")
async def stream(self, ctx: RequestContext, payload: StreamInput) -> AsyncIterator[TokenDelta]:
    async for chunk in self.client.stream(payload.id):
        yield TokenDelta(delta_text=chunk)
```

Validation (enforced at registration): an `async def` generator **must**
annotate `AsyncIterator[Delta]`. A sync `Iterator[Delta]` annotation on an
`async def` generator is rejected — they're contradictory shapes.

## GPU semaphore interaction (#337)

For `accelerator=cuda*` endpoints the per-GPU semaphore is acquired **on the
dispatcher thread, BEFORE the coroutine is scheduled** onto the loop. So:

- The coroutine never holds the GIL while waiting on the semaphore.
- Two GPU handlers can't race on `CUDA_VISIBLE_DEVICES`.
- The semaphore is released after the coroutine completes, on the same thread.

For `accelerator=none` endpoints (the high-concurrency I/O case) there is no
semaphore at all — handlers scale to the full coroutine ceiling.

## Pitfalls

1. **Calling a sync blocking library inside an async handler.** `requests.get`,
   `time.sleep`, a synchronous DB driver, or any CPU-heavy call **blocks the
   shared asyncio loop** — every other in-flight async request on the worker
   stalls. Use the async equivalent (`httpx.AsyncClient`, `asyncio.sleep`,
   async DB drivers) or offload to a thread with
   `await asyncio.to_thread(...)`.
2. **`async def setup` without async handlers.** If `setup` is `async def` but
   handlers are sync, the SDK still runs `setup` on the loop, but the endpoint
   is otherwise sync (ThreadPoolExecutor). That's allowed; just be intentional.
3. **Expecting GPU parallelism from async.** Async does not bypass the GPU
   semaphore — concurrent GPU work is still bounded by the per-GPU slot count.
   Async only helps the *I/O* portions of a handler.
4. **Cancellation.** `ctx.is_canceled()` works in async handlers; you can also
   `await ctx.wait_until_canceled()` for cooperative cancellation. Async
   generators are `aclose()`-d on completion/error so tenant `finally` blocks
   run.
