- **The entrypoint dispatch loop lands (`gen_worker.serving.serve_loop` +
  `envelope`).** The wire-facing serving path over the pgw#1382 split:
  the SIGNATURE-DERIVED request envelope (`{"model"/"models", "adapters":
  {"turbo", "loras"}, "input"}` — model picks keyed by slot name, adapter
  rows decoded into typed `Adapter`/`DistillationAdapter` values with the
  worker-side takeover guard re-asserted) decodes per entrypoint; every
  invocation runs under `ResidencyManager` leases taken in STABLE SLOT-NAME
  order (admission before allocation, LRU eviction between requests,
  serialized loads, single-flight per instance); the call is ctx-first and
  `ctx.warn` rows ride the `InvokeOutcome`. Instances are keyed
  (model class x checkpoint x lane) and reused across requests; deploy state
  arrives through the `BindingResolver` seam. `python -m gen_worker.serving
  --envelope` serves the full production wire shape locally with zero hub.
