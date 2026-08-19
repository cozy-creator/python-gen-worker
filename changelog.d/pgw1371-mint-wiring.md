- **A serving worker now mints its own compiled graphs.** `serving/mint.py` shipped complete and
  with **no production caller** — every reference to `mint_holes` at HEAD was a definition or a
  re-export — so no pod could mint anything and the typed `self_mint_wedged` /
  `self_mint_arm_missed` events pgw#985 landed were unreachable in production. The caller is
  `serving/self_mint.py`: it reads the adopt session's ordered hole list, mints on a daemon thread
  off the request path, publishes and arms each graph as it lands, and reports counted progress on
  a `self_mint_compile` activity carrying `compile:self_mint_graphs`.

- **The trigger is the post-load hook, and the earlier answer was measured wrong.** Holes are
  registered by the author's own `ctx.compile` calls INSIDE `Model.load(ctx)`, so the first instant
  the work-list is whole is the instant that load returns — `ServeLoop` gained `on_loaded` for
  exactly that. Arming at the `compile_sink_for` handover (the obvious place) reported
  `nothing_to_mint` on a pod with two real holes. Not a request-count trigger and not a delay: a
  pre-warm pod gets no traffic, and a clock against work is the anti-pattern this tier already paid
  for twice. Contention is handled where it was measured — `entry_workers <= vcpus - reserve` plus
  a niced mint tree.

- **Two more wiring gaps found on the way, both of which made the mint unreachable anyway.**
  (1) `Worker` built its `ServeLoop` with `compile_sink_for=None`, so **every `ctx.compile` on every
  real serving pod was a transparent pass-through** — pgw#1372's adopt-first boot existed only in
  `python -m gen_worker.serving`, never on a pod. `serving/serve_adoption.py` is the missing
  construction: th#2133's answer → `HubGraphStore` → `AdoptSession`, refusals typed and never fatal.
  (2) `HubGraphStore.publish_artifact` REFUSES by construction, so a mint handed the boot store
  would have failed every hole. `serving/mint_store.py` makes the store two-tiered — local tensorfs
  CAS first (durable, cannot refuse, so a restarted pod adopts what it already minted), hub second —
  and the fleet leg's absence is one typed `self_mint_publish_local_only` event naming
  pgw#1368/th#2132, never a per-graph failure.

- **`HubGraphStore.get_graphs` stops swallowing `ReleaseNotStamped`.** Flattening the typed 404 into
  `None` erased the only distinction it carries — "no adopt story, serve eager" versus "answered and
  rebuilt to nothing" — and made both callers' handlers dead code.

- Driven end to end on CPU with a fake compile callable and real everything else: worker A boots onto
  an unminted (lane x sm), registers 2 holes in document order, mints both, publishes both, arms both
  onto its live dispatch and serves through them with no reboot; worker B boots on the same store with
  0 holes and **0 compile invocations**. A hanging compile is condemned and emits `self_mint_wedged`
  rather than hanging. `not_armed` / `nothing_to_mint` / `unavailable` are three distinct states, none
  of which renders as a finished mint.
