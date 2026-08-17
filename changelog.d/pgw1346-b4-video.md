# pgw#1346 B4 — the video DiT fleet, and F2 resolved

Split into its own per-lane fragment rather than appended to `pgw1346.md`: ~10 lanes share that
one file, and every edit to it re-serializes the merge queue (two ejections measured). Same
convention `pgw1346-b3a-families.md` adopted.

- **The video DiT fleet is declared: three Wan 2.2 models, LTX-Video 2.3, and MiniMax-H3.**
  `gen_worker.model.catalog` gains `Wan22T2vA14b`, `Wan22I2vA14b`, `Wan22Ti2v5b` and `Ltx23`
  (typed bindings, committed exports) plus an eager `MINIMAX_H3`. **Wan is THREE models, not the
  two the batch plan scoped**: T2V and I2V-A14B publish different `in_channels` (16 vs 36 — 16
  noisy + 4 mask + 16 conditioning, channel-concatenated), TI2V-5B is a different network on five
  axes and takes a per-token float32 timestep where the A14B pair takes one scalar per batch, and
  the hub registers all three separately. A14B's expert pair publishes byte-identical configs, so
  it is ONE graph class run twice — declared as two runners over two counted stages, which states
  the MoE budget (`steps_high`, `steps_low`) that diffusers leaves implicit in a `boundary_ratio`
  threshold. LTX is one joint audio-video DiT: audio is a second token stream through the same
  weights, so `audio_tokens` is a bucket axis rather than a second family, and the two-stage
  distilled recipe is one runner named twice.
- **The "distilled flow-match" scheduler math B4 was told it owed is already implemented.**
  `FlowMatchEulerDiscrete` under a static shift resolves `sigma_i = s*x/(1+(s-1)*x)` over
  `x_i = (N-i)/N`, which is term for term wan-2.2's own `distilled_sigmas` fed through its
  `shifted_sigma`. Asserted against both live-verified ladders: 4 steps at shift 5.0 gives
  `[1000, 937.5, 833.3, 625]` and 8 steps gives all eight. The endpoint had to subclass diffusers
  only because diffusers double-shifts (its `__init__` shifts `sigma_max`/`sigma_min` and
  `set_timesteps` shifts again, landing a 4-step run on t=24). **And UniPC-on-flow-sigmas is not owed
  either — B3-math landed it. B4's scheduler-math column is EMPTY, which is what makes K10 the
  whole blocker: both of a Wan model's solvers are implemented and the declaration still cannot
  carry them.**
- **LTX's literal sigma ladders need no new scheduler machinery either.** `Schedule` is already a
  public frozen dataclass over an explicit sigma tuple; the synthesis from a step count is what
  `FlowMatchEulerDiscrete` adds on top. `ltx23_serve.schedule_from_sigmas` is three lines, and it
  refuses a stamped ladder that carries the terminal zero — a catalog document must not
  double-count a step that does not exist.
- **The three Wan models declare NO scheduler block, and the omission is the honest reading.**
  One Wan model serves two solvers selected by which adapter a request attaches: the checkpoint's
  own UniPC (`use_flow_sigmas`, `flow_prediction`, `solver_order=2`, `bh2`) for the base lineage,
  flow-match Euler on the trained uniform ladder for the distilled one. A single-valued
  `Scheduler` block would name one and serve the other lane a schedule it was not trained on. This
  is pgw#1346 K10 recurring on a stronger case than B2's — no payload enum is involved.
- **Codegen no longer imports `MappingProxyType` into a scheduler-less binding.** It has exactly
  one use (freezing `SCHEDULER_PARAMETERS`); the three Wan models are the catalog's first families
  that declare no scheduler, so nothing had surfaced it. No existing export digest moves.
- **K10 is now a declaration limit, not a missing implementation.** Both of Wan's solvers are
  implemented on this branch — `unipc_multistep` from B3-math and `flow_match_euler_discrete`
  which B4 proves reproduces the distilled ladder exactly — and the three declarations still
  cannot carry them, because `GraphModelSpec.scheduler` is ONE block and codegen emits ONE
  `scheduler()`. A family whose CHECKPOINT chooses the solver has nowhere to put the second one.
- **The runner -> component map W1b-2 said did not exist now does, and A14B is the family it was
  missing for.** `Runner(component=)` on the Wan declarations resolves the two expert runners to
  `transformer` and `transformer_2` — which is what turns "the expert pair is two weight sets over
  one graph class" from a comment into the module an eager backing reaches. LTX names
  `transformer`, matching the endpoint's own `Compile(targets=("transformer",))`. `component` is a
  SERVING fact and is not exported, asserted twice: the committed documents are byte-unchanged,
  and the export JSON contains no component path at all.
- **`MinimaxH3` is a type.** B5's eager-tier codegen gives the eager declaration a real `Model`
  subclass carrying the tuned schema and nothing else — no bucket literals, no runner callables,
  no `scheduler()` — so an endpoint can annotate a handler parameter with it without claiming a
  composition it does not have.
