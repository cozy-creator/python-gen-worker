- **pgw#1346 K10: the declaration carries a scheduler SET keyed by the tuned sampler, and
  `euler_a` / `ddim` are implemented.** B2 closed with a structural blocker rather than a
  budget one: the sampler is a CHECKPOINT value (`inst.tuned.scheduler`, six names on sdxl and
  nine on sd15, over four diffusers classes) while `GraphModelSpec.scheduler` was ONE block,
  so a second implemented kind would have been a class no declaration could attach to. The
  live consequence it measured: **SDXL's DEFAULT sampler is `euler_a`**, so the family's single
  declared `euler_discrete` block was its TRAINED schedule and not the one most requests ask
  for. `GraphModelSpec.schedulers` is now `{sampler: Scheduler(kind, block)}`; the export
  records a sorted `schedulers[]` array of `(sampler, name, parameters)`; codegen emits
  `SCHEDULERS` + per-sampler `SCHEDULER_PARAMETERS` and a `scheduler()` that resolves
  `inst.tuned.scheduler` to a concrete class over a closed union, with an optional `name=`
  typed by a generated `…DeclaredSampler` Literal. A family with ONE sampler declares a set of
  one and its accessor keeps the exact previous shape — no argument, one concrete return type —
  so the single-scheduler migration (flux1_dev) is mechanical and B2's held endpoint migration
  needs nothing from this change. **An undeclared stamp is `SCHEDULER_UNDECLARED`, never a
  substitution**: serving the family's other schedule under the requested sampler's name is a
  plausible wrong image and nothing else. New math, at B2's final instrument (relative 2e-4 vs
  diffusers, timesteps EXACT, loops compared in the L2 norm with our ladder injected, and our
  own ladder byte-identical under `ATEN_CPU_CAPABILITY=default`): `EulerAncestralDiscrete`
  (same ladder as `euler`, stochastic step, **noise is a required parameter** — defaulting it
  would be a silently different sampler) and `Ddim` (walks ALPHAS, not sigmas — `init_noise_sigma`
  1.0 and `scale_model_input` the identity, so it is its own schedule type). Both loops measured
  BIT-EXACT against diffusers with the table removed as a variable. Three sharp findings carved
  into tests: `DDIMScheduler` does **not** clamp the terminal alpha under zero-terminal-SNR where
  the euler family does; DDIM ROUNDS its linspace grid where the euler family interpolates at the
  fractional position; and a block carrying a parameter its kind never reads is now REFUSED
  (`EulerAncestralDiscrete` has no `final_sigmas_type`) rather than ignored. The per-step
  ancestral noise is CPU-seeded and keyed by `(seed, step index)` through splitmix64's finalizer —
  reproducible across pods AND across loop shapes, where an advancing generator is only the
  former. **B3-math landed first and is ADDITIVE to the set**, so the staging shrank to one name:
  `dpmsolver_multistep` and `unipc_multistep` slot in as two more members with no math change,
  and **sd15's own default `dpmpp_2m_karras` is servable** — five samplers on sdxl and eight on
  sd15/sd2, every one of them rendered end to end through the real `generate` in the suite. The
  loop dispatches on the SCHEDULE TYPE and not on the name, because the differences are
  load-bearing: a multistep solver carries history between steps and is NOT pre-scaled (its
  `init_noise_sigma` is 1.0, so dividing its latents by `sqrt(sigma**2+1)` is a wrong image and
  never an error), and `sde-dpmsolver++` consumes the same keyed noise `euler_a` does. The
  declared blocks are asserted against `gen_worker.view.SAMPLERS`, which already defines each
  sampler name completely for the `Slot`-served endpoints, rather than restated from memory.
  **`lcm` is the one name still owed** — `LCMScheduler` has no module in `model/solvers/` — and a
  checkpoint stamped with it refuses by name. B3's two solvers also gained the same
  unread-parameter refusal.
