## pgw#1664 — `sensenova-u1` enters the vocabulary, and a lane refusal learns to name the corpus that answered

- **`SenseNovaU1` / `SenseNovaU1Defaults`.** SenseTime's NEO-unify MoT family —
  understanding and generation in ONE set of weights, so text-to-image and
  reference-image editing serve under one root, the same shape as `QwenImage`.
  No `canonical_scheduler_config`: there is no scheduler and no diffusers
  pipeline to synthesize one for, because the trajectory is a shifted linspace
  built inside the sampler and the model is VAE-free — the flow state IS the
  image.

- 🔴 **Two of the platform values disagree with the checkpoint's own
  `config.json`, and the config is the one that is wrong.** `timestep_shift` is
  **3.0**, not the config's 1.0: every upstream example passes 3.0 and their
  `time_schedule` is hard-forced to `"standard"`, so the config field is dead
  and copying it would silently walk a different trajectory on every request.
  `guidance` 4.0 is a REAL classifier-free CFG — a second prefix over the empty
  prompt and a second KV cache — not a guidance embedding, so the arm count is a
  serving fact and never a batch axis. `img_guidance` 1.0 is the EDIT arm's
  second axis: above 1.0 a third images-only prefix is built and the denoise step
  runs three times per timestep.

- **A lane refusal now names the corpus that said no.** `lanes=` is checked
  against the VENDORED tensorfs snapshot and never the PEP 508 `tensorfs` pin, so
  an author who pins tensorfs at the exact commit that added their record is
  still refused — correctly, and with no way to tell "fix your declaration" from
  "wait for a re-vendor". `_stamp_half`'s not-registered message carries
  `gen_worker._vendor.tensorfs @ <rev>` and says the pin is not consulted. That
  is precisely how this issue's own blocker read for a day.
