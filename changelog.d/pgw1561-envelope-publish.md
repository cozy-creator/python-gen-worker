- **pgw#1561: the v2 artifact band was NEVER loadable by adoption — the
  publisher banked the bare `model.pt2` ZIP, the boot loader reads the tar+gzip
  envelope.** Latent since publishing existed (pgw#1471); va#3 arm 2 was the
  first real-loader exercise and holed 14/14 as `cannot decompress`, minutes
  after `SERVABLE` was printed over the same store. `publish_compiled` now
  repacks the ENVELOPE from the unpacked artifact (deterministic, validated
  against the package, carrying the metadata and literals the bare ZIP
  discarded — and byte-identical to the engine cache's object, so the CAS
  dedups the bands). torchcg tcg#75: a ZIP in the envelope band raises typed
  `ArtifactFormatSkew` (re-publish, not re-mint) and an envelope REPLACES a
  skewed incumbent at publish. `compile` treats a present-but-skewed position
  as a miss so a warm run repairs it, its census probes every position's shape
  (two magic bytes), and the pgw#1533 read-back now MATERIALIZES fetched bytes
  through the real loader — presence-only certification is what let this ship.
  The RUNTIME-mint seam now round-trips too: `_mint_one` publishes, and the test
  fetches those bytes back through the real store and opens them with
  `torchcg.serve.materialize`. That seam arms the unpacked directory it just
  wrote, so nothing it publishes was ever read back — the gap the field found.
