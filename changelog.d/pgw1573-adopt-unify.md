- **pgw#1573: the runtime mint arms the bytes the STORE hands back, never the
  directory the compiler wrote.** `_mint_one` published a tar+gzip ENVELOPE and
  then armed `Engine.compile`'s unpacked destination, so no run of the mint had
  ever read its own published bytes. That is the seam pgw#1471's bare-`.pt2`
  publish hid behind from the first publisher that ever existed until va#3
  arm 2 fetched one on a pod and holed 14/14 on `cannot decompress` — every
  local green was green over the directory. The mint now publishes, FETCHES
  BACK through `fetch_artifact`, and arms that, at the same position boot
  adoption fetches to: a self-minted artifact and a hub-fetched artifact
  traverse ONE load path, so a publish/load format skew breaks on the machine
  that made it, in seconds. A publish the store cannot hand back is the typed
  `MintNotServable`. A storeless mint is refused at construction — it had
  exactly one thing left to arm, and that thing was the second load path.

- **pgw#1573: a fleet-pool hit is BANKED into the local CAS on the way past.**
  `TieredGraphStore.fetch_artifact` returned the hub's bytes and kept none of
  them, so a pod re-downloaded every artifact on every restart and a hub outage
  turned a warm box cold. "Check local, then remote" is half a cache if the
  remote answer is never kept. Banking is best-effort — the bytes are already
  verified and in hand, so a failure costs one more download, never an arm —
  and an answer carrying no requirements manifest is deliberately NOT banked,
  because a cached artifact with no stated floors would be adoptable on a
  machine the fleet answer would have refused.

- **pgw#1573: ONE store constructor, `mint_store.graph_store` (was
  `worker_store`).** Six call sites built a store six ways — three of them a
  bare `LocalGraphStore` with no hub tier and no baked tier — so "do I have
  this graph" had three answers depending on which entry point asked. A pod,
  `gen-worker up`, `gen-worker compile` and the CI runner now build the same
  object; `upstream=None` is the stated difference between a box and a pod, not
  a different class. `TieredGraphStore` gained the `artifact_skew` passthrough
  pgw#1561 left owed, so a skewed local position is a miss for every reader
  instead of only for the compile CLI.

- **pgw#1573: `gen-worker compile`'s `FETCHED` outcome is deleted, along with
  the "FETCH-FIRST … the hub is asked before anything is built" paragraph that
  documented it.** `_store` passed `upstream=None` and `FETCHED` was defined
  and never assigned anywhere in `src/`. A vocabulary member nothing can emit
  is a proof condition that passes unconditionally.
