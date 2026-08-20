- **pgw#1564: a boot that arms ZERO claimed graphs says so — with the hole
  reasons — and a mint mints what it was ARMED on.** The 2026-08-20 field
  shape: a `gen-worker up` boot claimed 14 graphs, holed all 14 with
  `cannot decompress` sitting on each Hole (pgw#1561's ZIPs), and the resident
  log carried not one adoption line — the summary sat at INFO (`up` surfaces
  WARNING+) and reasons were logged nowhere, so the zero was investigated for
  hours as a new defect. Now: zero-armed on a declared-compiled lane is a
  WARNING with deduped hole reasons; the handle carries `hole_reasons` and an
  `adoption` verdict ({engaged, armed, claimed}) beside `adopted_graphs`.
  Separately, an L4 pod declared `self_mint_compile completed 0/14` in 13 ms:
  `arm` counted holes off the live `host.holes` property and `run` re-read
  it — the work-list is now read ONCE at arm and passed through, so an
  emptied second read can no longer settle an untouched mint as complete.
