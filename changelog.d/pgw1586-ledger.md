- **pgw#1586 phase 1: the residency ledger records what loads and requests actually cost.**
  `models/residency_ledger.py` banks four measured facts per
  `endpoint × checkpoint × {w}x{h}x{batch} × extras`: the activation peak
  (`max_memory_allocated − baseline`, which is residency-INDEPENDENT and therefore the property
  the old 1.25 GiB constant lacked — it was true of `model_offload` and 3.6× wrong on
  `partial_resident`), the `_placement_attribution` split at arm time, requests-per-boot, and
  allocator-retry counts as the softness signal for cached memory. **Phase 1 RECORDS AND
  DECIDES NOTHING** — no placement path consults it, and a test enforces that by grepping the
  tree, so landing it cannot change a rung.
- **pgw#1586: the ledger refuses to answer when it does not know.** Below
  `MIN_SAMPLES_FOR_PERCENTILE` the windowed p99 returns `None` and the caller must keep its
  default floor, because **at small n a "p99" IS the max** and offering one would dress a single
  observation as a distribution fact — the same derived-wearing-measured error the ledger
  exists to end. A cold, corrupt, or unwritable ledger reads as COLD and never raises into a
  placement. Samples are windowed so one anomalous request cannot pin the reserve forever, and
  the checkpoint identity is in the key so changed weights start cold with no invalidation
  machinery.
