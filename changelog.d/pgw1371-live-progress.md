- **pgw#1371: the runtime mint streams per-class progress, so a torn-down pool
  is legible as mid-flight work instead of `pool_busy_s 0 / n_entries 0`.** The
  "pool never dispatches" fleet read (pods `rzz5p4e7b2kcpp` / `c7bx4yxbh3wx87`,
  e2e#1892 runs 7/8) was a telemetry artifact: the pool dispatched and its
  children compiled — `zco8e1bx0t1jgk` completed the identical 36-class sdxl
  shape at 0.9989 pool efficiency in 3608 s — but a share of 36/K classes
  reported once, at its end, so both pods (shut down at 2798 s / 471 s, before
  the ~3600 s first-share horizon) rolled up zero everywhere and the hub read
  the healthy mint as `self_stalled=t` for its whole life. The compile child now
  writes one row per packed class and a position beat per phase boundary; the
  pool harvests them every poll — `class_spans` fills live (the phase snapshot
  and abandon table name what landed), the ledger carries `pool_classes_landed`
  and `pool_child_cpu_s`, the silence window admits landed classes and position
  beats as evidence, and the mint's progress counter beats per GRAPH CLASS
  against the class goal instead of once per share against the worker count.
