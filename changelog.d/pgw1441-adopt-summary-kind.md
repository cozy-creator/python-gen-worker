- **The adopt roll-up gets its own activity kind, `boot_adopt_summary`.** pgw#1371 emitted the
  per-BOOT reuse verdict under `boot_adopt`, whose `phase` is a per-KEY gate token
  (`hit`/`miss`/`no_export_declaration`, one row per graph). One kind then carried two `phase`
  vocabularies, and the hub keys `info.Activities` on kind alone — so
  `count(*) where kind='boot_adopt' and phase='reused'` reads 0 on every pod predating the code,
  which is indistinguishable from "nothing reused". This is pgw#1067's `warmup`/`warmup_summary`
  incident exactly, caught before it shipped instead of after.

- **The reuse counts are numeric, not prose.** `step` = graphs adopted, `total_steps` = graphs
  claimed, so the reuse ratio is a query rather than a regex over `detail` — and `total_steps` IS
  the lane's graph-class count, readable directly instead of inferred by counting per-key rows.
  `emit_event` gained `step`/`total_steps` for this; both default to 0, which honestly means "not
  a count" for every other event.
