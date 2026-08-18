- **pgw#1371: the runtime mint is holes-only, per-class adopted and published,
  and yields to the serving process.** The final-mandate shape: an obligation
  can be SCOPED to named holes (`PendingSelfMint.holes` — the handoff pgw#1372
  fills; the boot's per-class misses already ride `enable_compiled(holes=…)`
  via `BootAdoptOutcome.graph_classes`), the compile children intersect their
  shares with the hole list and prove coverage through their own
  `targeted_classes` reports, and `adopt_minted_class` runs the durable →
  §4.32 arm+parity → publish ladder for EACH class the moment its artifact
  lands — so a pod killed mid-mint keeps (and the fleet store holds) every
  completed graph, and the terminus folds the records instead of arming or
  uploading twice. The FLEET entry-child tree now nices itself
  (`FLEET_MINT_NICE=10`): e2e#1892 run 7 measured the core reserves failing
  to protect serving (a 15m50s invocation never returned in 65 minutes beside
  a 2-on-7 mint), and priority is the reserve mechanism that actually works;
  the rig mint child and the serving process itself stay at full priority.
