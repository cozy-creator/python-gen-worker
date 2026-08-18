- **The boot no longer calls the keyset derive+resolve ladder.** The §4.27
  boot-time derivation entry (`Executor._boot_adopt`) is deleted; a compiled
  family boots EAGER (self-mint per fleet policy — §4.28/§4.31 stand restored)
  and states the new typed gate reason `boot_derive_deleted` on the wire, so
  the fleet-wide cutover is a query. Identity derives at PUBLISH; compiled
  artifacts arrive via the adopt-first release pull (`gen_worker.serving`).
  Adopt-only pods refuse typed (`NOT_ADOPTABLE`). `keys_from=` telemetry is no
  longer emitted by any boot — the new flow reports `graphs_from=release` on
  the `adopt_pull` span. The `boot_adopt`/`keyset` modules remain one release
  as mint-lane tooling; their deletion is pgw#1373's.
