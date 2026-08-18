- **master was RED: pgw#1355's `boot_stages` roll-up broke gw#601's activity-kind allowlist.**
  `test_executor_setup_emits_monotonic_activity_phases` asserts that the kinds emitted during
  `ensure_setup` are a subset of the phase envelope's neighbours; `boot_stages` is a new kind that
  rides the same bind, so the subset assertion failed on every run. Caught by a consolidation lane
  whose own PR could not meet its full-suite bar while master was red.
- The fix adds `KIND_BOOT_STAGES` to that list, which is maintenance of the guard rather than a
  widening of it: `boot_stages` is the same shape as pgw#1309's boot row — a self-contained event
  under its own kind, deliberately NOT part of the activity's phase envelope, whose real content is
  asserted below over `KIND_WARMUP` alone. **Red-verified both ways:** shrinking the list back
  reddens the test, restoring it greens it, so the allowlist is still load-bearing.
- **A SECOND master red, from pgw#1363's `cell` → `compiled graph` rename:**
  `test_pod_privilege_isolation_pgw858` still reached `local_compiled_graph_store.CELLS_DIRNAME`,
  an attribute the rename replaced with `COMPILED_GRAPHS_DIRNAME` — an `AttributeError` on every CI
  run. It skips on a developer box (it needs docker + root), so only CI saw it.
- **Why mypy did not catch it, and the argument that makes:** that module sits on the
  `ignore_errors` mypy burn-down, so it was wholly unchecked. The pgw#1362 consolidation took it
  OFF the list on its way through, and mypy then flagged the stale attribute immediately. **A test
  module exempt from type checking is a test module that can silently stop running** — which is the
  whole case for the burn-down.
