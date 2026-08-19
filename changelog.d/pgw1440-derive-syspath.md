- **`gen-worker release derive` runs again.** `cli/release.py`'s `_run_derive` imported
  `_ensure_sys_path` from `cli/run.py`, which pgw#1373 (`cd46c957`) deleted with the v1 SDK — the
  symbol was defined nowhere in `src/`, so the derive raised `ModuleNotFoundError` on the second
  line of its handler, on every tree, since that hardcut. It blocked the pgw#1371 mint
  demonstration: the graph blob a runtime mint compiles has no publish leg and no fetch leg, and
  the one workaround (derive on the pod with `--graph-cas`) went through this CLI.

- **Fixed by de-duplication, not resurrection.** The deleted helper's own docstring said it existed
  to "match discover.py's sys.path priming", while `discovery/discover.py` had the same logic
  inlined — two copies of one rule, which is what let a deletion break a caller silently. There is
  now one exported `discovery.discover.prime_sys_path(root)`, called by both `discover_manifest`
  and `_run_derive`. `cli/run.py` stays deleted. Note the order is load-bearing: both inserts go to
  position 0, so the statements read root-then-src to leave `src` ahead of `root`, and that
  precedence is now asserted.

- **`test_cli_entry_orders` stops parametrizing over three deleted modules.** `cli.serve`,
  `cli.run` and `cli.invoke` all went in `cd46c957`, so three of its four cases had been failing on
  master ever since; re-pointed at the CLI's actual surface.
