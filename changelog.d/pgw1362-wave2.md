- **Wave 2 of the incident-name fold: the cli/config/sdk cluster.** 9 modules / 2,660 lines become
  3 domain modules / 2,226 lines — `test_config_authority.py` (the declared interpreter env and the
  torch flags it imposes), `test_config_boundary.py` (who may read settings, how a reconcile
  resolves), `test_sdk_authoring.py` (the decorators, the job contract, what a handler emits) — plus
  `test_model_sdk_pgw1332.py` renamed to `test_model_sdk.py`, which was already single-domain and
  needed a name, not a merge. **All 120 tests survive**, counted on both trees before the originals
  were deleted.
- **Four more modules come off the mypy `ignore_errors` burn-down and none go back on**
  (high-water 255 → 251 on this branch; it composes with wave 1's cut). The merged modules are
  typed clean at the test posture.
- **The torch gate did not widen.** `test_settings_authority_pgw1049` and `test_env_contract_pgw718`
  both carry a module-level `pytest.importorskip("torch")` and every test in each uses torch, so
  they merged with each other and *not* with the three torchless config modules — a merge that
  mixed them would have made a torchless runner skip five modules' worth of tests silently.
- **One redundant autouse fixture deleted, not carried:** pgw#718's `_reset_boot_seal` reset
  `env_seal._BOOT_READBACK`, which pgw#1049's `_fresh_boot` — now module-wide — already unwinds
  along with `_ESTABLISHED_OVERRIDES`.
- Mutation-probed, one load-bearing guard per module: `settings_authority.impose_torch` stops
  imposing `cudnn_benchmark` (2 RED), `runtime_config.observe` stops writing the snapshot (2 RED),
  `api/jobs._validate_job_shape` stops refusing class-declared jobs (1 RED).
