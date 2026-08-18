- **The first VALUE pass on the test corpus — and mutation probing refuted three quarters of it.**
  Paul's ruling (2026-08-17: *"just because a test exists doesn't mean it's useful"*) makes value,
  not vacuity, the deletion criterion. Wave 3a read all 36 short tests in the procsplit/child-fault
  cluster with an explicit deletion bias and condemned five. Each condemned test was then probed the
  only honest way — **break its subject in `src/` and check whether anything else goes red** — and
  **three of the five turned out to be the ONLY guard of their property**: `container_limits()` can
  return `{}`, `compile_crash_rows()` can drop its kind filter, and `memory_oom_group` can stop
  being recorded, all with the rest of the suite green. Those three were restored.
- **Two deletions survived the probe and stand:** `test_describe_exit_decodes_signals` (a decoder
  unit test carrying dead code — a real signal death through `test_signal_death_is_named` reddens
  five tests when the decoder breaks) and `test_report_is_a_noop_without_an_orchestrator_address`
  (uniquely guarded and deliberately given up: the guarded state is unreachable on a pod, which
  always has an orchestrator address).
- **A vacuity finding banked, not fixed here:** `test_signal_death_is_named` asserts
  `memory.max=` / `memory.current=` / `cpu.max=` / `host_cpu_count=` appear in the death dial, and
  those assertions **still pass when `container_limits()` returns `{}`** — the dial prints the keys
  regardless. The strings are checked; the facts are not.
- **The child-fault modules fold 5 → 1.** `test_child_faults.py` replaces `test_gw640_postmortem`,
  `test_worker_fatal_gw640`, `test_native_crash_streak_pgw676`, `test_child_stderr_postmortem_pgw833`
  and `test_pgw714_compile_crash`. 48 tests in, 46 out (the two deletions above), 1,105 lines → 963.
  Three more modules come off the mypy `ignore_errors` burn-down with none added (238 → 235).
- Probe that the fold kept its teeth: `postmortem.record_native_crash` stops de-duplicating attempts
  of one request → **4 tests RED**.
