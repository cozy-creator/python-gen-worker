- **pgw#1591: a WRAPPED compiled dispatcher is not a DISPLACED one — and a
  displaced one now reports what it measured.** `DispatchCounter` asked
  `module.forward is dispatcher`, and pgw#1573's adapter guard installs itself
  AS `module.forward`, so every guarded module read DISPLACED from the moment
  that guard landed. Measured in the field: sd15, `dispatch: DISPLACED on
  UNet2DConditionModel`, 12 of 12 requests on BOTH arms of the pgw#1548
  dynamic-dims benchmark — a whole GPU leg discarded because the lane could
  not tell eager from compiled. The arm was healthy the entire time; the
  question was wrong. `adapter_guard.dispatcher_of` is now the one resolver
  and resolves THROUGH wrappers, so adding a wrapper is not a change every
  reader has to learn about; a module it cannot resolve is genuinely displaced
  (accelerate restoring `_old_forward` after an offload rung is the real shape
  of that) and still reports so.

- **pgw#1591: the displaced verdict states the counts instead of inferring
  them.** The branch ended `"so all N call(s) ran eager"` — a claim it never
  measured. The same logs carried 120 AOTI wrapper invocations per arm, so the
  filing lane had two irreconcilable readings and correctly refused to pick
  one. **Neither was right:** compiled DID execute and the message overcounted
  eager. Displacement and dispatch are separate facts and both are now stated
  (`… still served COMPILED` vs `… ran eager`, whichever the counter measured).

- **pgw#1591: the three-leg adopt test now installs the DAEMON's witness.** It
  asserted on a loader stub's call list; `cli/daemon.py` installs a
  `DispatchCounter` after `host.setup` and prints its verdict per request. That
  difference is why the legs stayed green against a displacing daemon — the two
  booths were not measuring the same thing. All three legs now take the
  daemon's reading and refuse a displaced arm.
