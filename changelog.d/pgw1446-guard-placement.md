- **pgw#1444's missing-contract refusal no longer destroys the measurements it fires beside.**
  It refused unconditionally on `_CONTRACT_MISSING`, and `entry_phases` is banked *after* that gate
  — so every reap raised before the fold and `entry_phases` came back empty, breaking pgw#1189's
  law that the attempt which FAILED is exactly the attempt whose measurement the next one sizes
  against (pgw#877 states that rule three lines above the call site). The refusal is now
  conditioned on a child having actually reported a digest: real work whose provenance cannot be
  checked, which is the hazard pgw#840 names. No reported digest means no artifact of unknown
  origin to refuse.

- **The C10 fix that was right is kept.** The original defect was *silence* — `cd46c957` deleted
  `aot_compile_child.py`, `_code_digest()` returned `""` through a branch written for zipimport,
  and the guard skipped while still reading as a guard. `_CONTRACT_MISSING` records the cause and
  makes it inspectable; only the blanket raise is withdrawn.
