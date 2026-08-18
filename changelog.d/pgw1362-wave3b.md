- **The procsplit cluster folds 5 modules into 2, and its shared rig finally moves to `harness/`.**
  `test_procsplit.py` (the split boundary and the authorization it carries) and
  `test_pod_isolation.py` (what the compute child may touch on the host: uid, OOM rank, host RAM)
  replace `test_procsplit_pgw763`, `test_procsplit_security_pgw763`,
  `test_pod_privilege_isolation_pgw858`, `test_procsplit_oom_rank_pgw975` and
  `test_host_move_guard_pgw763`. 2,863 → 2,409 lines (incl. the 168-line hoisted rig); 86 tests in, 85 out.
- **A test module was doubling as a fixture library, and the fold made that impossible to keep.**
  Four other modules did `from test_procsplit_pgw763 import SplitHarness, captured_dials, ...`.
  The rig is now `tests/harness/split.py` where the other shared rigs live, and
  `test_procsplit.py`, `test_pod_isolation.py`, `test_child_faults.py` and `test_receipts_trust.py`
  all import it from there. **`TESTS_DIR` had to become `parent.parent`** — the hoist moved it a
  directory deeper, and the un-fixed version spawned every child from
  `tests/harness/harness/...`, which the suite caught as a boot-crash loop rather than a skip.
- **The value pass on this cluster found almost nothing to delete, and that is the finding.**
  Probing `actions.authorize` and `capability.decide` produced guard-groups of 3 and 8 tests — but
  each red set spanned *distinct* shapes, not variants. **Exactly one deletion survived probing**:
  the `"unlisted path"` row of `test_action_table_refuses`, whose subject (path matching) reddens
  `test_delta1_parent_refuses_a_hub_call_the_allowlist_does_not_name` — the same property through a
  real split child. The `"wrong method"` row beside it is uniquely guarded: breaking method
  matching alone reddens that row and nothing else.
- **The journey hypothesis is REFUTED for this cluster.** The plan was to replace 23 narrow
  security rigs with one broad journey. The probes say those 23 tests are not variants of each
  other, so a journey would have to assert every refusal individually anyway — it would save
  fixture setup, not properties. Recorded on pgw#1362 so no later wave re-proposes it from the
  file count alone.
- **A COARSE mutation fuses distinct guards.** Breaking path-and-method matching together reddened
  both `test_action_table_refuses` rows and made them look like one property; narrowing the
  mutation to each separately showed they are two. The probe must be as narrow as the property.
- Fold probes: the delta-1 credential strip removed → 2 RED **across both merged modules**;
  `privdrop.plan_drop` always returning None → 1 RED. Four more modules leave the mypy
  `ignore_errors` burn-down with none added (235 → 231); the pgw#1362 intake baseline shrinks
  375 → 370, which is the new gate's first real use — **it caught the fold and refused the branch
  until the five folded names were removed from the list.**
