- **A new test module may no longer be named for the issue that produced it — `fast gates` refuses
  it.** The number that bought this gate: across five hours on 2026-08-17/18 the pgw#1362
  consolidation landed three waves, removing 41 incident-named modules and ~1,900 lines, and
  `origin/master`'s test tree still went **441 files / 134,979 lines → 411 / 135,818** — thirty
  files fewer and **839 lines MORE**, because other lanes added ~2,700 lines of new issue-named
  tests in the same window. All of it correct, ruled work. **Nobody did anything wrong and the
  corpus still grew.** A cleanup epic cannot outrun its own intake, so the naming rule moves into
  the gate instead of into everyone's memory.
- `scripts/lint_incident_test_names.py` refuses a top-level `tests/test_*.py` or
  `tests_v2/test_*.py` whose filename carries an issue id (`pgw1234`, `gw640`, `th1130`, `cl72`,
  `ie655`, `te148`). The refusal names the domain modules that already exist and says where the
  lineage goes instead: a one-line `# pgw#1234: <one clause>` comment on the test, with the full
  story in the tracker issue. **There is no escape-hatch marker** — unlike the content fences, this
  one sweeps a FILENAME and the fix is `git mv`; a naming rule with an opt-out is a suggestion.
- **The 375 existing modules are grandfathered in a list that only SHRINKS**, and the mechanism is
  structural rather than a promise (the `lint_mypy_ratchet.py` shape): a baseline entry that no
  longer exists on disk is a hard failure, so folding a module *forces* deleting its line; and
  `BASELINE_HIGH_WATER` caps the length, so padding it requires raising a number in the diff.
- **Red-verified against the real repo, not only in fixtures** — a synthetic
  `tests/test_..._pgw9999.py` is refused; the same file domain-named passes; padding the baseline
  to 376 is refused; renaming a grandfathered module without shrinking the list is refused. Plus a
  13-case `--selftest` covering every issue-id spelling and the `_v2`/`_v4` version suffixes that
  must NOT be read as issue ids.
