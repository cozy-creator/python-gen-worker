- **pgw#1339: a changelog fragment is dated by its ISSUE'S COMMITS, not by the
  commit that added the fragment file — and the recipe that repaired one late
  note stops sweeping every other pending fragment in with it.** Attribution used
  to read the commit that introduced the fragment FILE. A fragment that does not
  exist yet is exactly the case a repair is *for*, so it had no such commit, took
  the "in no tag → the version being cut" fallback, and so did everything else
  pending. `--version 0.121.0`, run to date ONE late fragment into an
  already-released section, therefore swept **every** pending fragment into
  `## 0.121.0` — 5 on the first attempt and **16** on the second, including work
  that had shipped in no wheel at all. The recipe was safe only when nothing else
  was pending, which on this repo is never. It bit two cutters in two days.

  A fragment is now dated by the commits whose SUBJECT names its issue
  (`pgw#1323:`), not by the ones whose body merely mentions it, and it is
  assembled into the earliest release tag whose tree contains **all** of them — a
  note is true only once everything it describes has shipped. Work in no tag
  rides the version being cut, **and only if that version is not already
  released**; that guard is what tells a repair from a cut. Anything else stays
  pending and is PRINTED with its reason rather than silently re-dated, so
  `--dry-run` is a readable answer instead of a diff to audit. When no subject
  names the issue the tool still falls back to the fragment file's own commit.

- **pgw#1339: a fragment name may carry a per-lane suffix, so one issue is not
  one shared path.** `changelog.d/pgw1346.md` had become a shared path again —
  around ten batch lanes appending to one file, which re-serialised the merge
  queue and ejected PRs as `CONFLICTING`, the precise failure `changelog.d/` was
  built to remove (pgw#916, twice). A lane of a batched issue now writes
  `changelog.d/pgw1346-b4-video.md`: disjoint paths so they cannot conflict, the
  same issue number so they are dated by the same commits and land **adjacent in
  one section** — unsuffixed file first, then suffixes in order. The
  `<prefix><number>` core stays mandatory, because it is what dates and orders
  the fragment, so `b3-math.md` is refused by `--check` at the lane's own PR
  rather than at the cut.

- **pgw#1339: `--cut-ref` names the commit being released** (default `HEAD`).
  `docs/releasing.md` tells a cutter to branch from an older commit when a red is
  not theirs; passing the same ref being tagged makes the tool hold back the
  fragments for work that merged *after* it, instead of writing notes for code
  the wheel does not contain.
