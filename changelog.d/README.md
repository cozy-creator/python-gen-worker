# changelog.d — one fragment per issue

`CHANGELOG.md` is append-only history and is **not** edited by lanes. Write your
entry to `changelog.d/<issue>.md` instead — a path no sibling lane touches, so a
CHANGELOG merge conflict is impossible by construction rather than merely rare.

```
changelog.d/pgw968.md      # this repo's issues
changelog.d/th1566.md      # tensorhub-numbered work landing here
```

**One number, no suffixes.** The name must be `<prefix><number>.md` — the number
is what orders the release section, so `pgw984-pgw985.md`, `pgw1016-pgw1017.md`
or `pgw868-a4.md` parses as nothing and refuses the CUT. Work that closes several issues writes
ONE fragment under the lowest number and names the rest in its text. CI checks
this on the PR that adds the fragment (`assemble_changelog.py --check`, fast
gates) rather than weeks later on the cut.

The file holds the bullet(s) exactly as they should appear under the release
heading — no `##` heading of your own, no version, no date:

```markdown
- **pgw#968: lanes no longer serialise on one mutable CHANGELOG.** ...
```

At a cut, `scripts/assemble_changelog.py --version X.Y.Z` concatenates the
unconsumed fragments into a section (ordered by issue number, so the result does
not depend on filesystem order). See `docs/releasing.md`.

## The cut half (pgw#1226) — a fragment is dated by the TAGS, not by the cutter

The paragraphs above document only the lane-vs-lane half, and used to read as if
that were the whole problem. It is not: a fragment also races the CUT.

**A cut does not delete fragments.** It records what it consumed in
`consumed.tsv` — `<version>\t<fragment>`, written only by the assembler — and
which version a fragment belongs to is **derived, not chosen**: the earliest
`vX.Y.Z` tag whose tree contains the fragment, or the version being cut if no tag
does. Two things follow, and they are the reason this exists:

- your fragment merging while a cut is mid-queue is **fine**. It is in no tag, so
  the next cut takes it, and it cannot be deleted out from under your branch.
- a fragment that WAS in a tagged tree and never got assembled is written into
  **that** version's section, under a one-line "attributed after the cut" note —
  not silently re-dated onto the next wheel. `changelog.d/pgw1244.md` is the case
  that forced this: 0.114.3 shipped pgw#1244's code and left its fragment behind.

Consumed fragments stay on disk for one further release and are then pruned; the
`consumed.tsv` row outlives the file, so a re-added fragment is never re-consumed.
**Nothing here is a lane's job.** Do not edit `consumed.tsv`, and do not delete a
fragment because you see it in `CHANGELOG.md`.
