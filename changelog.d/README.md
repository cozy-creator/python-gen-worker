# changelog.d — one fragment per issue

`CHANGELOG.md` is append-only history and is **not** edited by lanes. Write your
entry to `changelog.d/<issue>.md` instead — a path no sibling lane touches, so a
CHANGELOG merge conflict is impossible by construction rather than merely rare.

```
changelog.d/pgw968.md      # this repo's issues
changelog.d/th1566.md      # tensorhub-numbered work landing here
```

The file holds the bullet(s) exactly as they should appear under the release
heading — no `##` heading of your own, no version, no date:

```markdown
- **pgw#968: lanes no longer serialise on one mutable CHANGELOG.** ...
```

At a cut, `scripts/assemble_changelog.py --version X.Y.Z` concatenates every
fragment into a new section (ordered by issue number, so the result does not
depend on filesystem order) and deletes them. See `docs/releasing.md`.
