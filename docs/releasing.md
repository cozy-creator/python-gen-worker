# Releasing gen-worker

**This file is the ONLY release procedure.** Publication is the **tag push** — `publish.yaml` refuses
to ship a tree no CI run has proven. `Taskfile.yaml` deliberately has no `publish` task: a local
`uv publish` walks around that gate, and a second written procedure is how a cutter ends up
following the stale one. **No local mint is a release gate** — the `rig:*` tasks are development
vehicles, they run real inductor/AOTI compiles, and Paul's 2026-08-10 hard cut puts every mint on a
remote pod. (pgw#1140, 2026-08-12: `task publish` still said otherwise, five days after the cut that
superseded it.)

Two rules, both from Paul (2026-08-02), both replacing an unwritten habit. If you are about to cut,
read the first one — it is usually the answer.

---

## 1. Batch releases. Cut when a POD RUN needs one, not when a lane finishes something.

> *"It would help also if you stopped creating so many frivolous pushes into PyPi / new versions
> randomly. That makes us look stupid on PyPi, having 10 new versions in a single day. It would be
> better if version-releases were more thoughtful and batched."*

**What this replaces:** the de facto rule was *"cut whenever something lands that a lane wants
released."* Nobody wrote that down either, which is why it was never weighed. It produced **0.90.2
through 0.90.6 in a single day**, each unblocking one lane.

**The rule:**

- **The only consumer that needs a PUBLISHED version is a pod run.** Lanes develop against `chaos`;
  nothing else requires PyPI. An endpoint image resolves `gen-worker` from `pypi.org/simple` at build
  time, so a pod is the one thing a tag actually unblocks.
- **So: cut when a pod run needs one, and batch everything ready at that moment.**
- **"My work should be in a release" is NOT a reason to cut.** It is a reason to add it to the next
  batch — and to say so in the tracker, so the next cutter knows it is waiting.
- **If two pod runs are close together, they share a cut.**

Rule 2 is what makes this cheap: once endpoints admit a *range*, a lane's work reaching `chaos` is
usually enough on its own, and the release stops being the bottleneck that justified cutting five
times in a day.

---

## 2. Endpoints pin a RANGE, not an exact version. Conversion tracks newest.

> *"You don't need to pin a worker to a specific gen-worker version; you can make it a range as
> needed. And the conversion endpoint should be using whatever the newest version is; keep it up to
> date."*

**The bound, and why:** `>=X.Y.Z,<X.(Y+1).0` — i.e. bounded at the **major.minor the hub gate already
admits**. `TENSORHUB_SUPPORTED_GEN_WORKER_VERSIONS` matches on `major.minor` — **read its CURRENT
value from the deployed hub rather than from this sentence, and bump it in lockstep with each
MINOR** (it read `0.91` when this was written and the number has moved many times since) — which is
exactly why every patch that week rode without a hub bounce. Pinning wider than the hub admits would
produce a worker the hub refuses at Hello; pinning narrower re-creates the problem this removes.

**The conversion endpoint tracks newest** rather than a range, because it is the *producer*: it should
be the first thing on a new SDK, not the last.

### What gw#391 was actually protecting — checked before loosening, as required

gw#391 is stated as *"the producer's pin must be exact and track the serving pin."* **That rule is
unsatisfiable as written** — there has never been a single serving pin to track (the fleet has run
0.90.3, 0.90.4 and 0.90.5 simultaneously), and `deploy_family.py` / `deploy_chaos_l4.py` **rewrite the
pin in a temporary copy at pack time**, so the value in git was never the value in the fleet.

**The invariant underneath it is real, and it is not a version string — it is artifact compatibility,
enforced at SERVE time.** `compile_cache.verify()`:

```python
want_gw, have_gw = str(meta.get("gen_worker") or ""), gen_worker_version()
if want_gw != have_gw:
    # gw#391: the producer's gen-worker shapes the traced graph; a version
    # drift means the FX-graph cache keys may no longer match.
    return f"gen_worker {want_gw!r} != runtime {have_gw!r}"
```

So a compiled graph is accepted **only** by a runtime whose `gen_worker` string matches the producer's
exactly, checked per-artifact, at arm time, and **named** on mismatch.

**Therefore loosening the pin does not weaken the invariant.** The `==` in `pyproject.toml` was a
crude *proxy* for a property that `verify()` enforces directly and more strictly: a packaging pin
cannot stop a compiled graph minted elsewhere from being offered to this pod, whereas `verify()` can and does.
Moving to ranges relocates nothing — it just stops pretending the packaging constraint was the
safety mechanism.

**State the honest trade:** with ranges, two endpoints may resolve different patches, and a compiled graph
minted under one will be **refused, typed** (`gen_worker 'a' != runtime 'b'`) by the other. That is a
**cache-hit-rate** cost — a re-mint — **not a correctness cost**. Nothing is ever silently
mis-served. If cross-endpoint compiled graph sharing matters for a given family, converge those endpoints on
one version deliberately; do not reach for a global exact pin to get it.

---

## The gate — what actually has to be green

`ci.yaml`. **A local `pytest` run is not the gate** and will let you push a red.

Since pgw#952 it is TWO PARALLEL JOBS, and both must be green — `publish.yaml` keys on
the run's conclusion, which is success only when both are. It runs on **every PR into `master`**
(pgw#977 narrowed the trigger back to `[master]` alone when `dev` was deleted, so that IS the
every-lane trigger), plus `workflow_dispatch` — so a cut no longer inherits a pile of unproven lane
merges. A third context, `drift`, also gates.

**`fast gates`** (~1m35s):

1. `uv sync --locked --extra dev` — a version bump without `uv lock` fails here. It has.
2. `mypy src/gen_worker` — GATING (gw#497; the tree is mypy-clean, no baseline)
3. `scripts/lint_http_timeouts.py` (gw#467)
4. `scripts/lint_unreached_surface.py` (pgw#849 guard 2)
5. `scripts/lint_config_reads.py` (pgw#931 guard)
6. `uv build` — a cut can still fail here, after every test is green.

**`tests`** (~15m15s):

1. `uv sync --locked --extra dev`
2. install pinned CPU `llama-server` (so the gw#402 runtime tests run for real)
3. `pytest tests/ -n 4 --dist loadfile`
4. `pytest tests_v2/ -n 4 --dist loadfile` — **`testpaths = ["tests"]`, so a bare `pytest` skips this.**

**`ruff check src/gen_worker` is NOT a gate.** `ci.yaml` runs it with
`continue-on-error: true` (lint debt, mostly `worker.py`) — it is visible in
the logs and blocks nothing. Do not spend cut time on red ruff output.

**Publishing additionally requires a green CI run carrying this exact TREE** (`publish.yaml` compares
tree SHAs, not commit SHAs). Tag a commit CI has already proven, or dispatch CI on the tag and re-run
publish.

### Why a cut used to cost hours — and what pgw#952 changed

**Historically:** `chaos` was a shared branch everyone committed to directly, and nothing gated a
chaos commit on being green. The publish gate was the first thing that ever demanded a green run on
an exact tree — so every red accumulated since the last cut landed on whoever cut next, who usually
authored none of it.

Measured on the 0.90.6 cut: three reds, none authored by the cutter — another lane's `mypy` error, a
stale pgw#849 baseline entry, and **two tests that had never passed CI in their lives** (established
by `git merge-base --is-ancestor` against the last green run).

**Since pgw#952** every lane PR is gated, so the debt stopped accruing: `ci.yaml` had been
`branches: [master]` while lanes targeted `dev`, and such a PR ran NO CI AT ALL (PR #466's
`statusCheckRollup` was literally `[]`). `dev` has since been deleted and pgw#977 pointed the
trigger back at `master`, which is now where every lane PRs — so a red is refused at the lane PR
that introduces it rather than at the cut. **This has held in practice:** the 0.113.0 cut was green
first try on all three contexts, inheriting nothing. Anything that merged before pgw#952 was never
gated, so while such history is in range, budget hours rather than minutes and establish by
ancestry — never from cut notes — whether a red is yours.

**If the red is a semantics decision inside another lane's work, do not fix it to unblock your tag.**
Branch the release from a commit that predates it and say so in the CHANGELOG. A cut that launders
someone else's red is worse than a slow cut.

---

## The CHANGELOG is assembled, not edited (pgw#968)

Lanes write `changelog.d/<issue>.md` and never touch `CHANGELOG.md`. One mutable
file that every concurrent lane appends to made a conflict near-certain for any PR
that sat through a sibling merge — one lane rebased three times in a night, every
time on the same single CHANGELOG hunk. A per-issue path cannot conflict.

The cut collapses them, ordered by issue number so the section does not depend on
filesystem order:

```bash
scripts/assemble_changelog.py --check                     # names parse, none empty
scripts/assemble_changelog.py --version <X.Y.Z> \
    --headline "**the one thing this cut is about**"      # writes; deletes nothing
```

### A cut MARKS fragments consumed, and the TAGS decide which version owns one (pgw#1226)

**What this replaces:** the cut used to `unlink()` every fragment it assembled, and
it dated every fragment it found to the version being cut. Both halves were wrong
in the same direction — they trusted the moment the cutter happened to run the
script.

- **Nothing is deleted at cut time.** The assembler appends
  `<version>\t<fragment>` rows to `changelog.d/consumed.tsv` and leaves the files
  alone. Fragments consumed by a release older than the previous one are pruned on
  a later cut; the ledger row outlives the file, so a re-added fragment can never
  be re-consumed.
- **Attribution is derived from `git tag --contains`,** not from what the cutter
  swept: a fragment belongs to the **earliest `vX.Y.Z` tag whose tree contains it**,
  and to the version being cut only if no tag contains it.

**The failure this kills, named:** *a fragment sitting in the tagged tree, absent
from that version's section, silently re-dated onto the next wheel by the next
cut.* It is not hypothetical — **0.114.3 was tagged with pgw#1244's code in the
tree and `changelog.d/pgw1244.md` unconsumed**, and under the old tooling 0.115.0
would have printed that bullet under `## 0.115.0`, pointing readers at a wheel
that did not contain the change. The fragment is now written into the existing
`## 0.114.3` section under a one-line *attributed after the cut* note, by the tool,
with no cutter decision involved.

The corollary the freeze convention kept failing to buy: **a lane fragment that
merges while your cut is mid-queue is simply the next cut's.** It is in no tag, so
it cannot be mis-dated, and the cut never deletes it, so it cannot be lost.

Two refusals rather than a wrong answer: the assembler **refuses on a shallow
repository** (a graft makes `tag --contains` answer about history the checkout does
not have — the failure recorded in th#1810) and **refuses if no `vX.Y.Z` tag
exists at all**. `--check` reads no git and stays the cheap `fast gates` guard; it
now prints each fragment as `pending` or `consumed -> <version>`.

**`--version` with nothing pending for it is not an error** — it is how a
mis-attribution is repaired without cutting: the older sections are corrected, no
new section is written, and the tool says so. **Nothing assembled AT ALL is a
refusal** (exit 1, nothing written): naming a version no pending fragment belongs
to is a mistyped invocation, not a repair.

### The version is derived from the WORK, not from the fragment file (pgw#1339)

**What this replaces, and it bit two cutters in two days:** attribution used to
read the commit that added the fragment FILE. A fragment that does not exist yet —
the case a repair is *for* — has no such commit, so it took the
"in no tag → the version being cut" fallback, and so did every other pending
fragment. `--version 0.121.0`, run to date one late fragment into an
already-released section, therefore swept **every** pending fragment into
`## 0.121.0`: 5 on the first attempt, **16** on the second, including work that
had shipped in no wheel at all. The recipe was safe only when nothing else was
pending, which on this repo is never.

A fragment is now dated by its issue's **authored commits** — the ones whose
SUBJECT says `pgw#1323:`, not the ones whose body mentions it — falling back to
the fragment file's own commit only when no subject names the issue. It is
assembled into the earliest release tag whose tree contains **all** of them; a
note is true only once everything it describes has shipped. Work in no tag rides
the version being cut, **and only if that version is not already released** — the
guard that tells a repair from a cut. Anything else stays pending and is printed:

```
$ scripts/assemble_changelog.py --version 0.121.0 --dry-run
pgw1323.md -> 0.121.0 (late; its code shipped in v0.121.0)
1 fragment(s) pending, not in 0.121.0:
  pgw1356.md: its work (commit subjects: e95000c7, 5c574917) is in no release tag,
  and 0.121.0 is already released -- unshipped work cannot be dated into it
```

**Repairing a silent release note is now the documented one-liner it always
claimed to be:** write `changelog.d/<issue>.md`, run
`assemble_changelog.py --version <the version whose tag contains the work>`, and
the bullet is appended to that existing section under the *attributed after the
cut* note — one heading, in place, with every unrelated fragment left pending.
Dry-run it first and read the pending list.

### One issue, many lanes: `pgw1346-<suffix>.md` (pgw#1339)

A fragment name is `<prefix><number>[-<suffix>].md`. **The suffix exists because
`changelog.d/pgw1346.md` became a shared path again** — around ten batch lanes
appended to the one file, which re-serialised the merge queue and ejected PRs as
`CONFLICTING` (pgw#916 twice), the precise failure `changelog.d/` was built to
remove. A lane of a batched issue writes its own file:

```
changelog.d/pgw1346-b3-math.md      changelog.d/pgw1346-b4-video.md
```

Disjoint paths, so they cannot conflict; the same issue number, so they are dated
by the same commits and land **adjacent in one section**, unsuffixed file first
then suffixes in order. **The `<prefix><number>` core is never optional** — it is
what dates and orders the fragment — so `pgw-b3-math.md` or `b3-math.md` is
refused by `--check` at the lane's own PR, not at the cut.

`--cut-ref` (default `HEAD`) names the commit being released. Cutting from an
older commit — which `docs/releasing.md` tells you to do when a lane's red is not
yours — means work merged after it is not in your wheel, so pass the same ref you
are tagging and the tool will hold those fragments for the next cut.

### SWEEP THE FRAGMENTS AGAINST `origin/master`, NOT YOUR CHECKOUT

**pgw#1226 took one half of this off you and left the other.** A fragment that EXISTS and was
never assembled is now handled by the tool — it is attributed back to the tag whose tree holds it,
loudly. What is still yours is the half no tool can see: **a merge that never wrote a fragment at
all.** That is the sweep below, and it is unchanged.

A merge with no fragment is a **silent release note**, and the sweep for one is the cutter's job —
lanes forget. Do it by listing the range's commits against the fragments **as they exist on the
ref you are cutting**:

```bash
git log --format='%h %s' v<prev>..origin/master          # every merge, with its issue number
git ls-tree --name-only origin/master changelog.d/       # NOT `ls changelog.d/`
```

**`ls changelog.d/` lies whenever your checkout is behind**, and at a cut it usually is — lanes are
merging while you assemble. The 0.113.0 cut's checkout was two commits behind and the working-tree
listing showed pgw#1159's fragment MISSING when it was merged and present; had that been trusted,
the cut would have written a duplicate. Same failure in the other direction is worse: a fragment
that exists only in your stale tree looks owed and gets written twice.

Expect the ledger's "known-owed" list to be **incomplete** — it names what a lane remembered to
file. 0.113.0's ledger named one owed fragment; the sweep found five.

## Mechanics

```bash
# 0. assemble the CHANGELOG section from changelog.d/ (above)

# 1. version + lock, in ONE commit
#    (edit pyproject.toml, then:)
uv lock                      # updates the root package version; --locked is the gate
git add pyproject.toml uv.lock CHANGELOG.md changelog.d

# 2. verify the range BY ANCESTRY, never from cut notes
git merge-base --is-ancestor <commit> HEAD    # per must-ride
git rev-list --count v<prev>..HEAD            # and COUNT it; do not estimate

# 3. green CI on this exact tree, then tag and push
gh workflow run ci.yaml --ref <branch>
git tag -s v<X.Y.Z> -m "..." && git push origin v<X.Y.Z>   # tag push triggers publish.yaml
```

**Tags are SSH-SIGNED** (`git tag -s`; the repo sets `gpg.format=ssh` + `user.signingkey`). Check
with `git tag -v v<X.Y.Z>` before pushing — a lightweight `git tag` produces an unsigned, unverifiable
release marker, which v0.110.0 is.

### Only a DISPATCHED run proves a tree — the gate enforces this, you do not have to remember it

Step 3 is `gh workflow run ci.yaml --ref <branch>` for a reason: a `workflow_dispatch` (or `push`) run
checks out the ref it names, so its `head_sha` genuinely names the tree it built. **A `pull_request`
run does not** — `ci.yaml` checks out with no `ref:`, so GitHub builds `refs/pull/<n>/merge`, your head
merged with whatever `master` is at that moment, while still recording your branch head as
`head_sha`.

**`publish.yaml` refuses a `pull_request` run as proof** (pgw#1191, `scripts/assert_ci_proof.py`), so
this is a rule the gate holds rather than a step you can forget. If you see it, the refusal tells you
which problem you have:

| refusal | what it means |
|---|---|
| `only_pull_request_proof` | your tree is fine, your **evidence** is the wrong kind — dispatch CI on the tag and re-run publish |
| `no_run_carries_tree` | this content has **never been tested anywhere** |

A `pull_request` run remains the right gate for *merging*. It is simply not proof for *publishing* —
and it was pgw#795's v0.78.0 hole arriving through a different door, found by a cut whose PR run went
green while master moved four times underneath it.

### PIN the commit you cut, and TAG IT — do not tag master's tip

Lanes merge while you assemble; master moved four times during the 0.113.0 cut. Cut from a pinned
commit, say which one in the release notes, and **never re-point at master's tip to pick up work that
landed under you** — that ships code with no changelog entry, which is the silent release note this
whole file exists to prevent.

That leaves the tag on a branch commit, so the cut PR must land as a **TRUE MERGE, not a squash**. A
squash rewrites your commit, the tag then points at something that is NOT an ancestor of `master`,
and the next cutter's `git rev-list v<X.Y.Z>..HEAD` silently re-lists everything you just released.

Since pgw#1209 you do not choose this per-PR: `master` is behind a **merge queue**, and the queue's
merge method is set to a true merge for exactly this reason — one method applies to every entry, and
a squash queue would break every release cut silently. So `gh pr merge <n>` enqueues, the queue
re-runs `fast gates` + `tests` against master's tip, and it lands as a merge commit. Assert it, do
not hope:

```bash
git merge-base --is-ancestor v<X.Y.Z> origin/master && echo OK
```

## Verify the PUBLISHED ARTIFACT — a green pipeline is not evidence of a published wheel

**Verify against the SIMPLE INDEX, not the JSON API** — `pypi.org/pypi/<pkg>/json` is cached and
showed the previous version for minutes after a successful upload. **`pip` and `uv` read that same
cached index**: minutes after 0.113.0 was live on the simple index, `pip download gen-worker==0.113.0`
still failed with *"No matching distribution found"* and listed every version up to 0.112.0. That is
the cache, not a missing wheel — do not re-cut, do not "fix" the publish. Consumers hitting it want
`uv lock --refresh`.

```bash
curl -s https://pypi.org/simple/gen-worker/ | grep gen_worker-<X.Y.Z>-
```

Then **open the artifact and read it**, because the index only proves an upload happened:

```bash
url=$(curl -s https://pypi.org/simple/gen-worker/ \
      | grep -o 'https://files.pythonhosted.org/[^"]*gen_worker-<X.Y.Z>-py3-none-any\.whl[^"]*' | head -1)
curl -sL "$url" -o /tmp/w.whl && sha256sum /tmp/w.whl      # must match the index's #sha256=
python3 -c "import zipfile;z=zipfile.ZipFile('/tmp/w.whl');\
print([l for l in z.read([n for n in z.namelist() if n.endswith('METADATA')][0]).decode().splitlines() if l.startswith('Version:')])"
```

Check the METADATA version **and one thing this cut actually changed** — a file the release deleted
should be absent, a restored function present. The 0.113.0 cut confirmed `mint_budget.py` and
`local_compiled_graphs.py` gone and `trt_engine.build()` present, which is the only check that distinguishes
"the right tree shipped" from "a wheel with the right number shipped".
