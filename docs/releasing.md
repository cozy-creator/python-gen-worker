# Releasing gen-worker

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
admits**. `TENSORHUB_SUPPORTED_GEN_WORKER_VERSIONS` matches on `major.minor` (`0.90`), which is
exactly why every patch this week rode without a hub bounce. Pinning wider than the hub admits would
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

So a compiled cell is accepted **only** by a runtime whose `gen_worker` string matches the producer's
exactly, checked per-artifact, at arm time, and **named** on mismatch.

**Therefore loosening the pin does not weaken the invariant.** The `==` in `pyproject.toml` was a
crude *proxy* for a property that `verify()` enforces directly and more strictly: a packaging pin
cannot stop a cell minted elsewhere from being offered to this pod, whereas `verify()` can and does.
Moving to ranges relocates nothing — it just stops pretending the packaging constraint was the
safety mechanism.

**State the honest trade:** with ranges, two endpoints may resolve different patches, and a cell
minted under one will be **refused, typed** (`gen_worker 'a' != runtime 'b'`) by the other. That is a
**cache-hit-rate** cost — a re-mint — **not a correctness cost**. Nothing is ever silently
mis-served. If cross-endpoint cell sharing matters for a given family, converge those endpoints on
one version deliberately; do not reach for a global exact pin to get it.

---

## The gate — what actually has to be green

`ci.yml`, in order. **A local `pytest` run is not the gate** and will let you push a red:

1. `uv sync --locked` — a version bump without `uv lock` fails here. It has.
2. `mypy src/gen_worker`
3. `ruff check src/gen_worker`
4. `scripts/lint_http_timeouts.py` (gw#467)
5. `scripts/lint_unreached_surface.py` (pgw#849 guard 2)
6. `pytest tests/ -n 4 --dist loadfile`
7. `pytest tests_v2/ -n 4 --dist loadfile` — **`testpaths = ["tests"]`, so a bare `pytest` skips this.**

**Publishing additionally requires a green CI run carrying this exact TREE** (`publish.yml` compares
tree SHAs, not commit SHAs). Tag a commit CI has already proven, or dispatch CI on the tag and re-run
publish.

### Why a cut costs hours: chaos accrues CI debt and the cut pays all of it

`chaos` is a shared branch everyone commits to directly (by design), and **nothing gates a chaos
commit on being green.** The publish gate is the first thing that ever demands a green run on an
exact tree — so every red accumulated since the last cut lands on whoever cuts next, who usually
authored none of it.

Measured on the 0.90.6 cut: three reds, none authored by the cutter — another lane's `mypy` error, a
stale pgw#849 baseline entry, and **two tests that had never passed CI in their lives** (established
by `git merge-base --is-ancestor` against the last green run). Budget hours, not minutes, and expect
to choose between fixing another lane's code and branching around it.

**If the red is a semantics decision inside another lane's work, do not fix it to unblock your tag.**
Branch the release from a commit that predates it and say so in the CHANGELOG. A cut that launders
someone else's red is worse than a slow cut.

---

## Mechanics

```bash
# 1. version + lock, in ONE commit
#    (edit pyproject.toml, then:)
uv lock                      # updates the root package version; --locked is the gate
git add pyproject.toml uv.lock CHANGELOG.md

# 2. verify the range BY ANCESTRY, never from cut notes
git merge-base --is-ancestor <commit> HEAD    # per must-ride
git rev-list --count v<prev>..HEAD            # and COUNT it; do not estimate

# 3. green CI on this exact tree, then tag and push
gh workflow run ci.yml --ref <branch>
git tag -a v<X.Y.Z> -m "..." && git push origin v<X.Y.Z>   # tag push triggers publish.yml
```

**Verify the publish landed against the SIMPLE INDEX, not the JSON API** — `pypi.org/pypi/<pkg>/json`
is cached and showed the previous version for minutes after a successful upload:

```bash
curl -s https://pypi.org/simple/gen-worker/ | grep gen_worker-<X.Y.Z>-
```
