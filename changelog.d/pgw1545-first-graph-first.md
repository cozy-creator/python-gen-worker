- **`gen-worker compile` builds the specialization you are about to RUN first, and can hand the
  rest to a background fill.** Paul, 2026-08-20: *"prioritize generating the graph specialization
  for what we want to run first; the other specializations can be compiled in the background
  while we do inference using the .so that was compiled for the workflow we're actually using."*
  Measured before this: 14 specializations at ~111 s each and **nothing servable for ~26 minutes**,
  because the verb built every one of them eagerly and published the graph-set document last.

  `--first SELECTOR` names the one to build first — a comma-separated conjunction over lane
  contract, target module path, input parameter, dtype, `AxBxC` shape, or a graph-identity prefix,
  matched against facets read off the ingress the lock actually carries. Unstated, it is the
  document's own first record, which is not "whatever came first": graph order is semantic and
  the derive puts the all-defaults specialization there (pgw#1384). A selector that matches
  nothing **refuses and prints what IS addressable** — silently building the default instead
  would report success over a specialization nobody asked for.

  `--fill background` returns as soon as that one artifact is servable and finishes the rest in a
  detached `nice -n 19` child; `--fill none` builds only the first; `--fill all` is the default
  and is the old behaviour, in priority order. The fill is not a second code path with its own
  resume state — it is **this verb** with `--fill all` and every resolved input restated on its
  argv (sm, CAS, module name, lock), so everything already built resolves as PRESENT and it
  continues from there. A killed fill therefore resumes as reuse, and `--verdict PATH` gives it
  somewhere durable to state its result per specialization, since a detached child's exit status
  reaches nobody.

- **The graph-set document is now published BEFORE the first build, and that reversal is what
  makes an incremental run servable at all.** It used to land last, on the runtime mint's
  durability rule that nothing is announced before it exists. That rule is about ARTIFACTS and
  still governs them — but the document is not a claim that artifacts exist. It is the authored
  lane list read out of the committed lock, and `AdoptSession` turns every record it cannot fetch
  into a HOLE: eager for that graph, queued for the mint, **never a refusal**. Publishing it last
  meant a partly-filled store adopted exactly nothing, because the one row adoption enumerates
  FROM had not landed yet.

- **A deferred specialization is not a missing one, and the verdict can tell them apart.** Every
  census row is now attributed to its graph (`Gap`), so a gap this run promised to close is still
  `NOT SERVABLE` with rc=1 while a deliberately deferred one is reported as pending with rc=0.
  The all-complete `SERVABLE — ... all N artifact(s)` line is chosen by asking the serving reader
  which artifacts it can still not find, never by asking what the run intended, so **no
  arrangement of half-done work reaches it**. pgw#1533's witness is untouched: every build is
  still read back through a store object the run did not publish through.

- **13 red arms, 13 caught** (`tests/test_compile_servability.py`, extended): ordering ignored, a
  selector silently falling back to the default, the document published last again, deferred gaps
  counted as fatal, promised gaps laundered as deferred, the all-complete line printed regardless,
  the presence short-circuit removed so a resume rebuilds, the fill never handed its specs, the
  fill child forgetting the resolved module name, a fill that will not start killing the run, no
  verdict written, an unmatched call refusing instead of serving eager, and the late arm made a
  no-op. The last two are the serving half, driven through the real `AdoptSession`: a request
  needing a specialization nobody has built yet runs the author's own forward, and the fill arms
  into the LIVE module with no reboot — which is the whole reason deferring is safe.
