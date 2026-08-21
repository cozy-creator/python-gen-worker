- **pgw#1645: a worker on a lower layout rung now reads as EARNING rather than as a
  slow pod nobody can explain.** A mint declares the byte layout it compiled against and,
  while compiling, records the layout inductor actually asked for. Those two facts, read
  off the artifact that is answering requests right now, are a POSITION:
  `no_wish` (this is the ideal), `at_ideal`, `earning` (a ratified wish is outstanding and
  deliverable), `declined` (ratified, and the platform will not deliver it yet — with the
  reason), `candidate` (outside the catalog, emitted for a human to ratify, never
  invented), `no_single_ideal` (two constants want two arrangements, so a re-mint would
  move the copies rather than delete them). The values are a wire contract in the same way
  `EagerPhase`'s are; they ride out in `ServeAdoption.facts()` beside the counts.

- **pgw#1645: the decline is TYPED and names what is missing, because today it is
  everything.** `tensorfs-py` exports no `fill` and varena implements no `FillSink`
  (varena#13), so no layout-applying transform is reachable from a worker process at all —
  and the confession says exactly that rather than dropping the wish. It also carries the
  measurement that will price the decision once a fill exists (tensorfs#157: an identity
  fill runs at 5.15 GiB/s and a `channels_last` fill at 0.12 GiB/s, ~31 ns/element,
  extrapolating to ~5 s of host time for SDXL's 635 MiB conv set). gen-worker deliberately
  computes no run count of its own — that would be a second implementation of the fold, and
  `FillStats::runs_per_element` is the number.
