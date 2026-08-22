- **pgw#1668 follow-up: the component dtype-pin fence reads a component's NARROWEST dtype, not
  what it is mostly.** pgw#1668 replaced a majority-by-tensor-count measure with a strict one and
  disarmed this fence for the case it exists for. The old measure answered `bf16` for a component
  that was 20 tensors cast and 3 not, so an fp32 pin refused it; the new one answers `mixed`, and
  `dtype_bits("mixed")` is 0, which makes `is_narrowing` False against everything. **A fully
  violated pin was refused and a HALF violated one published silently** — the fence got strictly
  less able to fire the more of the component survived, which is backwards, and it was found by
  testing the fence after the change rather than by reasoning that the change was contained. The
  rollup is still what the checkpoint REPORTS (`mixed` is the honest label and the whole point of
  pgw#1668); it is simply not a comparand, because a pin is violated by any tensor below it. The
  fence now reads `component_narrowest_dtype` on both sides — the produced tree and the source
  that excuses it — so a mirror of a tree that already ships narrow still passes. Red-armed: with
  the comparison put back on the rollup, the half-violated case publishes, 5/5.
