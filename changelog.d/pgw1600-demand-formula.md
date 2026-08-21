- **pgw#1600: a lane's demand is DATA — a term algebra, serialized into the
  release document, evaluable by a non-Python reader, and falsified on every
  serve before anything is allowed to consume it.** `lane(request=...)` already
  took a formula; it had no evaluator, no serialization and no instrument. It
  now has all three. Evaluation is exact 64-bit INTEGER arithmetic — each term
  is an integer numerator over a fixed scale (`megapixels` is `width*height`
  over 1_000_000) and the sum is floor division, computed without ever forming
  `coefficient * value`, which overflows int64 at plausible shapes. That is not
  fastidiousness: a float divide is where two languages stop agreeing, and the
  whole point of the serialization is that tensorhub's Go evaluator returns the
  same bytes. `gen_worker/contracts/demand_vectors.json` is the shared
  conformance corpus, and it is GENERATED from this repository's own evaluator
  rather than typed by hand — a corpus with hand-written expectations proves a
  human and Go agree, which is not the property anyone wants.

- **pgw#1600: every coefficient carries its provenance, and claiming a
  measurement without naming it is a refusal.** `Basis` has three values and
  the default is the honest one: `uncalibrated`, a declared prior. `measured`
  and `ledger` require a `source=` citing the run or the ledger key, and adding
  two same-named terms of DIFFERENT provenance refuses rather than laundering
  one claim into the other. A formula reports the claim its WEAKEST term
  supports, so a measured intercept beside a guessed slope reads as
  uncalibrated — which is what it is.

- **pgw#1600: the release document's lane rows carry a `demand` block, and the
  worst case in it is DERIVED.** Terms, the closed vocabulary they are written
  against (so a Go reader validates its own table instead of assuming), the
  advertised envelope, the shape inside that envelope which maximises the
  formula, and the bytes. Every envelope axis states its SOURCE — `advertised`
  from a bounded payload field, `declared` from the payload's own bucket table,
  or `default` with the reason. An aspect-ratio enum can now dimension itself
  via `Annotated[..., Shape(pixels=_BUCKETS)]`, passing the endpoint's own
  table by reference so there is no second spelling of its geometry to drift.
  The block states what it is NOT: request arena only, weights are manifest
  arithmetic the hub adds.

- **pgw#1600: `demand_miss` — predicted-vs-measured, banked every serve,
  counted and hub-visible per (lane × regime).** The request arena is measured
  across the handler and decomposed the way it is SPENT: the torch allocator's
  peak, growth outside the allocator (CUDA context, cuDNN/cuBLAS workspaces,
  AOTI's own `cudaMalloc`'d pool) and driver growth. Eager is judged on the
  allocator; compiled on the driver total, because AOTI's first-call allocation
  cannot spend the allocator's cache. A compiled miss is emitted as a P0 defect
  of the stamp, never as a statistic. Samples never pool across regimes or
  lanes, a regime the worker could not determine pools with nothing, and a
  concurrent request banks nothing at all rather than attributing a co-tenant's
  activations to this lane.

- **pgw#1600: provably zero admission decisions consume the number — the
  falsifier ships before the enforcer, and a test asserts the absence.** An AST
  guard names the admission surface and fails the moment any of it reaches the
  demand plane. `headroom_admits(demand_bytes=)` stays inert until pgw#1601's
  mint-time stamp exists, and the guard is falsified against a module that
  really does import the plane, so it is known to be armed rather than assumed
  to be.

- **pgw#1600: the cross-language corpus earned its keep on its first Go run, and
  the arithmetic changed because of it.** The evaluator originally avoided
  forming `coefficient * value` — it split `c = q*scale + r` and computed
  `q*v + floor(r*v/scale)` on the premise that the declared shape bounds kept
  both products inside int64. They do not: at the corpus's own ceiling row,
  `mp_batch` with a 1 GiB coefficient makes `r*v` 1.3e19, and Go wrapped by
  exactly 2^64/1e6. **Python's bignums could not have found this in a thousand
  green runs** — only a second language executing the same table could. Both
  sides now compute exactly (Python natively, Go through `math/big`) and the
  one shared constraint is that the ANSWER fits int64; a formula whose worst
  case does not is refused loudly rather than wrapped into a small number.
