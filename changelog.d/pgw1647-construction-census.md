- **pgw#1647: what a module IS becomes RELEASE-BUILD DATA, and the serve fence
  REPLAYS it instead of re-deriving trust.** Four defects were paid for on rented
  hardware in three days, each fixed as a symptom: pgw#1626 (`tie_weights` never
  ran), pgw#1638 (the config's quantizer never ran — 357 orphan
  `weight_scale_inv`; and neither did `model.eval()`, so 44/44 fleet components
  served with dropout armed and five T5/UMT5 conditioners randomized their
  conditioning on every request), pgw#1644 (the whole-module `.to(device)` never
  ran — three non-persistent RoPE buffers on the CPU under an all-CUDA model,
  $0.89, `mat1 is on cpu` raised inside `diffusers` 8 ms into a forward). One
  sentence, four endings: *the meta skeleton is built from the config alone, so
  the `from_pretrained` machinery never runs.*

  Every one was invisible to the fence that should have caught it, because that
  fence walked the checkpoint CONTAINER. A container names the tensors a
  checkpoint carries and can never name a tensor the CODE creates — a tie, a
  quantizer's scale grid, a computed `inv_freq`.

  The CONSTRUCTION CENSUS states the module's whole tensor identity as data:
  every parameter and buffer, persistent and non-persistent, with shape and
  dtype; the tied alias groups by object identity; the module classes the
  config's quantizer swapped in and the tensors that swap owns; and eval mode.
  It is computed at RELEASE BUILD inside the endpoint image on the config-only
  tree (the pgw#1370 derive seam), rides the release document
  (`construction_census`, digest-addressed, forwarded by the hub verbatim —
  th#2281), and is REPLAYED by the serve-time fence, which now walks the MODULE.

  Five named invariants, each with a red arm that neuters one step of the
  prepare seam: **I1** ties (identity, not `_tied_weights_keys`), **I2** the
  quantizer's swap, **I3** eval mode, **I4** placement (nothing on meta, nothing
  off the target — params AND buffers), **I5** totality — set equality in BOTH
  directions, so a construction side effect nobody has written down yet is a $0
  publish refusal instead of the fifth rental. One predicate, three moments:
  release build (refuse the release), the CPU-only conformance suite (catch
  image-bump drift), serve after the fill (catch store corruption and fill
  defects, which is the only jurisdiction serve cannot delegate).

  **Hardcut**: the container-walking fence and `_place_uninstalled` are DELETED,
  not flagged — no fallback arm, no dual-run window. A census computation that
  crashes at release build FAILS the build with a typed refusal naming the tree,
  never a soft row-marking. Refusals name the invariant and the first offending
  tensor, so pgw#1626's "the checkpoint does not carry…" — a loader defect
  blamed on the checkpoint — is unwritable.

  The prepare seam is now ONE enumerated pipeline (`skeleton.PREPARE_STEPS`,
  `build` + `finish`) and a source guard refuses a preparation step added
  anywhere else. `skeleton.build_modules` answers a `Skeleton`, so the census
  reads the quantizer's own record rather than a second derivation of it, and
  the meta build now reties on meta — `init_empty_weights` breaks the tie
  `post_init` established, which is why the skeleton was silent about its own
  tie structure.

  pgw#1633's suite is the test home: census round-trip for every fleet pipeline
  class, plus the `minimax-h3` fixture, which is the RoPE-buffer case AND the
  fp8-blockwise case at once because it is the incident. CPU-only, no card, no
  weights, no downloads.
