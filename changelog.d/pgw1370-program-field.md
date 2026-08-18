- **The stored graph's digest is named `program`, because that is what the miner reads.** pgw#962
  landed the runtime mint reading `PROGRAM_DIGEST_FIELD = "program"` off each `GraphRecord` and
  raising a typed `MissingProgramDigest` for any hole without one; the derive side had shipped the
  same field spelled `artifact`. The two halves of the blob-in design could not meet — every hole
  would have refused. The consumer is landed and the NAME is the contract, so the producer follows
  it (torchcg tcg#50), and the vendored `GraphRecord` the serving side decodes gains the field
  too. Pre-launch hardcut: no alias, no transitional acceptance of the old key.
