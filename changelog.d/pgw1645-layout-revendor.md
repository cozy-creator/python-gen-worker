- **pgw#1645: `declared_input_layout` arrives — the whole fleet re-mints, and that is
  the point.** The vendored torchcg moves `aa051792` → `1ade7222` (tcg#83 + tcg#85), which
  makes the byte layout an artifact was compiled against a MANDATORY key axis: `cg-key-v2`
  → `cg-key-v3`, `cg-env-v3` → `cg-env-v4`. **Every compiled artifact that exists today is
  a store miss under the new key and every pod re-mints.** Stated up front rather than
  discovered from a cold cache: it is the same price tcg#80 paid, deliberately, pre-launch,
  and it buys the property the whole layout-morphism program stands on — an artifact and
  the bytes it binds can no longer disagree about their arrangement without saying so.

- **pgw#1645: the binder stopped silently repacking, so a mismatch is now a refusal.**
  `runner.bind`'s unconditional `.contiguous()` is gone upstream. Bytes in the layout an
  artifact declares bind by reference at zero copies; bytes in any other layout raise
  `ConstantBindingError("layout_mismatch", ...)` naming both sides. The same question is
  asked at MINT, before the compile is paid for. Real checkpoints are row-major and the
  default declaration is `torch.contiguous@1`, so nothing in the fleet changes shape — but
  a derive that hands the mint a non-contiguous tensor, which used to buy a silent
  full-weight copy at bind, now refuses loudly.

- **pgw#1645: the vendored tensorfs moves `ac9c9d4c` → `201c32e1` and carries the RATIFIED
  layout catalog.** The Python surface is byte-identical at both revs; what actually
  arrives is `spec/v2/layouts/*.json` — seven layout-morphism records, each a permutation
  written as DATA with its provenance, auto-ratified by applying the map and its inverse
  rather than by somebody reading it. Two of them (`cublas.blockscale-128x4@1`,
  `nunchaku.micro-scale@1`) were transcribed from this repository's own
  `nvfp4_quant.py` and `svdq_layout.py`, which is why they exist at all: reading one nvfp4
  packaging as the other measured LPIPS 1.11, with every name, dtype and shape correct and
  every number wrong.

- **pgw#1645: `canonical_ingress` and its stride canonicalizer are DELETED.** That boundary
  forced every tensor entering a compiled callable to row-major contiguous strides — which
  is exactly the transform this program exists to stop doing by hand, and it would have
  silently undone a delivered layout at full copy cost. It cost nothing to remove: the
  wrapper had ZERO callers repo-wide, so the deletion changes no behaviour. Its dtype-drift
  assert is not lost in spirit — the binder's typed refusal is the wired form of "these
  bytes are not what this artifact was compiled for". The guard classifier no longer cites
  an ingress pin that does not exist.
