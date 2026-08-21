- **pgw#1590: serving admission charges the LANE'S OWN DECLARATION, not the whole
  checkpoint tree at its stored precision.** `ResidencyManager.lease` sized every
  instance from `tree_bytes(whole repo) x 1.25` and threw the `lane` argument
  away, so a multi-component repo charged each lane for every other lane's bytes
  at a precision the load path may not keep. On a real H100 pod it refused
  `minimax.h3-dit-diffusers@1` as needing 180,063,706,300 bytes against
  84,368,556,032 — `JOB_STATUS_FATAL` on the exact shape six pods of this fleet
  had served on single H100s two weeks earlier, because H3's DiT is `quantize_()`
  d to w8a8 inside `setup()` and no manifest can see that.

  A lane's declared VRAM floor (`lanes={contract: "vram78g"}`) now CAPS the
  charge. That number is a declaration in the author's own class header, already
  statically extracted, and already what the hub filters placement on — charging
  more than it was the incoherent position: "the platform placed me on a card
  sized by this floor, and I refuse to run because I need more than the floor."
  The cap is `min`, so it can only ever lower a charge and no lane admitted
  before is refused now; a lane that declares nothing keeps the conservative
  whole-tree charge, and its refusal says which declaration would fix it.
