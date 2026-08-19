- **pgw#1489: an artifact's environment is its COMPILE STACK — torch, triton,
  `nvidia-*` — and the whole-installed-set closure is deleted, key and audit.**
  The key was a hash of every resolved package, restated at serve time from
  `importlib.metadata`. That is two representations of an environment the
  endpoint's `uv.lock` already pins, unable to agree by construction (pgw#1472
  measured all three reasons: PEP 503 spelling, the `+cu129` local segment a
  lock cannot express, platform-conditional rows), and it split the artifact
  pool on packages no compiler can see — 43-package diffs between envs that
  serve identically, and a docs extra invalidating every compiled graph on the
  box. Measured on tonight's real artifacts: under the old key an irrelevant
  four-package diff turned 14/14 sd1.5 graphs into holes and the adopt audit
  refused; under the new key the same diff adopts 14/14 with a byte-identical
  key, while `sm_120` still refuses BY NAME and a `torch` bump still
  invalidates all 14. sd1.5 and SDXL, which had different closures, now share
  one env key because they pin one compile stack. Artifact positions move to
  `cg-env-v2` with no compat alias (pre-launch); the local rekey re-positioned
  32 artifacts + 32 manifests and re-minted nothing, because the CAS is
  content-addressed and only the ref moves. Driver floor and measured peak-VRAM
  remain admission metadata at adopt, never key inputs, and
  `scripts/lint_no_installed_set_keying.py` keeps the installed set out of the
  key path — one allowlisted diagnostic that returns strings for a log line.
