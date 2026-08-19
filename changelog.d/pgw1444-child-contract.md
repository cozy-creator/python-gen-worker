- **The v2 mint's compile child now proves it is the parent, and refuses before it compiles.**
  `serving/mint.py` spawns `python -m gen_worker.serving.mint_child`, which produces the bytes the
  parent then publishes and arms — while every decision around it runs in the parent. That
  assignment is only sound while both are the same code, and `-m` resolves whatever the
  interpreter's path yields (a second checkout, an inherited `PYTHONPATH`, a stale wheel, or this
  tree edited between the mint's import and the spawn). pgw#840 documented the hazard for the v1
  pool; the v2 path shipped with **no protection at all**. The parent now stamps a digest of the
  contract source into every request and the child recomputes and compares — in the CHILD, before
  any work, so skew costs nothing.

- **A guard that lost its digest source now says so instead of turning off.** `cd46c957` deleted
  `aot_compile_child.py`; `aot_compile_pool._CONTRACT_MODULES` kept naming it; `_code_digest()`
  returned `""` through a branch meant for zipimport; and `_verify_child_code`'s
  `if not CODE_DIGEST: return` became a silent no-op — the pgw#840 check disabled by an unrelated
  deletion, still looking like a guard. An absent NAMED contract module is now distinguished from
  "no source anywhere" and refuses. Recorded rather than raised at import, because the module is
  import-time reachable and a raise there would take the import with it.
