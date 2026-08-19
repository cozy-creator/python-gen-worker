- **A trace no longer needs a contract document, and an empty lane tuple no longer disables
  compilation in silence.** Paul's ruling ("A NORMAL TRACE MUST JUST WORK"): point the tracer at
  the endpoint's model and payloads, trace, compile, run. Two things stood in the way and both
  are gone.

  **Contracts are metadata, not a gate.** A model class that names no tensorfs contract used to
  be refused by name at publish — *"omits lanes= and its model type has no canonical contract
  yet"* — which cost the anima lane a throwaway one-tensor contract document invented purely to
  be ALLOWED to run `torch.export`. Such a class now gets a DERIVED lane, `derived.<model
  type>@1`, computed the same way at trace and at serve so both address the same row, and its
  load dtype comes from the checkpoint itself when no contract states one. A contract document
  attaches to the produced artifacts later, as fleet naming/pricing metadata.

  **Nothing rekeys.** A lane handle is a NAME: `cg-graph-v1` hashes the canonical trace plus its
  ingress and passes, `cg-key-v1` is (graph, sm, toolchain), and the contract string is in
  neither. Measured, not argued — a fixture identical to another but for its emptied `lanes=`
  derives byte-identical graph hashes under a different lane name, and re-locking the real sd1.5
  endpoint reproduces its document digest exactly.

  **Eager-forever is now a word, with a reason.** `eager_only="<why>"` on the class header, the
  mandatory-reason pattern `self_loading=` already uses. `lanes=()` no longer means it: an
  absent lane declaration says "no layout contract stated", and that traces. This closes
  pgw#1469, where `ctx.compile` under `lanes=()` was byte-identical-digest dead code — `load`
  was never called, so nothing observed the mark and the author got a green lock proving
  nothing. A compile mark under `eager_only=` is now a declaration refusal, read statically from
  the AST with no author code run.

  Every outcome `lock` can reach is printed as one word — `traced`,
  `traced-no-compile-targets`, `eager-by-declaration`, `weightless` — and carried in its JSON
  summary. Silence is not a posture.

- **`gen-worker lock --check`** — the freshness gate for a lock that is committed source. It
  re-derives only when the inputs moved, compares the DOCUMENT (an input that moves without
  moving the output is not drift), writes nothing either way, and exits 1 on real drift.
