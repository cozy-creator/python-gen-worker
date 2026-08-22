- **pgw#1645: the layout-rung vocabulary stopped claiming a wire contract that does not
  exist.** `LayoutState`'s docstring said "the hub groups activity rows on them", copied
  from `EagerPhase`, where it is true. Traced: the rungs reach `ServeAdoption.facts()` →
  `worker.mint_facts()` → a `logger.debug` call, and no hub query groups on them. The
  spellings are still worth freezing — they are cheap to fix now and unfixable once a query
  does exist — but the docstring says INTENDED, and says plainly that nothing consumes them
  yet. Found by auditing the merged change rather than by a test, which is the only way this
  class of defect is ever found.

- **pgw#1645: the decline names the lanes that are actually building a fill client.** It
  named varena#13, filed hours earlier on the belief that a varena-side `FillSink` was the
  only route to a Python-reachable fill. tensorfs#159 and pgw#1648 were already building
  one over a raw destination address, and the coordinator has since ruled that form
  sufficient for tensorfs#154's acceptance (b), re-classifying varena#13 to non-gating
  mechanism-hardening. The verdict was right; the reason was pointing at the wrong door.

- **pgw#1645: `fill_path()` now says out loud that it is aimed at the wrong surface.** It
  probes the vendored tensorfs for a `fill` attribute; the real client is
  `serving.streaming.fill_client.client_for` over the native `CudaFillClient`, and this
  probe could never go true regardless, because that class lives in the COMPILED extension
  and the vendored tensorfs deliberately carries none (pgw#1310). It is left aimed here and
  labelled rather than pointed at a module that does not exist yet: the verdict it produces
  is correct either way — there is no fill in reach — and a probe that lies about which
  absence it found is worse than one that names its own aim.
