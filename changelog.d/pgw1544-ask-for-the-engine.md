- **pgw#1544: `ctx.load` asks for the streaming engine itself, and the refusal
  stops asserting a cause it never measured.** `engine_for` had two production
  call sites — `EndpointHost` (the local CLI and the daemon) and pgw#1543's
  repair. The serverless worker builds `ServeLoop` with no `engine=`, so on a
  POD nothing ever asked: every projected checkpoint fell to the eager bridge
  and refused on every request. The refusal then blamed the store, because
  `_projection_declined_because` returned "the manifest pin is MISSING" as the
  else of two structural checks without ever looking — so a pod whose pin was
  present and perfectly resolvable produced `pin ... is MISSING | repair
  attempted: not needed: already pinned`, one string contradicting itself. The
  engine is now asked at the one place that always has the tree, the pin is
  measured ONCE and handed to both the decision and the sentence, and
  "already pinned" no longer reads as "cannot serve".
