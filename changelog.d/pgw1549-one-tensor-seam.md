### One tensorfs-aware seam turns checkpoint files into tensors

The ~21 h fleet serve outage of 2026-08-19 had one shape: a capability that two
callers both needed was added to each of them separately, and the pod's caller
was the one that got missed. `EndpointHost` asked for the streaming loader,
`ServeLoop` never did, and every local test passed because every local test was
the caller that asked.

- **pgw#1549** — `EndpointHost` no longer builds a loader engine. `ctx.load` is
  the sole binder (the one place that always has the tree), and both hosts now
  build their `LoadContext` through one `serving/worker_context.py` factory. In
  the process this fixes a live pod defect of the same shape: `ServeLoop` named
  no placement device, so pgw#1452's fix never reached a pod and every eagerly
  bridged pipeline there ran on the CPU.
- **pgw#1550** — `scripts/lint_tensor_read_seam.py` (`fast gates`) fails any raw
  `safe_open` / `load_file` / `torch.load` on a checkpoint path, in gen_worker
  and in the sibling endpoint checkout, AND a second `LoadContext` builder
  anywhere in `src/` — consolidating the two hosts is only worth something if a
  third cannot appear. Exemption is a proof at the line with a mandatory reason. The quantizer producers now read through the seam;
  `load_from_pretrained` and the skeleton's passthrough branch refuse typed.
- **pgw#1551** — `tests/test_pod_serve_loop_streams.py` builds `ServeLoop`
  exactly as `worker.py` does on a pod and serves a request off a real projected
  tree with real stubs on a real `ModelStore`. No mocks on the seam. Reverting
  the engine ask turns it red with the fleet's own refusal.
- **pgw#1552** — `docs/projected-trees.md`: the stub contract, the one reader,
  every other reader's obligation, and the catalogue of wrong inferences a stub
  invites.
