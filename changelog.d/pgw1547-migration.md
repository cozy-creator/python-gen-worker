pgw#1547 MIGRATION: the `cell` hardcut (Paul, 2026-08-20: *"please do these
renames now, fully. I want it fully renamed. You can go ahead and break
everything. No legacy support, hardcut. We are pre-launch it's fine."*). The
fence's deferred set goes from 83 line-proofs to 3 historical notes.

**`GEN_WORKER_LOCAL_CELLS_DIR` is DELETED, not renamed.** Paul: *"that's a
configuration on where to store them, and I believe we store them in local-CAS,
so you can delete this env entirely and just use whatever env we use to point to
the local-CAS directory."* `store_root()` now derives unconditionally as
`<TENSORHUB_CACHE_DIR>/compiled-graph-store` — one root, no branch, so the store
cannot disagree with the CAS it describes. It leaves the runtime-env corpus (one
variable out, nothing in: a derived path is not a configuration), the config
allowlist, `loader.py`, the docs and the test fixtures.

**It is still granted explicitly in `procsplit/parent.py`, and that is the
interesting part.** Deriving it looked like a reason to drop the grant as
"subsumed" by the cache-root entry. It is not: `grant_paths` mkdirs AND chowns
each path it is handed, and this subdirectory does not exist on a cold pod, so
only naming it gets it created owned by the compute uid. Dropping it reproduced
pgw#1349 exactly — `PermissionError` on the dropped child's first nested sidecar
write — and only `test_pod_isolation` under a real root parent saw it. The
parent restates the dirname (it must stay torch-free) and that row now pins the
two constants together.

**On disk**, `<root>/aot-cells/<ck1>/cell.tar.gz` becomes
`<root>/graphs/<ck1>/graph.tar.gz`. No dual-read, no boot-time move: a pod volume
still holding the old tree is simply not found and re-pays one mint, which Paul
accepted explicitly.

**On the wire**, field and message NAMES move with their NUMBERS UNCHANGED —
`CellLookup`→`CompiledGraphLookup`, `cell_lookups`/`cell_key`/`cell_ref`/
`requested_cell_key`/`cell_snapshot_digest`/`served_cell_ref` to their
`compiled_graph_*` spellings — so the binary wire stays byte-identical and the
two-repo landing window is safe in either order (the property tcg#56 relied on).
`reserved "requested_cell_axes"` is deliberately NOT respelled: a reserved name
exists to keep that exact string un-reusable. Wire VALUES move with the code that
emits them: serving modes `jit_cell`/`aot_cell` → `jit_graph`/`aot_graph`, boot
phases `cell_*` → `graph_*`, claims `cell_read_*` → `graph_read_*`, refusal code
`compile_cell_failed` → `compile_graph_failed`. PROTO_DIGEST,
WORKER_VALUE_CONTRACTS_DIGEST and COZY_RUNTIME_ENV_DIGEST all bumped, bindings
regenerated with the pinned toolchain.

`test_pod_isolation`'s "RELOCATED store" row is re-aimed rather than deleted: the
env it relocated is gone, so it now relocates the ONE knob and asks the CHILD's
environment for the expected root — the split harness overrides
`TENSORHUB_CACHE_DIR` for the child, so computing it from the pytest process was
a second, quieter way for the row to go green for the wrong reason.

Proven by a real store round-trip at the new names — root derivation, census,
verdict transitions, the owed-to-sink obligation and the memo shortcut — driven
through the module's own API rather than mocks.
