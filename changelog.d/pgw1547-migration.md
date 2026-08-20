pgw#1547 MIGRATION: the `cell` hardcut (Paul, 2026-08-20: *"please do these
renames now, fully. I want it fully renamed. You can go ahead and break
everything. No legacy support, hardcut. We are pre-launch it's fine."*). The
fence's deferred set goes from 83 line-proofs to 6 historical notes.

**`GEN_WORKER_LOCAL_CELLS_DIR` is DELETED, not renamed.** Paul: *"that's a
configuration on where to store them, and I believe we store them in local-CAS,
so you can delete this env entirely and just use whatever env we use to point to
the local-CAS directory."* `store_root()` now derives unconditionally as
`<TENSORHUB_CACHE_DIR>/compiled-graph-store` — one root, no branch, so the store
cannot disagree with the CAS it describes. It leaves the runtime-env corpus (one
variable out, nothing in: a derived path is not a configuration) and the config
allowlist. `procsplit/parent.py`'s dedicated grant entry is DELETED as
*subsumed*, not dropped: the store is now a subtree of the cache root that
`grant_paths` already chowns recursively, so pgw#1349's dropped-child
`PermissionError` is structurally impossible rather than merely fixed — there is
no second knob that can point the store somewhere unowned.

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
WORKER_VALUE_CONTRACTS_DIGEST and COZY_RUNTIME_ENV_DIGEST all bumped.

**A stale pointer fixed on the way past** (pgw#1554): the shared cg-key
conformance corpus named `gen_worker.cell_key.is_key`, a module the worker does
not have. It is `_vendor.torchcg.identity.is_compiled_graph_key`; KEY_GRAMMAR_DIGEST
bumped in both copies.

**Two defects this surfaced, both of the kind a rename is supposed to surface:**
`tests/test_store_corruption_pgw1283.py` still reached the deleted `ENV_STORE_DIR`
(caught by mypy, not by a test run), and `test_pod_isolation`'s
"RELOCATED store" row asserted a scenario that no longer exists — rewritten to
assert the property that still must hold, that relocating the ONE knob keeps the
dropped child able to write its own store.

Proven by a real store round-trip at the new names — root derivation, census,
verdict transitions, the owed-to-sink obligation and the memo shortcut — not by
mocks.
