### Fixed
- The `entrypoints[]` manifest block matches the hub's decode (th#2146). It is
  `functions[]`' successor spelling — a FLAT list of the same item shape, folded
  into `functions` at the one decode site — not a wrapped object with its own
  field names, which failed `ParseManifest` outright. Model slots carry
  `pipeline_class` (read statically from `load()`'s `ctx.load(...)` call),
  `family`, the lane stamps as `layouts`, and `requires=`'s ie#740 floors as
  `layout_requirements` in the machinereq term shape. Adapter slots carry
  th#2140 5c's `kind`/`adapter_kind`/`multiple`.
- The endpoint-lock validator folds `entrypoints[]` the same way and refuses a
  lock carrying both spellings.
