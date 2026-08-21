# checkpoint-configs — z-image

**Config files ONLY. No weights, ever.** Every file here was copied VERBATIM
from the tree the fleet actually serves — `tensorhub/z-image` checkpoint `sha256:e2b57d69fbfe…` (a clone of
`Tongyi-MAI/Z-Image@04cc4abb7c50`) — read through the hub's resolve API and filtered to
`*.json` under 1.5 MB, with the `*.safetensors.index.json` shard maps excluded
(they are weight maps, not configuration).

Two consumers, and they are why the tree has to be COMPLETE rather than just
the scheduler:

1. `tests/test_drive.py` builds the REAL `FlowMatchEulerDiscreteScheduler` from
   `scheduler/`, so the scheduler the drive exercises is the checkpoint's own
   rather than a library default.
2. `gen-worker release derive` enumerates the graph set against this tree
   (`author-ci.toml [derive] checkpoint_configs`, se#748 tier 2). A
   lanes-declaring endpoint without it is a `lint_author_derive` refusal.

`shift: 6.0` in `scheduler/` is the BASE checkpoint's; the served Turbo tree
ships the same class at `3.0`.

To refresh: re-read the pinned checkpoint through
`/api/v1/repos/<org>/<name>/resolve?release=prod&digest=…` and copy the JSON
back. Pin the digest — `?release=prod` alone answers `409 release_ambiguous` on
`qwen-image`.
