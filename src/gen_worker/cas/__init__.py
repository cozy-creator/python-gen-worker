"""The worker's own half of the content store.

`_vendor/tensorfs` is upstream's bytes at one rev. THIS package is what pgw
owns and upstream does not implement in Python: the object planner (a port of
the Rust planner, because pgw#1310 rules a compiled extension out of a
source-vendored wheel), admission built on top of it, and retention.

pgw#1575 is why the split is here rather than smuggled into the snapshot.
Before it, `_vendor/tensorfs` pinned its write half to `8bafdfbb` — a rev with
NO common ancestor with tensorfs master at all — purely to keep
`LocalCAS.ingest_file`, `ingest_repository` and `collect_garbage` alive, plus a
`planner.py` that existed at no upstream rev whatsoever. Every one of those is
this repo's code answering this repo's question, and none of them was ever a
reason to hold a dead lineage.

Whole-directory admission is NOT here, and that is the same audit's other
answer: nothing in `src/` has ever called it. It is a fixture, and it lives at
`tests/cas_fixture.py`.
"""

from .admission import ingest_file
from .planner import BLOB_V1, GGUF_V1, SAFETENSORS_V1, Plan, Region, plan, plan_chunks
from .retention import GCReport, collect_garbage

__all__ = [
    "BLOB_V1",
    "GGUF_V1",
    "GCReport",
    "Plan",
    "Region",
    "SAFETENSORS_V1",
    "collect_garbage",
    "ingest_file",
    "plan",
    "plan_chunks",
]
