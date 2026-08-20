# Projected trees: the stub contract

The one page for anyone whose code can meet a checkpoint directory.

## What is on disk

A **projected tree** is what the worker resolves for every served checkpoint
(`<TENSORHUB_CACHE_DIR>/cas/snapshots/<key>/`). It carries no tensor bytes.

| kind of file | what the path holds |
|---|---|
| non-tensor (config, tokenizer, `.so`, media) | a relative **symlink** into the CAS `objects/` — a real file, readable normally |
| tensor container (`*.safetensors`, …) | a ~128 B **`TFSSTUB1` pointer stub**; the bytes are chunked into the CAS and **no path-based read can reach them** |

The manifest is pinned in the same store under `snapshot:<key>`, so
`(cas, manifest)` is recoverable from the tree's own directory name — which is
its digest. Nothing needs a sidecar, and nothing may key that lookup on a
*ref*: the serving path holds the resolver's `pick.ref`, which need not be the
string the store banked under (pgw#1543).

## The one reader

```python
from gen_worker.models.tensor_source import open_tensor_source, load_state_dict
```

`open_tensor_source(path, why=...)` keeps `safetensors.safe_open`'s shape and
moves the source: `keys()`, `get_tensor()`, `metadata()` behave the same
whether the path is a real file or a stub. `load_state_dict` is the
`safetensors.torch.load_file` replacement. A lane is cut over by changing one
`with` line.

For a whole pipeline the seam is one level up: **`ctx.load(PipelineClass)`**,
which binds the pgw#1380 streaming engine and walks the chunk store straight to
VRAM. There is no `torch_dtype=` (the lane contract *is* the dtype) and no
`.to("cuda")` (placement is the worker's decision, handed down).

A raw `safe_open` / `load_file` / `torch.load` outside the seam is the
pgw#1550 outage and must not come back.

## Every other reader's obligation

A consumer that can meet a projected tree needs a stub-aware branch or an
**explicit typed refusal naming the stub and the remedy**. Falling back to a
default is the defect. The refusals that exist:

- `UnresolvedProjection` — the stub is here and its manifest is not recoverable.
- `ProjectedTreeNotStreamable` — `ctx.load` was handed a projected tree and no
  engine bound.
- `ProjectedTreeNotEagerlyLoadable` — `load_from_pretrained` (the eager
  whole-pipeline loader) met one; it would materialize a second copy of the
  tree to read weights `ctx.load` streams with no file at all.
- `SkeletonError` — a passthrough (non-`nn.Module`) component's directory holds
  a stub, so the stock `from_pretrained` cannot build it.

Third-party code that genuinely needs real files goes through
`models/materialized_view.third_party_dir(path, why=...)` — **tier 3** of the
pgw#1303 ladder, permanent for external binaries (`llama-server -m`,
`gguf.GGUFReader`) and AOT `.so` delivery, and a **defect** for a serving
pytorch endpoint since Paul's 2026-08-19 no-fill ruling.

## The catalogue of wrong inferences (pgw#1308 finding 3)

The stub format's safety property is that a naive `open()` fails **loudly at
the parse site**. That guarantee constrains nothing about what the caller then
concludes, and no change to the format can fix a wrong conclusion. Four have
been paid for:

| the observation | the wrong inference | what it cost |
|---|---|---|
| `SafetensorError: header too large` | **the checkpoint is corrupt** | `store.py` deleted the model on every boot; two days pointed at poisoned volumes and truncated downloads. The tell was that the error is identical for a 3.4 GB model and a 68 GB one — the stub is a fixed size. |
| a header read returns `{}` | **the tensor is absent** | `detect_on_disk_dtype` silently doubled VRAM; `_quantized_layers` stopped detecting a w8a8 artifact as quantized and routed it to the plain bf16 lane. |
| a dangling symlink / unreadable component | **the tree is missing a file** | `skeleton.build` reported `carries no model_index.json` about a tree that has one (pgw#1514). |
| the streaming engine did not bind | **the store is broken / the pin is missing** | the ~21 h fleet outage of 2026-08-19. The pin resolved the whole time; nothing had *asked* for an engine. Three lanes read that sentence and went to look at the store (pgw#1544). |

The shape is always the same: a correct loud failure, read as a fact about the
**bytes** when it is a fact about the **reader**. Before concluding anything
about a checkpoint from a parse failure, call
`gen_worker.models.projection.stub_at(path)`.
