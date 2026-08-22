- **A diffusers repack that is a KEY MAP, not a model load — `outputs=[{repack: "<family>"}]`
  on a clone job.** A mirror reproduces the upstream layout, and a flat root beside a
  `config.json` whose `auto_map` names `.py` files that do not exist cannot be `ctx.load`ed
  hollow-legally: the hollow session intercepts only diffusers/transformers loaders, so an
  endpoint's component classes have to be reachable through a `model_index.json`. The repack
  this repo already had could not do it — `repackage.singlefile_to_diffusers` instantiates a
  DECLARED diffusers pipeline class from a single file and saves it back, which needs a class
  the SDK can name and a card's worth of RAM, and SenseNova's serving class is
  `sensenova_u1.Model` in an endpoint package. `convert/tree_repack.py` is the transform the
  ruling actually asked for: it routes safetensors KEYS into component directories, derives each
  component's `config.json` from the source config field by declared field, moves the tokenizer
  files into their own directory and writes `model_index.json`. There is no torch import in the
  module. Tensor DATA is copied as byte RANGES out of the source header's own offsets, so every
  tensor in the produced tree is bit-identical to the one it came from and no dtype or shape can
  move here — asserted per tensor, before and after, and red when one byte of the copy loop is
  perturbed.
- **It composes with the cast in ONE submission and it MOVES what it can.** The repack is the
  last leg of `build_flavor_tree`, so the 50 GB read the cast already pays is not paid twice. A
  declaration whose key map is the identity and whose single weight component takes every key
  moves each member with `os.replace` — zero bytes read, zero written — and the disk preflight
  prices that property (`TreeRepack.is_pure_move`) rather than assuming either answer. Assuming
  a copy would refuse a job that fits, which is pgw#1666's under-count pointing the other way.
- **A requested repack can never be satisfied by publishing the source unchanged.** `spec_actions`
  now treats a repack as WORK: without that, a source whose dtype already matched took the
  `PUBLISH_SOURCE` arm, handed the ingest tree straight to the publisher and stamped it with a
  `tree_repack` attribute it had not earned — the same silent-substitution shape that cost this
  checkpoint pgw#1668 and pgw#1669.
- **Members are PRESERVED and the produced layout is READ BACK, never echoed.** N source members
  become N component members with an index; the repack neither shards nor de-shards, and
  `file_layout` comes from `observed_file_layout` on the produced tree. ⚠️ On a clone the CAST has
  already collapsed a shard set by the time the repack runs (`stream_reencode` writes one output
  file per weight set), so a 13-shard source publishes one member — that is the cast's doing, is
  asserted as such, and belongs to pgw#1669's axis.
- **Everything that could silently drop a tensor is a REFUSAL, most of them at declaration time.**
  Two catch-all components, a catch-all that is not last, a component that carries neither weights
  nor files, two source keys renaming onto one, a key no component claims, a config field the
  source document does not carry, a declared tokenizer file that is not there, a tree that is
  already component-shaped, and a tree that carries none of the declaration's required key
  prefixes. An unknown family name is refused by `normalize_outputs`, i.e. on the request, before
  a pod exists — and the refusal names what IS declared. A repack is NAMED by the request and
  never detected: a wrong key map produces a tree that loads and serves noise.
- Declared for `sensenova-u1.mot` (se#840): one `transformer` component whose key map is
  deliberately the identity — the upstream keys already ARE the endpoint's module paths, and what
  the repack supplies is the directory shape, the two config documents and the `model_index.json`.
