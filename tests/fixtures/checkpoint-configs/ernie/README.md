# ernie checkpoint configs (fixtures)

**CONFIG ONLY — no tensors, no weights, 224 KB.** Two consumers: the
`[derive] checkpoint_configs` tree `gen-worker release derive` runs against
(se#748 tier 2, required of every lanes-declaring endpoint), and the real
`FlowMatchEulerDiscreteScheduler` the drive suite clones per request.

Fetched VERBATIM from the hub's revision-pinned clone `tensorhub/ernie-image@prod`
(checkpoint `sha256:e8cd9241cfa57d2115f7bfdefd98fde5cd77a5f2fd40e586e511f7b807c05f0d`,
cloned from `huggingface:baidu/ERNIE-Image@5346b31d68c9c23758ba56ef8be5e9dc174c7f99`).
That is the packaging the fleet actually serves, not a hand-written approximation.

## What is deliberately absent

The three ~17 MB `tokenizer.json` blobs. They are vocabulary data, not the
component config the derive reads, and committing 51 MB to make a KB-scale check
pass is the opposite of this fixture's point. The `*.safetensors.index.json`
files ARE included — those are the shard weight MAP, not shard bytes.

## One value worth reading before you touch it

`scheduler/scheduler_config.json` carries **`shift: 4.0` with
`use_dynamic_shifting: false`** — deliberately NOT flux.2-klein's 3.0/true.
ERNIE is flow-matching too, so filling this by analogy with the nearest
flow-matching family looks reasonable and would silently change this family's
sigma ladder. The analogy is actively invited: ingest currently MISCLASSIFIES
both ernie checkpoints as flux (`model_index.json` declares
`vae: AutoencoderKLFlux2`, and `_class_name ErnieImagePipeline` is not in the
family map). It is byte-identical on `ernie-image-turbo`
(`sha256:96031c39fcd4651ae...`, 482 B), so one fixture serves both deployments.
