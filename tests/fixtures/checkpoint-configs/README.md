# Fleet checkpoint-config trees (pgw#1633)

Config-only copies of the `[derive] checkpoint_configs` fixtures each endpoint
ships in `serverless-endpoints/<endpoint>/tests/fixtures/checkpoint-configs`.
**No weights, ever** — the copies are verbatim, and each subdirectory keeps the
endpoint's own `README.md` naming the upstream repo and revision it was
transcribed from.

They are vendored rather than read across repos because the suite that uses
them is pgw CI, and pgw CI checks out pgw. A cross-repo read would mean a
private-repo token in the workflow and a suite that silently stops running the
day the token expires.

## What reads them

`tests/test_skeleton_conformance_pgw1633.py` — meta-build every weight-bearing
component, fill by the key set a saved checkpoint would carry, `retie()`,
assert nothing stays on `meta` and every alias IS its source. Seconds, no GPU,
no bytes.

## Coverage, and what is NOT covered

The roster lives in `FLEET` in that test file and is checked against this
directory, so a fixture that disappears turns the suite red instead of shrinking
it. Four fleet endpoints are deliberately absent, each for a reason that is a
property of the endpoint, not an oversight:

| endpoint | why not here |
|---|---|
| `minimax-h3` | its index names `diffusers.MiniMaxH3Scheduler`, which this image's diffusers does not carry |
| `internvl-U` | its index names endpoint-local classes (`internvlu.diffusion.*`) that exist only inside that endpoint's image |
| `hidream-o1-image` | a flat transformers tree with no `model_index.json` — `skeleton.build` is not its loader |
| `trellis-3d` | not a diffusers pipeline tree (`pipeline.json`, `ckpts/*.json`) |

The tie exposure `hidream-o1` carries (Qwen3) IS covered, through
`flux.2-klein-4b`/`-9b`, whose `text_encoder` is the same `Qwen3ForCausalLM`.

## Keeping them honest

These are architecture configs: they change when an endpoint repins a major,
which is rare and deliberate. When one does, re-copy the endpoint's fixture
directory whole. Nothing here is edited by hand.
