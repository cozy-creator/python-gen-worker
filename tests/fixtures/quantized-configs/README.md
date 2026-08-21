# Quantized checkpoint-config corpus (pgw#1638)

Configs only — never weights, exactly like `checkpoint-configs/`. The
`test_the_corpus_carries_no_weights_pgw1633` guard covers this tree too.

## Why it is a SECOND corpus and not a row in `checkpoint-configs/`

`checkpoint-configs/` is the FLEET roster, and every entry in it must name a
pipeline class this image's diffusers carries — `minimax-h3` names
`MiniMaxH3ModularPipeline`, which is the endpoint's own class and lives in the
`serverless-endpoints/minimax-h3` wheel, not in diffusers. Adding it there would
turn the roster check into a lie about what a fleet fixture is. The quantized
question is also not the fleet question: it is asked per COMPONENT CONFIG, and
one component is the whole article.

## `minimax-h3/text_encoder/config.json`

Byte-identical to
`serverless-endpoints/minimax-h3/tests/fixtures/checkpoint-configs/text_encoder/config.json`,
which came from `tensorhub/minimax-h3@serve-narrowed`'s own tree over the hub's
`resolve` route. It is the config that paid for pgw#1638 on an H200:

    "quantization_config": {"quant_method": "fp8", "fmt": "e4m3",
      "weight_block_size": [128, 128], "activation_scheme": "dynamic",
      "modules_to_not_convert": ["model.visual", "lm_head"]}

with `text_config.num_hidden_layers = 51`. **51 layers x 7 quantized linears =
357**, which is exactly the orphan count the pod reported, so the suite's
numbers are the incident's numbers rather than a shape chosen to pass.

`model_index.json` carries the REAL `text_encoder` entry from that tree (the
modular three-element form, `subfolder: "text_encoder"`), so the index walk
under test is the one production runs.
