- 🔻 **A lane resolved to a bare digest, and sdxl could not derive at all.** `lane_handle` read
  `contract` then `digest` off the lane object — but the SHIPPED `tensorfs.Contract` (tensorfs#111,
  landed) has no `contract` attribute and its `digest` is a bare 64-hex string, so every lane
  answered `f1455f56…` where `sdxl.diffusers-bf16@1` belonged and torchcg refused it. The code was
  written against the DESIGN of the contract object and never run against one. `stamp` is the
  shipped spelling; `digest` is only ever used `sha256:`-prefixed.

- 🔻 **The lane's layout document travelled as `null`.** The whole point of contract OBJECTS is
  that the full canonical layout ships inside the release metadata, so the platform needs no prior
  knowledge of it — but `Contract.document` is a canonical JSON **string** and the reader accepted
  only a `dict`, so every lane emitted a correct-looking `stamp` beside `"document": null`. It now
  parses the string, and a lane whose document cannot be read is a typed refusal: a stamp with no
  layout behind it is worse than no row.
