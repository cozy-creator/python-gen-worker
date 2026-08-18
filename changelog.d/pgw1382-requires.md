### Added
- `Model[MT]` class headers take `requires={contract: "vram12g"}` — the ie#740
  machine floor, per lane, statically extractable at publish. The release
  derive records it on each lane contract, so a deployment is sized without
  running author code. Omitting it leaves placement undeclared.
