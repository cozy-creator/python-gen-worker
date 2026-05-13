# Changelog

## 0.7.0

### New

- Chainable `Repo` + `Dispatch` binding model. Declare model dependencies on
  the decorator's `models={...}` kwarg:
  ```python
  flux = Repo("acme/flux")
  @inference_function(
      resources=Resources(requires_gpu=True, min_vram_gb=22.0),
      models={"pipe": flux.flavor("bf16")},
  )
  def generate(ctx, pipe, payload): ...
  ```
- Payload-driven dispatch via `dispatch(field, table)` — function pins a set
  of picks keyed by a `Literal[...]`-typed payload field.
- `Repo` / `Dispatch` support `.allow_override(*classes)` to permit caller
  substitution within an explicit pipeline-class allowlist.
- Reserved `_models` invocation field — invokers can substitute bindings via
  `{"_models": {"pipe": "owner/repo:tag#flavor"}}` (string or structured
  form). Substitution is atomic.
- `Resources` — merged hardware envelope + cost-shape struct, declared **per
  function**.
- Boot-time self-advertise: the worker compares each function's `Resources`
  against host hardware and marks unavailable functions automatically.

### Breaking

A lot of removed and renamed symbols. There are no compat shims; bare
imports of deleted names raise `ImportError` with a pointer to the new API.
See [docs/endpoint-authoring.md](docs/endpoint-authoring.md) for the full
reference.
