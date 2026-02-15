# Terminology (Canonical)

This repo follows Cozy's canonical naming. There is no backward-compat layer for legacy names.

- `owner`: publishing namespace (an org slug in URLs; canonical ID is an org UUID).
- `endpoint`: published unit (source code is published as endpoint releases).
- `function`: invokable unit inside an endpoint release.
- Invoke reference: `owner/endpoint/function[:tag]` (default tag is `prod`).
- `release_id`: immutable identifier for a published endpoint release.
- `invoker` / `invoker_id`: the identity performing an invocation.
- `artifacts`: uploaded source code and endpoint-owned build inputs/outputs.
  - Cozy Hub stores endpoint source tarballs in the endpoint artifacts bucket (`s3.endpoint_artifacts.*` / `S3_ENDPOINT_ARTIFACTS_*`).
