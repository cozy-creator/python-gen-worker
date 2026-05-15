# `endpoint.toml` reference

`endpoint.toml` declares the entry module + where the resulting image can run
(orchestrator placement). It is a build-time input; it does not need to be in
the final image.

The control plane reads `/app/.tensorhub/endpoint.lock` (the discovery output)
for everything else — per-function `Resources`, model bindings, payload
schemas. Model state lives entirely in Python.

---

## Minimum viable shape

CUDA endpoint (5 lines):

```toml
schema_version = 1
main = "myendpoint.main"

[[build.profiles]]
name = "default"
accelerator = "cuda"
cuda_min = "12.8"
compute_capabilities = [">=8.0"]
```

CPU endpoint (4 lines):

```toml
schema_version = 1
main = "myendpoint.main"

[[build.profiles]]
name = "default"
accelerator = "none"
```

What this declares:

- `main` — Python import path containing your `@inference_function`
  decorators. Discovery scans it at build time.
- `[[build.profiles]]` — at least one routing target. `accelerator` plus
  `cuda_min` + `compute_capabilities` for GPU profiles.

---

## The four build modes

Every profile is a **routing declaration** (`accelerator` + optional CUDA
hints for placement). The image-source dimension is independent:

| Mode                     | Profile fields                                | Image comes from                                    | Dockerfile required? |
|--------------------------|-----------------------------------------------|-----------------------------------------------------|----------------------|
| **Generated Dockerfile** | no Dockerfile + `python` / `torch` / `cuda` or `base_image` | Tensorhub generates a standard Dockerfile           | No                   |
| **Managed**              | Dockerfile + `python` / `torch` / `cuda`      | Tensorhub resolves via lookup table                 | Yes                  |
| **Explicit**             | Dockerfile + `base_image`                     | Tenant names the image ref directly                 | Yes                  |
| **Fully custom**         | Dockerfile + `accelerator` only (no build hints) | Whatever's in the tenant's `FROM` line              | Yes                  |

### Generated Dockerfile

Tensorhub generates the Dockerfile from the profile and project files when the
tarball has no root `Dockerfile` and the profile has build hints. Use this for
the common case where the endpoint can be installed with pip or `pyproject.toml`
metadata:

```toml
[[build.profiles]]
name = "cpu"
accelerator = "none"
python = "3.12"
dependencies = ["gen-worker>=0.7.5", "msgspec"]
```

There is no `kind` selector in `endpoint.toml`; add a Dockerfile only when you
need to own the base image, build stages, package manager setup, or shell
commands.

### Managed mode

Tensorhub picks a vetted PyTorch base image from the build hints:

```toml
[[build.profiles]]
name = "cu128"
accelerator = "cuda"
cuda_min = "12.8"
compute_capabilities = [">=8.0"]
python = "3.12"
torch = "2.12.x"
cuda = "12.8"
```

### Explicit mode

You name the base image; the build args are still injected:

```toml
[[build.profiles]]
name = "nvcr"
accelerator = "cuda"
cuda_min = "12.8"
compute_capabilities = [">=8.0"]
base_image = "nvcr.io/nvidia/pytorch:25.01-py3"
```

`base_image` accepts any ref — public registries, private ECR, your own
ghcr.io, etc. (Docker still needs pull credentials in your build environment.)

### Fully custom mode

Routing only; tensorhub does not resolve or inject a base image:

```toml
[[build.profiles]]
name = "custom"
accelerator = "cuda"
cuda_min = "12.8"
compute_capabilities = [">=8.0"]
# no python / torch / cuda / base_image — your Dockerfile's FROM wins
```

The profile's `accelerator = "cuda"` still drives orchestrator placement onto
a GPU host with `cuda >= 12.8`. The build step is opaque to tensorhub.

---

## Placement fields — always relevant

These describe what hardware the image needs to run on. The orchestrator reads
them to decide which workers can host this profile.

| Field                  | Required?                  | Meaning                                                                                  |
|------------------------|----------------------------|------------------------------------------------------------------------------------------|
| `accelerator`          | always                     | `"cuda"` or `"none"`                                                                     |
| `cuda_min`             | if `accelerator="cuda"`    | Minimum host CUDA driver version (e.g. `"12.8"`)                                         |
| `compute_capabilities` | if `accelerator="cuda"`    | Minimum GPU SM version constraint(s) (e.g. `[">=8.0"]`)                                  |
| `cpu_arch`             | optional, default `"x86"`  | `"x86"` (alias: `"amd64"`) or `"arm64"`                                                 |
| `os`                   | optional, default `"linux"` | OS family the image expects (`"linux"` / `"windows"` / `"macos"`)                       |

Per-function `Resources.min_vram_gb` drives VRAM placement decisions — that's
a Python-side declaration, not a `endpoint.toml` field.

---

## Build hints — only for managed-mode image resolution

These describe the contents of the image being built. Tensorhub only consumes
them when picking a managed base image; they're irrelevant to placement.

| Field         | Used for                                                                                  |
|---------------|-------------------------------------------------------------------------------------------|
| `python`      | Which Python version the managed base image bundles                                       |
| `torch`       | Which PyTorch version the managed base image bundles                                      |
| `cuda`        | Which CUDA version the managed base image is built against (separate from `cuda_min`)     |
| `image_kind`  | `"runtime"` vs `"devel"` flavor of the managed image                                      |
| `base_image`  | Explicit override; bypasses the managed lookup entirely                                   |
| `dependencies` | Extra pip requirements for generated Dockerfiles                                        |
| `system_dependencies` | Debian packages installed by generated Dockerfiles                                |

`cuda` (build) and `cuda_min` (placement) typically match but can differ — for
example, you might build against CUDA 12.4 yet require host driver >= 12.8.

---

## `BASE_IMAGE` / `CUDA_VERSION` / `TORCH_VERSION` / `PYTHON_VERSION` injection

What tensorhub passes to `docker build` as build args, by profile shape:

| Profile shape                                                | `BASE_IMAGE` | `CUDA_VERSION` / `TORCH_VERSION` / `PYTHON_VERSION` |
|--------------------------------------------------------------|--------------|------------------------------------------------------|
| `accelerator` + `python` + `torch` + (`cuda`)                | yes (resolved) | yes                                                |
| `accelerator` + `base_image`                                 | yes (literal)  | yes if `python` / `torch` / `cuda` also set        |
| `accelerator` only (custom)                                  | **no**         | yes if any of those fields are set                 |

Your Dockerfile decides which args to consume. Unused build args are silently
ignored by docker; if you use `ARG NAME` in the Dockerfile and the arg isn't
passed, it falls back to the `ARG NAME=default` declaration.

A managed-mode endpoint can pin a specific CUDA at install time without
hardcoding it:

```dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG CUDA_VERSION
RUN pip install some-package-pinned-to-${CUDA_VERSION}
```

A fully-custom endpoint doesn't use `BASE_IMAGE` at all:

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.01-py3
# no ARG BASE_IMAGE — tensorhub didn't inject one, your FROM wins
```

---

## Multi-profile endpoints

One endpoint can ship multiple profiles. The orchestrator picks one per
placement based on the host's capabilities.

```toml
schema_version = 1
main = "myendpoint.main"

# CUDA 12.8 managed
[[build.profiles]]
name = "cu128"
accelerator = "cuda"
cuda_min = "12.8"
compute_capabilities = [">=8.0"]
python = "3.12"
torch = "2.12.x"
cuda = "12.8"

# CPU fallback
[[build.profiles]]
name = "cpu"
accelerator = "none"

# ARM64 CPU
[[build.profiles]]
name = "cpu-arm"
accelerator = "none"
cpu_arch = "arm64"
```

Each profile is built into its own image. Tensorhub stamps the resolved
metadata onto the release; gen-orchestrator routes invocations to a worker
that satisfies the per-profile placement fields.

---

## What lives where

Model state, per-function hardware envelopes, and cost-shape declarations live
entirely in **Python** on the `@inference_function` decorator. `endpoint.toml`
carries only:

- `schema_version` + `main`
- One or more `[[build.profiles]]` entries, each with placement fields and
  optional build hints

Anything model-shaped or function-shaped goes in `main.py` — see
[endpoint-authoring.md](endpoint-authoring.md) for `Resources`, `Repo`,
`dispatch`, and `models={...}`.
