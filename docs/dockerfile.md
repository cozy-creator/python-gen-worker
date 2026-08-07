# Dockerfile contract

You only need a Dockerfile when you want to own the base image, build steps,
dependency manager, caching strategy, or multi-stage layout. For simple
Tensorhub endpoints, omit the Dockerfile and declare build hints in
`endpoint.toml`'s `[[build.profiles]]`; Tensorhub generates the Dockerfile
and satisfies the contract below.

When you do provide a Dockerfile, it is fully yours. Tensorhub does not own this
layer.

You satisfy three contract points; everything else is up to you.

## The three contract points

1. **`gen_worker` is importable in the runtime environment.** Whatever
   dependency manager you use, the resulting image must have `gen-worker`
   installed — `gen-worker[torch]` for PyTorch endpoints, plain `gen-worker`
   for non-PyTorch ones. (Other extras: `vision`, `images`, `signing`.)

2. **Discovery is baked into the image at `/app/.tensorhub/endpoint.lock`.**
   Run discovery during `docker build`:

   ```dockerfile
   RUN mkdir -p /app/.tensorhub \
       && python -m gen_worker.discovery > /app/.tensorhub/endpoint.lock
   ```

   This serializes every `@endpoint` object's `Resources`, bindings, and
   payload schemas. The control plane reads the lock from the built image.

3. **The entrypoint runs `gen_worker.entrypoint`.**

   ```dockerfile
   ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]
   ```

   The entrypoint reads `endpoint.lock`, connects to the orchestrator, and
   serves invocations.

---

## Minimum viable Dockerfile

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install -e .
RUN mkdir -p /app/.tensorhub \
    && python -m gen_worker.discovery > /app/.tensorhub/endpoint.lock
ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]
```

No `ARG BASE_IMAGE`, no version pass-throughs. The endpoint's `pyproject.toml`
pins `gen-worker` and the Dockerfile installs it.

If the Dockerfile only looks like this, prefer Tensorhub's
generated-Dockerfile path:

```toml
[[build.profiles]]
accelerator = "none"
python = "3.12"
```

---

## When to use `ARG BASE_IMAGE`

Use `ARG BASE_IMAGE` when your build profile uses **managed mode**
(declares `python` / `torch` / `cuda`) or **explicit mode** (declares
`base_image`). Tensorhub resolves or accepts the base image and passes it as
a build arg.

```dockerfile
ARG BASE_IMAGE=<a real pytorch/pytorch tag+digest matching your profile>
FROM ${BASE_IMAGE}
WORKDIR /app
COPY . /app
RUN pip install -e .
RUN mkdir -p /app/.tensorhub \
    && python -m gen_worker.discovery > /app/.tensorhub/endpoint.lock
ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]
```

Use a common upstream default matching the profile so local builds behave like
Tensorhub builds. In production, tensorhub overrides it with the resolved /
explicit ref.

Omit `ARG BASE_IMAGE` entirely when your profile is **fully custom**
(no `python` / `torch` / `cuda` / `base_image` declared). Tensorhub does not
inject `BASE_IMAGE` in that mode — your `FROM` line is the only source of
truth.

---

## Pinning library versions for `RUN` steps

If your profile sets any of `python` / `torch` / `cuda`, tensorhub injects
matching `PYTHON_VERSION` / `TORCH_VERSION` / `CUDA_VERSION` build args.
Consume them when needed:

```dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDA_VERSION
SHELL ["/bin/bash", "-c"]   # ${VAR//.} is a bash substitution; RUN defaults to sh
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//.} flash-attn
```

Most endpoints don't need this — the base image already ships the right
version. Add an `ARG` only when a specific `RUN` step depends on it.

If you're already in the image, you can read the version directly at build
time without an `ARG`:

```dockerfile
RUN python -c "import torch; print(torch.version.cuda)"
```

One source of truth: the base image.

---

## Multi-profile builds — one Dockerfile, different args per profile

A single Dockerfile is reused across every build profile.
Tensorhub passes the per-profile build args at build time; your Dockerfile
branches as needed on the args it cares about.

For most endpoints this is a no-op — the same Dockerfile works for every
profile because each profile resolves to a fully-formed base image.

For profiles that need different install steps (e.g. CUDA-specific wheels
vs. CPU wheels), branch on a build arg you set yourself:

```dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG ACCEL=cpu
RUN if [ "$ACCEL" = "cuda" ]; then \
        pip install -e .[torch]; \
    else \
        pip install -e .; \
    fi
```

Add an extra build-profile field convention or use the `cuda` field's
presence as your signal in the build script that drives `docker build`.

---

## Caching: tenant-controlled, with one platform ban

`BUILD_NONCE`, `DEPS_NONCE`, endpoint-specific commit pins — these are your
cache-bust knobs, unrelated to versioning. Layer caching is yours; Tensorhub
imposes no caching strategy.

```dockerfile
ARG BUILD_NONCE=2026-04-23-default
RUN echo "build-nonce=${BUILD_NONCE}" \
    && mkdir -p /app/.tensorhub \
    && python -m gen_worker.discovery > /app/.tensorhub/endpoint.lock
```

### BuildKit cache mounts are refused

A Dockerfile that asks BuildKit for a persistent cache directory is rejected at
publish:

```
400 invalid_tarball
     buildkit cache mounts are not allowed in org Dockerfiles
```

**Why.** The builder is multi-tenant, and such a cache is addressed by an id.
The obvious id is a constant — every org copies the same one off a page like
this one — and a constant id is one mutable directory shared across tenant
builds. Org A seeds a poisoned wheel into it; org B's build mounts the same id
and installs it. That is build-time code injection across a tenant boundary:
the same shape of trust failure as adopting a foreign compile cell, one layer
down.

Install uncached instead — `uv pip install --no-cache`. The ordinary layer
cache still covers the common case, and a cold dependency install is a
build-time cost paid once per release, not a per-pod one.

If build speed later justifies a real cache, the answer is a per-org
**namespaced** id the validator enforces (the id must embed the org; the
validator rewrites or refuses). Never a shared one — speed does not reopen a
cross-tenant channel.

> The validator matches the raw bytes of the whole Dockerfile, **comments
> included**. Do not paste the banned directive into your file even to explain
> why it is banned: an explanatory comment is refused exactly like a live
> instruction. Fail-closed is deliberate here.

---

## The AOT host toolchain — yours to install, the platform's to verify

AOTInductor is the only lane that host-compiles: it emits a C++ wrapper and
links a real `.so`, on the machine running the mint. The dynamo/JIT lane does
not — it emits Triton kernels behind a Python wrapper — so an endpoint that
declares no `compile=` export never needs any of this.

If any of your endpoints DOES declare an AOT export, the image must carry a C++
compiler. The pytorch runtime bases ship a C compiler and no C++ one. Install
it yourself: your Dockerfile is author-owned content and the platform never
injects layers into it.

```dockerfile
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl g++ \
 && rm -rf /var/lib/apt/lists/*
```

`g++`, recommends-off, and **not** `build-essential` — the latter drags ~250 MB
of make/dpkg-dev the wrapper compile never invokes. Measured cost of this layer
on a ~9.2 GB endpoint image: **+80 MB**.

**The platform verifies it rather than establishing it.** `python -m
gen_worker.discovery` already runs inside your final image, so it can ask the
question about the image that will actually serve. An image whose endpoints
declare an AOT export and whose PATH holds no C++ compiler fails the build, by
name:

```
error: aot precondition cxx_toolchain: no C++ compiler on this image, but
       ['micro-4d', 'micro-diffusion'] declare an AOT export
```

That refusal costs $0.00, at the build, naming the families. Before the check
existed the same defect cost 336 s of rented L4 time and surfaced as
`InvalidCxxCompiler` at the link step. A guarantee you can observe beats a
quieter one: the refusal IS the guarantee.

### On a CUDA image, `g++` is necessary and not sufficient

torch's `cpp_extension` also needs a CUDA root it can discover, and the pytorch
runtime bases ship CUDA as pip wheels without ever creating `/usr/local/cuda`.
An image with `g++` and no CUDA root reaches the link step and dies with
`CUDA_HOME environment variable is not set` — three separate facts are missing,
and none of them is a compiler.

One line, after your dependency install:

```dockerfile
RUN python -m gen_worker.cuda_root
```

It composes `/usr/local/cuda` out of parts the image already ships (the
`nvidia/*` wheels' headers and libs, the real `crt/` headers, and `nv/target`
from `cuda-cccl` fetched into a throwaway directory). It writes nothing inside
a pip package, is a no-op when the image already has a CUDA install, and never
fails a build — a CPU image simply has nothing to compose.

You invoke it; the SDK owns whether the recipe is right. That is deliberate:
the alternative is twenty lines of shell transcribed into every Dockerfile that
needs it, drifting apart the moment a base image changes.

The same build-time gate covers it:

```
error: aot precondition cuda_root: torch's cpp_extension cannot host-compile on
       this image and ['micro-diffusion'] declare an AOT export. Missing: …
```

---

## Full real-world example (managed mode + uv)

```dockerfile
ARG BASE_IMAGE=<a real pytorch/pytorch tag+digest matching your profile>
FROM ${BASE_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# The AOT host toolchain. Drop this layer and the cuda_root line below only if
# no endpoint in the image declares a compile export — Tensorhub's own
# generated Dockerfile does both in every image it synthesizes.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl g++ \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock /app/

RUN uv export --no-cache --link-mode copy \
      --no-dev --no-hashes --no-sources --no-emit-project --no-emit-local \
      -o /tmp/requirements.txt \
    && uv pip install --no-cache --link-mode copy \
      --system --break-system-packages --no-deps -r /tmp/requirements.txt

COPY . /app

RUN uv pip install --no-cache --link-mode copy \
      --system --break-system-packages --no-deps --no-sources /app \
    && python -m gen_worker.cuda_root \
    && mkdir -p /app/.tensorhub \
    && python -m gen_worker.discovery > /app/.tensorhub/endpoint.lock

ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]
```

Use this shape for GPU endpoints. For CPU-only endpoints, `python:3.12-slim`
is still a good common default.
