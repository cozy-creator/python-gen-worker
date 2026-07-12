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
   (or just `gen-worker` for non-PyTorch endpoints) installed.

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
ARG BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
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

## Caching: tenant-controlled

`BUILD_NONCE`, `DEPS_NONCE`, endpoint-specific commit pins — these are your
cache-bust knobs, unrelated to versioning. Tensorhub does not impose any
caching strategy.

```dockerfile
ARG BUILD_NONCE=2026-04-23-default
RUN echo "build-nonce=${BUILD_NONCE}" \
    && mkdir -p /app/.tensorhub \
    && python -m gen_worker.discovery > /app/.tensorhub/endpoint.lock
```

---

## Full real-world example (managed mode + uv)

```dockerfile
ARG BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_CACHE_DIR=/var/cache/uv \
    UV_LINK_MODE=copy \
    HOME=/root \
    XDG_CACHE_HOME=/root/.cache

COPY pyproject.toml uv.lock /app/

RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv export --no-dev --no-hashes --no-sources --no-emit-project --no-emit-local \
      -o /tmp/requirements.txt \
    && uv pip install --system --break-system-packages --no-deps -r /tmp/requirements.txt

COPY . /app

RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv pip install --system --break-system-packages --no-deps --no-sources /app \
    && mkdir -p /app/.tensorhub \
    && python -m gen_worker.discovery > /app/.tensorhub/endpoint.lock

ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]
```

Use this shape for GPU endpoints. For CPU-only endpoints, `python:3.12-slim`
is still a good common default.
