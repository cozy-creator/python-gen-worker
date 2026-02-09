# syntax=docker/dockerfile:1.7
# Cozy Worker Dockerfile
#
# Build your own worker image without gen-builder.
#
# Usage:
#   docker build -t my-worker .
#
# Build args:
#   --build-arg PYTHON_VERSION=3.12
#   --build-arg UV_TORCH_BACKEND=cu126   (cpu, cu126, cu128, cu130, ...)
#   --build-arg TORCH_SPEC="~=2.10.0"
#
# This Dockerfile matches cozy-hub's direct build path (no prebuilt cozycreator/gen-worker base).
ARG PYTHON_VERSION=3.12
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS cozy_base

ARG UV_TORCH_BACKEND=cu126
# Pin torch to 2.10.x by default. We rely on stable torch layers for caching and
# do not want automatic upgrades beyond 2.10.x without an explicit spec.
ARG TORCH_SPEC="~=2.10.0"
# Pin gen-worker to a known version for reproducible builds and stable caching.
# Override explicitly to test unreleased versions.
ARG GEN_WORKER_SPEC="==0.2.1"
# Set to 0 to skip installing torch entirely (for non-torch workers).
ARG USE_TORCH=1

WORKDIR /app

ENV UV_CACHE_DIR=/var/cache/uv
ENV UV_LINK_MODE=copy

# Install system dependencies for GPU worker parity with cozy-hub templates.
RUN --mount=type=cache,id=cozy-apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=cozy-apt-lists,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean

# Install shared runtime layers first for cache reuse.
RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    if [ "${USE_TORCH}" = "1" ]; then \
      uv pip install --system --break-system-packages --torch-backend ${UV_TORCH_BACKEND} \
        "torch${TORCH_SPEC}"; \
    fi
RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv pip install --system --break-system-packages "gen-worker${GEN_WORKER_SPEC}"
RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    if [ "${USE_TORCH}" = "1" ]; then \
      # Keep these as explicit shared-layer deps. They are excluded from project installs
      # when the tenant depends on gen-worker[torch], because we install gen-worker
      # separately to prevent torch replacement.
      uv pip install --system --break-system-packages \
        "safetensors>=0.7.0" \
        "flashpack>=0.2.1" \
        "numpy>=2.4.2"; \
    fi

# Final stage: app-specific deps + code.
FROM cozy_base

# Copy lock metadata first so dependency layers are cacheable across releases.
COPY pyproject.toml uv.lock /app/

# Install dependencies into the global Python environment (no project venv).
# Important:
# - --no-sources ignores tool.uv.sources so build contexts don’t need local path deps.
# - We exclude torch + gen-worker because those are installed above and must never be replaced.
RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv export --locked --no-dev --no-hashes --no-sources --no-emit-project \
      --no-emit-package torch --no-emit-package gen-worker \
      -o /tmp/requirements.all.txt \
    # If the project lock includes torch CUDA deps (often via accelerate), never install them here.
    # Torch (and its CUDA runtime deps) are installed in a stable layer above via --torch-backend.
    && grep -Ev '^(torch|triton|nvidia-|cuda-)' /tmp/requirements.all.txt > /tmp/requirements.txt \
    && uv pip install --system --break-system-packages --no-deps -r /tmp/requirements.txt

# Copy app code late so app edits only invalidate the final layers.
COPY . /app

# Install the project itself without altering dependency layers.
RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv pip install --system --break-system-packages --no-deps --no-sources /app

# Generate function manifest at build time.
# Backward-compat: published gen-worker currently exposes ModelRefSource.DEPLOYMENT.
RUN mkdir -p /app/.cozy && python -c 'from gen_worker.injection import ModelRefSource as M; import runpy; hasattr(M, "RELEASE") or setattr(M, "RELEASE", M.DEPLOYMENT); runpy.run_module("gen_worker.discover", run_name="__main__")' > /app/.cozy/manifest.json

# Run as non-root at runtime.
RUN groupadd --system --gid 10001 cozy \
    && useradd --system --uid 10001 --gid cozy --create-home --home-dir /home/cozy --shell /usr/sbin/nologin cozy \
    && chown -R cozy:cozy /app /home/cozy

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV HOME=/home/cozy
ENV XDG_CACHE_HOME=/home/cozy/.cache
ENV HF_HOME=/home/cozy/.cache/huggingface

USER cozy:cozy

CMD ["python", "-m", "gen_worker.entrypoint"]
