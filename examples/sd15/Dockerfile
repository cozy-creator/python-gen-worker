# Tenant-supplied Dockerfile example.
#
# - Installs torch + gen-worker in stable cacheable layers.
# - Installs tenant deps from uv.lock without replacing torch/gen-worker.
# - Bakes discovered functions into /app/.cozy/manifest.json at build time.
#
# Local build (GPU):
#   docker build -t sd15-worker -f Dockerfile .
#
ARG PYTHON_VERSION=3.12
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS cozy_base

ARG UV_TORCH_BACKEND=cu126
ARG TORCH_SPEC="~=2.10.0"

WORKDIR /app

ENV UV_CACHE_DIR=/var/cache/uv
ENV UV_LINK_MODE=copy

# Keep base stages identical across GPU examples to maximize local/remote cache hits.
RUN --mount=type=cache,id=cozy-apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=cozy-apt-lists,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean

# Stable runtime layers (avoid re-downloading torch/cu libs on every tenant build).
RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv pip install --system --break-system-packages --torch-backend ${UV_TORCH_BACKEND} \
      "torch${TORCH_SPEC}"
RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv pip install --system --break-system-packages gen-worker

FROM cozy_base AS cozy_final

# Copy lock metadata first so dependency layers are cacheable across source edits.
COPY pyproject.toml uv.lock /app/

# Install tenant dependencies into the global environment without replacing torch/gen-worker.
RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv export --no-dev --no-hashes --no-sources --no-emit-project --no-emit-local \
      --no-emit-package torch --no-emit-package gen-worker \
      -o /tmp/requirements.all.txt \
    && grep -Ev '^(torch|triton|nvidia-|cuda-)' /tmp/requirements.all.txt > /tmp/requirements.txt \
    && uv pip install --system --break-system-packages --no-deps -r /tmp/requirements.txt

# Copy app code late so app edits only invalidate the final layers.
COPY . /app

# Install the project itself without altering dependency layers.
RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv pip install --system --break-system-packages --no-deps --no-sources /app

# Bake discovered functions into the image so Cozy Hub can read them without executing tenant code.
RUN mkdir -p /app/.cozy \
    && python -m gen_worker.discover > /app/.cozy/manifest.json

# Run as root at runtime (needed for writable mounted cache dirs like /workspace).

ENV HOME=/root
ENV XDG_CACHE_HOME=/root/.cache
ENV HF_HOME=/root/.cache/huggingface


ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]
