# Tenant-supplied Dockerfile example.
#
# - Installs torch in a stable cacheable layer.
# - Installs tenant deps from uv.lock without replacing torch.
# - Bakes discovered functions into /app/.tensorhub/endpoint.lock at build time.
#
# Local build (GPU):
#   docker build -t flux2-klein-9b-worker -f Dockerfile .
#
ARG PYTHON_VERSION=3.12
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS cozy_base

ARG CUDA_VERSION=12.8
ARG UV_TORCH_BACKEND=
ARG TORCH_SPEC="~=2.10.0"

WORKDIR /app

ENV UV_CACHE_DIR=/var/cache/uv
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,id=cozy-apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=cozy-apt-lists,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean

# Stable runtime layers (avoid re-downloading torch/cu libs on every tenant build).
RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    backend="${UV_TORCH_BACKEND:-cu$(printf '%s' "${CUDA_VERSION}" | tr -d '.')}" \
    && uv pip install --system --break-system-packages --torch-backend "$backend" \
      "torch${TORCH_SPEC}"

FROM cozy_base AS cozy_final

# Copy lock metadata first so dependency layers are cacheable across source edits.
COPY pyproject.toml uv.lock /app/

# Install tenant dependencies into the global environment without replacing torch.
RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv export --no-dev --no-hashes --no-sources --no-emit-project --no-emit-local \
      --no-emit-package torch \
      -o /tmp/requirements.all.txt \
    && grep -Ev '^(torch|triton|nvidia-|cuda-)' /tmp/requirements.all.txt > /tmp/requirements.txt \
    && uv pip install --system --break-system-packages --no-deps -r /tmp/requirements.txt

# Copy app code late so app edits only invalidate the final layers.
COPY . /app

# Install the project itself without altering dependency layers.
RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv pip install --system --break-system-packages --no-deps --no-sources /app

# Bake discovered functions into the image so Cozy Hub can read them without executing tenant code.
RUN mkdir -p /app/.tensorhub \
    && python -m gen_worker.discover > /app/.tensorhub/endpoint.lock

# Run as root at runtime (needed for writable mounted cache dirs like /workspace).

ENV HOME=/root
ENV XDG_CACHE_HOME=/root/.cache
ENV HF_HOME=/root/.cache/huggingface


ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]
