# Tenant-supplied Dockerfile example (GPU).
#
# - Installs torch + gen-worker in stable cacheable layers.
# - Installs tenant deps from uv.lock without replacing torch/gen-worker.
# - Bakes discovered functions into /app/.cozy/manifest.json at build time.
#
ARG PYTHON_VERSION=3.12
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS cozy_base

ARG UV_TORCH_BACKEND=cu126
ARG TORCH_SPEC="~=2.10.0"

WORKDIR /app

ENV UV_CACHE_DIR=/var/cache/uv
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,id=cozy-apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=cozy-apt-lists,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean

RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv pip install --system --break-system-packages --torch-backend ${UV_TORCH_BACKEND} \
      "torch${TORCH_SPEC}"
RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv pip install --system --break-system-packages gen-worker==0.3.0

FROM cozy_base AS cozy_final

COPY pyproject.toml uv.lock /app/

RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv export --no-dev --no-hashes --no-sources --no-emit-project --no-emit-local \
      --no-emit-package torch --no-emit-package gen-worker \
      -o /tmp/requirements.all.txt \
    && grep -Ev '^(torch|triton|nvidia-|cuda-)' /tmp/requirements.all.txt > /tmp/requirements.txt \
    && uv pip install --system --break-system-packages --no-deps -r /tmp/requirements.txt

COPY . /app

RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv pip install --system --break-system-packages --no-deps --no-sources /app

RUN mkdir -p /app/.cozy \
    && python -m gen_worker.discover > /app/.cozy/manifest.json

ENV HOME=/root
ENV XDG_CACHE_HOME=/root/.cache
ENV HF_HOME=/root/.cache/huggingface

ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]
