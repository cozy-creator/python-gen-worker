# Tenant-supplied Dockerfile example.
#
# - Uses Tensorhub-resolved PyTorch base images for shared Torch/CUDA layers.
# - Installs tenant deps from uv.lock without replacing base-provided Torch.
# - Installs required system libs (libsndfile1) for soundfile.
# - Bakes discovered functions into /app/.tensorhub/endpoint.lock at build time.
#
ARG BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE} AS cozy_base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_CACHE_DIR=/var/cache/uv
ENV UV_LINK_MODE=copy

# Keep base stages identical across GPU examples to maximize local/remote cache hits.
RUN --mount=type=cache,id=cozy-apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=cozy-apt-lists,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean

FROM cozy_base AS cozy_final

# Runtime system deps for this tenant project.
RUN --mount=type=cache,id=cozy-apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=cozy-apt-lists,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && apt-get clean

COPY pyproject.toml uv.lock /app/

RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv export --no-dev --no-hashes --no-sources --no-emit-project --no-emit-local \
      --no-emit-package torch \
      --no-emit-package torchvision \
      --no-emit-package torchaudio \
      -o /tmp/requirements.all.txt \
    && grep -Ev '^(torch==|torchvision==|torchaudio==|triton==|nvidia-|cuda-)' /tmp/requirements.all.txt > /tmp/requirements.txt \
    && uv pip install --system --break-system-packages --no-deps -r /tmp/requirements.txt

COPY . /app

RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv pip install --system --break-system-packages --no-deps --no-sources /app

RUN mkdir -p /app/.tensorhub \
    && python3 -m gen_worker.discovery > /app/.tensorhub/endpoint.lock


ENV HOME=/root
ENV XDG_CACHE_HOME=/root/.cache
ENV HF_HOME=/root/.cache/huggingface


ENTRYPOINT ["python3", "-m", "gen_worker.entrypoint"]
