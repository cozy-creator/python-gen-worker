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
#   --build-arg TORCH_SPEC=">=2.9,<3"
#
# This Dockerfile matches cozy-hub's direct build path (no prebuilt cozycreator/gen-worker base).
ARG PYTHON_VERSION=3.12
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim

ARG UV_TORCH_BACKEND=cu126
ARG TORCH_SPEC=">=2.9,<3"

WORKDIR /app

# Install system dependencies for GPU worker parity with cozy-hub templates.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install shared runtime layers first for cache reuse.
RUN uv pip install --system --break-system-packages --no-cache --torch-backend ${UV_TORCH_BACKEND} \
    "torch${TORCH_SPEC}" \
    "torchvision${TORCH_SPEC}" \
    "torchaudio${TORCH_SPEC}"
RUN uv pip install --system --break-system-packages --no-cache gen-worker

# Copy app code late so app edits only invalidate lower layers.
COPY . /app

# Install dependencies into the global Python environment (no project venv).
RUN uv pip install --system --break-system-packages --no-sources /app

# Generate function manifest at build time
RUN mkdir -p /app/.cozy && python -m gen_worker.discover > /app/.cozy/manifest.json

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
