# Cozy Worker Dockerfile
#
# Build your own worker image without gen-builder.
#
# Usage:
#   docker build -t my-worker .
#
# Available base images:
#   - cozycreator/gen-worker:cpu-torch2.9        (CPU only)
#   - cozycreator/gen-worker:cuda12.6-torch2.9   (CUDA 12.6)
#   - cozycreator/gen-worker:cuda12.8-torch2.9   (CUDA 12.8)
#   - cozycreator/gen-worker:cuda13-torch2.9     (CUDA 13.0)
#
# This Dockerfile matches cozy-hub's GPU worker build path.
FROM cozycreator/gen-worker:cuda12.8-torch2.9
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY . /app

# Install system dependencies for GPU worker parity with cozy-hub templates.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies.
RUN uv sync --no-sources --no-dev

# Generate function manifest at build time
RUN mkdir -p /app/.cozy && /app/.venv/bin/python -m gen_worker.discover > /app/.cozy/manifest.json

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["/app/.venv/bin/python", "-m", "gen_worker.entrypoint"]
