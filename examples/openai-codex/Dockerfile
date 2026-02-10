# Tenant-supplied Dockerfile example: OpenAI Codex CLI worker (Ubuntu 24.04).
#
# Contract:
# - Installs gen-worker.
# - Runs discovery at build time, baking /app/.cozy/manifest.json into the image.
# - Uses gen-worker as the image ENTRYPOINT.
#
# Local build:
#   docker build -t cozy-example-openai-codex:dev -f Dockerfile .
#
# Runtime requires:
# - WORKER_JWT (for worker auth)
# - CODEX_API_KEY (for Codex `exec` headless auth)
#
FROM ghcr.io/astral-sh/uv:latest AS uv_bin

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12

# Pin Codex CLI version for reproducibility.
# See https://github.com/openai/codex/releases
ARG CODEX_VERSION=0.80.0
ARG TARGETARCH

WORKDIR /app

ENV UV_CACHE_DIR=/var/cache/uv
ENV UV_LINK_MODE=copy

COPY --from=uv_bin /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    python3 \
    python3-venv \
    tar \
  && rm -rf /var/lib/apt/lists/*

# Install Codex CLI binary.
RUN set -eu; \
  arch="x86_64"; \
  if [ "${TARGETARCH:-amd64}" = "arm64" ]; then arch="aarch64"; fi; \
  asset="codex-${arch}-unknown-linux-musl.tar.gz"; \
  url="https://github.com/openai/codex/releases/download/rust-v${CODEX_VERSION}/${asset}"; \
  curl -fsSL "$url" -o /tmp/codex.tar.gz; \
  tar -xzf /tmp/codex.tar.gz -C /tmp; \
  mv "/tmp/codex-${arch}-unknown-linux-musl" /usr/local/bin/codex; \
  chmod +x /usr/local/bin/codex; \
  rm -rf /tmp/codex.tar.gz

# Stable runtime layer.
RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
  uv pip install --system --break-system-packages gen-worker

# Copy lock metadata first (cache-friendly).
COPY pyproject.toml uv.lock /app/

RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
  uv export --no-dev --no-hashes --no-sources --no-emit-project --no-emit-local \
    --no-emit-package gen-worker \
    -o /tmp/requirements.all.txt \
  && grep -Ev '^(gen-worker)' /tmp/requirements.all.txt > /tmp/requirements.txt \
  && uv pip install --system --break-system-packages --no-deps -r /tmp/requirements.txt

# Copy app code late.
COPY . /app

# Install the project itself.
RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
  uv pip install --system --break-system-packages --no-deps --no-sources /app

# Bake discovered endpoints into the image.
RUN mkdir -p /app/.cozy \
  && python3 -m gen_worker.discover > /app/.cozy/manifest.json

# Non-root runtime user.
RUN groupadd --system --gid 10001 cozy \
  && useradd --system --uid 10001 --gid cozy --create-home --home-dir /home/cozy --shell /usr/sbin/nologin cozy \
  && chown -R cozy:cozy /app /home/cozy

ENV HOME=/home/cozy
ENV XDG_CACHE_HOME=/home/cozy/.cache

USER cozy:cozy

ENTRYPOINT ["python3", "-m", "gen_worker.entrypoint"]

