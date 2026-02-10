# Tenant-supplied Dockerfile example (CPU).
#
# - Installs gen-worker in a stable cacheable layer.
# - Installs torch (CPU) so Cozy Hub can record a stable backend version for tagging.
# - Installs tenant deps from uv.lock without reinstalling gen-worker.
# - Bakes discovered endpoints into /app/.cozy/manifest.json at build time.
#
ARG PYTHON_VERSION=3.12
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim

WORKDIR /app

ENV UV_CACHE_DIR=/var/cache/uv
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,id=cozy-apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=cozy-apt-lists,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean

ARG UV_TORCH_BACKEND=cpu
ARG TORCH_SPEC="~=2.10.0"

RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv pip install --system --break-system-packages --torch-backend ${UV_TORCH_BACKEND} \
      "torch${TORCH_SPEC}"

RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv pip install --system --break-system-packages gen-worker

COPY pyproject.toml uv.lock /app/

RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv export --no-dev --no-hashes --no-sources --no-emit-project --no-emit-local \
      --no-emit-package gen-worker --no-emit-package torch \
      -o /tmp/requirements.txt \
    && uv pip install --system --break-system-packages --no-deps -r /tmp/requirements.txt

COPY . /app

RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv pip install --system --break-system-packages --no-deps --no-sources /app

RUN mkdir -p /app/.cozy \
    && python -m gen_worker.discover > /app/.cozy/manifest.json

RUN groupadd --system --gid 10001 cozy \
    && useradd --system --uid 10001 --gid cozy --create-home --home-dir /home/cozy --shell /usr/sbin/nologin cozy \
    && chown -R cozy:cozy /app /home/cozy

ENV HOME=/home/cozy
ENV XDG_CACHE_HOME=/home/cozy/.cache

USER cozy:cozy

ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]
