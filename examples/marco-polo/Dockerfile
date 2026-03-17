ARG PYTHON_VERSION=3.12
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim

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
    && python -m gen_worker.discover > /app/.tensorhub/endpoint.lock

ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]
