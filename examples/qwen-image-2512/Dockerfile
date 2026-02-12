# Tenant-supplied Dockerfile example.
#
# - Installs torch in a stable cacheable layer.
# - Installs tenant deps from uv.lock without replacing torch/gen-worker.
# - Installs exactly one gen-worker source from Cozy Hub vendored build context.
# - Bakes discovered endpoints into /app/.cozy/manifest.json at build time.
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

RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv pip install --system --break-system-packages --torch-backend ${UV_TORCH_BACKEND} \
      "torch${TORCH_SPEC}"

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

RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    test -f /app/.cozy/vendor/gen-worker/pyproject.toml \
    && uv pip install --system --break-system-packages /app/.cozy/vendor/gen-worker

RUN mkdir -p /app/.cozy \
    && python - <<'PY' > /app/.cozy/manifest.json
import runpy
runpy.run_module("gen_worker.discover", run_name="__main__")
PY


ENV HOME=/root
ENV XDG_CACHE_HOME=/root/.cache
ENV HF_HOME=/root/.cache/huggingface


ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]
