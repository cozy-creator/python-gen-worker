# The three Dockerfile contract points (docs/dockerfile.md), plus one thing
# only this family does: it GENERATES its checkpoint at build time.
#
# The generation step is what keeps the family compliant with the
# weights-locality rule while still being a real endpoint. There is no
# checkpoint in git, none on a developer's box, and no download at boot — the
# bytes are a pure function of a seed, materialized into the image once, and
# byte-identical in every image built from this Dockerfile.
ARG BASE_IMAGE=pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime
FROM ${BASE_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock /app/

RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv export --cache-dir /var/cache/uv --link-mode copy \
      --no-dev --no-hashes --no-sources --no-emit-project --no-emit-local \
      -o /tmp/requirements.txt \
    && uv pip install --cache-dir /var/cache/uv --link-mode copy \
      --system --break-system-packages --no-deps -r /tmp/requirements.txt

COPY . /app

RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv pip install --cache-dir /var/cache/uv --link-mode copy \
      --system --break-system-packages --no-deps --no-sources /app

# The checkpoint, generated and VERIFIED in the same layer. `--verify`
# regenerates every tensor and byte-compares: an image whose weights did not
# reproduce from the seed fails the build instead of shipping a checkpoint
# nobody else can rebuild.
ARG MICRO_SEED=997
RUN python -m micro_diffusion.weights --out /app/.micro-weights \
      --seed ${MICRO_SEED} --verify

# Contract point 2: discovery baked into the image.
ARG BUILD_NONCE=2026-08-06-default
RUN echo "build-nonce=${BUILD_NONCE}" \
    && mkdir -p /app/.tensorhub \
    && python -m gen_worker.discovery > /app/.tensorhub/endpoint.lock

# Contract point 3.
ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]
