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

# The AOT host toolchain. This family declares a compile export, so its image
# must be able to link the C++ wrapper AOTInductor emits — the pytorch runtime
# bases carry a C compiler and no C++ one. Recommends-off and no
# build-essential: +80 MB measured, against ~250 MB of make/dpkg-dev the
# wrapper compile never invokes. `gen_worker.discovery` below verifies this at
# build time and refuses the image if it is missing (pgw#996, pgw#1017).
#
# NOT SUFFICIENT ON ITS OWN: torch's cpp_extension also needs a discoverable
# CUDA root, which this base does not have (CUDA arrives as pip wheels and
# /usr/local/cuda is never created). The `gen_worker.cuda_root` step below
# composes it, and the `cuda_root` precondition refuses the build without it.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl g++ \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock /app/

RUN uv export --no-cache --link-mode copy \
      --no-dev --no-hashes --no-sources --no-emit-project --no-emit-local \
      -o /tmp/requirements.txt \
    && uv pip install --no-cache --link-mode copy \
      --system --break-system-packages --no-deps -r /tmp/requirements.txt

COPY . /app

RUN uv pip install --no-cache --link-mode copy \
      --system --break-system-packages --no-deps --no-sources /app

# pgw#823/pgw#1017: the pytorch runtime bases ship CUDA as pip wheels and never
# create /usr/local/cuda, so AOT export reaches the LINK step and dies. Compose
# the root from parts the image already ships. Dockerfile-less families get this
# from the hub's synthesized Dockerfile (cudaRootLine); a family that writes its
# own Dockerfile must ask for it explicitly. No-op if a root already exists.
RUN python -m gen_worker.cuda_root

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
ENTRYPOINT ["python", "-m", "gen_worker.worker_main"]
