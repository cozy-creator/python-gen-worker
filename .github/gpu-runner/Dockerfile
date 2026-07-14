# Ephemeral GitHub Actions runner image for the GPU CI lane (gw#421).
# Runs on a RunPod GPU pod; torch cu128 comes from `uv sync` at job time.
# `base` flavor: torch wheels bundle their own CUDA runtime; the host injects the driver.
FROM nvidia/cuda:12.8.1-base-ubuntu22.04

ARG RUNNER_VERSION=2.335.1

ENV DEBIAN_FRONTEND=noninteractive \
    RUNNER_ALLOW_RUNASROOT=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git jq tar \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN mkdir /runner && cd /runner \
    && curl -fsSL -o runner.tar.gz \
        "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
    && tar xzf runner.tar.gz && rm runner.tar.gz \
    && ./bin/installdependencies.sh \
    && rm -rf /var/lib/apt/lists/*

# Bake the interpreter so job-time `uv sync` only resolves wheels.
RUN uv python install 3.11

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
