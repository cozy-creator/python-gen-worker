ARG BASE_IMAGE=cozycreator/python-worker:cuda12.8-torch2.9
FROM ${BASE_IMAGE}

WORKDIR /app
COPY . /app

WORKDIR /app/examples/sd15

RUN pip install --no-cache-dir uv
RUN uv sync --no-dev

RUN mkdir -p /app/.cozy && .venv/bin/python -m gen_worker.discover > /app/.cozy/manifest.json

ENTRYPOINT ["/app/examples/sd15/.venv/bin/python", "-m", "gen_worker.entrypoint"]
