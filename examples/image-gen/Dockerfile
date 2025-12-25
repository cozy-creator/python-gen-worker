ARG BASE_IMAGE=cozy/check-python-worker-cuda
FROM ${BASE_IMAGE}

WORKDIR /app/user_project

COPY pyproject.toml /app/user_project/pyproject.toml
COPY README.md /app/user_project/README.md
COPY src /app/user_project/src

RUN uv pip install --system .

ENV USER_MODULES=image_gen
ENV ENABLE_DEFAULT_MODEL_MANAGER=true
