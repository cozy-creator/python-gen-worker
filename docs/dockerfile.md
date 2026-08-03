# Dockerfile contract

You only need a Dockerfile when you want to own the base image, build steps,
dependency manager, caching strategy, or multi-stage layout. For simple
Tensorhub endpoints, omit the Dockerfile and declare build hints in
`endpoint.toml`'s `[[build.profiles]]`; Tensorhub generates the Dockerfile
and satisfies the contract below.

When you do provide a Dockerfile, it is fully yours. Tensorhub does not own this
layer.

You satisfy three contract points; everything else is up to you.

## The three contract points

1. **`gen_worker` is importable in the runtime environment.** Whatever
   dependency manager you use, the resulting image must have `gen-worker`
   (or just `gen-worker` for non-PyTorch endpoints) installed.

2. **Discovery is baked into the image at `/app/.tensorhub/endpoint.lock`.**
   Run discovery during `docker build`:

   ```dockerfile
   RUN mkdir -p /app/.tensorhub \
       && python -m gen_worker.discovery > /app/.tensorhub/endpoint.lock
   ```

   This serializes every `@endpoint` object's `Resources`, bindings, and
   payload schemas. The control plane reads the lock from the built image.

3. **The entrypoint runs `gen_worker.entrypoint`.**

   ```dockerfile
   ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]
   ```

   The entrypoint reads `endpoint.lock`, connects to the orchestrator, and
   serves invocations.

---

## Minimum viable Dockerfile

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install -e .
RUN mkdir -p /app/.tensorhub \
    && python -m gen_worker.discovery > /app/.tensorhub/endpoint.lock
ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]
```

No `ARG BASE_IMAGE`, no version pass-throughs. The endpoint's `pyproject.toml`
pins `gen-worker` and the Dockerfile installs it.

If the Dockerfile only looks like this, prefer Tensorhub's
generated-Dockerfile path:

```toml
[[build.profiles]]
accelerator = "none"
python = "3.12"
```

---

## The base-image digest is duplicated across repos

When your profile declares `python` / `torch` / `cuda` (managed mode) or
`base_image` (explicit mode), tensorhub resolves the base image and passes it
as the `BASE_IMAGE` build arg, plus matching `PYTHON_VERSION` /
`TORCH_VERSION` / `CUDA_VERSION` args. Omit `ARG BASE_IMAGE` entirely for a
**fully custom** profile — tensorhub injects nothing and your `FROM` is the
only source of truth.

**The pinned digest is not single-sourced.** The authority is one map entry
in tensorhub's `internal/builder/baseimage.go`; copies of the same
`pytorch/pytorch:2.13.0-cuda13.0-...@sha256:db80a41f...` string were spread
across four sites in two repos (that map, a builder test fixture, and two
samples in this file — the two samples are now deleted). A torch bump has to
chase every copy; if a local build and a tensorhub build disagree about the
base, this is why.

## Cut from this document

The Docker tutorial that used to live here (`ARG BASE_IMAGE` walkthrough,
version-pinning `RUN` steps, multi-profile branching, cache-bust nonces, the
full uv example) was generic Docker knowledge with a Cozy label on it. Write
whatever Dockerfile you want; satisfy the three contract points above.

