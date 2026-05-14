# Cookbook: `@inference.stage` — Multi-Stage Pipelines

One page on when (and when not) to annotate the internal stages of a
multi-stage endpoint with `@inference.stage`. The annotation is a
forward-compatibility hook: today the SDK calls stages in-process as
method calls on `self`; tomorrow the SDK can route a stage to a
remote worker on a different GPU class.

Cross-links:
- [endpoint-authoring.md](endpoint-authoring.md) — base SDK reference.
- [cookbook-image-diffusion.md](cookbook-image-diffusion.md) — single-stage
  image DiTs (don't need stage annotations).
- [cookbook-video-diffusion.md](cookbook-video-diffusion.md) — video DiTs.
- [cookbook-audio.md](cookbook-audio.md) — audio (SerialWorker and
  BatchedWorker).

---

## TL;DR

```python
@inference(models={...}, resources=...)
class MyPipeline:
    def setup(self, ...): ...

    @inference.stage(name="encode", gpu_class="small")
    def encode(self, image) -> Embeds: ...

    @inference.stage(name="structure", gpu_class="large")
    def make_structure(self, embeds) -> Structure: ...

    @inference.function
    def generate(self, ctx, payload):
        embeds = self.encode(payload.image)        # plain method call
        structure = self.make_structure(embeds)    # plain method call
        return ...

    def shutdown(self): ...
```

Two rules:

1. **Annotate when** the pipeline has stages with materially different
   compute / VRAM profiles AND you want forward-compat with
   disaggregated deployment.
2. **Skip when** the workload is single-stage (one big forward), or
   when the stages are too small to matter for placement decisions.

If in doubt: skip. The `@inference.function` method alone is the right
shape for 90% of endpoints. Add stage annotations when you have
evidence (measured) that the stages would benefit from sharing
hardware across many requests.

---

## When to use `@inference.stage`

Three signals point to "yes, annotate":

### 1. The pipeline has distinct compute classes

TRELLIS.2 image-to-3D is the canonical example:

| Stage         | Compute class | Why                                                |
|---------------|---------------|----------------------------------------------------|
| `encode`      | small         | Vision transformer; batch-friendly; runs comfortably on a small GPU. |
| `structure`   | large         | Sparse-structure DiT; the heaviest component; saturates one large GPU. |
| `texture`     | large         | Texture flow + FlexGEMM sparse conv; second-heaviest. |
| `mesh_extract`| small         | Sparse marching cubes; small GPU sufficient.       |

Today, all four stages run in-process on whichever worker handles the
request. Tomorrow, the SDK can route `encode` and `mesh_extract` to a
small-GPU pool that's shared across many in-flight 3D requests, while
each `structure` + `texture` invocation owns one large GPU. The
annotations carry the metadata the SDK needs to do this without
tenant rewrites.

### 2. The text encoder / VAE could batch across many image requests

Very large image DiTs (Flux.2 at 9B, Qwen-Image-20B) have a similar
shape: the text encoder is small enough to share across many
concurrent image generations, while the DiT itself fills a large GPU.

```python
@inference.stage(name="encode_text", gpu_class="small")
def encode_text(self, prompt: str) -> TextEmbeds: ...

@inference.stage(name="denoise", gpu_class="large")
def denoise(self, embeds: TextEmbeds, ...) -> Latents: ...

@inference.stage(name="decode_vae", gpu_class="small")
def decode_vae(self, latents: Latents) -> Image: ...
```

Today this is sequential method calls. Tomorrow the SDK can run
many `encode_text` calls together on a single small-GPU worker,
ship the embeddings to a large-GPU worker for `denoise`, and route
the VAE decode back to the small-GPU pool. The pattern mirrors
NVIDIA Dynamo's prefill/decode split for LLM serving.

### 3. You want clean isolation of an in-process pipeline today

Even before disaggregation lands, annotating stages buys you:

- **Typed contracts between stages.** Each stage's input and output
  types are visible in code and in the discovery manifest. Refactoring
  one stage doesn't silently break the next.
- **Progress reporting points** at natural stage boundaries (the
  generate() function calls `ctx.progress(0.3, stage="texture")` after
  the structure stage and the orchestrator surfaces the labeled
  stage as the current activity).
- **Warmup discipline.** Warmup loops naturally fan out to per-stage
  dummy forwards — encoder warmup, DiT warmup, VAE warmup — matching
  the stage shape.

---

## When NOT to use `@inference.stage`

**Single-stage workloads.** A bog-standard SDXL or Flux generation
has one big forward. There's no internal disaggregation boundary;
annotating "denoise" as a stage gains nothing. Just put the work in
`@inference.function`.

```python
# NO — stage annotation buys nothing here.
@inference.function
def generate(self, ctx, payload):
    return self._do_one_forward(payload)

@inference.stage(name="denoise", gpu_class="large")    # unnecessary
def _do_one_forward(self, payload): ...
```

**Stages too small to matter.** If the entire pipeline runs in
<500ms, stage routing overhead (serialization, transport, scheduling)
will eat any gains from disaggregation. Annotate stages whose
in-process cost is large enough that routing them remotely is a
sensible trade.

**Stages with shared mutable state.** `@inference.stage` is best for
pure-function-like stages: input data in, output data out, no
implicit state shared with the next stage beyond what's typed in the
return value. If the stages share a `self.kv_cache` that survives
across calls, disaggregation can't preserve it transparently. Stick
with `@inference.function` and a clear in-class implementation.

**You don't have evidence the disaggregation will pay off.** The
annotation is forward-compat metadata — it costs nothing today, but
also gains nothing today. Adding it before you have measured per-stage
times and identified a target small-GPU pool is yak-shaving.

---

## How stages are called — today vs tomorrow

### Today: in-process method calls on `self`

The SDK does no special routing. Calling `self.encode(image)` is
exactly that — a regular Python method call. The `@inference.stage`
decoration attaches metadata (`name`, `gpu_class`) that discovery
emits to the manifest, but at runtime the SDK is transparent:

```python
@inference.function
def generate(self, ctx, payload):
    # All three calls are local method calls. No serialization,
    # no scheduling, no IPC. SDK is invisible at runtime.
    embeds = self.encode(payload.image)
    structure = self.make_structure(embeds)
    mesh = self.extract_mesh(structure)
    return mesh
```

This means stage annotations have **zero runtime cost today**. You
can annotate as soon as you know the pipeline shape; the cost is
the annotation lines themselves.

### Tomorrow: routed across GPU pools by `gpu_class`

When the SDK ships disaggregated inference, calling `self.encode(image)`
may transparently become a typed RPC to a remote small-GPU worker.
The contract:

- **Inputs/outputs serialize cleanly.** Use msgspec structs for
  stage handles — same shape as wire payloads. Don't pass raw
  `torch.Tensor` references between stages that the SDK might
  disaggregate; produce typed handles that can serialize.
- **Stages are stateless.** Each call is independent of the previous;
  any state needed by the next stage rides in the return value.
- **No global mutable state.** Don't write to `self._cache` from
  within a `@inference.stage` method; the next call may land on a
  different worker.

If your existing code violates these contracts and you want to keep
the annotation, mark the violating fast-path private and call it
without `@inference.stage`. The annotations are opt-in per method.

### `gpu_class` is the placement signal

| `gpu_class` | Maps to (in future deployments)          | Use for                                          |
|-------------|------------------------------------------|--------------------------------------------------|
| `"small"`   | T4 / L4 / A10 / 4090 / small B40         | Encoders, VAEs, mesh extraction, text tokenization. |
| `"large"`   | A100-80 / H100 / H200 / B200             | DiTs, large transformers, the heavy denoise loop. |

Default is `"large"` — sensible for the steady state where the heavy
component dominates. Set `"small"` explicitly for the lightweight
ancillary stages.

There's no `"medium"` or numeric tier today. The two-class split
captures the practical disaggregation boundary (cheap vs expensive
hardware) without committing the SDK to a finer-grained policy that
might not match real fleet shapes.

---

## Complete working example: TRELLIS.2 with 4 stages annotated

Microsoft's TRELLIS.2 image-to-3D pipeline has exactly the right
shape for staged disaggregation. Image encoder + mesh extraction are
small-GPU candidates; structure DiT + texture flow saturate a large
GPU each.

```python
"""TRELLIS.2 image-to-3D — SerialWorker with 4 stages annotated."""

from __future__ import annotations

from typing import Any

import msgspec

from gen_worker import (
    Asset,
    Repo,
    RequestContext,
    Resources,
    inference,
)


# Three model bindings — encoder + structure DiT + texture flow.
trellis_structure = Repo("microsoft/TRELLIS.2-image-large-structure")
trellis_texture = Repo("microsoft/TRELLIS.2-image-large-texture")
trellis_encoder = Repo("microsoft/TRELLIS.2-image-large-encoder")


# ---------------------------------------------------------------------------
# Wire shapes.
# ---------------------------------------------------------------------------


class ThreeDInput(msgspec.Struct):
    image: Asset
    resolution: int = 1024              # 512 / 1024 / 1536 on H100
    seed: int = 0
    num_structure_steps: int = 25
    num_texture_steps: int = 25


class ThreeDOutput(msgspec.Struct):
    mesh: Asset
    num_vertices: int
    num_faces: int
    resolution: int


# ---------------------------------------------------------------------------
# Internal stage handles — typed msgspec structs that travel between
# stages. NOT exposed on the wire; only ThreeDInput / ThreeDOutput
# cross the worker boundary. Each handle is small and serializable,
# so the future disaggregated path can ship it between workers.
# ---------------------------------------------------------------------------


class Embeds(msgspec.Struct):
    tokens: int
    width: int
    height: int


class Structure(msgspec.Struct):
    resolution: int
    num_voxels: int


class Textured(msgspec.Struct):
    resolution: int
    num_voxels: int
    num_texture_features: int


class Mesh(msgspec.Struct):
    num_vertices: int
    num_faces: int
    resolution: int

    def serialize(self) -> bytes:
        """Serialize to glb bytes (placeholder; real impl uses trimesh)."""
        header = b"glTF\x02\x00\x00\x00"
        return header + f"trellis2:v={self.num_vertices},f={self.num_faces}".encode()


# ---------------------------------------------------------------------------
# Endpoint class.
# ---------------------------------------------------------------------------


@inference(
    label="TRELLIS.2 image-to-3D",
    description=(
        "Microsoft TRELLIS.2 image-to-3D pipeline. SerialWorker with "
        "four stages annotated for future disaggregation: encode "
        "(small GPU), structure DiT (large GPU), texture flow "
        "(large GPU), mesh extract (small GPU). Today every stage "
        "runs in-process on the request's owning worker."
    ),
    resources=Resources(
        accelerator="cuda",
        requires_gpu=True,
        min_vram_gb=24.0,
        min_compute_capability=8.0,
        peak_vram_per_request_gb=24.0,
        vram_must_fit="full_model",
        runtime_scales_with=(
            "resolution", "num_structure_steps", "num_texture_steps",
        ),
    ),
    models={
        "structure": trellis_structure.flavor("bf16"),
        "texture": trellis_texture.flavor("bf16"),
        "encoder": trellis_encoder.flavor("bf16"),
    },
)
class Trellis2Generate:
    def setup(self, structure: Any, texture: Any, encoder: Any) -> None:
        """Load weights and apply per-component acceleration."""
        self._structure = structure
        self._texture = texture
        self._encoder = encoder
        # In production code, each component would be wrapped here:
        #   import torch
        #   self._structure = torch.compile(structure, mode='reduce-overhead')
        #   gen_worker.cache.FBCache(threshold=0.12).apply(self._structure)

    def warmup(self) -> None:
        """Dummy forward at each declared shape, per stage.

        With stage annotations, warmup naturally splits across stages
        so each component's compile cache is populated before traffic.
        """
        embeds = self.encode("warmup://image")
        structure = self.make_structure(embeds, resolution=1024, steps=1)
        textured = self.make_texture(structure, steps=1)
        _ = self.extract_mesh(textured)

    # ------------------------------------------------------------------
    # The four stages. @inference.stage attaches placement metadata so
    # future SDK releases can route each to a separate worker.
    # ------------------------------------------------------------------

    @inference.stage(name="encode", gpu_class="small")
    def encode(self, image_ref: str) -> Embeds:
        """Stage 1: image encoder → conditioning embeddings.

        Small-GPU class: the encoder is a vision transformer that
        runs comfortably on small hardware AND is batch-friendly.
        Future disaggregation lets many in-flight 3D requests share
        a single small-GPU encoder pool.
        """
        # Real impl: load image, tokenize, run encoder.
        return Embeds(tokens=1369, width=518, height=518)

    @inference.stage(name="structure", gpu_class="large")
    def make_structure(self, embeds: Embeds, resolution: int, steps: int) -> Structure:
        """Stage 2: sparse-structure DiT → coarse voxel occupancy.

        Large-GPU class: the structure DiT is the heaviest component
        and saturates one large GPU end-to-end. SerialWorker keeps a
        single request on the GPU for the duration of this stage.
        """
        active_voxels = max(1, (resolution // 16) ** 3 // 64)
        return Structure(resolution=resolution, num_voxels=active_voxels)

    @inference.stage(name="texture", gpu_class="large")
    def make_texture(self, structure: Structure, steps: int) -> Textured:
        """Stage 3: texture flow → SLAT features on the voxel set.

        Large-GPU class: FlexGEMM sparse conv; second-heaviest.
        """
        return Textured(
            resolution=structure.resolution,
            num_voxels=structure.num_voxels,
            num_texture_features=8,
        )

    @inference.stage(name="mesh_extract", gpu_class="small")
    def extract_mesh(self, textured: Textured) -> Mesh:
        """Stage 4: mesh extraction → triangle mesh.

        Small-GPU class: dominated by sparse marching cubes; runs
        comfortably on small hardware. Candidate for sharing across
        in-flight requests in a disaggregated deployment.
        """
        verts = textured.num_voxels * 4
        faces = textured.num_voxels * 2
        return Mesh(
            num_vertices=verts,
            num_faces=faces,
            resolution=textured.resolution,
        )

    # ------------------------------------------------------------------
    # The externally invocable function. Calls the four stages in
    # sequence. With staging annotations, the SDK is free to route
    # each call to a different worker in future releases.
    # ------------------------------------------------------------------

    @inference.function(
        timeout_ms=120_000,
        description="Run TRELLIS.2 image-to-3D end-to-end.",
    )
    def generate(self, ctx: RequestContext, payload: ThreeDInput) -> ThreeDOutput:
        ctx.raise_if_canceled()

        resolution = int(payload.resolution)
        if resolution not in (512, 1024, 1536):
            raise ValueError(
                f"resolution must be 512, 1024, or 1536; got {resolution}"
            )
        num_structure_steps = max(1, min(100, int(payload.num_structure_steps)))
        num_texture_steps = max(1, min(100, int(payload.num_texture_steps)))

        image_ref = (payload.image.ref or payload.image.local_path or "").strip()
        if image_ref == "":
            raise ValueError("payload.image must have a ref or local_path")

        # Stage 1: encode (small GPU class).
        ctx.progress(0.0, stage="encode")
        embeds = self.encode(image_ref)
        ctx.raise_if_canceled()

        # Stage 2: structure DiT (large GPU class).
        ctx.progress(0.15, stage="structure")
        structure = self.make_structure(
            embeds, resolution=resolution, steps=num_structure_steps,
        )
        ctx.raise_if_canceled()

        # Stage 3: texture flow (large GPU class).
        ctx.progress(0.55, stage="texture")
        textured = self.make_texture(structure, steps=num_texture_steps)
        ctx.raise_if_canceled()

        # Stage 4: mesh extraction (small GPU class).
        ctx.progress(0.85, stage="mesh_extract")
        mesh = self.extract_mesh(textured)
        ctx.raise_if_canceled()

        # Serialize and save.
        ctx.progress(0.95, stage="serialize")
        mesh_asset = ctx.save_bytes(
            f"jobs/{ctx.request_id}/outputs/mesh.glb",
            mesh.serialize(),
        )

        ctx.progress(1.0, stage="done")
        return ThreeDOutput(
            mesh=mesh_asset,
            num_vertices=mesh.num_vertices,
            num_faces=mesh.num_faces,
            resolution=resolution,
        )

    def shutdown(self) -> None:
        self._structure = None
        self._texture = None
        self._encoder = None
```

What the annotations buy you today:

- The discovery manifest lists all four stages with their `gpu_class`,
  so the orchestrator can surface "this endpoint has small-GPU-friendly
  stages" in metrics and capacity planning.
- `ctx.progress(0.55, stage="texture")` calls reference stage names that
  match the annotation labels — clients see consistent stage labels in
  progress events.
- Future SDK releases can route stages without changing the endpoint
  code. The tenant doesn't migrate; the SDK does.

What the annotations cost you today:

- Four extra lines of decoration.
- A typed contract for each stage's input and output. (Arguably a
  feature, not a cost — your refactoring discipline is now enforced.)

---

## A counter-example: when to skip

Standard Flux.2-klein-4B at 1024² is one big forward followed by VAE
decode. The two "stages" (denoise + VAE) are NOT good candidates for
disaggregation: the VAE is small enough that the IPC round-trip would
cost more than running it locally, and shipping the latent across
workers is a sizable serialization tax.

```python
# DON'T annotate stages here — the workload is single-stage.
@inference(models={"pipe": flux_klein}, ...)
class FluxGenerate:
    def setup(self, pipe): ...

    @inference.function
    def generate(self, ctx, payload):
        # No stage annotations. This is a single bag of work.
        return self.pipe(payload.prompt, ...).images[0]

    def shutdown(self): ...
```

If you later identify a real disaggregation boundary (say, sharing
the T5-XXL text encoder across many concurrent Flux requests),
that's the moment to add `@inference.stage(name="encode_text",
gpu_class="small")`. Not before.

---

## Next steps

- **3D endpoints** (TRELLIS.2, Hunyuan3D-2.5)? Annotate the four
  natural stages (encode / structure / texture / mesh_extract) with
  `@inference.stage`. The shape above is a working template.
- **Very large image / video DiTs** with an obvious text encoder /
  VAE split that justifies disaggregation? Annotate the encoder
  and VAE stages as `gpu_class="small"`. Keep the DiT itself as
  `gpu_class="large"` (or unannotated if it's the only invocation
  point — it stays inside `@inference.function`).
- **Want to measure the disaggregation gain before annotating?**
  Run a per-stage timing pass first. Log per-stage seconds via
  `ctx.emit("stage_timing", {...})` and look at the histogram:
  if one stage takes >70% of total time and the others are
  small-GPU-friendly, you have a candidate for staging.

The `@inference.stage` annotation pattern is a long-game investment:
zero cost today, large lever tomorrow. Add it when the pipeline
shape clearly fits; skip it when it doesn't.
