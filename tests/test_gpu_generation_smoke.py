"""Real image-generation smoke for the GPU CI lane (gw#421).

Generates one image with FLUX.2-klein-4B turbo (step-distilled, ungated HF
repo) through the production loading path — ``load_from_pretrained`` +
``place_pipeline`` — and fails if the output trips the image-garbage
tripwire.

Excluded from default runs: requires ``GEN_WORKER_GPU_SMOKE=1`` (it
downloads ~16GB of weights) and skips cleanly without CUDA/diffusers, so
plain ``pytest tests/`` never touches the network.

The quality gate is a python port of the canonical e2e tripwire
(cozy-creator/e2e ``quality/quality.go``, calibrated 2026-07-06): histogram
entropy, uniform-frame, neighbor correlation, multi-scale latent-garbage,
chroma clipping. Thresholds are copied verbatim; e2e stays canonical — the
duplication exists because the e2e repo is private and the pod job carries
no credentials beyond its single-use runner token.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GEN_WORKER_GPU_SMOKE") != "1",
    reason="generation smoke runs only with GEN_WORKER_GPU_SMOKE=1 (~16GB download)",
)

# Turbo (step-distilled) klein — verified ungated (anonymous download OK).
REPO_ID = "black-forest-labs/FLUX.2-klein-4B"
# Diffusers layout only: the repo also ships a redundant 7.75GB single-file
# checkpoint + sample jpgs we must not pull.
ALLOW_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "tokenizer/*",
    "transformer/*",
    "vae/*",
]
PROMPT = (
    "a red fox standing in a snowy birch forest at golden hour, "
    "photorealistic, shallow depth of field"
)
SEED = 42
STEPS = 4  # klein turbo's distilled step count (endpoint default)


# ---------------------------------------------------------------------------
# Tripwire port (e2e quality/quality.go — thresholds verbatim)

TILE_GRID = 8
QUIET_TILE_THRESHOLD = 8.0
BUSY_TILE_THRESHOLD = 24.0
MIN_ENTROPY = 3.0
MIN_LUMA_STDDEV = 8.0
MIN_NEIGHBOR_CORR = 0.80
MAX_COARSE_BUSY_WITH_NO_QUIET = 0.95
MAX_SAT_WITH_NO_QUIET = 0.08


@dataclass
class Verdict:
    metrics: dict
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def _tile_stats(luma, w: int, h: int) -> tuple[float, float, float]:
    """(quiet_frac, busy_frac, lap_var) of mean |4-neighbor Laplacian| over an
    8x8 tile grid, interior pixels only — mirrors quality.go tileStats."""
    import numpy as np

    c = luma[1:-1, 1:-1]
    lap = 4 * c - luma[1:-1, :-2] - luma[1:-1, 2:] - luma[:-2, 1:-1] - luma[2:, 1:-1]
    lap_var = float(lap.var())
    ys, xs = np.mgrid[1 : h - 1, 1 : w - 1]
    ti = (ys * TILE_GRID // h) * TILE_GRID + xs * TILE_GRID // w
    sums = np.bincount(ti.ravel(), weights=np.abs(lap).ravel(), minlength=TILE_GRID**2)
    cnts = np.bincount(ti.ravel(), minlength=TILE_GRID**2)
    tile_hf = sums[cnts > 0] / cnts[cnts > 0]
    quiet = float((tile_hf < QUIET_TILE_THRESHOLD).mean())
    busy = float((tile_hf > BUSY_TILE_THRESHOLD).mean())
    return quiet, busy, lap_var


def measure(image) -> dict:
    """Raw tripwire metrics for a PIL image — mirrors quality.go Measure."""
    import numpy as np

    rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    h, w = rgb.shape[:2]
    n = float(w * h)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    hist = np.bincount(np.clip(luma.astype(np.int64), 0, 255).ravel(), minlength=256)
    p = hist[hist > 0] / n
    entropy = float(-(p * np.log2(p)).sum())
    std = float(luma.std())

    a, b = luma[:, :-1].ravel(), luma[:, 1:].ravel()
    den = a.std() * b.std()
    corr = float(((a * b).mean() - a.mean() * b.mean()) / den) if den > 0 else 0.0

    quiet, busy, lap_var = _tile_stats(luma, w, h)
    cw, ch = w // 4, h // 4
    coarse = luma[: ch * 4, : cw * 4].reshape(ch, 4, cw, 4).mean(axis=(1, 3))
    _, coarse_busy, _ = _tile_stats(coarse, cw, ch)

    sat = float(((rgb.max(axis=2) - rgb.min(axis=2)) > 200).mean())
    return {
        "width": w, "height": h, "entropy": entropy, "luma_std": std,
        "neighbor_corr": corr, "lap_var": lap_var, "quiet_tile_frac": quiet,
        "busy_tile_frac": busy, "coarse_busy_tile_frac": coarse_busy,
        "sat_frac": sat,
    }


def check_image(image) -> Verdict:
    m = measure(image)
    fails = []
    if m["entropy"] < MIN_ENTROPY:
        fails.append(f"entropy {m['entropy']:.2f} < {MIN_ENTROPY} (histogram collapsed)")
    if m["luma_std"] < MIN_LUMA_STDDEV:
        fails.append(f"luma stddev {m['luma_std']:.1f} < {MIN_LUMA_STDDEV} (uniform frame)")
    if m["neighbor_corr"] < MIN_NEIGHBOR_CORR:
        fails.append(
            f"neighbor correlation {m['neighbor_corr']:.3f} < {MIN_NEIGHBOR_CORR} "
            "(unstructured noise)"
        )
    if m["quiet_tile_frac"] == 0 and m["coarse_busy_tile_frac"] >= MAX_COARSE_BUSY_WITH_NO_QUIET:
        fails.append(
            f"no quiet tiles and {m['coarse_busy_tile_frac']:.0%} coarse-busy tiles "
            "(latent-garbage texture at every scale)"
        )
    if m["quiet_tile_frac"] == 0 and m["sat_frac"] > MAX_SAT_WITH_NO_QUIET:
        fails.append(
            f"no quiet tiles and {m['sat_frac']:.1%} chroma-clipped pixels "
            "(garbled saturation)"
        )
    return Verdict(metrics=m, failures=fails)


# ---------------------------------------------------------------------------


def test_flux2_klein_turbo_generates_real_image(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    pytest.importorskip("numpy")
    diffusers = pytest.importorskip("diffusers")
    from huggingface_hub import snapshot_download

    from gen_worker.models.loading import load_from_pretrained
    from gen_worker.models.memory import place_pipeline

    t0 = time.monotonic()
    snapshot = snapshot_download(REPO_ID, allow_patterns=ALLOW_PATTERNS)
    t_download = time.monotonic() - t0

    # The production loading path: dtype mirrors the endpoint binding
    # (HF("black-forest-labs/FLUX.2-klein-4B", dtype="bf16")), placement is
    # worker-owned.
    t0 = time.monotonic()
    pipe = load_from_pretrained(diffusers.Flux2KleinPipeline, snapshot, dtype="bf16")
    placement = place_pipeline(pipe)
    t_load = time.monotonic() - t0

    # NATIVE-PARAMS RULE: prompt + fixed seed only; native resolution and the
    # distilled step count. Turbo ignores guidance.
    t0 = time.monotonic()
    image = pipe(
        PROMPT,
        num_inference_steps=STEPS,
        generator=torch.Generator("cpu").manual_seed(SEED),
    ).images[0]
    t_generate = time.monotonic() - t0

    out_dir = Path(os.environ.get("GEN_WORKER_SMOKE_OUT", tmp_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "flux2-klein-4b-turbo-smoke.png"
    image.save(out_path)

    verdict = check_image(image)
    metrics = " ".join(
        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in verdict.metrics.items()
    )
    print(
        f"\nsmoke: download={t_download:.0f}s load={t_load:.0f}s "
        f"generate={t_generate:.1f}s placement={placement} -> {out_path}\n"
        f"tripwire: {'PASS' if verdict.ok else 'FAIL'} {metrics}"
    )
    assert not math.isnan(verdict.metrics["entropy"])
    assert verdict.ok, f"generated image tripped the quality gate: {verdict.failures} [{metrics}]"
