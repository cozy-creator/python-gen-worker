"""gw#593 item 2: explicit source-file selection on the clone request.

``Lightricks/LTX-2.3`` bundles dev/distilled/distilled-lora/upscaler
checkpoints together at repo root; the classifier's dtype-variant heuristic
has no way to know the caller wants specifically
``ltx-2.3-22b-dev.safetensors`` (gw#593's item-1 fix makes every file land
untagged, so they group into one ~147GB bundle and hit the size refusal
instead of silently publishing the wrong file — a strict improvement, but
still unclonable). ``source_include`` is the caller's explicit selector:
dual-form (compact single glob string, or a structured list of globs),
matched against repo-relative paths, applied BEFORE classification so
``classify_repo`` only ever sees the caller-approved subset and every
existing strategy branch keeps working unchanged.

All synthetic file-listing fixtures — no network, no weight downloads.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gen_worker.convert.classifier import (
    SourceIncludeError,
    apply_source_include,
    classify_repo,
)
from gen_worker.convert.clone import normalize_source_include, run_clone
from gen_worker.convert.ingest import HFSourcePlan, IngestedSource

# Real Lightricks/LTX-2.3 root listing (2026-07-19 HF tree API; confirmed live
# via `curl https://huggingface.co/api/models/Lightricks/LTX-2.3/tree/main`).
_LTX23_FILES = {
    ".gitattributes": 1571,
    "LICENSE": 21399,
    "README.md": 6570,
    "ltx-2.3-22b-dev.safetensors": 46149344974,
    "ltx-2.3-22b-distilled-1.1.safetensors": 46149345334,
    "ltx-2.3-22b-distilled-lora-384-1.1.safetensors": 7605507256,
    "ltx-2.3-22b-distilled-lora-384.safetensors": 7605507256,
    "ltx-2.3-22b-distilled.safetensors": 46149345038,
    "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors": 1090125794,
    "ltx-2.3-spatial-upscaler-x2-1.0.safetensors": 995743504,
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors": 995743504,
    "ltx-2.3-temporal-upscaler-x2-1.0.safetensors": 995743504,
    "ltx2.3-open.png": 2259601,
}


# ---------------------------------------------------------------------------
# classifier.apply_source_include — pure glob-filter semantics
# ---------------------------------------------------------------------------

def test_apply_source_include_noop_when_absent() -> None:
    paths = list(_LTX23_FILES)
    assert apply_source_include(paths, ()) == paths
    assert apply_source_include(paths, None) == paths  # type: ignore[arg-type]


def test_apply_source_include_single_glob_pins_one_file() -> None:
    out = apply_source_include(list(_LTX23_FILES), ["ltx-2.3-22b-dev.safetensors"])
    assert out == ["ltx-2.3-22b-dev.safetensors"]


def test_apply_source_include_multiple_globs_union() -> None:
    out = apply_source_include(
        list(_LTX23_FILES),
        ["ltx-2.3-22b-dev.safetensors", "*.md", "LICENSE"],
    )
    assert set(out) == {"ltx-2.3-22b-dev.safetensors", "README.md", "LICENSE"}


def test_apply_source_include_glob_star_pattern() -> None:
    out = apply_source_include(list(_LTX23_FILES), ["*dev*"])
    assert out == ["ltx-2.3-22b-dev.safetensors"]


def test_apply_source_include_nested_path_glob() -> None:
    paths = [
        "model_index.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/model.safetensors",
    ]
    out = apply_source_include(paths, ["transformer/*", "vae/*"])
    assert set(out) == {
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    }


def test_apply_source_include_unmatched_glob_fails_loud() -> None:
    with pytest.raises(SourceIncludeError) as exc:
        apply_source_include(list(_LTX23_FILES), ["ltx-2.3-22b-dev.safetensors", "*.typo-ext"])
    err = exc.value
    assert err.unmatched == ["*.typo-ext"]
    # The error names the bad glob AND what the good glob(s) matched.
    assert "*.typo-ext" in str(err)
    assert "ltx-2.3-22b-dev.safetensors" in str(err)
    assert err.matched["ltx-2.3-22b-dev.safetensors"] == ["ltx-2.3-22b-dev.safetensors"]


def test_apply_source_include_all_globs_unmatched() -> None:
    with pytest.raises(SourceIncludeError) as exc:
        apply_source_include(list(_LTX23_FILES), ["nope-1.safetensors", "nope-2.safetensors"])
    assert set(exc.value.unmatched) == {"nope-1.safetensors", "nope-2.safetensors"}


# ---------------------------------------------------------------------------
# classify_repo end-to-end with the filtered listing (the actual gw#593
# blocker: pinning JUST the dev checkpoint out of the 147GB root bundle)
# ---------------------------------------------------------------------------

def test_source_include_resolves_the_ltx23_dev_checkpoint() -> None:
    filtered = apply_source_include(list(_LTX23_FILES), ["ltx-2.3-22b-dev.safetensors"])
    sizes = {p: _LTX23_FILES[p] for p in filtered}
    c = classify_repo(filtered, sizes=sizes)
    assert c.strategy == "aio_singlefile"
    assert c.allow_patterns == ["ltx-2.3-22b-dev.safetensors"]


def test_without_source_include_the_full_bundle_still_refuses_too_large() -> None:
    """Guardrail: confirms this fixture is the REAL gw#593 blocker (unrestricted
    classify_repo still can't resolve it) — source_include is what unblocks it."""
    from gen_worker.convert.classifier import RepoRefusal

    with pytest.raises(RepoRefusal) as exc:
        classify_repo(list(_LTX23_FILES), sizes=_LTX23_FILES)
    assert exc.value.reason == "too_large"


# ---------------------------------------------------------------------------
# clone.normalize_source_include — dual-form input (compact string / list)
# ---------------------------------------------------------------------------

def test_normalize_source_include_none_is_empty() -> None:
    assert normalize_source_include(None) == ()


def test_normalize_source_include_compact_string_form() -> None:
    assert normalize_source_include("ltx-2.3-22b-dev.safetensors") == (
        "ltx-2.3-22b-dev.safetensors",
    )


def test_normalize_source_include_blank_string_is_empty() -> None:
    assert normalize_source_include("   ") == ()


def test_normalize_source_include_structured_list_form() -> None:
    assert normalize_source_include(
        ["ltx-2.3-22b-dev.safetensors", "text_encoder/**"]
    ) == ("ltx-2.3-22b-dev.safetensors", "text_encoder/**")


def test_normalize_source_include_dedupes_preserving_order() -> None:
    assert normalize_source_include(["a/*", "b/*", "a/*"]) == ("a/*", "b/*")


def test_normalize_source_include_rejects_other_types() -> None:
    with pytest.raises(ValueError):
        normalize_source_include(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# run_clone plumbing — source_include reaches plan_huggingface/
# ingest_huggingface, and provider guard rails.
# ---------------------------------------------------------------------------

class _Ctx:
    def __init__(self) -> None:
        self._file_api_base_url = "http://127.0.0.1:1"
        self._worker_capability_token = "cap-token"
        self.owner = "acme"
        self.request_id = "req-1"
        self.destination = {"repo": "acme/fallback"}


def test_run_clone_threads_source_include_into_plan_and_ingest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    calls: dict[str, object] = {}

    def fake_plan(source_ref, **kwargs):
        calls["plan_source_include"] = kwargs.get("source_include")
        return HFSourcePlan(
            repo_id=source_ref, revision="sha-1", paths=["ltx-2.3-22b-dev.safetensors"],
            sizes={"ltx-2.3-22b-dev.safetensors": 64},
            side={}, classification=classify_repo(
                ["ltx-2.3-22b-dev.safetensors"], sizes={"ltx-2.3-22b-dev.safetensors": 64}),
            content_ids={},
        )

    def fake_ingest(source_ref, dest_dir, **kwargs):
        calls["ingest_source_include"] = kwargs.get("source_include")
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "ltx-2.3-22b-dev.safetensors").write_bytes(b"\x00" * 64)
        return IngestedSource(
            provider="huggingface", source_ref=source_ref, source_revision="sha-1",
            dir=dest_dir, layout="singlefile", model_family="ltx2", model_family_variant="ltx2",
            classification=SimpleNamespace(strategy="aio_singlefile"),
            attrs={"dtype": "bf16", "file_layout": "singlefile"},
            metadata={"source_provider": "huggingface"},
            repo_spec={"kind": "model", "library_name": ""},
        )

    monkeypatch.setattr("gen_worker.convert.clone.plan_huggingface", fake_plan)
    monkeypatch.setattr("gen_worker.convert.clone.ingest_huggingface", fake_ingest)

    class _Hub:
        def __init__(self, *a, **k) -> None:
            pass

        @staticmethod
        def from_ctx(ctx):
            return _Hub()

        def commit(self, **kwargs):
            return SimpleNamespace(
                revision_id="rev-1", checkpoint_id="ck-1", uploaded=1,
                deduped=0, total_bytes=64,
            )

    monkeypatch.setattr("gen_worker.convert.clone.HubClient", _Hub)

    result = run_clone(
        _Ctx(),
        provider="huggingface",
        source_ref="Lightricks/LTX-2.3",
        destination_repo="acme/ltx-2.3-dev",
        outputs=[{"dtype": "bf16", "file_layout": "diffusers", "file_type": "safetensors"}],
        source_include="ltx-2.3-22b-dev.safetensors",
    )
    assert calls["plan_source_include"] == ("ltx-2.3-22b-dev.safetensors",)
    assert calls["ingest_source_include"] == ("ltx-2.3-22b-dev.safetensors",)
    assert result.published


def test_run_clone_rejects_source_include_for_civitai() -> None:
    with pytest.raises(ValueError, match="only supported for provider='huggingface'"):
        run_clone(
            _Ctx(),
            provider="civitai",
            civitai_model_version_id=1,
            destination_repo="acme/thing",
            source_include="*.safetensors",
        )
