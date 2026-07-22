"""th#1017: inference regimes — @endpoint's ``regimes=`` decorator surface,
discovery manifest emission, Slot resolution's ``.regime``/backstop, and the
converter's scheduler-config regime stamp. Real discovery/registry/convert
code, no mocking (same idiom as test_variant_th1004.py).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import msgspec
import pytest


@pytest.fixture()
def tmp_pkg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


def _write(pkg: Path, main_src: str) -> None:
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(textwrap.dedent(main_src))


# ---------------------------------------------------------------------------
# decorator surface + discovery manifest emission
# ---------------------------------------------------------------------------


def test_class_per_method_regimes_round_trip_into_manifest(tmp_pkg: Path) -> None:
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_r_method"
    _write(pkg, """
        import msgspec
        from gen_worker import RequestContext, endpoint

        class In_(msgspec.Struct):
            prompt: str = ""

        class Out_(msgspec.Struct):
            y: str

        @endpoint(regimes={"generate_distilled": ("distilled",)})
        class Gen:
            def generate(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="base")

            def generate_distilled(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="distilled")
    """)

    fns = {f["name"]: f for f in discover_functions(tmp_pkg, main_module="ep_r_method.main")}
    # Undeclared method defaults to ("standard",) -> omitted from the manifest.
    assert "regimes" not in fns["generate"]
    assert fns["generate-distilled"]["regimes"] == ["distilled"]


def test_function_form_tuple_regimes_round_trips(tmp_pkg: Path) -> None:
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_r_fn"
    _write(pkg, """
        import msgspec
        from gen_worker import RequestContext, endpoint

        class In_(msgspec.Struct):
            prompt: str = ""

        class Out_(msgspec.Struct):
            y: str

        @endpoint(regimes=("v_prediction", "standard"))
        def render(ctx: RequestContext, data: In_) -> Out_:
            return Out_(y="base")
    """)

    fns = {f["name"]: f for f in discover_functions(tmp_pkg, main_module="ep_r_fn.main")}
    assert set(fns["render"]["regimes"]) == {"v_prediction", "standard"}


def test_function_form_default_regimes_omitted_from_manifest(tmp_pkg: Path) -> None:
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_r_fn_default"
    _write(pkg, """
        import msgspec
        from gen_worker import RequestContext, endpoint

        class In_(msgspec.Struct):
            prompt: str = ""

        class Out_(msgspec.Struct):
            y: str

        @endpoint
        def render(ctx: RequestContext, data: In_) -> Out_:
            return Out_(y="base")
    """)

    fns = {f["name"]: f for f in discover_functions(tmp_pkg, main_module="ep_r_fn_default.main")}
    assert "regimes" not in fns["render"]


# Module scope: typing.get_type_hints resolves handler annotations from the
# declaring module's globals, so spec-time structs cannot be test-local.
from gen_worker import RequestContext, endpoint


class RIn(msgspec.Struct):
    prompt: str = ""


class ROut(msgspec.Struct):
    y: str


def test_unknown_regime_string_rejected_at_decoration() -> None:
    with pytest.raises(ValueError, match="unknown regime"):
        @endpoint(regimes=("fried",))
        def render(ctx: RequestContext, data: RIn) -> ROut:  # pragma: no cover
            return ROut(y="x")


def test_class_regimes_unknown_method_rejected_at_decoration() -> None:
    with pytest.raises(ValueError, match="unknown handler method"):
        @endpoint(regimes={"nonexistent": ("distilled",)})
        class Gen:
            def generate(self, ctx: RequestContext, data: RIn) -> ROut:
                return ROut(y="x")


def test_function_form_rejects_per_method_mapping() -> None:
    with pytest.raises(TypeError, match="tuple of"):
        @endpoint(regimes={"render": ("distilled",)})
        def render(ctx: RequestContext, data: RIn) -> ROut:  # pragma: no cover
            return ROut(y="x")


def test_class_form_rejects_bare_tuple() -> None:
    with pytest.raises(TypeError, match="mapping"):
        @endpoint(regimes=("distilled",))
        class Gen:
            def generate(self, ctx: RequestContext, data: RIn) -> ROut:
                return ROut(y="x")


# ---------------------------------------------------------------------------
# Slot resolution: .regime + the executor-level backstop
# ---------------------------------------------------------------------------


def test_resolved_slot_carries_regime_from_resolve_slot() -> None:
    from gen_worker.api.binding import HF
    from gen_worker.api.slot import Slot, resolve_slot
    from gen_worker.families import SdxlDefaults

    slot = Slot(object, default_config=SdxlDefaults())
    ref = HF("acme/gonzalomo-xl")
    resolved = resolve_slot("pipeline", slot, ref=ref, inference_regime="distilled")
    assert resolved.regime == "distilled"


def test_resolve_slot_defaults_regime_to_standard() -> None:
    from gen_worker.api.binding import HF
    from gen_worker.api.slot import Slot, resolve_slot
    from gen_worker.families import SdxlDefaults

    slot = Slot(object, default_config=SdxlDefaults())
    resolved = resolve_slot("pipeline", slot, ref=HF("acme/plain-xl"))
    assert resolved.regime == "standard"


def test_regime_mismatch_raises_typed_backstop_error() -> None:
    from gen_worker.api.binding import HF
    from gen_worker.api.slot import RegimeMismatchError, Slot, resolve_slot
    from gen_worker.families import SdxlDefaults

    slot = Slot(object, default_config=SdxlDefaults())
    ref = HF("acme/gonzalomo-xl")
    with pytest.raises(RegimeMismatchError, match="distilled"):
        resolve_slot(
            "pipeline", slot, ref=ref,
            inference_regime="distilled", allowed_regimes=("standard",),
        )


def test_regime_within_allowed_set_resolves_cleanly() -> None:
    from gen_worker.api.binding import HF
    from gen_worker.api.slot import Slot, resolve_slot
    from gen_worker.families import SdxlDefaults

    slot = Slot(object, default_config=SdxlDefaults())
    ref = HF("acme/gonzalomo-xl")
    resolved = resolve_slot(
        "pipeline", slot, ref=ref,
        inference_regime="distilled", allowed_regimes=("standard", "distilled"),
    )
    assert resolved.regime == "distilled"


# ---------------------------------------------------------------------------
# hub_client: resolve_repo parses inference_regime (mirrors model_family)
# ---------------------------------------------------------------------------


def test_resolve_repo_parses_inference_regime(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    from gen_worker.models.hub_client import resolve_repo
    from gen_worker.models.refs import TensorhubRef

    class _FakeResp:
        status_code = 200

        def json(self) -> dict:
            return {
                "snapshot_digest": "abc123",
                "files": [{"path": "model.safetensors", "blake3": "b3", "url": "http://x/f",
                           "size_bytes": 10}],
                "model_family": "sdxl",
                "inference_regime": "distilled",
            }

    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResp())
    ref = TensorhubRef(owner="acme", repo="gonzalomo-xl", tag="prod", flavor="")
    resolved = resolve_repo(ref, base_url="https://hub.example")
    assert resolved.inference_regime == "distilled"


def test_resolve_repo_defaults_inference_regime_empty_on_older_hubs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests

    from gen_worker.models.hub_client import resolve_repo
    from gen_worker.models.refs import TensorhubRef

    class _FakeResp:
        status_code = 200

        def json(self) -> dict:
            return {
                "snapshot_digest": "abc123",
                "files": [{"path": "model.safetensors", "blake3": "b3", "url": "http://x/f",
                           "size_bytes": 10}],
            }

    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResp())
    ref = TensorhubRef(owner="acme", repo="plain-xl", tag="prod", flavor="")
    resolved = resolve_repo(ref, base_url="https://hub.example")
    assert resolved.inference_regime == ""


# ---------------------------------------------------------------------------
# convert: regime-correct scheduler config written into produced snapshots
# ---------------------------------------------------------------------------


def _diffusers_source(tmp_path: Path) -> "IngestedSource":  # noqa: F821
    from gen_worker.convert.ingest import IngestedSource

    src = tmp_path / "source"
    (src / "scheduler").mkdir(parents=True)
    (src / "scheduler" / "config.json").write_text(json.dumps({
        "_class_name": "EulerDiscreteScheduler", "prediction_type": "epsilon",
    }))
    (src / "unet").mkdir()
    (src / "unet" / "model.safetensors").write_bytes(b"\x00" * 8)
    (src / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionXLPipeline"}))
    return IngestedSource(
        provider="huggingface", source_ref="acme/gonzalomo-xl", source_revision="sha1",
        dir=src, layout="diffusers", model_family="sdxl", model_family_variant="",
        attrs={"dtype": "fp16"},
    )


def test_distilled_regime_writes_trailing_timestep_spacing(tmp_path: Path) -> None:
    from gen_worker.convert.clone import OutputSpec, build_flavor_tree

    source = _diffusers_source(tmp_path)
    spec = OutputSpec(dtype="fp16", file_layout="diffusers", file_type="safetensors")
    out_dir = tmp_path / "flavor-distilled"
    tree, _attrs = build_flavor_tree(source, spec, out_dir, inference_regime="distilled")

    cfg = json.loads((tree / "scheduler" / "config.json").read_text())
    assert cfg["timestep_spacing"] == "trailing"
    assert cfg["prediction_type"] == "epsilon"  # untouched


def test_v_prediction_regime_writes_zero_snr_v_pred(tmp_path: Path) -> None:
    from gen_worker.convert.clone import OutputSpec, build_flavor_tree

    source = _diffusers_source(tmp_path)
    spec = OutputSpec(dtype="fp16", file_layout="diffusers", file_type="safetensors")
    out_dir = tmp_path / "flavor-vpred"
    tree, _attrs = build_flavor_tree(source, spec, out_dir, inference_regime="v_prediction")

    cfg = json.loads((tree / "scheduler" / "config.json").read_text())
    assert cfg["prediction_type"] == "v_prediction"
    assert cfg["rescale_betas_zero_snr"] is True


def test_standard_regime_leaves_scheduler_config_untouched(tmp_path: Path) -> None:
    from gen_worker.convert.clone import OutputSpec, build_flavor_tree

    source = _diffusers_source(tmp_path)
    spec = OutputSpec(dtype="fp16", file_layout="diffusers", file_type="safetensors")
    out_dir = tmp_path / "flavor-standard"
    tree, _attrs = build_flavor_tree(source, spec, out_dir, inference_regime="standard")

    cfg = json.loads((tree / "scheduler" / "config.json").read_text())
    assert cfg == {"_class_name": "EulerDiscreteScheduler", "prediction_type": "epsilon"}


def test_apply_regime_scheduler_config_noop_without_scheduler_dir(tmp_path: Path) -> None:
    from gen_worker.convert.writer import apply_regime_scheduler_config

    out_dir = tmp_path / "singlefile-out"
    out_dir.mkdir()
    apply_regime_scheduler_config(out_dir, "distilled")  # must not raise
    assert not (out_dir / "scheduler").exists()
