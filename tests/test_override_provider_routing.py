"""Tests for gen-worker #18: route override `provider` through the
worker so cross-provider `_models` overrides resolve through the right
downloader branch + reject pickle-only override snapshots.

The bug (pre #18): `Worker._resolved_models_for_request` extracted only
`{ref, tag, flavor}` from each stamped entry; `_resolve_model_id_for_injection`
then called `_resolved_repo_id(...)` with no `provider=`, silently
defaulting to "tensorhub". So an `HFRepo` binding overridden with another
HF ref was mis-routed to the tensorhub path and failed at download time.

The fix:
  1. Extract `provider` on the wire (with "tensorhub" back-compat default).
  2. Pass `provider=` into `_resolved_repo_id`.
  3. Layer per-request override entries into `_provider_by_ref` so the
     downloader's bare-ref→provider lookup finds the override.
  4. Belt-and-braces safetensors-only gate for override downloads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from gen_worker import HFRepo, Repo
from gen_worker.models.hf_downloader import HuggingFaceDownloadResult, HuggingFaceRef
from gen_worker.models.ref_downloader import (
    ModelRefDownloader,
    lookup_provider_for_ref,
    reset_override_ref_keys,
    reset_provider_by_ref,
    set_override_ref_keys,
    set_provider_by_ref,
)
from gen_worker.models.unsafe_format import (
    UnsafeFileFormat,
    assert_safe_weight_format,
)
from gen_worker.worker import InjectionSpec, Worker, _resolved_repo_id


# -----------------------------------------------------------------------------
# (1) Provider extraction from the wire
# -----------------------------------------------------------------------------


class _Pipe:
    pass


def _bare_worker() -> Worker:
    w = Worker.__new__(Worker)
    w._release_allowed_model_ids = None
    return w


def test_resolved_models_extracts_provider_from_dict_entry() -> None:
    """Dict-shaped entries (test fixtures) carry `provider` — extract it."""
    w = _bare_worker()
    fake_request = MagicMock()
    fake_request.resolved_models = {
        "pipeline": {
            "ref": "other-org/their-flux",
            "tag": "prod",
            "flavor": "bf16",
            "provider": "hf",
        }
    }
    out = w._resolved_models_for_request(fake_request)
    assert out == {
        "pipeline": {
            "ref": "other-org/their-flux",
            "tag": "prod",
            "flavor": "bf16",
            "provider": "hf",
        }
    }


def test_resolved_models_extracts_provider_from_proto_shape() -> None:
    """Proto-shaped entries expose `provider` as an attribute, not a dict
    key. The wire format ships both shapes through the same parser.
    """
    w = _bare_worker()

    class _Entry:
        pass

    entry = _Entry()
    entry.ref = "owner/repo"
    entry.tag = "prod"
    entry.flavor = "bf16"
    entry.provider = "hf"

    fake_request = MagicMock()
    fake_request.resolved_models = {"pipeline": entry}
    out = w._resolved_models_for_request(fake_request)
    assert out["pipeline"]["provider"] == "hf"
    assert out["pipeline"]["ref"] == "owner/repo"


def test_resolved_models_defaults_provider_to_tensorhub_for_back_compat() -> None:
    """Pre-#358 orchestrators stamped `{ref, tag, flavor}` with no provider
    field. Those entries are tensorhub by definition of the old wire
    format — the worker must accept them and assume tensorhub.
    """
    w = _bare_worker()
    fake_request = MagicMock()
    fake_request.resolved_models = {
        "pipeline": {"ref": "acme/flux", "tag": "prod", "flavor": "bf16"},
    }
    out = w._resolved_models_for_request(fake_request)
    assert out["pipeline"]["provider"] == "tensorhub"


def test_resolved_models_empty_provider_falls_back_to_tensorhub() -> None:
    """Empty-string provider on the wire must round-trip to tensorhub
    rather than propagate as an empty provider that downstream code
    would mis-handle.
    """
    w = _bare_worker()
    fake_request = MagicMock()
    fake_request.resolved_models = {
        "pipeline": {"ref": "x/y", "tag": "prod", "flavor": "", "provider": ""},
    }
    out = w._resolved_models_for_request(fake_request)
    assert out["pipeline"]["provider"] == "tensorhub"


# -----------------------------------------------------------------------------
# (2) Cross-provider override resolution: model_id key carries the right
#     provider so the cache and the downloader don't collide tensorhub and HF.
# -----------------------------------------------------------------------------


def test_hf_binding_with_hf_override_resolves_through_hf_key() -> None:
    """HFRepo binding overridden with `{provider: "hf", ref: ...}` must
    produce a model_id keyed on the HF provider — pre-#18 this defaulted
    to tensorhub.
    """
    w = _bare_worker()
    inj = InjectionSpec(
        param_name="pipeline",
        param_type=_Pipe,
        binding=HFRepo("orig/hf-flux").flavor("bf16").allow_override(_Pipe),
    )
    import msgspec

    class _P(msgspec.Struct):
        prompt: str = ""

    model_id, _ = w._resolve_model_id_for_injection(
        "fn",
        inj,
        payload=_P(),
        resolved_models={
            "pipeline": {
                "ref": "other-org/their-flux",
                "tag": "prod",
                "flavor": "bf16",
                "provider": "hf",
            }
        },
    )
    # The HF-provider key form is `hf::owner/repo#flavor` — distinct from
    # the tensorhub form which has no prefix.
    assert model_id.startswith("hf::")
    assert "other-org/their-flux" in model_id


def test_hf_binding_with_tensorhub_override_resolves_through_tensorhub_key() -> None:
    """HFRepo binding overridden with `{provider: "tensorhub", ...}` lands
    on the tensorhub key shape (no `provider::` prefix).
    """
    w = _bare_worker()
    inj = InjectionSpec(
        param_name="pipeline",
        param_type=_Pipe,
        binding=HFRepo("orig/hf-flux").flavor("bf16").allow_override(_Pipe),
    )
    import msgspec

    class _P(msgspec.Struct):
        prompt: str = ""

    model_id, _ = w._resolve_model_id_for_injection(
        "fn",
        inj,
        payload=_P(),
        resolved_models={
            "pipeline": {
                "ref": "acme/cozy-mirror",
                "tag": "prod",
                "flavor": "bf16",
                "provider": "tensorhub",
            }
        },
    )
    # Tensorhub keys have no provider prefix.
    assert not model_id.startswith("hf::")
    assert "acme/cozy-mirror" in model_id


def test_tensorhub_binding_with_hf_override_resolves_through_hf_key() -> None:
    """Reverse cross-provider: tensorhub binding overridden with an HF ref
    must produce an HF-keyed model_id so the downloader routes to HF.
    """
    w = _bare_worker()
    inj = InjectionSpec(
        param_name="pipeline",
        param_type=_Pipe,
        binding=Repo("acme/flux").flavor("bf16").allow_override(_Pipe),
    )
    import msgspec

    class _P(msgspec.Struct):
        prompt: str = ""

    model_id, _ = w._resolve_model_id_for_injection(
        "fn",
        inj,
        payload=_P(),
        resolved_models={
            "pipeline": {
                "ref": "owner/their-hf",
                "tag": "prod",
                "flavor": "bf16",
                "provider": "hf",
            }
        },
    )
    assert model_id.startswith("hf::")
    assert "owner/their-hf" in model_id


def test_override_without_provider_field_defaults_to_tensorhub() -> None:
    """Back-compat: a stamped entry that omits `provider` must resolve
    through the tensorhub path (pre-#358 orchestrator behavior).
    """
    w = _bare_worker()
    inj = InjectionSpec(
        param_name="pipeline",
        param_type=_Pipe,
        binding=Repo("acme/flux").flavor("bf16").allow_override(_Pipe),
    )
    import msgspec

    class _P(msgspec.Struct):
        prompt: str = ""

    model_id, _ = w._resolve_model_id_for_injection(
        "fn",
        inj,
        payload=_P(),
        resolved_models={
            "pipeline": {"ref": "acme/other", "tag": "prod", "flavor": "bf16"},
        },
    )
    # No provider prefix — landed on tensorhub.
    assert not model_id.startswith("hf::")
    assert "acme/other" in model_id


# -----------------------------------------------------------------------------
# (3) Safetensors-only gate on override downloads
# -----------------------------------------------------------------------------


def _make_weight_files(root: Path, files: List[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for f in files:
        p = root / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 16)


def test_safetensors_gate_accepts_safetensors_only_snapshot(tmp_path: Path) -> None:
    snap = tmp_path / "snap"
    _make_weight_files(snap, ["model.safetensors", "config.json"])
    # No exception — accepted.
    assert_safe_weight_format(snap, ref="owner/repo")


def test_safetensors_gate_accepts_flashpack_only_snapshot(tmp_path: Path) -> None:
    snap = tmp_path / "snap"
    _make_weight_files(snap, ["model.flashpack"])
    assert_safe_weight_format(snap)


def test_safetensors_gate_accepts_when_safetensors_sibling_exists(tmp_path: Path) -> None:
    """HF back-compat repos often ship both pytorch_model.bin and
    model.safetensors. The safetensors sibling is enough — accept.
    """
    snap = tmp_path / "snap"
    _make_weight_files(snap, ["pytorch_model.bin", "model.safetensors"])
    assert_safe_weight_format(snap)


def test_safetensors_gate_rejects_bin_only_snapshot(tmp_path: Path) -> None:
    snap = tmp_path / "snap"
    _make_weight_files(snap, ["pytorch_model.bin"])
    with pytest.raises(UnsafeFileFormat, match="refusing to load"):
        assert_safe_weight_format(snap, ref="bad/repo")


def test_safetensors_gate_rejects_pt_only_snapshot(tmp_path: Path) -> None:
    snap = tmp_path / "snap"
    _make_weight_files(snap, ["model.pt"])
    with pytest.raises(UnsafeFileFormat):
        assert_safe_weight_format(snap)


def test_safetensors_gate_rejects_ckpt_only_snapshot(tmp_path: Path) -> None:
    snap = tmp_path / "snap"
    _make_weight_files(snap, ["model.ckpt"])
    with pytest.raises(UnsafeFileFormat):
        assert_safe_weight_format(snap)


def test_safetensors_gate_no_op_on_metadata_only_snapshot(tmp_path: Path) -> None:
    """No weight files at all — let the downstream loader produce its own
    error rather than failing the safety gate prematurely.
    """
    snap = tmp_path / "snap"
    _make_weight_files(snap, ["config.json", "tokenizer.json"])
    assert_safe_weight_format(snap)


# -----------------------------------------------------------------------------
# (4) End-to-end: downloader applies the safetensors gate only to override
#     refs, not to binding-default refs.
# -----------------------------------------------------------------------------


class _FakeHFDownloader:
    """Stand-in for HuggingFaceHubDownloader. The local_dir we return
    becomes the snapshot the safetensors gate inspects.
    """

    def __init__(self, snapshot_dir: Path) -> None:
        self.snapshot_dir = snapshot_dir
        self.calls: List[HuggingFaceRef] = []

    def download(self, ref: HuggingFaceRef, progress_callback: Any = None) -> HuggingFaceDownloadResult:
        if progress_callback is not None:
            progress_callback(1, 1)
        self.calls.append(ref)
        return HuggingFaceDownloadResult(local_dir=self.snapshot_dir)


def test_downloader_runs_gate_for_override_ref_with_pickle_only(tmp_path: Path) -> None:
    """A ref marked as override + a snapshot containing only `.bin` →
    UnsafeFileFormat raised by the downloader.
    """
    snap = tmp_path / "snap"
    _make_weight_files(snap, ["pytorch_model.bin"])

    dl = ModelRefDownloader()
    dl._hf = _FakeHFDownloader(snap)  # type: ignore[assignment]

    ref = "bad-org/pickle-repo"
    prov_tok = set_provider_by_ref({ref: "hf"})
    keys_tok = set_override_ref_keys([ref])
    try:
        with pytest.raises(UnsafeFileFormat, match="refusing to load"):
            dl.download(ref, str(tmp_path))
    finally:
        reset_override_ref_keys(keys_tok)
        reset_provider_by_ref(prov_tok)


def test_downloader_skips_gate_for_non_override_ref(tmp_path: Path) -> None:
    """Binding-default refs don't carry the override-key marker → the
    gate doesn't fire even if the snapshot is pickle-only. (Binding-
    default refs went through the build-time validator at deploy time;
    the runtime gate is purely belt-and-braces for overrides.)
    """
    snap = tmp_path / "snap"
    _make_weight_files(snap, ["pytorch_model.bin"])

    dl = ModelRefDownloader()
    dl._hf = _FakeHFDownloader(snap)  # type: ignore[assignment]

    ref = "binding/default-repo"
    prov_tok = set_provider_by_ref({ref: "hf"})
    # No set_override_ref_keys() — this isn't an override.
    try:
        out = dl.download(ref, str(tmp_path))
    finally:
        reset_provider_by_ref(prov_tok)
    assert Path(out) == snap


def test_downloader_skips_gate_when_override_ref_has_safetensors(tmp_path: Path) -> None:
    """Override ref + safetensors snapshot → download succeeds; gate
    is satisfied.
    """
    snap = tmp_path / "snap"
    _make_weight_files(snap, ["model.safetensors"])

    dl = ModelRefDownloader()
    dl._hf = _FakeHFDownloader(snap)  # type: ignore[assignment]

    ref = "good/safetensors-repo"
    prov_tok = set_provider_by_ref({ref: "hf"})
    keys_tok = set_override_ref_keys([ref])
    try:
        out = dl.download(ref, str(tmp_path))
    finally:
        reset_override_ref_keys(keys_tok)
        reset_provider_by_ref(prov_tok)
    assert Path(out) == snap


# -----------------------------------------------------------------------------
# (5) Per-request contextvar isolation: `_provider_by_ref` overrides set
#     during one request must not leak into the next.
# -----------------------------------------------------------------------------


def test_override_keys_contextvar_resets_cleanly() -> None:
    """After `reset_override_ref_keys` the contextvar is back to its
    pre-set value — no leak across requests sharing the same thread.
    """
    from gen_worker.models.ref_downloader import _override_ref_keys

    assert _override_ref_keys.get() is None
    tok = set_override_ref_keys({"a/b", "c/d"})
    assert _override_ref_keys.get() == frozenset({"a/b", "c/d"})
    reset_override_ref_keys(tok)
    assert _override_ref_keys.get() is None


def test_provider_by_ref_layering_no_leak() -> None:
    """Two nested `set_provider_by_ref` calls (build-time index, then
    per-request overrides) tear down correctly — the var lands back at
    its original value after both resets.
    """
    from gen_worker.models.ref_downloader import _provider_by_ref

    assert _provider_by_ref.get() is None
    base_tok = set_provider_by_ref({"binding/ref": "hf"})
    req_tok = set_provider_by_ref({"binding/ref": "hf", "override/ref": "tensorhub"})
    # Both layers visible.
    assert lookup_provider_for_ref("override/ref") == "tensorhub"
    # Tear down request layer first, then base.
    reset_provider_by_ref(req_tok)
    assert lookup_provider_for_ref("override/ref") == "tensorhub"  # default fallback
    assert lookup_provider_for_ref("binding/ref") == "hf"
    reset_provider_by_ref(base_tok)
    assert _provider_by_ref.get() is None
