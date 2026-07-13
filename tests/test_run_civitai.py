"""Civitai ref resolution for the local CLI (issue #351 / #341).

``provision.resolve_local_path`` treats a ``CivitaiRepo`` ref as a MODEL id and resolves
it to the latest published version — unless a version is pinned via
``CivitaiRepo.version()`` (threaded in as ``civitai_version_id``), in which case
it is used directly with no model lookup. A failed model lookup or a model with
no versions fails loud (no silent fallback to treating the id as a version).

Stubs the network leaves (``fetch_civitai_model`` / ``download_civitai``) so
the resolution logic is exercised without hitting civitai.com.
"""

from __future__ import annotations

import pytest

import gen_worker.models.download as dl_mod
import gen_worker.models.provision as prov_mod


@pytest.fixture(autouse=True)
def _cas_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORHUB_CAS_DIR", str(tmp_path))


def _no_emit(_e):  # event sink
    pass


def test_pinned_version_skips_model_lookup(monkeypatch, tmp_path):
    def _must_not_call(_mid, **_kw):
        raise AssertionError("fetch_civitai_model must not run when a version is pinned")

    downloaded = {}

    def _fake_download(vid, out, **kw):
        downloaded["vid"] = vid
        return out / "model.safetensors"

    monkeypatch.setattr(dl_mod, "fetch_civitai_model", _must_not_call)
    monkeypatch.setattr(dl_mod, "download_civitai", _fake_download)
    path = prov_mod.resolve_local_path(
        ref="123456", provider="civitai", offline=False, emit=_no_emit,
        civitai_version_id="789012",
    )
    assert downloaded["vid"] == 789012
    assert path.endswith("model.safetensors")


def test_happy_path_resolves_latest_version(monkeypatch):
    monkeypatch.setattr(
        dl_mod, "fetch_civitai_model",
        lambda mid, **_kw: {"id": mid, "modelVersions": [{"id": 555}, {"id": 111}]},
    )
    captured = {}
    monkeypatch.setattr(
        dl_mod, "download_civitai",
        lambda vid, out, **kw: captured.setdefault("vid", vid) or out,
    )
    prov_mod.resolve_local_path(ref="123456", provider="civitai", offline=False, emit=_no_emit)
    assert captured["vid"] == 555  # modelVersions[0] = latest


def test_failed_model_lookup_raises_no_download(monkeypatch):
    monkeypatch.setattr(
        dl_mod, "fetch_civitai_model",
        lambda mid, **_kw: (_ for _ in ()).throw(ValueError("404 not a model")),
    )
    dl = {"n": 0}
    monkeypatch.setattr(
        dl_mod, "download_civitai",
        lambda *a, **k: dl.__setitem__("n", dl["n"] + 1) or a[1],
    )
    with pytest.raises(prov_mod.ModelResolutionError):
        prov_mod.resolve_local_path(ref="999", provider="civitai", offline=False, emit=_no_emit)
    assert dl["n"] == 0  # never silently downloaded a guessed version


def test_model_with_no_versions_raises_no_download(monkeypatch):
    monkeypatch.setattr(
        dl_mod, "fetch_civitai_model", lambda mid, **_kw: {"id": mid, "modelVersions": []}
    )
    dl = {"n": 0}
    monkeypatch.setattr(
        dl_mod, "download_civitai",
        lambda *a, **k: dl.__setitem__("n", dl["n"] + 1) or a[1],
    )
    with pytest.raises(prov_mod.ModelResolutionError):
        prov_mod.resolve_local_path(ref="123456", provider="civitai", offline=False, emit=_no_emit)
    assert dl["n"] == 0


def test_offline_civitai_raises(monkeypatch):
    with pytest.raises(prov_mod.ModelResolutionError):
        prov_mod.resolve_local_path(ref="123456", provider="civitai", offline=True, emit=_no_emit)
