"""Civitai ref resolution in ``gen-worker run`` (issue #351 / #341).

``_resolve_local_path`` treats a ``CivitaiRepo`` ref as a MODEL id and resolves
it to the latest published version — unless a version is pinned via
``CivitaiRepo.version()`` (threaded in as ``civitai_version_id``), in which case
it is used directly with no model lookup. A failed model lookup or a model with
no versions fails loud (no silent fallback to treating the id as a version).

Stubs the network leaves (``fetch_civitai_model`` /
``download_civitai_model_version_files``) so the resolution logic is exercised
without hitting civitai.com.
"""

from __future__ import annotations

import pytest

import gen_worker.cli.run as run_mod
import gen_worker.conversion.ingest as ingest
import gen_worker.models.ref_downloader as rd


@pytest.fixture(autouse=True)
def _stub_leaves(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORHUB_CAS_DIR", str(tmp_path))
    # numeric-id parser + artifact path are not what we're testing — make them trivial.
    monkeypatch.setattr(rd, "_parse_civitai_model_version_id", lambda s: int(str(s).strip()))
    monkeypatch.setattr(rd, "_civitai_local_artifact_path", lambda out, info: str(out) + "/model.safetensors")


def _no_emit(_e):  # event sink
    pass


def test_pinned_version_skips_model_lookup(monkeypatch):
    def _must_not_call(_mid):
        raise AssertionError("fetch_civitai_model must not run when a version is pinned")

    downloaded = {}
    monkeypatch.setattr(ingest, "fetch_civitai_model", _must_not_call)
    monkeypatch.setattr(
        ingest, "download_civitai_model_version_files",
        lambda vid, out, civitai_api_key="": downloaded.setdefault("vid", vid) or {},
    )
    path = run_mod._resolve_local_path(
        ref="123456", provider="civitai", offline=False, emit=_no_emit,
        civitai_version_id="789012",
    )
    assert downloaded["vid"] == 789012
    assert path.endswith("model.safetensors")


def test_happy_path_resolves_latest_version(monkeypatch):
    monkeypatch.setattr(
        ingest, "fetch_civitai_model",
        lambda mid: {"id": mid, "modelVersions": [{"id": 555}, {"id": 111}]},
    )
    captured = {}
    monkeypatch.setattr(
        ingest, "download_civitai_model_version_files",
        lambda vid, out, civitai_api_key="": captured.setdefault("vid", vid) or {},
    )
    run_mod._resolve_local_path(ref="123456", provider="civitai", offline=False, emit=_no_emit)
    assert captured["vid"] == 555  # modelVersions[0] = latest


def test_failed_model_lookup_raises_no_download(monkeypatch):
    monkeypatch.setattr(
        ingest, "fetch_civitai_model",
        lambda mid: (_ for _ in ()).throw(ValueError("404 not a model")),
    )
    dl = {"n": 0}
    monkeypatch.setattr(
        ingest, "download_civitai_model_version_files",
        lambda *a, **k: dl.__setitem__("n", dl["n"] + 1) or {},
    )
    with pytest.raises(run_mod._ModelResolutionError):
        run_mod._resolve_local_path(ref="999", provider="civitai", offline=False, emit=_no_emit)
    assert dl["n"] == 0  # never silently downloaded a guessed version


def test_model_with_no_versions_raises_no_download(monkeypatch):
    monkeypatch.setattr(ingest, "fetch_civitai_model", lambda mid: {"id": mid, "modelVersions": []})
    dl = {"n": 0}
    monkeypatch.setattr(
        ingest, "download_civitai_model_version_files",
        lambda *a, **k: dl.__setitem__("n", dl["n"] + 1) or {},
    )
    with pytest.raises(run_mod._ModelResolutionError):
        run_mod._resolve_local_path(ref="123456", provider="civitai", offline=False, emit=_no_emit)
    assert dl["n"] == 0


def test_offline_civitai_raises(monkeypatch):
    with pytest.raises(run_mod._ModelResolutionError):
        run_mod._resolve_local_path(ref="123456", provider="civitai", offline=True, emit=_no_emit)
