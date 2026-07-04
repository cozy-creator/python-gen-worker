"""publish_flavors + run_clone against the fake /commits server.

Covers the producer publish contract (#375) and clone robustness (#374):
explicit publish, HF-cache junk never uploaded, keyed workdir retained on
failure / removed on success, and no-publish-cannot-read-as-success.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cozy_convert import ProducedFlavor, publish_flavors
from cozy_convert.clone import run_clone
from cozy_convert.ingest import IngestedSource

from conftest import _FakeHub


class _Ctx:
    def __init__(self, server) -> None:
        self._file_api_base_url = f"http://127.0.0.1:{server.server_port}"
        self._worker_capability_token = "cap-token"
        self.owner = "acme"
        self.request_id = "req-1"
        self.destination = {"repo": "acme/fallback"}


def test_publish_flavors_file_and_dir(fake_hub, tmp_path: Path) -> None:
    _FakeHub.state["finalize_calls"] = 1
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"\x00" * 32)
    tree = tmp_path / "tree"
    (tree / ".cache" / "huggingface").mkdir(parents=True)
    (tree / ".cache" / "huggingface" / "x.lock").write_text("junk")
    (tree / "config.json").write_text("{}")

    results = publish_flavors(
        _Ctx(fake_hub),
        [
            ProducedFlavor(path=weights, flavor="fp32"),
            ProducedFlavor(path=tree, flavor="bf16"),
        ],
        destination_repo="acme/dest",
    )
    assert [r.revision_id for r in results] == ["rev-1", "rev-1"]
    reqs = _FakeHub.state["commit_requests"]
    assert len(reqs) == 2
    assert reqs[0]["flavor"] == "fp32"
    assert [op["path"] for op in reqs[0]["operations"]] == ["weights.safetensors"]
    # directory flavor: HF-cache junk never reaches the commit
    assert [op["path"] for op in reqs[1]["operations"]] == ["config.json"]


def test_publish_flavors_destination_falls_back_to_ctx(fake_hub, tmp_path: Path) -> None:
    _FakeHub.state["finalize_calls"] = 1
    f = tmp_path / "weights.safetensors"
    f.write_bytes(b"\x01" * 16)
    publish_flavors(_Ctx(fake_hub), [ProducedFlavor(path=f, flavor="fp32")])
    assert _FakeHub.state["auth"] == "Bearer cap-token"


def test_publish_flavors_requires_destination(fake_hub, tmp_path: Path) -> None:
    ctx = _Ctx(fake_hub)
    ctx.destination = {}
    f = tmp_path / "w.safetensors"
    f.write_bytes(b"\x01")
    with pytest.raises(ValueError, match="destination_repo"):
        publish_flavors(ctx, [ProducedFlavor(path=f)])


# ---------------------------------------------------------------------------
# run_clone: junk filtering, workdir lifecycle, empty publish
# ---------------------------------------------------------------------------


def _fake_source(dest_dir: Path) -> IngestedSource:
    return IngestedSource(
        provider="huggingface",
        source_ref="org/tiny",
        source_revision="sha-1",
        dir=dest_dir,
        layout="diffusers",
        model_family="",
        model_family_variant="",
        classification=None,
        attrs={"dtype": "bf16"},
        metadata={"source_provider": "huggingface"},
        repo_spec={"kind": "model", "library_name": "diffusers"},
    )


def _install_fake_ingest(monkeypatch, *, fail_first: bool = False) -> dict:
    calls = {"n": 0}

    def fake_ingest(source_ref, dest_dir, **kwargs):
        calls["n"] += 1
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "partial.bin").write_bytes(b"partial")
        if fail_first and calls["n"] == 1:
            raise RuntimeError("network died mid-download")
        (dest_dir / "config.json").write_text("{}")
        junk = dest_dir / ".cache" / "huggingface"
        junk.mkdir(parents=True, exist_ok=True)
        (junk / "config.json.metadata").write_text("etag")
        (junk / ".gitignore").write_text("*")
        return _fake_source(dest_dir)

    monkeypatch.setattr("cozy_convert.clone.ingest_huggingface", fake_ingest)
    return calls


def test_run_clone_publishes_clean_tree_and_removes_workdir(
    fake_hub, tmp_path: Path, monkeypatch
) -> None:
    _FakeHub.state["finalize_calls"] = 1
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    _install_fake_ingest(monkeypatch)

    result = run_clone(
        _Ctx(fake_hub), provider="huggingface", source_ref="org/tiny",
        destination_repo="acme/dest",
    )
    assert len(result.published) == 1
    ops = [op["path"] for op in _FakeHub.state["commit_request"]["operations"]]
    assert "config.json" in ops
    assert not any(o.startswith(".cache/") for o in ops)
    # success: keyed workdir is gone
    assert list((tmp_path / "work").glob("clone-*")) == []


def test_run_clone_failure_retains_workdir_for_resume(
    fake_hub, tmp_path: Path, monkeypatch
) -> None:
    _FakeHub.state["finalize_calls"] = 1
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    calls = _install_fake_ingest(monkeypatch, fail_first=True)

    with pytest.raises(RuntimeError, match="network died"):
        run_clone(
            _Ctx(fake_hub), provider="huggingface", source_ref="org/tiny",
            destination_repo="acme/dest",
        )
    kept = list((tmp_path / "work").glob("clone-*"))
    assert len(kept) == 1
    assert (kept[0] / "source" / "partial.bin").exists()

    # retry lands in the SAME keyed workdir and succeeds; workdir removed
    result = run_clone(
        _Ctx(fake_hub), provider="huggingface", source_ref="org/tiny",
        destination_repo="acme/dest",
    )
    assert calls["n"] == 2
    assert len(result.published) == 1
    assert list((tmp_path / "work").glob("clone-*")) == []


def test_run_clone_publishing_nothing_is_an_error(
    fake_hub, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))

    def empty_ingest(source_ref, dest_dir, **kwargs):
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        return _fake_source(Path(dest_dir))

    monkeypatch.setattr("cozy_convert.clone.ingest_huggingface", empty_ingest)
    with pytest.raises(RuntimeError, match="no publishable flavor"):
        run_clone(
            _Ctx(fake_hub), provider="huggingface", source_ref="org/tiny",
            destination_repo="acme/dest",
        )
