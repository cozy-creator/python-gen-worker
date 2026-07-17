"""publish_flavors + run_clone against the fake /commits server.

Covers the producer publish contract (#375) and clone robustness (#374):
explicit publish, HF-cache junk never uploaded, keyed workdir retained on
failure / removed on success, and no-publish-cannot-read-as-success.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker.convert import ProducedFlavor, publish_flavors
from gen_worker.convert.clone import run_clone
from gen_worker.convert.ingest import IngestedSource

from fake_hub import _FakeHub


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


def test_publish_flavors_defaults_to_replace(fake_hub, tmp_path: Path) -> None:
    """th#597 C2 regression (te#44): a producer's flavor export is a complete
    replacement tree — publish_flavors must default to mode=replace so an
    #fp8 export can never merge with (and silently carry) the repo's prior
    fp16 base tree. mode=merge remains available only as an explicit opt-in
    for deliberate overlay publishes."""
    _FakeHub.state["finalize_calls"] = 1
    f = tmp_path / "weights.safetensors"
    f.write_bytes(b"\x02" * 16)
    publish_flavors(_Ctx(fake_hub), [ProducedFlavor(path=f, flavor="fp8")], destination_repo="acme/dest")
    assert _FakeHub.state["commit_requests"][-1]["mode"] == "replace"

    publish_flavors(
        _Ctx(fake_hub),
        [ProducedFlavor(path=f, flavor="vae-fix")],
        destination_repo="acme/dest",
        mode="merge",
    )
    assert _FakeHub.state["commit_requests"][-1]["mode"] == "merge"


def test_publish_flavors_destination_falls_back_to_ctx(fake_hub, tmp_path: Path) -> None:
    _FakeHub.state["finalize_calls"] = 1
    f = tmp_path / "weights.safetensors"
    f.write_bytes(b"\x01" * 16)
    publish_flavors(_Ctx(fake_hub), [ProducedFlavor(path=f, flavor="fp32")])
    assert _FakeHub.state["auth"] == "Bearer cap-token"
    # The commit actually landed on the ctx.destination repo.
    assert "acme/fallback" in _FakeHub.state["commit_path"]


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

    monkeypatch.setattr("gen_worker.convert.clone.ingest_huggingface", fake_ingest)
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


def test_run_clone_failure_cleans_workdir_then_retry_succeeds(
    fake_hub, tmp_path: Path, monkeypatch
) -> None:
    # gw#462 flipped the old retain-on-failure default: a long-running
    # conversion worker must not leak each failed job's scratch (the disk IS
    # the scarce resource). Cross-run resume lives in the publish bank + CAS
    # dedup, not in retained local bytes.
    _FakeHub.state["finalize_calls"] = 1
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    monkeypatch.delenv("COZY_CONVERT_RETAIN_WORKDIR", raising=False)
    calls = _install_fake_ingest(monkeypatch, fail_first=True)

    with pytest.raises(RuntimeError, match="network died"):
        run_clone(
            _Ctx(fake_hub), provider="huggingface", source_ref="org/tiny",
            destination_repo="acme/dest",
        )
    assert list((tmp_path / "work").glob("clone-*")) == []

    # retry re-creates the keyed workdir and succeeds; workdir removed again
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

    monkeypatch.setattr("gen_worker.convert.clone.ingest_huggingface", empty_ingest)
    with pytest.raises(RuntimeError, match="no publishable flavor"):
        run_clone(
            _Ctx(fake_hub), provider="huggingface", source_ref="org/tiny",
            destination_repo="acme/dest",
        )


def test_publish_flavors_stamps_placement(fake_hub, tmp_path: Path) -> None:
    """th#697 P1: recognized quant flavors carry a structured `placement`
    block in commit metadata (arch requirements the SKU-aware ladder reads
    back); base/unknown flavors stay unstamped; explicit placement_* attrs
    override the token-derived defaults and never leak as flat keys."""
    _FakeHub.state["finalize_calls"] = 1
    f = tmp_path / "weights.safetensors"
    f.write_bytes(b"\x03" * 16)

    publish_flavors(
        _Ctx(fake_hub),
        [
            ProducedFlavor(
                path=f,
                flavor="svdq-int4-r128",
                attributes={"quantization_method": "svdquant",
                            "quantization_library": "nunchaku"},
            ),
            ProducedFlavor(path=f, flavor="fp8"),
            ProducedFlavor(path=f, flavor="bf16"),
            ProducedFlavor(path=f, flavor="vae-fix"),
            ProducedFlavor(
                path=f,
                flavor="fp8",
                attributes={"placement_sm_min": "89", "placement_engines": "transformer_engine"},
            ),
        ],
        destination_repo="acme/dest",
    )
    reqs = _FakeHub.state["commit_requests"][-5:]
    assert reqs[0]["metadata"]["placement"] == {
        "precision_class": "svdq-int4",
        "sm_allowed": [75, 80, 86, 89],
        "engines": ["nunchaku"],
    }
    assert reqs[1]["metadata"]["placement"] == {"precision_class": "fp8"}
    assert "placement" not in (reqs[2].get("metadata") or {})
    assert "placement" not in (reqs[3].get("metadata") or {})
    assert reqs[4]["metadata"]["placement"] == {
        "precision_class": "fp8",
        "sm_min": 89,
        "engines": ["transformer_engine"],
    }
    assert "placement_sm_min" not in reqs[4]["metadata"]
