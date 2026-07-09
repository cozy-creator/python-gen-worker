"""th#592 provider-hash download-skip: bank keys, skip path, fail-open.

Real HTTP against the fake tensorhub (fake_hub.py) — the same code path a
production clone takes, minus the provider network (plans/ingest are
substituted with local fixtures; the *twice-clone* test runs the full
run_clone pipeline both times and proves the second pass downloads nothing).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from gen_worker.convert import clone as clone_mod
from gen_worker.convert.bank import BANK_KEY_PREFIX, build_bank_payload, flavor_bank_key
from gen_worker.convert.clone import OutputSpec, run_clone
from gen_worker.convert.hub import blake3_file

from fake_hub import _FakeHub, _client


class _Plan:
    """Minimal SourcePlan test double."""

    def __init__(self, files, extra=None, provider="huggingface",
                 source_ref="acme/src", revision="deadbeef"):
        self._files = files
        self._extra = extra or {"strategy": "diffusers", "attrs": "{}"}
        self.provider = provider
        self.source_ref = source_ref
        self.revision = revision

    def bank_files(self):
        return sorted(self._files)

    def bank_extra(self):
        return dict(self._extra)


_SPEC = OutputSpec(dtype="bf16", file_layout="diffusers", file_type="safetensors")


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------

def test_bank_key_deterministic_and_input_sensitive() -> None:
    files = [("model.safetensors", 100, "sha256:" + "a" * 64),
             ("config.json", 10, "git:abc123")]
    k1 = flavor_bank_key(_Plan(files), _SPEC.label, layout_hint="diffusers")
    k2 = flavor_bank_key(_Plan(list(reversed(files))), _SPEC.label, layout_hint="diffusers")
    assert k1 == k2 and k1.startswith(BANK_KEY_PREFIX)

    # Any input change -> a different key.
    changed_content = [("model.safetensors", 100, "sha256:" + "b" * 64),
                       ("config.json", 10, "git:abc123")]
    assert flavor_bank_key(_Plan(changed_content), _SPEC.label, layout_hint="diffusers") != k1
    assert flavor_bank_key(_Plan(files), "fp8-diffusers-safetensors", layout_hint="diffusers") != k1
    assert flavor_bank_key(_Plan(files), _SPEC.label, layout_hint="singlefile") != k1
    assert flavor_bank_key(_Plan(files), _SPEC.label, layout_hint="diffusers",
                           quantize_components=["transformer"]) != k1
    assert flavor_bank_key(_Plan(files, extra={"strategy": "peft", "attrs": "{}"}),
                           _SPEC.label, layout_hint="diffusers") != k1


def test_bank_key_empty_when_content_ids_missing() -> None:
    assert flavor_bank_key(_Plan([]), _SPEC.label) == ""


# ---------------------------------------------------------------------------
# _publish_from_bank against the fake hub
# ---------------------------------------------------------------------------

def _seed_banked_manifest(key: str, *, blobs_in_cas: bool = True) -> dict:
    payload = build_bank_payload(
        files=[
            {"path": "model_index.json", "blake3": "1" * 64, "size_bytes": 10},
            {"path": "unet/diffusion_pytorch_model.safetensors",
             "blake3": "2" * 64, "size_bytes": 100},
        ],
        flavor="bf16", dtype="bf16", file_layout="diffusers", file_type="safetensors",
        metadata={"source_provider": "huggingface", "source_repo": "acme/src"},
        repo_spec={"kind": "model", "library_name": "diffusers"},
        source_revision="deadbeef",
    )
    st = _FakeHub.state
    st.setdefault("bank_manifests", {})[key] = payload
    if blobs_in_cas:
        st.setdefault("cas_blobs", set()).update({"1" * 64, "2" * 64})
    return payload


def _bank_args(plan):
    return dict(
        plan=plan, provider="huggingface", specs=[_SPEC],
        bank_keys={_SPEC.label: flavor_bank_key(plan, _SPEC.label, layout_hint="diffusers")},
        destination="acme/dst", tags=["prod"], mode="replace", progress=None,
    )


def test_publish_from_bank_hit_commits_by_reference(fake_hub) -> None:
    plan = _Plan([("model.safetensors", 100, "sha256:" + "a" * 64)])
    key = flavor_bank_key(plan, _SPEC.label, layout_hint="diffusers")
    _seed_banked_manifest(key)

    result = clone_mod._publish_from_bank(_client(fake_hub), **_bank_args(plan))
    assert result is not None
    assert result.published[0]["banked"] is True
    assert result.published[0]["uploaded"] == 0
    assert result.published[0]["deduped"] == 2
    assert result.metadata["download_skip"] == "bank"
    assert result.metadata["source_bytes_downloaded"] == "0"
    assert result.metadata["source_bytes_avoided"] == "100"

    st = _FakeHub.state
    # ZERO bytes moved: no part PUTs, no completes.
    assert "put_bytes" not in st and "completed" not in st
    req = st["commit_request"]
    ops = {op["path"]: op for op in req["operations"]}
    assert set(ops) == {"model_index.json", "unet/diffusion_pytorch_model.safetensors"}
    assert ops["model_index.json"]["blake3"] == "1" * 64
    assert req["mode"] == "replace" and req["flavor"] == "bf16"
    assert req["provenance"] == {"upstream_revision": "deadbeef"}
    assert req["metadata"]["download_skip"] == "bank"


def test_publish_from_bank_miss_returns_none(fake_hub) -> None:
    plan = _Plan([("model.safetensors", 100, "sha256:" + "a" * 64)])
    result = clone_mod._publish_from_bank(_client(fake_hub), **_bank_args(plan))
    assert result is None
    assert "commit_request" not in _FakeHub.state


def test_publish_from_bank_not_ready_when_blob_gced(fake_hub) -> None:
    plan = _Plan([("model.safetensors", 100, "sha256:" + "a" * 64)])
    key = flavor_bank_key(plan, _SPEC.label, layout_hint="diffusers")
    _seed_banked_manifest(key, blobs_in_cas=False)  # found but not ready
    result = clone_mod._publish_from_bank(_client(fake_hub), **_bank_args(plan))
    assert result is None
    assert "commit_request" not in _FakeHub.state


def test_publish_from_bank_lookup_error_fails_open(fake_hub, monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)
    plan = _Plan([("model.safetensors", 100, "sha256:" + "a" * 64)])
    key = flavor_bank_key(plan, _SPEC.label, layout_hint="diffusers")
    _seed_banked_manifest(key)
    _FakeHub.state["fail_bank_lookups"] = 99  # every attempt 503s
    result = clone_mod._publish_from_bank(_client(fake_hub), **_bank_args(plan))
    assert result is None  # fail-open: caller downloads as today


def test_publish_from_bank_blob_gone_at_commit_falls_back(fake_hub, monkeypatch) -> None:
    """Lookup says ready but the blob vanishes before the commit (GC race):
    the by-reference commit has no bytes to upload -> BankedBlobGone -> the
    revision is aborted and the bank path reports a miss."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    plan = _Plan([("model.safetensors", 100, "sha256:" + "a" * 64)])
    key = flavor_bank_key(plan, _SPEC.label, layout_hint="diffusers")
    _seed_banked_manifest(key)
    _FakeHub.state["commit_pretend_missing"] = {"2" * 64}

    result = clone_mod._publish_from_bank(_client(fake_hub), **_bank_args(plan))
    assert result is None


# ---------------------------------------------------------------------------
# run_clone twice: first pass records, second pass downloads NOTHING
# ---------------------------------------------------------------------------

class _Ctx:
    def __init__(self, server):
        self._file_api_base_url = f"http://127.0.0.1:{server.server_port}"
        self._worker_capability_token = "cap-token"
        self.owner = "acme"
        self.stages: list[str] = []

    def progress(self, p: float, stage: str = "") -> None:
        self.stages.append(stage)


def _fake_source_tree(tmp_path: Path) -> Path:
    src = tmp_path / "fixture-src"
    (src / "unet").mkdir(parents=True)
    (src / "model_index.json").write_text('{"_class_name":"X"}')
    (src / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"\x11" * 256)
    (src / "unet" / "config.json").write_text('{"c":1}')
    return src


def _install_fake_provider(monkeypatch, tmp_path: Path, calls: dict) -> None:
    """Substitute the provider network: plan_huggingface returns a fixed
    identity; ingest_huggingface 'downloads' by copying a local fixture.
    Everything downstream (flavor build, hashing, commit HTTP) is real."""
    import shutil as _shutil

    from gen_worker.convert.ingest import IngestedSource

    src = _fake_source_tree(tmp_path)
    files = sorted(p.relative_to(src).as_posix() for p in src.rglob("*") if p.is_file())
    plan = _Plan(
        [(p, (src / p).stat().st_size, "sha256:" + blake3_file(src / p))  # any stable id
         for p in files],
        extra={"strategy": "diffusers", "attrs": '{"dtype":"bf16"}'},
    )

    def fake_plan(*a, **k):
        calls["plan"] = calls.get("plan", 0) + 1
        return plan

    def fake_ingest(source_ref, dest_dir, **kwargs):
        calls["ingest"] = calls.get("ingest", 0) + 1
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        _shutil.copytree(src, dest_dir, dirs_exist_ok=True)
        progress = kwargs.get("progress")
        total = sum((src / p).stat().st_size for p in files)
        if progress is not None:
            progress(total, total)  # "downloaded" bytes
        return IngestedSource(
            provider="huggingface", source_ref="acme/src", source_revision="deadbeef",
            dir=dest_dir, layout="diffusers", model_family="sdxl",
            model_family_variant="", classification=None,
            attrs={"dtype": "bf16", "file_layout": "diffusers", "file_type": "safetensors"},
            metadata={"source_provider": "huggingface", "source_repo": "acme/src",
                      "source_revision": "deadbeef"},
            repo_spec={"kind": "model", "library_name": "diffusers"},
        )

    monkeypatch.setattr(clone_mod, "plan_huggingface", fake_plan)
    monkeypatch.setattr(clone_mod, "ingest_huggingface", fake_ingest)


def test_run_clone_twice_second_pass_skips_download(fake_hub, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    calls: dict = {}
    _install_fake_provider(monkeypatch, tmp_path, calls)
    ctx = _Ctx(fake_hub)

    # First clone: full path (download + convert + upload) records the bank
    # manifest after publishing.
    r1 = run_clone(ctx, provider="huggingface", source_ref="acme/src",
                   destination_repo="acme/dst", destination_repo_tags=["prod"])
    assert calls == {"plan": 1, "ingest": 1}
    assert r1.metadata.get("download_skip") is None
    assert int(r1.metadata["source_bytes_downloaded"]) > 0
    st = _FakeHub.state
    assert len(st["bank_records"]) == 1
    assert st["bank_records"][0]["manifests"][0]["key"].startswith(BANK_KEY_PREFIX)
    ops1 = {op["path"]: op["blake3"] for op in st["commit_requests"][0]["operations"]}

    # Second clone of the identical source: bank hit -> ingest NEVER runs,
    # zero provider bytes downloaded, byte-identical commit operations
    # (content-addressed checkpoint id is a function of the ops set).
    put_count_after_first = len(st.get("put_bytes", {}))
    r2 = run_clone(ctx, provider="huggingface", source_ref="acme/src",
                   destination_repo="acme/dst", destination_repo_tags=["prod"])
    assert calls == {"plan": 2, "ingest": 1}, "second pass must not ingest/download"
    assert r2.metadata["download_skip"] == "bank"
    assert r2.metadata["source_bytes_downloaded"] == "0"
    assert len(st.get("put_bytes", {})) == put_count_after_first, "0 content bytes uploaded"
    ops2 = {op["path"]: op["blake3"] for op in st["commit_requests"][-1]["operations"]}
    assert ops2 == ops1, "banked commit must be byte-identical to the original"
    assert r2.published[0]["banked"] is True
    assert r2.published[0]["flavor"] == r1.published[0]["flavor"] == "bf16"


def test_run_clone_bank_unavailable_is_fail_open(fake_hub, tmp_path, monkeypatch) -> None:
    """Bank lookup hard-down must not block the clone."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    calls: dict = {}
    _install_fake_provider(monkeypatch, tmp_path, calls)
    _FakeHub.state["fail_bank_lookups"] = 99
    r = run_clone(_Ctx(fake_hub), provider="huggingface", source_ref="acme/src",
                  destination_repo="acme/dst")
    assert calls["ingest"] == 1
    assert r.published and r.metadata.get("download_skip") is None


def test_run_clone_plan_failure_is_fail_open(fake_hub, tmp_path, monkeypatch) -> None:
    """Provider metadata plan blowing up must not block the clone either."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    calls: dict = {}
    _install_fake_provider(monkeypatch, tmp_path, calls)

    def boom(*a, **k):
        raise RuntimeError("hf api down")

    monkeypatch.setattr(clone_mod, "plan_huggingface", boom)
    r = run_clone(_Ctx(fake_hub), provider="huggingface", source_ref="acme/src",
                  destination_repo="acme/dst")
    assert calls["ingest"] == 1
    assert r.published
    # No key -> nothing recorded either.
    assert "bank_records" not in _FakeHub.state


def test_hf_repackaged_flavor_never_banked(fake_hub, tmp_path, monkeypatch) -> None:
    """HF flavors that repackaged depend on post-download family detection —
    they must not be recorded under a pre-download key."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    calls: dict = {}
    _install_fake_provider(monkeypatch, tmp_path, calls)

    real_build = clone_mod.build_flavor_tree

    def fake_build(source, spec, out_dir, **kwargs):
        tree, attrs = real_build(source, spec, out_dir, **kwargs)
        attrs["repackage_toolchain"] = "singlefile_to_diffusers:v1"
        return tree, attrs

    monkeypatch.setattr(clone_mod, "build_flavor_tree", fake_build)
    r = run_clone(_Ctx(fake_hub), provider="huggingface", source_ref="acme/src",
                  destination_repo="acme/dst")
    assert r.published
    assert "bank_records" not in _FakeHub.state
