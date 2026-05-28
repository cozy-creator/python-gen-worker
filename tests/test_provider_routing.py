"""Provider routing + binding resolution — collapsed integration suite.

Replaces the ~97-test binding/routing permutation cluster (test_api_binding,
test_binding_api, test_provider_index_routing, test_override_provider_routing,
test_model_selectable_endpoints) with a handful of parametrized cases that
each exercise the REAL routing path:

  * build_provider_index_from_manifest over a real endpoint.lock-shaped dict,
  * lookup_provider_for_ref against that real index (contextvar + global),
  * ModelRefDownloader.download_with_progress routing through the right
    provider branch against a REAL on-disk snapshot (civitai/hf leaf stubbed
    only at the network edge — every routing decision is real),
  * the safetensors-only override gate against real tmp files,
  * Worker._resolve_model_id_for_injection cross-provider key shape.

Named regressions kept as explicit cases:
  * tag-stripping (live 2026-05-16 failure: ``:latest`` stamped HF ref),
  * contextvar no-leak across requests sharing a thread,
  * HF #flavor suffix stripped before producing the huggingface_hub repo_id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import msgspec
import pytest

from gen_worker import HFRepo, Repo
from gen_worker._worker_support import build_provider_index_from_manifest
from gen_worker.models.hf_downloader import HuggingFaceDownloadResult, HuggingFaceRef
from gen_worker.models.ref_downloader import (
    ModelRefDownloader,
    lookup_provider_for_ref,
    reset_override_ref_keys,
    reset_provider_by_ref,
    set_override_ref_keys,
    set_provider_by_ref,
    set_provider_by_ref_global,
)
from gen_worker.models.unsafe_format import UnsafeFileFormat, assert_safe_weight_format
from gen_worker.worker import InjectionSpec, Worker, _resolved_repo_id


# --------------------------------------------------------------------------- #
# Test doubles that touch ONLY the network leaf — every routing decision the
# downloader makes (provider lookup, branch selection, gate) is real.
# --------------------------------------------------------------------------- #


class _RecordingHF:
    """HuggingFaceHubDownloader stand-in: records the parsed ref it was
    handed (so we can assert routing landed on the HF branch with the right
    repo_id) and returns ``snapshot_dir`` as the materialized snapshot the
    safetensors gate then inspects on real disk.
    """

    def __init__(self, snapshot_dir: Path) -> None:
        self.snapshot_dir = snapshot_dir
        self.calls: List[HuggingFaceRef] = []

    def download(self, ref: HuggingFaceRef, progress_callback: Any = None) -> HuggingFaceDownloadResult:
        self.calls.append(ref)
        return HuggingFaceDownloadResult(local_dir=self.snapshot_dir)


def _make_weight_files(root: Path, files: List[str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for f in files:
        p = root / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 16)
    return root


# --------------------------------------------------------------------------- #
# build_provider_index_from_manifest — real endpoint.lock shapes
# --------------------------------------------------------------------------- #


def _manifest(*binding_blocks: dict) -> dict:
    return {
        "functions": [
            {"name": f"fn{i}", "bindings": b} for i, b in enumerate(binding_blocks)
        ]
    }


@pytest.mark.parametrize(
    "block,expect",
    [
        # Fixed HF binding: bare ref + #flavor both index to hf.
        (
            {"pipeline": {"kind": "fixed", "provider": "hf",
                          "ref": "bfl/FLUX.2-klein-4B", "flavor": "bf16", "tag": "prod"}},
            {"bfl/FLUX.2-klein-4B#bf16": "hf", "bfl/FLUX.2-klein-4B": "hf"},
        ),
        # Fixed tensorhub binding: canonical owner/repo:tag#flavor + tagless form.
        (
            {"pipeline": {"kind": "fixed", "provider": "tensorhub",
                          "ref": "acme/flux", "flavor": "fp8", "tag": "prod"}},
            {"acme/flux:prod#fp8": "tensorhub", "acme/flux#fp8": "tensorhub"},
        ),
        # Dispatch table mixing providers across variants — each indexed.
        (
            {"pipeline": {"kind": "dispatch", "field": "variant", "table": {
                "bf16": {"provider": "hf", "ref": "owner/flux", "flavor": "bf16", "tag": "prod"},
                "fp8": {"provider": "tensorhub", "ref": "owner/flux", "flavor": "fp8", "tag": "prod"},
            }}},
            {"owner/flux#bf16": "hf", "owner/flux:prod#fp8": "tensorhub"},
        ),
    ],
)
def test_provider_index_extracted_from_manifest(block: dict, expect: dict) -> None:
    index = build_provider_index_from_manifest(_manifest(block))
    for ref, provider in expect.items():
        assert index.get(ref) == provider, (ref, index)

    # Robust to empty / malformed input: skipped, not fatal.
    assert build_provider_index_from_manifest(None) == {}
    assert build_provider_index_from_manifest({"functions": []}) == {}
    bad = build_provider_index_from_manifest({
        "functions": [
            {"bindings": {"pipeline": {"kind": "fixed", "provider": "hf"}}},  # no ref
            {"bindings": "not-a-dict"},
            {"bindings": {"pipeline": {"kind": "fixed", "provider": "hf", "ref": "ok/ref"}}},
        ]
    })
    assert bad == {"ok/ref": "hf"}


# --------------------------------------------------------------------------- #
# lookup_provider_for_ref — contextvar, global fallback, tag-stripping
# --------------------------------------------------------------------------- #


def test_lookup_contextvar_default_and_no_leak() -> None:
    # Default when unset (overridable); per-request index resolves; and the
    # regression: the per-request contextvar must NOT leak into the next request.
    assert lookup_provider_for_ref("foo/bar") == "tensorhub"
    assert lookup_provider_for_ref("foo/bar", default="hf") == "hf"

    tok = set_provider_by_ref({"acme/flux#bf16": "hf"})
    try:
        assert lookup_provider_for_ref("acme/flux#bf16") == "hf"
        assert lookup_provider_for_ref("not/in-index") == "tensorhub"
    finally:
        reset_provider_by_ref(tok)
    assert lookup_provider_for_ref("acme/flux#bf16") == "tensorhub"  # no lingering state


@pytest.mark.parametrize(
    "wire_ref",
    [
        "bfl/FLUX.2-klein-4B:latest#bf16",  # canonicalizer stamped :latest
        "bfl/FLUX.2-klein-4B:prod#bf16",    # non-default tag
        "bfl/FLUX.2-klein-4B#bf16",         # bare form (no regression)
    ],
)
def test_lookup_global_fallback_and_tag_strip(wire_ref: str) -> None:
    """gRPC stream threads don't inherit the contextvar — the process-global
    index installed at boot is the fallback. Live 2026-05-16 failure: such a
    thread stamps a ``:tag`` onto an HF ref but the index only carries the bare
    HF form, so the lookup must strip the tag (tag is meaningless for HF)."""
    assert lookup_provider_for_ref(wire_ref) == "tensorhub"  # default before install
    set_provider_by_ref_global({"bfl/FLUX.2-klein-4B#bf16": "hf"})
    try:
        assert lookup_provider_for_ref(wire_ref) == "hf"
    finally:
        set_provider_by_ref_global({})


# --------------------------------------------------------------------------- #
# ModelRefDownloader routing — real branch selection
# --------------------------------------------------------------------------- #


def test_hf_indexed_ref_routes_to_hf_branch(tmp_path: Path) -> None:
    """The #17 fix end-to-end: an HF-provider binding routes to the HF
    downloader instead of raising the tensorhub 'not in resolved_repos_by_id'
    error."""
    snap = _make_weight_files(tmp_path / "snap", ["model.safetensors"])
    dl = ModelRefDownloader()
    hf = _RecordingHF(snap)
    dl._hf = hf

    ref = "bfl/FLUX.2-klein-4B#bf16"
    tok = set_provider_by_ref({ref: "hf"})
    try:
        out = dl.download(ref, str(tmp_path))
    finally:
        reset_provider_by_ref(tok)

    assert len(hf.calls) == 1
    assert hf.calls[0].repo_id == "bfl/FLUX.2-klein-4B"  # #flavor stripped
    assert out == snap.as_posix() or Path(out) == snap


@pytest.mark.parametrize(
    "ref,index",
    [
        ("acme/cozy-only#fp8", {"acme/cozy-only#fp8": "tensorhub"}),  # indexed tensorhub
        ("acme/unindexed", {"other/ref": "hf"}),                     # defaults to tensorhub
        ("acme/no-index", None),                                     # no index at all
    ],
)
def test_tensorhub_routed_refs_raise_without_resolved_repos(
    tmp_path: Path, ref: str, index: Optional[dict]
) -> None:
    """Workers never resolve tensorhub refs themselves — the contract is the
    orchestrator pre-resolves them. Any ref that routes to the tensorhub
    branch without a resolved manifest raises (and the HF branch is NOT
    touched)."""
    dl = ModelRefDownloader()
    hf = _RecordingHF(tmp_path / "snap")
    dl._hf = hf

    tok = set_provider_by_ref(index) if index is not None else None
    try:
        with pytest.raises(RuntimeError, match="not in resolved_repos_by_id"):
            dl.download(ref, str(tmp_path))
    finally:
        if tok is not None:
            reset_provider_by_ref(tok)
    assert hf.calls == []  # tensorhub refs never hit the HF branch


# --------------------------------------------------------------------------- #
# Safetensors-only override gate — real files, fires only for override refs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "files,raises",
    [
        (["model.safetensors", "config.json"], False),
        (["model.flashpack"], False),
        (["pytorch_model.bin", "model.safetensors"], False),  # safetensors sibling ok
        (["config.json", "tokenizer.json"], False),           # no weights -> no-op
        (["pytorch_model.bin"], True),
        (["model.pt"], True),
        (["model.ckpt"], True),
    ],
)
def test_safetensors_gate(tmp_path: Path, files: List[str], raises: bool) -> None:
    snap = _make_weight_files(tmp_path / "snap", files)
    if raises:
        with pytest.raises(UnsafeFileFormat, match="refusing to load"):
            assert_safe_weight_format(snap, ref="x/y")
    else:
        assert_safe_weight_format(snap, ref="x/y")


def test_downloader_gates_only_override_refs(tmp_path: Path) -> None:
    """The runtime safetensors gate fires for invoker-supplied OVERRIDE refs
    (which skipped the build-time validator) but not for binding-default refs.
    Pickle-only snapshot + override marker -> reject; same snapshot without
    the marker -> accept."""
    snap = _make_weight_files(tmp_path / "snap", ["pytorch_model.bin"])
    dl = ModelRefDownloader()
    dl._hf = _RecordingHF(snap)
    ref = "bad-org/pickle-repo"

    # Override marker set -> gate fires.
    prov = set_provider_by_ref({ref: "hf"})
    keys = set_override_ref_keys([ref])
    try:
        with pytest.raises(UnsafeFileFormat, match="refusing to load"):
            dl.download(ref, str(tmp_path))
    finally:
        reset_override_ref_keys(keys)
        reset_provider_by_ref(prov)

    # Same pickle-only snapshot, but NOT an override -> gate stays quiet.
    prov = set_provider_by_ref({ref: "hf"})
    try:
        out = dl.download(ref, str(tmp_path))
    finally:
        reset_provider_by_ref(prov)
    assert Path(out) == snap


# --------------------------------------------------------------------------- #
# Cross-provider override resolution — real Worker key shaping
# --------------------------------------------------------------------------- #


class _Pipe:
    pass


class _P(msgspec.Struct):
    prompt: str = ""


def _bare_worker() -> Worker:
    w = Worker.__new__(Worker)
    w._release_allowed_model_ids = None
    return w


@pytest.mark.parametrize(
    "binding,override_provider,expect_hf_prefix",
    [
        # HF binding + HF override -> HF-keyed model_id (pre-#18 bug: defaulted tensorhub).
        (HFRepo("orig/hf-flux").dtype("bf16").allow_override(_Pipe), "hf", True),
        # HF binding + tensorhub override -> tensorhub key (no hf:: prefix).
        (HFRepo("orig/hf-flux").dtype("bf16").allow_override(_Pipe), "tensorhub", False),
        # tensorhub binding + HF override -> HF key (reverse cross-provider).
        (Repo("acme/flux").flavor("bf16").allow_override(_Pipe), "hf", True),
        # tensorhub binding + no provider field -> tensorhub default (back-compat).
        (Repo("acme/flux").flavor("bf16").allow_override(_Pipe), None, False),
    ],
)
def test_cross_provider_override_key_shape(
    binding: Repo, override_provider: Optional[str], expect_hf_prefix: bool
) -> None:
    w = _bare_worker()
    inj = InjectionSpec(param_name="pipeline", param_type=_Pipe, binding=binding)
    entry = {"ref": "other/model", "tag": "prod", "flavor": "bf16"}
    if override_provider is not None:
        entry["provider"] = override_provider

    model_id, _ = w._resolve_model_id_for_injection(
        "fn", inj, payload=_P(), resolved_models={"pipeline": entry},
    )
    assert model_id.startswith("hf::") is expect_hf_prefix
    assert "other/model" in model_id


def test_override_rejected_for_non_overridable_binding_and_default_path() -> None:
    """Defense-in-depth: orchestrator drift stamping an override for a
    non-overridable binding errors loudly; the binding-default path resolves to
    the declared ref/flavor."""
    w = _bare_worker()
    inj = InjectionSpec(
        param_name="pipeline", param_type=_Pipe,
        binding=Repo("acme/flux").flavor("bf16"),  # no allow_override
    )
    with pytest.raises(ValueError, match="no allow_override"):
        w._resolve_model_id_for_injection(
            "fn", inj, payload=_P(),
            resolved_models={"pipeline": {"ref": "acme/other", "tag": "prod", "flavor": ""}},
        )
    model_id, _ = w._resolve_model_id_for_injection("fn", inj, payload=_P(), resolved_models={})
    assert model_id == _resolved_repo_id("acme/flux", flavor="bf16", tag="prod")
