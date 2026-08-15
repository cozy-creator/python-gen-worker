"""HARDCUT E5 — the two model entry lanes pgw#1273 did not close.

pgw#1273 refuses a pickle on the HF lane before a byte moves. The chokepoint
audit for E5 asked the obvious next question — *which OTHER lane materializes
tenant-named weights?* — and found two:

* **modelscope**, in BOTH `download.ensure_local` and
  `provision.resolve_local_path`, handed `snapshot_download`'s return straight
  to a loader with no refusal anywhere. ModelScope mirrors HF-shaped repos
  verbatim, `.bin` components included.
* **civitai** was safe by construction (its selector allow-lists
  `.safetensors`/`.gguf`) but said `civitai_no_supported_files`, which reads as
  "civitai is broken" rather than "we refuse pickles".

Both are driven through the PRODUCTION dispatch. The only fake is the registry
client itself — a real `snapshot_download` is a network call, and the defect is
what we do with the tree it returns, not the fetch.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any, Dict

import pytest

from gen_worker.models import download as dl
from gen_worker.models import provision
from gen_worker.models.errors import PickleWeightRefused


def _fake_modelscope(monkeypatch: pytest.MonkeyPatch, tree: Path) -> Dict[str, Any]:
    """Install a `modelscope` module whose snapshot_download returns `tree`."""
    calls: Dict[str, Any] = {}

    def _snapshot_download(**kwargs: Any) -> str:
        calls.update(kwargs)
        return str(tree)

    mod = types.ModuleType("modelscope")
    mod.snapshot_download = _snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modelscope", mod)
    return calls


def _pickle_tree(tmp_path: Path) -> Path:
    root = tmp_path / "ms" / "unet"
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}")
    (root / "diffusion_pytorch_model.bin").write_bytes(b"not really a pickle")
    return tmp_path / "ms"


def _clean_tree(tmp_path: Path) -> Path:
    root = tmp_path / "clean" / "unet"
    root.mkdir(parents=True)
    (root / "diffusion_pytorch_model.safetensors").write_bytes(b"ok")
    return tmp_path / "clean"


def test_the_modelscope_lane_refuses_a_pickle_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_modelscope(monkeypatch, _pickle_tree(tmp_path))
    with pytest.raises(PickleWeightRefused) as excinfo:
        dl.download_modelscope("org/pickled")
    assert "diffusion_pytorch_model.bin" in str(excinfo.value)


def test_a_clean_modelscope_repo_still_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control. A refusal that refuses everything proves nothing."""
    tree = _clean_tree(tmp_path)
    _fake_modelscope(monkeypatch, tree)
    assert dl.download_modelscope("org/clean") == tree


def test_the_refusal_is_reached_through_ensure_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Through the executor's own dispatch, not the helper directly."""
    _fake_modelscope(monkeypatch, _pickle_tree(tmp_path))
    with pytest.raises(PickleWeightRefused):
        asyncio.run(dl.ensure_local("org/pickled", provider="modelscope",
                                    cache_dir=tmp_path / "cas"))


def test_the_refusal_is_reached_through_the_cli_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`provision.resolve_local_path` is the SECOND modelscope call site, and a
    fix in one is exactly the drift that left this lane open in the first
    place. It must refuse TYPED — wrapping it in `ModelResolutionError` would
    make a pickle repo look like a transient fetch failure."""
    _fake_modelscope(monkeypatch, _pickle_tree(tmp_path))
    monkeypatch.setenv("TENSORHUB_CACHE_DIR", str(tmp_path / "cas"))
    with pytest.raises(PickleWeightRefused):
        provision.resolve_local_path(
            ref="org/pickled", provider="modelscope", offline=False,
            emit=lambda _event: None,
        )


def test_a_pickle_only_civitai_version_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """It was already unreachable — the selector allow-lists safetensors/gguf —
    but `civitai_no_supported_files` named the wrong party."""
    payload = {"files": [{"id": 1, "name": "model.ckpt", "primary": True,
                          "downloadUrl": "https://example.invalid/x",
                          "sizeKB": 1, "hashes": {"SHA256": "ab" * 32}}]}
    monkeypatch.setattr(dl, "fetch_civitai_model_version", lambda *a, **k: payload)
    with pytest.raises(PickleWeightRefused) as excinfo:
        dl.download_civitai(1234, Path("/nonexistent/should-not-be-reached"))
    assert "model.ckpt" in str(excinfo.value)
