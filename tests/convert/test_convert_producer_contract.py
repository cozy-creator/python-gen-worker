"""Convert/publish producer lane (th#960/pgw#609 Phase 2b): a consolidated
kept file — the coordinator's one authorized "no P-test home" lane for the
clone/publish producer contract (separate from P8's dtype/classifier
contract, which stays in test_p8_convert_publish_contract.py).

Absorbed from (all deleted after this file lands): test_clone_concurrency.py
(gw#442, e2e J19 double-clone), test_clone_hygiene.py (gw#462, J24 ENOSPC),
test_download_skip.py (th#592), test_publish_resilience.py (gw#462, J24
lost-staged-object). Their other tests (disk-budget arithmetic variants,
GGUF intermediate-peak sizing, sweep/lock edge cases) have no distinct
incident pin beyond what's kept here and are git-history-archived.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from fake_hub import _FakeHub, _client

# ---------------------------------------------------------------------------
# gw#442 (e2e J19): concurrent duplicate clones must serialize on the keyed
# workdir — a crash-recovery re-queue put two clones of the same source on
# one worker; unserialized, hf_hub's local-dir download unlinked files a
# peer clone was mid-read on.
# ---------------------------------------------------------------------------


def test_concurrent_same_source_clones_serialize(fake_hub, tmp_path: Path, monkeypatch) -> None:
    from gen_worker.convert.clone import CloneResult, run_clone
    from gen_worker.convert.ingest import IngestedSource

    class _Ctx:
        def __init__(self, server) -> None:
            self._file_api_base_url = f"http://127.0.0.1:{server.server_port}"
            self._worker_capability_token = "cap-token"
            self.owner = "acme"
            self.request_id = "req-1"
            self.destination = {"repo": "acme/fallback"}

    def _fake_source(dest_dir: Path) -> IngestedSource:
        return IngestedSource(
            provider="huggingface", source_ref="org/tiny", source_revision="sha-1",
            dir=dest_dir, layout="diffusers", model_family="", model_family_variant="",
            classification=None, attrs={"dtype": "bf16"},
            metadata={"source_provider": "huggingface"},
            repo_spec={"kind": "model", "library_name": "diffusers"},
        )

    _FakeHub.state["finalize_calls"] = 1
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setattr(
        "gen_worker.convert.clone.plan_huggingface",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    guard = threading.Lock()
    state = {"active": 0, "max_active": 0}

    def fake_ingest(source_ref, dest_dir, **kwargs):
        with guard:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.5)  # hold the window open: an unserialized peer would overlap
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "config.json").write_text("{}")
        with guard:
            state["active"] -= 1
        return _fake_source(dest_dir)

    monkeypatch.setattr("gen_worker.convert.clone.ingest_huggingface", fake_ingest)

    results: dict = {}

    def _clone(i: int) -> None:
        try:
            results[i] = run_clone(
                _Ctx(fake_hub), provider="huggingface", source_ref="org/tiny",
                destination_repo="acme/dest",
            )
        except BaseException as exc:  # noqa: BLE001
            results[i] = exc

    threads = [threading.Thread(target=_clone, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    for i in range(2):
        assert isinstance(results.get(i), CloneResult), f"clone {i}: {results.get(i)!r}"
    assert state["max_active"] == 1, "concurrent clones must never share the workdir"


# ---------------------------------------------------------------------------
# gw#462 (J24 qwen postmortem): a 20GB conversion pod ENOSPC-died mid-
# download with no preflight — disk preflight now refuses loud before ever
# starting a download it cannot finish.
# ---------------------------------------------------------------------------


class _DiskPlan:
    def __init__(self, sizes: list) -> None:
        self._sizes = sizes
        self.classification = SimpleNamespace(strategy="", attrs={})
        self.provider = ""
        self.paths = [f"f{i}.safetensors" for i in range(len(sizes))]

    def bank_files(self):
        return [(p, s, f"cid{i}") for i, (p, s) in enumerate(zip(self.paths, self._sizes))]


def test_disk_preflight_refuses_oversized_source_with_actionable_message(tmp_path: Path) -> None:
    from gen_worker.convert.clone import CloneDiskSpaceError, _preflight_disk, normalize_outputs

    specs = normalize_outputs([{"dtype": "bf16", "file_layout": "diffusers", "file_type": "safetensors"}])
    with pytest.raises(CloneDiskSpaceError, match=r"need ~.* GiB free .*have .* GiB"):
        _preflight_disk(tmp_path, _DiskPlan([10 * 1024**5]), specs)  # 10 PiB: no real fs fits


# ---------------------------------------------------------------------------
# th#592: download-skip bank keys are deterministic + input-sensitive (a
# wrong-but-stable key would either over-skip real re-downloads onto stale
# content, or never hit at all).
# ---------------------------------------------------------------------------


class _BankPlan:
    def __init__(self, files, extra=None) -> None:
        self._files = files
        self._extra = extra or {"strategy": "diffusers", "attrs": "{}"}
        self.provider = "huggingface"
        self.source_ref = "acme/src"
        self.revision = "deadbeef"

    def bank_files(self):
        return sorted(self._files)

    def bank_extra(self):
        return dict(self._extra)


def test_download_skip_bank_key_deterministic_and_input_sensitive() -> None:
    from gen_worker.convert.bank import BANK_KEY_PREFIX, flavor_bank_key
    from gen_worker.convert.clone import OutputSpec

    spec = OutputSpec(dtype="bf16", file_layout="diffusers", file_type="safetensors")
    files = [("model.safetensors", 100, "sha256:" + "a" * 64), ("config.json", 10, "git:abc123")]
    k1 = flavor_bank_key(_BankPlan(files), spec.label, layout_hint="diffusers")
    k2 = flavor_bank_key(_BankPlan(list(reversed(files))), spec.label, layout_hint="diffusers")
    assert k1 == k2 and k1.startswith(BANK_KEY_PREFIX), "key must be order-independent"

    changed = [("model.safetensors", 100, "sha256:" + "b" * 64), ("config.json", 10, "git:abc123")]
    assert flavor_bank_key(_BankPlan(changed), spec.label, layout_hint="diffusers") != k1
    assert flavor_bank_key(_BankPlan(files), spec.label, layout_hint="singlefile") != k1


# ---------------------------------------------------------------------------
# gw#462 (J24 qwen postmortem): a lost staged object during commit must cost
# exactly ONE file's re-upload, never the whole job.
# ---------------------------------------------------------------------------


def test_publish_lost_staged_object_reuploads_only_that_file(fake_hub, tmp_path: Path, monkeypatch) -> None:
    from gen_worker.convert.hub import files_from_tree

    monkeypatch.setattr("time.sleep", lambda *_: None)
    _FakeHub.state["staging_missing"] = {"shard-00004.safetensors": 1}

    (tmp_path / "config.json").write_text('{"a":1}')
    (tmp_path / "shard-00004.safetensors").write_bytes(b"\x04" * 96)

    result = _client(fake_hub).commit(
        destination_repo="acme/qwen-image", files=files_from_tree(tmp_path),
    )
    assert result.uploaded == 2
    st = _FakeHub.state
    assert st["reopens"] == ["shard-00004.safetensors"], "only the poisoned file re-opens"
    assert sum(st["put_counts"].values()) == 3  # 2 files + 1 re-upload of the poisoned one
