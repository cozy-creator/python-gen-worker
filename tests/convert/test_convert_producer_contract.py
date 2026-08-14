"""Convert/publish producer lane (th#960/pgw#609 Phase 2b): a consolidated
kept file — the coordinator's one authorized "no P-test home" lane for the
clone/publish producer contract (separate from P8's dtype/classifier
contract, which stays in test_p8_convert_publish_contract.py).

Absorbed from (all deleted after this file lands): test_clone_concurrency.py
(gw#442, e2e J19 double-clone), test_clone_hygiene.py (gw#462, J24 ENOSPC),
test_download_skip.py, test_publish_resilience.py (gw#462, J24
lost-staged-object). Their other tests (disk-budget arithmetic variants,
GGUF intermediate-peak sizing, sweep/lock edge cases) have no distinct
incident pin beyond what's kept here and are git-history-archived.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path


from fake_hub import _FakeHub

# ---------------------------------------------------------------------------
# concurrent duplicate clones must serialize on the keyed
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
            dir=dest_dir, layout="multi-file", model_family="", model_family_variant="",
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
