from __future__ import annotations

import hashlib
import http.server
import json
import threading
from pathlib import Path
from typing import Any, Callable, List, Tuple

import pytest
import torch

import tcg_artifacts
from gen_worker._vendor.torchcg.serve import materialize
from gen_worker.serving import DeployBinding
from gen_worker.serving.mint_store import graph_store

from test_serving_adopt_first import (  # noqa: E402
    ENV,
    LANE,
    OVERRIDES,
    SM,
    STACK,
    fresh_host,
    publish_document,
)

RELEASE = "rel-pgw1573"


@pytest.fixture()
def binding(tmp_path: Path) -> DeployBinding:
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"seed": 7, "scheduler": {"prediction_type": "epsilon"}}))
    return DeployBinding(
        checkpoint_ref="ckpt:tiny@1", checkpoint_dir=root, model="sdxl",
        defaults=dict(OVERRIDES),
    )


@pytest.fixture()
def document(binding: DeployBinding, tmp_path: Path) -> Any:
    """The real derive, run once: two aspect buckets -> two specializations."""
    host = fresh_host(binding, tmp_path / "derive")
    host.setup()
    return publish_document(host)


def opening_loader(calls: List[str]) -> "Callable[[Path, Any, Any], Callable[..., Any]]":
    """The production load, minus only the AOTI handle."""

    def load(path: Path, record: Any, module: Any) -> "Callable[..., Any]":
        assert isinstance(module, torch.nn.Module), (
            "the loader binds constants from the LIVE module (tcg#58)")
        materialize(Path(path), Path(path).parent / f"{Path(path).name}.opened")

        def compiled(sample: torch.Tensor) -> torch.Tensor:
            calls.append(record.graph)
            return sample

        return compiled

    return load


def real_compiler(built: List[str]) -> "Callable[[Path, Any, Path], Path]":
    """The compile seam: a REAL unpacked artifact directory, no inductor."""

    def build(blob: Path, record: Any, destination: Path) -> Path:
        built.append(record.graph)
        return tcg_artifacts.unpacked(
            destination, graph_specialization=record.graph, sm=SM)

    return build


def programs(root: Path) -> "Callable[[str, Path], Path]":
    def fetch(graph: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"exported-program")
        return destination

    return fetch


def ratios(document: Any) -> Tuple[str, str]:
    """(ratio for graph 0, ratio for graph 1) — the fixture's two buckets."""
    out = []
    for record in document.lanes[0].graphs:
        shape = tuple(int(d) for d in record.ingress.inputs[0].shape)
        out.append("1:1" if shape == (16, 16) else "3:4")
    return out[0], out[1]


def mint_the_holes(host: Any, store: Any, tmp_path: Path,
                   built: List[str]) -> Any:
    from gen_worker import compile_posture
    from gen_worker.serving import mint as mint_mod

    return mint_mod.BackgroundMint(
        host=host, store=store, artifacts_dir=tmp_path / "artifacts",
        compiler=real_compiler(built), program_source=programs(tmp_path),
        posture=compile_posture.FLEET, vcpus=4,
    ).run()


def dispatch_counts(host: Any) -> Any:
    from gen_worker.serving.dispatch_counter import DispatchCounter

    return DispatchCounter().install(host)


def assert_served_compiled(counter: Any, *, at_least: int = 1) -> Any:
    """The reading a compiled claim rests on, in the daemon's own terms."""
    counts = counter.take()
    assert counts.displaced_modules == (), (
        f"the compiled dispatcher is not reachable as the module's forward: "
        f"{counts.facts()} — this is what the live sd15 benchmark read on "
        f"12/12 requests of both arms (pgw#1591)")
    assert counts.compiled_graph_calls >= at_least, (
        f"served eager while claiming compiled: {counts.facts()}")
    return counts


def bank_both(store: Any, document: Any, tmp_path: Path) -> None:
    """Put a real envelope at both positions of ``store``."""
    from gen_worker.serving.mint import publish_compiled

    for index, record in enumerate(document.lanes[0].graphs):
        unpacked = tmp_path / "seed" / f"{index}"
        tcg_artifacts.unpacked(
            unpacked, graph_specialization=record.graph, sm=SM)
        publish_compiled(store, record.graph, ENV, unpacked)


def test_cold_boot_serves_EAGER_then_mints_then_serves_COMPILED(
    binding: DeployBinding, document: Any, tmp_path: Path
) -> None:
    """Paul's branch ③, end to end, on one live host."""
    cas = tmp_path / "cas"
    store = graph_store(cas, None, tmp_path / "no-baked")
    calls: List[str] = []
    host = fresh_host(binding, tmp_path)
    host.setup(store=store, document=document, sm=SM,
               loader=opening_loader(calls),
               artifacts_dir=tmp_path / "artifacts", stack=STACK)

    assert len(host.adoption.adopted) == 0
    assert len(host.holes) == 2
    first, second = ratios(document)
    host.dispatch("generate", {"prompt": "x", "aspect_ratio": first},
                  request_id="cold-1")
    assert calls == [], "a cold boot must serve EAGER, not wait for a compile"

    built: List[str] = []
    outcome = mint_the_holes(host, store, tmp_path, built)
    assert outcome.landed == 2 and not outcome.failed, outcome.failed
    assert sorted(built) == sorted(
        record.graph for record in document.lanes[0].graphs)
    assert all(entry.armed for entry in outcome.entries), (
        "a minted graph that did not arm leaves this host eager forever")

    counter = dispatch_counts(host)
    host.dispatch("generate", {"prompt": "x", "aspect_ratio": first},
                  request_id="cold-2")
    host.dispatch("generate", {"prompt": "x", "aspect_ratio": second},
                  request_id="cold-3")
    armed = {record.graph for record in document.lanes[0].graphs}
    assert set(calls) == armed, (
        f"the eager forward is still serving after a landed mint: {calls}")
    assert_served_compiled(counter, at_least=2)

    for record in document.lanes[0].graphs:
        assert store.has_artifact(record.graph, ENV)
        assert store.artifact_skew(record.graph, ENV) is None


def test_warm_local_reboot_arms_from_the_store_and_compiles_NOTHING(
    binding: DeployBinding, document: Any, tmp_path: Path
) -> None:
    """Paul's branch ①."""
    cas = tmp_path / "cas"
    store = graph_store(cas, None, tmp_path / "no-baked")
    bank_both(store, document, tmp_path)

    calls: List[str] = []
    host = fresh_host(binding, tmp_path)
    host.setup(store=store, document=document, sm=SM,
               loader=opening_loader(calls),
               artifacts_dir=tmp_path / "artifacts", stack=STACK)

    assert len(host.adoption.adopted) == 2 and host.holes == ()
    first, _second = ratios(document)
    counter = dispatch_counts(host)
    host.dispatch("generate", {"prompt": "x", "aspect_ratio": first},
                  request_id="warm-1")
    assert calls, "a warm boot served eager over a full store"
    assert_served_compiled(counter)

    from gen_worker.serving.self_mint import NOTHING_TO_MINT, SelfMint

    built: List[str] = []
    box = SelfMint(store=store, artifacts_dir=tmp_path / "artifacts",
                   compiler=real_compiler(built),
                   program_source=programs(tmp_path), vcpus=2)
    assert box.arm(host).state == NOTHING_TO_MINT
    assert built == [], "a warm boot recompiled a graph it already had"


class _Hub(http.server.BaseHTTPRequestHandler):

    answer: dict = {}
    blobs: dict = {}

    def log_message(self, *_args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's spelling
        path = self.path.split("?", 1)[0]
        if path.startswith("/blob/"):
            payload = type(self).blobs.get(path[len("/blob/"):])
            if payload is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if path.endswith("/compiled-graphs"):
            body = json.dumps(type(self).answer).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)


def _serve_hub(answer: dict, blobs: dict) -> Tuple[str, Any]:
    _Hub.answer, _Hub.blobs = answer, blobs
    server = http.server.HTTPServer(("127.0.0.1", 0), _Hub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_port}", server


def _fleet_answer(document: Any, base_url: str, blobs: dict,
                  artifacts: List[Path]) -> dict:
    rows = []
    for record, artifact in zip(document.lanes[0].graphs, artifacts):
        payload = Path(artifact).read_bytes()
        name = f"{record.graph}.tar.gz"
        blobs[name] = payload
        rows.append({
            "graph_hash": record.graph,
            "graph_specialization": record.graph,
            "status": "hit",
            "found": True,
            "module_path": record.target,
            "ingress": record.ingress.as_dict(),
            "content_digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "artifact_path": name,
            "requirements_manifest": {
                "v": 1, "autotuned_on": None, "cuda_floor": None,
                "include_set": [["torch", torch.__version__]],
                "sm_compiled": SM,
            },
            "transport": {
                "snapshot_digest": "sha256:0",
                "files": [{"path": name, "url": f"{base_url}/blob/{name}"}],
            },
        })
    return {
        "object": "release_compiled_graphs", "release_id": RELEASE,
        "env_compile_stack": [[name, value] for name, value in STACK],
        "lane": LANE, "lane_stamped": True, "sm": SM, "empty": False,
        "hits": len(rows), "misses": 0, "graphs": rows,
        "targets": [record.target for record in document.lanes[0].graphs],
        "unobserved_targets": [], "passes": [],
    }


def _remote_store(cas: Path, base_url: str) -> Any:
    from gen_worker.serving.__main__ import HttpReleaseGraphTransport
    from gen_worker.serving.hub_store import HubGraphStore

    upstream = HubGraphStore(
        HttpReleaseGraphTransport(base_url), RELEASE, LANE, SM)
    return graph_store(cas, upstream, cas.parent / "no-baked")


def test_warm_remote_fetches_verifies_arms_serves_and_BANKS_locally(
    binding: DeployBinding, document: Any, tmp_path: Path
) -> None:
    """Paul's branch ②, over a real HTTP hub, into a live dispatcher."""
    artifacts = []
    for index, record in enumerate(document.lanes[0].graphs):
        artifacts.append(tcg_artifacts.build(
            tmp_path / "fleet" / f"{index}.tar.gz",
            graph_specialization=record.graph, sm=SM))
    blobs: dict = {}
    base_url, server = _serve_hub({}, blobs)
    _Hub.answer = _fleet_answer(document, base_url, blobs, artifacts)
    try:
        cas = tmp_path / "cold-cas"
        store = _remote_store(cas, base_url)
        assert not store.local.has_artifact(
            document.lanes[0].graphs[0].graph, ENV), "the local tier is not cold"

        calls: List[str] = []
        host = fresh_host(binding, tmp_path)
        host.setup(store=store, document=document, sm=SM,
                   loader=opening_loader(calls),
                   artifacts_dir=tmp_path / "artifacts", stack=STACK)

        assert len(host.adoption.adopted) == 2, (
            f"the fleet pool answered and nothing armed; holes: "
            f"{[(h.record.graph[-12:], h.reason) for h in host.holes]}")
        first, _second = ratios(document)
        counter = dispatch_counts(host)
        host.dispatch("generate", {"prompt": "x", "aspect_ratio": first},
                      request_id="remote-1")
        assert calls, "adopted from the hub and still served eager"
        assert_served_compiled(counter)

        for record in document.lanes[0].graphs:
            assert store.local.has_artifact(record.graph, ENV), (
                f"{record.graph[-12:]} was fetched from the fleet pool and not "
                f"kept — 'check local then remote' is half a cache if the "
                f"remote answer is never banked")
    finally:
        server.shutdown()

    offline: List[str] = []
    rebooted = fresh_host(binding, tmp_path / "second")
    rebooted.setup(
        store=graph_store(tmp_path / "cold-cas", None, tmp_path / "no-baked"),
        document=document, sm=SM, loader=opening_loader(offline),
        artifacts_dir=tmp_path / "artifacts2", stack=STACK)
    assert len(rebooted.adoption.adopted) == 2 and rebooted.holes == ()


def test_a_SKEWED_remote_artifact_refuses_BY_TYPE_and_never_arms(
    binding: DeployBinding, document: Any, tmp_path: Path
) -> None:
    artifacts = []
    for index, record in enumerate(document.lanes[0].graphs):
        artifacts.append(tcg_artifacts.aoti_package(
            tmp_path / "fleet" / f"{index}.pt2",
            graph_specialization=record.graph))
    blobs: dict = {}
    base_url, server = _serve_hub({}, blobs)
    _Hub.answer = _fleet_answer(document, base_url, blobs, artifacts)
    try:
        calls: List[str] = []
        host = fresh_host(binding, tmp_path)
        host.setup(store=_remote_store(tmp_path / "cold-cas", base_url),
                   document=document, sm=SM, loader=opening_loader(calls),
                   artifacts_dir=tmp_path / "artifacts", stack=STACK)
    finally:
        server.shutdown()

    assert len(host.adoption.adopted) == 0, (
        "a bare .pt2 package armed — the loader is not opening its bytes")
    assert len(host.holes) == 2
    for hole in host.holes:
        assert hole.reason.startswith("ArtifactFormatSkew:"), (
            f"the refusal must NAME the skew, so the remedy is 're-publish' "
            f"and not 'scrub the disk': {hole.reason}")
        assert "bare AOTI .pt2 package" in hole.reason

    first, _second = ratios(document)
    host.dispatch("generate", {"prompt": "x", "aspect_ratio": first},
                  request_id="skew-1")
    assert calls == []


def test_a_TRUNCATED_remote_artifact_is_refused_before_it_is_ever_stored(
    binding: DeployBinding, document: Any, tmp_path: Path
) -> None:
    """Corruption in transit is the hub store's own digest gate, not the loader's — and a refused fetch must never leave bytes in the local CAS."""
    artifacts = []
    for index, record in enumerate(document.lanes[0].graphs):
        artifacts.append(tcg_artifacts.build(
            tmp_path / "fleet" / f"{index}.tar.gz",
            graph_specialization=record.graph, sm=SM))
    blobs: dict = {}
    base_url, server = _serve_hub({}, blobs)
    answer = _fleet_answer(document, base_url, blobs, artifacts)
    for name in list(blobs):
        blobs[name] = blobs[name][:-64]
    _Hub.answer = answer
    cas = tmp_path / "cold-cas"
    try:
        host = fresh_host(binding, tmp_path)
        host.setup(store=_remote_store(cas, base_url), document=document,
                   sm=SM, loader=opening_loader([]),
                   artifacts_dir=tmp_path / "artifacts", stack=STACK)
    finally:
        server.shutdown()

    assert len(host.adoption.adopted) == 0 and len(host.holes) == 2
    for hole in host.holes:
        assert "failed digest verification" in hole.reason, hole.reason
    for record in document.lanes[0].graphs:
        assert not graph_store(cas, None, tmp_path / "no-baked").has_artifact(
            record.graph, ENV), (
            "bytes that failed their own digest gate were banked locally; a "
            "poisoned fetch would then be a permanent local hit")


def test_the_flow_has_ONE_arming_path_for_local_remote_and_self_minted(
    binding: DeployBinding, document: Any, tmp_path: Path
) -> None:
    seen: List[Path] = []

    def recording_loader(path: Path, record: Any, module: Any) -> Any:
        seen.append(Path(path))
        return opening_loader([])(path, record, module)

    cas = tmp_path / "cas"
    store = graph_store(cas, None, tmp_path / "no-baked")
    host = fresh_host(binding, tmp_path)
    host.setup(store=store, document=document, sm=SM, loader=recording_loader,
               artifacts_dir=tmp_path / "artifacts", stack=STACK)
    assert len(host.holes) == 2
    mint_the_holes(host, store, tmp_path, [])
    minted = list(seen)

    seen.clear()
    warm = fresh_host(binding, tmp_path / "warm")
    warm.setup(store=graph_store(cas, None, tmp_path / "no-baked"),
               document=document, sm=SM, loader=recording_loader,
               artifacts_dir=tmp_path / "artifacts-warm", stack=STACK)
    adopted = list(seen)

    assert len(minted) == 2 and len(adopted) == 2
    for path in minted + adopted:
        assert path.is_file(), f"{path} is not a file — a directory reached the loader"
        with path.open("rb") as handle:
            assert handle.read(2) == b"\x1f\x8b", f"{path} is not an envelope"
    assert sorted(p.name for p in minted) == sorted(p.name for p in adopted)


def test_a_mint_with_no_store_is_unrepresentable(tmp_path: Path) -> None:
    from gen_worker.serving import mint as mint_mod

    with pytest.raises(ValueError, match="needs the store"):
        mint_mod.BackgroundMint(
            host=object(), store=None, compiler=lambda *a: Path("."),
            artifacts_dir=tmp_path)


def test_every_entry_point_builds_the_SAME_store(tmp_path: Path) -> None:
    import gen_worker

    root = Path(gen_worker.__file__).parent
    offenders = []
    for path in (root / "cli" / "compile.py", root / "cli" / "daemon.py",
                 root / "serving" / "__main__.py",
                 root / "serving" / "serve_adoption.py"):
        text = path.read_text()
        if "LocalGraphStore(" in text:
            offenders.append(str(path))
    assert offenders == [], (
        f"these entry points construct a store directly instead of through "
        f"`mint_store.graph_store`, which is how a hub tier goes missing on "
        f"one path and not another: {offenders}")
