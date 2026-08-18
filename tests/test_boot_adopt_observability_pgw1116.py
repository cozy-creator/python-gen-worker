"""pgw#1116: a boot-adopt refusal must NAME ITS GATE, on the wire.

The measured defect (pgw#1108's POD PROOF, three real pods on 0.103.0):
``/v1/worker/compiled-graphs/resolve`` was called **zero** times, every pod self-minted,
and nothing off-pod could say why — ``BootAdoptOutcome.reason``/``.detail``/
``.derived_key``/``.derive_ms`` had zero readers outside ``boot_adopt.py`` (the
executor consumed only ``.adoption``), and the executor's own three pre-attempt
gates returned a bare ``None`` that built no outcome at all. So a merged fix and
a published wheel both sailed past a completely broken adopt path.

What is proved here:

1. **Every gate is individually distinguishable on the wire.** One typed
   ``boot_adopt`` activity event per decision, ``phase`` = the gate's own token.
   Parametrized over every terminus — the four executor pre-attempt gates, the
   eager-only non-entry, each ``attempt`` refusal, the miss and the hit.
2. **The vocabulary is exhaustive.** Every refusal token any path in the tree
   can produce is in ``boot_adopt.REASONS``; a new refusal that forgets to name
   itself fails here rather than becoming the next silent one.
3. **A pod that DERIVED a key and missed is distinguishable from a pod that
   never derived one** (the issue's regression box).
4. **A boot with no local compiled graph and a reachable hub actually issues the resolve
   call** — driven end to end through the real ``Executor._boot_adopt``, the
   real ``boot_key.derive`` (three structure-only trace children, fake tensors,
   no compile anywhere) and a real HTTP hub, against ``examples/micro-diffusion``
   — the exact vehicle the three pods ran.

None of this traces a real checkpoint, links a ``.so`` or mints: the derivation
is fake-tensor tracing, which is what the "mints run on remote machines only"
rule explicitly permits.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest

from harness.slot_facts import TEST_FACTS as _TEST_FACTS

from gen_worker import activity, boot_adopt, boot_key, compiled_graph_resolve
from gen_worker import executor as executor_mod
from gen_worker.api import export_contract as export_contract_mod


def _raise(exc: BaseException) -> Any:
    def _f(*_a: Any, **_k: Any) -> Any:
        raise exc

    return _f

REPO = Path(__file__).resolve().parent.parent
MICRO_SRC = REPO / "examples" / "micro-diffusion" / "src"

KEY = "cg-key-v1-" + "3f" * 28


# ---------------------------------------------------------------------------
# Capture: the REAL sink, so the assertions read the same ActivityUpdate the
# hub's worker_activity_events row is built from.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _empty_local_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """pgw#1127: the pre-derive gate asks whether ANYBODY could answer, and this
    machine's own compiled graph store is one of the two answerers. Pin it to an empty
    root so every row here reads a fact about the test rather than about
    whatever `~/.cache/cozy/compiled graphs` happens to hold on the box."""
    from gen_worker import local_compiled_graph_store

    monkeypatch.setenv(
        local_compiled_graph_store.ENV_STORE_DIR,
        str(tmp_path_factory.mktemp("empty-compiled graphs")))


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    seen: List[Any] = []
    monkeypatch.setattr(activity, "_sink", seen.append, raising=False)
    return seen


def _adopt_events(seen: List[Any]) -> List[Any]:
    return [u for u in seen if u.kind == activity.KIND_BOOT_ADOPT]


def _one(seen: List[Any]) -> Any:
    """The boot-adopt event, when the decision is one per CLASS.

    pgw#1176: the rule this helper enforces was "EXACTLY one typed event per
    boot-adopt decision", and it is UNCHANGED — what changed is what a
    decision is about. A boot derives a KEY SET and decides per class, so a
    3-class declaration emits three events, one per class, each carrying its
    own key. That is the anti-silence property pgw#1116 exists for, applied at
    the granularity the atom made real: a pod that resolved 30 of 36 keys must
    say so 36 times, not once.

    Every row must still reach the SAME terminus here — these fixtures fail
    identically for all classes — so a divergence between them is a defect
    this assertion catches.
    """
    rows = _adopt_events(seen)
    assert rows, (
        "a boot-adopt decision emitted NO typed event — got "
        f"{[(u.kind, u.phase) for u in seen]}")
    phases = {u.phase for u in rows}
    assert len(phases) == 1, (
        "the classes of one declaration reached DIFFERENT boot-adopt termini "
        f"in a fixture that fails identically for all of them: {phases}")
    return rows[0]


# ---------------------------------------------------------------------------
# 1. The vocabulary is exhaustive — read out of the tree, not restated
# ---------------------------------------------------------------------------


def _tokens(path: Path, pattern: str) -> List[str]:
    return re.findall(pattern, path.read_text())


def test_every_refusal_token_in_the_tree_is_in_the_vocabulary() -> None:
    """A refusal path that names a token ``REASONS`` does not carry is a path
    whose event nobody can enumerate, count or alert on — i.e. the next silent
    one. Read the sites out of the source so adding a refusal without adding
    its name fails HERE."""
    src = REPO / "src" / "gen_worker"
    found: Dict[str, str] = {}
    for token in _tokens(
        src / "boot_trace_child.py", r"_fail\(\s*report_path,\s*\"([a-z_]+)\"",
    ):
        found[token] = "boot_trace_child._fail"
    for token in _tokens(
        src / "boot_key.py", r"BootKeyUnavailable\(\s*\n?\s*\"([a-z_]+)\"",
    ):
        found[token] = "boot_key.BootKeyUnavailable"
    for token in _tokens(
        src / "boot_key.py", r"reason=\"([a-z_]+)\"",
    ):
        found[token] = "boot_key trace report"
    for token in _tokens(
        src / "executor.py", r"boot_adopt\.refused\(\s*\n?\s*\"([a-z_]+)\"",
    ):
        found[token] = "executor gate"
    for token in _tokens(
        src / "boot_adopt.py", r"reason=\"([a-z_]+)\"",
    ):
        found[token] = "boot_adopt.attempt"
    # pgw#1123: `boot_trace_child` now reports the structure-only refusal under
    # whichever of these two `structure_only.refusal_token` picks, so the
    # tokens themselves live there and must still be read out of the tree.
    for token in _tokens(
        src / "models" / "structure_only.py", r"TOKEN_[A-Z_]+ = \"([a-z_]+)\"",
    ):
        found[token] = "structure_only.refusal_token"

    assert found, "the scan found no refusal sites at all — the patterns rotted"
    missing = sorted(
        f"{token} ({where})" for token, where in found.items()
        if token not in boot_adopt.REASONS)
    assert not missing, (
        "these refusal paths can produce a token boot_adopt.REASONS does not "
        f"carry, so their events are unenumerable: {missing}")


def test_the_vocabulary_names_the_gates_that_used_to_return_a_bare_none() -> None:
    """The three executor pre-attempt gates are the ones that produced NOTHING
    — not even a discarded reason. They are the reason one pod could not name
    its own refusal."""
    # pgw#1127 replaced `no_hub` with `no_compiled_graph_source`: the gate now refuses
    # only when BOTH answerers are absent (no hub AND an empty local store),
    # because the derived ck1 key is `local_compiled_graph_store`'s own address.
    for token in ("no_export_declaration", "declaration_unreadable",
                  "no_compiled_graph_source"):
        assert token in boot_adopt.GATE_REASONS
    assert "no_hub" not in boot_adopt.REASONS, (
        "a token nothing can emit is a query nobody can write — pgw#1127 "
        "split it into `no_compiled_graph_source` (pre-derive, nobody at all) and "
        "`local_miss_no_hub` (derived, this machine does not hold it, and "
        "there is no hub either)")


# ---------------------------------------------------------------------------
# 2. The executor's pre-attempt gates: each names itself, and none returns None
# ---------------------------------------------------------------------------


class _Cfg:
    family = "micro-diffusion"
    targets = ("transformer",)
    shapes = ((64, 64),)
    text_lens = (8,)
    guidance_scales = ()
    lora_bucket = 0


class _Spec:
    name = "generate"
    module = "micro_diffusion.main"
    compile = _Cfg()

    def compile_contract(self) -> _Cfg:
        return _Cfg()


def _executor(tmp_path: Path) -> Any:
    from gen_worker.executor import Executor
    from gen_worker.models.store import ModelStore

    async def _send(msg: Any) -> None:
        pass

    ex = Executor([], _send, store=ModelStore(_send, cache_dir=tmp_path / "cas"))
    ex.file_base_url = "http://hub.local"
    ex.worker_jwt_provider = lambda: "worker-jwt"
    return ex


# ---------------------------------------------------------------------------
# 3. Every terminus inside `attempt` emits, hit included
# ---------------------------------------------------------------------------


def _derived(wall_ms: int = 1234) -> Any:
    from gen_worker._vendor.torchcg import identity as ck

    from gen_worker import keyset

    return keyset.DerivedKeySet(
        # pgw#1176: a boot derives a KEY SET. These declarations trace to one
        # class, so the set has one member and callers take it from `keys`.
        entry_keys={keyset.GraphClassName("a"): keyset.CompiledGraphKey(
            ck.from_axes({
                "graph": "c0ffee0000000000",
                "sm": "sm_89", "toolchain": "t" * 16}).value)},
        source=keyset.KeySource.TRACED,
        closure=keyset.parse_closure_digest("ab" * 16),
        workers=2, width_reason="test", traced=1, wall_ms=wall_ms)


class _Graph:
    publisher_org = "org-a"
    publisher_tier = "platform"
    cg_ref = "root/family-micro-diffusion#" + KEY
    content_digest = "sha256:" + "ab" * 32


def _attempt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **wires: Any) -> Any:
    """The FIRST of the per-class outcomes a boot returns (pgw#1176).

    This declaration traces THREE classes, so `attempt` answers three times —
    one per key, which is the point of the key set. These rows are about the
    TERMINUS vocabulary (which reason, which event), and every class reaches
    the same terminus here because the wires under test fail identically for
    all of them. Rows that care about per-class divergence index the tuple;
    `test_the_verdict_is_bisectable_to_ONE_named_axis` is the one that does.
    """
    # pgw#1327: the deriver is INJECTED, not imported — `boot_adopt` no longer
    # names `boot_key`, so patching that module reaches nothing. These rows are
    # about the terminus VOCABULARY, so they hand `attempt` a deriver directly;
    # the key-set-as-data route has its own file
    # (`tests/test_cgkey_as_data_pgw1327.py`).
    if "resolve" in wires:
        # pgw#1224: the wire is a BATCH. The per-key wires below are lifted to
        # the batch shape here rather than rewritten one by one, so each row
        # keeps stating the TERMINUS it is about.
        monkeypatch.setattr(
            compiled_graph_resolve, "resolve_batch", _batched(wires["resolve"]))
    if "materialize" in wires:
        monkeypatch.setattr(compiled_graph_resolve, "materialize", wires["materialize"])
    return boot_adopt.attempt(
        function="generate", modules=("micro_diffusion.main",), cfg=_Cfg(),
        slots={}, declared_hint=3,
        work_root=tmp_path, derive=wires.get("derive"))[0]


def _batched(per_key: Any) -> Any:
    """A single-key wire -> the batch wire, answer for answer.

    A raise stays a WHOLE-BATCH raise (that is what a caller-scoped refusal is,
    and every key in the batch reports it); a returned compiled graph/None becomes that
    key's own answer.
    """
    def _call(_family: str, keys: Any, **_kw: Any) -> Any:
        out = []
        for key in keys:
            graph = per_key(_family, key)
            out.append(compiled_graph_resolve.ResolveAnswer(
                compiled_graph_key=key,
                status="miss" if graph is None else "hit", graph=graph))
        return tuple(out)
    return _call


def _refuse_hub(code: str) -> Any:
    return _raise(compiled_graph_resolve.CompiledGraphResolveRefused(code, "the hub said so", status=409))


@pytest.mark.parametrize(
    "phase,wires",
    [
        # Step 1 — the derivation. `structure_unsupported` is the one that
        # names a family whose structure-only build does not exist; collapsing
        # it into "derive failed" loses the only signal that says which family
        # to build next.
        ("structure_unsupported", {"derive": _raise(boot_key.BootKeyUnavailable(
            "structure_unsupported",
            "MicroEscapeDenoiser has no from_config"))}),
        ("child_died", {"derive": _raise(boot_key.BootKeyUnavailable(
            "child_died", "trace child exited 1 without a report"))}),
        ("derive_failed", {"derive": _raise(MemoryError("no headroom"))}),
        # Step 2 — the ask. A typed hub refusal is NOT a miss.
        ("compiled_graph_resolve_ambiguous",
         {"derive": lambda **_k: _derived(), "resolve": _refuse_hub(
             "compiled_graph_resolve_ambiguous")}),
        ("resolve_unreachable",
         {"derive": lambda **_k: _derived(),
          "resolve": _raise(OSError("connection reset"))}),
        ("miss", {"derive": lambda **_k: _derived(),
                  "resolve": lambda *_a, **_k: None}),
        # Step 3 — materialize + the pgw#1031 witness floor.
        ("materialize_failed",
         {"derive": lambda **_k: _derived(),
          "resolve": lambda *_a, **_k: _Graph(),
          "materialize": _raise(RuntimeError("content_digest_mismatch"))}),
    ],
)
def test_every_attempt_terminus_emits_its_own_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, events: List[Any],
    phase: str, wires: Dict[str, Any],
) -> None:
    out = _attempt(monkeypatch, tmp_path, **wires)
    assert out.reason == phase
    row = _one(events)
    assert row.phase == phase
    assert row.phase in boot_adopt.REASONS
    assert "family=micro-diffusion" in row.detail


def test_the_derive_wall_rides_the_event_as_a_number(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, events: List[Any],
) -> None:
    """``duration_ms`` is a numeric column hub-side. A derivation timed by the
    pod and then interpolated into free text cannot be grouped or percentiled,
    which is how "the derive is slow" stayed an anecdote."""
    _attempt(monkeypatch, tmp_path, derive=lambda **_k: _derived(8350),
             resolve=lambda *_a, **_k: None)
    assert _one(events).duration_ms == 8350


# ---------------------------------------------------------------------------
# 4. END TO END: a boot with no local compiled graph and a reachable hub ASKS
# ---------------------------------------------------------------------------


class _ResolveHub(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_a: Any) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = json.loads(raw or b"{}")
        self.server.calls.append((self.path, body))  # type: ignore[attr-defined]
        # th#1788: the live hub withholds every self-minted compiled graph from a resolve,
        # so a MISS is what a correct worker gets today. The property under test
        # is that it ASKED.
        # pgw#1224: the real hub answers the BATCH — one answer per requested
        # key, in request order, and a miss is an ANSWER rather than an
        # omission. This double answers exactly that, so the end-to-end row
        # exercises the arity and order checks on a real socket.
        keys = list(body.get("keys") or [])
        out = json.dumps({
            "object": "compiled_graph_resolve_batch",
            "family": body.get("family"),
            "answers": [{"compiled_graph_key": k, "status": "miss", "found": False}
                        for k in keys],
            "hits": 0, "misses": len(keys),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def hub() -> Iterator[Any]:
    srv = HTTPServer(("127.0.0.1", 0), _ResolveHub)
    srv.calls = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def micro_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("torch")
    # No `importorskip("accelerate")`: pgw#1123 removed it from this path, and
    # the row below proves the derivation completes when it is unimportable.
    if str(MICRO_SRC) not in sys.path:
        sys.path.insert(0, str(MICRO_SRC))
    from micro_diffusion.weights import SEED, materialize

    return materialize(tmp_path_factory.mktemp("micro-tree"), seed=SEED)


@pytest.fixture
def micro_declaration(micro_tree: Path) -> None:
    """Restore micro-diffusion's export declaration before the run.

    On a pod the import IS the registration (`micro_diffusion/main.py` imports
    `.aot_declaration`, which registers at module scope). In a SUITE that is not
    enough: several files call `export_contract.reset_export_declarations()`,
    which empties the process-global registry, and by then
    `micro_diffusion.main` is already in `sys.modules` — so `collect_endpoints`
    re-imports nothing and re-registers nothing. Measured on CI run
    31475315256 (`-n 4 --dist loadfile`, worker gw2): this row reported
    `no_export_declaration` while passing in isolation. Restated here rather
    than depended on, because the property under test is the RESOLVE, not
    whichever file happened to share the worker.
    """
    from gen_worker.api import export_contract as ec

    import micro_diffusion.aot_declaration as decl

    ec.register_export_declaration(decl.DECLARATION, replace=True)


@pytest.fixture
def sm_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """``sm`` is a KEY AXIS, so a cardless box cannot state a key at all — which
    is correct behaviour, not something to test around. Patch only the PROBES;
    every value is a real fact of a real runtime and nothing about how they are
    combined into a key is faked."""
    import torch

    from gen_worker import compile_cache

    full = {
        "sku": "l4", "sm": "sm_89", "torch": str(torch.__version__),
        "triton": "3.6.0", "cuda": "13.0",
        "image_digest": "sha256:" + "ab" * 32,
    }
    monkeypatch.setattr(compile_cache, "runtime_key", lambda: dict(full))


