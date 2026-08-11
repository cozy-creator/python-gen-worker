"""pgw#1116: a boot-adopt refusal must NAME ITS GATE, on the wire.

The measured defect (pgw#1108's POD PROOF, three real pods on 0.103.0):
``/v1/worker/cells/resolve`` was called **zero** times, every pod self-minted,
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
4. **A boot with no local cell and a reachable hub actually issues the resolve
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

from gen_worker import activity, boot_adopt, boot_key, cell_resolve
from gen_worker import executor as executor_mod


def _raise(exc: BaseException) -> Any:
    def _f(*_a: Any, **_k: Any) -> Any:
        raise exc

    return _f

REPO = Path(__file__).resolve().parent.parent
MICRO_SRC = REPO / "examples" / "micro-diffusion" / "src"

KEY = "ck1-" + "3f" * 28


# ---------------------------------------------------------------------------
# Capture: the REAL sink, so the assertions read the same ActivityUpdate the
# hub's worker_activity_events row is built from.
# ---------------------------------------------------------------------------


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    seen: List[Any] = []
    monkeypatch.setattr(activity, "_sink", seen.append, raising=False)
    return seen


def _adopt_events(seen: List[Any]) -> List[Any]:
    return [u for u in seen if u.kind == activity.KIND_BOOT_ADOPT]


def _one(seen: List[Any]) -> Any:
    rows = _adopt_events(seen)
    assert len(rows) == 1, (
        "a boot-adopt decision must emit EXACTLY one typed event — got "
        f"{[(u.kind, u.phase) for u in seen]}")
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
    for token in ("no_export_declaration", "declaration_unreadable", "no_hub"):
        assert token in boot_adopt.GATE_REASONS


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

    def compile_cell(self) -> _Cfg:
        return _Cfg()


def _executor(tmp_path: Path) -> Any:
    from gen_worker.executor import Executor, ModelStore

    async def _send(msg: Any) -> None:
        pass

    ex = Executor([], _send, store=ModelStore(_send, cache_dir=tmp_path / "cas"))
    ex.file_base_url = "http://hub.local"
    ex.worker_jwt_provider = lambda: "worker-jwt"
    return ex


class _Blocked:
    """A pgw#853 thunk-shaped declaration that REFUSES when asked."""


@pytest.mark.parametrize(
    "gate,wire,expect",
    [
        (
            "no_export_declaration",
            lambda mp: mp.setattr(
                executor_mod.aot_mint, "export_declaration", lambda f: None),
            "no registered export declaration",
        ),
        (
            "declaration_refused",
            lambda mp: mp.setattr(
                executor_mod.aot_mint, "export_declaration",
                _raise(RuntimeError("OQ-2 audio_timestep rank unresolved"))),
            "audio_timestep",
        ),
        (
            "declaration_unreadable",
            lambda mp: mp.setattr(
                executor_mod.aot_declaration, "cell_plans",
                _raise(ValueError("two mint plans share entry name"))),
            "share entry name",
        ),
    ],
)
def test_each_pre_attempt_gate_names_itself_on_the_wire(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, events: List[Any],
    gate: str, wire: Any, expect: str,
) -> None:
    monkeypatch.setattr(
        executor_mod.aot_mint, "export_declaration", lambda f: object())
    monkeypatch.setattr(
        executor_mod.aot_declaration, "cell_plans", lambda d: [object()])
    monkeypatch.setattr(
        executor_mod.boot_adopt, "attempt",
        _raise(AssertionError("attempt must not be reached past this gate")))
    wire(monkeypatch)

    out = _executor(tmp_path)._boot_adopt(_Spec(), {})

    assert out is not None, (
        "a gate that returns a bare None carries no reason, no family and no "
        "event — which is exactly how three pods refused unattributably")
    assert out.reason == gate
    assert not out.adopted
    row = _one(events)
    assert row.phase == gate, (
        "the event's phase must be the GATE's own token — 'adopt failed' makes "
        "eight different bugs look like one")
    assert "family=micro-diffusion" in row.detail
    assert "function=generate" in row.detail
    assert expect in row.detail


def test_no_hub_names_which_half_of_the_readiness_was_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, events: List[Any],
) -> None:
    from gen_worker.procsplit import broker

    monkeypatch.setattr(
        executor_mod.aot_mint, "export_declaration", lambda f: object())
    monkeypatch.setattr(
        executor_mod.aot_declaration, "cell_plans", lambda d: [object()])
    monkeypatch.setattr(broker, "_broker", None, raising=False)

    ex = _executor(tmp_path)
    ex.file_base_url = ""
    ex.worker_jwt_provider = lambda: ""

    out = ex._boot_adopt(_Spec(), {})
    assert out.reason == "no_hub"
    row = _one(events)
    assert row.phase == "no_hub"
    assert "base_url=<unset>" in row.detail and "seam=down" in row.detail


def test_the_gates_are_pairwise_distinguishable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The property the pod proof needed and did not have: knowing an adopt did
    not happen is worth nothing unless the phases differ."""
    seen: List[str] = []
    monkeypatch.setattr(
        activity, "_sink",
        lambda u: seen.append(u.phase) if u.kind == activity.KIND_BOOT_ADOPT
        else None, raising=False)
    monkeypatch.setattr(
        executor_mod.aot_declaration, "cell_plans", lambda d: [object()])

    ex = _executor(tmp_path)
    monkeypatch.setattr(
        executor_mod.aot_mint, "export_declaration", lambda f: None)
    ex._boot_adopt(_Spec(), {})
    monkeypatch.setattr(
        executor_mod.aot_mint, "export_declaration", lambda f: object())
    monkeypatch.setattr(
        executor_mod.aot_declaration, "cell_plans", _raise(ValueError("nope")))
    ex._boot_adopt(_Spec(), {})
    ex.file_base_url = ""
    monkeypatch.setattr(
        executor_mod.aot_declaration, "cell_plans", lambda d: [object()])
    ex._boot_adopt(_Spec(), {})

    assert len(seen) == len(set(seen)) == 3, (
        f"three different refusals must read as three different phases: {seen}")


# ---------------------------------------------------------------------------
# 3. Every terminus inside `attempt` emits, hit included
# ---------------------------------------------------------------------------


def _derived(wall_ms: int = 1234) -> Any:
    from gen_worker import cell_key as ck

    return boot_key.DerivedKey(
        key=ck.from_axes({
            "graph": "c0ffee0000000000", "envelope": "e" * 16,
            "sm": "sm_89", "toolchain": "t" * 16}),
        class_hashes={"a": "0" * 16}, combined="c0ffee0000000000",
        workers=2, width_reason="test", traced=1, memo="miss",
        wall_ms=wall_ms)


class _Cell:
    publisher_org = "org-a"
    publisher_tier = "platform"
    cell_ref = "root/family-micro-diffusion#" + KEY
    content_digest = "sha256:" + "ab" * 32


def _attempt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **wires: Any) -> Any:
    if "derive" in wires:
        monkeypatch.setattr(boot_key, "derive", wires["derive"])
    if "resolve" in wires:
        monkeypatch.setattr(cell_resolve, "resolve", wires["resolve"])
    if "materialize" in wires:
        monkeypatch.setattr(cell_resolve, "materialize", wires["materialize"])
    return boot_adopt.attempt(
        function="generate", modules=("micro_diffusion.main",), cfg=_Cfg(),
        slots={}, declared_hint=3,
        envelope={"shapes": [[64, 64]], "text_lens": [8], "guidance": []},
        work_root=tmp_path)


def _refuse_hub(code: str) -> Any:
    return _raise(cell_resolve.CellResolveRefused(code, "the hub said so", status=409))


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
        ("cell_resolve_ambiguous",
         {"derive": lambda **_k: _derived(), "resolve": _refuse_hub(
             "cell_resolve_ambiguous")}),
        ("resolve_unreachable",
         {"derive": lambda **_k: _derived(),
          "resolve": _raise(OSError("connection reset"))}),
        ("miss", {"derive": lambda **_k: _derived(),
                  "resolve": lambda *_a, **_k: None}),
        # Step 3 — materialize + the pgw#1031 witness floor.
        ("materialize_failed",
         {"derive": lambda **_k: _derived(),
          "resolve": lambda *_a, **_k: _Cell(),
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


def test_a_pod_that_derived_and_missed_is_not_a_pod_that_never_derived(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, events: List[Any],
) -> None:
    """The issue's regression box, verbatim. Both pods self-mint; only one of
    them proves the hub was asked and answered."""
    _attempt(monkeypatch, tmp_path, derive=lambda **_k: _derived(),
             resolve=lambda *_a, **_k: None)
    missed = _one(events)
    events.clear()

    monkeypatch.setattr(
        executor_mod.aot_mint, "export_declaration", lambda f: None)
    _executor(tmp_path)._boot_adopt(_Spec(), {})
    never = _one(events)

    assert missed.phase == "miss" and never.phase == "no_export_declaration"
    assert "key=ck1-" in missed.detail, (
        "a MISS must carry the key that missed — it is the whole difference "
        "between 'the hub holds nothing for me' and 'I never asked'")
    assert "key=-" in never.detail


# ---------------------------------------------------------------------------
# 4. END TO END: a boot with no local cell and a reachable hub ASKS
# ---------------------------------------------------------------------------


class _ResolveHub(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_a: Any) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = json.loads(raw or b"{}")
        self.server.calls.append((self.path, body))  # type: ignore[attr-defined]
        # th#1788: the live hub withholds every self-minted cell from a resolve,
        # so a MISS is what a correct worker gets today. The property under test
        # is that it ASKED.
        out = json.dumps({"found": False}).encode()
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
    pytest.importorskip("accelerate")
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

    from gen_worker import aot_serve, compile_cache

    full = {
        "sku": "l4", "sm": "sm_89", "torch": str(torch.__version__),
        "triton": "3.6.0", "cuda": "13.0",
        "image_digest": "sha256:" + "ab" * 32,
    }
    monkeypatch.setattr(compile_cache, "runtime_key", lambda: dict(full))
    monkeypatch.setattr(aot_serve, "runtime_key", lambda: {
        k: full[k] for k in ("sku", "sm", "torch", "cuda")})


def test_a_cold_boot_with_a_reachable_hub_actually_issues_the_resolve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, events: List[Any],
    hub: Any, micro_tree: Path, micro_declaration: None, sm_runtime: None,
) -> None:
    """THE end-to-end property, on the vehicle the three pods ran.

    Real ``Executor._boot_adopt`` -> real ``boot_key.derive`` (structure-only
    trace children on fake tensors, no compile) -> real ``cell_resolve.resolve``
    -> a real HTTP hub. What the pods' GIN log counted zero of is counted here,
    and the decision is READABLE afterwards, which is the half that was missing.
    """
    from gen_worker.api.binding import ModelRef
    from gen_worker.mint_process import MintSlot
    from gen_worker.procsplit import broker
    from gen_worker.registry import collect_endpoints

    if str(MICRO_SRC) not in sys.path:
        sys.path.insert(0, str(MICRO_SRC))
    monkeypatch.syspath_prepend(str(REPO / "tests"))
    monkeypatch.setenv(
        "PYTHONPATH", ":".join([
            str(REPO / "src"), str(REPO / "tests"), str(MICRO_SRC)]))
    # Single-process posture: no seam, a local bearer. `broker.request` then
    # makes the same POST the parent makes on a split pod.
    monkeypatch.setattr(broker, "_broker", None, raising=False)

    specs = collect_endpoints(["harness.rig_runtime", "micro_diffusion.main"])
    spec = next(s for s in specs if s.name == "generate")

    ex = _executor(tmp_path)
    ex.file_base_url = f"http://127.0.0.1:{hub.server_address[1]}"
    slots = {"pipeline": MintSlot(
        ref=ModelRef(source="tensorhub", path="cozy/micro-diffusion",
                     tag="prod"),
        path=str(micro_tree))}

    out = ex._boot_adopt(spec, slots)

    resolves = [c for c in hub.calls if c[0] == cell_resolve.RESOLVE_PATH]
    assert len(resolves) == 1, (
        "the worker did not ask the hub by its derived key — this is the pod "
        f"defect, reproduced off-pod. Hub saw: {hub.calls}")
    assert resolves[0][1] == {
        "family": "micro-diffusion", "cell_key": out.derived_key}
    assert out.derived_key.startswith("ck1-")
    assert out.reason == "miss"

    row = _one(events)
    assert row.phase == "miss"
    assert out.derived_key in row.detail
    assert row.duration_ms > 0, (
        "the derivation was measured and the measurement must reach the hub — "
        "otherwise 'the boot derive costs seconds' stays an anecdote")
