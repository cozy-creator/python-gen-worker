"""pgw#1327: a serve pod states its ``cg-key-v1`` set from DATA, not from traces.

The measured defect: ``boot_key.derive`` learned the boot key set by running
``torch.export`` in child processes — its own docstring, *"< 60 s, every time,
~99 % of it the traces"* — so a pod that will never compile anything still needed
torch, diffusers and the endpoint's model code importable at boot, only to
compute a pure function of code the mint lane already ran.

What is proved here, in the order the checklist asks for it:

1. **The key set is a versioned, typed schema.** Unknown versions, unknown
   fields and malformed identifiers are refused LOUDLY; nothing reaches a fold
   as a bare ``dict[str, str]``.
2. **THE HEADLINE GATE — a fresh pod boots with ZERO ``torch.export`` calls.**
   Driven through the real ``boot_adopt.attempt`` with a real closure digest,
   a shipped document, an armed ``torch.export`` sentinel that raises on any
   call, an armed ``subprocess.Popen`` sentinel, and a deriver spy that fails
   the test if it is reached. The pod resolves, materializes and hands back an
   adoption.
3. **Drift fails safe.** A closure the document does not carry is a stated
   ``keyset_absent`` refusal, never a fold of the nearest row; a document
   carrying a DIFFERENT graph's hash folds to a different key, so the hub
   answers MISS and no wrong cell is ever armed.
4. **The tracer is unreachable from the serve-boot path**, statically.
5. **The round trip on the real vehicle:** what ``boot_key.derive`` traces is,
   key for key, what a fresh pod reads out of the shipped document — so the data
   is the same value, not an approximation of it.

Only row 5 runs a tracer, and only as the MINT lane: structure-only
``torch.export`` on fake tensors, in child processes, with no compile, no
``.so`` and no publish — which the standing "mints run on remote machines only"
rule explicitly permits. Every other row's graph hash is a fixture, which is the
point: a serving pod should not have to produce one.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, cast

import msgspec
import pytest

from harness.slot_facts import TEST_FACTS as _TEST_FACTS

from gen_worker import boot_adopt, cell_resolve, compile_cache as cc, keyset
from gen_worker.child_contract import CompileSpec
from gen_worker.keyset import document as doc_mod, store as keyset_store

REPO = Path(__file__).resolve().parent.parent
MICRO_SRC = REPO / "examples" / "micro-diffusion" / "src"

FAMILY = "micro-diffusion"
FUNCTION = "generate"
MODULES: Tuple[str, ...] = ("micro_diffusion.main",)

#: Two fixture graph hashes. TCG's grammar (16 lowercase hex) and nothing else —
#: a class hash is opaque to everything downstream of the exporter that made it.
HASH_A = "a1b2c3d4e5f60718"
HASH_B = "0f1e2d3c4b5a6978"
INGRESS = "9" * 32


@pytest.fixture(autouse=True)
def _micro_on_path() -> Iterator[None]:
    added = str(MICRO_SRC)
    if added not in sys.path:
        sys.path.insert(0, added)
    yield


@pytest.fixture(autouse=True)
def _stable_sm(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI has no card. The fold needs an ``sm``, and pinning it here is what
    makes the machine-independence assertions below mean something: the same
    document under two ``sm`` values must produce two different key sets."""
    monkeypatch.setattr(
        cc, "runtime_key", lambda: {"sm": "sm_89", "torch": "", "triton": ""})


def _hashes(**classes: str) -> Dict[keyset.GraphClassName, keyset.ClassHash]:
    """The graph axis, through the parsers — never a bare ``dict[str, str]``."""
    return {
        keyset.parse_graph_class_name(name): keyset.parse_class_hash(value)
        for name, value in classes.items()
    }


def _spec() -> CompileSpec:
    return CompileSpec(
        shapes=((64, 64),), targets=("transformer",), family=FAMILY,
        lora_bucket=0, guidance_scales=(), text_lens=(8,))


class _Cfg:
    family = FAMILY
    targets = ("transformer",)
    shapes = ((64, 64),)
    text_lens = (8,)
    guidance_scales = ()
    lora_bucket = 0


def _digest() -> keyset.ClosureDigest:
    return keyset.closure_of(
        family=FAMILY, function=FUNCTION, modules=MODULES, cfg=_spec(),
        slots={})


def _row(**classes: str) -> doc_mod.ClosureRow:
    return doc_mod.closure_row(
        family=FAMILY,
        function=FUNCTION,
        tcg_version=keyset.tcg_version(),
        classes={
            name: doc_mod.GraphClassRow(
                graph_class=name, class_hash=class_hash,
                ingress_digest=INGRESS, target="transformer")
            for name, class_hash in classes.items()
        },
        emitted_by="pgw#1327 fixture",
    )


def _ship(root: Path, digest: keyset.ClosureDigest, row: doc_mod.ClosureRow) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / keyset.KEYSET_FILENAME
    path.write_bytes(keyset.encode(keyset.KeySetDocument(
        schema=keyset.KEYSET_SCHEMA, version=keyset.KEYSET_VERSION,
        closures={str(digest): row})))
    return path


# ---------------------------------------------------------------------------
# 1. The schema is versioned and typed, and refuses what it does not understand
# ---------------------------------------------------------------------------


def test_the_document_round_trips_and_is_canonically_ordered() -> None:
    digest = _digest()
    document = keyset.KeySetDocument(
        schema=keyset.KEYSET_SCHEMA, version=keyset.KEYSET_VERSION,
        closures={str(digest): _row(denoiser=HASH_A, vae=HASH_B)})
    raw = keyset.encode(document)
    assert keyset.encode(keyset.decode(raw)) == raw, (
        "a document must re-encode byte-identically, or whoever bakes it into "
        "an image cannot content-address it")
    closure = keyset.parse_closure(keyset.decode(raw), digest)
    assert [row.graph_class for row in closure.classes] == ["denoiser", "vae"]
    assert closure.class_hashes == {"denoiser": HASH_A, "vae": HASH_B}


@pytest.mark.parametrize(
    "mutate,expect",
    [
        (lambda d: {**d, "version": 2}, "refuses to read a document it does not"),
        (lambda d: {**d, "schema": "cg-keyset-v2"}, "refuses to read a document"),
        (lambda d: {**d, "unexpected": 1}, "does not match cg-keyset-v1"),
        (lambda d: {"version": 1}, "refuses to read a document"),
    ],
)
def test_an_unknown_version_or_field_is_refused_loudly(
    mutate: Any, expect: str,
) -> None:
    """A partially-understood key set is not a narrower answer, it is a wrong
    one — and its wrong answer selects a wrong graph."""
    digest = _digest()
    payload = msgspec.json.decode(keyset.encode(keyset.KeySetDocument(
        schema=keyset.KEYSET_SCHEMA, version=keyset.KEYSET_VERSION,
        closures={str(digest): _row(denoiser=HASH_A)})))
    with pytest.raises(keyset.KeySetError) as caught:
        keyset.decode(msgspec.json.encode(mutate(payload)))
    assert caught.value.reason == "keyset_invalid"
    assert expect in caught.value.detail


@pytest.mark.parametrize(
    "bad", ["", "A1B2C3D4E5F60718", "a1b2c3d4e5f607", "a1b2c3d4e5f60718 ", "zz" * 8],
)
def test_a_malformed_class_hash_never_becomes_a_key(bad: str) -> None:
    with pytest.raises(keyset.KeySetError):
        keyset.parse_class_hash(bad)


def test_identifier_types_are_not_interchangeable_strings() -> None:
    """pgw#1326 "Strict typing": a class hash and an ingress digest are both
    hex strings and only the parsers tell them apart."""
    with pytest.raises(keyset.KeySetError):
        keyset.parse_ingress_digest(HASH_A)          # 16 hex is not an ingress
    with pytest.raises(keyset.KeySetError):
        keyset.parse_class_hash(INGRESS)             # 32 hex is not a class hash
    with pytest.raises(keyset.KeySetError):
        keyset.parse_compiled_graph_key(HASH_A)      # nor is either one a key


def test_a_key_is_never_admitted_as_an_identity_fact() -> None:
    """TCG's own boundary grammar, delegated rather than re-spelled."""
    folded = keyset.fold_entry_keys(_hashes(denoiser=HASH_A), family=FAMILY)
    key = str(folded[keyset.GraphClassName("denoiser")])
    assert keyset.parse_compiled_graph_key(key) == key
    with pytest.raises(keyset.KeySetError):
        keyset.parse_class_hash(key)


# ---------------------------------------------------------------------------
# 2. THE HEADLINE GATE: a boot with a shipped key set makes ZERO export calls
# ---------------------------------------------------------------------------


class _ExportSentinel:
    """Any attribute access on ``torch.export`` is a failure of this leg.

    Blunt on purpose: pgw#1327's claim is not "we call a cheaper exporter", it is
    that a serving pod's key path does not reach one at all.
    """

    def __init__(self) -> None:
        self.touched: List[str] = []

    def __getattr__(self, name: str) -> Any:
        self.touched.append(name)
        raise AssertionError(
            f"pgw#1327: the boot path reached torch.export.{name} — a pod with "
            f"a shipped cg-keyset-v1 must state its keys from data")


@pytest.fixture
def no_export(monkeypatch: pytest.MonkeyPatch) -> _ExportSentinel:
    sentinel = _ExportSentinel()
    torch = sys.modules.get("torch")
    if torch is not None:
        monkeypatch.setattr(torch, "export", sentinel, raising=False)
    monkeypatch.setitem(sys.modules, "torch.export", sentinel)
    return sentinel


@pytest.fixture
def no_children(monkeypatch: pytest.MonkeyPatch) -> None:
    """No trace child may be spawned either — the exporter runs in a child, so
    a sentinel on this process's ``torch.export`` alone would not see it."""
    import subprocess

    def _refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            f"pgw#1327: the boot path spawned a process ({args[:1]!r}) — a "
            f"shipped key set costs one JSON parse, not a process pool")

    monkeypatch.setattr(subprocess, "Popen", _refuse)


class _Deriver:
    """The mint-lane fallback. Reaching it at all fails the zero-export leg."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, **kwargs: Any) -> keyset.DerivedKeySet:
        self.calls += 1
        raise AssertionError(
            "pgw#1327: the mint-lane deriver was reached even though a "
            "cg-keyset-v1 document holds this pod's closure")


class _Cell:
    publisher_org = "org-a"
    publisher_tier = "platform"
    content_digest = "sha256:" + "ab" * 32

    def __init__(self, key: str) -> None:
        self.cell_ref = f"root/family-{FAMILY}#{key}"


def _hub(seen: List[str]) -> Any:
    def _resolve_batch(family: str, keys: Any, **_kw: Any) -> Any:
        seen.extend(str(k) for k in keys)
        return tuple(
            cell_resolve.ResolveAnswer(
                compiled_graph_key=str(key), status="hit",
                cell=cast(Any, _Cell(str(key))))
            for key in keys)
    return _resolve_batch


def test_a_fresh_pod_with_a_shipped_key_set_adopts_with_zero_torch_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    no_export: _ExportSentinel, no_children: None,
) -> None:
    """THE acceptance row. Fresh pod = empty cache, no memo, no volume.

    Red-provable: delete the data path in ``boot_adopt._key_set`` and this fails
    on the deriver spy; leave the data path but let it fall back on a miss and it
    fails on the export/subprocess sentinels.
    """
    digest = _digest()
    shipped = _ship(tmp_path / "image", digest, _row(denoiser=HASH_A, vae=HASH_B))
    asked: List[str] = []
    monkeypatch.setattr(cell_resolve, "resolve_batch", _hub(asked))
    monkeypatch.setattr(
        cell_resolve, "materialize",
        lambda cell, **_kw: tmp_path / "artifact.pt2")
    deriver = _Deriver()

    outcomes = boot_adopt.attempt(
        function=FUNCTION, modules=MODULES, cfg=_Cfg(), slots={},
        declared_hint=2, work_root=tmp_path / "work",
        cache_dir=tmp_path / "empty-cache",
        memo_dir=tmp_path / "empty-cache",
        base_url="http://hub.local", bearer="jwt",
        keyset_roots=(shipped.parent,), derive=deriver)

    assert deriver.calls == 0
    assert not no_export.touched
    assert len(outcomes) == 2, (
        "one outcome per declared graph class, exactly as a traced boot "
        "produced (pgw#1176)")
    assert {o.reason for o in outcomes} == {boot_adopt.HIT}
    assert all(o.adopted for o in outcomes)
    assert {o.key_source for o in outcomes} == {keyset.KeySource.SHIPPED}, (
        "the event must SAY the keys came from data — otherwise 'did any pod "
        "still trace at boot' is an inference rather than a query")
    assert sorted(asked) == sorted(
        str(k) for k in keyset.fold_entry_keys(
            _hashes(denoiser=HASH_A, vae=HASH_B), family=FAMILY).values()), (
        "the keys asked of the hub must be the shipped hashes folded with THIS "
        "pod's runtime axes — not the shipped values themselves")


def test_the_shipped_document_holds_no_folded_key(tmp_path: Path) -> None:
    """A document that carried keys would be pinned to one SKU and one
    toolchain. Assert the bytes, so a future field cannot quietly add one."""
    digest = _digest()
    path = _ship(tmp_path, digest, _row(denoiser=HASH_A))
    raw = path.read_text()
    assert "cg-key-v1" not in raw, (
        "a cg-keyset-v1 document ships the machine-independent graph axis only")
    assert HASH_A in raw and INGRESS in raw


def test_the_same_document_folds_differently_on_a_different_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Machine independence, stated as the property that makes it worth
    shipping: ONE document serves every SKU."""
    hashes = _hashes(denoiser=HASH_A)
    on_89 = keyset.fold_entry_keys(hashes, family=FAMILY)
    monkeypatch.setattr(
        cc, "runtime_key", lambda: {"sm": "sm_100", "torch": "", "triton": ""})
    on_100 = keyset.fold_entry_keys(hashes, family=FAMILY)
    assert on_89 != on_100


def test_a_pod_that_cannot_read_a_compute_capability_refuses_to_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cc, "runtime_key", lambda: {"sm": "", "torch": "", "triton": ""})
    with pytest.raises(keyset.KeySetError) as caught:
        keyset.fold_entry_keys(_hashes(denoiser=HASH_A), family=FAMILY)
    assert caught.value.reason == "no_runtime_sm"


# ---------------------------------------------------------------------------
# 3. Drift fails safe
# ---------------------------------------------------------------------------


def test_a_closure_the_document_does_not_carry_is_a_stated_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_export: _ExportSentinel,
    no_children: None,
) -> None:
    """A stale shipped key set must resolve to MISS, never to the nearest row.

    The document holds a DIFFERENT closure — the shape a shipped key set takes
    the moment the endpoint's code, its shape ladder or its checkpoint moves.
    With no deriver injected — which is how pgw#1328's adopt-only role states
    its posture — the pod states no key at all.
    """
    stale = keyset.parse_closure_digest("dead" + "0" * 28)
    shipped = _ship(tmp_path / "image", stale, _row(denoiser=HASH_A))
    monkeypatch.setattr(
        cell_resolve, "resolve_batch",
        lambda *a, **k: pytest.fail("no key was stated; nothing may be asked"))

    (out,) = boot_adopt.attempt(
        function=FUNCTION, modules=MODULES, cfg=_Cfg(), slots={},
        declared_hint=1, work_root=tmp_path / "work",
        cache_dir=tmp_path / "cache", memo_dir=tmp_path / "cache",
        base_url="http://hub.local", bearer="jwt",
        keyset_roots=(shipped.parent,), derive=None)

    assert out.reason == "keyset_absent"
    assert not out.adopted
    assert out.derived_key == ""
    assert out.reason in boot_adopt.REASONS


def test_a_drifted_hash_folds_to_a_key_the_hub_misses_and_arms_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_export: _ExportSentinel,
    no_children: None,
) -> None:
    """The other half of fail-safe: a document that ADDRESSES correctly but
    carries a hash for a graph this code no longer produces asks for a key the
    hub does not hold. A miss, and nothing is armed — never a wrong cell."""
    digest = _digest()
    shipped = _ship(tmp_path / "image", digest, _row(denoiser=HASH_B))
    only_a = str(keyset.fold_entry_keys(
        _hashes(denoiser=HASH_A), family=FAMILY)[
            keyset.GraphClassName("denoiser")])

    def _resolve(family: str, keys: Any, **_kw: Any) -> Any:
        return tuple(
            cell_resolve.ResolveAnswer(
                compiled_graph_key=str(key),
                status="hit" if str(key) == only_a else "miss",
                cell=cast(Any, _Cell(str(key))) if str(key) == only_a else None)
            for key in keys)

    monkeypatch.setattr(cell_resolve, "resolve_batch", _resolve)
    (out,) = boot_adopt.attempt(
        function=FUNCTION, modules=MODULES, cfg=_Cfg(), slots={},
        declared_hint=1, work_root=tmp_path / "work",
        cache_dir=tmp_path / "cache", memo_dir=tmp_path / "cache",
        base_url="http://hub.local", bearer="jwt",
        keyset_roots=(shipped.parent,), derive=None)

    assert out.reason == "miss"
    assert not out.adopted
    assert out.key_source is keyset.KeySource.SHIPPED


def test_a_malformed_shipped_document_refuses_loudly_rather_than_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_export: _ExportSentinel,
    no_children: None,
) -> None:
    """A broken key set is a MINT-LANE defect. Degrading it to 'absent' is how
    every pod in a release silently traces forever with the artifact beside
    it."""
    root = tmp_path / "image"
    root.mkdir(parents=True)
    (root / keyset.KEYSET_FILENAME).write_text(
        '{"schema": "cg-keyset-v1", "version": 9, "closures": {}}')

    (out,) = boot_adopt.attempt(
        function=FUNCTION, modules=MODULES, cfg=_Cfg(), slots={},
        declared_hint=1, work_root=tmp_path / "work",
        cache_dir=tmp_path / "cache", memo_dir=tmp_path / "cache",
        base_url="http://hub.local", bearer="jwt",
        keyset_roots=(root,), derive=None)

    assert out.reason == "keyset_invalid"
    assert "cg-keyset-v1" in out.detail


def test_the_closure_moves_when_the_declaration_moves() -> None:
    """The address is the whole safety story, so assert it actually binds."""
    base = _digest()
    wider = keyset.closure_of(
        family=FAMILY, function=FUNCTION, modules=MODULES,
        cfg=CompileSpec(
            shapes=((64, 64), (128, 128)), targets=("transformer",),
            family=FAMILY, lora_bucket=0, guidance_scales=(), text_lens=(8,)),
        slots={})
    other_fn = keyset.closure_of(
        family=FAMILY, function="other", modules=MODULES, cfg=_spec(), slots={})
    assert len({str(base), str(wider), str(other_fn)}) == 3


# ---------------------------------------------------------------------------
# 4. The cache is the same document, and the tracer is unreachable
# ---------------------------------------------------------------------------


def test_this_machines_cache_answers_under_its_own_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_export: _ExportSentinel,
    no_children: None,
) -> None:
    """§4.28's compile-once-run-forever is the SAME document out of the pod's
    own cache — read after the shipped roots and labelled as itself."""
    digest = _digest()
    cache = tmp_path / "cache"
    keyset_store.write_closure(cache, digest, _row(denoiser=HASH_A))
    monkeypatch.setattr(
        cell_resolve, "resolve_batch",
        lambda family, keys, **k: tuple(
            cell_resolve.ResolveAnswer(
                compiled_graph_key=str(key), status="miss", cell=None)
            for key in keys))

    (out,) = boot_adopt.attempt(
        function=FUNCTION, modules=MODULES, cfg=_Cfg(), slots={},
        declared_hint=1, work_root=tmp_path / "work",
        cache_dir=cache, memo_dir=cache,
        base_url="http://hub.local", bearer="jwt",
        keyset_roots=(tmp_path / "no-image",), derive=None)

    assert out.key_source is keyset.KeySource.MEMO
    assert out.reason == "miss"


def test_a_shipped_root_wins_over_this_machines_cache(tmp_path: Path) -> None:
    digest = _digest()
    cache = tmp_path / "cache"
    keyset_store.write_closure(cache, digest, _row(denoiser=HASH_B))
    shipped = _ship(tmp_path / "image", digest, _row(denoiser=HASH_A))
    hit = keyset_store.lookup(
        digest, cache_dir=cache, extra_roots=(shipped.parent,))
    assert hit is not None
    assert hit.source is keyset.KeySource.SHIPPED
    assert hit.closure.class_hashes == {"denoiser": HASH_A}


def test_the_serve_boot_key_path_cannot_reach_a_tracer() -> None:
    """The static half of the same claim as the sentinel above: a runtime test
    proves one boot did not export; this proves no boot can.

    pgw#1328 SUPERSEDED the narrow fence this used to read
    (`lint_serve_keyset_closure`, whose own docstring nominated its
    replacement). The claim is unchanged and is asserted here against the
    successor's declarations, so pgw#1327's guarantee keeps its own test
    instead of being folded into somebody else's suite: the boot-adopt and
    keyset roots must still be present, and the three tracer modules must
    still be banned.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    import lint_serve_role_closure as fence

    roots = fence._declared_tuple("SERVE_ROLE_MODULES")
    banned = fence._declared_tuple("MINT_MACHINERY")
    for root in ("gen_worker.boot_adopt", "gen_worker.keyset"):
        assert root in roots, f"{root} left the serve role's declared set"
    for tracer in (
        "gen_worker.boot_key", "gen_worker.boot_trace_child",
        "gen_worker.keyset.emit",
    ):
        assert tracer in banned, f"{tracer} stopped being banned"
    seen, via, _, _ = fence.closure(roots)
    assert len(seen) > 20, "the roots resolved to nothing — the fence rotted"
    reached = sorted(name for name in banned if name in seen)
    assert not reached, (
        f"the serve-boot key path reaches {reached} "
        f"(via {[via.get(n) for n in reached]})")
    # THE GATE'S OWN GREEN, through the entry point CI invokes — a detector
    # disconnected from `main()` would pass every assertion above and guard
    # nothing (the th#1820 shape this repo has already been bitten by).
    assert fence.main([]) == 0


def test_the_tracer_fence_actually_fires() -> None:
    """THE GATE'S OWN RED. A root that legitimately reaches the tracer must
    make the check non-zero; if it does not, the fence is decoration."""
    sys.path.insert(0, str(REPO / "scripts"))
    import lint_serve_role_closure as fence

    banned = fence._declared_tuple("MINT_MACHINERY")
    problems = fence.check(
        ("gen_worker.boot_adopt", "gen_worker.boot_trace_child"), banned)
    assert problems, "a root that IS a tracer produced no violation"
    assert any("boot_trace_child" in line for line in problems)


def test_boot_adopts_vocabulary_carries_every_key_set_terminus() -> None:
    """pgw#1116's rule, applied to the tokens pgw#1327 adds: a refusal the
    vocabulary does not carry is a refusal nobody can count."""
    for token in (
        "keyset_absent", "keyset_invalid", "closure_unavailable",
        "no_runtime_sm",
    ):
        assert token in boot_adopt.REASONS


def test_the_deriver_is_an_argument_and_not_a_module_lookup() -> None:
    """The structural half of pgw#1327: an adopt-only role states its posture by
    passing no deriver, and there is deliberately NO env knob for it — the role
    the worker already declares on the wire (`process_role`, pgw#1309) is the
    one answer to "may this pod compile", and pgw#1328 is where it becomes the
    deciding one."""
    import inspect

    signature = inspect.signature(boot_adopt.attempt)
    assert "derive" in signature.parameters
    assert signature.parameters["derive"].default is None
    assert not hasattr(boot_adopt, "adopt_only"), (
        "a second answer to 'may this pod compile', beside the declared role")


# ---------------------------------------------------------------------------
# 5. THE ROUND TRIP, on the real vehicle: what the mint lane derives is what a
#    fresh pod reads, key for key, with no exporter on the second boot
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def micro_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("torch")
    if str(MICRO_SRC) not in sys.path:
        sys.path.insert(0, str(MICRO_SRC))
    from micro_diffusion.weights import SEED, materialize

    return materialize(tmp_path_factory.mktemp("micro-tree"), seed=SEED)


@pytest.fixture
def micro_declaration(micro_tree: Path) -> None:
    """The export declaration, restored. On a pod the import IS the
    registration; in a suite several files empty the process-global registry
    and `micro_diffusion.main` is already in `sys.modules`, so nothing
    re-registers. Restated rather than depended on."""
    from gen_worker.api import export_contract as ec

    import micro_diffusion.aot_declaration as decl

    ec.register_export_declaration(decl.DECLARATION, replace=True)


def test_the_minted_key_set_is_the_key_set_a_fresh_pod_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    micro_tree: Path, micro_declaration: None,
) -> None:
    """THE round trip, on the vehicle three real pods ran.

    Leg 1 (the MINT LANE): the real ``boot_key.derive`` — structure-only
    ``torch.export`` trace children on fake tensors, no compile, no ``.so``, no
    publish — records a ``cg-keyset-v1`` closure.

    Leg 2 (a FRESH POD): the recorded document is staged as a shipped root, the
    cache is thrown away, and ``boot_adopt.attempt`` runs with the exporter and
    the process spawner both booby-trapped. It must produce the IDENTICAL key
    set.

    Equality of the two key sets is the whole claim of pgw#1327: the shipped
    data is not an approximation of what the traces would have said, it is the
    same value.
    """
    from gen_worker import aot_declaration, boot_key
    from gen_worker.api.binding import ModelRef
    from gen_worker.api.export_contract import export_declaration
    from gen_worker.child_contract import MintSlot
    from gen_worker.registry import collect_endpoints

    monkeypatch.setenv(
        "PYTHONPATH", ":".join([str(REPO / "src"), str(MICRO_SRC)]))
    specs = collect_endpoints(["micro_diffusion.main"])
    spec = next(s for s in specs if s.name == FUNCTION)
    cfg = spec.compile_cell()
    assert cfg is not None
    compile_spec = CompileSpec(
        shapes=tuple(tuple(int(v) for v in row) for row in (cfg.shapes or ())),
        targets=tuple(str(t) for t in (cfg.targets or ())),
        family=str(cfg.family),
        lora_bucket=int(cfg.lora_bucket or 0),
        guidance_scales=tuple(float(v) for v in (cfg.guidance_scales or ())),
        text_lens=tuple(int(v) for v in (cfg.text_lens or ())),
    )
    slots = {"pipeline": MintSlot(
        ref=ModelRef(source="tensorhub", path="cozy/micro-diffusion",
                     release="prod"),
        path=str(micro_tree), facts=_TEST_FACTS)}
    declaration = export_declaration(str(cfg.family))
    assert declaration is not None

    mint_cache = tmp_path / "mint-cache"
    minted = boot_key.derive(
        function=FUNCTION,
        modules=MODULES,
        family=str(cfg.family),
        cfg=compile_spec,
        slots=slots,
        declared_hint=len(list(aot_declaration.cell_plans(declaration))),
        work_root=tmp_path / "trace",
        memo_dir=mint_cache,
        emitted_by="pgw#1327 round-trip",
    )
    assert minted.source is keyset.KeySource.TRACED
    assert minted.traced >= 1

    # Stage what the mint lane produced exactly as an image bake would: copy the
    # one small JSON. Nothing is recomputed, reformatted or re-signed.
    shipped = tmp_path / "image"
    shipped.mkdir()
    (shipped / keyset.KEYSET_FILENAME).write_bytes(
        (mint_cache / keyset.KEYSET_FILENAME).read_bytes())

    asked: List[str] = []
    monkeypatch.setattr(cell_resolve, "resolve_batch", _hub(asked))
    monkeypatch.setattr(
        cell_resolve, "materialize", lambda cell, **_kw: tmp_path / "artifact")

    sentinel = _ExportSentinel()
    torch = sys.modules.get("torch")
    if torch is not None:
        monkeypatch.setattr(torch, "export", sentinel, raising=False)
    monkeypatch.setitem(sys.modules, "torch.export", sentinel)
    import subprocess

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail(
        "a fresh pod with a shipped key set spawned a trace child"))

    outcomes = boot_adopt.attempt(
        function=FUNCTION, modules=MODULES, cfg=cfg, slots=slots,
        declared_hint=99, work_root=tmp_path / "work2",
        cache_dir=tmp_path / "fresh-cache", memo_dir=tmp_path / "fresh-cache",
        base_url="http://hub.local", bearer="jwt",
        keyset_roots=(shipped,), derive=_Deriver())

    assert not sentinel.touched
    assert {o.key_source for o in outcomes} == {keyset.KeySource.SHIPPED}
    assert sorted(asked) == sorted(str(k) for k in minted.keys), (
        "the shipped key set must fold to the SAME cg-key-v1 values the traces "
        "produced — not merely to a plausible set")
    assert len(outcomes) == len(minted.entry_keys)
