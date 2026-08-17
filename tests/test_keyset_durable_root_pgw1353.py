"""pgw#1353: the 805 s key set survives the pod that paid for it.

The measured defect, on four independent sdxl pods: ``keys_from=traced``,
778-833 s of ``torch.export`` per cold boot, forever. pgw#1327 shipped the
document that deletes that cost and offered a fleet pod exactly two ways to read
one — neither of which reaches it:

* the BAKED image root cannot be produced at all. The closure digest binds the
  checkpoint ref (``keyset.closure``), and for an sdxl-shaped
  ``Slot(selected_by=...)`` that ref is a deploy-time hub pick the image build
  cannot state;
* the pod's own cache is ``tensorhub_cache_dir or tempfile.gettempdir()``, and
  the hub deliberately leaves ``TENSORHUB_CACHE_DIR`` unset (th#850). **The
  document is written to /tmp and thrown away with the pod.**

So the document goes on the root the platform ALREADY places for exactly this
class of artifact: th#1813's ``GEN_WORKER_LOCAL_CELLS_DIR``, the endpoint's
network volume when one is attached (shared by every pod of the endpoint) and
the pod's volume disk otherwise.

What is proved here:

1. **THE HEADLINE — pod 1 traces and pod 2 does not.** Two ``boot_adopt.attempt``
   calls with DISJOINT caches and one shared durable root, driven through the
   real store. Pod 1's deriver runs once and its document lands on the volume;
   pod 2 boots with an armed ``torch.export`` sentinel, an armed
   ``subprocess.Popen`` sentinel and a deriver that fails the test if reached.
2. **The evidence is queryable**: pod 2 reports ``keys_from=durable``, which is
   its own ``KeySource`` member precisely so the fleet can count it.
3. **Read order**: shipped beats durable beats memo.
4. **RED — the root is an optimization, not a gate.** No root placed, an
   unwritable root, and a root holding a document for a different closure each
   fall through to the deriver and BOOT.
5. **RED — a corrupt document still refuses loudly** (``keyset_invalid``,
   pgw#1327's rule), rather than degrading to a silent re-trace.
6. **A proven-dishonest row is dropped from BOTH roots**, and the honesty audit
   reads the root the pod actually answered from — otherwise a bad row survives
   on a volume every pod of the endpoint reads.
7. **Concurrent writers cannot tear the document** — the shared temp name this
   replaces could interleave two encodes into one file on a shared volume.

Only row 1's pod-1 leg runs a deriver, and it is a spy rather than a tracer: what
is under test is the STORE, and a real ``torch.export`` here would prove nothing
this file claims. The real-tracer round trip is pgw#1327's row 5.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, cast

import pytest

from gen_worker import boot_adopt, cell_resolve, compile_cache as cc, keyset
from gen_worker import local_cell_store
from gen_worker.child_contract import CompileSpec
from gen_worker.keyset import document as doc_mod, store as keyset_store

REPO = Path(__file__).resolve().parent.parent
MICRO_SRC = REPO / "examples" / "micro-diffusion" / "src"

FAMILY = "micro-diffusion"
FUNCTION = "generate"
MODULES: Tuple[str, ...] = ("micro_diffusion.main",)

HASH_A = "a1b2c3d4e5f60718"
HASH_B = "0f1e2d3c4b5a6978"
HASH_C = "1234567890abcdef"
INGRESS = "9" * 32


@pytest.fixture(autouse=True)
def _micro_on_path() -> Iterator[None]:
    added = str(MICRO_SRC)
    if added not in sys.path:
        sys.path.insert(0, added)
    yield


@pytest.fixture(autouse=True)
def _stable_sm(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI has no card, and the fold needs an ``sm``."""
    monkeypatch.setattr(
        cc, "runtime_key", lambda: {"sm": "sm_89", "torch": "", "triton": ""})


@pytest.fixture(autouse=True)
def _no_inherited_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """The developer running this may have a real root placed. Every test states
    its own, so none of them can pass by reading somebody's laptop."""
    monkeypatch.delenv(keyset_store.ENV_LOCAL_CELLS_DIR, raising=False)
    monkeypatch.delenv(keyset_store.ENV_KEYSET_PATH, raising=False)


def _hashes(**classes: str) -> Dict[keyset.GraphClassName, keyset.ClassHash]:
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
        emitted_by="pgw#1353 fixture",
    )


def _place(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Place the durable root the way th#1813's hub does — as a PATH in the env.

    Returns the key-set subtree, which is what the store owns; the caller's
    ``root`` is the mount, and ``aot-cells/`` is the cell store's neighbour in
    it.
    """
    monkeypatch.setenv(keyset_store.ENV_LOCAL_CELLS_DIR, str(root))
    subtree = root / keyset_store.DURABLE_KEYSET_DIRNAME
    assert keyset_store.durable_root() == subtree
    return subtree


def _write(root: Path, digest: keyset.ClosureDigest, row: doc_mod.ClosureRow) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / keyset.KEYSET_FILENAME
    path.write_bytes(keyset.encode(keyset.KeySetDocument(
        schema=keyset.KEYSET_SCHEMA, version=keyset.KEYSET_VERSION,
        closures={str(digest): row})))
    return path


class _ExportSentinel:
    """Any attribute touch is an export attempt."""

    def __init__(self) -> None:
        self.touched = False

    def __getattr__(self, name: str) -> Any:
        self.touched = True
        raise AssertionError(
            f"pgw#1353: the boot path reached torch.export.{name} — a durable "
            f"key set costs one JSON parse, not a trace")


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
    import subprocess

    def _refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            f"pgw#1353: the boot path spawned a process ({args[:1]!r}) — the "
            f"whole point of the durable root is that pod 2 spawns none")

    monkeypatch.setattr(subprocess, "Popen", _refuse)


class _TracingDeriver:
    """Pod 1's mint-lane fallback: records the closure exactly as the real
    ``boot_key.derive`` does, through the real ``keyset.store`` writer."""

    def __init__(self, **classes: str) -> None:
        self.calls = 0
        self.classes = classes

    def __call__(self, **kwargs: Any) -> keyset.DerivedKeySet:
        self.calls += 1
        digest = keyset.closure_of(
            family=kwargs["family"], function=kwargs["function"],
            modules=kwargs["modules"], cfg=kwargs["cfg"], slots=kwargs["slots"])
        memo_dir = kwargs.get("memo_dir")
        keyset_store.write_closure(memo_dir, digest, _row(**self.classes))
        return keyset.DerivedKeySet(
            entry_keys=keyset.fold_entry_keys(
                _hashes(**self.classes), family=kwargs["family"]),
            source=keyset.KeySource.TRACED,
            closure=digest,
            wall_ms=804_700,
        )


class _RefusingDeriver:
    """Pod 2's deriver. Reaching it at all is the regression."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, **kwargs: Any) -> keyset.DerivedKeySet:
        self.calls += 1
        raise AssertionError(
            "pgw#1353: pod 2 re-derived even though pod 1 left the document on "
            "the durable root — this is the 805 s the issue exists to delete")


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


def _boot(
    tmp_path: Path, pod: str, deriver: Any, monkeypatch: pytest.MonkeyPatch,
) -> Tuple[boot_adopt.BootAdoptOutcome, ...]:
    """One pod's boot-adopt, through the REAL ``boot_adopt.attempt``.

    Each pod gets its own cache dir, which is the honest model of a fresh
    rental: nothing but the durable root is shared between two of these.
    """
    cache = tmp_path / pod / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cell_resolve, "resolve_batch", _hub([]))
    monkeypatch.setattr(
        cell_resolve, "materialize", lambda cell, **_kw: tmp_path / "artifact.pt2")
    return boot_adopt.attempt(
        function=FUNCTION, modules=MODULES, cfg=_Cfg(), slots={},
        declared_hint=2, work_root=tmp_path / pod / "work",
        cache_dir=cache, memo_dir=cache,
        base_url="http://hub.local", bearer="jwt", derive=deriver)


# ---------------------------------------------------------------------------
# 1-2. THE HEADLINE: pod 1 traces, pod 2 reads, and the event says which
# ---------------------------------------------------------------------------


def test_pod_two_reads_what_pod_one_derived_and_spawns_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """THE acceptance row, in the shape the fleet runs it.

    Pod 1: empty cache, empty volume — derives, and the document lands on the
    volume. Pod 2: a DIFFERENT empty cache, the same volume — reads it, and no
    exporter and no child process is reachable.

    Red-provable three ways: drop the durable root from ``writable_roots`` and
    pod 2 hits ``_RefusingDeriver``; drop it from the READ leg only and the same;
    report the hit as ``MEMO`` and the ``keys_from`` assertion fails.
    """
    volume = _place(tmp_path / "runpod-volume", monkeypatch)
    assert not volume.exists(), "the volume starts empty, like a new endpoint's"

    tracing = _TracingDeriver(denoiser=HASH_A, vae=HASH_B)
    first = _boot(tmp_path, "pod1", tracing, monkeypatch)

    assert tracing.calls == 1, "pod 1 is the pod that pays; it must have traced"
    assert {o.key_source for o in first} == {keyset.KeySource.TRACED}
    document = volume / keyset.KEYSET_FILENAME
    assert document.is_file(), (
        "pod 1's derivation must land on the DURABLE root, not only in the "
        "container-local cache it is about to lose")

    # ── the pod is destroyed; only the volume survives ────────────────────
    with _no_export_and_children(monkeypatch):
        refusing = _RefusingDeriver()
        second = _boot(tmp_path, "pod2", refusing, monkeypatch)

    assert refusing.calls == 0, (
        "pod 2 must not re-derive — that is the 805 s this issue deletes")
    assert {o.key_source for o in second} == {keyset.KeySource.DURABLE}, (
        "the boot event must SAY the keys came off the durable root; "
        "'how many pods still trace' has to be a query, not an inference")
    assert [o.derived_key for o in second] == [o.derived_key for o in first], (
        "same closure, same document, same folded keys — a durable read that "
        "produced different keys would arm graphs pod 1 never minted")
    assert all(o.reason == boot_adopt.HIT for o in second)


class _no_export_and_children:
    """The two sentinels as a context manager, so one test can arm them for
    HALF of itself: pod 1 legitimately derives, pod 2 may not."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.monkeypatch = monkeypatch
        self.sentinel = _ExportSentinel()

    def __enter__(self) -> "_no_export_and_children":
        import subprocess

        torch = sys.modules.get("torch")
        if torch is not None:
            self.monkeypatch.setattr(torch, "export", self.sentinel, raising=False)
        self.monkeypatch.setitem(sys.modules, "torch.export", self.sentinel)

        def _refuse(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError(
                f"pgw#1353: pod 2 spawned a process ({args[:1]!r})")

        self.monkeypatch.setattr(subprocess, "Popen", _refuse)
        return self

    def __exit__(self, *exc: Any) -> None:
        assert not self.sentinel.touched


def test_the_durable_hit_is_its_own_key_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``durable`` is not a flavour of ``memo``.

    On a fleet pod the pod-local cache is ``/tmp`` and a fresh pod's is empty, so
    ``keys_from=memo`` is very nearly a contradiction there. Reporting a durable
    hit as ``memo`` would make it indistinguishable from the impossible case and
    the one number this issue moves unreadable off the boot events.
    """
    volume = _place(tmp_path / "vol", monkeypatch)
    digest = _digest()
    _write(volume, digest, _row(denoiser=HASH_A))

    hit = keyset_store.lookup(digest, cache_dir=tmp_path / "empty")
    assert hit is not None
    assert hit.source is keyset.KeySource.DURABLE
    assert hit.source.value == "durable"
    assert hit.path == volume / keyset.KEYSET_FILENAME


# ---------------------------------------------------------------------------
# 3. Read order
# ---------------------------------------------------------------------------


def test_shipped_beats_durable_beats_memo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One closure, three roots, three DIFFERENT class sets — so the assertion
    reads which root answered rather than merely that one did."""
    digest = _digest()
    volume = _place(tmp_path / "vol", monkeypatch)
    cache = tmp_path / "cache"
    image = tmp_path / "image"

    _write(cache, digest, _row(denoiser=HASH_C))
    memo = keyset_store.lookup(digest, cache_dir=cache)
    assert memo is not None and memo.source is keyset.KeySource.MEMO

    _write(volume, digest, _row(denoiser=HASH_B))
    durable = keyset_store.lookup(digest, cache_dir=cache)
    assert durable is not None and durable.source is keyset.KeySource.DURABLE

    _write(image, digest, _row(denoiser=HASH_A))
    monkeypatch.setenv(keyset_store.ENV_KEYSET_PATH, str(image))
    shipped = keyset_store.lookup(digest, cache_dir=cache)
    assert shipped is not None and shipped.source is keyset.KeySource.SHIPPED, (
        "the mint lane's own document must still outrank a root any pod of the "
        "endpoint can write")


def test_a_durable_document_for_another_closure_is_not_a_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drift fails safe on the durable root exactly as on a shipped one: the
    volume outlives RELEASES, so a stale closure sitting on it is the normal
    steady state, not an anomaly."""
    volume = _place(tmp_path / "vol", monkeypatch)
    stale = keyset.parse_closure_digest("dead" + "0" * 28)
    _write(volume, stale, _row(denoiser=HASH_A))

    assert keyset_store.lookup(_digest(), cache_dir=tmp_path / "empty") is None


# ---------------------------------------------------------------------------
# 4. RED — an optimization, never a gate
# ---------------------------------------------------------------------------


def test_no_root_placed_still_boots_by_deriving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """th#1813 places no root on trusted hardware and none where the provider
    has no durable storage. Both must boot exactly as they do today."""
    assert keyset_store.durable_root() is None
    assert keyset_store.writable_roots(None) == ()

    tracing = _TracingDeriver(denoiser=HASH_A)
    outcomes = _boot(tmp_path, "rootless", tracing, monkeypatch)

    assert tracing.calls == 1
    assert {o.key_source for o in outcomes} == {keyset.KeySource.TRACED}
    assert all(o.reason == boot_adopt.HIT for o in outcomes)


def test_an_unwritable_durable_root_still_boots_and_keeps_the_local_memo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A read-only volume must not cost this pod its own cache write, and must
    not fail the boot. One root failing is a slower next boot, never a wrong
    key."""
    mount = tmp_path / "ro-mount"
    (mount / keyset_store.DURABLE_KEYSET_DIRNAME).mkdir(parents=True)
    (mount / keyset_store.DURABLE_KEYSET_DIRNAME).chmod(0o500)
    monkeypatch.setenv(keyset_store.ENV_LOCAL_CELLS_DIR, str(mount))
    cache = tmp_path / "cache"

    try:
        wrote = keyset_store.write_closure(cache, _digest(), _row(denoiser=HASH_A))
        assert wrote is True, "the cache must still take it"
        assert (cache / keyset.KEYSET_FILENAME).is_file()
        hit = keyset_store.lookup(_digest(), cache_dir=cache)
        assert hit is not None and hit.source is keyset.KeySource.MEMO
    finally:
        (mount / keyset_store.DURABLE_KEYSET_DIRNAME).chmod(0o700)


# ---------------------------------------------------------------------------
# 5. RED — a corrupt document still refuses LOUDLY
# ---------------------------------------------------------------------------


def test_a_corrupt_durable_document_refuses_rather_than_retracing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1327's rule, extended to the new root and not weakened for it.

    Reading past a malformed document is how every pod of an endpoint silently
    re-traces forever with a broken artifact sitting on the volume beside it. It
    is a defect and it must be visible as itself.
    """
    volume = _place(tmp_path / "vol", monkeypatch)
    volume.mkdir(parents=True, exist_ok=True)
    (volume / keyset.KEYSET_FILENAME).write_bytes(b"{not json at all")

    with pytest.raises(keyset.KeySetError) as caught:
        keyset_store.lookup(_digest(), cache_dir=tmp_path / "empty")
    assert caught.value.reason == "keyset_invalid"

    # BOTH refusal shapes, because they travel different lines: bytes that do
    # not DECODE raise out of `read_document`, while a document that decodes and
    # carries a bad row raises out of `parse_closure`. A guard that only caught
    # one would let the other degrade to a silent 805 s re-trace.
    (volume / keyset.KEYSET_FILENAME).write_bytes(
        b'{"schema":"cg-keyset-v1","version":9999,"closures":{}}')
    with pytest.raises(keyset.KeySetError) as unversioned:
        keyset_store.lookup(_digest(), cache_dir=tmp_path / "empty")
    assert unversioned.value.reason == "keyset_invalid"

    tracing = _TracingDeriver(denoiser=HASH_A)
    (out,) = _boot(tmp_path, "corrupt", tracing, monkeypatch)
    assert out.reason == "keyset_invalid", (
        "the boot states the defect under its own name; it does not report a "
        "miss and quietly pay 805 s")
    assert tracing.calls == 0


# ---------------------------------------------------------------------------
# 6. A dishonest row dies on EVERY root
# ---------------------------------------------------------------------------


def test_a_dishonest_row_is_invalidated_on_the_volume_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``assert_honest`` is the mint lane's gate on a stored key set. A row it
    proves wrong surviving on the shared volume would re-poison every pod of the
    endpoint on its next boot — and this pod would have no reason to look."""
    volume = _place(tmp_path / "vol", monkeypatch)
    cache = tmp_path / "cache"
    digest = _digest()
    keyset_store.write_closure(cache, digest, _row(denoiser=HASH_A))
    assert (volume / keyset.KEYSET_FILENAME).is_file(), (
        "one write_closure must reach every writable root")
    assert (cache / keyset.KEYSET_FILENAME).is_file(), (
        "…and must not STOP at the first root that took it. The cache is what "
        "serves this pod's own next boot when the volume is detached or the "
        "endpoint is redeployed onto storage this pod cannot see")

    verdict = keyset_store.assert_honest(
        cache, digest, {"denoiser": {"class_hash": HASH_B}})

    assert "DISHONEST" in verdict
    assert keyset_store.lookup(digest, cache_dir=cache) is None, (
        "neither root may still answer for a closure just proven wrong")


def test_the_honesty_audit_reads_the_root_the_pod_answered_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit asks what a writable root WOULD have answered. Auditing only
    ``/tmp`` while the pod read the volume audits nothing at all."""
    volume = _place(tmp_path / "vol", monkeypatch)
    digest = _digest()
    _write(volume, digest, _row(denoiser=HASH_A))
    empty_cache = tmp_path / "empty"

    assert keyset_store.class_hashes(digest, cache_dir=empty_cache) == _hashes(
        denoiser=HASH_A)
    verdict = keyset_store.assert_honest(
        empty_cache, digest, {"denoiser": {"class_hash": HASH_B}})
    assert "DISHONEST" in verdict


# ---------------------------------------------------------------------------
# 7. Concurrency, and the cross-repo name
# ---------------------------------------------------------------------------


def test_concurrent_writers_cannot_tear_the_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A network volume is mounted by SEVERAL pods of one endpoint at once —
    that is the whole reason it is worth writing to.

    The fixed ``cg-keyset-v1.tmp`` this replaces is a name two pods would write
    to simultaneously, interleaving two encodes into one file. Driven with real
    threads through the real writer; every reader must see a parseable document
    and the last writer's row.
    """
    import threading

    volume = _place(tmp_path / "vol", monkeypatch)
    digests = [
        keyset.parse_closure_digest(f"{index:032x}") for index in range(1, 9)]
    errors: List[BaseException] = []

    def _writer(digest: keyset.ClosureDigest) -> None:
        try:
            for _ in range(12):
                keyset_store.write_closure(None, digest, _row(denoiser=HASH_A))
                keyset_store.read_document(volume / keyset.KEYSET_FILENAME)
        except BaseException as exc:  # noqa: BLE001 — reported, not swallowed
            errors.append(exc)

    threads = [threading.Thread(target=_writer, args=(d,)) for d in digests]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"a concurrent read saw a torn document: {errors[:2]}"
    document = keyset_store.read_document(volume / keyset.KEYSET_FILENAME)
    assert document.closures, "the document must not end up empty"
    stray = [p.name for p in volume.iterdir() if p.name.endswith(".tmp")]
    assert not stray, f"temp files leaked: {stray}"


def test_the_document_is_recorded_before_the_machine_dependent_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete set of traces must not be thrown away by a failing fold.

    The document is machine-INDEPENDENT — TCG class hashes, nothing folded —
    while ``fold_entry_keys`` restates this process's ``sm`` and refuses
    ``no_runtime_sm`` when it cannot read one. With the fold ordered first, a
    producer that traced every class correctly on a box with no usable card
    discarded all of it at the last step, which is how
    ``scripts/emit_cg_keyset.py --derive`` was found unable to emit anything at
    all.

    Red-provable: move ``record_closure`` back below ``fold_entry_keys`` in
    ``boot_key.derive`` and this fails on the ordering.
    """
    import inspect

    from gen_worker import boot_key
    from gen_worker.keyset import emit as keyset_emit

    # (a) THE ORDERING ITSELF, read off the real function. Driving it would mean
    # running 36 real traces to reach the tail, so the property is asserted where
    # it lives: within `derive`'s body, the emission precedes the fold.
    body = inspect.getsource(boot_key.derive)
    assert body.count("keyset_emit.record_closure(") == 1, (
        "one emission site; a second would make the comparison below ambiguous")
    assert body.count("keyset.fold_entry_keys(") == 2, (
        "two fold sites are expected and the comparison depends on knowing it: "
        "the FIRST is the cache-hit path, which returns before any trace runs "
        "and so has no document to record, and the SECOND is the trace tail "
        "this row is about. A third would need this test rewritten, not "
        "renumbered")
    recorded = body.index("keyset_emit.record_closure(")
    traced_fold = body.rindex("keyset.fold_entry_keys(")
    assert recorded < traced_fold, (
        "boot_key.derive folds a machine-dependent key BEFORE recording the "
        "machine-independent document, so a box that cannot state an `sm` "
        "discards a complete set of traces at the last step (pgw#1353)")

    # (b) …and the emission it now precedes reaches BOTH writable roots.
    volume = _place(tmp_path / "vol", monkeypatch)
    cache = tmp_path / "cache"
    digest = _digest()
    rows = {
        "denoiser": doc_mod.GraphClassRow(
            graph_class="denoiser", class_hash=HASH_A,
            ingress_digest=INGRESS, target="transformer")}
    keyset_emit.record_closure(
        cache, digest, rows, family=FAMILY, function=FUNCTION,
        tcg_version=keyset.tcg_version(), emitted_by="pgw#1353 ordering guard")

    assert (volume / keyset.KEYSET_FILENAME).is_file(), (
        "the emission must land on both writable roots without ever folding a "
        "key — the fold is this pod's business, the document is the fleet's")
    document = keyset_store.read_document(volume / keyset.KEYSET_FILENAME)
    assert str(digest) in document.closures
    assert b"cg-key-v1" not in (volume / keyset.KEYSET_FILENAME).read_bytes(), (
        "a document that carried a folded key would be pinned to one SKU")


def test_a_rootless_write_still_reaches_the_cache_only(tmp_path: Path) -> None:
    """``write_closure(None, ...)`` with no root placed is a no-op that says so,
    rather than an exception a caller has to guard."""
    assert keyset_store.durable_root() is None
    assert keyset_store.write_closure(None, _digest(), _row(denoiser=HASH_A)) is False
    assert keyset_store.invalidate(None, _digest()) is False


def test_the_durable_root_name_matches_the_cell_store(tmp_path: Path) -> None:
    """``keyset.store`` RESTATES the env name rather than importing
    ``local_cell_store`` — that module imports the vendored TCG package, and
    reading a key set must not drag a tracer's import graph onto a serve pod.
    A restatement that drifts is a root the hub places and nothing reads."""
    assert keyset_store.ENV_LOCAL_CELLS_DIR == local_cell_store.ENV_STORE_DIR
    assert keyset_store.ENV_LOCAL_CELLS_DIR == "GEN_WORKER_LOCAL_CELLS_DIR", (
        "th#1813 owns this spelling on every create "
        "(localcompiledgraphs.EnvName); changing it goes dark on the hub")


def test_the_keyset_subtree_does_not_collide_with_the_cell_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Namespaced, so an operator reading a mount can say which tier owns what."""
    volume = _place(tmp_path / "vol", monkeypatch)
    assert volume.name == keyset_store.DURABLE_KEYSET_DIRNAME
    assert volume.name != local_cell_store.CELLS_DIRNAME
    assert volume.parent == tmp_path / "vol"


def test_an_expanduser_root_is_honoured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cozy-local exports this variable from its own ``workerEnv`` and a ``~``
    path is what an operator types there."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv(keyset_store.ENV_LOCAL_CELLS_DIR, "~/cells")
    root = keyset_store.durable_root()
    assert root is not None
    assert not str(root).startswith("~")
    assert root == tmp_path / "home" / "cells" / keyset_store.DURABLE_KEYSET_DIRNAME


def test_a_blank_env_is_the_same_as_an_absent_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hub writes ``""`` for "nothing placed" on some paths; whitespace must
    never become a root at the filesystem's own root."""
    for blank in ("", "   ", "\t"):
        monkeypatch.setenv(keyset_store.ENV_LOCAL_CELLS_DIR, blank)
        assert keyset_store.durable_root() is None
