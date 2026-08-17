"""pgw#1353 option (b) / th#2123: the key set outlives the ENDPOINT, not just the pod.

pgw#1353 row (a) put the document on th#1813's durable root and fixed every
endpoint that HAS one. An **ephemeral private deployment** has none — no network
volume, no baked document, a fresh container per pod — so it still paid the
measured 778-833 s of ``torch.export`` once per pod, forever. This is the tier
that fixes that one: the hub's own store, addressed by the same closure digest,
``GET``/``PUT /v1/worker/keysets/<closure_digest>``.

What is proved here:

1. **THE HEADLINE — pod 1 traces and UPLOADS, pod 2 reads the hub and spawns
   nothing.** Two ``boot_adopt.attempt`` calls with disjoint caches, NO durable
   root at all (the ephemeral shape), one in-memory hub between them. Pod 2 runs
   with an armed ``torch.export`` sentinel, an armed ``subprocess.Popen``
   sentinel, and a deriver that fails the test if reached.
2. **The evidence is queryable**: pod 2 reports ``keys_from=hub``, its own
   ``KeySource`` member, so "which fix is reaching the fleet" stays a query.
3. **What is uploaded is the MACHINE-INDEPENDENT document** — one closure, keyed
   by its own address, with no folded ``cg-key-v1`` anywhere in the bytes.
4. **Search order**: every local root beats the hub, and the hub is not even
   asked when one answers.
5. **RED — the tier is an optimization, never a gate.** A hub that 404s, 500s,
   times out, or answers a DIFFERENT closure than the address it was asked at
   all fall through to the deriver and BOOT.
6. **RED — a failed upload never blocks a boot.** The pod that could not publish
   still serves; only the next pod pays again.

No tracer runs anywhere in this file. What is under test is the TIER, and a real
``torch.export`` here would prove nothing this file claims.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, cast

import pytest

from gen_worker import boot_adopt, cell_resolve, compile_cache as cc, keyset
from gen_worker.child_contract import CompileSpec
from gen_worker.keyset import document as doc_mod, hub as keyset_hub
from gen_worker.keyset import store as keyset_store
from gen_worker.procsplit import broker

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
def _ephemeral_pod(monkeypatch: pytest.MonkeyPatch) -> None:
    """THE SHAPE UNDER TEST: no durable root, no shipped root.

    This is not tidying — it is the fixture. An ephemeral private deployment
    gets no ``GEN_WORKER_LOCAL_CELLS_DIR`` it can share and no baked document,
    and a developer's own laptop may have both. Every test that let one leak in
    would pass without the hub tier existing.
    """
    monkeypatch.delenv(keyset_store.ENV_LOCAL_CELLS_DIR, raising=False)
    monkeypatch.delenv(keyset_store.ENV_KEYSET_PATH, raising=False)


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
        family=FAMILY, function=FUNCTION, modules=MODULES, cfg=_spec(), slots={})


def _hashes(**classes: str) -> Dict[keyset.GraphClassName, keyset.ClassHash]:
    return {
        keyset.parse_graph_class_name(name): keyset.parse_class_hash(value)
        for name, value in classes.items()
    }


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
        emitted_by="pgw#1353b fixture",
    )


# ---------------------------------------------------------------------------
# The hub, in memory: th#2123's two routes and its write-once rule, and nothing
# else. Deliberately a REIMPLEMENTATION of the contract rather than a mock of
# this client's calls — a mock that returned whatever the client asked for
# would prove the client talks to itself.
# ---------------------------------------------------------------------------


class _Hub:
    def __init__(self) -> None:
        self.rows: Dict[str, bytes] = {}
        self.gets: List[str] = []
        self.puts: List[Tuple[str, bytes]] = []
        self.get_status = 200
        self.put_status = 0
        self.raise_on: str = ""
        self.answer_override: bytes = b""

    def request(
        self, method: str, path: str, *, base_url: str = "", bearer: str = "",
        params: Any = None, json: Any = None, timeout: float = 30.0,
    ) -> broker.HubResponse:
        if self.raise_on and self.raise_on == method:
            raise ConnectionError("hub is down")
        assert path.startswith(keyset_hub.KEYSET_PATH_PREFIX), path
        address = path[len(keyset_hub.KEYSET_PATH_PREFIX):]
        if method == "GET":
            self.gets.append(address)
            if self.answer_override:
                return broker.HubResponse(
                    status_code=200, text=self.answer_override.decode())
            if self.get_status != 200:
                return broker.HubResponse(
                    status_code=self.get_status, text='{"error":"forced"}')
            stored = self.rows.get(address)
            if stored is None:
                return broker.HubResponse(
                    status_code=404, text='{"found": false}')
            return broker.HubResponse(status_code=200, text=stored.decode())
        if method == "PUT":
            body = _dumps(json)
            self.puts.append((address, body))
            if self.put_status:
                return broker.HubResponse(
                    status_code=self.put_status, text='{"error":"forced"}')
            # th#2123's admission, in the two clauses that matter here: the
            # document must NAME the address, and the store is write-once.
            parsed = keyset.decode(body)
            if list(parsed.closures) != [address]:
                return broker.HubResponse(
                    status_code=400,
                    text='{"error":{"code":"keyset_address_mismatch"}}')
            if address in self.rows:
                return broker.HubResponse(
                    status_code=200 if self.rows[address] == body else 409,
                    text='{"stored": false}')
            self.rows[address] = body
            return broker.HubResponse(status_code=201, text='{"stored": true}')
        raise AssertionError(f"the key-set tier used {method}, which th#2123 does not serve")


def _dumps(payload: Any) -> bytes:
    return json.dumps(payload).encode("utf-8")


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> _Hub:
    fake = _Hub()
    monkeypatch.setattr(broker, "request", fake.request)
    return fake


# ---------------------------------------------------------------------------
# Sentinels: the whole claim is "pod 2 does no work", so the absence of work is
# what is armed, not the presence of an answer.
# ---------------------------------------------------------------------------


class _ExportSentinel:
    def __init__(self) -> None:
        self.touched = False

    def __getattr__(self, name: str) -> Any:
        self.touched = True
        raise AssertionError(
            f"pgw#1353b: the boot path reached torch.export.{name} — a hub key "
            f"set costs one round trip, not a trace")


class _RefusingDeriver:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, **kwargs: Any) -> keyset.DerivedKeySet:
        self.calls += 1
        raise AssertionError(
            "pgw#1353b: this pod re-derived even though the hub holds the "
            "document — this is the 805 s the issue exists to delete")


class _TracingDeriver:
    """The mint-lane fallback, recording through the REAL store writer.

    A spy, not a tracer. It does exactly what ``boot_key.derive`` does at the
    step that matters here: write the machine-INDEPENDENT row into the writable
    roots, which is where the publisher reads it back from.
    """

    def __init__(self, **classes: str) -> None:
        self.calls = 0
        self.classes = classes

    def __call__(self, **kwargs: Any) -> keyset.DerivedKeySet:
        self.calls += 1
        digest = keyset.closure_of(
            family=kwargs["family"], function=kwargs["function"],
            modules=kwargs["modules"], cfg=kwargs["cfg"], slots=kwargs["slots"])
        keyset_store.write_closure(
            kwargs.get("memo_dir"), digest, _row(**self.classes))
        return keyset.DerivedKeySet(
            entry_keys=keyset.fold_entry_keys(
                _hashes(**self.classes), family=kwargs["family"]),
            source=keyset.KeySource.TRACED,
            closure=digest,
            wall_ms=804_700,
        )


class _Cell:
    publisher_org = "org-a"
    publisher_tier = "platform"
    content_digest = "sha256:" + "ab" * 32

    def __init__(self, key: str) -> None:
        self.cell_ref = f"root/family-{FAMILY}#{key}"


def _resolve_stub(family: str, keys: Any, **_kw: Any) -> Any:
    return tuple(
        cell_resolve.ResolveAnswer(
            compiled_graph_key=str(key), status="hit",
            cell=cast(Any, _Cell(str(key))))
        for key in keys)


def _boot(
    tmp_path: Path, pod: str, deriver: Any, monkeypatch: pytest.MonkeyPatch,
    *, hub_absent: str = "",
) -> Tuple[boot_adopt.BootAdoptOutcome, ...]:
    """One pod's boot-adopt, through the REAL ``boot_adopt.attempt``.

    Each pod gets its OWN cache dir and no shared root of any kind — the honest
    model of an ephemeral private deployment, where the hub is the only thing
    two pods have in common.
    """
    cache = tmp_path / pod / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cell_resolve, "resolve_batch", _resolve_stub)
    monkeypatch.setattr(
        cell_resolve, "materialize", lambda cell, **_kw: tmp_path / "artifact.pt2")
    return boot_adopt.attempt(
        function=FUNCTION, modules=MODULES, cfg=_Cfg(), slots={},
        declared_hint=2, work_root=tmp_path / pod / "work",
        cache_dir=cache, memo_dir=cache,
        base_url="http://hub.local", bearer="jwt", hub_absent=hub_absent,
        derive=deriver)


# ---------------------------------------------------------------------------
# 1-3. THE HEADLINE, and what actually crosses the wire
# ---------------------------------------------------------------------------


def test_pod_two_reads_the_hub_and_traces_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub: _Hub,
) -> None:
    """The acceptance row for an EPHEMERAL deployment: no volume, two pods.

    Red-provable four ways: delete the hub read from ``key_set_from_data`` and
    pod 2 hits ``_RefusingDeriver``; delete the publish from ``_key_set`` and
    the hub holds nothing for pod 2 to read; report the hit as ``MEMO`` and the
    ``keys_from`` assertion fails; publish the FOLDED keys and the
    machine-independence assertion below fails.
    """
    assert keyset_store.durable_root() is None, (
        "this fixture must be the ephemeral shape — with a durable root placed, "
        "row (a) would carry the document and this test would prove nothing")

    tracing = _TracingDeriver(denoiser=HASH_A, vae=HASH_B)
    first = _boot(tmp_path, "pod1", tracing, monkeypatch)

    assert tracing.calls == 1, "pod 1 is the pod that pays; it must have traced"
    assert {o.key_source for o in first} == {keyset.KeySource.TRACED}
    assert list(hub.rows) == [str(_digest())], (
        "pod 1 traced for 805 s and did not hand the answer to the hub; every "
        "pod of this endpoint will pay it again")

    # ── the pod is destroyed. Nothing survives it but the hub ──────────────
    sentinel = _ExportSentinel()
    torch = sys.modules.get("torch")
    if torch is not None:
        monkeypatch.setattr(torch, "export", sentinel, raising=False)
    monkeypatch.setitem(sys.modules, "torch.export", sentinel)
    import subprocess

    monkeypatch.setattr(subprocess, "Popen", _no_children)

    refusing = _RefusingDeriver()
    second = _boot(tmp_path, "pod2", refusing, monkeypatch)

    assert refusing.calls == 0, "pod 2 re-derived; the hub tier did not answer"
    assert not sentinel.touched
    assert {o.key_source for o in second} == {keyset.KeySource.HUB}, (
        "pod 2 must report keys_from=hub — a distinct KeySource is what makes "
        "'how many pods still pay the 805 s' a query instead of a story")
    assert hub.gets == [str(_digest())] * 2, (
        f"the hub was asked at {hub.gets}; the address is the closure digest "
        f"and nothing else")

    # The two pods derived the SAME keys, which is the property the whole tier
    # rests on: the document is machine-independent and the fold is local.
    assert {o.derived_key for o in first} == {o.derived_key for o in second}


def _no_children(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError(
        f"pgw#1353b: the boot path spawned a process ({args[:1]!r}) — a pod "
        f"reading the hub spawns none")


def test_what_is_published_is_machine_independent_and_names_its_address(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub: _Hub,
) -> None:
    """The bytes on the wire, inspected.

    Three properties, and the middle one is the one that would be silently
    wrong: ONE closure (the address is per-closure and the hub refuses more),
    keyed by the address it is PUT at, and carrying NO folded ``cg-key-v1``
    value — those fold this pod's ``sm`` and toolchain in, and a document
    carrying them would be a wrong answer for every pod on another SKU.
    """
    tracing = _TracingDeriver(denoiser=HASH_A, vae=HASH_B)
    outcomes = _boot(tmp_path, "pod1", tracing, monkeypatch)

    assert len(hub.puts) == 1, f"expected exactly one PUT, got {hub.puts}"
    address, body = hub.puts[0]
    assert address == str(_digest())

    document = keyset.decode(body)
    assert list(document.closures) == [address], (
        "the document must name exactly the address it was PUT at; that check "
        "IS the hub's admission, and a body naming anything else is refused")
    closure = keyset.parse_closure(document, _digest())
    assert {str(c.graph_class) for c in closure.classes} == {"denoiser", "vae"}
    assert {str(c.class_hash) for c in closure.classes} == {HASH_A, HASH_B}

    folded = {str(key) for key in
              (o.derived_key for o in outcomes) if key}
    assert folded, "the fixture produced no folded keys, so this check is vacuous"
    text = body.decode()
    for key in folded:
        assert key not in text, (
            f"the published document carries the folded key {key}; the graph "
            f"axis is machine-independent and the sm/toolchain axes must be "
            f"restated by every reader")
    assert "cg-key-v1" not in text


# ---------------------------------------------------------------------------
# 4. SEARCH ORDER — every local root beats the hub, and the hub is not asked
# ---------------------------------------------------------------------------


def test_a_local_document_wins_and_the_hub_is_not_asked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub: _Hub,
) -> None:
    """shipped -> durable -> cache -> hub, and the ordering is observable.

    Not just "the right answer wins" — the hub must not even be ASKED, because
    a round trip in front of an answer this machine is already holding is pure
    added boot latency on the path this issue exists to shorten.
    """
    digest = _digest()
    for name, place in (
        ("shipped", lambda root: monkeypatch.setenv(
            keyset_store.ENV_KEYSET_PATH, str(root))),
        ("durable", lambda root: monkeypatch.setenv(
            keyset_store.ENV_LOCAL_CELLS_DIR, str(root.parent))),
    ):
        monkeypatch.delenv(keyset_store.ENV_KEYSET_PATH, raising=False)
        monkeypatch.delenv(keyset_store.ENV_LOCAL_CELLS_DIR, raising=False)
        hub.gets.clear()
        hub.rows[str(digest)] = keyset.encode(keyset.KeySetDocument(
            schema=keyset.KEYSET_SCHEMA, version=keyset.KEYSET_VERSION,
            closures={str(digest): _row(denoiser=HASH_C)}))

        root = tmp_path / name / keyset_store.DURABLE_KEYSET_DIRNAME
        root.mkdir(parents=True, exist_ok=True)
        (root / keyset.KEYSET_FILENAME).write_bytes(
            keyset.encode(keyset.KeySetDocument(
                schema=keyset.KEYSET_SCHEMA, version=keyset.KEYSET_VERSION,
                closures={str(digest): _row(denoiser=HASH_A, vae=HASH_B)})))
        place(root)

        outcomes = _boot(tmp_path, f"pod-{name}", _RefusingDeriver(), monkeypatch)
        assert {o.key_source for o in outcomes} == {
            keyset.KeySource.SHIPPED if name == "shipped" else keyset.KeySource.DURABLE
        }, f"the {name} root did not win"
        assert hub.gets == [], (
            f"the {name} root answered and the hub was asked anyway ({hub.gets}); "
            f"that is a network round trip in front of a local answer")
        assert len(outcomes) == 2, (
            "the local document declares two classes and the hub's declares "
            "one, so an answer of length 1 means the hub won after all")


# ---------------------------------------------------------------------------
# 5. RED — the tier is an OPTIMIZATION and every failure falls through to boot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arrange,why",
    [
        (lambda h: None,
         "the hub simply holds nothing — the first pod of a release"),
        (lambda h: setattr(h, "get_status", 503),
         "the hub is down or rebuilding"),
        (lambda h: setattr(h, "get_status", 401),
         "this pod's credential was refused"),
        (lambda h: setattr(h, "raise_on", "GET"),
         "the connection failed outright"),
        (lambda h: setattr(h, "answer_override", b"{not json"),
         "something in between rewrote the body"),
        (lambda h: setattr(h, "answer_override", json.dumps({
            "schema": keyset.KEYSET_SCHEMA, "version": 99, "closures": {}}
        ).encode()),
         "the hub answered a version this worker does not read"),
    ],
    ids=["miss", "503", "401", "connection-error", "garbage", "unknown-version"],
)
def test_a_hub_that_cannot_answer_never_blocks_a_boot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub: _Hub,
    arrange: Any, why: str,
) -> None:
    """Every failure mode of a network peer degrades to the derive.

    The asymmetry against a LOCAL document is deliberate and is asserted
    separately below: a malformed document in the image is a mint-lane defect
    that must be visible, while a hub is a peer that can be down — and a pod
    that refuses to boot because a cache was unreachable is strictly worse than
    the 805 s it was avoiding.
    """
    arrange(hub)
    tracing = _TracingDeriver(denoiser=HASH_A)
    outcomes = _boot(tmp_path, "pod", tracing, monkeypatch)

    assert tracing.calls == 1, f"the pod did not fall through to the deriver ({why})"
    assert {o.key_source for o in outcomes} == {keyset.KeySource.TRACED}
    assert all(o.reason in ("local_miss_no_hub", "adopted", "local_hit", "miss")
               or not o.reason.startswith("keyset_")
               for o in outcomes), (
        f"a hub fault surfaced as a KEY SET refusal ({why}); it must be a miss")


def test_an_answer_at_the_wrong_address_is_refused_not_trusted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub: _Hub,
) -> None:
    """ADMISSION (§4.29): the answer must NAME the address it was asked at.

    A hub answering a document for some OTHER closure is either a hub defect or
    something in between; either way the pod must derive rather than fold a
    stranger's class hashes into its own keys. This is the check that makes it
    safe for an untrusted pod to have written the row in the first place.
    """
    other = "f" * 32
    hub.answer_override = keyset.encode(keyset.KeySetDocument(
        schema=keyset.KEYSET_SCHEMA, version=keyset.KEYSET_VERSION,
        closures={other: _row(denoiser=HASH_C)}))

    tracing = _TracingDeriver(denoiser=HASH_A)
    outcomes = _boot(tmp_path, "pod", tracing, monkeypatch)

    assert tracing.calls == 1, (
        "the pod accepted a document addressed at a different closure")
    assert {o.key_source for o in outcomes} == {keyset.KeySource.TRACED}
    assert {str(k) for k in _hashes(denoiser=HASH_C).values()} == {HASH_C}
    for outcome in outcomes:
        assert HASH_C not in str(outcome.derived_key or ""), (
            "the stranger's class hash reached this pod's key")


def test_a_local_document_that_is_corrupt_still_refuses_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub: _Hub,
) -> None:
    """THE OTHER HALF of the asymmetry, and the reason it is not laziness.

    pgw#1327's rule is unchanged by this tier: a malformed document on a LOCAL
    root is ``keyset_invalid`` and propagates, because it is a mint-lane or
    image defect that must be visible as itself. Only the HUB degrades.
    """
    root = tmp_path / "shipped"
    root.mkdir(parents=True, exist_ok=True)
    (root / keyset.KEYSET_FILENAME).write_bytes(b"{ not a document")
    monkeypatch.setenv(keyset_store.ENV_KEYSET_PATH, str(root))

    tracing = _TracingDeriver(denoiser=HASH_A)
    outcomes = _boot(tmp_path, "pod", tracing, monkeypatch)

    assert tracing.calls == 0, (
        "a corrupt LOCAL document degraded to a derive; it must refuse, or "
        "every pod in the release silently re-traces forever beside a broken "
        "artifact")
    assert [o.reason for o in outcomes] == ["keyset_invalid"]
    assert hub.gets == [], "the hub was asked past a corrupt local document"


# ---------------------------------------------------------------------------
# 6. RED — the UPLOAD never blocks a boot either
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arrange,why",
    [
        (lambda h: setattr(h, "put_status", 503), "the hub is down"),
        (lambda h: setattr(h, "put_status", 409), "another pod stored a different document"),
        (lambda h: setattr(h, "put_status", 403), "this pod may not write"),
        (lambda h: setattr(h, "raise_on", "PUT"), "the connection failed outright"),
    ],
    ids=["503", "409", "403", "connection-error"],
)
def test_a_failed_upload_never_blocks_the_boot_that_paid_for_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub: _Hub,
    arrange: Any, why: str,
) -> None:
    """The pod that traced still serves. Only the NEXT pod pays again.

    The publish runs strictly after the keys this boot needs already exist, so
    nothing it can do is allowed to reach the boot's outcome.
    """
    arrange(hub)
    tracing = _TracingDeriver(denoiser=HASH_A, vae=HASH_B)
    outcomes = _boot(tmp_path, "pod", tracing, monkeypatch)

    assert tracing.calls == 1
    assert len(outcomes) == 2, f"the boot lost its outcomes when the upload failed ({why})"
    assert {o.key_source for o in outcomes} == {keyset.KeySource.TRACED}
    assert all(o.derived_key for o in outcomes), (
        f"the boot produced no keys after a failed upload ({why})")


def test_nothing_is_published_when_the_keys_came_from_a_document(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub: _Hub,
) -> None:
    """A pod that READ its keys has learned nothing new to publish.

    Re-uploading would be a write-once conflict at best and a wasted round trip
    at worst — and it is exactly the shape that would make every pod of a busy
    endpoint PUT on every boot.

    TWO ARMS, and the second is the one with teeth. On a HUB hit the pod's own
    writable roots hold nothing, so a missing source guard would be masked by
    the read-back finding no row. On a DURABLE hit the row IS on a writable
    root — every pod of a volume-backed endpoint would re-PUT what it just
    read — so this arm is what makes the guard load-bearing rather than
    decorative.
    """
    digest = _digest()
    hub.rows[str(digest)] = keyset.encode(keyset.KeySetDocument(
        schema=keyset.KEYSET_SCHEMA, version=keyset.KEYSET_VERSION,
        closures={str(digest): _row(denoiser=HASH_A)}))

    outcomes = _boot(tmp_path, "pod", _RefusingDeriver(), monkeypatch)

    assert {o.key_source for o in outcomes} == {keyset.KeySource.HUB}
    assert hub.puts == [], f"a hub-reading pod wrote back: {hub.puts}"

    # ── the durable arm ──────────────────────────────────────────────────
    volume = tmp_path / "volume"
    monkeypatch.setenv(keyset_store.ENV_LOCAL_CELLS_DIR, str(volume))
    subtree = volume / keyset_store.DURABLE_KEYSET_DIRNAME
    subtree.mkdir(parents=True, exist_ok=True)
    (subtree / keyset.KEYSET_FILENAME).write_bytes(
        keyset.encode(keyset.KeySetDocument(
            schema=keyset.KEYSET_SCHEMA, version=keyset.KEYSET_VERSION,
            closures={str(digest): _row(denoiser=HASH_A)})))
    hub.gets.clear()

    durable = _boot(tmp_path, "pod-vol", _RefusingDeriver(), monkeypatch)
    assert {o.key_source for o in durable} == {keyset.KeySource.DURABLE}
    assert hub.puts == [], (
        f"a pod that READ its key set off the endpoint volume PUT it back "
        f"({hub.puts}); every pod of that endpoint would do this on every boot")


def test_a_producer_read_back_that_explodes_never_blocks_the_boot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub: _Hub,
) -> None:
    """The publisher's OUTER guard, made load-bearing rather than defensive.

    ``closure_row`` reads a file off a volume this pod does not own. A mount
    that has gone away mid-boot, a permissions change, a truncated read — none
    of them are this boot's problem, because this boot already has its keys.
    """
    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise OSError("the volume went away mid-boot")

    monkeypatch.setattr(keyset_store, "closure_row", _explode)
    tracing = _TracingDeriver(denoiser=HASH_A, vae=HASH_B)
    outcomes = _boot(tmp_path, "pod", tracing, monkeypatch)

    assert tracing.calls == 1
    assert len(outcomes) == 2 and all(o.derived_key for o in outcomes), (
        "a read-back failure in the PRODUCER reached the boot's outcome")
    assert hub.puts == []


def test_a_derivers_own_memo_hit_is_published_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub: _Hub,
) -> None:
    """The publish has NO source guard, and this is why.

    ``boot_key.derive`` answers ``MEMO`` when its own cache already holds the
    closure — a cozy-local machine on its second boot, a pod whose container
    restarted. Every read tier has already missed to reach the deriver at all,
    INCLUDING the hub, so a memo-sourced document is exactly as worth
    publishing as a freshly traced one: this machine holds it and the platform
    does not. A "publish only TRACED" guard reads plausible and would strand
    every one of those.
    """
    class _MemoDeriver:
        def __call__(self, **kwargs: Any) -> keyset.DerivedKeySet:
            digest = keyset.closure_of(
                family=kwargs["family"], function=kwargs["function"],
                modules=kwargs["modules"], cfg=kwargs["cfg"], slots=kwargs["slots"])
            keyset_store.write_closure(
                kwargs.get("memo_dir"), digest, _row(denoiser=HASH_A))
            return keyset.DerivedKeySet(
                entry_keys=keyset.fold_entry_keys(
                    _hashes(denoiser=HASH_A), family=kwargs["family"]),
                source=keyset.KeySource.MEMO, closure=digest, wall_ms=3)

    outcomes = _boot(tmp_path, "pod", _MemoDeriver(), monkeypatch)

    assert {o.key_source for o in outcomes} == {keyset.KeySource.MEMO}
    assert [a for a, _ in hub.puts] == [str(_digest())], (
        "a machine holding a document the hub does not have kept it to itself")


def test_publish_closure_returns_a_reason_and_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The producer's OWN contract, asserted where it lives.

    ``boot_adopt`` wraps this call too, so a raise here is survivable — which
    is exactly why it must also be asserted HERE: a guard covered only by a
    second guard is a guard nobody can prove works, and the day the outer one
    is narrowed the boot path acquires a raise nobody added.
    """
    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise ConnectionError("hub is down")

    monkeypatch.setattr(broker, "request", _explode)
    tier = keyset_hub.HubTier(base_url="http://hub.local", bearer="jwt")
    reason = keyset_hub.publish_closure(_digest(), _row(denoiser=HASH_A), tier)
    assert reason and "ConnectionError" in reason

    # …and the READ half has the same contract, for the same reason.
    closure, why = keyset_hub.fetch_closure(_digest(), tier)
    assert closure is None and "ConnectionError" in why


def test_a_pod_with_no_hub_asks_none_and_still_boots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub: _Hub,
) -> None:
    """``hub_absent`` is the CALLER's sentence for "there is nobody to ask".

    cozy-local and an embedded worker are the real cases. The tier must make no
    call at all — not a call that fails — and the boot must be exactly what it
    was before this tier existed.
    """
    tracing = _TracingDeriver(denoiser=HASH_A)
    outcomes = _boot(
        tmp_path, "pod", tracing, monkeypatch,
        hub_absent="cozy-local runs with no hub")

    assert hub.gets == [] and hub.puts == [], (
        "a pod told there is no hub dialled one anyway")
    assert tracing.calls == 1
    assert {o.key_source for o in outcomes} == {keyset.KeySource.TRACED}


# ---------------------------------------------------------------------------
# The tier's own units — the pieces the boot test exercises only indirectly
# ---------------------------------------------------------------------------


def test_the_address_is_a_validated_type_at_the_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``keyset_path`` takes a ``ClosureDigest``, and that is not decoration.

    The path segment is the one place the hub reads as an address. A folded key
    or a class hash landing there would be stored at an address no pod ever
    looks up — a silent miss, forever, for the whole endpoint.
    """
    digest = keyset.parse_closure_digest("a" * 32)
    assert keyset_hub.keyset_path(digest) == "/v1/worker/keysets/" + "a" * 32
    for bad in ("", "A" * 32, "a" * 31, "cg-key-v1-" + "a" * 56):
        with pytest.raises(keyset.KeySetError) as caught:
            keyset.parse_closure_digest(bad)
        assert caught.value.reason == "keyset_invalid"


def test_the_published_body_is_one_closure_keyed_by_its_address() -> None:
    digest = keyset.parse_closure_digest("b" * 32)
    body = keyset_hub.single_closure_document(digest, _row(denoiser=HASH_A))
    assert list(keyset.decode(body).closures) == [str(digest)]


def test_closure_row_reads_back_what_the_deriver_wrote(tmp_path: Path) -> None:
    """The producer's own read-back, which is where the published bytes come
    from. Reading the RAW row rather than a parsed view is what keeps the
    writer and the reader from disagreeing about a field neither of them
    reads."""
    digest = _digest()
    assert keyset_store.closure_row(tmp_path, digest) is None
    keyset_store.write_closure(tmp_path, digest, _row(denoiser=HASH_A))
    row = keyset_store.closure_row(tmp_path, digest)
    assert row is not None and set(row.classes) == {"denoiser"}
    assert keyset_store.closure_row(tmp_path, keyset.parse_closure_digest("c" * 32)) is None


# ---------------------------------------------------------------------------
# THE MINT LANE'S EMITTER — the producer that makes "a serving pod never
# derives" true, rather than "a serving pod derives once per closure".
# ---------------------------------------------------------------------------


def _emitter() -> Any:
    """`scripts/emit_cg_keyset.py`, imported as a module.

    Loaded by path rather than added to `sys.path` permanently: it is an
    operator script, not an importable part of the package, and a test that
    made it importable would be testing a shape the mint lane does not have.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "emit_cg_keyset_under_test", REPO / "scripts" / "emit_cg_keyset.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_emitter_publishes_every_staged_closure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub: _Hub,
) -> None:
    """`--publish` sends the STAGED document, closure by closure.

    This is the row that makes a serving pod never derive at all: the mint lane
    pays the traces once and fills the store, instead of the first pod of every
    release paying them.
    """
    emit = _emitter()
    digest = _digest()
    cache = tmp_path / "podcache"
    keyset_store.write_closure(cache, digest, _row(denoiser=HASH_A, vae=HASH_B))
    out = tmp_path / "build"

    monkeypatch.setenv(emit.ENV_HUB_URL, "http://hub.local")
    monkeypatch.setenv(emit.ENV_HUB_TOKEN, "jwt")
    status = emit.main(
        ["--from-cache", str(cache), "--out", str(out), "--publish"])

    assert status == 0
    assert [a for a, _ in hub.puts] == [str(digest)]
    stored = keyset.decode(hub.rows[str(digest)])
    assert list(stored.closures) == [str(digest)]
    assert set(stored.closures[str(digest)].classes) == {"denoiser", "vae"}


def test_the_emitter_refuses_a_publish_it_cannot_perform(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub: _Hub,
) -> None:
    """A `--publish` with no credential REFUSES BY NAME.

    Never a silent skip. A publish flag that quietly did nothing is the exact
    failure this issue is about — a fleet paying 805 s per pod while a green
    log line says the document was emitted — and it would be invisible in a
    build log that scrolls.
    """
    emit = _emitter()
    cache = tmp_path / "podcache"
    keyset_store.write_closure(cache, _digest(), _row(denoiser=HASH_A))
    monkeypatch.delenv(emit.ENV_HUB_URL, raising=False)
    monkeypatch.delenv(emit.ENV_HUB_TOKEN, raising=False)

    with pytest.raises(SystemExit) as caught:
        emit.main(["--from-cache", str(cache), "--out",
                   str(tmp_path / "build"), "--publish"])
    assert emit.ENV_HUB_TOKEN in str(caught.value)
    assert hub.puts == []


def test_the_emitter_reports_a_publish_that_did_not_land(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub: _Hub,
) -> None:
    """A refused PUT is a NON-ZERO exit, so a build step can see it.

    A write-once conflict is the interesting case: it means two runs of the
    same code against the same subjects traced different class hashes, which is
    a finding rather than a retry.
    """
    emit = _emitter()
    hub.put_status = 409
    cache = tmp_path / "podcache"
    keyset_store.write_closure(cache, _digest(), _row(denoiser=HASH_A))
    monkeypatch.setenv(emit.ENV_HUB_URL, "http://hub.local")
    monkeypatch.setenv(emit.ENV_HUB_TOKEN, "jwt")

    status = emit.main(
        ["--from-cache", str(cache), "--out", str(tmp_path / "b"), "--publish"])
    assert status == 1, "a publish that did not land exited 0"


# ---------------------------------------------------------------------------
# THE SEAM — the layer every test above is structurally blind to.
#
# Under the process split (which is the real fleet configuration) the compute
# child holds no credential and NEVER NAMES A FREE-FORM PATH: the parent
# refuses any request that does not match `procsplit/actions.py` by method AND
# full-path regex. Every test above patches `broker.request`, so all of them
# stay green with the tier COMPLETELY DEAD on every fleet pod — the refusal
# would arrive as a `BrokerError`, `keyset.hub` would correctly read it as a
# miss, and the pod would derive for 805 s exactly as it does today.
#
# That is pgw#1309's shape precisely (its child-PID rows were merged and green
# for a full cycle before a live pod emitted one), so it is asserted against the
# TABLE rather than against this client.
# ---------------------------------------------------------------------------


def test_the_keyset_actions_are_allowlisted_pgw1353b() -> None:
    from gen_worker.procsplit import actions

    digest = _digest()
    path = keyset_hub.keyset_path(digest)

    fetch, query, body = actions.authorize({"method": "GET", "path": path})
    assert fetch.name == "keysets.fetch"
    assert query == {} and body is None

    document = keyset_hub.single_closure_document(digest, _row(denoiser=HASH_A))
    publish, _q, sent = actions.authorize({
        "method": "PUT", "path": path, "json": json.loads(document.decode())})
    assert publish.name == "keysets.publish"
    assert sent is not None and set(sent) == {"schema", "version", "closures"}

    # The timeouts the parent will hold its own control loop for must cover the
    # client's own bound, or the seam gives up before the call it authorized.
    assert fetch.timeout_s >= keyset_hub.KEYSET_TIMEOUT_S
    assert publish.timeout_s >= keyset_hub.KEYSET_TIMEOUT_S


def test_the_allowlisted_address_grammar_is_the_closure_digests_pgw1353b() -> None:
    """The path regex pins the ADDRESS SHAPE, and it must be the same one
    `parse_closure_digest` admits.

    A looser `[^/]+` would let a process running tenant code put anything it
    liked into a path segment the parent attaches the pod's credential to.
    """
    from gen_worker.procsplit import actions

    for bad in (
        "/v1/worker/keysets/" + "A" * 32,          # uppercase
        "/v1/worker/keysets/" + "a" * 31,          # short
        "/v1/worker/keysets/" + "a" * 33,          # long
        "/v1/worker/keysets/../compiled-graphs",   # traversal
        "/v1/worker/keysets/",                     # empty address
        "/v1/worker/keysets/" + "a" * 32 + "/x",   # extra segment
    ):
        for method in ("GET", "PUT"):
            with pytest.raises(actions.ActionRefused):
                actions.authorize({"method": method, "path": bad, "json": {}})

    # …and every address `parse_closure_digest` admits is one the seam allows,
    # which is the direction that would strand the tier if it were wrong.
    for good in ("0" * 32, "f" * 32, "a1b2c3d4e5f60718293a4b5c6d7e8f90"):
        digest = keyset.parse_closure_digest(good)
        action, _q, _b = actions.authorize(
            {"method": "GET", "path": keyset_hub.keyset_path(digest)})
        assert action.name == "keysets.fetch"


def test_a_keyset_publish_is_not_a_PUBLISH_ACTION_pgw1353b() -> None:
    """A key set is not a cell, and the probe disarm must not swallow it.

    pgw#980 disarms `PUBLISH_ACTIONS` on a live-edit probe because a probe
    writing a CELL into a shared family namespace poisons every pod that later
    adopts from it. A key set carries no artifact and no signature — it is a
    statement about a closure the writer can already compute — and its address
    is a pure function of the code being run, so a probe's document lands at a
    closure only that probe resolves. Including it here would disarm the tier on
    exactly the pods a dev loop uses most.
    """
    from gen_worker.procsplit import actions

    assert "keysets.publish" not in actions.PUBLISH_ACTIONS
    assert "keysets.fetch" not in actions.PUBLISH_ACTIONS


def test_the_parent_actually_SENDS_a_PUT_body_pgw1353b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parent's half of the seam, at the one line that decides it.

    `parent._http_call` is the ONLY place the worker JWT reaches the wire on a
    child's behalf, and it used to attach the body ONLY on `method == "POST"` —
    true while POST was the only verb in the action table, and silently fatal
    the moment a PUT joined it. A key-set publish would have gone out with NO
    BODY, the hub would have answered a typed `keyset_invalid`, and the child
    would have read a well-formed "no" to a request it believed it sent. Nothing
    above this test can see that: every other test in this file stops at
    `broker.request`, which is the CHILD's side.

    Asserted against the real function with `requests.request` captured, because
    the defect is a keyword argument and only the call itself carries it.
    """
    from gen_worker.procsplit import parent as procsplit_parent

    sent: List[Any] = []

    class _Resp:
        status_code = 201
        text = '{"stored": true}'

    import requests as _requests

    def _capture(method: str, url: str, **kw: Any) -> Any:
        sent.append((method, url, kw))
        return _Resp()

    monkeypatch.setattr(_requests, "request", _capture)

    digest = _digest()
    document = json.loads(
        keyset_hub.single_closure_document(digest, _row(denoiser=HASH_A)).decode())
    status, _text = procsplit_parent._http_call(
        "PUT", "http://hub.local" + keyset_hub.keyset_path(digest),
        "worker-jwt", {}, document, 10.0)

    assert status == 201
    assert len(sent) == 1
    method, url, kwargs = sent[0]
    assert method == "PUT" and url.endswith(str(digest))
    assert kwargs["json"] == document, (
        "the parent dropped the PUT body; the hub would receive an empty "
        "document and the child would read a typed refusal to a request it "
        "believes it sent")
    assert kwargs["headers"]["Authorization"] == "Bearer worker-jwt"

    # …and a GET still carries none, which is what its action declares.
    sent.clear()
    procsplit_parent._http_call(
        "GET", "http://hub.local" + keyset_hub.keyset_path(digest),
        "worker-jwt", {}, None, 10.0)
    assert sent[0][2]["json"] is None
