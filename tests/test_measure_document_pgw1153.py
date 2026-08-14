"""The documented command must work on the artifacts that exist.

``python -m gen_worker.measure_child <request>.mint.json`` is documented in
endpoint sources and in ``MintBlocker``. Every ``aot/*.mint.json`` an endpoint
repo commits is a flattened DECLARATION payload. The fence:

1. **Every committed ``aot/*.mint.json`` in the fleet parses through the real
   entry** — the corpus under ``tests/fixtures/fleet_mint_requests/`` is a
   verbatim copy of all 24 of them, and each
   one is driven through :func:`measure_child.load_document`, which is the ONE
   decoder ``main()`` uses. The corpus is a SNAPSHOT (inference-endpoints
   ``dd41755``); the half that cannot go stale is that repo's own
   ``scripts/lint_mint_requests.py``, which reads the real tree on every PR.
2. **The derivation is exercised, not asserted** — the three answers a committed
   payload does not spell out (which function, which targets, which checkpoint)
   are taken from the declaration the payload NAMES, on a real endpoint with a
   real ``@endpoint(compile=...)`` block.
3. **The documented invocation runs a real measurement end to end** — a
   committed-SHAPE payload for micro-diffusion, through ``main(argv)``, through
   the real loader and the real export loop, to a real ``MeasureReport``.
4. **Nothing the widened door admits can publish** — the new document type is
   input-side only, exactly as :class:`measure_child.MeasureJob` is.

Cardless: CPU throughout, inductor faked at the one seam it is exercised at, no
GPU, no mint, no pod, no network (slot resolution is offline by construction).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import msgspec
import pytest

from gen_worker import measure_child

REPO = Path(__file__).resolve().parent.parent
MICRO_SRC = REPO / "examples" / "micro-diffusion" / "src"
FLEET = REPO / "tests" / "fixtures" / "fleet_mint_requests"
FAMILY = "micro-diffusion"

#: Every committed request in the fleet at the time this fence was written.
#: A count, so that a fixture quietly disappearing fails here rather than
#: shrinking the corpus in silence.
FLEET_COUNT = 24

#: The committed files that name no checkpoint at all (``source_ref: ""``).
#: SHRINK-ONLY, and each one costs an operator a `--slot` flag they have to
#: source themselves — which is the reason to fill them, not a reason to widen
#: this. inference-endpoints' own `scripts/lint_mint_requests.py` holds the
#: same inventory on the artifact side.
NO_SOURCE_REF = frozenset({
    "flux.2-klein-4b/transformer-4b.mint.json",
    "flux.2-klein-9b/transformer-9b.mint.json",
})


def _committed() -> List[Path]:
    return sorted(FLEET.rglob("*.mint.json"))


# ---------------------------------------------------------------------------
# 1. THE FENCE: every committed file, through the real entry.
# ---------------------------------------------------------------------------


def test_the_corpus_is_the_whole_fleet() -> None:
    files = _committed()
    assert len(files) == FLEET_COUNT, (
        f"the corpus is {len(files)} files, not {FLEET_COUNT} — a fence that "
        f"shrinks silently is not a fence")
    assert len({p.parent.name for p in files}) == 7


@pytest.mark.parametrize(
    "path", _committed(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_committed_mint_request_parses_through_the_documented_entry(
    path: Path,
) -> None:
    """RED on the pre-fix tree, on file one.

    ``load_document`` is the decoder ``main()`` calls, so this is the actual
    first instruction of the documented command — not a re-implementation of
    it, which is the mistake that let the two shapes drift apart.
    """
    doc, flat = measure_child.load_document(path.read_bytes())

    body = json.loads(path.read_text())
    family = (doc.family or flat.family).strip()

    assert family, f"{path}: names no family, so nothing selects a declaration"
    assert family == body["family"]
    assert doc.declaration_module.strip(), (
        f"{path}: names no `declaration_module`, so the documented command has "
        f"no image to collect endpoints from (pgw#1107)")
    if f"{path.parent.name}/{path.name}" in NO_SOURCE_REF:
        assert not doc.source_ref.strip(), (
            f"{path}: names a source_ref now — take it OUT of NO_SOURCE_REF, "
            f"which is shrink-only")
    else:
        assert doc.source_ref.strip(), (
            f"{path}: names no `source_ref`, so nothing binds the slot that "
            f"owns the compile targets and every invocation of the documented "
            f"command needs a --slot flag. Fill it, or record it in "
            f"NO_SOURCE_REF with the cost stated")
    # The committed payload is the DECLARATION half: it carries none of the
    # runtime envelope, which is exactly why decoding it as one could not work.
    assert not doc.function and not doc.modules and doc.cfg is None
    assert not doc.slots


@pytest.mark.parametrize(
    "path", _committed(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_no_committed_request_can_carry_an_output_destination(
    path: Path,
) -> None:
    """pgw#1134's structural property, held at the WIDENED door.

    The reason a measure run cannot publish is that the destination never
    enters the process. Widening what the file may say is exactly the change
    that could have undone it.
    """
    doc, _flat = measure_child.load_document(path.read_bytes())
    for field in measure_child.WITHHELD_FIELDS:
        assert not hasattr(doc, field), field


def test_the_document_type_is_input_side_only() -> None:
    fields = set(measure_child.MeasureDocument.__struct_fields__)
    assert fields.isdisjoint(measure_child.WITHHELD_FIELDS)
    assert {"declaration_module", "source_ref", "family"} <= fields


# ---------------------------------------------------------------------------
# 2. THE DERIVATION, on a real declaration.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def micro_src() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    if str(MICRO_SRC) not in sys.path:
        sys.path.insert(0, str(MICRO_SRC))


@pytest.fixture(scope="module")
def w8a8_tree(
    micro_src: None, tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    from micro_diffusion.weights import SEED, materialize_w8a8

    return materialize_w8a8(
        tmp_path_factory.mktemp("micro-1153") / "w8a8", seed=SEED)


MODULES = ("harness.rig_runtime", "micro_diffusion.main_w8a8")


def _committed_shape(**over: Any) -> Dict[str, Any]:
    """A committed ``aot/*.mint.json`` for micro-diffusion, in the shape the
    fleet actually commits — flattened, declaration-named, no envelope."""
    body: Dict[str, Any] = {
        "_declaration": "micro_diffusion/main_w8a8.py — the @endpoint block",
        "family": FAMILY,
        "declaration_module": "micro_diffusion.main_w8a8",
        "weight_lane": "w8a8",
        "precision": "w8a8",
        "batch": 1,
        "lora_bucket": 0,
        "text_lens": [],
        "guidance_scales": [],
        "strict": True,
        "source_ref": "cozy/micro-diffusion:prod",
        "source_digest": "",
        "declared_vram_gb": 8.0,
    }
    body.update(over)
    return body


@pytest.fixture
def on_path(monkeypatch: pytest.MonkeyPatch, micro_src: None) -> None:
    monkeypatch.syspath_prepend(str(REPO / "tests"))
    monkeypatch.setenv("PYTHONPATH", ":".join(
        [str(REPO / "src"), str(REPO / "tests"), str(MICRO_SRC)]))


def _resolve(body: Dict[str, Any], **kw: Any) -> measure_child.MeasureJob:
    raw = json.dumps(body).encode()
    doc, flat = measure_child.load_document(raw)
    return measure_child.resolve_job(doc, flat, **kw)


def test_the_three_answers_the_payload_omits_come_off_the_declaration(
    on_path: None, w8a8_tree: Path,
) -> None:
    """A committed payload names a FAMILY and a MODULE. The function, the
    compile targets and the target slot's checkpoint are all derived — which
    is the whole of the fix, and each one of them is separately load-bearing.
    """
    job = _resolve(
        _committed_shape(), slot_flags=[f"pipeline={w8a8_tree}"])

    # (a) modules <- declaration_module.
    assert job.modules == ("micro_diffusion.main_w8a8",)
    # (b) function <- the endpoint declaring Compile(family=).
    assert job.function == "generate-w8a8"
    assert job.family == FAMILY and job.cfg.family == FAMILY
    # (c) targets <- that endpoint's own Compile(targets=). Since pgw#1107 no
    # committed payload carries them, and `compile_cache.resolve_targets`
    # returns NOTHING for an empty tuple — the same defect one step later.
    assert job.cfg.targets == ("transformer", "decoder")
    assert job.slots["pipeline"].path == str(w8a8_tree)


def test_source_ref_binds_to_the_slot_that_owns_the_targets(
    on_path: None, tmp_path: Path,
) -> None:
    """The payload's one checkpoint goes to the one slot the declared targets
    live on, decided from the declaration alone — the question has to be
    answered BEFORE a checkpoint is opened."""
    from gen_worker.registry import collect_endpoints

    specs = collect_endpoints(list(MODULES))
    spec = next(s for s in specs if s.name == "generate-w8a8")
    assert measure_child._target_owner(spec, ("transformer", "decoder")) == (
        "pipeline")

    # And when the operator names it, the flag wins over the payload — without
    # the payload's ref being resolved first and refusing on the way past.
    tree = tmp_path / "tree"
    tree.mkdir()
    job = _resolve(_committed_shape(), slot_flags=[f"pipeline={tree}"])
    assert job.slots["pipeline"].path == str(tree)


def test_an_explicit_function_overrides_the_derivation(
    on_path: None, tmp_path: Path,
) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    job = _resolve(
        _committed_shape(), function="generate-w8a8-turbo",
        slot_flags=[f"pipeline={tree}"])
    assert job.function == "generate-w8a8-turbo"


# ---------------------------------------------------------------------------
# 3. EVERY REFUSAL NAMES ITSELF, and none of them is a ValidationError.
# ---------------------------------------------------------------------------


def _refusal(body: Dict[str, Any], **kw: Any) -> measure_child.MeasureRefused:
    with pytest.raises(measure_child.MeasureRefused) as caught:
        _resolve(body, **kw)
    assert caught.value.reason in measure_child.REASONS, caught.value.reason
    return caught.value


def test_a_payload_naming_no_module_refuses_by_name() -> None:
    body = _committed_shape()
    body.pop("declaration_module")
    assert _refusal(body).reason == "no_declaration_module"


def test_a_module_this_image_cannot_import_refuses_by_name() -> None:
    body = _committed_shape(declaration_module="not_in_this_image.main")
    refusal = _refusal(body)
    assert refusal.reason == "declaration_module_unimportable"
    assert "ENDPOINT'S OWN IMAGE" in refusal.detail


def test_a_family_no_endpoint_declares_refuses_by_name(on_path: None) -> None:
    refusal = _refusal(_committed_shape(family="not-a-family"))
    assert refusal.reason == "function_underivable"
    assert FAMILY in refusal.detail, (
        "the refusal must say what this image DOES declare, or the operator "
        "cannot tell a stale payload from the wrong pod")


def test_a_ref_the_local_store_does_not_hold_refuses_without_downloading(
    on_path: None,
) -> None:
    """A measure run never downloads, so a slot that is not already on this
    machine is a refusal rather than a multi-gigabyte fetch."""
    refusal = _refusal(_committed_shape())
    assert refusal.reason == "slots_unresolvable"
    assert "never downloads" in refusal.detail
    assert "--slot pipeline=" in refusal.detail


def test_a_malformed_slot_flag_refuses_by_name(on_path: None) -> None:
    assert _refusal(
        _committed_shape(), slot_flags=["pipeline"]).reason == (
        "slots_unresolvable")


# ---------------------------------------------------------------------------
# 4. THE DOCUMENTED INVOCATION, end to end, on a committed-shape file.
# ---------------------------------------------------------------------------


def test_the_documented_command_measures_a_committed_shape_file(
    tmp_path: Path, w8a8_tree: Path, on_path: None,
) -> None:
    """THE acceptance, executed: a file in the shape endpoint repos commit,
    the exact ``main()`` an operator invokes, a real load, a real export loop
    and a real typed report — with no hand-written wire struct anywhere.
    """
    request = tmp_path / "transformer.mint.json"
    request.write_text(json.dumps(_committed_shape()))
    report_path = tmp_path / "measure.json"

    rc = measure_child.main([
        str(request), str(report_path), "--export-only",
        f"--slot=pipeline={w8a8_tree}",
    ])

    report = msgspec.json.decode(
        report_path.read_bytes(), type=measure_child.MeasureReport)
    assert rc == measure_child.EXIT_OK, f"{report.reason}: {report.detail[:600]}"
    assert report.ok and report.family == FAMILY
    assert report.function == "generate-w8a8"
    assert report.declared_classes == 3 and len(report.entries) == 3
    assert all(e.ok and e.nodes > 0 for e in report.entries), report.entries
    assert report.weights == "real"


def test_a_file_that_is_not_a_request_at_all_exits_bad_job(
    tmp_path: Path,
) -> None:
    request = tmp_path / "x.mint.json"
    request.write_text("not json")
    assert measure_child.main(
        [str(request), str(tmp_path / "r.json")]) == measure_child.EXIT_BAD_JOB


def test_every_committed_file_reaches_the_declaration_walk(
    tmp_path: Path,
) -> None:
    """The end of the cardless road for the REAL fleet files: each one gets
    past the decode and refuses only because this machine is not the
    endpoint's image. On the pre-fix tree not one of them got that far.
    """
    seen = set()
    for path in _committed():
        doc, flat = measure_child.load_document(path.read_bytes())
        with pytest.raises(measure_child.MeasureRefused) as caught:
            measure_child.resolve_job(doc, flat)
        seen.add(caught.value.reason)
        assert caught.value.reason == "declaration_module_unimportable", (
            f"{path}: {caught.value.reason}: {caught.value.detail[:300]}")
    assert seen == {"declaration_module_unimportable"}
