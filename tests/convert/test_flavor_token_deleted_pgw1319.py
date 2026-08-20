"""A18 / §1.32(d), pgw#1319: the flavor token is DELETED, and the precision
class is DECLARED at every producer that publishes a non-base row.

`ProducedFlavor.flavor` was the last of the axis. It named no catalog row, but
`classify_flavor_token` turned it into `checkpoints.metadata["placement"]
["precision_class"]` — the hub's strongest evidence for a stored class where no
tensor-layout contract is proven — so a producer-local LABEL decided how the
artifact is served. It also got the answer wrong in both directions: the
classifier matched `fp8` and `fp8-*` and never `*-fp8`, so `coherent-fp8` and
`encoder-trunc-fp8` published as base for as long as they existed.

The cross-repo leg landed first, in two parts, because pgw cannot drop the
inference while a producer's only statement of class is a token: te#225
(`7e5e6196`) declared at seven sites, and te#227 (`e52ad87d`) took the census
to NINE after a third one made from the tree — `fuse`'s fp8 arm passes the
literal `"fp8"`, and the H3 modulation bake splits an fp8 DiT it never decodes.
Eleven of te's twenty production constructions publish a non-base row.

What is fenced here:

  1. Every one of those producer SHAPES publishes its declared class, through
     the real `publish_flavors` against the fake hub. The bags are transcribed
     literals: a fence that derived them from the code under test would agree
     with itself.
  2. The two refusals that replace the guess. An unclassifiable declaration and
     a tree of narrow bytes nobody classified are typed errors, never an
     unstamped publish — the hub's fallback for an unstamped row is `ClassBase`,
     which is exactly how the two live mis-stamps stayed invisible.
  3. The deletion itself: no field, no reader, no classifier.

    pytest tests/convert/test_flavor_token_deleted_pgw1319.py -q
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fake_hub import _FakeHub

from gen_worker.convert.produced import ProducedFlavor
from gen_worker.convert.publish import PrecisionClassRefusal, publish_flavors
from gen_worker.models import ladder

#: Cut by `tests/convert/conftest.py`'s fake repo.
RELEASE = "2026.08"

#: Every training-endpoints producer shape that publishes a NON-BASE row, as
#: its attribute bag reaches `publish_flavors` — transcribed from te
#: `origin/master` after te#227, not derived. The `dtype`/`file_layout` keys
#: are server-inferred and ride along; what matters is that the class arrives.
PRODUCER_SHAPES: dict[str, tuple[dict[str, str], dict[str, Any] | None]] = {
    # modelopt `produce_quant_flavor`, both schemes.
    "modelopt-fp8": (
        {"precision_class": "fp8", "quantization_method": "w8a8",
         "quantization_library": "modelopt", "dtype": "fp8"},
        {"precision_class": "fp8"}),
    "modelopt-nvfp4": (
        {"precision_class": "nvfp4-w4a4", "quantization_method": "nvfp4",
         "quantization_library": "modelopt"},
        {"precision_class": "nvfp4-w4a4"}),
    # svdq `mirror_svdq` / `produce_svdq_flavor` — `svdq_precision` is the
    # artifact's own field, and the class is derived from it, not from the
    # `svdq-fp4-r128` token that used to be the only statement.
    "svdq-fp4": (
        {"precision_class": "svdq-fp4", "svdq_precision": "fp4",
         "svdq_rank": "128", "quantization_library": "nunchaku"},
        {"precision_class": "svdq-fp4"}),
    "svdq-int4": (
        {"precision_class": "svdq-int4", "svdq_precision": "int4",
         "svdq_rank": "32", "quantization_library": "nunchaku"},
        {"precision_class": "svdq-int4"}),
    # h3_svdq `produce_h3_svdq` — the site pgw#1307's census missed entirely.
    "h3-svdq-fp4": (
        {"precision_class": "svdq-fp4", "svdq_precision": "fp4",
         "h3_native": "true"},
        {"precision_class": "svdq-fp4"}),
    # transform `cast_dtype`, both arms.
    "cast-w8a8": (
        {"precision_class": "fp8", "dtype": "fp8",
         "quantization_method": "w8a8"},
        {"precision_class": "fp8"}),
    "cast-fp8": ({"precision_class": "fp8", "dtype": "fp8"},
                 {"precision_class": "fp8"}),
    # The two LIVE mis-stamps te#225 closed: fp8 artifacts whose token
    # classified to "" because it ended in `-fp8` instead of starting with it.
    "coherent-fp8": (
        {"precision_class": "fp8", "dtype": "fp8",
         "coherent_checkpoint": "true"},
        {"precision_class": "fp8"}),
    "encoder-trunc-fp8": (
        {"precision_class": "fp8", "dtype": "fp8", "kept_layers": "20"},
        {"precision_class": "fp8"}),
    # te#227's two: `fuse`'s fp8 arm and both H3 modulation-bake variants.
    "fuse-fp8": (
        {"precision_class": "fp8", "fuse_component": "transformer",
         "fuse_scale": "1.0"},
        {"precision_class": "fp8"}),
    "adaln-full": (
        {"precision_class": "fp8", "component_set": "transformer",
         "adaln_projections": "present"},
        {"precision_class": "fp8"}),
    "adaln-baked": (
        {"precision_class": "fp8", "component_set": "transformer",
         "modulation_tables": "present"},
        {"precision_class": "fp8"}),
    # And the base rows, which declare NOTHING on purpose: the hub's own
    # fallback for an unstamped row is ClassBase, so a stamp would restate it.
    "fuse-bf16": ({"fuse_component": "transformer"}, None),
    "prompt-corpus": ({"corpus_rows": "512"}, None),
    "lora-card": ({"output_kind": "lora", "lora_rank": "32"}, None),
}


class _Ctx:
    def __init__(self, base_url: str) -> None:
        self._file_api_base_url = base_url
        self._worker_capability_token = "cap-token"
        self.owner = "acme"

    def log(self, message: str, **fields: Any) -> None:
        pass


def _opaque_tree(tmp_path: Path, name: str = "out") -> Path:
    """A tree whose bytes state nothing: no readable safetensors header, so
    `component_dtypes_on_disk` reports nothing and only the DECLARATION can
    speak. That is the shape every row above is asserted in."""
    out = tmp_path / name
    out.mkdir()
    (out / "diffusion.safetensors").write_bytes(b"\x11" * 2048)
    return out


def _fp8_tree(tmp_path: Path, name: str, *, dtype: Any) -> Path:
    """A tree whose bytes DO state their width — a real safetensors header the
    publish gate reads back off disk."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    out = tmp_path / name
    (out / "transformer").mkdir(parents=True)
    save_file({"proj_out.weight": torch.zeros(8, 8).to(dtype)},
              str(out / "transformer" / "diffusion_pytorch_model.safetensors"))
    return out


def _publish(fake_hub: Any, tree: Path, attrs: dict[str, str]) -> dict:
    ctx = _Ctx(f"http://127.0.0.1:{fake_hub.server_port}")
    publish_flavors(
        ctx,
        [ProducedFlavor(path=tree, attributes=dict(attrs))],
        destination_repo="acme/qwen-image",
        release=RELEASE,
    )
    return dict(_FakeHub.state["publish_request"].get("metadata") or {})


# ---------------------------------------------------------------------------
# 1. every producer shape, through the real publish path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", sorted(PRODUCER_SHAPES))
def test_every_producer_shape_publishes_the_class_it_declared(
    fake_hub: Any, tmp_path: Path, shape: str
) -> None:
    """Read off the declare request the fake hub actually receives. With the
    token gone the attribute bag is the ONLY statement of class, so a shape
    that stopped declaring lands as `ClassBase` — silently, which is the
    failure this row exists to make loud."""
    attrs, expected = PRODUCER_SHAPES[shape]
    meta = _publish(fake_hub, _opaque_tree(tmp_path), attrs)

    if expected is None:
        assert "placement" not in meta, (
            f"{shape} is a base row and must publish unstamped; a block here "
            "restates the hub's own ClassBase fallback")
        return
    assert meta.get("placement") == expected, (
        f"{shape} declared {attrs.get('precision_class')!r} and published "
        f"{meta.get('placement')!r}")
    assert set(meta["placement"]) == {"precision_class"}


def test_the_token_is_not_published_as_metadata_under_any_spelling(
    fake_hub: Any, tmp_path: Path
) -> None:
    """A18's end state: the word names nothing at the wire either."""
    attrs, _ = PRODUCER_SHAPES["svdq-fp4"]
    meta = _publish(fake_hub, _opaque_tree(tmp_path), attrs)
    assert "flavor" not in meta
    assert "flavor" not in _FakeHub.state["publish_request"]


# ---------------------------------------------------------------------------
# 2. the two refusals that replace the guess
# ---------------------------------------------------------------------------


def test_narrow_bytes_with_no_declaration_are_REFUSED_not_published_unstamped(
    fake_hub: Any, tmp_path: Path
) -> None:
    """The regression the te legs exist to prevent, asserted from pgw's side.

    Before A18 an fp8 tree whose producer declared nothing was stamped by the
    token classifier; after it, nothing would stamp it and the hub would serve
    it as base. So the publish REFUSES, naming the attribute to declare, before
    a byte moves.
    """
    torch = pytest.importorskip("torch")
    tree = _fp8_tree(tmp_path, "undeclared", dtype=torch.float8_e4m3fn)

    with pytest.raises(PrecisionClassRefusal) as exc:
        _publish(fake_hub, tree, {"dtype": "fp8"})

    msg = str(exc.value)
    assert "transformer=fp8" in msg, msg
    assert "precision_class" in msg, msg


def test_the_same_narrow_tree_publishes_once_its_class_is_declared(
    fake_hub: Any, tmp_path: Path
) -> None:
    """The refusal is about the SILENCE, not about the bytes: declare the class
    and the identical tree publishes, carrying it."""
    torch = pytest.importorskip("torch")
    tree = _fp8_tree(tmp_path, "declared", dtype=torch.float8_e4m3fn)

    meta = _publish(fake_hub, tree, {"dtype": "fp8", "precision_class": "fp8"})

    assert meta["placement"] == {"precision_class": "fp8"}
    assert meta["component_dtypes"] == {"transformer": "fp8"}


def test_a_base_tree_needs_no_declaration(fake_hub: Any, tmp_path: Path) -> None:
    """bf16 is 16 bits and every ladder rung is narrower, so the backstop stays
    silent for the rows that are correctly unstamped — otherwise it would
    refuse every base publish on the platform."""
    torch = pytest.importorskip("torch")
    tree = _fp8_tree(tmp_path, "base", dtype=torch.bfloat16)

    meta = _publish(fake_hub, tree, {"dtype": "bf16"})

    assert "placement" not in meta
    assert meta["component_dtypes"] == {"transformer": "bf16"}


@pytest.mark.parametrize("bad", ["fp8-w8a8", "svdq-fp4-r128", "int4", "q4_k_m"])
def test_a_class_outside_the_vocabulary_is_REFUSED(
    fake_hub: Any, tmp_path: Path, bad: str
) -> None:
    """te#225's rule, on pgw's side of the seam: unclassifiable is a refusal,
    not an unstamped publish. `fp8-w8a8` and `svdq-fp4-r128` are the two most
    likely mistakes — they are TOKENS, and the token is what died."""
    with pytest.raises(PrecisionClassRefusal) as exc:
        _publish(fake_hub, _opaque_tree(tmp_path), {"precision_class": bad})
    assert "not a class tensorhub reads" in str(exc.value)


def test_a_class_is_matched_case_and_whitespace_insensitively(
    fake_hub: Any, tmp_path: Path
) -> None:
    """The declaration is a producer's literal; `"FP8"` is the same statement
    as `"fp8"` and refusing it would be a spelling gate, not a class gate."""
    meta = _publish(fake_hub, _opaque_tree(tmp_path), {"precision_class": " FP8"})
    assert meta["placement"] == {"precision_class": "fp8"}


# ---------------------------------------------------------------------------
# 3. the deletion itself
# ---------------------------------------------------------------------------


def test_the_produced_struct_has_no_flavor_field() -> None:
    assert "flavor" not in ProducedFlavor.__struct_fields__
    with pytest.raises(TypeError):
        ProducedFlavor(path=Path("/nonexistent"), flavor="fp8")  # type: ignore[call-arg]


def test_the_classifier_is_deleted_and_the_vocabulary_survives() -> None:
    """The vocabulary is what a DECLARATION is checked against, so it stays —
    one home, in this repo, which te imports rather than re-listing.

    SEVEN classes since pgw#1498 (`4d2bce0c`), which added `gguf`: ONE class
    for every ggml block encoding, because the qtype is a property of the
    artifact's bytes (it travels in the tensor-layout contract) while the CLASS
    is what the hub's ladder ranks, and every qtype ranks the same way against
    fp8 and base. The exact-set form is deliberate — the vocabulary is a
    cross-repo contract (tensorhub `precision.Class*`, which te imports rather
    than re-lists), so a member added or removed here must be a decision
    somebody took, not a drift somebody absorbed.
    """
    assert not hasattr(ladder, "classify_flavor_token")
    assert ladder.PRECISION_CLASSES == frozenset({
        "base", "fp8", "gguf", "nvfp4", "nvfp4-w4a4", "svdq-fp4", "svdq-int4"})


@pytest.mark.parametrize(
    "dead", ["svdq-fp4-r128", "fp8-w8a8", "q4_k_m", "q8_0", "bnb-nf4", "int4"])
def test_a_dead_spelling_can_never_re_enter_the_vocabulary(dead: str) -> None:
    """The positive half of the fence above, stated as the thing that must not
    come back. Each of these was once a real spelling somewhere — `-r128` and
    `fp8-w8a8` are TOKENS (the axis A18 deleted), the two ggml qtypes are what
    pgw#1498 refused to make classes of, and `bnb-nf4` is the rung pgw#1206 D
    deleted. A class is refused at publish (`convert.publish`) precisely
    because it is not a member here, so membership is the whole gate."""
    assert dead not in ladder.PRECISION_CLASSES


def test_the_publish_leg_names_the_artifact_not_a_flavor(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The leg's label was the token. A produced path's own name is the thing
    that actually exists, is always present, and tells one leg of an
    N-artifact export from another."""
    import gen_worker.activity as activity

    seen: list[str] = []
    monkeypatch.setattr(
        activity, "emit_event",
        lambda kind, detail, **kw: seen.append(f"{kind}: {detail}"))

    _publish(fake_hub, _opaque_tree(tmp_path, "svdq-tree"), {})

    legs = [s for s in seen if s.startswith("convert_publish:")]
    assert legs, "the publish emitted no legs at all"
    assert all("artifact=svdq-tree" in leg for leg in legs), legs
    assert not any("flavor=" in leg for leg in legs), legs
