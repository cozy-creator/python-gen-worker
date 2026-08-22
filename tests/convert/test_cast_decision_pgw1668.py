"""pgw#1668 — a clone casts when casting would change bytes, and declares what it produced.

The measured defect: `clone-huggingface` of SenseNova-U1.5-8B-MoT-Preview with
`outputs=[{dtype: "bf16"}]` SUCCEEDED in 349 GPU-s, moved 50.19 GB, and
published the UPSTREAM tree unchanged with `dtype: bf16` on it. The hub's own
header read of the same bytes said `component_dtypes: {"model": "mixed"}`.

Two functions in one file disagreed about what dtype a tree is. The cast
decision asked a majority by TENSOR COUNT, and this checkpoint has 601 small
BF16 norm/bias islands against 515 enormous F32 weights — so "bf16" won a vote
that "would casting change any bytes?" was never a vote about. The label was
then the REQUEST rather than the bytes, so the lie was stamped as well as told.

The shape of the fix, and what these cases hold:
  * one measure, weighed by BYTES and strict — a tree IS a dtype only when
    every float tensor in it is that dtype, and anything else is `mixed`;
  * `mixed` is not requestable, so it matches no request and the cast RUNS;
  * a uniform source still short-circuits — the cheap path is taken when it is
    provably a no-op, not when a majority guesses it is;
  * the published `dtype` is READ OFF the produced tree, so no arm can stamp a
    clean token on mixed bytes.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fake_hub import _FakeHub

from gen_worker.convert.clone import (
    CAST_OUTPUT,
    PUBLISH_SOURCE,
    OutputSpec,
    run_clone,
    spec_actions,
)
from gen_worker.hubio.client import files_from_tree

from gen_worker.convert.ingest import (
    MIXED_DTYPE,
    IngestedSource,
    detect_snapshot_dtype,
    rollup_dtype,
    snapshot_float_dtype_bytes,
)

_WIDTH = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "I64": 8}


def _safetensors(tensors: dict[str, tuple[str, int]]) -> bytes:
    """One valid safetensors file: `name -> (dtype, element count)`."""

    header: dict[str, Any] = {}
    offset = 0
    for name, (dtype, count) in tensors.items():
        end = offset + count * _WIDTH[dtype]
        header[name] = {"dtype": dtype, "shape": [count], "data_offsets": [offset, end]}
        offset = end
    blob = json.dumps(header).encode()
    body = bytes(bytearray((i * 37 + 11) % 251 for i in range(offset)))
    return struct.pack("<Q", len(blob)) + blob + body


def _sensenova_shaped() -> dict[str, tuple[str, int]]:
    """se#840's real header profile, at fixture scale.

    601 BF16 tensors of 1116 — a tensor-count majority — against an F32 bulk
    that is 98% of the bytes. This is the exact shape that defeats a count
    vote and does not defeat a byte one.
    """

    islands = {f"blocks.{i}.norm.weight": ("BF16", 8) for i in range(601)}
    bulk = {f"blocks.{i}.mlp.weight": ("F32", 4096) for i in range(30)}
    return {**islands, **bulk}


def _uniform_bf16() -> dict[str, tuple[str, int]]:
    return {f"blocks.{i}.mlp.weight": ("BF16", 4096) for i in range(30)}


def _tree(root: Path, tensors: dict[str, tuple[str, int]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_bytes(b'{"architectures":["Fake"]}')
    (root / "model.safetensors").write_bytes(_safetensors(tensors))
    return root


def _published_dtypes(root: Path) -> set[str]:
    """Every float storage dtype actually present in a tree's headers."""

    out: set[str] = set()
    for p in sorted(Path(root).rglob("*.safetensors")):
        with open(p, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        out |= {str(v["dtype"]) for v in header.values()
                if isinstance(v, dict) and "dtype" in v}
    return out


def _bytes_under(root: Path) -> int:
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


# ------------------------------------------------------------- the measure


def test_the_dtype_of_a_tree_is_weighed_in_bytes_and_a_mixed_tree_says_so(
    tmp_path: Path,
) -> None:
    """THE DEFECT, isolated: the count vote and the bytes point opposite ways.

    Both halves are asserted, because "not bf16" alone would pass on a
    function that returned nothing at all.
    """

    tree = _tree(tmp_path / "src", _sensenova_shaped())

    mass = snapshot_float_dtype_bytes(tree)
    assert mass == {"bf16": 601 * 8 * 2, "fp32": 30 * 4096 * 4}
    assert mass["fp32"] > 40 * mass["bf16"], "fixture is not F32-dominant"

    # The vote the fix deleted, reconstructed here so the two are compared
    # on the same bytes rather than on a memory of what the old one did.
    with open(tree / "model.safetensors", "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    counts: dict[str, int] = {}
    for value in header.values():
        counts[str(value["dtype"])] = counts.get(str(value["dtype"]), 0) + 1
    assert max(counts, key=lambda k: counts[k]) == "BF16", (
        "the fixture no longer reproduces the count majority that caused this")

    assert detect_snapshot_dtype(tree) == MIXED_DTYPE


def test_a_uniform_tree_is_named_and_an_empty_one_abstains(tmp_path: Path) -> None:
    """Strictness must not turn every tree into `mixed` — that would be the
    same defect pointing the other way, and it would cast forever."""

    assert detect_snapshot_dtype(_tree(tmp_path / "bf16", _uniform_bf16())) == "bf16"
    # Non-float tensors ride along untouched by any cast, so they cannot make
    # a tree mixed.
    mixedish = {**_uniform_bf16(), "position_ids": ("I64", 512)}
    assert detect_snapshot_dtype(_tree(tmp_path / "ints", mixedish)) == "bf16"
    (tmp_path / "empty").mkdir()
    assert detect_snapshot_dtype(tmp_path / "empty") == ""
    assert rollup_dtype({"bf16": 0}) == ""


# ------------------------------------------------------------ the decision


@pytest.mark.parametrize(
    "source_dtype, expected",
    [("mixed", CAST_OUTPUT), ("fp32", CAST_OUTPUT), ("bf16", PUBLISH_SOURCE)],
)
def test_the_cast_runs_unless_it_is_provably_a_no_op(
    source_dtype: str, expected: str,
) -> None:
    """`mixed` matches no request, which is the whole decision."""

    assert spec_actions(
        [OutputSpec(dtype="bf16", file_layout="multi-file",
                    file_type="safetensors")],
        publish_as_is=True, source_dtype=source_dtype,
        explicit_outputs=True, cast_eligible=True,
    ) == [expected]


# --------------------------------------------------------- the whole clone


class _Ctx:
    def __init__(self, server: Any) -> None:
        self._file_api_base_url = f"http://127.0.0.1:{server.server_port}"
        self._worker_capability_token = "cap-token"
        self.owner = "tensorhub"
        self.request_id = "req-1668"
        self.destination = {"repo": "tensorhub/fallback"}


def _pinned_pipeline_tree(root: Path) -> Path:
    """A diffusers tree one of whose components a bf16 cast MUST NOT touch.

    `AutoencoderKLWan` carries an fp32 load pin (`families.facts`), so a
    tree-wide bf16 cast converts the transformer and steps over the VAE. What
    comes out is genuinely mixed while the request, and every attribute
    derived from it, still says `bf16`.
    """

    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "FakePipeline",
        "transformer": ["diffusers", "FakeTransformer"],
        "vae": ["diffusers", "AutoencoderKLWan"],
    }))
    for comp in ("transformer", "vae"):
        (root / comp).mkdir(exist_ok=True)
        (root / comp / "config.json").write_bytes(b"{}")
        (root / comp / "diffusion_pytorch_model.safetensors").write_bytes(
            _safetensors({f"{comp}.{i}.weight": ("F32", 4096) for i in range(8)}))
    return root


def _fake_plan(source_dir: Path, strategy: str, layout: str) -> Any:
    files = [
        (p.relative_to(source_dir).as_posix(), p.stat().st_size,
         hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(source_dir.rglob("*")) if p.is_file()
    ]
    return SimpleNamespace(
        provider="huggingface",
        paths=[name for name, _, _ in files],
        source_storage_bits=32,
        classification=SimpleNamespace(
            strategy=strategy,
            attrs={"file_layout": layout, "file_type": "safetensors"},
        ),
        bank_files=lambda: list(files),
    )


def _clone(
    monkeypatch: pytest.MonkeyPatch, fake_hub: Any, tmp_path: Path,
    source_dir: Path, published: list[Any] | None = None,
    strategy: str = "transformers", layout: str = "single-file", **kwargs: Any,
) -> Any:
    """Everything but the network, with the dtype MEASURED rather than declared.

    The download is faked (nothing here rents a pod), but `attrs["dtype"]` is
    produced by the same call the real `ingest_huggingface` makes on the
    materialized snapshot (`ingest.py`, `on_disk_dtype = detect_snapshot_dtype(
    dest_dir)`) — so the fixture states bytes, not verdicts.
    """

    plan = _fake_plan(source_dir, strategy, layout)
    attrs = dict(plan.classification.attrs)
    attrs["dtype"] = detect_snapshot_dtype(source_dir)
    plan.classification.attrs.update(attrs)

    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setattr("gen_worker.convert.clone.plan_huggingface",
                        lambda *a, **k: plan)

    # The tree HANDED TO THE PUBLISHER — the run deletes its workdir, and the
    # thing under test is which bytes went to the hub, not which files
    # survived the cleanup.
    if published is not None:
        def _capture(tree: Any, *a: Any, **k: Any) -> Any:
            root = Path(tree)
            published.append(SimpleNamespace(
                path=root,
                is_the_source_tree=(root.resolve() == source_dir.resolve()),
                storage_dtypes=_published_dtypes(root),
                per_component={
                    d.name: _published_dtypes(d) for d in sorted(root.iterdir())
                    if d.is_dir() and any(d.rglob("*.safetensors"))},
                dtype=detect_snapshot_dtype(root),
                bytes=_bytes_under(root),
            ))
            return files_from_tree(tree, *a, **k)

        monkeypatch.setattr("gen_worker.convert.clone.files_from_tree", _capture)

    monkeypatch.setattr(
        "gen_worker.convert.clone.ingest_huggingface",
        lambda source_ref, dest_dir, **kw: IngestedSource(
            provider="huggingface", source_ref=source_ref,
            source_revision="13a8d0f3", dir=source_dir, layout=layout,
            model_family="fake", model_family_variant="fake1",
            classification=plan.classification, attrs=attrs,
            metadata={"source_provider": "huggingface"},
            repo_spec={"kind": "model", "library_name": "transformers"},
        ))
    return run_clone(
        _Ctx(fake_hub), provider="huggingface", source_ref="sensenova/fake",
        destination_repo="sensenova/fake-tree", destination_release="r1",
        **kwargs,
    )


def _wire_dtype() -> str:
    """The dtype as the HUB received it, not as the producer remembers it."""

    published = list(_FakeHub.state["publishes"].values())
    assert len(published) == 1, published
    return str(published[0].get("dtype") or "")


def test_a_mixed_tree_asked_for_bf16_is_CAST_not_relabelled(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """se#840's job, end to end: 50.19 GB in must not be 50.19 GB out.

    Pre-fix this published the source unchanged and stamped it `bf16`. The
    three things that were wrong are the three things asserted: the cast ran,
    every F32 tensor is gone, and the tree shrank.
    """

    source_dir = _tree(tmp_path / "source", _sensenova_shaped())
    source_bytes = _bytes_under(source_dir)
    published: list[Any] = []

    result = _clone(
        monkeypatch, fake_hub, tmp_path, source_dir, published,
        outputs=[{"dtype": "bf16", "file_layout": "multi-file",
                  "file_type": "safetensors"}],
    )

    assert not result.failed_flavors, result.failed_flavors
    assert len(result.published) == 1
    assert len(published) == 1
    tree = published[0]

    assert not tree.is_the_source_tree, (
        "the SOURCE tree was handed to the publisher — no cast ran")
    assert tree.storage_dtypes == {"BF16"}, (
        f"an F32 tensor survived a bf16 cast: {sorted(tree.storage_dtypes)}")
    assert tree.bytes < source_bytes, (
        f"the produced tree is {tree.bytes} bytes against a {source_bytes}-byte "
        "source: nothing was rewritten")
    # The label and the bytes, checked against each other rather than each
    # against the request.
    assert tree.dtype == "bf16"
    assert _wire_dtype() == tree.dtype == result.published[0]["dtype"]


def test_a_uniform_bf16_source_asked_for_bf16_still_publishes_as_is(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cheap path survives, and that is not incidental.

    A rule that cast whenever it was unsure would pay a full re-encode on
    every already-correct mirror. The source tree itself is what gets
    published — no flavor tree is built at all.
    """

    source_dir = _tree(tmp_path / "source", _uniform_bf16())
    published: list[Any] = []

    result = _clone(
        monkeypatch, fake_hub, tmp_path, source_dir, published,
        outputs=[{"dtype": "bf16", "file_layout": "multi-file",
                  "file_type": "safetensors"}],
    )

    assert not result.failed_flavors, result.failed_flavors
    assert len(published) == 1
    assert published[0].is_the_source_tree, (
        "a provably no-op cast materialized a whole second tree")
    assert published[0].bytes == _bytes_under(source_dir)
    assert _wire_dtype() == "bf16"


def test_a_publish_as_is_of_a_mixed_tree_declares_MIXED(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second half of the lie: the label was the REQUEST.

    With no explicit outputs the source is published as-is by design — that
    arm is correct. What was not correct is stamping it with the default
    `bf16` spec. The bytes are mixed, so the checkpoint says mixed, and it
    agrees with the `component_dtypes` the hub derives from the same headers.
    """

    source_dir = _tree(tmp_path / "source", _sensenova_shaped())

    result = _clone(monkeypatch, fake_hub, tmp_path, source_dir)

    assert not result.failed_flavors, result.failed_flavors
    assert _wire_dtype() == MIXED_DTYPE
    assert result.published[0]["dtype"] == MIXED_DTYPE


def test_a_cast_that_a_PIN_made_mixed_declares_mixed_not_the_request(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The label must come off the TREE, and this is the arm that proves it.

    Everywhere else the honest measure alone already lands the right token,
    because the intent and the outcome agree. Here they cannot: the request is
    bf16, the flavor builder's `attrs["dtype"]` is bf16, the cast runs — and it
    steps over an fp32-pinned VAE by design, so the tree that reaches the hub
    is mixed. Reading the label off the request or off the builder's intent
    stamps `bf16` on it; reading it off the produced headers cannot.
    """

    source_dir = _pinned_pipeline_tree(tmp_path / "source")
    published: list[Any] = []

    result = _clone(
        monkeypatch, fake_hub, tmp_path, source_dir, published,
        strategy="diffusers_pipeline", layout="multi-file",
        outputs=[{"dtype": "bf16", "file_layout": "multi-file",
                  "file_type": "safetensors"}],
    )

    assert not result.failed_flavors, result.failed_flavors
    assert len(published) == 1
    tree = published[0]
    # The cast DID run and the pin DID hold — without both, this case is not
    # testing a mixed produced tree at all.
    assert tree.storage_dtypes == {"BF16", "F32"}, sorted(tree.storage_dtypes)
    assert tree.per_component == {"transformer": {"BF16"}, "vae": {"F32"}}

    assert _wire_dtype() == MIXED_DTYPE, (
        "a tree with an fp32 component was published as bf16")
    assert result.published[0]["dtype"] == MIXED_DTYPE


# ------------------------------------------------ the phase the fix reaches


class _RecordingCtx(_Ctx):
    """A ctx that keeps what reached `ctx.progress`, the way the hub sees it."""

    def __init__(self, server: Any) -> None:
        super().__init__(server)
        self.ticks: list[tuple[str, int]] = []

    def progress(self, progress: Any = None, stage: Any = None, *,
                 step: Any = None, total: Any = None, position: Any = None,
                 phase: Any = None) -> None:
        self.ticks.append((str(stage or phase), int(position or 0)))


def _accepted(ticks: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """`AdvanceJobProgress` updates the row — and with it the stall clock —
    only on a STRICT increase. Every other tick is dropped."""

    out: list[tuple[str, int]] = []
    last = -1
    for stage, position in ticks:
        if position > last:
            last = position
            out.append((stage, position))
    return out


def test_the_CAST_declares_a_position_because_it_is_now_the_longest_phase(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1667's rule, in the phase this fix made reachable.

    `clone.convert` used to be instantaneous on a publish-as-is source, because
    the cast was almost never chosen — a majority vote said the tree already
    matched. Now a 50 GB F32-dominant source really is re-encoded there, for
    tens of minutes, against a hub that kills a job whose declared position
    stops advancing inside its budget. Entering the phase is one tick; the
    re-encode has to be the rest.

    The fixture is sized past the position's MiB unit on purpose: a cast that
    writes less than 1 MiB advances nothing and would pass a weaker assertion
    while proving nothing about a real one.
    """

    mib = 1024 * 1024
    source_dir = _tree(tmp_path / "source", {
        # 4 x 1 MiB of BF16 output from 2 MiB of F32 input, plus the islands
        # that make the tree mixed in the first place.
        **{f"blocks.{i}.mlp.weight": ("F32", mib // 2) for i in range(4)},
        **{f"blocks.{i}.norm.weight": ("BF16", 8) for i in range(601)},
    })
    ctx = _RecordingCtx(fake_hub)

    plan = _fake_plan(source_dir, "transformers", "single-file")
    attrs = dict(plan.classification.attrs)
    attrs["dtype"] = detect_snapshot_dtype(source_dir)
    plan.classification.attrs.update(attrs)
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setattr("gen_worker.convert.clone.plan_huggingface",
                        lambda *a, **k: plan)
    monkeypatch.setattr(
        "gen_worker.convert.clone.ingest_huggingface",
        lambda source_ref, dest_dir, **kw: IngestedSource(
            provider="huggingface", source_ref=source_ref,
            source_revision="13a8d0f3", dir=source_dir, layout="single-file",
            model_family="fake", model_family_variant="fake1",
            classification=plan.classification, attrs=attrs,
            metadata={"source_provider": "huggingface"},
            repo_spec={"kind": "model", "library_name": "transformers"},
        ))

    result = run_clone(
        ctx, provider="huggingface", source_ref="sensenova/fake",
        destination_repo="sensenova/fake-tree", destination_release="r1",
        outputs=[{"dtype": "bf16", "file_layout": "multi-file",
                  "file_type": "safetensors"}],
    )
    assert not result.failed_flavors, result.failed_flavors

    convert = [p for stage, p in _accepted(ctx.ticks) if stage == "clone.convert"]
    assert len(convert) >= 4, (
        f"the cast declared {len(convert)} accepted position(s) while writing "
        f"4 MiB — a frozen position is a job the hub kills as wedged: "
        f"{_accepted(ctx.ticks)}")
    assert convert == sorted(set(convert)), "positions must strictly increase"
