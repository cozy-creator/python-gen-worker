"""A checkpoint whose key names predate the installed library still loads.

# pgw#1453: sd1.5's text_encoder matched 0 of 197 tensors on the streaming path.

No mocks. A REAL diffusers pipeline is saved to real safetensors, then its
containers are REWRITTEN under the names the *previous* library generation
spelled — transformers' ``text_model.`` prefix and diffusers'
``query/key/value/proj_attn`` attention block — ingested into a REAL chunked
CAS, projected the way the chokepoint projects it, and streamed through the
REAL engine. This is exactly the shape of the stock sd1.5 mirror that matched
**0 of 197** tensors in ``text_encoder``.

⚠️ **The red arm is guarded, and it is guarded on the premise rather than on
the outcome.** Each legacy container is first asserted to name NOTHING the
skeleton carries — if a future library made the legacy names valid again, the
fixture would stop being a legacy fixture and this suite would say so instead
of passing for the wrong reason. The counter-case is the same tree left at its
MODERN names: the migration must rename nothing there, so a migration that
renamed indiscriminately cannot pass both arms.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from gen_worker._vendor.tensorfs import LocalCAS, project_snapshot  # noqa: E402
from cas_fixture import ingest_repository  # noqa: E402
from gen_worker.models.projection import REF_PREFIX, SNAPSHOTS_DIR  # noqa: E402
from gen_worker.serving.streaming import StreamingLoader, engine_for  # noqa: E402
from gen_worker.serving.streaming import keymap  # noqa: E402
from gen_worker.serving.streaming.skeleton import (  # noqa: E402
    build as build_skeleton,
)
from gen_worker.serving.streaming.skeleton import meta_survivors  # noqa: E402
from streaming_fixture import Lane, build_source  # noqa: E402

#: transformers v5 flattened the `text_model.` prefix CLIP checkpoints carry.
_LEGACY_CLIP_PREFIX = "text_model."

#: diffusers renamed the deprecated attention block's projections. These are
#: NOT expressible as any string rule — `key` is not a suffix of `to_k` — which
#: is why the map must come from the library and never from a table here.
_LEGACY_ATTN: Dict[str, str] = {
    "to_q": "query", "to_k": "key", "to_v": "value", "to_out.0": "proj_attn",
}


def _rewrite_keys(path: Path, rename: Any) -> int:
    """Rewrite a safetensors container's HEADER keys, byte-for-byte identical
    payload. Returns how many keys changed."""
    raw = path.read_bytes()
    (size,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + size])
    body = raw[8 + size :]
    rebuilt: Dict[str, Any] = {}
    changed = 0
    for key, spec in header.items():
        if key == "__metadata__":
            rebuilt[key] = spec
            continue
        new = rename(key)
        changed += new != key
        rebuilt[new] = spec
    blob = json.dumps(rebuilt, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + body)
    return changed


def _to_legacy_clip(key: str) -> str:
    return _LEGACY_CLIP_PREFIX + key


def _deprecated_attn_paths(module: Any) -> Tuple[str, ...]:
    """The submodule paths diffusers ITSELF marks as the deprecated attention
    block. Asking the library which blocks ever carried the old spelling is
    what keeps this fixture honest — writing `query` under a block that was
    never an `AttentionBlock` would be inventing a checkpoint nobody published,
    and diffusers would be right to refuse it."""
    return tuple(
        name for name, sub in module.named_modules()
        if getattr(sub, "_from_deprecated_attn_block", False)
    )


def _legacy_attn_renamer(paths: Tuple[str, ...]) -> Any:
    def rename(key: str) -> str:
        for path in paths:
            if not key.startswith(path + "."):
                continue
            for modern, legacy in _LEGACY_ATTN.items():
                for tail in ("weight", "bias"):
                    if key == f"{path}.{modern}.{tail}":
                        return f"{path}.{legacy}.{tail}"
        return key

    return rename


def _project(base: Path, source: Path, key: str) -> Path:
    cas = LocalCAS(base)
    manifest = ingest_repository(cas, source)
    cas.compare_and_swap_ref(
        REF_PREFIX + key, cas.store_manifest(manifest), expected=None
    )
    tree = base / SNAPSHOTS_DIR / key
    project_snapshot(cas, manifest, tree)
    return tree


def _container_keys(path: Path) -> Tuple[str, ...]:
    raw = path.read_bytes()
    (size,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + size])
    return tuple(k for k in header if k != "__metadata__")


def _skeleton_names(module: Any) -> set[str]:
    names = {n for n, _ in module.named_parameters(remove_duplicate=False)}
    names |= {n for n, _ in module.named_buffers(remove_duplicate=False)}
    return names


@pytest.fixture(scope="module")
def modern(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    """The pipeline exactly as the library writes it today."""
    base = tmp_path_factory.mktemp("pgw1453-modern")
    source = base / "source-model"
    pipeline_cls = build_source(source)
    return {"base": base, "source": source, "pipeline_cls": pipeline_cls,
            "tree": _project(base, source, key="c" * 64)}


@pytest.fixture(scope="module")
def legacy(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    """The SAME pipeline, its containers renamed to the previous generation."""
    base = tmp_path_factory.mktemp("pgw1453-legacy")
    source = base / "source-model"
    pipeline_cls = build_source(source)
    renamed = 0
    for component in ("text_encoder", "text_encoder_2"):
        for container in sorted((source / component).glob("*.safetensors")):
            renamed += _rewrite_keys(container, _to_legacy_clip)
    attn = 0
    from diffusers import AutoencoderKL, UNet2DConditionModel

    for component, cls in (("unet", UNet2DConditionModel), ("vae", AutoencoderKL)):
        paths = _deprecated_attn_paths(cls.from_config(  # type: ignore[attr-defined]
            cls.load_config(str(source / component))  # type: ignore[attr-defined]
        ))
        if not paths:
            continue
        for container in sorted((source / component).glob("*.safetensors")):
            attn += _rewrite_keys(container, _legacy_attn_renamer(paths))
    assert renamed, "the CLIP fixture was not actually made legacy"
    return {"base": base, "source": source, "pipeline_cls": pipeline_cls,
            "tree": _project(base, source, key="d" * 64),
            "clip_renamed": renamed, "attn_renamed": attn}


# -- the premise: this fixture really is unloadable without a migration -----


def test_the_legacy_clip_container_names_nothing_the_skeleton_carries(
    legacy: Dict[str, Any],
) -> None:
    """The go-red condition, asserted rather than assumed.

    sd1.5's measured overlap was 0 of 197. If a library change ever made these
    names valid again, the fixture would stop testing what it claims to and
    this assertion is what says so.
    """
    built = build_skeleton(legacy["pipeline_cls"], legacy["tree"])
    module = built.modules["text_encoder"]
    names = _skeleton_names(module)
    keys = _container_keys(
        next((legacy["source"] / "text_encoder").glob("*.safetensors"))
    )
    assert keys, "no container to read"
    assert all(key.startswith(_LEGACY_CLIP_PREFIX) for key in keys)
    assert not (set(keys) & names), (
        "the legacy container overlaps the skeleton, so it is not a legacy "
        "fixture and the green arm below would prove nothing"
    )


# -- the map comes from the library ----------------------------------------


def test_transformers_own_migration_places_every_legacy_clip_tensor(
    legacy: Dict[str, Any],
) -> None:
    built = build_skeleton(legacy["pipeline_cls"], legacy["tree"])
    module = built.modules["text_encoder"]
    names = _skeleton_names(module)
    keys = _container_keys(
        next((legacy["source"] / "text_encoder").glob("*.safetensors"))
    )
    renames = keymap.migration(module, keys)
    placed = [renames.get(key, key) for key in keys]
    assert len(renames) == len(keys), "every legacy key must be migrated"
    assert set(placed) <= names, sorted(set(placed) - names)[:5]
    assert len(set(placed)) == len(placed), "two keys landed on one name"


def test_diffusers_own_migration_places_the_deprecated_attention_block(
    legacy: Dict[str, Any],
) -> None:
    """`key -> to_k` and `proj_attn -> to_out.0` are SEMANTIC renames no string
    rule expresses, so this is the arm that proves the map is the library's."""
    if not legacy["attn_renamed"]:
        pytest.skip("this diffusers version builds no deprecated attn block")
    built = build_skeleton(legacy["pipeline_cls"], legacy["tree"])
    for component in ("unet", "vae"):
        module = built.modules[component]
        names = _skeleton_names(module)
        for container in sorted((legacy["source"] / component).glob("*.safetensors")):
            keys = _container_keys(container)
            renames = keymap.migration(module, keys)
            placed = [renames.get(key, key) for key in keys]
            assert set(placed) <= names, sorted(set(placed) - names)[:5]


def test_a_modern_checkpoint_is_migrated_by_nothing(
    modern: Dict[str, Any],
) -> None:
    """The counter-case. Every published/converted artifact is already spelled
    the way the installed library spells it, so the migration must be a
    no-op — otherwise the green arm above is just "renames everything"."""
    built = build_skeleton(modern["pipeline_cls"], modern["tree"])
    for component, module in built.modules.items():
        for container in sorted((modern["source"] / component).glob("*.safetensors")):
            keys = _container_keys(container)
            assert keymap.migration(module, keys) == {}, component


# -- the whole engine, end to end ------------------------------------------


def _load(tree: Path, pipeline_cls: type) -> Any:
    store = engine_for(tree, device="cpu")
    assert store is not None, "the projected tree carries no chunk store"
    return store.build(pipeline_cls, checkpoint_dir=tree, lane=Lane())


def test_the_engine_loads_a_legacy_checkpoint_with_nothing_left_on_meta(
    legacy: Dict[str, Any],
) -> None:
    """The defect, end to end: this raised ``NameMismatch`` on every tensor."""
    pipeline = _load(legacy["tree"], legacy["pipeline_cls"])
    for component in ("text_encoder", "text_encoder_2", "unet", "vae"):
        module = getattr(pipeline, component)
        assert meta_survivors(module) == (), component


def test_the_legacy_tree_loads_to_the_same_bytes_as_the_modern_one(
    legacy: Dict[str, Any], modern: Dict[str, Any],
) -> None:
    """Renaming a key must move a tensor, never change one. Both trees are
    built from the same seed, so every parameter must be byte-identical."""
    was = _load(legacy["tree"], legacy["pipeline_cls"])
    now = _load(modern["tree"], modern["pipeline_cls"])
    checked = 0
    for component in ("text_encoder", "text_encoder_2", "unet", "vae"):
        left = dict(getattr(was, component).named_parameters(remove_duplicate=False))
        right = dict(getattr(now, component).named_parameters(remove_duplicate=False))
        assert set(left) == set(right), component
        for name, tensor in left.items():
            other = right[name]
            assert tensor.dtype == other.dtype, f"{component}/{name}"
            assert torch.equal(
                tensor.reshape(-1).view(torch.uint8),
                other.reshape(-1).view(torch.uint8),
            ), f"{component}/{name} moved to the wrong slot"
            checked += 1
    assert checked > 50, checked


def test_a_container_the_library_cannot_explain_is_still_refused(
    legacy: Dict[str, Any],
) -> None:
    """The refusal is not softened. A name neither the skeleton nor the
    library's own history knows is still a ``NameMismatch``, and the message
    now says the migration ran — the fact that separates "wrong checkpoint"
    from "old checkpoint"."""
    from gen_worker.serving.streaming import NameMismatch

    base = legacy["base"] / "unknown"
    source = base / "source-model"
    pipeline_cls = build_source(source)
    for container in sorted((source / "text_encoder").glob("*.safetensors")):
        _rewrite_keys(container, lambda key: "not_a_name_any_version_had." + key)
    tree = _project(base, source, key="e" * 64)
    with pytest.raises(NameMismatch) as raised:
        _load(tree, pipeline_cls)
    assert "pgw#1453" in str(raised.value)


def test_the_streaming_loader_reports_the_legacy_load_as_a_normal_one(
    legacy: Dict[str, Any],
) -> None:
    """A migrated load is a load, not a degraded one: the report must carry the
    same tensor count the container holds."""
    store = engine_for(legacy["tree"], device="cpu")
    assert isinstance(store, StreamingLoader)
    store.build(legacy["pipeline_cls"], checkpoint_dir=legacy["tree"], lane=Lane())
    report = store.last_report
    assert report is not None
    assert report.tensors > 50, report.tensors
    assert report.weights_streamed_bytes > 0
