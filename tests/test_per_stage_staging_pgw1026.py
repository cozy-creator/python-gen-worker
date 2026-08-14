"""pgw#1026: a modular tree the CARD holds but the HOST does not must boot.

A 134.1 GiB tree plus the 8 GiB staging floor against 116.4 GiB of host RAM
refuses structurally (``HostRamCapacityError``) on a pod whose card holds it
fine. Host RAM binds ~26 GiB tighter than VRAM purely because staging is
all-or-nothing while the load is already component-sequential. Two halves, both
here:

* the LOADER places each component as it lands and drops the host copy, so
  the host-RAM high-water mark is one component instead of the tree;
* the ADMISSION GATE charges that same largest component, so the boot is not
  refused before the loader gets to run.

Real trees throughout. The loader tests hydrate REAL diffusers modular
pipelines (the pgw#1036 harness). The admission tests need those
magnitudes, so their trees are real directories of SPARSE files — real
``st_size``, real ``os.walk``, real index parsing, no disk cost.

NOT COVERED HERE: the placement itself lands on ``meta`` rather than
``cuda``. ``Module.to`` is the same call either way and ``meta`` frees the
host storage exactly as a device move does, which is the property under
test — but no CUDA device runs in this suite, so the device-side residency
of a streamed hydration is proved on a pod, not here.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from gen_worker.capability import HostRamCapacityError
from gen_worker.models import loading as loading_mod
from gen_worker.models.loading import (
    ModularHydrationError,
    decide_streamed_hydration,
    hydrate_modular_pipeline,
    modular_staging_units,
    plan_streamed_hydration,
)
from gen_worker.models.memory import HostRam

from harness.modular_endpoint import (
    TinyModularPipeline,
    build_base_tree,
)

_GIB = 1024 ** 3


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")


def _ram(total_gb: float, available_gb: float) -> HostRam:
    return HostRam(
        total_gb=total_gb, available_gb=available_gb,
        meminfo_total_gb=total_gb, meminfo_available_gb=available_gb,
        cgroup_limit_gb=total_gb, source="cgroup",
    )


# ---------------------------------------------------------------------------
# the staging units: what one component's staging actually costs
# ---------------------------------------------------------------------------


def test_the_unit_of_staging_is_a_component_dir(tmp_path) -> None:
    tree = build_base_tree(tmp_path / "base", fill=1.0)
    units = modular_staging_units(tree)

    assert set(units) == {"unet", "vae", "vae_ref", "scheduler"}
    for name, nbytes in units.items():
        assert nbytes == loading_mod.disk_gc.tree_bytes(tree / name), name
    # The config-only unselected partition (H3's `transformer_ref` shape)
    # costs its config and nothing else.
    assert units["vae_ref"] < units["vae"]
    assert max(units.values()) < sum(units.values())


def test_a_tree_with_no_modular_index_yields_no_units(tmp_path) -> None:
    (tmp_path / "plain").mkdir()
    assert modular_staging_units(tmp_path / "plain") == {}


# ---------------------------------------------------------------------------
# the decision, at ie#615's magnitudes
# ---------------------------------------------------------------------------


def _h3(**over: Any) -> Any:
    """ie#615's measured shape: 134.1 GiB tree on a 116.4 GiB host, largest
    component 46 GiB, on a card set that holds the tree."""
    kwargs: Dict[str, Any] = dict(
        tree_bytes=int(134.1 * _GIB),
        largest_unit_bytes=int(46.0 * _GIB),
        unit_count=6,
        host_total_bytes=int(116.4 * _GIB),
        device_free_bytes=int(141.0 * _GIB),
    )
    kwargs.update(over)
    return decide_streamed_hydration(**kwargs)


def test_the_ie615_shape_engages() -> None:
    plan = _h3()
    assert plan.engaged, plan.summary()
    assert "134.1GiB" in plan.summary() and "46.0GiB" in plan.summary()


def test_a_tree_that_fits_the_host_is_left_alone() -> None:
    """Nothing is wrong on this path, so nothing changes on it."""
    plan = _h3(tree_bytes=int(40.0 * _GIB), largest_unit_bytes=int(20.0 * _GIB))
    assert not plan.engaged
    assert plan.reason == "the whole tree fits host RAM"


def test_a_component_bigger_than_the_host_still_refuses_honestly() -> None:
    """Sequencing cannot place a component that cannot be staged at all —
    the pgw#752 structural verdict has to survive this issue."""
    plan = _h3(largest_unit_bytes=int(116.0 * _GIB))
    assert not plan.engaged
    assert plan.reason == "the largest component alone exceeds host RAM"


def test_a_card_that_does_not_hold_the_tree_does_not_engage() -> None:
    """The 1x H100-80 ie#615 actually ran on: the components have nowhere to
    go, so this is te#172's (or a bigger pod's) problem, not this one's."""
    assert not _h3(device_free_bytes=int(80.0 * _GIB)).engaged
    # ... and the margin is real: exactly the tree is not enough.
    assert not _h3(device_free_bytes=int(134.1 * _GIB)).engaged
    assert _h3(device_free_bytes=int(137.0 * _GIB)).engaged


def test_an_unreadable_host_probe_changes_nothing() -> None:
    assert not _h3(host_total_bytes=0).engaged
    assert not _h3(unit_count=0, largest_unit_bytes=0).engaged


def test_the_plan_measures_a_real_tree(tmp_path, monkeypatch) -> None:
    """The measuring wrapper reads the same units, the real host probe and
    the real free-VRAM probe — the tiny harness tree fits any host, so it
    does not engage, and that is the honest answer for it."""
    tree = build_base_tree(tmp_path / "base", fill=1.0)
    monkeypatch.setattr(loading_mod, "probe_host_ram", lambda: _ram(64.0, 60.0))
    plan = plan_streamed_hydration(tree, device_free_bytes=80 * _GIB)
    assert not plan.engaged
    assert plan.tree_bytes == sum(modular_staging_units(tree).values())
    assert plan.unit_count == 4


# ---------------------------------------------------------------------------
# the admission gate: ie#615's tree, on a real Executor
# ---------------------------------------------------------------------------


def _sparse_h3_tree(root: Path) -> Path:
    """A real modular tree whose component dirs measure H3's sizes. The
    files are sparse: `st_size` (what `tree_bytes` reads) is real, the
    blocks are not allocated."""
    root.mkdir(parents=True, exist_ok=True)
    sizes = {
        "text_encoder": int(60.0 * _GIB),
        "transformer": int(46.0 * _GIB),
        "transformer_2": int(26.0 * _GIB),
        "vae": int(2.1 * _GIB),
    }
    index: Dict[str, Any] = {
        "_class_name": "TinyModularPipeline",
        "_blocks_class_name": "TinyBlocks",
    }
    for name, nbytes in sizes.items():
        d = root / name
        d.mkdir()
        with open(d / "model.safetensors", "wb") as f:
            f.truncate(nbytes)
        index[name] = ["diffusers", "UNet2DConditionModel", {
            "pretrained_model_name_or_path": "upstream/h3",
            "subfolder": name, "variant": None, "revision": None,
        }]
    (root / "modular_model_index.json").write_text(json.dumps(index))
    return root


async def _admit(spec, paths):
    from gen_worker.executor import Executor

    async def _send(_msg: Any) -> None:
        return None

    ex = Executor([spec], _send)
    await ex._ensure_host_ram_for(spec, paths)


def _modular_spec():
    from gen_worker.registry import extract_specs

    from harness.modular_endpoint import ModularEndpoint

    return extract_specs(ModularEndpoint)[0]


def test_the_gate_charges_the_largest_component_for_a_modular_slot(
    tmp_path, monkeypatch,
) -> None:
    """The whole point: 134.1 GiB of tree on a 116.4 GiB host is admitted,
    because what stages at once is the 60 GiB text encoder."""
    tree = _sparse_h3_tree(tmp_path / "h3")
    monkeypatch.setattr(loading_mod, "probe_host_ram", lambda: _ram(116.4, 110.0))
    monkeypatch.setattr(
        "gen_worker.models.memory.probe_host_ram", lambda **_: _ram(116.4, 110.0))
    monkeypatch.setattr(
        loading_mod, "get_available_vram_gb", lambda *a, **k: 141.0)

    asyncio.run(_admit(_modular_spec(), {"pipeline": str(tree)}))


def test_the_gate_still_refuses_when_no_card_can_hold_the_tree(
    tmp_path, monkeypatch,
) -> None:
    """1x H100-80, the pod ie#615 actually had: the refusal is correct and
    must stay — with its measured numbers, not a softened verdict."""
    tree = _sparse_h3_tree(tmp_path / "h3")
    monkeypatch.setattr(loading_mod, "probe_host_ram", lambda: _ram(116.4, 110.0))
    monkeypatch.setattr(
        "gen_worker.models.memory.probe_host_ram", lambda **_: _ram(116.4, 110.0))
    monkeypatch.setattr(
        loading_mod, "get_available_vram_gb", lambda *a, **k: 80.0)

    with pytest.raises(HostRamCapacityError) as exc:
        asyncio.run(_admit(_modular_spec(), {"pipeline": str(tree)}))
    assert exc.value.required_bytes > exc.value.total_bytes


# ---------------------------------------------------------------------------
# the loader: one component on the host at a time
# ---------------------------------------------------------------------------


def _device_of(comp: Any) -> Optional[str]:
    if comp is None:
        return None
    try:
        return next(iter(comp.parameters())).device.type
    except StopIteration:
        return None  # weightless (scheduler)
    except AttributeError:
        return None


def _hydrate_watching_devices(tree: Path, monkeypatch, *, place_device: str):
    """Hydrate for real, recording every component's device at the moment
    each NEXT component starts staging (the pgw#1041 phase marker)."""
    pipe = TinyModularPipeline.from_pretrained(tree)
    observed: List[Dict[str, Optional[str]]] = []
    real = loading_mod.load_progress.set_phase

    def watch(phase: str, nbytes: int = 0, **kw):
        if str(phase).startswith("hydrate:"):
            observed.append({
                n: _device_of(getattr(pipe, n, None))
                for n in ("unet", "vae", "scheduler")
            })
        return real(phase, nbytes, **kw)

    monkeypatch.setattr(loading_mod.load_progress, "set_phase", watch)
    hydrate_modular_pipeline(pipe, tree, place_device=place_device)
    return pipe, observed


def test_each_component_leaves_the_host_before_the_next_one_stages(
    tmp_path, monkeypatch,
) -> None:
    tree = build_base_tree(tmp_path / "base", fill=1.0)
    pipe, observed = _hydrate_watching_devices(
        tree, monkeypatch, place_device="meta")

    assert len(observed) == 3  # scheduler, unet, vae (vae_ref is excluded)
    for i, snapshot in enumerate(observed):
        on_host = sorted(n for n, d in snapshot.items() if d == "cpu")
        assert not on_host, (
            f"staging component {i} with {on_host} still on the host — the "
            "high-water mark is the tree again"
        )
    # and every component really is hydrated, just not on the host
    assert _device_of(pipe.unet) == "meta"
    assert _device_of(pipe.vae) == "meta"
    assert pipe.scheduler is not None


def test_without_a_place_device_the_tree_accumulates_on_the_host(
    tmp_path, monkeypatch,
) -> None:
    """The control, and today's behaviour: unchanged for every slot the
    plan does not engage for."""
    tree = build_base_tree(tmp_path / "base", fill=1.0)
    pipe, observed = _hydrate_watching_devices(
        tree, monkeypatch, place_device="")

    assert any(
        d == "cpu" for snapshot in observed for d in snapshot.values()), (
        "expected the pre-pgw#1026 shape: hydrated components stay on the "
        "host until placement"
    )
    assert _device_of(pipe.unet) == "cpu"
    assert _device_of(pipe.vae) == "cpu"


def test_a_placement_that_fails_mid_hydration_is_typed(
    tmp_path, monkeypatch,
) -> None:
    """Free VRAM is read once, before the first component loads. If it moved
    under us the `.to()` fails, and it must fail NAMING the component — not
    leave a half-placed pipeline behind a nameless torch error."""
    tree = build_base_tree(tmp_path / "base", fill=1.0)
    pipe = TinyModularPipeline.from_pretrained(tree)
    with pytest.raises(ModularHydrationError) as exc:
        hydrate_modular_pipeline(pipe, tree, place_device="cuda:77")
    assert "per-component staging could not place component" in str(exc.value)
