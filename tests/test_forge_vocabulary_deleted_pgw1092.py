"""pgw#1092 / th#1751 W4 (DESIGN-RULINGS §4.28) — the forge vocabulary is GONE.

Paul, 2026-08-10: *"the forge-system is kind of stupid and should be deleted
entirely honestly… untrusted machines that need a cell that doesn't exist can
always create it themselves on their end; they don't need to request anything.
Yeah just delete the entire mint-request + forge system."*

This file REPLACES ``test_worker_goals_pgw930.py``, whose whole subject — the
composable ``(serve, mint)`` goal pair, the ``mint`` goal's driver and the
three tenant-reserve relaxations it licensed — no longer exists. What pgw#930
was protecting (a mint goal must not silently drop the reserves of a pod that
is ALSO serving) is now structural: there is one pod class, it always serves,
and the reserves are unconditional. See ``test_mint_vram_cap_pgw848`` and
``test_pool_simultaneity_pgw992`` for the arithmetic half.

Each row below names the state it was RED in before this change.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from gen_worker import config as gw_config
from gen_worker import env_seal, worker_goals

_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# The module surface
# ---------------------------------------------------------------------------


def test_the_mint_goal_module_is_gone_and_the_package_still_imports() -> None:
    """RED before: ``gen_worker.mint_goal`` was a 230-line driver importable
    from ``lifecycle`` and ``executor``, and deleting it without its callers
    raised ``NameError`` out of ``Executor._mark_warm_complete``."""
    with pytest.raises(ImportError):
        __import__("gen_worker.mint_goal")

    from gen_worker import executor, lifecycle

    assert not hasattr(lifecycle, "mint_goal_mod")
    assert not hasattr(executor, "mint_goal_mod")
    # ...and the accessors that existed ONLY so the driver could decide it
    # was finished go with it. A serving pod never retires and never asks.
    from gen_worker import fleet_cells

    assert not hasattr(executor.Executor, "background_mint_tasks")
    assert not hasattr(fleet_cells, "published_cells")
    assert not hasattr(fleet_cells, "refused_publishes")
    # The serve-path readback the executor's own publish wait uses SURVIVES.
    assert hasattr(fleet_cells, "publishes_in_flight")
    assert hasattr(fleet_cells, "publish_durable_progress")
    assert hasattr(executor.Executor, "declares_compile")


def test_the_goal_set_has_exactly_one_goal() -> None:
    """RED before: ``WorkerGoals.mint`` existed, ``MINT_ONLY`` was a module
    constant, and ``drives_mint`` / ``retires_when_mint_completes`` /
    ``tenant_reserve_applies`` / ``wire_declaration`` were all readable."""
    assert worker_goals.WorkerGoals.__struct_fields__ == ("serve",)
    assert worker_goals.SERVE_ONLY.serve_admitted() is True
    for gone in ("MINT_ONLY", "from_settings", "_DECL_FORGE"):
        assert not hasattr(worker_goals, gone), gone
    for gone in ("mint", "drives_mint", "retires_when_mint_completes",
                 "tenant_reserve_applies", "wire_declaration", "declared",
                 "declaration_understood"):
        assert not hasattr(worker_goals.SERVE_ONLY, gone), gone


# ---------------------------------------------------------------------------
# The env, and the pod the fleet may still boot with it
# ---------------------------------------------------------------------------


def test_a_pod_booted_with_a_stale_worker_mode_serves_normally(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """§4.28's own acceptance. The hub no longer stamps ``WORKER_MODE``, but a
    pod resumed from an old create request may still carry it, and nothing in
    this image may read it as a logic gate (§1.18: envs carry values, never
    decisions).

    RED before: ``Settings.worker_mode`` existed, ``worker_goals.from_settings``
    interpreted ``"forge"`` into ``serve=False``, and
    ``Executor._dispatch`` then REFUSED tenant dispatch on
    ``serve_admitted() is False``.
    """
    monkeypatch.setenv("WORKER_MODE", "forge")
    settings = gw_config.reload_for_test()

    assert not hasattr(settings, "worker_mode")
    assert "WORKER_MODE" not in gw_config.loader._ENV_TO_FIELD
    assert worker_goals.current().serve_admitted() is True


def test_the_env_seal_is_untouched_by_a_stale_mode(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """pgw#846's gate, carried forward: the key axis must not move because a
    pod's launch env still mentions a mode. It never was a sealed knob and it
    must not become one on the way out."""
    monkeypatch.delenv("WORKER_MODE", raising=False)
    gw_config.reload_for_test()
    clean = env_seal.seal_digest(env_seal.effective_seal())

    monkeypatch.setenv("WORKER_MODE", "forge")
    gw_config.reload_for_test()
    stale = env_seal.seal_digest(env_seal.effective_seal())

    assert stale == clean

    from gen_worker import settings_authority as sa

    assert not any("WORKER_MODE" in str(k) for k in sa.DECLARED_TORCH)


# ---------------------------------------------------------------------------
# The tree-level sweep — th#1751 §2.5's wave-4 acceptance, worker half
# ---------------------------------------------------------------------------


def test_the_word_forge_survives_only_as_the_english_verb() -> None:
    """th#1751 §2.5: *"``git grep -nw forge`` across both trees returns only
    the English verb, migration comments and historical tracker prose — zero
    live producers, zero live readers."*

    The allowlist below is the English verb and the ZERO-DOWNLOAD FORGE
    (``meta_instantiation`` / ``models.meta_init`` / ``structure_only``), which
    is an unrelated pgw#1080 concept — a structure-only instantiation, not a
    pod class. Anything else is a live forge reference and this row says so.

    RED before: ``mint_goal.py``, ``worker_goals._DECL_FORGE``,
    ``aot_compile_pool.FORGE_RSS_RESERVE_BYTES`` and
    ``config.settings.worker_mode``'s docstring all matched.
    """
    allowed_files = {
        # the zero-download forge — structure, not a pod class
        "src/gen_worker/meta_instantiation.py",
        "src/gen_worker/models/meta_init.py",
        "src/gen_worker/models/structure_only.py",
        "src/gen_worker/api/compile_axis.py",
        "src/gen_worker/dist_records.py",
        "src/gen_worker/warmup.py",
    }
    out = subprocess.run(
        ["git", "grep", "-nw", "-i", "forge", "--", "src/"],
        cwd=_ROOT, capture_output=True, text=True).stdout
    live = [
        line for line in out.splitlines()
        if line and line.split(":", 1)[0] not in allowed_files
    ]
    assert not live, "live `forge` references survive the §4.28 cut:\n" + \
        "\n".join(live)
