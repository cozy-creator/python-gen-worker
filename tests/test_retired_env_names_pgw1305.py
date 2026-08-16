"""pgw#1305 arm (1): four retired env NAMES leave ``_OWNED_NON_SETTINGS``.

pgw#1270 deleted the four read sites and then held the NAMES on a predicate —
*"retire these names only on a fresh measured count of zero"* — justified by a
stated hazard: **dropping a declared name makes a deployed pod fail config
load**. This file measures the MECHANISM instead of the count, because the
hazard is what makes a count matter, and the hazard is not real.

``UnknownSettingError`` is raised only from ``_normalize_key(strict=True)``,
and its only callers are the FILE sources — ``.env``, yaml, ``/run/secrets``.
The process environment never reaches it, and the hub delivers a release's
declared env ONLY as container process env (RunPod ``CreatePodRequest.Env``,
Vast ``CreateInstanceRequest.Env``, k8s inline ``corev1.EnvVar``, local
``docker run -e``); it writes no config FILE into a worker container.

So retiring a name cannot fail a pod's config load. It moves the name from
"silently swallowed" to "named at boot by ``unrecognised_owned_env``", which is
the outcome retiring a name is supposed to produce.
"""

from __future__ import annotations

import pytest

from gen_worker.config import loader as loader_mod
from gen_worker.config.loader import (
    UnknownSettingError,
    load_settings,
    unrecognised_owned_env,
)

#: The four pgw#1270 held. Their read sites died with the modules TCG replaced.
RETIRED = (
    "GEN_WORKER_AOT_RUN_IMPL_SPLIT_OFF",
    "GEN_WORKER_AOT_HOST_COMPILE_JOBS",
    "GEN_WORKER_MINT_RESUME_DIR",
    "GEN_WORKER_MINT_RESUME_MAX_BYTES",
)


@pytest.mark.parametrize("name", RETIRED)
def test_the_retired_name_is_no_longer_accepted_as_owned_non_settings(name: str) -> None:
    """RED before pgw#1305: every one of these was in the frozenset."""
    assert name not in loader_mod._OWNED_NON_SETTINGS, (
        f"{name} still declared as owned-non-settings; its read site died with "
        f"pgw#1270 and nothing reads it"
    )
    assert name not in loader_mod._ENV_TO_FIELD
    assert name not in loader_mod._ENV_ALIASES


@pytest.mark.parametrize("name", RETIRED)
def test_a_pod_still_declaring_a_retired_name_boots_and_the_name_is_NAMED(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole predicate, discharged: the hub hands a pod its declared env as
    PROCESS env, and a retired name arriving that way does not refuse — it is
    reported. RED before pgw#1305 on the second assertion: the name was in
    ``_OWNED_NON_SETTINGS``, so ``unrecognised_owned_env`` filtered it out and
    a pod declaring a dead name was told nothing at all.
    """
    monkeypatch.setenv(name, "1")

    load_settings()  # must not raise — this is the claimed fleet-killer

    assert name in unrecognised_owned_env(), (
        f"{name} arrived in the process env and was neither read nor NAMED; "
        f"a retired name has to be visible at boot or it is just inert residue"
    )


def test_the_instrument_can_go_red_a_file_source_DOES_refuse(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves the two halves of the asymmetry are really different code paths:
    the same unknown owned-namespace name that the process env merely REPORTS
    is a hard refusal from a hand-authored file source. Without this, the test
    above could pass because nothing ever refuses anything.
    """
    env_file = tmp_path / ".env"
    env_file.write_text("GEN_WORKER_AOT_RUN_IMPL_SPLIT_OFF=1\n", encoding="utf-8")
    monkeypatch.setattr(loader_mod, "_DOTENV_PATH", str(env_file))

    with pytest.raises(UnknownSettingError) as exc:
        load_settings()
    assert "GEN_WORKER_AOT_RUN_IMPL_SPLIT_OFF" in str(exc.value)


def test_no_read_site_survives_for_any_retired_name() -> None:
    """A name is retired only if nothing reads it. Text fence over the package,
    not a symbol grep: a read can be `os.environ[...]` with the literal.
    """
    import pathlib

    src = pathlib.Path(loader_mod.__file__).resolve().parents[1]
    offenders: list[str] = []
    for py in src.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="replace")
        for name in RETIRED:
            if name in text:
                offenders.append(f"{py}: {name}")
    assert not offenders, f"retired env names still named in source: {offenders}"
