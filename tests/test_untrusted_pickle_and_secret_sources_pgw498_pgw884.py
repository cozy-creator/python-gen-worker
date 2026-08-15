"""Untrusted bytes and refused secrets, at the two boundaries that must stay
closed.

**PICKLES ARE BANNED — refused always, never read safely.** The convert/clone
lane ingests ARBITRARY tenant-submitted repos, so a `.bin` reaching this process
is hostile by assumption and unpickling it is arbitrary code execution inside a
pod holding hub credentials and other tenants' work. The sanctioned remedy for a
legacy repo is to MIRROR it without the pickle. There is no pickle reader in
this package at all: `classifier.py` refuses a pickle-only repo with
`pickle_only`, and both component-level doors
(`repackage._load_component_state_dict`, `writer.iter_component_tensors`) raise
the same refusal instead of converting. The SHAPE scan that asserts zero
deserializer sites — not "one safe one" — is `scripts/lint_pickle_readers.py`,
in `fast gates` (HARDCUT E5); the two copies that lived here scanned `src/`
only, in a job that gates nothing.

A bare `torch.load` is NOT a substitute. On torch >= 2.6 `weights_only` defaults
to `True`, but that safety is a DEFAULT, not a decision — and torch publishes
the switch that flips it back:

    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

which torch applies *"only if callsite did not explicitly set weights_only"*
(`torch/serialization.py:1496`). One environment variable in a pod turns every
argument-less `torch.load` in the process back into an unpickle, and the
variable is outside this program's owned namespaces, so nothing here would ever
report it. `test_a_poisoned_bin_cannot_execute_code` runs exactly that.

**Secret SOURCES.** The "a pod holding C2PA key material refuses to start" guard
must not read `os.environ` alone. A PEM delivered as
`/run/secrets/GEN_WORKER_C2PA_KEY_PEM`, or as a `.env` / yaml entry, is
otherwise neither loaded (the loader drops the name as "not a Settings field")
nor refused — the pod boots green with a private key world-readable to tenant
code at a mounted path.

Every payload below writes a marker file into a pytest `tmp_path` and does
nothing else. It proves EXECUTION; it does not do anything harmful.
"""

from __future__ import annotations

import pathlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gen_worker.config import loader as loader_mod  # noqa: E402
from gen_worker.convert import repackage, writer  # noqa: E402


# ---------------------------------------------------------------------------
# The untrusted-pickle boundary in the convert lane
# ---------------------------------------------------------------------------


class _MarkerPayload:
    """A pickle whose reconstruction calls `Path.write_text`.

    `__reduce__` returns the (callable, args) pair; the callable runs at
    UNPICKLE time, not at pickle time, so building and saving this object has
    no side effect at all. Only reading it back does.
    """

    def __init__(self, marker: Path) -> None:
        self._marker = marker

    def __reduce__(self) -> tuple:  # type: ignore[type-arg]
        return (pathlib.Path.write_text, (self._marker, "arbitrary code ran"))


def _poisoned_component(tmp_path: Path) -> tuple[Path, Path]:
    """A diffusers-shaped component dir whose only weight file is a torch
    checkpoint carrying the payload. Returns (component_dir, marker_path)."""
    component = tmp_path / "unet"
    component.mkdir(parents=True)
    marker = tmp_path / "EXECUTED.txt"
    # torch.save writes its real container format, so the pre-fix call path
    # loads this file successfully rather than dying on a header check.
    torch.save({"weight": _MarkerPayload(marker)}, component / "diffusion_pytorch_model.bin")
    return component, marker


def test_the_payload_is_inert_until_it_is_read(tmp_path: Path) -> None:
    """Sanity: building and saving the fixture executes nothing. Without this
    a green result below could mean the harness never armed."""
    component, marker = _poisoned_component(tmp_path)
    assert (component / "diffusion_pytorch_model.bin").is_file()
    assert not marker.exists()


def test_the_payload_really_does_execute_through_an_unguarded_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The RED half, pinned as a fact about torch rather than about us.

    This is the exact call `repackage.py` used to make, under the exact pod
    environment that makes it unsafe. It executes the payload. If this ever
    stops being true the guard below is measuring nothing and must be re-cut,
    which is why the negative result is not left implicit.
    """
    component, marker = _poisoned_component(tmp_path)
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    with pytest.warns(UserWarning, match="TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"):
        torch.load(str(component / "diffusion_pytorch_model.bin"),  # pickle-ban: proves-the-refusal
                   map_location="cpu")
    assert marker.exists(), (
        "the payload did not run — torch no longer honours "
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD, so this file's guard is stale"
    )


def test_a_poisoned_bin_cannot_execute_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production path refuses, and nothing runs.

    Same bytes, same environment, through `_load_component_state_dict` —
    reached from the clone lane's layout-repackage branch
    (`convert/clone.py:335-350`) on arbitrary cloned upstream sources.
    """
    component, marker = _poisoned_component(tmp_path)
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    with pytest.raises(writer.ConversionImplementationError) as excinfo:
        repackage._load_component_state_dict(
            component,
            safetensors_bases=("diffusion_pytorch_model",),
            bin_base="diffusion_pytorch_model",
        )

    assert not marker.exists(), "a tenant-supplied pickle executed code in this process"
    # A named refusal, not a generic load failure. The file is never opened at
    # all: the refusal is that it IS a pickle, not that its bytes are hostile.
    assert "pickle_only" in str(excinfo.value)


def test_even_a_benign_bin_is_refused(tmp_path: Path) -> None:
    """The ban has NO benign arm — that is the whole point of it.

    A pickle carrying nothing but tensors is refused exactly like a poisoned
    one, because the safe/unsafe distinction can only be drawn by READING the
    file, which is the act being banned. The conversion endpoint still declares
    `bin_base` for seven families
    (`training-endpoints/.../declarations/families.py`); those declarations are
    now inert and their repos must be mirrored without the pickle.
    """
    component = tmp_path / "unet"
    component.mkdir(parents=True)
    torch.save(
        {"a.weight": torch.zeros(2, 3), "a.bias": torch.ones(3)},
        component / "diffusion_pytorch_model.bin",
    )

    with pytest.raises(writer.ConversionImplementationError) as excinfo:
        repackage._load_component_state_dict(
            component,
            safetensors_bases=("diffusion_pytorch_model",),
            bin_base="diffusion_pytorch_model",
        )
    assert "pickle_only" in str(excinfo.value)
    # The refusal must name the remedy, not just the rule.
    assert "mirror" in str(excinfo.value).lower()


def test_safetensors_still_wins_over_the_pickle(tmp_path: Path) -> None:
    """Preference order is unchanged: a component carrying both loads the
    safetensors and never opens the pickle at all."""
    component, marker = _poisoned_component(tmp_path)
    from safetensors.torch import save_file

    save_file({"a.weight": torch.zeros(1)}, str(component / "diffusion_pytorch_model.safetensors"))

    state = repackage._load_component_state_dict(
        component,
        safetensors_bases=("diffusion_pytorch_model",),
        bin_base="diffusion_pytorch_model",
    )
    assert list(state) == ["a.weight"]
    assert not marker.exists()


# ---------------------------------------------------------------------------
# refused key material, at every source that can deliver it
# ---------------------------------------------------------------------------


_PEM = "-----BEGIN PRIVATE KEY-----\nnot-a-real-key\n-----END PRIVATE KEY-----\n"


@pytest.mark.parametrize("key", sorted(loader_mod.REFUSED_KEY_MATERIAL))
def test_key_material_mounted_at_run_secrets_refuses(tmp_path: Path, key: str) -> None:
    """RED for pgw#884's headline: a PEM at `/run/secrets/<KEY>` used to boot
    green, because `_normalize_key` dropped the name before anything looked at
    it and the th#1307 refusal only ever read `os.environ`."""
    (tmp_path / key).write_text(_PEM, encoding="utf-8")
    with pytest.raises(loader_mod.RefusedKeyMaterialError) as excinfo:
        loader_mod._load_secrets_dir(str(tmp_path))
    assert key in str(excinfo.value)


@pytest.mark.parametrize("key", sorted(loader_mod.REFUSED_KEY_MATERIAL))
def test_key_material_in_yaml_refuses(tmp_path: Path, key: str) -> None:
    cfg = tmp_path / "gen-worker.yaml"
    cfg.write_text(f"{key}: {_PEM!r}\n", encoding="utf-8")
    with pytest.raises(loader_mod.RefusedKeyMaterialError):
        loader_mod._load_yaml([str(cfg)])


@pytest.mark.parametrize("key", sorted(loader_mod.REFUSED_KEY_MATERIAL))
def test_key_material_in_dotenv_refuses(tmp_path: Path, key: str) -> None:
    env = tmp_path / ".env"
    env.write_text(f"{key}=inline-key-material\n", encoding="utf-8")
    with pytest.raises(loader_mod.RefusedKeyMaterialError):
        loader_mod._load_dotenv(str(env))


def test_load_settings_dies_on_a_mounted_pem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole boot, not one source reader: the acceptance is that nothing
    tenant-facing comes up beside the key."""
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "GEN_WORKER_C2PA_KEY_PEM").write_text(_PEM, encoding="utf-8")
    monkeypatch.setattr(loader_mod, "_SECRETS_DIR", str(secrets))
    monkeypatch.setattr(loader_mod, "_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.setattr(loader_mod, "_YAML_CANDIDATE_PATHS", ())
    with pytest.raises(loader_mod.RefusedKeyMaterialError):
        loader_mod.load_settings()


def test_an_ordinary_secret_still_mounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal is keyed on the two names, not on `/run/secrets` existing —
    a mounted `HF_TOKEN` is still an ordinary, loadable secret."""
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "HF_TOKEN").write_text("hf_abc\n", encoding="utf-8")
    monkeypatch.setattr(loader_mod, "_SECRETS_DIR", str(secrets))
    monkeypatch.setattr(loader_mod, "_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.setattr(loader_mod, "_YAML_CANDIDATE_PATHS", ())
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert loader_mod.load_settings().hf_token == "hf_abc"


def test_the_env_arm_is_still_the_ratchet(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader arm ADDS to th#1307's env refusal; it does not replace it.
    Pod env is the hub's only delivery vector today, and that arm is checked
    at every read of the signing state, not once at boot."""
    from gen_worker import content_credentials

    monkeypatch.setenv("GEN_WORKER_C2PA_KEY_PEM", _PEM)
    with pytest.raises(content_credentials.C2paSigningError):
        content_credentials._refuse_pod_private_key_material()


def test_refused_names_are_known_to_the_process_env_census() -> None:
    """They must not read as unrecognised owned-namespace names: they are
    recognised precisely, and refused."""
    assert set(loader_mod.REFUSED_KEY_MATERIAL) <= loader_mod._OWNED_NON_SETTINGS


# ---------------------------------------------------------------------------
# pgw#884 box 3 — the pgw#848 sweep test was blind to a rename here
# ---------------------------------------------------------------------------


def test_worker_credential_reads_the_literal_field_name() -> None:
    """`worker_credential` is the ONE sanctioned reader of the boot token. It
    used to reach it as `getattr(get_settings(), "bootstrap_worker_jwt", "")`
    inside a bare `except Exception: return ""` — the C8 shape that swallows
    exactly the `AttributeError` pgw#848's rename existed to raise. pgw#931
    replaced it with direct attribute access; this pins that, because the
    pgw#848 sweep test greps `settings.worker_jwt` and would never have seen
    a getattr on the NEW name.
    """
    import ast
    import inspect

    from gen_worker import worker_credential

    src = inspect.getsource(worker_credential)
    assert "settings.bootstrap_worker_jwt" in src
    # AST, not text: the module's own comments narrate the old getattr shape,
    # and a grep that its documentation can turn red is a grep nobody keeps.
    defaulted = [
        node.lineno for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "getattr"
    ]
    assert defaulted == [], (
        f"a defaulted attribute read in worker_credential (line(s) {defaulted}) "
        "re-arms the pgw#848 silent-stale-credential class"
    )
