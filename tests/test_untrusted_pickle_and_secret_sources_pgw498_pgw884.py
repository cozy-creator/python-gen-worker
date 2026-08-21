"""Untrusted bytes and refused secrets, at the two boundaries that must stay closed."""

from __future__ import annotations

import pathlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gen_worker.config import loader as loader_mod  # noqa: E402
from gen_worker.convert import repackage, writer  # noqa: E402


class _MarkerPayload:

    def __init__(self, marker: Path) -> None:
        self._marker = marker

    def __reduce__(self) -> tuple:  # type: ignore[type-arg]
        return (pathlib.Path.write_text, (self._marker, "arbitrary code ran"))


def _poisoned_component(tmp_path: Path) -> tuple[Path, Path]:
    component = tmp_path / "unet"
    component.mkdir(parents=True)
    marker = tmp_path / "EXECUTED.txt"
    torch.save({"weight": _MarkerPayload(marker)}, component / "diffusion_pytorch_model.bin")
    return component, marker


def test_the_payload_is_inert_until_it_is_read(tmp_path: Path) -> None:
    """Sanity: building and saving the fixture executes nothing."""
    component, marker = _poisoned_component(tmp_path)
    assert (component / "diffusion_pytorch_model.bin").is_file()
    assert not marker.exists()


def test_the_payload_really_does_execute_through_an_unguarded_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The RED half, pinned as a fact about torch rather than about us."""
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
    """The production path refuses, and nothing runs."""
    component, marker = _poisoned_component(tmp_path)
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    with pytest.raises(writer.ConversionImplementationError) as excinfo:
        repackage._load_component_state_dict(
            component,
            safetensors_bases=("diffusion_pytorch_model",),
            bin_base="diffusion_pytorch_model",
        )

    assert not marker.exists(), "a tenant-supplied pickle executed code in this process"
    assert "pickle_only" in str(excinfo.value)


def test_even_a_benign_bin_is_refused(tmp_path: Path) -> None:
    """The ban has NO benign arm — that is the whole point of it."""
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
    assert "mirror" in str(excinfo.value).lower()


def test_safetensors_still_wins_over_the_pickle(tmp_path: Path) -> None:
    """Preference order is unchanged: a component carrying both loads the safetensors and never opens the pickle at all."""
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


_PEM = "-----BEGIN PRIVATE KEY-----\nnot-a-real-key\n-----END PRIVATE KEY-----\n"


@pytest.mark.parametrize("key", sorted(loader_mod.REFUSED_KEY_MATERIAL))
def test_key_material_mounted_at_run_secrets_refuses(tmp_path: Path, key: str) -> None:
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
    """The whole boot, not one source reader: the acceptance is that nothing tenant-facing comes up beside the key."""
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
    """The refusal is keyed on the two names, not on `/run/secrets` existing — a mounted `HF_TOKEN` is still an ordinary, loadable secret."""
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "HF_TOKEN").write_text("hf_abc\n", encoding="utf-8")
    monkeypatch.setattr(loader_mod, "_SECRETS_DIR", str(secrets))
    monkeypatch.setattr(loader_mod, "_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.setattr(loader_mod, "_YAML_CANDIDATE_PATHS", ())
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert loader_mod.load_settings().hf_token == "hf_abc"


def test_the_env_arm_is_still_the_ratchet(monkeypatch: pytest.MonkeyPatch) -> None:
    from gen_worker import content_credentials

    monkeypatch.setenv("GEN_WORKER_C2PA_KEY_PEM", _PEM)
    with pytest.raises(content_credentials.C2paSigningError):
        content_credentials._refuse_pod_private_key_material()


def test_refused_names_are_known_to_the_process_env_census() -> None:
    """They must not read as unrecognised owned-namespace names: they are recognised precisely, and refused."""
    assert set(loader_mod.REFUSED_KEY_MATERIAL) <= loader_mod._OWNED_NON_SETTINGS


def test_worker_credential_reads_the_literal_field_name() -> None:
    """`worker_credential` is the ONE sanctioned reader of the boot token."""
    import ast
    import inspect

    from gen_worker import worker_credential

    src = inspect.getsource(worker_credential)
    assert "settings.bootstrap_worker_jwt" in src
    defaulted = [
        node.lineno for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "getattr"
    ]
    assert defaulted == [], (
        f"a defaulted attribute read in worker_credential (line(s) {defaulted}) "
        "re-arms the pgw#848 silent-stale-credential class"
    )
