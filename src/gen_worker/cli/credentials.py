"""USER-GLOBAL hub credentials, shared by every endpoint venv on the machine."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

MACHINE_TOKEN_MARKER = "cozy_st_"

DEFAULT_HUB_NAME = "tensorhub"
DEFAULT_PROFILE = "default"

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


class CredentialError(RuntimeError):
    """The credential store could not be read or written."""


def _settings() -> Any:
    from .. import config

    return config.current_or(config.Settings())


def cozy_home() -> Path:
    """``~/.cozy``, or whatever ``COZY_HOME`` configured."""
    return Path(_settings().cozy_home or (Path.home() / ".cozy"))


def credentials_dir() -> Path:
    return cozy_home() / "credentials.d"


def current_selection() -> tuple[str, str]:
    """``(name, profile)`` from the SHARED ``~/.cozy/config.json`` pointer."""
    try:
        document = json.loads((cozy_home() / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_HUB_NAME, DEFAULT_PROFILE
    if not isinstance(document, dict):
        return DEFAULT_HUB_NAME, DEFAULT_PROFILE
    return (
        str(document.get("current_name") or DEFAULT_HUB_NAME),
        str(document.get("current_profile") or DEFAULT_PROFILE),
    )


def current_hub_url() -> str:
    """The hub URL the shared config names, or ``""``."""
    try:
        document = json.loads((cozy_home() / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(document.get("tensorhub_url") or "") if isinstance(document, dict) else ""


def _encode(component: str) -> str:
    return _UNSAFE.sub(lambda m: f"%{ord(m.group()):02X}", component)


def credential_path(name: str = DEFAULT_HUB_NAME, profile: str = DEFAULT_PROFILE) -> Path:
    return credentials_dir() / f"{_encode(name)}.{_encode(profile)}.json"


@dataclass(frozen=True, slots=True)
class Credential:
    token: str
    org: str = ""
    hub_url: str = ""

    @property
    def is_machine_token(self) -> bool:
        return self.token.startswith(MACHINE_TOKEN_MARKER)


def load(
    name: str = "", profile: str = ""
) -> Optional[Credential]:
    """The stored credential, or the environment's, or None. A configured token (Settings.tensorhub_token) WINS over the file store — a pod is configured by its environment and must not depend on a home directory it does not have."""
    settings = _settings()
    configured_token = (settings.tensorhub_token or "").strip()
    if configured_token:
        return Credential(
            token=configured_token,
            hub_url=(settings.tensorhub_url or "").strip() or current_hub_url(),
        )
    selected_name, selected_profile = current_selection()
    name = name or selected_name
    profile = profile or selected_profile
    path = credential_path(name, profile)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CredentialError(f"{path}: {exc}") from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CredentialError(f"{path} is not readable JSON: {exc}") from exc
    token = str(document.get("token") or "").strip()
    if not token:
        return None
    return Credential(
        token=token,
        org=str(document.get("org") or ""),
        hub_url=str(document.get("hub_url") or "") or current_hub_url(),
    )


def save(
    credential: Credential,
    name: str = "",
    profile: str = "",
) -> Path:
    """Write the credential 0600, atomically."""
    selected_name, selected_profile = current_selection()
    name = name or selected_name
    profile = profile or selected_profile
    directory = credentials_dir()
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, stat.S_IRWXU)
    path = credential_path(name, profile)
    document = {
        "token": credential.token,
        "org": credential.org,
        "hub_url": credential.hub_url,
    }
    fd, tmp = tempfile.mkstemp(dir=str(directory), prefix=".cred-", suffix=".json")
    try:
        os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def clear(name: str = "", profile: str = "") -> bool:
    """Delete the credential."""
    selected_name, selected_profile = current_selection()
    name = name or selected_name
    profile = profile or selected_profile
    try:
        credential_path(name, profile).unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise CredentialError(f"{credential_path(name, profile)}: {exc}") from exc
    return True


__all__ = [
    "DEFAULT_HUB_NAME",
    "DEFAULT_PROFILE",
    "MACHINE_TOKEN_MARKER",
    "Credential",
    "CredentialError",
    "clear",
    "cozy_home",
    "credential_path",
    "credentials_dir",
    "load",
    "save",
]
