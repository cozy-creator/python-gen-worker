"""USER-GLOBAL hub credentials, shared by every endpoint venv on the machine.

pgw#1491. Paul's ruling: ``login`` writes credentials user-global "so one login
serves every endpoint venv on this machine". Each endpoint has its own venv
with its own pinned torch, so a credential stored per-venv would mean logging
in once per endpoint — and N copies of one secret is how they diverge.

## This is the store cozy-local already uses, deliberately

``~/.cozy/credentials.d/<name>.<profile>.json``, mode 0600. Not a new location:
cl#85 requires cozy-local to read the SAME store gen-worker writes, and that
store is where cozy-local's already is. Choosing a different path would create
a second home for one secret, which is the exact defect this module's shape
exists to prevent.

## Machine tokens only — nothing here rotates

The stored credential is a MACHINE TOKEN (``cozy_st_``): long-lived, scoped,
and carrying no refresh token. That is a deliberate correction, not a
convenience. On 2026-08-11 eight profiles on one box shared a single rotating
refresh token; every use invalidated the copy the others held, and 737 of 740
refresh sessions were revoked server-side. The same mechanism is what kills a
session token in the middle of a long ``cozy build``. A credential with nothing
to rotate cannot reproduce either failure — and this module writes a credential
back exactly never, which is a structural absence rather than a flag.

Precedence, and it matters for pods: ``TENSORHUB_TOKEN`` in the environment
WINS over the file store, and its presence makes a missing store a non-error.
A container with the env var and no ``~/.cozy`` at all is a working install.
"""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

#: Marker every machine token carries. Lets a client tell a static credential
#: from a session token without asking the server.
MACHINE_TOKEN_MARKER = "cozy_st_"

DEFAULT_HUB_NAME = "tensorhub"
DEFAULT_PROFILE = "default"

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


class CredentialError(RuntimeError):
    """The credential store could not be read or written."""


def _settings() -> Any:
    """Installed Settings, or a zero-config default.

    §1.18: config is loaded ONCE by the pipeline and PASSED; a module reading
    `os.environ` itself is a second loader that can disagree with the first.
    `current_or` takes the fallback AS A VALUE so the zero-config case is
    visible here rather than hidden in an env read — this module is imported
    both by the CLI (which installs Settings at process entry) and by scripts
    that never bring a worker up.
    """
    from .. import config

    return config.current_or(config.Settings())


def cozy_home() -> Path:
    """``~/.cozy``, or whatever ``COZY_HOME`` configured. The same root
    cozy-local resolves, so one login serves both tools."""
    return Path(_settings().cozy_home or (Path.home() / ".cozy"))


def credentials_dir() -> Path:
    return cozy_home() / "credentials.d"


def current_selection() -> tuple[str, str]:
    """``(name, profile)`` from the SHARED ``~/.cozy/config.json`` pointer.

    Read rather than defaulted, because the store is shared: cozy-local writes
    ``current_name``/``current_profile`` there when a user switches profiles,
    and a gen-worker that ignored it would authenticate as a different identity
    than the tool the user just ran — with both of them reading files out of
    the same directory. Absent config falls back to the plain defaults.
    """
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
    """Percent-encode a path component.

    Without this, ``a/b`` + ``c`` and ``a`` + ``b/c`` collide on one filename
    and ``..`` escapes the directory — both real, both cheap to prevent here.
    """
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
    """The stored credential, or the environment's, or ``None``.

    A CONFIGURED token wins over the file store: a pod is configured by its
    environment and must not depend on a home directory it does not have. It
    arrives through `Settings.tensorhub_token` rather than an `os.environ` read
    here — same precedence, but the value now also comes from `.env`, yaml and
    `/run/secrets` like every other setting, instead of only from the one
    source this module happened to look at. Empty ``name``/``profile`` take the
    shared pointer (:func:`current_selection`), not a local default.
    """
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
    """Write the credential 0600, atomically. Never merges — a login replaces.

    The temp file is created INSIDE the destination directory with 0600 from
    birth, so the secret is never briefly world-readable and never briefly on a
    different filesystem where the rename would degrade to a copy.
    """
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
    """Delete the credential. A logout is a DELETION, never an empty file —
    an empty file reads as "logged in with nothing" to the next reader."""
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
