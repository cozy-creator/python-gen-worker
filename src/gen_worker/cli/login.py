"""``gen-worker login`` / ``logout`` — hub auth, once per machine."""

from __future__ import annotations

import argparse
import getpass
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict

from . import credentials
from .credentials import Credential, CredentialError

HTTP_TIMEOUT_S = 60.0

PUBLISH_SCOPES = ("org:endpoint:upsert", "org:endpoint:list", "org:endpoint:deploy")


class LoginError(RuntimeError):
    """Authentication failed."""


def _hub_url(stated: str) -> str:
    from .. import config

    base = (stated or config.current_or(config.Settings()).tensorhub_url or "").strip()
    if not base:
        raise LoginError(
            "no hub URL: pass --hub-url, or set TENSORHUB_URL "
            "(e.g. https://tensorhub.com)"
        )
    return base.rstrip("/")


def _post(url: str, body: Dict[str, Any], *, bearer: str = "") -> Dict[str, Any]:
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    if bearer:
        request.add_header("Authorization", f"Bearer {bearer}")
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
            return dict(json.loads(response.read().decode("utf-8")))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise LoginError(f"{url} answered {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LoginError(f"{url} unreachable: {exc.reason}") from exc


def _get(url: str, *, bearer: str) -> Dict[str, Any]:
    request = urllib.request.Request(url)
    request.add_header("Authorization", f"Bearer {bearer}")
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
            return dict(json.loads(response.read().decode("utf-8")))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise LoginError(f"{url} answered {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LoginError(f"{url} unreachable: {exc.reason}") from exc


def whoami(base_url: str, token: str) -> Dict[str, Any]:
    """``GET /api/v1/tokens/self`` — the only way a bare token learns its org."""
    return _get(f"{base_url}/api/v1/tokens/self", bearer=token)


def _password_login(base_url: str, login: str, password: str) -> str:
    answer = _post(
        f"{base_url}/api/v1/password/login", {"login": login, "password": password}
    )
    access = str(answer.get("access_token") or "")
    if not access:
        raise LoginError("password login returned no access_token")
    return access


def _mint_machine_token(
    base_url: str, session: str, org: str, name: str
) -> str:
    answer = _post(
        f"{base_url}/api/v1/orgs/{urllib.parse.quote(org)}/tokens",
        {"name": name, "scopes": list(PUBLISH_SCOPES)},
        bearer=session,
    )
    token = str(answer.get("token") or "")
    if not token:
        raise LoginError(
            "the hub minted a machine token but did not return its plaintext "
            "(it is returned exactly once); nothing was stored"
        )
    return token


def add_subparser(sub: "argparse._SubParsersAction[Any]") -> None:
    parser = sub.add_parser(
        "login",
        help="Authenticate to the hub (credentials shared by every endpoint venv).",
        description=(
            "Store a hub credential in ~/.cozy/credentials.d, readable by "
            "every endpoint venv on this machine. Prefers a machine token: "
            "long-lived, scoped, and with nothing that rotates."
        ),
    )
    parser.add_argument("--token", default="",
                        help="a machine token (cozy_st_...) to store directly")
    parser.add_argument("--email", default="",
                        help="log in with a password, then MINT a machine token")
    parser.add_argument("--password", default="",
                        help="password (prompted if omitted)")
    parser.add_argument("--org", default="",
                        help="org to mint the machine token under (default: "
                             "the one the session belongs to)")
    parser.add_argument("--hub-url", default="", help="hub base URL")
    parser.add_argument("--profile", default=credentials.DEFAULT_PROFILE)
    parser.add_argument("--token-name", default="gen-worker",
                        help="label for the minted machine token")
    parser.set_defaults(_handler=run_login)

    out = sub.add_parser(
        "logout",
        help="Forget this machine's stored hub credential.",
        description=(
            "Delete the stored credential. Does NOT revoke it hub-side — "
            "forgetting it here and killing it everywhere are different acts."
        ),
    )
    out.add_argument("--profile", default=credentials.DEFAULT_PROFILE)
    out.set_defaults(_handler=run_logout)


def run_login(args: argparse.Namespace) -> int:
    try:
        base_url = _hub_url(args.hub_url)
    except LoginError as exc:
        sys.stderr.write(f"gen-worker login: {exc}\n")
        return 2
    if not args.token and not args.email:
        sys.stderr.write(
            "gen-worker login: pass --token cozy_st_... (a machine token) or "
            "--email to log in with a password and mint one.\n"
        )
        return 2
    try:
        token = args.token.strip()
        if not token:
            password = args.password or getpass.getpass("password: ")
            session = _password_login(base_url, args.email, password)
            org = args.org or str(whoami(base_url, session).get("org_slug") or "")
            if not org:
                raise LoginError(
                    "could not determine which org to mint the machine token "
                    "under; pass --org"
                )
            token = _mint_machine_token(base_url, session, org, args.token_name)
            sys.stderr.write(
                f"gen-worker login: minted a machine token under {org} "
                f"(the password session was discarded, not stored)\n"
            )
        identity = whoami(base_url, token)
    except LoginError as exc:
        sys.stderr.write(f"gen-worker login: {exc}\n")
        return 1
    credential = Credential(
        token=token,
        org=str(identity.get("org_slug") or args.org or ""),
        hub_url=base_url,
    )
    if not credential.is_machine_token:
        sys.stderr.write(
            "gen-worker login: WARNING — this is not a machine token "
            f"({credentials.MACHINE_TOKEN_MARKER}...). Session tokens expire "
            "and rotate; a long publish can die holding one. Mint a machine "
            "token with `gen-worker login --email ...` instead.\n"
        )
    try:
        path = credentials.save(credential, profile=args.profile)
    except CredentialError as exc:
        sys.stderr.write(f"gen-worker login: {exc}\n")
        return 1
    sys.stderr.write(
        f"gen-worker login: authenticated as {identity.get('kind', 'user')}"
        + (f" in {credential.org}" if credential.org else "")
        + f"; credential stored at {path} (0600)\n"
    )
    return 0


def run_logout(args: argparse.Namespace) -> int:
    try:
        removed = credentials.clear(profile=args.profile)
    except CredentialError as exc:
        sys.stderr.write(f"gen-worker logout: {exc}\n")
        return 1
    sys.stderr.write(
        "gen-worker logout: credential removed\n" if removed
        else "gen-worker logout: nothing was stored\n"
    )
    return 0


def require_credential(hub_url: str = "") -> Credential:
    """The credential a hub-touching verb needs, or a refusal naming login."""
    credential = credentials.load()
    if credential is None:
        raise LoginError(
            "not logged in. Run `gen-worker login --token cozy_st_...` "
            "(or --email to mint one). The credential is stored once per "
            "machine and shared by every endpoint venv."
        )
    if hub_url and not credential.hub_url:
        credential = Credential(
            token=credential.token, org=credential.org, hub_url=hub_url
        )
    return credential


__all__ = [
    "LoginError",
    "PUBLISH_SCOPES",
    "add_subparser",
    "require_credential",
    "run_login",
    "run_logout",
    "whoami",
]
