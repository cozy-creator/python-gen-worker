"""``gen-worker publish`` — push this endpoint's source to the hub; it builds.

pgw#1491. The endpoint uploads ITSELF: source plus the committed
``endpoint.lock``, to ``POST /api/v1/endpoints/:org/:name/releases``. The hub
builds the image. Releases are born only from source — there is no path that
uploads an image.

## The wheel preflight, and why it is not optional

Before a single byte goes over the wire this runs ``uv build --wheel`` into a
throwaway directory. It costs about five seconds, needs no network, no GPU and
no compile — and it is the only thing that can catch an entire class of failure
before the hub pays for it. se#775 burned a **6h28m** image build that died at
step 23/23 because ``allow-direct-references`` sat under ``[tool.uv]`` (not a
uv key — a warning, silently unapplied), so hatchling refused the wheel's
``git+https://`` SDK riders. Every endpoint riding uncut SDK commits has that
exact shape, and a source or editable install NEVER invokes the build backend's
``validate_fields()``, so no CPU drive test can reach it. Only actually
building a wheel can. Skippable with ``--no-preflight``, which is there so the
flag shows up in the log of anyone who chose to skip it.

## What the hub does with the lock — and it is load-bearing in two places

The lock is INGESTED, at both ends of the build (verified with the th#2162
owner against the deployed hub pin, not read off master):

* **at publish**, the uploaded tarball's ``endpoint.lock`` is captured and its
  ``[derive]`` table parsed and digest-verified, which decides ONE thing —
  whether a derive step is rendered into the build at all. With a committed
  lock the build pays no trace;
* **from the built image**, the stamped graph document is read back out of
  ``/app/endpoint.lock`` and written onto the release rows.

Without a committed lock the build regenerates one itself — "regenerate when
missing" is LEGAL and this command warns rather than refuses, because refusing
would turn a ruled-supported publish into an error whose only remedy is a trace
the author may not want to pay at that moment. Including the lock when there is
one is what makes the build cheap.

A correction worth recording, because the method failed and not just the
answer: this module first said the hub ignored the lock, on the strength of
``grep 'case "endpoint.lock"'`` returning nothing and a
``DeriveAtBakeArmed = false`` constant. The inspector matches a CONSTANT
(``CommittedLockFileName``), not the literal, and the disarm constant had been
DELETED — a grep for a string value cannot see either. Absence of a literal is
not absence of a mechanism.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import endpoint_lock as el
from .login import LoginError, require_credential

#: Hub-side cap on a source upload. Checked here so a 6-minute upload does not
#: end in a 413 the user could have been told about instantly.
MAX_SOURCE_BYTES = 100 * 1024 * 1024

HTTP_TIMEOUT_S = 600.0

#: Never uploaded, whatever git says: weights belong in repos, venvs and caches
#: are machine-local, and a .env is a secret someone forgot to gitignore.
_EXCLUDE_DIRS = frozenset({
    # Build/tool detritus.
    ".venv", "venv", "__pycache__", ".git", ".mypy_cache", ".pytest_cache",
    ".tox", ".ruff_cache", "node_modules",
    # A LOCAL RUN'S OUTPUT. `outputs` is still a cwd-relative default (media
    # from `run`). `.compiled-graphs` is NO LONGER ONE — pgw#1526 moved the
    # artifacts default to the box cache (`cli/workspace.artifacts_root`), so
    # the only way it appears in a source tree now is an explicit
    # `--artifacts-dir .compiled-graphs`. It stays in this set anyway: this is
    # a FLOOR, and a floor that only holds for the default is not a floor.
    # Measured via cozy-local's twin of this set (cl#88): 172 MB of artifacts
    # took one tarball 75 KB -> 59 MB.
    ".compiled-graphs", "outputs",
    # CREDENTIAL DIRECTORIES. Not bloat — a secret-leak vector. `publish`
    # uploads a whole tree when the endpoint is not a git work tree, and an
    # `.ssh/id_ed25519` sitting beside an endpoint would go to the hub with
    # it. cozy-local's archiver has excluded these since it was written and
    # asserts it in a test; this client did not, and the two clients publish
    # the same trees.
    ".aws", ".azure", ".gnupg", ".kube", ".secrets", ".ssh",
})
_EXCLUDE_SUFFIXES = (".safetensors", ".ckpt", ".bin", ".pt", ".pth", ".gguf",
                     ".onnx", ".so", ".pyc")
_EXCLUDE_NAMES = frozenset({".env"})


class PublishError(RuntimeError):
    """Publish could not proceed. Always names the step that refused."""


def _tracked_files(root: Path) -> Optional[List[Path]]:
    """What git tracks, or ``None`` outside a work tree.

    A release is built from a commit, so the manifest is what git tracks — not
    whatever happens to be sitting in the directory. Outside a repo the whole
    tree is packaged (minus the exclusions), because there is no commit to
    speak of.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True, check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    names = [n for n in completed.stdout.decode("utf-8").split("\0") if n]
    return [root / name for name in names]


def _excluded(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part in _EXCLUDE_DIRS for part in relative.parts):
        return True
    if path.name in _EXCLUDE_NAMES:
        return True
    return path.suffix.lower() in _EXCLUDE_SUFFIXES


def source_files(root: Path) -> List[Path]:
    tracked = _tracked_files(root)
    candidates = tracked if tracked is not None else [
        p for p in root.rglob("*") if p.is_file()
    ]
    files = [p for p in candidates if p.is_file() and not _excluded(p, root)]
    lock = root / el.LOCK_FILENAME
    # The committed lock rides even when git does not track it yet: an author
    # who just ran `gen-worker lock` and has not committed should be told the
    # lock is uncommitted, not silently publish without it.
    if lock.is_file() and lock not in files:
        files.append(lock)
    return sorted(set(files))


def archive(root: Path, files: List[Path]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in files:
            tar.add(str(path), arcname=str(path.relative_to(root)))
    return buffer.getvalue()


def wheel_preflight(root: Path) -> None:
    """``uv build --wheel`` into a temp dir. Raises with the backend's own words."""
    with tempfile.TemporaryDirectory(prefix="gen-worker-preflight-") as out:
        completed = subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", out, str(root)],
            capture_output=True, check=False,
        )
    if completed.returncode == 0:
        return
    stderr = completed.stderr.decode("utf-8", "replace").strip()
    raise PublishError(
        f"wheel preflight FAILED — the hub's image build would die on this "
        f"after paying for the whole CUDA layer, so it refuses here instead "
        f"(about five seconds in).\n\n{stderr}\n\n"
        f"  A source or editable install never runs the build backend's "
        f"validation, which is why a working dev env proves nothing about "
        f"this. Skip with --no-preflight if you know better."
    )


def _endpoint_identity(root: Path, org: str, name: str) -> Tuple[str, str]:
    resolved_name = name
    if not resolved_name:
        try:
            import tomllib

            document = tomllib.loads(
                (root / "pyproject.toml").read_text(encoding="utf-8")
            )
            resolved_name = str((document.get("project") or {}).get("name") or "")
        except (OSError, ValueError):
            resolved_name = ""
    if not resolved_name:
        raise PublishError(
            "cannot tell which endpoint this is: pass --name, or give "
            "pyproject.toml a [project].name"
        )
    if not org:
        raise PublishError("no org: pass --org (or log in, which records one)")
    return org, resolved_name


def upload(
    *, base_url: str, token: str, org: str, name: str, blob: bytes,
    promote: Optional[bool], dev: bool,
) -> Dict[str, Any]:
    query: List[str] = []
    if promote is not None:
        query.append(f"promote={'true' if promote else 'false'}")
    if dev:
        query.append("dev=true")
    url = (
        f"{base_url.rstrip('/')}/api/v1/endpoints/{org}/{name}/releases"
        + ("?" + "&".join(query) if query else "")
    )
    request = urllib.request.Request(url, data=blob, method="POST")
    request.add_header("Content-Type", "application/gzip")
    request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:600]
        raise PublishError(f"hub answered {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise PublishError(f"{url} unreachable: {exc.reason}") from exc
    try:
        return dict(json.loads(body))
    except json.JSONDecodeError:
        return {"raw": body}


def add_subparser(sub: "argparse._SubParsersAction[Any]") -> None:
    parser = sub.add_parser(
        "publish",
        help="Push this endpoint's source (+ endpoint.lock) to the hub to build.",
        description=(
            "Package what git tracks, run the wheel preflight, and upload to "
            "the hub, which builds the image. Releases are born only from "
            "source."
        ),
    )
    parser.add_argument("endpoint_dir", nargs="?", default=".",
                        help="directory holding endpoint.toml (default: .)")
    parser.add_argument("--org", default="", help="owning org (default: the "
                                                  "one the credential names)")
    parser.add_argument("--name", default="",
                        help="endpoint name (default: pyproject [project].name)")
    parser.add_argument("--hub-url", default="", help="hub base URL")
    parser.add_argument("--promote", dest="promote", action="store_true",
                        default=None,
                        help="promote the built release (hub default: yes)")
    parser.add_argument("--no-promote", dest="promote", action="store_false",
                        help="build without promoting")
    parser.add_argument("--dev", action="store_true",
                        help="a content-addressed dev release from the working "
                             "tree (packages untracked files too)")
    # NO --skip-profiling, deliberately. Paul ruled build-time profiling
    # DECOUPLED from the builder (th#2214): builds never rent hardware, "skip"
    # becomes the only build behavior, and profiling becomes an explicit
    # publisher verb (`cozy profile <endpoint>@<version>`) that names its cost
    # before renting a pod. A flag here would be a knob for a coupling that is
    # being deleted, and every endpoint that set it would have to unset it.
    parser.add_argument("--no-preflight", action="store_true",
                        help="skip `uv build --wheel` (see the module docstring "
                             "for the 6h28m reason not to)")
    parser.add_argument("--dry-run", action="store_true",
                        help="package and preflight, upload nothing")
    parser.set_defaults(_handler=run_publish)


def run_publish(args: argparse.Namespace) -> int:
    root = Path(args.endpoint_dir).resolve()
    if not (root / "endpoint.toml").is_file():
        sys.stderr.write(
            f"gen-worker publish: {root} has no endpoint.toml — it is not an "
            f"endpoint directory\n"
        )
        return 2

    lock = root / el.LOCK_FILENAME
    if not lock.is_file():
        # WARN, never refuse. Paul's builder addendum is explicit: "the builder
        # USES the committed lock when present, and REGENERATES it when missing
        # — missing is allowed, not a refusal." A client that refused would
        # turn a legal publish into an error whose only remedy is a trace the
        # author may not want to pay right now.
        sys.stderr.write(
            f"gen-worker publish: WARNING — no {el.LOCK_FILENAME} beside the "
            f"endpoint, so this build will TRACE inside the image (~2 min for "
            f"an sd15-sized endpoint).\n"
            f"  `gen-worker lock` once, commit the result, and every later "
            f"build skips that step.\n"
        )

    try:
        if not args.no_preflight:
            sys.stderr.write("gen-worker publish: wheel preflight...\n")
            wheel_preflight(root)
            sys.stderr.write("gen-worker publish: wheel preflight ok\n")
        files = source_files(root)
        blob = archive(root, files)
    except PublishError as exc:
        sys.stderr.write(f"gen-worker publish: {exc}\n")
        return 1
    if len(blob) > MAX_SOURCE_BYTES:
        sys.stderr.write(
            f"gen-worker publish: the source archive is {len(blob) / 1e6:.1f} MB, "
            f"over the hub's {MAX_SOURCE_BYTES / 1e6:.0f} MB cap. Move large "
            f"artifacts into repos/datasets/media and reference them from "
            f"endpoint.toml.\n"
        )
        return 1

    sys.stderr.write(
        f"gen-worker publish: {len(files)} file(s), {len(blob) / 1024:.0f} KiB "
        + (f"(including {el.LOCK_FILENAME}, so the build skips the trace)\n"
           if lock.is_file() else "(no committed lock; the build traces)\n")
    )
    if args.dry_run:
        return 0

    try:
        credential = require_credential(args.hub_url)
        base_url = args.hub_url or credential.hub_url
        if not base_url:
            raise PublishError("no hub URL: pass --hub-url or set TENSORHUB_URL")
        org, name = _endpoint_identity(root, args.org or credential.org, args.name)
        answer = upload(
            base_url=base_url, token=credential.token, org=org, name=name,
            blob=blob, promote=args.promote, dev=args.dev,
        )
    except (PublishError, LoginError) as exc:
        sys.stderr.write(f"gen-worker publish: {exc}\n")
        return 1
    sys.stdout.write(json.dumps(answer) + "\n")
    build_id = answer.get("BuildID") or answer.get("build_id") or ""
    if build_id:
        sys.stderr.write(f"gen-worker publish: build {build_id} started\n")
    return 0


__all__ = [
    "MAX_SOURCE_BYTES",
    "PublishError",
    "add_subparser",
    "archive",
    "run_publish",
    "source_files",
    "wheel_preflight",
]
