#!/usr/bin/env python3
"""Cache a serving release's published CONFIG documents as an authoring fixture.

pgw#1346. Authoring a ``ModelSpec`` needs the checkpoint's own published
configuration twice over, and neither half may be guessed:

* the **architecture** block is class-level truth — it is what makes two
  checkpoints one model or two;
* the **scheduler** block rides the export digest, so, in
  ``catalog/flux1_dev.py``'s own words, "a re-declared schedule changes the
  family's identity instead of silently changing every request."

Neither is in this repo, and deliberately so: the endpoints carry no checkpoint
ref because ie#524/th#980 made it a deploy-time binding. So the source is the
SERVING RELEASE ITSELF, read through the hub the endpoints already resolve
against — not a gated third party, which would re-point authoring at someone
else for values our own hub already validates.

**This fetches CONFIGS, never weights.** Every caller names the exact paths it
wants, each must be a ``.json`` document under :data:`MAX_CONFIG_BYTES`, and a
tensor-bearing path is refused by name. The weights-locality rule is not being
excused here — a few hundred bytes of JSON is simply not a weight, and the
fence below is what keeps that true as this script is reused.

Usage::

    scripts/fetch_model_configs.py tests/fixtures/flux1_schnell \\
        --why "FLUX.1-schnell's architecture and schedule, for the catalog" \\
        --release tensorhub/flux1-schnell@prod:schnell \\
        --file transformer/config.json --file scheduler/scheduler_config.json

``--release`` may be repeated; ``--file`` applies to the release that precedes
it. The result is the config documents plus a ``PROVENANCE.json`` recording the
repo, release, checkpoint id, byte size and digest of every one — which
:func:`verify_fixture` re-checks, so a silently edited fixture fails a test
instead of quietly re-keying a family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

#: A config document is small. The cap is deliberately far above any real one
#: and far below anything tensor-bearing, so it is a fence and not a tuning
#: knob.
MAX_CONFIG_BYTES = 256 * 1024

#: Path fragments that mean "this entry carries weights". Refused by name even
#: when the extension looks innocent — `model.safetensors.index.json` is a
#: shard map, not a config, and nothing here has a reason to want one.
WEIGHT_MARKERS = (".safetensors", ".bin", ".pth", ".pt", ".gguf", ".onnx", ".index.json")

PROVENANCE = "PROVENANCE.json"


class FetchRefused(RuntimeError):
    """A refusal with a reason, never a silent skip."""


def _check_path(path: str) -> None:
    lowered = path.lower()
    if not lowered.endswith(".json"):
        raise FetchRefused(
            f"{path!r} is not a .json config document. This script caches "
            "CONFIGS; anything else belongs on a pod, not on an authoring box."
        )
    for marker in WEIGHT_MARKERS:
        if marker in lowered:
            raise FetchRefused(
                f"{path!r} looks weight-bearing ({marker!r}). Refused by name: a "
                "shard index is not a config and no declaration needs one."
            )


def _fixture_name(role: str, path: str) -> str:
    """``transformer/config.json`` under role ``turbo`` -> ``turbo.transformer.config.json``.

    Flattened because two releases of one family contribute files at the same
    tree path, and a nested fixture directory would hide that they differ.
    """

    flat = path.replace("/", ".")
    return f"{role}.{flat}" if role else flat


def fetch_release(
    *,
    owner: str,
    repo: str,
    release: str,
    role: str,
    paths: Sequence[str],
    out_dir: Path,
    base_url: str,
    token: str = "",
) -> Dict[str, Any]:
    """Fetch one release's named configs into ``out_dir``; return its provenance row."""

    from gen_worker.models.hub_client import resolve_repo
    from gen_worker.models.refs import TensorhubRef

    for path in paths:
        _check_path(path)

    resolved = resolve_repo(
        TensorhubRef(owner=owner, repo=repo, release=release),
        base_url=base_url,
        token=token or None,
    )
    by_path = {f.path: f for f in resolved.files}

    import requests

    files: Dict[str, Any] = {}
    for path in paths:
        entry = by_path.get(path)
        if entry is None:
            raise FetchRefused(
                f"{owner}/{repo}@{release} publishes no {path!r}. It has: "
                f"{sorted(p for p in by_path if p.endswith('.json'))!r}"
            )
        size = int(getattr(entry, "size_bytes", 0) or 0)
        if size > MAX_CONFIG_BYTES:
            raise FetchRefused(
                f"{path!r} is {size} bytes, over the {MAX_CONFIG_BYTES}-byte config "
                "cap. That is not a config document."
            )
        url = getattr(entry, "url", "") or ""
        if not url:
            raise FetchRefused(
                f"{path!r} resolved with no whole-file URL (it is chunked), which a "
                "config never is — refusing rather than reassembling."
            )
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        raw = resp.content
        if len(raw) > MAX_CONFIG_BYTES:
            raise FetchRefused(f"{path!r} downloaded {len(raw)} bytes, over the cap")
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        declared = str(getattr(entry, "digest", "") or "").strip().lower()
        if declared and declared.removeprefix("sha256:") != digest.removeprefix("sha256:"):
            raise FetchRefused(
                f"{path!r} hashed to {digest} but the hub declared {declared}. "
                "Refusing a document the catalog does not vouch for."
            )
        # Parsed before writing: a fixture that is not valid JSON is a fixture
        # nobody can read a number out of.
        json.loads(raw.decode("utf-8"))
        name = _fixture_name(role, path)
        (out_dir / name).write_bytes(raw)
        files[name] = {
            "path_in_tree": path,
            "size_bytes": len(raw),
            "digest": digest,
        }

    return {
        "role": role,
        "repo": f"{owner}/{repo}",
        "release": release,
        "checkpoint_id": str(getattr(resolved, "snapshot_digest", "") or ""),
        "files": files,
    }


def verify_fixture(fixture_dir: Path) -> Tuple[int, List[str]]:
    """Re-hash every cached config against ``PROVENANCE.json``.

    Returns ``(checked, problems)``. The one function every lane's tests call,
    so "the fixture is what it says it is" has ONE implementation rather than a
    copy per family that can drift into a weaker check.
    """

    problems: List[str] = []
    provenance_path = fixture_dir / PROVENANCE
    if not provenance_path.is_file():
        return 0, [f"{fixture_dir} holds no {PROVENANCE}"]
    document = json.loads(provenance_path.read_text())

    recorded: Dict[str, Any] = {}
    for release in document.get("releases") or []:
        for name, row in (release.get("files") or {}).items():
            recorded[name] = row

    checked = 0
    for name, row in sorted(recorded.items()):
        path = fixture_dir / name
        if not path.is_file():
            problems.append(f"{name}: recorded in {PROVENANCE} but absent")
            continue
        raw = path.read_bytes()
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        if digest != row.get("digest"):
            problems.append(f"{name}: hashes to {digest}, {PROVENANCE} says {row.get('digest')}")
        if len(raw) != int(row.get("size_bytes", -1)):
            problems.append(f"{name}: is {len(raw)} bytes, {PROVENANCE} says {row.get('size_bytes')}")
        checked += 1

    # The reverse direction, which is the one that catches a weight sneaking in:
    # a file in the directory that no provenance row vouches for.
    for path in sorted(fixture_dir.iterdir()):
        if path.name == PROVENANCE or not path.is_file():
            continue
        if path.name not in recorded:
            problems.append(f"{path.name}: present but no {PROVENANCE} row vouches for it")
        try:
            _check_path(path.name.rsplit(".", 2)[-2] + "." + path.name.rsplit(".", 1)[-1])
        except FetchRefused as exc:
            problems.append(f"{path.name}: {exc}")

    return checked, problems


def _parse_release(spec: str) -> Tuple[str, str, str, str]:
    """``tensorhub/flux1-schnell@prod:schnell`` -> (owner, repo, release, role)."""

    body, _, role = spec.partition(":")
    repo_part, _, release = body.partition("@")
    owner, _, repo = repo_part.partition("/")
    if not owner or not repo:
        raise SystemExit(f"--release {spec!r} must look like owner/repo@release[:role]")
    return owner, repo, release or "prod", role


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fixture_dir", type=Path)
    parser.add_argument("--why", required=True, help="why this fixture exists, for the record")
    parser.add_argument("--release", action="append", default=[], metavar="OWNER/REPO@REL[:ROLE]")
    parser.add_argument("--file", action="append", default=[], metavar="PATH",
                        help="config path; applies to the --release before it")
    parser.add_argument("--base-url", default="http://127.0.0.1:31550")
    parser.add_argument("--token", default="", help="only needed for a private repo")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)

    if args.verify_only:
        checked, problems = verify_fixture(args.fixture_dir)
        for problem in problems:
            print(f"  - {problem}")
        print(f"{args.fixture_dir}: {checked} config(s) verified, {len(problems)} problem(s)")
        return 1 if problems else 0

    # Re-walk argv so --file binds to the --release it follows.
    groups: List[Tuple[str, List[str]]] = []
    for token in (argv if argv is not None else sys.argv[1:]):
        if token.startswith("--release="):
            groups.append((token.split("=", 1)[1], []))
        elif token.startswith("--file="):
            if not groups:
                raise SystemExit("--file before any --release")
            groups[-1][1].append(token.split("=", 1)[1])
    if not groups:
        raise SystemExit("pass --release=owner/repo@rel:role and --file=path/config.json")

    args.fixture_dir.mkdir(parents=True, exist_ok=True)
    releases = []
    for spec, paths in groups:
        owner, repo, release, role = _parse_release(spec)
        if not paths:
            raise SystemExit(f"--release {spec!r} names no --file")
        print(f"resolving {owner}/{repo}@{release} ({len(paths)} config(s))")
        releases.append(
            fetch_release(
                owner=owner, repo=repo, release=release, role=role, paths=paths,
                out_dir=args.fixture_dir, base_url=args.base_url, token=args.token,
            )
        )

    document = {
        "why": args.why,
        "not_weights": (
            "Every file here is a config document. No tensor bytes were fetched, and "
            "none may be added: scripts/fetch_model_configs.py refuses a non-.json or "
            "weight-bearing path by name, and verify_fixture() refuses a file no "
            "provenance row vouches for."
        ),
        "source": {
            "kind": "tensorhub repo catalog",
            "base_url": args.base_url,
            "route": "GET /api/v1/repos/{owner}/{repo}/resolve -> per-file presigned GETs",
            "tool": "scripts/fetch_model_configs.py",
        },
        "releases": releases,
    }
    (args.fixture_dir / PROVENANCE).write_text(json.dumps(document, indent=2) + "\n")

    total = sum(len(r["files"]) for r in releases)
    size = sum(row["size_bytes"] for r in releases for row in r["files"].values())
    print(f"wrote {total} config(s), {size} bytes, into {args.fixture_dir}")
    checked, problems = verify_fixture(args.fixture_dir)
    for problem in problems:
        print(f"  - {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
