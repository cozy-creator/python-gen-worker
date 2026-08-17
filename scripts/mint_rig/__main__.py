"""``python -m mint_rig`` — the pod mint-rig from a shell.

    # pgw#1331's owed leg: the smallest real compile, on the cheapest sm_86 card
    python -m mint_rig mint --gpu a40 --rail 2.00 --lane pgw1331-clip \
        --target gen_worker.model.catalog.flux1_dev:FLUX1_DEV --runner clip \
        --deliver sdist --issue pgw#1331

    # the same command against a PUBLISHED release
    python -m mint_rig mint --gpu a40 --rail 2.00 --deliver wheel \
        --spec 'gen-worker[torch]==0.121.0' --target ... --runner clip

    # an arbitrary named command (the general primitive; `mint` is one preset)
    python -m mint_rig run --gpu a40 --rail 1.00 --name adopt \
        --command 'python3 adopt_and_infer.py && echo RIG_DONE' \
        --upload ./adopt_and_infer.py --artifact /root/rig/out

    python -m mint_rig sweep                 # what is running vs what we recorded
    python -m mint_rig terminate --pod <id>  # or --name <pod name>
    python -m mint_rig cards

`--rail` is REQUIRED on anything that rents. That is the point.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import cards as cards_mod
from .driver import Rig, uploads_for
from .rail import Rail
from .row import RigRow
from .workload import POD_ROOT, Upload, Workload, install_sdist, install_wheel, mint_model

REPO = Path(__file__).resolve().parents[2]


def _rig(args: argparse.Namespace) -> Rig:
    return Rig(
        rail=Rail(max_usd=float(args.rail)),
        lane=args.lane,
        issue=getattr(args, "issue", ""),
        out_dir=Path(args.out) if args.out else REPO / "rig-runs",
        tick_s=float(args.tick),
        stall_ticks=int(args.stall_ticks),
        dry_run=bool(getattr(args, "dry_run", False)),
        cloud_type="" if getattr(args, "cloud", "") == "ANY" else getattr(args, "cloud", "SECURE"),
        boot_budget=float(getattr(args, "boot_budget", 0.15)),
    )


def build_sdist(out_dir: Path) -> Path:
    """Build the worktree's own wheel, so the pod runs THIS lane's code.

    Not a convenience: pgw#1337 has the wheel cut blocked, so a lane whose
    surface is newer than PyPI cannot reach a card any other honest way. The
    row records the dist's sha256, so the answer to "which code ran" is a
    digest rather than a claim. Packaging only — nothing compiles here.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.whl"):
        stale.unlink()
    subprocess.run(
        ["nice", "-n", "19", "uv", "build", "--wheel", "--out-dir", str(out_dir)],
        cwd=REPO,
        check=True,
    )
    wheels = sorted(out_dir.glob("*.whl"))
    if not wheels:
        raise RuntimeError(f"uv build produced no wheel in {out_dir}")
    return wheels[-1]


def _delivery(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[Upload, ...]]:
    if args.deliver == "image":
        return (), ()
    if args.deliver == "wheel":
        if not args.spec:
            raise SystemExit("--deliver wheel needs --spec, e.g. 'gen-worker[torch]==0.121.0'")
        return install_wheel(args.spec, args.index_args), ()
    dist = Path(args.dist) if args.dist else build_sdist(REPO / "dist")
    return install_sdist(dist, args.extras, args.index_args), (Upload(local=dist),)


def cmd_mint(args: argparse.Namespace) -> int:
    install, uploads = _delivery(args)
    fleet_line = Path(args.fleet_line) if args.fleet_line else None
    # REFUSED BEFORE RENTING, because this refusal is free and the pod that
    # taught us was not. `gen-worker model mint` asserts the fleet line first
    # (RIG-ENV §1) and rigcheck reads endpoint.toml / fleet-floors.toml /
    # ENDPOINT dist metadata — none of which exist on a pod carrying only this
    # repo's wheel. Only `--deliver image` is exempt: an endpoint image ships
    # its own endpoint.toml, which is the authority.
    if fleet_line is None and args.deliver != "image":
        raise SystemExit(
            "pgw#1347: `mint` needs --fleet-line <endpoint.toml|fleet-floors.toml>. "
            "gen-worker's own torch requirement is NOT an authority (rigcheck refuses "
            "to let the SDK certify its own floor), and this repo ships neither file, so "
            "the mint would abort FleetLineUnknown on a pod you already paid for. The "
            "workspace's authority is ~/cozy/serverless-endpoints/fleet-floors.toml; a "
            "per-endpoint endpoint.toml is stronger, because it also declares CUDA."
        )
    if fleet_line is not None and not fleet_line.is_file():
        raise SystemExit(f"pgw#1347: --fleet-line {fleet_line} does not exist")
    workload = mint_model(
        args.target,
        runners=tuple(args.runner),
        install=install + tuple(args.setup),
        uploads=uploads,
        fleet_line=fleet_line,
        name=args.name,
    )
    row = _rig(args).run(cards_mod.pick(args.gpu), workload, image=args.image)
    return 0 if row.verdict == "green" else 1


def cmd_run(args: argparse.Namespace) -> int:
    install, uploads = _delivery(args)
    workload = Workload(
        name=args.name,
        command=args.command,
        setup=install + tuple(args.setup),
        uploads=uploads + uploads_for([Path(p) for p in args.upload]),
        artifacts=tuple(args.artifact) or (POD_ROOT,),
        progress_paths=tuple(args.progress) or (POD_ROOT,),
    )
    row = _rig(args).run(cards_mod.pick(args.gpu), workload, image=args.image)
    return 0 if row.verdict == "green" else 1


#: `terminate` and `sweep` rent nothing, but `Rig` refuses to exist without a
#: rail — deliberately, so no code path can construct one "just to look". A cent
#: is the smallest honest statement of "this invocation will not buy anything".
_NO_SPEND = 0.01


def cmd_terminate(args: argparse.Namespace) -> int:
    rig = Rig(rail=Rail(max_usd=_NO_SPEND), lane="terminate", out_dir=Path(args.out or "."))
    row: RigRow = rig.terminate(pod_id=args.pod, name=args.name)
    print(json.dumps(row.teardown.__dict__, indent=2))
    return 0 if row.teardown.confirmed else 1


def cmd_sweep(args: argparse.Namespace) -> int:
    rig = Rig(rail=Rail(max_usd=_NO_SPEND), lane="sweep", out_dir=Path(args.out or "."))
    report = rig.sweep()
    print(json.dumps(report, indent=2, sort_keys=True))
    # A live pod that NO record anywhere attends is the failure this package
    # exists to prevent. A sibling lane's attended pod is not that.
    return 1 if report["unattended"] else 0


def cmd_cards(_: argparse.Namespace) -> int:
    for card in cards_mod.CARDS.values():
        print(
            f"{card.slug:8s} sm{card.sm_expected:5s} ~${card.usd_per_hour_hint:5.2f}/hr  "
            f"{'data-center' if card.data_center_part else 'workstation':12s} "
            f"{', '.join(card.gpu_type_ids)}"
        )
    return 0


def _rent_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpu", default="sm86", choices=sorted(cards_mod.CARDS))
    parser.add_argument("--image", default=cards_mod.FLEET_IMAGE)
    parser.add_argument(
        "--cloud",
        default="SECURE",
        choices=("SECURE", "COMMUNITY", "ANY"),
        help="ANY lets the provider choose. SECURE has little sm_86 capacity; a "
        "compile proof carries no weights, so COMMUNITY is a legitimate place to buy one.",
    )
    parser.add_argument("--lane", required=True)
    parser.add_argument("--issue", default="")
    parser.add_argument(
        "--rail",
        required=True,
        help="Spend cap for THIS invocation, in dollars. Mandatory: there is no default.",
    )
    parser.add_argument("--out", default="")
    parser.add_argument("--tick", default="15", help="Seconds between progress observations.")
    parser.add_argument(
        "--stall-ticks",
        default="12",
        help="Consecutive observations with an unchanged progress token that mean STUCK. "
        "Not a timeout: work that is moving is never stopped by this.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write the kill-set and rent nothing.")
    parser.add_argument("--deliver", default="sdist", choices=("sdist", "wheel", "image"))
    parser.add_argument("--spec", default="", help="pip requirement for --deliver wheel.")
    parser.add_argument("--dist", default="", help="Prebuilt dist for --deliver sdist.")
    parser.add_argument("--extras", default="", help="Extras for the shipped dist, e.g. 'torch'.")
    parser.add_argument(
        "--boot-budget",
        default="0.15",
        help="Fraction of the rail bring-up may spend before the pod answers. Bring-up "
        "has no progress signal (RunPod cannot distinguish an image pull from a wedged "
        "host), so money is its only honest bound. Raise it for a big single-blob image.",
    )
    parser.add_argument("--index-args", default="", help="Extra pip index flags.")
    parser.add_argument(
        "--setup",
        action="append",
        default=[],
        help="Extra shell line to run before the command; repeatable, order preserved. "
        "The delivery lines run first.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mint_rig", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    mint = sub.add_parser("mint", help="gen-worker model mint on a rented pod.")
    _rent_flags(mint)
    mint.add_argument("--target", required=True, help="module:ATTR of the GraphModelSpec.")
    mint.add_argument("--runner", action="append", default=[], help="Repeatable; omit for all.")
    mint.add_argument(
        "--fleet-line",
        default="",
        help="endpoint.toml or fleet-floors.toml to ship as the fleet-line authority "
        "(RIG-ENV §2). REQUIRED unless --deliver image. It becomes "
        "GEN_WORKER_FLEET_LINE_FILE on the pod.",
    )
    mint.add_argument("--name", default="mint")
    mint.set_defaults(fn=cmd_mint)

    run = sub.add_parser("run", help="Any named command on a rented pod.")
    _rent_flags(run)
    run.add_argument("--command", required=True, help="Must print RIG_DONE on success.")
    run.add_argument("--name", default="run")
    run.add_argument("--upload", action="append", default=[])
    run.add_argument("--artifact", action="append", default=[])
    run.add_argument("--progress", action="append", default=[])
    run.set_defaults(fn=cmd_run)

    terminate = sub.add_parser("terminate", help="DELETE + 404 + absent-from-list.")
    terminate.add_argument("--pod", default="")
    terminate.add_argument("--name", default="")
    terminate.add_argument("--out", default="")
    terminate.set_defaults(fn=cmd_terminate)

    sweep = sub.add_parser(
        "sweep", help="Live pods vs every record; exit 1 on a pod nobody attends."
    )
    sweep.add_argument("--out", default="")
    sweep.set_defaults(fn=cmd_sweep)

    show = sub.add_parser("cards", help="The card catalogue.")
    show.set_defaults(fn=cmd_cards)

    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
