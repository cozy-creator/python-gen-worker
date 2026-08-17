"""The card catalogue — asked ids, and what the row must record instead.

A `Card` names a SET of RunPod `gpuTypeIds` to ask for and the compute
capability they share. It is a plan input and nothing more:
:class:`~mint_rig.row.RigRow` carries `asked_gpu` and `observed_gpu` as separate
fields because e2e#privatedeploy's matrix learned that conflating them measures
an intention rather than a machine — a set of three ids resolves to whichever
one the provider had, and the sm the row quotes must come from the pod's own
`nvidia-smi`, never from this table.

Prices here are LAST-OBSERVED, for sizing a rail before renting. The rail is
re-armed against the create response's real `costPerHr` the moment it lands.

The image default is the one `research/RIG-ENV.md` §3a settles on for RunPod:
the upstream public `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime`, because
the tensorhub repack is not publicly pullable and RunPod's failure mode for an
unpullable tag is to exit the pod ~1 s after rent with no diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

#: RIG-ENV §3a. Do not replace with the tensorhub repack — see this module's
#: docstring and RIG-ENV's own warning box.
FLEET_IMAGE = "pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime"  # oci-image: RunPod pulls it

#: The CUDA line the fleet is on. `gen_worker.rigboot --cuda` takes it; the
#: value is what RIG-ENV §2 resolves to today, and the pod re-reads the real
#: authority itself via `gen_worker.rigcheck`, so this is only the preflight's
#: argument.
FLEET_CUDA = "13.0"


@dataclass(frozen=True)
class Card:
    """One asked-for card class."""

    slug: str
    #: RunPod `gpuTypeIds`, in preference order. A set, because a single id is
    #: how a lane waits an hour for capacity that a sibling SKU had all along.
    gpu_type_ids: tuple[str, ...]
    #: Compute capability of the class. RECORDED FROM THE POD, never trusted
    #: from here — this is the value the row's `sm_expected` is checked against
    #: so a provider substituting a different card is visible.
    sm_expected: str
    #: Container disk. AOTI object caches and an inductor scratch tree are the
    #: bulk of it, not weights: a model mint downloads no checkpoint (§4.27).
    disk_gb: int
    #: Last-observed whole-pod rate, for sizing a rail before renting.
    usd_per_hour_hint: float
    data_center_part: bool
    note: str = ""


#: Ampere sm_86 first: it is the cheapest place a real AOTI compile can be
#: proven, and pgw#1331's owed leg asks for exactly that.
CARDS: dict[str, Card] = {
    "sm86": Card(
        slug="sm86",
        # MEASURED 2026-08-17: asking for `["NVIDIA A40"]` alone answered
        # HTTP 500 "This machine does not have the resources to deploy your
        # pod" — SECURE A40 capacity, at that hour, in that region. There is no
        # capacity query in the REST API (see PodApi), so the SET is the whole
        # availability strategy: one create call, several acceptable machines,
        # and the row records which one arrived.
        gpu_type_ids=(
            "NVIDIA A40",
            "NVIDIA RTX A5000",
            "NVIDIA RTX A4500",
            "NVIDIA RTX A4000",
            "NVIDIA GeForce RTX 3090",
        ),
        sm_expected="8.6",
        disk_gb=40,
        usd_per_hour_hint=0.30,
        # Conservative: the set MIXES data-center (A40) and workstation parts,
        # and NVIDIA's forward-compat libcuda is only supported on the former.
        # Claiming True here would promise a repair rigboot cannot always make.
        data_center_part=False,
        note="every sm_86 RunPod sells, cheapest-acceptable-first. THE default "
        "for a compile proof: the graph is what is under test, not the card.",
    ),
    "a40": Card(
        slug="a40",
        gpu_type_ids=("NVIDIA A40",),
        sm_expected="8.6",
        disk_gb=60,
        usd_per_hour_hint=0.40,
        data_center_part=True,
        note="sm_86 data-center part: cuda-compat forward-compat is supported, "
        "so rigboot can repair a 570-driver host instead of re-rolling it.",
    ),
    "a4000": Card(
        slug="a4000",
        gpu_type_ids=("NVIDIA RTX A4000", "NVIDIA RTX A4500"),
        sm_expected="8.6",
        disk_gb=60,
        usd_per_hour_hint=0.17,
        data_center_part=False,
        note="cheapest sm_86, but a WORKSTATION part: NVIDIA's forward-compat "
        "libcuda is not supported here, so a 570-driver host is a RE-ROLL, not "
        "a repair. rigboot says which.",
    ),
    "a5000": Card(
        slug="a5000",
        gpu_type_ids=("NVIDIA RTX A5000",),
        sm_expected="8.6",
        disk_gb=60,
        usd_per_hour_hint=0.26,
        data_center_part=False,
    ),
    "sm89": Card(
        slug="sm89",
        # MEASURED 2026-08-17, in the same probe that found sm_86 empty: an Ada
        # card was created on the first try at $0.74/hr while every sm_86 SKU
        # answered out-of-capacity. Ada is the cheapest AVAILABLE place to prove
        # a real compile, which is a different question from the cheapest place.
        gpu_type_ids=(
            "NVIDIA L4",
            "NVIDIA RTX 4000 Ada Generation",
            "NVIDIA GeForce RTX 4090",
            "NVIDIA L40S",
        ),
        sm_expected="8.9",
        disk_gb=40,
        usd_per_hour_hint=0.74,
        data_center_part=False,
        note="every sm_89 RunPod sells, cheapest-first. The FALLBACK when sm_86 "
        "is out of capacity — and on 2026-08-17 it was.",
    ),
    "sm80": Card(
        slug="sm80",
        gpu_type_ids=("NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-40GB", "NVIDIA A30"),
        sm_expected="8.0",
        disk_gb=60,
        usd_per_hour_hint=1.39,
        data_center_part=True,
        note="Ampere data-center. Forward-compat libcuda IS supported here, so a "
        "570-driver host is a repair rather than a re-roll.",
    ),
    "sm90": Card(
        slug="sm90",
        gpu_type_ids=("NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe", "NVIDIA H100 NVL", "NVIDIA H200"),
        sm_expected="9.0",
        disk_gb=100,
        usd_per_hour_hint=3.29,
        data_center_part=True,
    ),
    "sm100": Card(
        slug="sm100",
        gpu_type_ids=("NVIDIA B200",),
        sm_expected="10.0",
        disk_gb=100,
        usd_per_hour_hint=6.81,
        data_center_part=True,
        note="THE EXPENSIVE ONE. pgw#1084 §7.2's rule: nothing larger is bought on "
        "an sm the micro probe has not cleared — which is what makes this a $1.84 "
        "question rather than a $20 one.",
    ),
    "sm120": Card(
        slug="sm120",
        gpu_type_ids=(
            "NVIDIA GeForce RTX 5090",
            "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        ),
        sm_expected="12.0",
        disk_gb=100,
        usd_per_hour_hint=0.89,
        data_center_part=False,
        note="pgw#1084 §8.2 calls rtx-pro-6000 'the only mintable sm_120 SKU' and it "
        "is NOT a registered gpu_sku_set, so sm_120 is bare-pod-only until th#2112 "
        "lands — which is exactly the route this rig takes.",
    ),
    "l40s": Card(
        slug="l40s",
        gpu_type_ids=("NVIDIA L40S", "NVIDIA L40"),
        sm_expected="8.9",
        disk_gb=100,
        usd_per_hour_hint=0.86,
        data_center_part=True,
    ),
    "h100": Card(
        slug="h100",
        gpu_type_ids=("NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe", "NVIDIA H100 NVL"),
        sm_expected="9.0",
        disk_gb=220,
        usd_per_hour_hint=2.39,
        data_center_part=True,
    ),
    "b200": Card(
        slug="b200",
        gpu_type_ids=("NVIDIA B200",),
        sm_expected="10.0",
        disk_gb=220,
        usd_per_hour_hint=5.98,
        data_center_part=True,
    ),
    "5090": Card(
        slug="5090",
        gpu_type_ids=("NVIDIA GeForce RTX 5090", "NVIDIA RTX PRO 6000 Blackwell Server Edition"),
        sm_expected="12.0",
        disk_gb=220,
        usd_per_hour_hint=0.94,
        data_center_part=False,
    ),
}


def pick(slug: str) -> Card:
    try:
        return CARDS[slug]
    except KeyError:
        raise KeyError(f"unknown card {slug!r}; known: {', '.join(sorted(CARDS))}") from None
