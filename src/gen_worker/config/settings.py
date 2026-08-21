"""Worker `Settings` struct — the canonical typed config for the worker process."""
import msgspec

BOOT_CONFIG_GENERATION_ABSENT = -1


class Settings(msgspec.Struct, frozen=True, kw_only=True):
    """Worker-process configuration."""

    hf_token: str = ""
    hf_home: str = ""

    tensorhub_public_url: str = ""
    orchestrator_public_addr: str = ""

    worker_id: str = ""
    endpoint_lock_path: str = "/app/.tensorhub/endpoint.lock"
    weights_cas_root: str = ""
    graph_cas_root: str = ""
    artifacts_root: str = ""
    cozy_home: str = ""
    endpoint_state_root: str = ""
    baked_program_cas_root: str = ""
    # NOT "the worker's JWT" — the BOOTSTRAP copy the hub injected at pod create, frozen forever. The live credential is rotated over the scheduler stream; gen_worker.worker_credential is its only source of truth. Read it there, never here — presenting this expired copy wedges the pod.
    bootstrap_worker_jwt: str = ""
    worker_release_id: str = ""

    runpod_pod_id: str = ""

    config_snapshot_path: str = ""
    # The sentinel default -1 means "no boot-only environment was ever injected", which is NOT "generation 0": the hub injects WORKER_CONFIG_GENERATION only into pod-launch env, so a host-process worker never receives one, and conflating the two makes it report BOOT_STALE for its whole life.
    boot_config_generation: int = BOOT_CONFIG_GENERATION_ABSENT

    worker_image_digest: str = ""

    boot_record_path: str = ""
    # pgw#1630: the FLOOR under the compute-child flatness window, in seconds.
    # The window itself is derived from that child's own observed
    # inter-progress gaps (procsplit/liveness.py); this only bounds it from
    # below, and the ladder needs FOUR consecutive windows of provably-zero
    # kernel work before anything is killed. 0 = the derived default. A knob
    # because a fleet meeting a legitimately slower shape must be able to widen
    # the window without a release.
    watchdog_flatness_floor_s: float = 0.0

    tensorhub_url: str = ""
    tensorhub_token: str = ""
    tensorhub_cache_dir: str = ""
    tensorhub_fill_source_dir: str = ""

    civitai_api_key: str = ""

    c2pa_cert_pem: str = ""
    c2pa_cert_path: str = ""
    c2pa_alg: str = "es256"
    c2pa_ta_url: str = ""
