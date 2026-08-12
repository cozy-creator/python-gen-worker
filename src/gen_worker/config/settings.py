"""Worker `Settings` struct — the canonical typed config for the worker process.

Ruling §1.18 (Paul, 2026-08-02): *"we should NEVER be loading random envs in
the middle of code; we should only load it from our config pipeline and then
pass it around."* Exactly one component in this process reads the process
environment — `gen_worker.config` — and it produces exactly one typed struct,
which is **passed by parameter**.

Each process entry performs one bootstrap-owned `load_settings()`:
`entrypoint._run_main()` for the worker, `procsplit.parent.main()` for the
control parent, and the `cli/` command group for the standalone CLI. Nothing
else loads; nothing else reads env. There is deliberately **no** cached
process-global accessor — `get_settings()` was deleted in pgw#931 because a
`lru_cache`d singleton is the same defect as a raw env read wearing better
clothes: unreachable by a caller that wants to pass a different value,
invisible to a test that does not clear the cache, and latched by whichever
module happened to import first.

The list of sanctioned raw-read sites is not prose here — prose drifted from
5 files to 41 before pgw#931 measured it. It is
`scripts/config_reads_allowlist.txt`, every line classified, enforced by
`scripts/lint_config_reads.py` in CI. A new `os.environ` read outside this
package fails the build.

Built on msgspec.Struct (already a worker dep) instead of pydantic-settings to
avoid pulling in pydantic. The source-loader layering (env → .env → secrets dir
→ yaml → defaults) lives next to this in `loader.py`.
"""
import msgspec

# gw#668 sentinel: no boot-only environment (``WORKER_CONFIG_GENERATION``) was
# ever injected into this process. Distinct from generation 0, which is a real
# — and genuinely ancient — injected generation.
BOOT_CONFIG_GENERATION_ABSENT = -1


class Settings(msgspec.Struct, frozen=True, kw_only=True):
    """Worker-process configuration. Loaded once at startup, passed by reference.

    All fields default to their zero value so the loader can omit any field
    that isn't explicitly set in any source — msgspec.convert applies struct
    defaults for missing keys. Required-ness is enforced by callers (e.g.
    `Worker.__init__` raises if `orchestrator_public_addr` is empty).
    """

    # HuggingFace credentials/cache. Universal across endpoints — read by the
    # huggingface_hub library and our HF download helpers.
    hf_token: str = ""
    hf_home: str = ""

    # Service endpoints injected by gen-orchestrator at pod launch.
    tensorhub_public_url: str = ""
    # Public address of the orchestrator router (ALPN-multiplexed HTTP + gRPC).
    # Single shared value across the cluster — the router resolves us to the
    # per-release lease owner. gen-orchestrator issue #317.
    orchestrator_public_addr: str = ""

    # Per-pod worker identity (orchestrator-injected).
    worker_id: str = ""
    # Path to the discovery manifest (endpoint.lock). Default is the baked
    # container location; non-container runs (e2e, bare-metal dev) override it.
    endpoint_lock_path: str = "/app/.tensorhub/endpoint.lock"
    # pgw#848 / Paul: NOT "the worker's JWT". It is the BOOTSTRAP copy — the
    # value the hub injected at pod create, frozen there forever and updated by
    # nothing. The live credential is rotated over the scheduler stream at ~80 %
    # of TTL, and `gen_worker.worker_credential` is the ONLY source of truth for
    # it. Renamed from `worker_jwt` deliberately: it read exactly like "the
    # worker's JWT" to anyone who did not already know the live value lived in
    # another module's private state, and the attestation carrier read it for
    # precisely that reason — every dial past T+30 min then presented an expired
    # token, three of which wedge the pod (pgw#846 attempts 16 and 17).
    #
    # The rename is the fix rather than a comment because it makes the mistake
    # IMPOSSIBLE instead of detectable: any surviving `settings.worker_jwt`
    # reader is now an AttributeError at the call site, not a stale string. Read
    # this field from `worker_credential` and nowhere else.
    bootstrap_worker_jwt: str = ""
    # pgw#763 delta 1: the worker's release identity, delivered SEPARATELY from
    # the JWT. Under the process split the compute child holds no JWT (the
    # parent strips it), so it cannot read `release_id` out of the claims — but
    # the intent registry and every lifecycle snapshot are keyed on it. A claim
    # is not a credential: the control parent decodes its own token and hands
    # this one value down to the compute child.
    worker_release_id: str = ""

    # Runtime introspection (set by the RunPod runtime; not configuration).
    runpod_pod_id: str = ""

    # th#1087: local mutable-config snapshot file (atomic-rewritten on each
    # config-generation push; per-invoke subprocesses read it). Empty =
    # runtime_config.DEFAULT_SNAPSHOT_PATH.
    config_snapshot_path: str = ""  # GEN_WORKER_CONFIG_SNAPSHOT_PATH
    # Config generation whose boot-only environment was injected when this pod
    # was created. Never inferred from the first desired-state command.
    #
    # gw#668: the DEFAULT is the sentinel -1, meaning "no boot-only
    # environment was ever injected into this process" — which is not the same
    # fact as "the injected boot config is generation 0". Tensorhub injects
    # WORKER_CONFIG_GENERATION only into pod-launch env, so a host-process
    # worker (the e2e local-worker shape, the dev loop, any BYO/externally
    # managed fleet) never receives one; conflating the two made every such
    # worker report BOOT_STALE for its whole life, with a remedy (pod
    # replacement) that does not exist for a pod-less worker.
    boot_config_generation: int = BOOT_CONFIG_GENERATION_ABSENT  # WORKER_CONFIG_GENERATION

    # Immutable image provenance stamped by Tensorhub from the release image
    # variant it selected for this pod.
    worker_image_digest: str = ""  # WORKER_IMAGE_DIGEST

    # pgw#931 VIOLATION-A #14/#15: where the post-mortem boot record is written.
    # It had NO field at all and was read raw in three places, which is why the
    # parent and the child could disagree about the carrier. Empty = let
    # `postmortem` pick a DURABLE default (the model-cache volume when it is not
    # itself volatile; /tmp only as a last resort, and loudly).
    boot_record_path: str = ""  # GEN_WORKER_BOOT_RECORD

    # tensorhub access for standalone clients (run/serve/prefetch). Production
    # workers get orchestrator-resolved manifests and never dial these.
    tensorhub_url: str = ""        # TENSORHUB_URL
    tensorhub_token: str = ""      # TENSORHUB_TOKEN
    # CAS/cache roots. cache_dir moves the whole tensorhub cache off /tmp
    # (cozy local persists weights across reboots); cas_dir points the
    # standalone CLI (`gen-worker run`) at an explicit CAS root — also the
    # test-isolation knob for the CLI resolver tests (real consumer:
    # models/provision.py::resolve_local_path).
    tensorhub_cache_dir: str = ""  # TENSORHUB_CACHE_DIR
    tensorhub_cas_dir: str = ""    # TENSORHUB_CAS_DIR
    # th#850 managed-tier ruling (gw#599): endpoint-scoped datacenter-warm
    # fill source (RunPod network volume mount), tried before R2. Never the
    # CAS root itself — see models/cache_paths.py::tensorhub_fill_source_dir.
    tensorhub_fill_source_dir: str = ""  # TENSORHUB_FILL_SOURCE_DIR

    # Civitai provider credential (CIVITAI_API_KEY, alias CIVITAI_TOKEN).
    civitai_api_key: str = ""

    # C2PA Content Credentials signing (th#714, EU AI Act Art. 50).
    # Signing is ON iff cert material is set (inline PEM or path): every
    # generated media asset gets a signed provenance manifest at save time
    # (content_credentials.py). Inline *_PEM carries the PEM content itself —
    # the hub injects it into pod env at launch (RunPod pods have no file
    # mounts) — and takes precedence over *_PATH. Chain = leaf first.
    # th#1307: there is NO key field. The PRIVATE KEY never reaches a pod;
    # the hub signs claims over POST /v1/worker/c2pa/sign. A pod that finds
    # GEN_WORKER_C2PA_KEY_PEM/_KEY_PATH in ANY config source refuses to start
    # — env via content_credentials._refuse_pod_private_key_material (a
    # ratchet, re-checked at every read of the signing state), and .env /
    # /run/secrets / yaml via loader.REFUSED_KEY_MATERIAL (pgw#884: this
    # sentence used to say "in its env", and a PEM mounted at
    # /run/secrets/GEN_WORKER_C2PA_KEY_PEM booted green).
    c2pa_cert_pem: str = ""   # GEN_WORKER_C2PA_CERT_PEM (inline PEM chain)
    c2pa_cert_path: str = ""  # GEN_WORKER_C2PA_CERT_PATH
    c2pa_alg: str = "es256"   # GEN_WORKER_C2PA_ALG (COSE alg matching the cert key)
    c2pa_ta_url: str = ""     # GEN_WORKER_C2PA_TA_URL (optional RFC3161 TSA)
