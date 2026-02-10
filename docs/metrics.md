# Worker-Reported Perf Metrics (v1)

`gen-worker` can optionally report best-effort performance/debug metrics to `gen-orchestrator` via the existing gRPC stream `WorkerSchedulerMessage.worker_event`.

These metrics are:

- Best-effort: metrics emission must never fail a run.
- Safe: numbers and small strings only. No URLs, secrets, or file paths.
- Optional: only emit keys when known; omit unknown fields entirely.

## Canonical Events

These event types are designed to be stable and low-cardinality. `gen-orchestrator` can persist them into dedicated columns.

- `metrics.compute.started` payload: `{ "at": "<rfc3339>" }`
- `metrics.compute.completed` payload: `{ "at": "<rfc3339>" }`
- `metrics.fetch` payload: `{ "ms": <int> }` (use `0` for warm disk hits)
- `metrics.gpu_load` payload: `{ "ms": <int> }`
- `metrics.inference` payload: `{ "ms": <int> }`
- `metrics.tokens` payload: `{ "output_tokens": <int> }` (only when applicable)

All times are milliseconds as integers.

## Extended Debug Event

Additionally, the worker emits one extended event at the end of each run:

- `metrics.run` payload: JSON object (schema versioned)

### `metrics.run` payload (schema_version=1)

Top-level keys (all optional unless noted):

- `schema_version` (required): `1`
- `function_name`: string
- `cache_state`: `hot_vram | warm_disk | cold_remote`
- `models`: array of objects (best-effort per required model)
- `pipeline_init_ms`: int
- `gpu_load_ms`: int
- `warmup_ms`: int (only for first warmup run; otherwise omit)
- `inference_ms`: int
- diffusion-ish extras (optional): `steps`, `iters_per_s`, `width`, `height`, `guidance`
- post (optional): `png_encode_ms`, `upload_ms`
- resources (optional): `peak_vram_bytes`, `peak_ram_bytes`

Per-model object keys (all optional unless noted):

- `model_id` (required): canonical model id used by worker/scheduler
- `variant_label`: string
- `snapshot_digest`: string
- `cache_state`: `hot_vram | warm_disk | cold_remote`
- `bytes_downloaded`: int (0 if none)
- `download_ms`: int (0 if warm disk hit)
- `bytes_read_disk`: int
- `disk_fstype`: string (e.g. `nfs4`, `ext4`, `overlay`) (best-effort)
- `disk_backend`: `local | nfs` (best-effort)
- `localized`: bool (true when an NFS snapshot was copied into the local cache dir before load)
- `nfs_to_local_copy_ms`: int (only when a copy occurred)
- `bytes_copied`: int (only when a copy occurred)

## Notes

- `metrics.fetch` is primarily the time spent ensuring required model blobs are present on disk (remote download vs warm disk hit).
- `metrics.gpu_load` is best-effort and currently reflects time spent moving injected model objects to the worker device when supported.
- `metrics.inference` is best-effort and currently reflects time spent executing the user function body (not including scheduler queueing).

## Model Cache Inventory (NFS-Aware)

Separately from `metrics.*`, the worker emits best-effort events that help `gen-orchestrator` understand which shared volumes (e.g. NFS) have which models.

- `model.cached` (run_id="") payload:
  - `model_variant_id`: string (canonical model id)
  - `disk_backend`: `local | nfs` (best-effort)
  - `disk_fstype`: string (best-effort)
  - `disk_volume_key`: string (sha256 hash of mount identity; does not expose raw mount source)

- `models.disk_inventory` (run_id="") payload:
  - `disk_backend`, `disk_fstype`, `disk_volume_key`
  - `disk_models`: string[] (canonical model ids present on disk)

These are emitted best-effort and must never fail a run.
