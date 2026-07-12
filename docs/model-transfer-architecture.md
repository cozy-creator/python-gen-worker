# Model transfer

gen-worker has a closed model-weight provider surface: **Tensorhub**,
**Hugging Face**, **Civitai**. It does not accept arbitrary URL model refs;
storage URLs Tensorhub hands the worker are an internal transport detail,
not a fourth provider.

## Uploads (worker → Tensorhub)

Tensorhub is the source of truth for commits, dedup, manifests, BLAKE3
digests, and finalization; R2 is only the byte store.

- **Repo/model uploads require a Tensorhub-issued scoped transfer grant**
  (`transfer_grant` in the upload-create response: short-lived S3
  credentials + bucket/key/expiry, scoped per object). The worker uploads
  the seekable local file via `gen_worker.s3_transfer.upload_file_with_grant`
  (boto3/s3transfer) and reports size + BLAKE3 on complete. There is no
  presigned-multipart fallback for model uploads.
- **Non-model platform uploads** (media, datasets) may still return presigned
  multipart URLs; those upload part-by-part with fixed internal defaults
  (file fanout 4, part fanout 4, process-wide presigned PUT budget 8 —
  implementation constants, not knobs).
- Integrity is Tensorhub's BLAKE3 verification, not AWS checksum headers
  (R2 doesn't support them all).

## Downloads

- **HF**: `huggingface_hub` owns transfer/cache/resume; worker code plans the
  file set (`files=` allow-patterns) but never reimplements Hub transfer.
  Large HF artifacts stage to local disk before any Tensorhub upload
  (seekable files make multipart retry safe).
- **Civitai**: a bounded provider-specific downloader/API integration.
- **Tensorhub**: presigned R2 GETs against the resolved snapshot manifest,
  blake3-verified into the local CAS.

Transfer defaults are chosen from benchmarks
(`scripts/benchmark_model_transfer.py` — reliability first, then wall-clock,
RSS, retries), not exposed as knobs.
