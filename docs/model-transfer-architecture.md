# Model Transfer Architecture

python-gen-worker has a closed model-weight provider surface:

- Tensorhub
- Hugging Face
- Civitai

It does not accept arbitrary URL model refs. Tensorhub may currently hand the
worker storage URLs for resolved Tensorhub artifacts, but those URLs are an
internal Tensorhub transport detail, not a fourth provider.

## Tensorhub

Tensorhub is the source of truth for artifact sessions, dedupe, revision
manifests, BLAKE3 digests, and finalization. R2 is only the byte store.

Trusted worker model uploads use Tensorhub-issued scoped transfer grants and
`boto3`/`botocore`/`s3transfer` against Cloudflare R2. Tensorhub returns bucket,
object key, endpoint URL, short-lived credentials, permitted operations, and
expiry; the worker uploads the seekable local file through the SDK and reports
size, BLAKE3, and object metadata back on complete.

The legacy Tensorhub presigned multipart uploader remains only for non-model
platform plumbing that has not moved to grants yet. While it exists, it uses
fixed internal defaults:

- file upload fanout: 4
- part fanout per file: 4
- process-wide presigned R2 PUT budget: 8

These are not tenant or operator knobs. They are implementation defaults chosen
to preserve useful throughput while preventing file-level and part-level pools
from multiplying into a retry storm.

Credentials must be narrowly scoped by bucket, prefix or object, TTL, and
required permissions. Tensorhub still validates the final object with BLAKE3
and commits the artifact metadata.

Cloudflare R2 supports temporary S3 credentials with bucket, operation, and
path scope. Tensorhub should mint those credentials in the trusted control
plane; workers should only receive the short-lived access key, secret key,
session token, bucket, object key or prefix, endpoint URL, and expiry for the
current transfer.

python-gen-worker accepts this grant shape in the upload-create response as
`transfer_grant` / `s3_transfer_grant` and routes it through
`gen_worker.s3_transfer.upload_file_with_grant`. Repo/model uploads require a
grant and do not fall back to presigned multipart.

Do not rely on AWS checksum headers that R2 does not support. Tensorhub BLAKE3
validation is the integrity check.

## Hugging Face

Hugging Face artifacts are downloaded with `huggingface_hub`. Worker code may
plan/select the file set and probe metadata, but should not reimplement Hub
transfer, cache, or resume behavior.

Large Hugging Face artifacts should stage to local disk before Tensorhub upload.
Streaming HF directly into multipart upload is not the default because retrying
multipart uploads is safer with seekable files.

## Civitai

Civitai uses a provider-specific bounded downloader/API integration. Do not
route Civitai through arbitrary URL download plumbing.

## Benchmarks

Transfer defaults should be chosen from benchmarks, not exposed as knobs.
Benchmarks should compare reliability first, then wall-clock time, peak RSS,
CPU, disk I/O, retry count, and R2 error rate on representative model sizes.
Use `scripts/benchmark_model_transfer.py` so candidate paths report the same
resource metrics.

The benchmark set should include FLUX.2-klein-4B and compare:

- `boto3`/`s3transfer` from seekable local files with throughput-oriented
  fixed defaults
- `boto3`/`s3transfer` with a lower internal concurrency default
- the legacy presigned path with the safety limiter, as a non-model baseline
- optional HF direct streaming only if retry correctness can be preserved
