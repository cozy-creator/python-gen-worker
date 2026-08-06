- **pgw#973 (§4.24 execution wave 2): the safetensors header cap is now one
  number, not six.** A safetensors file opens with an 8-byte declared header
  length taken straight from the file, and every reader turned it directly into
  `json.loads(f.read(n))`. That threat was bounded six times — `models/w4a4.py`,
  `models/w8a8.py`, `models/svdq.py`, `models/loading.py` and
  `convert/ingest.py` at 100 MiB, and `convert/writer.py` at **512 MiB**. The
  outlier was not harmless: the writer accepted headers the loader would refuse,
  so the re-shard path could emit a shard the serving path could not open — same
  bytes, two verdicts. All six now read
  `gen_worker.models.safetensors_header.MAX_HEADER_BYTES` / `header_len_ok()`,
  where the threat, the reason nothing else prevents it, and why 100 MiB is a
  plausibility floor rather than a measurement are stated once. RED-verified:
  restoring the 512 MiB cap makes the writer parse a 200 MiB-declared header the
  loader rejects.
- **pgw#973: a dead offload threshold deleted.**
  `models/memory.py:_DEFAULT_VAE_SLICE_THRESHOLD_GB = 10.0` had zero references
  anywhere in the repo — src, tests, docs and scripts.
- **pgw#973: two verbatim-duplicated bounds given single owners.**
  `_READ_CHUNK_BYTES` (4 MiB) was defined identically in `models/chunk_cas.py`
  and `models/chunk_upload.py`, which already imports the rest of the chunk
  vocabulary from `chunk_cas`; and `input_assets._DEFAULT_MAX_BYTES` /
  `_CHUNK` restated `url_fetch.DEFAULT_MAX_BYTES` / `_CHUNK`. **Correcting the
  census on the second one:** these are *not* two caps on one fetch path.
  `url_fetch.open_guarded_stream` deliberately caps nothing ("the caller owns
  the read and its byte cap"), and `url_fetch.fetch_bytes` and
  `input_assets._download_one` are separate entry points that each enforce their
  own. Deleting either would leave one path with no cap at all, so the value is
  aliased to a single owner rather than removed.
- **pgw#973: two limit justifications that cited a deleted module are corrected.**
  `presigned_upload._PRESIGNED_PUT_BUDGET = 8` called itself "the authoritative
  cap" on the grounds that "file-level fan-out is fixed at 4" — but the module
  that owned file-level fan-out (`_concurrent_upload.py`) no longer exists and
  the only in-repo caller is sequential, so the file axis is 1 and the binding
  bound is `optimal_part_concurrency`'s `min(total_parts, 4)`. The semaphore is
  KEPT because it covers the one axis nothing else can see — an endpoint author
  calling `ctx.save()` from their own threads — and both docstrings now say so.
