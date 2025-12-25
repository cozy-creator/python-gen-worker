# E2E Stub Worker

Minimal example worker function used for end-to-end tests. It downloads a model
via the cozy-hub downloader and returns the local path, and optionally uploads
the result to S3-compatible storage when configured.
