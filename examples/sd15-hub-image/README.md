# sd15-hub-image

Real GPU inference: Stable Diffusion 1.5 served from OUR R2-backed
repo-cas instead of HuggingFace directly. Same handler as
`examples/sd15-image`; the only difference is the model binding:

```python
@endpoint(model=Hub("tensorhub/sd15-mirror", tag="prod"), resources=Resources(vram_gb=6))
```

`Hub(...)` (provider `"tensorhub"`) resolves through tensorhub's repo-cas
instead of huggingface.co: the platform pre-resolves the ref to a
snapshot and the worker downloads every blob via presigned R2 GETs,
blake3-verified — no `HF_TOKEN`, no huggingface.co traffic.

Driven end to end by the cozy e2e J7 ingest-then-serve journey (`task
e2e-ingest-cloud` in the e2e repo): a prior `clone-huggingface` mirror
step publishes `stable-diffusion-v1-5/stable-diffusion-v1-5` into
`tensorhub/sd15-mirror`, then this endpoint serves straight from that
mirror on a real RunPod GPU pod.
