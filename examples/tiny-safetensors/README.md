# Tiny Safetensors Example

This example downloads a minimal safetensors file from cozy-hub and runs a tiny
linear "inference" (y = w @ x + b). It is intentionally small so it can be
used as a quick end-to-end test.

## Expected input

```json
{
  "x": [3.0, 4.0],
  "model_ref": "download/model-files/2"
}
```

`model_ref` defaults to `download/model-files/2`, which is seeded by the local
docker-compose stack in `~/cozy`.

## Expected output

```json
{
  "y": 2.5
}
```

## Notes
- The model is stored in cozy-hub as `tiny-linear.safetensors` (model file id 2).
- `COZY_HUB_URL` must be set (e.g. `http://cozy-hub:3001/api/v1` for docker,
  or `http://localhost:3101/api/v1` when running locally).
