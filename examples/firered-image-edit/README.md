# firered-image-edit

Example Cozy worker for FireRed Image Edit.

Upstream reference implementation:
- `FireRedTeam/FireRed-Image-Edit` `inference.py`

Model:
- `hf:FireRedTeam/FireRed-Image-Edit-1.0`

This example uses `diffusers.QwenImageEditPlusPipeline` and exposes a single function:
- `edit`

Payload shape:

```py
class EditInput(msgspec.Struct):
    image: Asset          # input image (materialized to image.local_path)
    prompt: str           # edit instruction
    negative_prompt: str = " "
    num_inference_steps: int = 50
    true_cfg_scale: float = 4.0
    seed: Optional[int] = None
    num_images_per_prompt: int = 1
```
