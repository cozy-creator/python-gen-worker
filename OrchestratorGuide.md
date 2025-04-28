# Guide on Deploying Functions for the Gen Orchestrator

This guide explains how to package your Python functions, manage secrets, and register your code so it can be run on demand by the Gen orchestrator.

---

## Table of Contents

- [Core Concepts](#core-concepts)
- [Steps to Deploy](#steps-to-deploy)
  - [Step 1: Write Your Function(s)](#step-1-write-your-functions)
  - [Step 2: Create requirements.txt](#step-2-create-requirementstxt)
  - [Step 3: Create Your Dockerfile](#step-3-create-your-dockerfile)
  - [Step 4: Build & Push Your Docker Image](#step-4-build--push-your-docker-image)
  - [Step 5: Create Secrets in Platform's Runpod Account](#step-5-create-secrets-in-platforms-runpod-account)
  - [Step 6: Register Your Deployment](#step-6-register-your-deployment)
  - [Step 7: Use Your Function](#step-7-use-your-function)

---

## Core Concepts

- **Worker**: Write standard Python functions decorated with `@worker_function` from the `gen-worker` library.
- **Base Image**: Platform-provided image with Python, PyTorch (CUDA), and core runtime.
- **Developer Image**: Custom Dockerfile starting from base image, adding your code and dependencies.
- **Secrets**: Store sensitive data like API keys via **Runpod Secrets**.
- **Deployment**: Links a `deployment_id` to your Docker image, functions, and secrets.
- **Orchestrator**: Reads deployment, launches containers, injects secrets, routes requests.
- **Execution**: Functions run inside containers using environment-injected secrets.

---

## Steps to Deploy

### Step 1: Write Your Function(s)

Install `gen-worker`:

```bash
uv pip install --index-url http://137.184.153.104:3141/root/mydev/+simple gen-worker
```

Import and decorate your functions:

```python
from gen_worker import worker_function, ResourceRequirements, ActionContext
import os
import boto3

# S3 upload example
s3_resources = ResourceRequirements()

@worker_function(resources=s3_resources)
def upload_image_to_s3(ctx: ActionContext, upload_details: dict) -> dict:
    access_key = os.environ.get("S3_ACCESS_KEY_ID")
    secret_key = os.environ.get("S3_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("S3_BUCKET_NAME")
    region = os.environ.get("S3_REGION")

    if not all([access_key, secret_key, bucket_name, region]):
        raise ValueError("Missing required S3 environment variables in worker")

    s3_client = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )

    image_bytes = upload_details.get("image_bytes")
    # Upload code here
    return {"s3_url": "http://..."}

# Image generation example
sdxl_resources = ResourceRequirements(
    model_name="<huggingface_model_id>",
    min_vram_gb=8.0,
    recommended_vram_gb=12.0
)

@worker_function(resources=sdxl_resources)
def generate_image(ctx: ActionContext, prompt_details: dict) -> bytes:
    return b''  # Your model logic here
```

Place functions in a module, e.g., `my_functions/`.

---

### Step 2: Create `requirements.txt`

```txt
boto3>=1.28.0
blake3>=0.3.0
diffusers>=0.20.0
transformers>=4.30.0
accelerate>=0.21.0
safetensors>=0.3.1
Pillow>=9.0.0
```

> **Do not include** `torch`, `grpcio`, `psutil`, `gen-worker`.

---

### Step 3: Create Your Dockerfile

```dockerfile
FROM cozycreator/base-worker:cuda118-py310-base-v2

WORKDIR /app

COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

COPY my_functions ./my_functions

ENV USER_MODULES=my_functions
```

---

### Step 4: Build & Push Your Docker Image

```powershell
$env:MY_IMAGE = "your-dockerhub-user/my-app-worker:v1.0.0"
docker build -t $env:MY_IMAGE .
docker push $env:MY_IMAGE
```

---

### Step 5: Create Secrets in Platform's Runpod Account

Obtain the \*\*platform's \*\*\`\`.

```powershell
$env:RUNPOD_API_KEY="PLATFORM_RUNPOD_API_KEY_HERE"

# Upload secrets
cd cmd/secret-uploader

go run main.go --name your-tenant-id-s3-key

go run main.go --name your-tenant-id-s3-secret

go run main.go --name your-tenant-id-s3-bucket

go run main.go --name your-tenant-id-s3-region

Remove-Item Env:\RUNPOD_API_KEY
```

> **Tip**: Use clear, tenant-specific secret names.

---

### Step 6: Register Your Deployment

```powershell
cd cmd/scheduler-admin

go run main.go `
    --db-path "C:\path\to\platform\scheduler-db" `
    --id "your-tenant-id-app-v1.0.0" `
    --image "your-dockerhub-user/my-app-worker:v1.0.0" `
    --functions "generate_image,upload_image_to_s3" `
    --tenant "your-tenant-id" `
    --secrets "S3_ACCESS_KEY_ID=your-tenant-id-s3-key,S3_SECRET_ACCESS_KEY=your-tenant-id-s3-secret,S3_BUCKET_NAME=your-tenant-id-s3-bucket,S3_REGION=your-tenant-id-s3-region"
```

---

### Step 7: Use Your Function

```python
deployment_id = "your-tenant-id-app-v1.0.0"

# Generate image
image_bytes = client.execute(deployment_id, "generate_image", {"prompt": "test", "seed": 1})

# Upload image
upload_result = client.execute(deployment_id, "upload_image_to_s3", {"image_bytes": image_bytes})
print(f"Upload Result: {upload_result}")
```

The orchestrator will automatically:

- Find the deployment
- Start a container (if needed)
- Inject secrets
- Route your request

---

