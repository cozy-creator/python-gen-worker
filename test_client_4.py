import grpc
import msgpack
import time
from gen_worker.pb import frontend_pb2_grpc, frontend_pb2

import os
from dotenv import load_dotenv

load_dotenv()

addr = os.getenv("SCHEDULER_ADDR", "localhost:8080")
# Connect to the orchestrator
channel = grpc.insecure_channel(addr)
stub = frontend_pb2_grpc.FrontendServiceStub(channel)

# Prepare your request
request_params = {
    "prompt": "a majestic dragon flying over mountains at sunset",
    "seed": 12345,
    "num_inference_steps": 30,
    "guidance_scale": 8.0,
    "width": 1024,
    "height": 1024,
    "filename": "dragon_sunset.png"
}

# Submit the job
request = frontend_pb2.ExecuteActionRequest(
    function_name="generate_and_upload_image",
    deployment_id="tenant-a-image-gen-app-v1",
    required_model_id="ebara-pony-xl",
    input_payload=msgpack.packb(request_params)
)

response = stub.ExecuteAction(request)
run_id = response.run_id
print(f"Job submitted with ID: {run_id}")

# Wait for completion
get_request = frontend_pb2.GetRunRequest(run_id=run_id)
result = stub.GetRun(get_request, timeout=300)  # 5 minute timeout

if result.success:
    output = msgpack.unpackb(result.output_payload)
    print(f"Image generated successfully!")
    print(f"URL: {output['s3_url']}")
    print(f"Size: {output['image_size_bytes']} bytes")
else:
    print(f"Generation failed: {result.error_message}")