import os
import sys
import time
import grpc
import msgpack
import concurrent.futures
from typing import Optional

try:
    from gen_worker.pb import frontend_pb2
    from gen_worker.pb import frontend_pb2_grpc
except ImportError:
    print("Error: Could not import protobuf definitions.", file=sys.stderr)
    sys.exit(1)

# === Configuration (only these two come from ENV) ===
SCHEDULER_ADDR   = os.getenv("SCHEDULER_ADDR", "localhost:8080")
DEPLOYMENT_ID    = os.getenv("DEPLOYMENT_ID", "tenant-a-image-gen-app-v1")
# === Everything else is hard-coded here ===
TEST_PROMPT      = "cowgirl riding a horse, cinematic lighting"
TEST_SEED        = 9876
CONCURRENCY      = 2
REQUIRED_MODEL   = "ebara-pony-xl"

# =====================================================
def execute_and_await(stub: frontend_pb2_grpc.FrontendServiceStub,
                      deployment_id: str,
                      function_name: str,
                      input_data: dict,
                      required_model_id: Optional[str] = None,
                      wait_timeout: int = 120) -> Optional[dict]:
    """Executes an action and waits for its result."""
    print(f"  [execute_and_await] function={function_name}, timeout={wait_timeout}")
    if required_model_id:
        print(f"  [execute_and_await] required_model={required_model_id}")
    else:
        print(f"  [execute_and_await] no required_model")
    try:
        payload = msgpack.packb(input_data, use_bin_type=True)
        # print(f"  [execute_and_await] payload={payload}")
    except Exception as e:
        print(f"  [Error] serializing input: {e}", file=sys.stderr)
        return None

    # Submit
    try:
        print(f"  [execute_and_await] required_model_id={required_model_id}")
        request = frontend_pb2.ExecuteActionRequest(
            deployment_id   = deployment_id,
            function_name   = function_name,
            input_payload   = payload,
            # required_model_id = required_model_id
        )
        if required_model_id: # This is the critical part
            request.required_model_id = required_model_id

        # print(f"  [execute_and_await] request={request}")

        resp = stub.ExecuteAction(request, timeout=30)
        print(f"    → Submitted. Run ID: {resp.run_id}")
        run_id = resp.run_id
    except grpc.RpcError as e:
        print(f"  [gRPC Error] ExecuteAction: {e.code()} {e.details()}", file=sys.stderr)
        return None

    # Wait
    print(f"  [Waiting] for result of run_id={run_id} (up to {wait_timeout}s)...")
    try:
        start = time.time()
        get_resp = stub.GetRun(
            frontend_pb2.GetRunRequest(run_id=run_id),
            timeout=wait_timeout
        )
        elapsed = time.time() - start
        print(f"    ← Received in {elapsed:.2f}s")

        if not get_resp.success:
            print(f"  [Task Failed] {get_resp.error_message}", file=sys.stderr)
            return None

        if not get_resp.output_payload:
            print("  [Warning] succeeded but empty payload")
            return {}

        data = msgpack.unpackb(get_resp.output_payload, raw=False)
        print(f"  [Success] unpacked payload of type {type(data)}")
        return data

    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
            print(f"  [Timeout] waiting for result ({wait_timeout}s)", file=sys.stderr)
        else:
            print(f"  [gRPC Error] GetRun: {e.code()} {e.details()}", file=sys.stderr)
        return None

def run_workflow(task_id: int,
                 stub: frontend_pb2_grpc.FrontendServiceStub,
                 deployment_id: str,
                 base_prompt: str,
                 base_seed: int,
                 model_id_for_generation: Optional[str] = None):
    """One end-to-end workflow: generate and upload in single function call."""
    print(f"\n--- Task {task_id} started ---")
    prompt = f"{base_prompt} [task {task_id}]"
    seed = base_seed + task_id
    filename = f"task_{task_id}_{seed}.png"

    # Single combined call: generate + upload
    print(f"[Task {task_id}] Generating and uploading image with seed={seed}")
    result = execute_and_await(
        stub, 
        deployment_id, 
        "generate_and_upload_image",
        {
            "prompt": prompt, 
            "seed": seed,
            "filename": filename,
            # Optional: Add other generation parameters
            "num_inference_steps": 28,
            "guidance_scale": 7.5,
            "width": 1024,
            "height": 1024
        },
        required_model_id=model_id_for_generation,
        wait_timeout=600  # Allow more time for both generation + upload
    )
    
    if not isinstance(result, dict):
        print(f"[Task {task_id}] generate_and_upload_image failed.", file=sys.stderr)
        return

    # Extract results from combined response
    s3_url = result.get("s3_url")
    if not s3_url:
        print(f"[Task {task_id}] No S3 URL in response.", file=sys.stderr)
        return

    print(f"[Task {task_id}] ✅ Complete! S3 URL: {s3_url}")
    
    # Print additional metadata if available
    if "image_size_bytes" in result:
        size_kb = result["image_size_bytes"] / 1024
        print(f"[Task {task_id}] Generated image: {size_kb:.1f} KB")
    
    if "width" in result and "height" in result:
        print(f"[Task {task_id}] Dimensions: {result['width']}x{result['height']}")

if __name__ == "__main__":
    # Print startup info
    print("--- Running Client Workflow Test (Combined Function) ---")
    print(f"Scheduler Address: {SCHEDULER_ADDR}")
    print(f"Deployment ID:     {DEPLOYMENT_ID}")
    print(f"Test Prompt:       {TEST_PROMPT!r}")
    print(f"Test Seed:         {TEST_SEED}")
    print(f"Required Model:    {REQUIRED_MODEL}")
    print(f"Concurrency:       {CONCURRENCY}")
    print("Using combined generate_and_upload_image function")
    print("------------------------------------")

    # Connect and fire off concurrent tasks
    with grpc.insecure_channel(SCHEDULER_ADDR) as channel:
        stub = frontend_pb2_grpc.FrontendServiceStub(channel)

        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = [
                executor.submit(
                    run_workflow, i, stub, DEPLOYMENT_ID, TEST_PROMPT, TEST_SEED, REQUIRED_MODEL
                )
                for i in range(CONCURRENCY)
            ]
            concurrent.futures.wait(futures)

    print("\n--- All tasks completed ---")