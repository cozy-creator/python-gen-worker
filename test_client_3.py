# client_test.py (Concurrent Workflow Example)

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
CONCURRENCY      = 10
REQUIRED_MODEL   = "playground2.5"

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
    """One end-to-end workflow: generate then upload."""
    print(f"\n--- Task {task_id} started ---")
    prompt = f"{base_prompt} [task {task_id}]"
    seed = base_seed + task_id

    # Step 1: generate image
    print(f"[Task {task_id}] Generating image with seed={seed}")
    img_bytes = execute_and_await(
        stub, 
        deployment_id, 
        "generate_image",
        {"prompt": prompt, "seed": seed},
        required_model_id=model_id_for_generation,
        wait_timeout=600
    )
    if not isinstance(img_bytes, (bytes, bytearray)):
        print(f"[Task {task_id}] generate_image failed.", file=sys.stderr)
        return

    local_file = f"output_task_{task_id}.png"
    # print(f"[Task {task_id}] Saving image to {local_file}")
    try:
        with open(local_file, "wb") as f:
            f.write(img_bytes)
    except IOError as e:
        print(f"[Task {task_id}] Error saving image: {e}", file=sys.stderr)

    # Step 2: upload to S3
    print(f"[Task {task_id}] Uploading to S3")
    upload_resp = execute_and_await(
        stub, deployment_id, "upload_image_to_s3",
        {"image_bytes": img_bytes, "filename": f"task_{task_id}_{seed}.png"},
        required_model_id=None,
        wait_timeout=60
    )
    if isinstance(upload_resp, dict) and upload_resp.get("s3_url"):
        print(f"[Task {task_id}] Uploaded: {upload_resp['s3_url']}")
    else:
        print(f"[Task {task_id}] upload_image_to_s3 failed.", file=sys.stderr)

if __name__ == "__main__":
    # Print startup info
    print("--- Running Client Workflow Test ---")
    print(f"Scheduler Address: {SCHEDULER_ADDR}")
    print(f"Deployment ID:     {DEPLOYMENT_ID}")
    print(f"Test Prompt:       {TEST_PROMPT!r}")
    print(f"Test Seed:         {TEST_SEED}")
    print(f"Required Model:    {REQUIRED_MODEL}")
    print(f"Concurrency:       {CONCURRENCY}")
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
