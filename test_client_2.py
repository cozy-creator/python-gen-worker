# client_test.py (Workflow Example)
import grpc
import msgpack
import os
import sys
import time
from typing import Optional

try:
    from gen_worker.pb import frontend_pb2
    from gen_worker.pb import frontend_pb2_grpc
except ImportError:
    print("Error: Could not import protobuf definitions.", file=sys.stderr)
    sys.exit(1)

# Helper function to execute and await a single action
def execute_and_await(stub: frontend_pb2_grpc.FrontendServiceStub,
                      deployment_id: str,
                      function_name: str,
                      input_data: dict,
                      wait_timeout: int = 120) -> Optional[dict]:
    """Executes an action and waits for its result."""
    print("-" * 20)
    print(f"Executing function '{function_name}' in deployment '{deployment_id}'...")

    try:
        input_payload_bytes = msgpack.packb(input_data, use_bin_type=True)
    except Exception as e:
        print(f"  Error serializing input: {e}", file=sys.stderr)
        return None

    execute_request = frontend_pb2.ExecuteActionRequest(
        deployment_id=deployment_id,
        function_name=function_name,
        input_payload=input_payload_bytes,
    )

    run_id = None
    try:
        start_time = time.time()
        execute_response = stub.ExecuteAction(execute_request, timeout=10)
        run_id = execute_response.run_id
        print(f"  Action submitted. Run ID: {run_id}")
    except grpc.RpcError as e:
        print(f"  Error calling ExecuteAction: {e.code()} - {e.details()}", file=sys.stderr)
        return None
    except Exception as e:
         print(f"  An unexpected error occurred during ExecuteAction: {e}", file=sys.stderr)
         return None

    # Wait for result
    print(f"  Waiting for result (timeout: {wait_timeout}s)...")
    get_run_request = frontend_pb2.GetRunRequest(run_id=run_id)
    try:
        get_run_response = stub.GetRun(get_run_request, timeout=wait_timeout)
        end_time = time.time()
        print(f"  Received result after {end_time - start_time:.2f} seconds.")

        if get_run_response.success:
            print("  Task completed successfully!")
            output_payload_bytes = get_run_response.output_payload
            if not output_payload_bytes:
                 print("  Warning: Task succeeded but returned empty payload.")
                 return {} # Return empty dict for success but no payload

            try:
                # Important: Unpack the result from msgpack!
                result_data = msgpack.unpackb(output_payload_bytes, raw=False)
                print(f"  Deserialized result: {type(result_data)}")
                return result_data
            except Exception as e:
                print(f"  Error deserializing result payload: {e}", file=sys.stderr)
                return None # Indicate failure to unpack
        else:
            print(f"  Task failed: {get_run_response.error_message}", file=sys.stderr)
            return None # Indicate failure

    except grpc.RpcError as e:
         if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
             print(f"  Timed out waiting for result ({wait_timeout}s).", file=sys.stderr)
         else:
            print(f"  Error calling GetRun: {e.code()} - {e.details()}", file=sys.stderr)
         return None
    except Exception as e:
         print(f"  An unexpected error occurred during GetRun: {e}", file=sys.stderr)
         return None


# Main Workflow Orchestration
if __name__ == "__main__":
    # --- Configuration ---
    scheduler_target_addr = os.getenv("SCHEDULER_ADDR", "localhost:8080")
    # Use the Deployment ID registered for the image containing both functions
    test_deployment_id = os.getenv("DEPLOYMENT_ID", "tenant-a-image-gen-app-v1")

    test_prompt = "cowgirl riding a horse, cinematic lighting"
    test_seed = 9876
    output_image_filename = "workflow_output_local.png" # Save locally for verification
    # --- End Configuration ---

    print("--- Running Client Workflow Test ---")
    print(f"Scheduler Address: {scheduler_target_addr}")
    print(f"Deployment ID: {test_deployment_id}")
    print("----------------------------------")

    image_bytes_result = None

    # Connect to server
    try:
        with grpc.insecure_channel(scheduler_target_addr) as channel:
            stub = frontend_pb2_grpc.FrontendServiceStub(channel)

            # --- Step 1: Generate Image ---
            gen_input = {"prompt": test_prompt, "seed": test_seed}
            # Use a longer timeout for image generation
            image_bytes_result = execute_and_await(stub, test_deployment_id, "generate_image", gen_input, wait_timeout=180)

            if image_bytes_result is None or not isinstance(image_bytes_result, bytes):
                 print("\nWorkflow failed: Image generation did not return valid bytes.")
                 sys.exit(1)

            print(f"\n  Image generated ({len(image_bytes_result)} bytes). Saving locally to {output_image_filename}")
            try:
                with open(output_image_filename, "wb") as f:
                    f.write(image_bytes_result)
            except IOError as e:
                 print(f"  Error saving image locally: {e}")
                 # Continue to upload attempt anyway? Or exit? Let's continue.


            # --- Step 2: Upload Image to S3 ---
            upload_input = {
                "image_bytes": image_bytes_result,
                "filename": f"cat_sunglasses_{test_seed}.png" # Optional filename base
            }
            # Use a shorter timeout for upload usually
            upload_result = execute_and_await(stub, test_deployment_id, "upload_image_to_s3", upload_input, wait_timeout=60)

            if upload_result is not None and isinstance(upload_result, dict):
                s3_url = upload_result.get("s3_url")
                if s3_url:
                    print(f"\nWorkflow successful! Image uploaded to: {s3_url}")
                else:
                    print("\nWorkflow partially failed: S3 upload succeeded but returned no URL.")
            else:
                print("\nWorkflow failed: S3 upload step failed.")
                sys.exit(1)


    except grpc.RpcError as e:
        print(f"\nWorkflow interrupted by gRPC connection error: {e.code()} - {e.details()}", file=sys.stderr)
    except Exception as e:
        print(f"\nWorkflow interrupted by unexpected error: {e}", file=sys.stderr)

    print("\n--- Client Workflow Test Finished ---")
