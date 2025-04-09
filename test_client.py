import grpc
import msgpack
import os
import sys
import time


try:
    from gen_worker.pb import frontend_pb2
    from gen_worker.pb import frontend_pb2_grpc
except ImportError:
    print("Error: Could not import protobuf definitions.", file=sys.stderr)
    print("Ensure the generated 'frontend_pb2.py' and 'frontend_pb2_grpc.py' are accessible.", file=sys.stderr)
    print("You might need to adjust the import path or ensure the 'pb' directory is in your PYTHONPATH.", file=sys.stderr)
    sys.exit(1)

def run_generate_image(scheduler_addr: str, prompt: str, seed: int, output_file: str):
    """Connects to the scheduler, requests image generation, and saves the result."""

    print(f"Connecting to scheduler at: {scheduler_addr}")
    try:
        with grpc.insecure_channel(scheduler_addr) as channel:
            stub = frontend_pb2_grpc.FrontendServiceStub(channel)

            print(f"Preparing request for function 'generate_image'...")
            input_data = {
                "prompt": prompt,
                "seed": seed,
            }

            try:
                input_payload_bytes = msgpack.packb(input_data, use_bin_type=True)
                print(f"Serialized input data ({len(input_payload_bytes)} bytes).")
            except Exception as e:
                print(f"Error serializing input data with msgpack: {e}", file=sys.stderr)
                return

            execute_request = frontend_pb2.ExecuteActionRequest(
                function_name="generate_image",
                input_payload=input_payload_bytes,
            )

            # Call ExecuteAction on the scheduler
            print("Sending ExecuteAction request...")
            try:
                start_time = time.time()
                execute_response = stub.ExecuteAction(execute_request, timeout=10)
                run_id = execute_response.run_id
                print(f"Action submitted successfully. Run ID: {run_id}")
            except grpc.RpcError as e:
                print(f"Error calling ExecuteAction: {e.code()} - {e.details()}", file=sys.stderr)
                return
            except Exception as e:
                 print(f"An unexpected error occurred during ExecuteAction: {e}", file=sys.stderr)
                 return


            # Poll or wait for the result using GetRun
            print(f"Waiting for result of run {run_id}...")
            get_run_request = frontend_pb2.GetRunRequest(run_id=run_id)

            wait_timeout_seconds = 60 # How long the client will wait overall
            try:
                get_run_response = stub.GetRun(get_run_request, timeout=wait_timeout_seconds)
                end_time = time.time()
                print(f"Received result after {end_time - start_time:.2f} seconds.")

                if get_run_response.success:
                    print("Task completed successfully!")
                    output_payload_bytes = get_run_response.output_payload
                    print(f"Received output payload ({len(output_payload_bytes)} bytes).")

                    try:
                        image_bytes = msgpack.unpackb(output_payload_bytes, raw=False)
                        if not isinstance(image_bytes, bytes):
                             print("Error: Deserialized payload is not bytes!", file=sys.stderr)
                             return
                        print(f"Deserialized msgpack payload into raw image bytes ({len(image_bytes)} bytes).")
                    except Exception as e:
                        print(f"Error deserializing output payload with msgpack: {e}", file=sys.stderr)
                        return

                    # Save the image bytes to a file
                    try:
                        with open(output_file, "wb") as f:
                            f.write(image_bytes)
                        print(f"Image saved successfully to: {output_file}")
                    except IOError as e:
                        print(f"Error saving image to file '{output_file}': {e}", file=sys.stderr)

                else:
                    print(f"Task failed: {get_run_response.error_message}", file=sys.stderr)

            except grpc.RpcError as e:
                 # Handle potential timeouts or other gRPC errors during GetRun
                 if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                     print(f"Timed out waiting for result ({wait_timeout_seconds}s). The task might still be running.", file=sys.stderr)
                 else:
                    print(f"Error calling GetRun: {e.code()} - {e.details()}", file=sys.stderr)
            except Exception as e:
                 print(f"An unexpected error occurred during GetRun: {e}", file=sys.stderr)

    except grpc.RpcError as e:
        print(f"gRPC connection failed: {e.code()} - {e.details()}", file=sys.stderr)
        print("Please ensure the scheduler server is running and accessible.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    scheduler_target_addr = os.getenv("SCHEDULER_ADDR", "localhost:8080")
    test_prompt = "a futuristic cityscape at sunset, cyberpunk style"
    test_seed = 12345
    test_output_filename = "generated_test_image.png"

    print("--- Running Test Client ---")
    print(f"Scheduler Address: {scheduler_target_addr}")
    print(f"Prompt: {test_prompt}")
    print(f"Seed: {test_seed}")
    print(f"Output File: {test_output_filename}")
    print("--------------------------")

    run_generate_image(
        scheduler_addr=scheduler_target_addr,
        prompt=test_prompt,
        seed=test_seed,
        output_file=test_output_filename
    )

    print("--- Test Client Finished ---")

