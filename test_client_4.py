# minimal_client.py
import grpc
import os
import sys
try:
    from gen_worker.pb import frontend_pb2
    from gen_worker.pb import frontend_pb2_grpc
except ImportError:
    print("Import Error. Ensure PYTHONPATH is correct and protos are compiled.", file=sys.stderr)
    sys.exit(1)

SCHEDULER_ADDR = os.getenv("SCHEDULER_ADDR", "localhost:8080")
DEPLOYMENT_ID = "tenant-a-image-gen-app-v1"
FUNCTION_NAME = "generate_image"
# REQUIRED_MODEL = "ebara-pony-xl" # Test with this set
REQUIRED_MODEL = "auraflow" # Test with this None/empty to compare

print(f"--- Minimal Client Test ---")
print(f"Targeting: {SCHEDULER_ADDR}")
print(f"Model ID to send: {REQUIRED_MODEL}")

try:
    with grpc.insecure_channel(SCHEDULER_ADDR) as channel:
        print("Attempting channel ready check...")
        grpc.channel_ready_future(channel).result(timeout=5)
        print("Channel is ready.")
        stub = frontend_pb2_grpc.FrontendServiceStub(channel)

        # Dummy payload, msgpack not important for this connectivity test
        dummy_payload = b'\x81\xa3str\xa5hello' # msgpack for {"str": "hello"}

        request = frontend_pb2.ExecuteActionRequest(
            deployment_id=DEPLOYMENT_ID,
            function_name=FUNCTION_NAME,
            input_payload=dummy_payload
        )

        if REQUIRED_MODEL: # Only set it if it's not None/empty
            print(f"Setting request.required_model_id to: '{REQUIRED_MODEL}'")
            request.required_model_id = REQUIRED_MODEL
        else:
            print("Not setting request.required_model_id (it's None/empty).")
         
        print(f"Sending request: {request}")

        try:
            response = stub.ExecuteAction(request, timeout=10) # 10 sec timeout for the RPC
            print(f"SUCCESS: ExecuteAction response: {response.run_id}")
        except grpc.RpcError as e:
            print(f"GRPC ERROR during ExecuteAction: {e.code()} - {e.details()}", file=sys.stderr)
        except Exception as e_exec:
            print(f"GENERAL ERROR during ExecuteAction: {e_exec}", file=sys.stderr)

except grpc.FutureTimeoutError:
    print(f"TIMEOUT: Channel to {SCHEDULER_ADDR} not ready.", file=sys.stderr)
except Exception as e:
    print(f"Overall script error: {e}", file=sys.stderr)