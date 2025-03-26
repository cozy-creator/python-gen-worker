import os
import time
import grpc
import msgpack

from gen_orchestrator.pb import frontend_pb2, frontend_pb2_grpc

def main():
    scheduler_addr = os.environ.get("SCHEDULER_ADDR", "localhost:8080")
    
    channel = grpc.insecure_channel(scheduler_addr)
    client = frontend_pb2_grpc.FrontendServiceStub(channel)
    
    input_data = {
        "prompt": "A futuristic cityscape at sunset",
        "negative_prompt": "low resolution, blurry",
        "steps": 30,
        "num_images": 1,
        "guidance_scale": 7.5
    }
    
    input_payload = msgpack.packb(input_data, use_bin_type=True)
    
    action_options = frontend_pb2.ActionOptions(
        id="test-run-009", 
        timeout_ms=60000
    )
    
    request = frontend_pb2.ExecuteActionRequest(
        function_name="image_gen_action",
        input_payload=input_payload,
        options=action_options
    )
    
    response = client.ExecuteAction(request)
    run_id = response.run_id
    print("Submitted action with Run ID:", run_id)
    
    while True:
        time.sleep(2)
        get_request = frontend_pb2.GetRunRequest(run_id=run_id)
        get_response = client.GetRun(get_request)
        if get_response.success or get_response.error_message:
            break
        print("Waiting for result...")
    
    # Process the result
    if get_response.success:
        result = msgpack.unpackb(get_response.output_payload, raw=False)
        print("Task succeeded. Result:")
        print(result)
    else:
        print("Task failed with error:")
        print(get_response.error_message)

if __name__ == "__main__":
    main()
