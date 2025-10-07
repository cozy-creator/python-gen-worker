import os
import sys
import time
import grpc
import msgpack
import concurrent.futures
import random
from typing import Optional
from dotenv import load_dotenv

try:
    from gen_worker.pb import frontend_pb2
    from gen_worker.pb import frontend_pb2_grpc
except ImportError:
    print("Error: Could not import protobuf definitions.", file=sys.stderr)
    sys.exit(1)

load_dotenv()

# === Configuration ===
SCHEDULER_ADDR = os.getenv("SCHEDULER_ADDR", "localhost:8080")
DEPLOYMENT_ID = os.getenv("DEPLOYMENT_ID", "tenant-a-image-gen-app-v1")

# Your actual model list
AVAILABLE_MODELS = [
    "ebara-pony-xl",
    "real.dream.pony",
    "playground2.5",
    "pony.realism",
    "illustrious.xl",
]

# Test prompts variety
TEST_PROMPTS = [
    "cowgirl riding a horse, cinematic lighting",
    "futuristic cityscape at night, neon lights",
    "portrait of a warrior, dramatic lighting",
    "sunset over mountains, landscape photography",
    "abstract art, colorful geometric shapes",
]

# =====================================================
def execute_and_await(stub: frontend_pb2_grpc.FrontendServiceStub,
                      deployment_id: str,
                      function_name: str,
                      input_data: dict,
                      required_model_id: Optional[str] = None,
                      wait_timeout: int = 120) -> Optional[dict]:
    """Executes an action and waits for its result."""
    try:
        payload = msgpack.packb(input_data, use_bin_type=True)
    except Exception as e:
        print(f"  [Error] serializing input: {e}", file=sys.stderr)
        return None

    # Submit
    try:
        request = frontend_pb2.ExecuteActionRequest(
            deployment_id=deployment_id,
            function_name=function_name,
            input_payload=payload,
        )
        if required_model_id:
            request.required_model_id = required_model_id

        resp = stub.ExecuteAction(request, timeout=30)
        run_id = resp.run_id
    except grpc.RpcError as e:
        print(f"  [gRPC Error] ExecuteAction: {e.code()} {e.details()}", file=sys.stderr)
        return None

    # Wait
    try:
        start = time.time()
        get_resp = stub.GetRun(
            frontend_pb2.GetRunRequest(run_id=run_id),
            timeout=wait_timeout
        )
        elapsed = time.time() - start

        if not get_resp.success:
            print(f"  [Task Failed] {get_resp.error_message}", file=sys.stderr)
            return None

        if not get_resp.output_payload:
            return {}

        data = msgpack.unpackb(get_resp.output_payload, raw=False)
        return {"elapsed": elapsed, "data": data}

    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
            print(f"  [Timeout] waiting for result ({wait_timeout}s)", file=sys.stderr)
        else:
            print(f"  [gRPC Error] GetRun: {e.code()} {e.details()}", file=sys.stderr)
        return None


def run_workflow(task_id: int,
                 stub: frontend_pb2_grpc.FrontendServiceStub,
                 deployment_id: str,
                 prompt: str,
                 model_id: str,
                 delay_before: float = 0):
    """One end-to-end workflow with optional delay."""
    
    # Optional delay to simulate staggered arrivals
    if delay_before > 0:
        time.sleep(delay_before)
    
    start_time = time.time()
    print(f"\n[Task {task_id}] Starting at T+{start_time - SCRIPT_START:.1f}s")
    print(f"  Model: {model_id}")
    print(f"  Prompt: {prompt[:50]}...")
    
    seed = 9876 + task_id
    filename = f"task_{task_id}_{seed}.png"

    result = execute_and_await(
        stub, 
        deployment_id, 
        "generate_and_upload_image",
        {
            "prompt": prompt, 
            "seed": seed,
            "filename": filename,
            "num_inference_steps": 28,
            "guidance_scale": 7.5,
            "width": 1024,
            "height": 1024
        },
        required_model_id=model_id,
        wait_timeout=600
    )
    
    total_time = time.time() - start_time
    
    if result and isinstance(result, dict):
        s3_url = result.get("data", {}).get("s3_url")
        if s3_url:
            print(f"[Task {task_id}] ✅ Complete in {total_time:.1f}s (processing: {result.get('elapsed', 0):.1f}s)")
        else:
            print(f"[Task {task_id}] ❌ Failed - no S3 URL")
    else:
        print(f"[Task {task_id}] ❌ Failed after {total_time:.1f}s")


# =====================================================
# TEST SCENARIOS
# =====================================================

def test_burst_traffic(stub, deployment_id):
    """Simulate sudden burst of traffic (like your original test)"""
    print("\n" + "="*60)
    print("TEST 1: BURST TRAFFIC (11 concurrent jobs)")
    print("="*60)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=11) as executor:
        futures = [
            executor.submit(
                run_workflow, 
                i, 
                stub, 
                deployment_id,
                random.choice(TEST_PROMPTS),
                random.choice(AVAILABLE_MODELS),
                0  # No delay - all at once
            )
            for i in range(11)
        ]
        concurrent.futures.wait(futures)


def test_gradual_ramp(stub, deployment_id):
    """Simulate gradually increasing traffic"""
    print("\n" + "="*60)
    print("TEST 2: GRADUAL RAMP (15 jobs over 2 minutes)")
    print("="*60)
    
    # Start with 3 jobs, then add more every 30 seconds
    waves = [
        (3, 0),    # 3 jobs immediately
        (4, 30),   # 4 more after 30s
        (4, 60),   # 4 more after 60s
        (4, 90),   # 4 more after 90s
    ]
    
    all_futures = []
    task_id = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        for count, delay in waves:
            for _ in range(count):
                future = executor.submit(
                    run_workflow,
                    task_id,
                    stub,
                    deployment_id,
                    random.choice(TEST_PROMPTS),
                    random.choice(AVAILABLE_MODELS),
                    delay
                )
                all_futures.append(future)
                task_id += 1
        
        concurrent.futures.wait(all_futures)


def test_sustained_load(stub, deployment_id):
    """Simulate sustained steady traffic"""
    print("\n" + "="*60)
    print("TEST 3: SUSTAINED LOAD (20 jobs, one every 5 seconds)")
    print("="*60)
    
    all_futures = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for i in range(20):
            future = executor.submit(
                run_workflow,
                i,
                stub,
                deployment_id,
                random.choice(TEST_PROMPTS),
                random.choice(AVAILABLE_MODELS),
                i * 5  # 5 second intervals
            )
            all_futures.append(future)
        
        concurrent.futures.wait(all_futures)


def test_model_switching(stub, deployment_id):
    """Test rapid model switching to validate model loading"""
    print("\n" + "="*60)
    print("TEST 4: MODEL SWITCHING (8 jobs, different models)")
    print("="*60)
    
    # Intentionally use different models to test loading
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                run_workflow,
                i,
                stub,
                deployment_id,
                random.choice(TEST_PROMPTS),
                AVAILABLE_MODELS[i % len(AVAILABLE_MODELS)],  # Cycle through models
                i * 2  # 2 second stagger
            )
            for i in range(8)
        ]
        concurrent.futures.wait(futures)


def test_spike_after_quiet(stub, deployment_id):
    """Test scaling after quiet period (tests scale-down and scale-up)"""
    print("\n" + "="*60)
    print("TEST 5: SPIKE AFTER QUIET (3 jobs, wait 6min, then 10 jobs)")
    print("="*60)
    
    # Initial small batch
    print("\n[Phase 1] Sending 3 initial jobs...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                run_workflow,
                i,
                stub,
                deployment_id,
                random.choice(TEST_PROMPTS),
                random.choice(AVAILABLE_MODELS),
                0
            )
            for i in range(3)
        ]
        concurrent.futures.wait(futures)
    
    # Wait for idle timeout (6 minutes to trigger scale-down)
    print("\n[Phase 2] Waiting 6 minutes for scale-down (idle timeout)...")
    for remaining in range(360, 0, -30):
        print(f"  {remaining//60}m {remaining%60}s remaining...")
        time.sleep(30)
    
    # Then sudden spike
    print("\n[Phase 3] Sending 10-job spike after quiet period...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(
                run_workflow,
                i + 100,  # Different IDs
                stub,
                deployment_id,
                random.choice(TEST_PROMPTS),
                random.choice(AVAILABLE_MODELS),
                0
            )
            for i in range(10)
        ]
        concurrent.futures.wait(futures)


def test_mixed_models_burst(stub, deployment_id):
    """Test burst with mixed models (most realistic)"""
    print("\n" + "="*60)
    print("TEST 6: MIXED MODEL BURST (12 jobs, 3 different models)")
    print("="*60)
    
    # Use 3 specific models, 4 jobs each
    model_distribution = [
        ("ebara-pony-xl", 4),
        ("flux.1-dev", 4),
        ("sdxl.base", 4),
    ]
    
    all_futures = []
    task_id = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        for model, count in model_distribution:
            for _ in range(count):
                future = executor.submit(
                    run_workflow,
                    task_id,
                    stub,
                    deployment_id,
                    random.choice(TEST_PROMPTS),
                    model,
                    0  # All at once
                )
                all_futures.append(future)
                task_id += 1
        
        concurrent.futures.wait(all_futures)


# =====================================================
# MAIN TEST RUNNER
# =====================================================

SCRIPT_START = None

if __name__ == "__main__":
    print("="*60)
    print("COMPREHENSIVE SCALING TEST SUITE")
    print("="*60)
    print(f"Scheduler: {SCHEDULER_ADDR}")
    print(f"Deployment: {DEPLOYMENT_ID}")
    print(f"Available Models: {len(AVAILABLE_MODELS)}")
    print("="*60)
    
    SCRIPT_START = time.time()
    
    with grpc.insecure_channel(SCHEDULER_ADDR) as channel:
        stub = frontend_pb2_grpc.FrontendServiceStub(channel)
        
        # Run tests based on user choice
        print("\nSelect test to run:")
        print("1. Burst Traffic (11 jobs at once)")
        print("2. Gradual Ramp (15 jobs over 2 min)")
        print("3. Sustained Load (20 jobs, one every 5s)")
        print("4. Model Switching (8 jobs, different models)")
        print("5. Spike After Quiet (tests scale-down + scale-up)")
        print("6. Mixed Model Burst (12 jobs, 3 models)")
        print("7. Run ALL tests sequentially")
        
        choice = input("\nEnter choice (1-7): ").strip()
        
        test_map = {
            "1": test_burst_traffic,
            "2": test_gradual_ramp,
            "3": test_sustained_load,
            "4": test_model_switching,
            "5": test_spike_after_quiet,
            "6": test_mixed_models_burst,
        }
        
        if choice == "7":
            # Run all tests
            for test_func in test_map.values():
                test_func(stub, DEPLOYMENT_ID)
                print("\n⏳ Waiting 2 minutes before next test...")
                time.sleep(120)
        elif choice in test_map:
            test_map[choice](stub, DEPLOYMENT_ID)
        else:
            print("Invalid choice, running burst test as default...")
            test_burst_traffic(stub, DEPLOYMENT_ID)
    
    total_duration = time.time() - SCRIPT_START
    print("\n" + "="*60)
    print(f"✅ ALL TESTS COMPLETE - Total time: {total_duration/60:.1f} minutes")
    print("="*60)