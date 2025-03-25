from pydantic import BaseModel, Field
# from gen_orchestrator import Worker, ActionContext


# TO DO: this should connect to an orchestrator based on environment variables
# TO DO: install packages based on evnrionment variables
# TO DO: import installed packages here, then register their functions with the orchestrator via worker.register_function()
# TO DO: worker.run() should block until interrupted
def main():
    """Run the example worker"""
    print("Hello from pytorch-worker!")
    
    # Create a worker that connects to the scheduler
    # worker = Worker(
    #     scheduler_addr="localhost:8080",
    #     worker_id="image-gen-worker-1"
    # )
    
    # Register functions here...
    # worker.register_function(image_gen_action)
    
    # Run the worker (this blocks until interrupted)
    # try:
    #     worker.run()
    # except KeyboardInterrupt:
    #     print("Worker interrupted, shutting down")
    # finally:
    #     worker.stop()

