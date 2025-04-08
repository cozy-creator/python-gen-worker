This is a python package, called gen_worker, which is used to decorate worker functions. It implements a grpc-client which communicates witht he gen-orchestrator, receives jobs, executes them using a running (entrypoint.py) and then returns the results.

The worker can also load models into and out of CUDA memory using PyTorch. It's designed to manage models in memory with a LRU cache.

---

Files in src/gen_worker/pb must be auto-generated in the gen-orchestrator repo, using the proto files. Go in there and run `task proto`

---
