These are worker-runtimes and worker functions.

The runtimes provide a set of dependencies and an environment for execution inside of a Docker container. They then install a set of functions (which are separate packages) to execute, and register them with the orchestrator. When the worker receives an Action (function call + input), it executes the function and returns the result to the orchestrator.

Worker functions are not isolated from each other: they all run in the same environment, and share the same installed dependencies and filesystem. It's up to the worker to share usage of the GPU amongst functions. This is necessary because Kubernetes can only assign GPUs to pods, and doesn't have any notion of pods sharing a GPU.
