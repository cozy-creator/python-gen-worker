import grpc
import logging
import time
import threading
import os
import signal
import queue
import psutil
import importlib
import inspect
import functools
from typing import Any, Callable, Dict, Optional, TypeVar, Iterator, List
import msgpack
import torch
import asyncio

# Use relative imports within the package
from .pb import worker_scheduler_pb2 as pb
from .pb import worker_scheduler_pb2_grpc as pb_grpc 
from .decorators import ResourceRequirements # Import ResourceRequirements for type hints if needed

from .model_interface import ModelManagementInterface

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use __name__ for logger

# Type variables for generic function signatures
I = TypeVar('I')  # Input type
O = TypeVar('O')  # Output type

# Generic type for action functions
ActionFunc = Callable[[Any, I], O]

HEARTBEAT_INTERVAL = 10  # seconds

class ActionContext:
    """Context object passed to action functions, allowing cancellation."""
    def __init__(self, run_id: str):
        self._run_id = run_id
        self._canceled = False
        self._cancel_event = threading.Event()

    @property
    def run_id(self) -> str:
        return self._run_id

    def is_canceled(self) -> bool:
        """Check if the action was canceled."""
        return self._canceled

    def cancel(self):
        """Mark the action as canceled."""
        if not self._canceled:
            self._canceled = True
            self._cancel_event.set()
            logger.info(f"Action {self.run_id} marked for cancellation.")

    def done(self) -> threading.Event:
        """Returns an event that is set when the action is cancelled."""
        return self._cancel_event

# Define the interceptor class correctly
class _AuthInterceptor(grpc.StreamStreamClientInterceptor):
    def __init__(self, token: str):
        self._token = token

    def intercept_stream_stream(self, continuation, client_call_details, request_iterator):
        metadata = list(client_call_details.metadata or [])
        metadata.append(('authorization', f'Bearer {self._token}'))
        new_details = client_call_details._replace(metadata=metadata)
        return continuation(new_details, request_iterator)

class Worker:
    """Worker implementation that connects to the scheduler via gRPC."""

    def __init__(
        self,
        scheduler_addr: str = "localhost:8080",
        user_module_names: List[str] = ["functions"], # Add new parameter for user modules
        worker_id: Optional[str] = None,
        auth_token: Optional[str] = None,
        use_tls: bool = False,
        reconnect_delay: int = 5,
        max_reconnect_attempts: int = 0,  # 0 means infinite retries
        model_manager: Optional[ModelManagementInterface] = None, # Optional model manager
    ):
        """Initialize a new worker.

        Args:
            scheduler_addr: Address of the scheduler service.
            user_module_names: List of Python module names containing user-defined @worker_function functions.
            worker_id: Unique ID for this worker (generated if not provided).
            auth_token: Optional authentication token.
            use_tls: Whether to use TLS for the connection.
            reconnect_delay: Seconds to wait between reconnection attempts.
            max_reconnect_attempts: Max reconnect attempts (0 = infinite).
            model_manager: Optional model manager.
        """
        self.scheduler_addr = scheduler_addr
        self.user_module_names = user_module_names # Store module names
        self.worker_id = worker_id or f"py-worker-{os.getpid()}"
        self.auth_token = auth_token
        self.use_tls = use_tls
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts

        self.deployment_id = os.getenv("DEPLOYMENT_ID", "") # Read DEPLOYMENT_ID env var
        if not self.deployment_id:
            logger.warning("DEPLOYMENT_ID environment variable not set for this worker!")

        self.tenant_id = os.getenv("TENANT_ID", "")
        self.runpod_pod_id = os.getenv("RUNPOD_POD_ID", "") # Read injected pod ID
        if not self.runpod_pod_id:
            logger.warning("RUNPOD_POD_ID environment variable not set for this worker!")

        logger.info(f"RUNPOD_POD_ID: {self.runpod_pod_id}")

        self._actions: Dict[str, Callable[[ActionContext, bytes], bytes]] = {}
        self._active_tasks: Dict[str, ActionContext] = {}
        self._active_tasks_lock = threading.Lock()
        self._discovered_resources: Dict[str, ResourceRequirements] = {} # Store resources per function

        self._gpu_busy_lock = threading.Lock()
        self._is_gpu_busy = False

        self._channel = None
        self._stub = None
        self._stream = None
        self._running = False
        self._stop_event = threading.Event()
        self._reconnect_count = 0
        self._outgoing_queue = queue.Queue()

        self._receive_thread = None
        self._heartbeat_thread = None

        self._model_manager = model_manager
        self._supported_model_ids_from_scheduler: Optional[List[str]] = None # To store IDs from scheduler
        self._model_init_done_event = threading.Event() # To signal model init is complete

        if self._model_manager:
            logger.info(f"ModelManager of type '{type(self._model_manager).__name__}' provided.")
        else:
            logger.info("No ModelManager provided. Worker operating in simple mode regarding models.")
            self._model_init_done_event.set() # No model init to wait for if no manager

        logger.info(f"Created worker: ID={self.worker_id}, DeploymentID={self.deployment_id or 'N/A'}, Scheduler={scheduler_addr}")

        # Discover functions before setting signals? Maybe after. Let's do it here.
        self._discover_and_register_functions()

        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)


    def _set_gpu_busy_status(self, busy: bool, func_name_for_log: str = ""):
        with self._gpu_busy_lock:
            if self._is_gpu_busy == busy:
                return
            self._is_gpu_busy = busy
        if func_name_for_log:
            logger.info(f"GPU status changed to {busy} due to function '{func_name_for_log}'.")
        else:
            logger.info(f"GPU status changed to {busy}.")


    def _get_gpu_busy_status(self) -> bool:
        with self._gpu_busy_lock:
            return self._is_gpu_busy


    def _discover_and_register_functions(self):
        """Discover and register functions marked with @worker_function."""
        logger.info(f"Discovering worker functions in modules: {self.user_module_names}...")
        discovered_count = 0
        for module_name in self.user_module_names:
            try:
                module = importlib.import_module(module_name)
                logger.debug(f"Inspecting module: {module_name}")
                for name, obj in inspect.getmembers(module):
                    if inspect.isfunction(obj) and hasattr(obj, '_is_worker_function'):
                        if getattr(obj, '_is_worker_function') is True:
                            # Found a decorated function
                            original_func = obj # Keep reference to the actual decorated function
                            func_name = original_func.__name__ # Use the real function name

                            if func_name in self._actions:
                                logger.warning(f"Function '{func_name}' from module '{module_name}' conflicts with an already registered function. Skipping.")
                                continue

                            resources: ResourceRequirements = getattr(original_func, '_worker_resources', ResourceRequirements())
                            self._discovered_resources[func_name] = resources

                            expects_pipeline_flag = resources.expects_pipeline_arg

                            # Create the wrapper for gRPC/msgpack interaction
                            def create_wrapper(captured_func: Callable, captured_name: str, func_expects_pipeline: bool = False) -> Callable[[ActionContext, bytes], bytes]:
                                @functools.wraps(captured_func) # Preserve metadata of original user func
                                def wrapper(ctx: ActionContext, pipeline_instance: Optional[Any], input_bytes: bytes) -> bytes:
                                    try:
                                        input_obj = msgpack.unpackb(input_bytes, raw=False)
                                        # Pass the context and deserialized input to the *original* user function
                                        if func_expects_pipeline: # Only pass pipeline if function expects it
                                            if pipeline_instance is None:
                                                err_msg = f"Function '{captured_name}' expected a pipeline argument, but None was provided by the Worker core."
                                                logger.error(err_msg)
                                                raise ValueError(err_msg)
                                            result = captured_func(ctx, pipeline_instance, input_obj)
                                        else:
                                            result = captured_func(ctx, input_obj) # For functions not needing a model

                                        if ctx.is_canceled():
                                             raise InterruptedError("Task was canceled during execution")
                                        # Ensure result is bytes after msgpack serialization
                                        packed_result = msgpack.packb(result, use_bin_type=True)
                                        # Check type explicitly after packing
                                        if not isinstance(packed_result, bytes):
                                            raise TypeError(f"Function {captured_name} did not return msgpack-serializable data resulting in bytes")
                                        return packed_result
                                    except InterruptedError as ie: # Catch cancellation specifically
                                        logger.warning(f"Function {captured_name} run {ctx.run_id} was interrupted.")
                                        raise # Re-raise to be handled in _execute_function
                                    except Exception as e:
                                        logger.exception(f"Error during execution of function {captured_name} (run_id: {ctx.run_id})")
                                        raise # Re-raise to be caught by _execute_function
                                return wrapper

                            self._actions[func_name] = create_wrapper(original_func, func_name, func_expects_pipeline=expects_pipeline_flag)
                            logger.info(f"Registered function: '{func_name}' from module '{module_name}' with resources: {resources}")
                            discovered_count += 1

            except ImportError:
                logger.error(f"Could not import user module: {module_name}")
            except Exception as e:
                logger.exception(f"Error during discovery in module {module_name}: {e}")

        if discovered_count == 0:
             logger.warning(f"No functions decorated with @worker_function found in specified modules: {self.user_module_names}")
        else:
             logger.info(f"Discovery complete. Found {discovered_count} worker functions.")

    def _send_message(self, message: pb.WorkerSchedulerMessage):
        """Add a message to the outgoing queue."""
        if self._running and not self._stop_event.is_set():
            try:
                self._outgoing_queue.put_nowait(message)
            except queue.Full:
                 logger.error("Outgoing message queue is full. Message dropped!")
        else:
            logger.warning("Attempted to send message while worker is stopping or stopped.")

    def connect(self) -> bool:
        """Connect to the scheduler.

        Returns:
            bool: True if connection was successful, False otherwise.
        """
        try:
            if self.use_tls:
                # TODO: Add proper credential loading if needed
                creds = grpc.ssl_channel_credentials()
                self._channel = grpc.secure_channel(self.scheduler_addr, creds)
            else:
                self._channel = grpc.insecure_channel(self.scheduler_addr)

            interceptors = []
            if self.auth_token:
                interceptors.append(_AuthInterceptor(self.auth_token))

            if interceptors:
                self._channel = grpc.intercept_channel(self._channel, *interceptors)

            self._stub = pb_grpc.SchedulerWorkerServiceStub(self._channel)

            # Start the bidirectional stream
            request_iterator = self._outgoing_message_iterator()
            self._stream = self._stub.ConnectWorker(request_iterator)

            logger.info(f"Attempting to connect to scheduler at {self.scheduler_addr}...")

            # Send initial registration immediately
            self._register_worker(is_heartbeat=False)

            # Start the receive loop in a separate thread *after* stream is initiated
            self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._receive_thread.start()

            # Start heartbeat thread
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()

            logger.info(f"Successfully connected to scheduler at {self.scheduler_addr}")
            self._reconnect_count = 0
            return True

        except grpc.RpcError as e:
            # Access code() and details() methods for RpcError
            code = e.code() if hasattr(e, 'code') and callable(e.code) else grpc.StatusCode.UNKNOWN # type: ignore
            details = e.details() if hasattr(e, 'details') and callable(e.details) else str(e) # type: ignore
            logger.error(f"Failed to connect to scheduler: {code} - {details}")
            self._close_connection()
            return False
        except Exception as e:
            logger.exception(f"Unexpected error connecting to scheduler: {e}")
            self._close_connection()
            return False

    def _outgoing_message_iterator(self) -> Iterator[pb.WorkerSchedulerMessage]:
        """Yields messages from the outgoing queue to send to the scheduler."""
        while not self._stop_event.is_set():
            try:
                # Block for a short time to allow stopping gracefully
                message = self._outgoing_queue.get(timeout=0.1)
                yield message
                # self._outgoing_queue.task_done() # Not needed if not joining queue
            except queue.Empty:
                continue
            except Exception as e:
                 if not self._stop_event.is_set():
                     logger.exception(f"Error in outgoing message iterator: {e}")
                     self._handle_connection_error()
                     break # Exit iterator on error

    def _heartbeat_loop(self):
        """Periodically sends heartbeat messages."""
        while not self._stop_event.wait(HEARTBEAT_INTERVAL):
            try:
                self._register_worker(is_heartbeat=True)
                logger.debug("Sent heartbeat to scheduler")
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"Error sending heartbeat: {e}")
                    self._handle_connection_error()
                    break # Stop heartbeating on error

    def _register_worker(self, is_heartbeat: bool = False):
        """Create and send a registration/heartbeat message."""
        try:
            mem = psutil.virtual_memory()
            cpu_cores = os.cpu_count() or 0

            gpu_count = 0
            gpu_total_mem = 0
            vram_models = []
            gpu_used_mem = 0

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    try:
                        props = torch.cuda.get_device_properties(0)
                        gpu_total_mem = props.total_memory
                        gpu_used_mem = torch.cuda.memory_allocated(0)
                    except Exception as gpu_err:
                         logger.warning(f"Could not get GPU properties: {gpu_err}")

            supports_model_loading_flag = False
            # current_models = []
            if self._model_manager:
                vram_models = self._model_manager.get_vram_loaded_models()
                supports_model_loading_flag = True 

            resources = pb.WorkerResources(
                worker_id=self.worker_id,
                deployment_id=self.deployment_id,
                # tenant_id=self.tenant_id,
                runpod_pod_id=self.runpod_pod_id,
                gpu_is_busy=self._get_gpu_busy_status(),
                cpu_cores=cpu_cores,
                memory_bytes=mem.total,
                gpu_count=gpu_count,
                gpu_memory_bytes=gpu_total_mem,
                gpu_memory_used_bytes=gpu_used_mem,
                available_functions=list(self._actions.keys()),
                available_models=vram_models,
                supports_model_loading=supports_model_loading_flag,
            )
            registration = pb.WorkerRegistration(
                resources=resources,
                is_heartbeat=is_heartbeat
            )
            message = pb.WorkerSchedulerMessage(worker_registration=registration)
            # logger.info(f"DEBUG: Preparing to send registration. Resource object: {resources}")
            # logger.info(f"DEBUG: Value being sent for runpod_pod_id: '{resources.runpod_pod_id}'")
            self._send_message(message)
        except Exception as e:
            logger.error(f"Failed to create or send registration/heartbeat: {e}")

    def run(self) -> None:
        """Run the worker, connecting to the scheduler and processing tasks."""
        if self._running:
            logger.warning("Worker is already running")
            return

        self._running = True
        self._stop_event.clear()
        self._reconnect_count = 0 # Reset reconnect count on new run

        while self._running and not self._stop_event.is_set():
            self._reconnect_count += 1
            logger.info(f"Connection attempt {self._reconnect_count}...")
            if self.connect():
                # Successfully connected, wait for stop signal or disconnection
                logger.info("Connection successful. Worker running.")
                self._stop_event.wait() # Wait here until stopped or disconnected
                logger.info("Worker run loop received stop/disconnect signal.")
                # If stopped normally (self.stop() called), _running will be False
                # If disconnected, connect() failed, threads stopped, _handle_connection_error called _stop_event.set()
            else:
                # Connection failed
                if self.max_reconnect_attempts > 0 and self._reconnect_count >= self.max_reconnect_attempts:
                    logger.error("Failed to connect after maximum attempts. Stopping worker.")
                    self._running = False # Ensure loop terminates
                    break

                if self._running and not self._stop_event.is_set():
                    logger.info(f"Connection attempt {self._reconnect_count} failed. Retrying in {self.reconnect_delay} seconds...")
                    # Wait for delay, but break if stop event is set during wait
                    if self._stop_event.wait(self.reconnect_delay):
                        logger.info("Stop requested during reconnect delay.")
                        break # Exit if stopped while waiting
            # After a failed attempt or disconnect, clear stop event for next retry
            if self._running:
                 self._stop_event.clear()

        # Cleanup after loop exits (either max attempts reached or manual stop)
        self.stop()

    def _handle_interrupt(self, sig, frame):
        """Handle interrupt signal (Ctrl+C)."""
        logger.info(f"Received signal {sig}, shutting down gracefully.")
        self.stop()

    def stop(self) -> None:
        """Stop the worker and clean up resources."""
        if not self._running and not self._stop_event.is_set(): # Check if already stopped or stopping
            # Avoid multiple stop calls piling up
            # logger.debug("Stop called but worker already stopped or stopping.")
            return

        logger.info("Stopping worker...")
        self._running = False # Signal loops to stop
        self._stop_event.set() # Wake up any waiting threads

        # Cancel any active tasks
        active_task_ids = []
        with self._active_tasks_lock:
            active_task_ids = list(self._active_tasks.keys())
            for run_id in active_task_ids:
                 ctx = self._active_tasks.get(run_id)
                 if ctx:
                     logger.debug(f"Cancelling active task {run_id} during stop.")
                     ctx.cancel()
            # Don't clear here, allow _execute_function to finish and remove

        # Wait for threads (give them a chance to finish)
        # Stop heartbeat first
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
             logger.debug("Joining heartbeat thread...")
             self._heartbeat_thread.join(timeout=1.0)

        # The outgoing iterator might be blocked on queue.get, stop_event wakes it

        # Close the gRPC connection (this might interrupt the receive loop)
        self._close_connection()

        # Wait for receive thread
        if self._receive_thread and self._receive_thread.is_alive():
            logger.debug("Joining receive thread...")
            self._receive_thread.join(timeout=2.0)

        # Clear outgoing queue after threads are stopped
        logger.debug("Clearing outgoing message queue...")
        while not self._outgoing_queue.empty():
            try:
                self._outgoing_queue.get_nowait()
            except queue.Empty:
                break

        logger.info("Worker stopped.")
        # Reset stop event in case run() is called again
        self._stop_event.clear()

    def _close_connection(self):
        """Close the gRPC channel and reset state."""
        if self._stream:
             try:
                  # Attempt to cancel the stream from the client side
                  # This might help the server side release resources quicker
                  # Note: Behavior might vary depending on server implementation
                  if hasattr(self._stream, 'cancel') and callable(self._stream.cancel):
                     self._stream.cancel() # type: ignore
                     logger.debug("gRPC stream cancelled.")
             except Exception as e:
                  logger.warning(f"Error cancelling gRPC stream: {e}")
        self._stream = None

        if self._channel:
            try:
                self._channel.close()
                logger.debug("gRPC channel closed.")
            except Exception as e:
                 logger.error(f"Error closing gRPC channel: {e}")
        self._channel = None
        self._stub = None


    def _receive_loop(self) -> None:
        """Loop to receive messages from the scheduler via the stream."""
        logger.info("Receive loop started.")
        try:
            if not self._stream:
                 logger.error("Receive loop started without a valid stream.")
                 # Don't call _handle_connection_error here, connect should have failed
                 return

            for message in self._stream:
                # Check stop event *before* processing
                if self._stop_event.is_set():
                    logger.debug("Stop event set during iteration, exiting receive loop.")
                    break
                try:
                    self._process_message(message)
                except Exception as e:
                    # Log errors processing individual messages but continue loop
                    logger.exception(f"Error processing message: {e}")

        except grpc.RpcError as e:
            # RpcError indicates a problem with the gRPC connection itself
            code = e.code() if hasattr(e, 'code') and callable(e.code) else grpc.StatusCode.UNKNOWN # type: ignore
            details = e.details() if hasattr(e, 'details') and callable(e.details) else str(e) # type: ignore

            if self._stop_event.is_set():
                 # If stopping, cancellation is expected
                 if code == grpc.StatusCode.CANCELLED:
                     logger.info("gRPC stream cancelled gracefully during shutdown.")
                 else:
                     logger.warning(f"gRPC error during shutdown: {code} - {details}")
            elif code == grpc.StatusCode.CANCELLED:
                logger.warning("gRPC stream unexpectedly cancelled by server or network.")
                self._handle_connection_error()
            elif code in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED, grpc.StatusCode.INTERNAL):
                 logger.warning(f"gRPC connection lost ({code}). Attempting reconnect.")
                 self._handle_connection_error()
            else:
                 logger.error(f"Unhandled gRPC error in receive loop: {code} - {details}")
                 self._handle_connection_error() # Attempt reconnect on unknown errors too
        except Exception as e:
            # Catch-all for non-gRPC errors in the loop
            if not self._stop_event.is_set():
                logger.exception(f"Unexpected error in receive loop: {e}")
                self._handle_connection_error() # Attempt reconnect
        finally:
             logger.info("Receive loop finished.")

    def _handle_connection_error(self):
         """Handles actions needed when a connection error occurs during run."""
         if self._running and not self._stop_event.is_set():
             logger.warning("Connection error detected. Signaling main loop to reconnect...")
             self._close_connection() # Ensure resources are closed before reconnect attempt
             self._stop_event.set() # Signal run loop to attempt reconnection
         # else: # Already stopping or stopped
             # logger.debug("Connection error detected but worker is already stopping.")


    def _process_message(self, message: pb.WorkerSchedulerMessage):
        """Process a single message received from the scheduler."""
        msg_type = message.WhichOneof('msg')
        # logger.debug(f"Received message of type: {msg_type}")

        if msg_type == 'run_request':
            self._handle_run_request(message.run_request)
        elif msg_type == 'load_model_cmd':
            # TODO: Implement model loading logic
            # model_id = message.load_model_cmd.model_id
            # logger.warning(f"Received load_model_cmd for {model_id}, but not yet implemented.")
            # # Send result back (failure for now)
            # result = pb.LoadModelResult(model_id=model_id, success=False, error_message="Model loading not implemented")
            # self._send_message(pb.WorkerSchedulerMessage(load_model_result=result))
            self._handle_load_model_cmd(message.load_model_cmd)
        elif msg_type == 'unload_model_cmd':
            # TODO: Implement model unloading logic
            model_id = message.unload_model_cmd.model_id
            logger.warning(f"Received unload_model_cmd for {model_id}, but not yet implemented.")
            result = pb.UnloadModelResult(model_id=model_id, success=False, error_message="Model unloading not implemented")
            self._send_message(pb.WorkerSchedulerMessage(unload_model_result=result))
        elif msg_type == 'interrupt_run_cmd':
            run_id = message.interrupt_run_cmd.run_id
            self._handle_interrupt_request(run_id)
        # Add handling for other message types if needed (e.g., config updates)
        elif msg_type == 'deployment_model_config':
            if self._model_manager:
                logger.info(f"Received DeploymentModelConfig: {message.deployment_model_config.supported_model_ids}")
                self._supported_model_ids_from_scheduler = list(message.deployment_model_config.supported_model_ids)
                self._model_init_done_event.clear() # Clear before starting new init
                model_init_thread = threading.Thread(target=self._process_deployment_config_async_wrapper, daemon=True)
                model_init_thread.start()
            else:
                logger.info("Received DeploymentModelConfig, but no model manager configured. Ignoring.")
                self._model_init_done_event.set() # Signal completion as there's nothing to do
        elif msg_type is None:
             logger.warning("Received empty message from scheduler.")
        else:
            logger.warning(f"Received unhandled message type: {msg_type}")

    def _process_deployment_config_async_wrapper(self):
        if not self._model_manager or self._supported_model_ids_from_scheduler is None:
            self._model_init_done_event.set()
            return
        
        loop = None
        try:
            # Get or create an event loop for this thread
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            loop.run_until_complete(
                self._model_manager.process_supported_models_config(
                    self._supported_model_ids_from_scheduler,
                    None # Pass downloader instance here if Worker creates/manages it
                )
            )
            logger.info("Model configuration and downloads (if any) processed.")
        except Exception as e:
            logger.exception(f"Error during model_manager.process_supported_models_config: {e}")
        finally:
            if loop and not loop.is_running() and not loop.is_closed(): # Clean up loop if we created it
                loop.close()
            self._model_init_done_event.set() # Signal completion or failure

    def _handle_load_model_cmd(self, cmd: pb.LoadModelCommand):
        model_id = cmd.model_id
        logger.info(f"Received LoadModelCommand for: {model_id}")
        success = False; error_msg = ""
        if not self._model_manager:
            error_msg = "LoadModelCommand: No model manager configured on worker."
            logger.error(error_msg)
        else:
            try:
                # Wait for initial model downloads if they haven't finished
                if not self._model_init_done_event.is_set():
                    logger.info(f"LoadModelCmd ({model_id}): Waiting for initial model setup...")
                    # Timeout for this wait, can be adjusted
                    if not self._model_init_done_event.wait(timeout=300.0): # 5 minutes
                         raise TimeoutError("Timeout waiting for model initialization before VRAM load.")
                
                logger.info(f"Model Memory Manager attempting to load '{model_id}' into VRAM...")
                # load_model_into_vram is async
                success = asyncio.run(self._model_manager.load_model_into_vram(model_id))
                if success: logger.info(f"Model '{model_id}' loaded to VRAM by Model Memory Manager.")
                else: error_msg = f"MMM.load_model_into_vram failed for '{model_id}'."; logger.error(error_msg)
            except Exception as e: error_msg = f"Exception in mmm.load_model_into_vram for '{model_id}': {e}"; logger.exception(error_msg)
        
        result = pb.LoadModelResult(model_id=model_id, success=success, error_message=error_msg)
        self._send_message(pb.WorkerSchedulerMessage(load_model_result=result))


    def _handle_run_request(self, request: pb.TaskExecutionRequest):
        """Handle a task execution request from the scheduler."""
        run_id = request.run_id
        function_name = request.function_name
        input_payload = request.input_payload
        required_model_id_for_exec = ""

        if request.required_models and len(request.required_models) > 0:
            required_model_id_for_exec = request.required_models[0]

        logger.info(f"Received Task request: run_id={run_id}, function={function_name}, model='{required_model_id_for_exec or 'None'}'")

        func_wrapper = self._actions.get(function_name)
        if not func_wrapper:
            error_msg = f"Unknown function requested: {function_name}"
            logger.error(error_msg)
            self._send_task_result(run_id, False, None, error_msg)
            return

        ctx = ActionContext(run_id)
        # Add to active tasks *before* starting thread
        with self._active_tasks_lock:
             # Double-check if task is already active (race condition mitigation)
             if run_id in self._active_tasks:
                  error_msg = f"Task with run_id {run_id} is already active (race condition?)."
                  logger.error(error_msg)
                  return # Avoid starting duplicate thread
             self._active_tasks[run_id] = ctx

        # Execute function in a separate thread to avoid blocking the receive loop
        thread = threading.Thread(target=self._execute_function, args=(ctx, function_name, func_wrapper, input_payload, required_model_id_for_exec), daemon=True)
        thread.start()

    def _handle_interrupt_request(self, run_id: str):
        """Handle a request to interrupt/cancel a running task."""
        logger.info(f"Received interrupt request for run_id={run_id}")
        with self._active_tasks_lock:
            ctx = self._active_tasks.get(run_id)
            if ctx:
                ctx.cancel() # Set internal flag and event
            else:
                logger.warning(f"Could not interrupt task {run_id}: Not found in active tasks.")

    def _execute_function(self, ctx: ActionContext, function_name: str, func_to_execute: Callable[[ActionContext, bytes], bytes], input_payload: bytes, required_model_id: str):
        """Execute the registered function and send the result/error back."""
        run_id = ctx.run_id
        output_payload: Optional[bytes] = None
        error_message: str = ""
        success = False

        # Determine if this function requires GPU and manage worker's GPU state
        func_requires_gpu = False
        resource_req = self._discovered_resources.get(function_name)
        if resource_req:
            func_requires_gpu = resource_req.requires_gpu
            func_expects_pipeline = resource_req.expects_pipeline_arg

        # Variable to track if this specific thread execution set the GPU busy
        this_thread_set_gpu_busy = False
        if func_requires_gpu:
            with self._gpu_busy_lock: # Lock to check and set self._is_gpu_busy atomically
                if not self._is_gpu_busy:
                    self._is_gpu_busy = True
                    this_thread_set_gpu_busy = True
                    logger.info(f"Worker GPU marked as BUSY by task {run_id} ({function_name}).")
                else:
                    logger.warning(f"Task {run_id} ({function_name}) requires GPU, but worker GPU was already marked busy. Proceeding...")

        active_pipeline_instance = None # To hold the pipeline for the user function
        try:
            if ctx.is_canceled(): 
                raise InterruptedError("Task cancelled before execution")
            
            if func_expects_pipeline:
                if not required_model_id:
                    raise ValueError(f"Function '{function_name}' expects a pipeline argument, but no model ID was provided.")
                
                if not self._model_manager:
                    raise RuntimeError(f"Function '{function_name}' expects a pipeline argument, but no model manager configured on worker.")
                
                if not self._model_init_done_event.is_set():
                    logger.info(f"Task {run_id} ({function_name}) waiting for initial model setup...")
                    if not self._model_init_done_event.wait(timeout=300.0): # 5 min timeout
                        raise TimeoutError(f"Timeout waiting for model initialization for task {run_id}")
                    logger.info(f"Initial model setup complete. Proceeding for task {run_id}.")
                
                logger.info(f"Task {run_id} ({function_name}) getting active pipeline for model '{required_model_id}'...")
                # get_active_pipeline is async
                active_pipeline_instance = asyncio.run(self._model_manager.get_active_pipeline(required_model_id))
                if not active_pipeline_instance:
                    raise RuntimeError(f"ModelManager failed to provide active pipeline for '{required_model_id}' for task {run_id}.")
                
                logger.info(f"Task {run_id} ({function_name}) obtained pipeline for model '{required_model_id}'.")

            # Execute the function wrapper (which handles deserialization/serialization)
            output_payload = func_to_execute(ctx, active_pipeline_instance, input_payload)
            # Check for cancellation *during* execution (func should check ctx.is_canceled)
            if ctx.is_canceled():
                raise InterruptedError("Task was cancelled during execution")

            success = True
            logger.info(f"Task {run_id} completed successfully.")

        except InterruptedError as e:
             error_message = str(e) or "Task was canceled"
             logger.warning(f"Task {run_id} was canceled: {error_message}")
             success = False # Explicitly set success to False on cancellation
        except (ValueError, RuntimeError, TimeoutError) as ve_rte_to: # Catch specific errors we raise
            error_message = f"{type(ve_rte_to).__name__}: {str(ve_rte_to)}"
            logger.error(f"Task {run_id} ({function_name}) failed pre-execution or during model acquisition: {error_message}")
            success = False
        except Exception as e:
            error_message = f"{type(e).__name__}: {str(e)}"
            logger.exception(f"Error executing function for run_id={run_id}: {error_message}")
            success = False
        finally:
            # Release the GPU if this thread set it busy
            if this_thread_set_gpu_busy:
                with self._gpu_busy_lock: # Lock to set self._is_gpu_busy
                    self._is_gpu_busy = False
                logger.info(f"Worker GPU marked as NOT BUSY by task {run_id} ({function_name}).")

            # Always send a result back, regardless of success, failure, or cancellation
            self._send_task_result(run_id, success, output_payload, error_message)
            # Remove from active tasks *after* sending result
            with self._active_tasks_lock:
                if run_id in self._active_tasks:
                    del self._active_tasks[run_id]
                # else: # Might have been removed by stop() already
                     # logger.warning(f"Task {run_id} not found in active tasks during cleanup.")


    def _send_task_result(self, run_id: str, success: bool, output_payload: Optional[bytes], error_message: str):
        """Send a task execution result back to the scheduler via the queue."""
        try:
            result = pb.TaskExecutionResult(
                run_id=run_id,
                success=success,
                output_payload=(output_payload or b'') if success else b'', # Default to b'' if None
                error_message=error_message if not success else ""
            )
            msg = pb.WorkerSchedulerMessage(run_result=result)
            self._send_message(msg)
            logger.debug(f"Queued task result for run_id={run_id}, success={success}")
        except Exception as e:
             # This shouldn't generally fail unless message creation has issues
             logger.error(f"Failed to create or queue task result for run_id={run_id}: {e}")

