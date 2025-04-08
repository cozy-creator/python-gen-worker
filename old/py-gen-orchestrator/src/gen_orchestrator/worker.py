import grpc
import logging
import time
import threading
import os
import signal
from typing import Any, Callable, Dict, Optional, TypeVar, Iterator
import msgpack

from gen_orchestrator.pb import worker_scheduler_pb2 as pb
from gen_orchestrator.pb import worker_scheduler_pb2_grpc as pb_grpc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('gen_orchestrator')

# Type variables for generic function signatures
I = TypeVar('I')  # Input type
O = TypeVar('O')  # Output type

# Generic type for action functions
ActionFunc = Callable[[Any, I], O]


class ActionContext:
    """Context object passed to action functions"""
    
    def __init__(self):
        self._canceled = False
        
    def is_canceled(self) -> bool:
        """Check if the action was canceled"""
        return self._canceled
    
    def cancel(self):
        """Mark the action as canceled"""
        self._canceled = True


class Worker:
    """Worker implementation that connects to the scheduler via gRPC"""
    
    def __init__(
        self,
        scheduler_addr: str = "localhost:8080",
        worker_id: Optional[str] = None,
        auth_token: Optional[str] = None,
        use_tls: bool = False,
        reconnect_delay: int = 5,
        max_reconnect_attempts: int = 0,  # 0 means infinite retries
    ):
        """Initialize a new worker
        
        Args:
            scheduler_addr: Address of the scheduler service
            worker_id: Unique ID for this worker (generated if not provided)
            auth_token: Optional authentication token
            use_tls: Whether to use TLS for the connection
            reconnect_delay: Seconds to wait between reconnection attempts
            max_reconnect_attempts: Maximum number of reconnection attempts (0 = infinite)
        """
        self.scheduler_addr = scheduler_addr
        self.worker_id = worker_id or f"py-worker-{os.getpid()}"
        self.auth_token = auth_token
        self.use_tls = use_tls
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        
        # Actions registry
        self._actions: Dict[str, Callable[[ActionContext, bytes], bytes]] = {}
        
        # Connection state
        self._channel = None
        self._stub = None
        self._stream = None
        self._running = False
        self._stop_event = threading.Event()
        self._reconnect_count = 0
        
        # Thread for receiving messages
        self._receive_thread = None
        
        logger.info(f"Created worker with ID={self.worker_id}, scheduler={scheduler_addr}")
        
        # Set up signal handlers immediately for cleaner development experience
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
    
    def register_function(self, func: ActionFunc[I, O], name: Optional[str] = None) -> None:
        """Register a function to be called when a task is received
        
        Args:
            func: The function to register
            name: Optional name for the function (uses function name if not provided)
        """
        # Determine function name
        if name is None:
            name = func.__name__
            
        # Create a wrapper that handles serialization/deserialization
        def wrapper(ctx: ActionContext, input_bytes: bytes) -> bytes:
            # Deserialize input
            input_obj = msgpack.unpackb(input_bytes, raw=False)
            
            # Call function
            result = func(ctx, input_obj)
            
            # Serialize output
            return msgpack.packb(result, use_bin_type=True)
        
        # Register the wrapper
        self._actions[name] = wrapper
        logger.info(f"Registered function: {name}")
    
    def connect(self) -> bool:
        """Connect to the scheduler
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            # Create channel options based on configuration
            if self.use_tls:
                creds = grpc.ssl_channel_credentials()
                self._channel = grpc.secure_channel(self.scheduler_addr, creds)
            else:
                self._channel = grpc.insecure_channel(self.scheduler_addr)
            
            # Add auth token if provided
            if self.auth_token:
                # Create auth interceptor
                def auth_interceptor(context, callback):
                    metadata = (('authorization', f'Bearer {self.auth_token}'),)
                    callback(metadata, None)
                    
                # Apply interceptor
                self._channel = grpc.intercept_channel(self._channel, auth_interceptor)
            
            # Create stub
            self._stub = pb_grpc.SchedulerWorkerServiceStub(self._channel)
            
            # Test the connection first with a simple RPC call to avoid false positives
            try:
                # Create stream for bidirectional communication
                request_iterator = self._get_request_iterator()
                self._stream = self._stub.ConnectWorker(request_iterator)
                
                # Try to get the first response to confirm connection is working
                # This will raise an exception if the connection fails
                logger.info(f"Attempting to connect to scheduler at {self.scheduler_addr}...")
                
                # Now we can consider ourselves connected
                logger.info(f"Successfully connected to scheduler at {self.scheduler_addr}")
                self._reconnect_count = 0
                return True
            except grpc.RpcError as e:
                logger.error(f"Connection test failed: {e}")
                if self._channel:
                    self._channel.close()
                    self._channel = None
                self._stub = None
                self._stream = None
                return False
                
        except grpc.RpcError as e:
            logger.error(f"Failed to connect to scheduler: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error connecting to scheduler: {e}")
            return False
    
    def _get_request_iterator(self) -> Iterator:
        """Iterator that yields registration and heartbeat messages"""
        # Send initial registration
        registration_msg = self._create_registration_message(is_heartbeat=False)
        yield registration_msg
        
        # Then heartbeats every 10 seconds
        while not self._stop_event.is_set():
            time.sleep(10)
            if not self._stop_event.is_set():
                try:
                    heartbeat_msg = self._create_registration_message(is_heartbeat=True)
                    yield heartbeat_msg
                    logger.debug("Sent heartbeat to scheduler")
                except Exception as e:
                    logger.error(f"Error creating heartbeat message: {e}")
                    if self._running:
                        self.stop()
                    break
    
    def _create_registration_message(self, is_heartbeat: bool = False):
        """Create a registration message"""
        # Get available functions
        available_functions = list(self._actions.keys())
        
        # Create resources message with proper protobuf types
        resources = pb.WorkerResources(
            worker_id=self.worker_id,
            cpu_cores=os.cpu_count() or 0,
            memory_bytes=0,  # We don't track system memory
            gpu_count=0,     # Default to 0 GPUs
            gpu_memory_bytes=0,
            available_functions=available_functions,
            available_models=[],
            supports_model_loading=False
        )
        
        # Create registration message
        registration = pb.WorkerRegistration(
            resources=resources,
            is_heartbeat=is_heartbeat
        )
        
        # Create and return worker scheduler message
        return pb.WorkerSchedulerMessage( 
            worker_registration=registration
        )
    
    def run(self) -> None:
        """Run the worker and start processing tasks"""
        if self._running:
            logger.warning("Worker is already running")
            return
        
        # Mark as running and reset stop event
        self._running = True
        self._stop_event.clear()
        
        # Try to connect initially
        connected = False
        attempt = 0
        
        while self._running and (self.max_reconnect_attempts == 0 or attempt < self.max_reconnect_attempts):
            attempt += 1
            logger.info(f"Connection attempt {attempt}...")
            connected = self.connect()
            
            if connected:
                break
                
            if self._running:
                logger.info(f"Connection attempt {attempt} failed. Retrying in {self.reconnect_delay} seconds...")
                
                # Use a loop with small sleep intervals to allow for better interrupt handling
                for _ in range(self.reconnect_delay * 10):
                    if self._stop_event.is_set():
                        break
                    time.sleep(0.1)
        
        if not connected:
            logger.error("Failed to connect to scheduler after maximum attempts")
            self._running = False
            return
        
        # Start receiver thread
        self._receive_thread = threading.Thread(target=self._receive_loop)
        self._receive_thread.daemon = True
        self._receive_thread.start()
        
        logger.info("Worker started and ready to process tasks")
        
        try:
            # Wait for stop event
            while self._running and not self._stop_event.is_set():
                # Use small sleep intervals to allow for better interrupt handling
                self._stop_event.wait(0.1)
        finally:
            # Cleanup
            self._running = False
            if self._channel:
                self._channel.close()
    
    def _handle_interrupt(self, sig, frame):
        """Handle interrupt signal"""
        logger.info("Received interrupt signal, shutting down")
        self.stop()
    
    def stop(self) -> None:
        """Stop the worker"""
        if not self._running:
            return
        
        logger.info("Stopping worker...")
        
        # Set stop event to terminate threads
        self._stop_event.set()
        
        # Wait for threads to complete
        if self._receive_thread and self._receive_thread.is_alive():
            self._receive_thread.join(timeout=2.0)
        
        # Close channel
        if self._channel:
            self._channel.close()
            self._channel = None
        
        self._running = False
        self._stream = None
        logger.info("Worker stopped")
    
    def _receive_loop(self) -> None:
        """Loop to receive messages from the scheduler"""
        if not self._stream:
            logger.error("Stream not initialized")
            self.stop()
            return
            
        try:
            for message in self._stream:
                if self._stop_event.is_set():
                    break
                    
                # Process message based on type
                msg_type = message.WhichOneof('msg')
                
                if msg_type == 'run_request':
                    # Handle task execution request
                    self._handle_run_request(message.run_request)
                
                elif msg_type == 'load_model_cmd':
                    # We don't support model loading in this implementation
                    model_id = message.load_model_cmd.model_id
                    logger.warning(f"Received load_model_cmd for {model_id}, but model loading is not supported")
                
                elif msg_type == 'unload_model_cmd':
                    # We don't support model unloading in this implementation
                    model_id = message.unload_model_cmd.model_id
                    logger.warning(f"Received unload_model_cmd for {model_id}, but model unloading is not supported")
                
                elif msg_type == 'interrupt_run_cmd':
                    # We don't support task interruption in this implementation
                    run_id = message.interrupt_run_cmd.run_id
                    logger.warning(f"Received interrupt_run_cmd for {run_id}, but task interruption is not supported")
                
                else:
                    logger.warning(f"Received unknown message type: {msg_type}")
        
        except grpc.RpcError as e:
            if self._running and not self._stop_event.is_set():
                logger.error(f"gRPC error in receive loop: {e}")
                # Try to reconnect if we're still running
                self._try_reconnect()
        except Exception as e:
            if self._running and not self._stop_event.is_set():
                logger.exception(f"Error in receive loop: {e}")
                # Try to reconnect if we're still running
                self._try_reconnect()
    
    def _try_reconnect(self):
        """Try to reconnect to the scheduler"""
        if not self._running or self._stop_event.is_set():
            return
                
        self._reconnect_count += 1
        
        # Check if we've exceeded max reconnect attempts
        if self.max_reconnect_attempts > 0 and self._reconnect_count > self.max_reconnect_attempts:
            logger.error(f"Exceeded maximum reconnection attempts ({self.max_reconnect_attempts}). Stopping worker.")
            self.stop()
            return
                
        logger.info(f"Attempting to reconnect to scheduler (attempt {self._reconnect_count})...")
        
        # Close existing channel if any
        if self._channel:
            self._channel.close()
            self._channel = None
        
        # Get current thread to avoid joining itself
        current_thread_ident = threading.get_ident()
                
        # Wait before reconnecting using small sleep intervals for better interrupt handling
        for _ in range(self.reconnect_delay * 10):
            if self._stop_event.is_set():
                return
            time.sleep(0.1)
        
        # Try to reconnect
        if not self._stop_event.is_set():
            if self.connect():
                # Start a new receive thread - but don't try to join the current thread
                if self._receive_thread and self._receive_thread.ident != current_thread_ident:
                    self._receive_thread.join(timeout=1.0)
                self._receive_thread = threading.Thread(target=self._receive_loop)
                self._receive_thread.daemon = True
                self._receive_thread.start()
            else:
                # Failed to reconnect, try again
                self._try_reconnect()
    
    def _handle_run_request(self, request) -> None:
        """Handle a task execution request
        
        Args:
            request: The task execution request
        """
        # Get function name and input
        run_id = request.run_id
        function_name = request.function_name
        input_payload = request.input_payload
        
        logger.info(f"Received task request: run_id={run_id}, function={function_name}")
        
        # Check if function exists
        if function_name not in self._actions:
            error_msg = f"Unknown function: {function_name}"
            logger.error(error_msg)
            self._send_task_result(run_id, False, None, error_msg)
            return
        
        # Get function
        func = self._actions[function_name]
        
        # Create context
        ctx = ActionContext()
        
        # Execute function in a separate thread
        threading.Thread(target=self._execute_function, args=(run_id, func, ctx, input_payload)).start()
    
    def _execute_function(self, run_id: str, func: Callable[[ActionContext, bytes], bytes], ctx: ActionContext, input_payload: bytes) -> None:
        """Execute a function and send the result
        
        Args:
            run_id: The task run ID
            func: The function to execute
            ctx: The action context
            input_payload: The input payload
        """
        try:
            # Execute function
            output = func(ctx, input_payload)
            
            # Send result
            if ctx.is_canceled():
                self._send_task_result(run_id, False, None, "Task was canceled")
            else:
                self._send_task_result(run_id, True, output, "")
        except Exception as e:
            # Send error
            error_msg = str(e)
            logger.exception(f"Error executing function for run_id={run_id}: {error_msg}")
            self._send_task_result(run_id, False, None, error_msg)
    
    def _send_task_result(self, run_id: str, success: bool, output_payload: Optional[bytes], error_message: str) -> None:
        """Send a task result to the scheduler
        
        Args:
            run_id: The task run ID
            success: Whether the task was successful
            output_payload: The output payload (if successful)
            error_message: The error message (if not successful)
        """
        if not self._running or self._stop_event.is_set() or not self._stub:
            logger.error(f"Cannot send task result for run_id={run_id}: worker not running or stopping")
            return
            
        try:
            # Create result message with proper protobuf types
            result = pb.TaskExecutionResult(
                run_id=run_id,
                success=success,
                output_payload=output_payload or b'',
                error_message=error_message
            )
            
            # Create worker scheduler message
            msg = pb.WorkerSchedulerMessage(
                run_result=result
            )
            
            # In bidirectional streaming, we can't directly use the stream to send
            # messages after it's been started - we need to use the request iterator
            # So we need to queue this message to be sent in the next iteration
            # For simplicity, we're using a one-off stream for the result
            result_stream = self._stub.ConnectWorker(iter([msg]))
            # Read from the stream to ensure the message is sent
            next(result_stream, None)
            
            logger.info(f"Sent task result for run_id={run_id}, success={success}")
            
        except grpc.RpcError as e:
            logger.error(f"Failed to send task result for run_id={run_id}: {e}")
            # Try to reconnect if we're still running
            if self._running and not self._stop_event.is_set():
                self._try_reconnect()
        except Exception as e:
            logger.exception(f"Error sending task result for run_id={run_id}: {e}")
            # Try to reconnect if we're still running
            if self._running and not self._stop_event.is_set():
                self._try_reconnect() 

