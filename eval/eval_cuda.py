"""
Helpers for Evaluations
Copied and then Adapted from the KernelBench evaluation code

Key Enhancement for Multi-Stream CUDA Models:

In eval_kernel_against_ref(), we modified the timing logic to properly handle models 
that create their own CUDA streams. This ensures accurate performance measurements 
for models with complex stream management.

Original timing code:
    with torch.cuda.stream(custom_model_stream):
        start_event.record(custom_model_stream)
        custom_model(*inputs)                    
        end_event.record(custom_model_stream)

Enhanced timing code:
    with torch.cuda.stream(custom_model_stream):
        start_event.record(custom_model_stream)
        custom_model(*inputs)
        
        # Wait for all model streams to complete before recording end event
        if custom_model_streams:
            for stream in custom_model_streams:
                custom_model_stream.wait_stream(stream)
        
        end_event.record(custom_model_stream)

This enhancement prevents timing inaccuracies when models use internal streams 
for operations like CUDA graphs, asynchronous kernels, or parallel execution.
Without this synchronization, timing measurements could complete before the 
model's actual GPU work finishes, leading to artificially fast results.

"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
import torch
import torch.nn as nn
import subprocess
import random
import json
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time
from datetime import datetime, timezone, timedelta
from typing import Tuple, List, Dict, Union, Optional, Callable




pst_tz = timezone(timedelta(hours=-8))

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")


def execute_model_with_timeout(
    model_src: str, 
    context: Dict, 
    timeout: float = 300.0,
    build_directory: Optional[str] = None,
    use_process_isolation: bool = False,
    info_string: str = ""
) -> Tuple[bool, str, Optional[float]]:
    """
    Execute model source code with a time limit.
    
    Args:
        model_src: Source code to execute (can be original_model_src or custom_model_src)
        context: Dictionary to execute the code in
        timeout: Maximum time in seconds to allow for execution (default: 300s = 5 minutes)
        build_directory: Optional build directory for CUDA extensions
        use_process_isolation: Use multiprocessing instead of threading (slower but more robust)
        
    Returns:
        Tuple[bool, str, Optional[float]]: (success, error_message, execution_time)
            - success: True if execution completed within timeout, False otherwise
            - error_message: Error details if execution failed, empty string if successful
            - execution_time: Time taken for execution in seconds, None if failed
            
    Note:
        ThreadPoolExecutor cannot interrupt blocking operations like time.sleep(), 
        network requests, or infinite loops. The timeout detection works correctly,
        but background threads may continue running until the blocking operation completes.
        For CUDA code, this is usually not an issue as compilation errors are detected quickly.
    """
    # Format info_string for consistent display
    info_prefix = f"[{info_string}] " if info_string else ""
    
    # Prepare source code with build directory if provided
    if build_directory:
        context["BUILD_DIRECTORY"] = build_directory
        model_src = (
            "import os\n" 
            f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
        ) + model_src

    # Static analysis for potentially problematic patterns
    potentially_hanging_patterns = [
        ('time.sleep(', 'time.sleep() calls'),
        ('requests.get(', 'network requests'),
        ('urllib.request.', 'URL requests'),
        ('input(', 'user input'),
        ('while True:', 'infinite loops'),
        ('subprocess.', 'subprocess calls'),
    ]
    
    detected_patterns = []
    for pattern, description in potentially_hanging_patterns:
        if pattern in model_src:
            detected_patterns.append(description)
    
    if detected_patterns:
        print(f"{info_prefix}[execute_model_with_timeout] WARNING: Detected potentially blocking operations:")
        for pattern in detected_patterns:
            print(f"{info_prefix}  - {pattern}")
        print(f"{info_prefix}[execute_model_with_timeout] These may cause hanging if they block indefinitely.")
        print(f"{info_prefix}[execute_model_with_timeout] Consider using use_process_isolation=True for risky code.")
        
        # Check for extremely problematic patterns that should be blocked
        blocking_patterns = ['time.sleep(', 'input(', 'while True:']
        should_block = any(pattern in model_src for pattern, _ in potentially_hanging_patterns 
                          if pattern in blocking_patterns)
        
        if should_block and not use_process_isolation:
            error_msg = f"Code contains blocking patterns that may cause indefinite hanging: {detected_patterns}"
            print(f"{info_prefix}[execute_model_with_timeout] BLOCKING EXECUTION: {error_msg}")
            print(f"{info_prefix}[execute_model_with_timeout] Use use_process_isolation=True to override")
            return False, error_msg, None

    def _execute_code():
        """Helper function to execute the code in a separate thread"""
        try:
            compile(model_src, "<string>", "exec")
            exec(model_src, context)
            return True
        except Exception as e:
            raise e

    try:
        isolation_method = "process isolation" if use_process_isolation else "thread isolation"
        print(f"{info_prefix}Executing model code with {timeout}s timeout using {isolation_method}...")
        
        if use_process_isolation:
            # Use multiprocessing (more robust but has limitations with CUDA)
            import multiprocessing as mp
            print(f"{info_prefix}[execute_model_with_timeout] WARNING: Process isolation may not work well with CUDA contexts")
            
            def _execute_in_process():
                try:
                    compile(model_src, "<string>", "exec")
                    local_context = {}
                    exec(model_src, local_context)
                    return True
                except Exception as e:
                    raise e
            
            process = mp.Process(target=_execute_in_process)
            t1 = time.time()
            process.start()
            process.join(timeout=timeout)
            t2 = time.time()
            execution_time = t2 - t1
            
            if process.is_alive():
                print(f"{info_prefix}[execute_model_with_timeout] Process timeout - terminating")
                process.terminate()
                process.join(timeout=5.0)
                if process.is_alive():
                    process.kill()
                    process.join()
                
                error_msg = f"Execution timeout after {execution_time:.6f} seconds"
                print(f"{info_prefix}[execute_model_with_timeout] {error_msg}")
                return False, error_msg, None
            
            if process.exitcode == 0:
                print(f"{info_prefix}Model code execution completed successfully")
                # Note: Process isolation doesn't update the context
                print(f"{info_prefix}[execute_model_with_timeout] Note: Context not updated due to process isolation")
                return True, "", execution_time
            else:
                error_msg = f"Process exited with code {process.exitcode}"
                return False, error_msg, None
        
        else:
            # Use threading (faster, works with CUDA, but can't interrupt blocking operations)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_execute_code)
                try:
                    t1 = time.time()
                    future.result(timeout=timeout)
                    t2 = time.time()
                    execution_time = t2 - t1
                    print(f"{info_prefix}Model code execution completed successfully")
                    return True, "", execution_time
                    
                except TimeoutError:
                    future.cancel()  # This won't stop blocking operations
                    elapsed_time = time.time() - t1
                    error_msg = f"Execution timeout after {elapsed_time:.6f} seconds"
                    print(f"{info_prefix}[execute_model_with_timeout] {error_msg}")
                    print(f"{info_prefix}[execute_model_with_timeout] Source code length: {len(model_src)} chars")
                    print(f"{info_prefix}[execute_model_with_timeout] First 200 chars: {model_src[:200]}...")
                    if detected_patterns:
                        print(f"{info_prefix}[execute_model_with_timeout] Note: Background thread may still be running due to blocking operations")
                    return False, error_msg, None
                
    except SyntaxError as e:
        error_msg = f"Syntax Error: {e}"
        print(f"{info_prefix}[execute_model_with_timeout] {error_msg}")
        print(f"{info_prefix}[execute_model_with_timeout] Source code length: {len(model_src)} chars")
        print(f"{info_prefix}[execute_model_with_timeout] First 200 chars: {model_src[:200]}...")
        return False, error_msg, None
        
    except Exception as e:
        error_msg = f"Runtime Error: {e}"
        print(f"{info_prefix}[execute_model_with_timeout] {error_msg}")
        print(f"{info_prefix}[execute_model_with_timeout] Source code length: {len(model_src)} chars")
        print(f"{info_prefix}[execute_model_with_timeout] First 200 chars: {model_src[:200]}...")
        return False, error_msg, None
    


def set_seed(seed: int):
    torch.manual_seed(seed)
    # NOTE: this only sets on current cuda device
    torch.cuda.manual_seed(seed)


def load_original_model_and_inputs(
    model_original_src: str, context: Dict, timeout: float = 300.0, info_string: str = ""
) -> Tuple[nn.Module, Callable, Callable]:
    """
    Load class from original NN.module pytorch code
    this is pytorch reference and we feed that to model to see if there will be any improvement
    
    Args:
        model_original_src: Source code for the original model
        context: Dictionary to execute the code in
        timeout: Maximum time in seconds to allow for code execution (default: 300s = 5 minutes)
        info_string: Information string for consistent logging
    """
    # Format info_string for consistent display
    info_prefix = f"[{info_string}] " if info_string else ""
    
    # Execute the model source code with timeout
    success, error_msg, execution_time = execute_model_with_timeout(
        model_src=model_original_src,
        context=context,
        timeout=timeout,
        build_directory=None,  # Original models typically don't need CUDA extensions
        info_string=info_string
    )
    
    if not success:
        print(f"{info_prefix}[load_original_model_and_inputs] Failed to execute original model code: {error_msg}")
        return None

    # these should be defined in the original model code and present in the context
    get_init_inputs_fn = context.get("get_init_inputs")
    get_inputs_fn = context.get("get_inputs")
    Model = context.get("Model")
    return (Model, get_init_inputs_fn, get_inputs_fn)


def load_custom_model(
    model_custom_src: str, context: Dict, build_directory: Optional[str] = None, timeout: float = 300.0, info_string: str = ""
) -> Optional[nn.Module]:
    """
    Load class from custom NN.module pytorch code
    this is the code output by LLM with calls to custom cuda kernels
    
    Args:
        model_custom_src: Source code for the custom model
        context: Dictionary to execute the code in
        build_directory: Directory for CUDA extensions
        timeout: Maximum time in seconds to allow for code execution (default: 300s = 5 minutes)
        info_string: Information string for consistent logging
    """
    # Format info_string for consistent display
    info_prefix = f"[{info_string}] " if info_string else ""
    
    # Execute the model source code with timeout
    success, error_msg, execution_time = execute_model_with_timeout(
        model_src=model_custom_src,
        context=context,
        timeout=timeout,
        build_directory=build_directory,
        info_string=info_string
    )
    
    if not success:
        print(f"{info_prefix}[load_custom_model] Failed to execute custom model code: {error_msg}")
        return None
    
    if execution_time is not None:
        print(f"{info_prefix}[load_custom_model] Model loaded successfully in {execution_time:.2f}s")
    
    ModelNew = context.get("ModelNew")
    
    # Debug: Show what's in the context
    print(f"{info_prefix}[load_custom_model] Context keys: {list(context.keys())}")
    print(f"{info_prefix}[load_custom_model] ModelNew from context: {ModelNew}")
    
    # Validate that ModelNew was properly defined
    if ModelNew is None:
        print(f"{info_prefix}[load_custom_model] Error: ModelNew was not defined in the custom model source code")
        print(f"{info_prefix}[load_custom_model] Make sure your custom model source includes: ModelNew = YourModelClass")
        print(f"{info_prefix}[load_custom_model] Available in context: {[k for k in context.keys() if not k.startswith('__')]}")
        return None
    
    if not callable(ModelNew):
        print(f"{info_prefix}Error: ModelNew is not callable (got {type(ModelNew)})")
        print(f"{info_prefix}Make sure ModelNew is a class that can be instantiated")
        return None
    
    # Additional validation - check if it's a class
    if not isinstance(ModelNew, type):
        print(f"{info_prefix}Error: ModelNew should be a class, got {type(ModelNew)}")
        print(f"{info_prefix}Example: class MyModel(nn.Module): ... then ModelNew = MyModel")
        return None
    
    return ModelNew


def graceful_eval_cleanup(curr_context: Dict, device: torch.device):
    """
    Clean up env, gpu cache, and compiled CUDA extensions after evaluation
    """  # delete ran-specific function definitions before next eval run
    del curr_context
    # Clear CUDA cache and reset GPU state
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

        # does this help?
        torch.cuda.reset_peak_memory_stats(device=device)

        torch.cuda.synchronize(
            device=device
        )  # Wait for all CUDA operations to complete

    # _cleanup_cuda_extensions() # SIMON NOTE: is this necessary?


def check_kernel_correctness(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 5,
    verbose: bool = False,
    build_dir: Optional[os.PathLike] = None,
    device: Optional[torch.device] = None,
    timeout: float = 300.0,
    info_string: str = ""
) -> Tuple[bool, str, Dict]:
    """
    Check correctness of custom CUDA kernel against reference implementation.
    
    Args:
        original_model_src: Source code for the original/reference model
        custom_model_src: Source code for the custom CUDA kernel model
        seed_num: Base seed for reproducible testing
        num_correct_trials: Number of trials with different inputs to test
        verbose: Whether to print detailed progress
        build_dir: Directory for CUDA extensions
        device: CUDA device to run on (defaults to current device)
        timeout: Timeout for model loading in seconds
        
    Returns:
        tuple[bool, str, dict]: (success, error_message, metadata)
            - success: True if all correctness trials pass
            - error_message: Error details if failed, empty string if successful
            - metadata: Dictionary with trial details and statistics
    """
    if device is None:
        raise Exception("Device is not set for check_kernel_correctness")
    
    if not torch.cuda.is_available():
        return False, "CUDA is not available", {}
    
    # Define pst_tz at the beginning of the function
    pst_tz = timezone(timedelta(hours=-8))
    
    # Format info_string for consistent display
    info_prefix = f"[{info_string}] " if info_string else ""
    
    # Set CUDA device
    torch.cuda.set_device(device)
    
    metadata = {
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device=device),
        "num_trials": num_correct_trials,
        "trials_passed": 0,
        "trials_failed": 0,
        "max_difference": [],
        "avg_difference": []
    }
    
    if verbose:
        print(f"{info_prefix}[Correctness] Starting correctness check on device: {device}")
        print(f"{info_prefix}[Correctness] Running {num_correct_trials} trials")
    
    # Load original model
    context_original = {}
    if verbose:
        print(f"{info_prefix}[Correctness] Loading original model...")
        
    try:
        Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
            original_model_src, context_original, timeout=timeout, info_string=info_string
        )
        if Model is None:
            return False, "Failed to load original model", metadata
            
        # Initialize original model
        set_seed(seed_num)
        init_inputs = get_init_inputs()
        init_inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
        ]
        
        with torch.no_grad():
            set_seed(seed_num)
            original_model = Model(*init_inputs).to(device)
            
    except Exception as e:
        return False, f"Failed to initialize original model: {e}", metadata
    
    # Load custom model
    context_custom = {}
    if verbose:
        print(f"{info_prefix}[Correctness] Loading custom model...")
        
    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Enable device-side assertions
        ModelNew = load_custom_model(custom_model_src, context_custom, build_dir, timeout=timeout, info_string=info_string)
        if ModelNew is None:
            return False, "Failed to load custom model", metadata
            
        # Initialize custom model
        with torch.no_grad():
            set_seed(seed_num)
            custom_model = ModelNew(*init_inputs).to(device)
            
        torch.cuda.synchronize(device=device)
        
    except Exception as e:
        return False, f"Failed to initialize custom model: {e}", metadata
    
    # Run correctness trials
    if verbose:
        print(f"{info_prefix}[Correctness] Running {num_correct_trials} correctness trials...")
    
    # Generate trial seeds deterministically
    torch.manual_seed(seed_num)
    trial_seeds = [torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)]
    
    pass_count = 0
    
    with torch.no_grad():
        for trial in range(num_correct_trials):
            trial_seed = trial_seeds[trial]
            
            # if verbose:
            #     print(f"{info_prefix}[Correctness {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] Trial {trial + 1}/{num_correct_trials} (seed: {trial_seed})")
            
            try:
                # Generate inputs for this trial
                set_seed(trial_seed)
                inputs = get_inputs()
                inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]
                
                # Run original model
                set_seed(trial_seed)
                original_model.eval()
                original_output = original_model(*inputs)
                torch.cuda.synchronize(device=device)
                
                # Run custom model  
                set_seed(trial_seed)
                custom_model.eval()
                custom_output = custom_model(*inputs)
                torch.cuda.synchronize(device=device)
                
                # Check output shapes
                if original_output.shape != custom_output.shape:
                    error_msg = f"Shape mismatch to the original model"
                    metadata["trials_failed"] += 1
                    # if verbose:
                    #     print(f"{info_prefix}[Correctness] ❌ {error_msg}")
                    return False, error_msg, metadata
                
                # Check output values
                if not torch.allclose(original_output, custom_output, atol=1e-02, rtol=1e-02):
                    max_diff = torch.max(torch.abs(original_output - custom_output)).item()
                    avg_diff = torch.mean(torch.abs(original_output - custom_output)).item()
                    
                    metadata["max_difference"].append(f"{max_diff:.6f}")
                    metadata["avg_difference"].append(f"{avg_diff:.6f}")
                    metadata["trials_failed"] += 1
                    
                    error_msg = f"Value mismatch to the original model"
                    # if verbose:
                    #     print(f"{info_prefix}[Correctness] ❌ {error_msg}")
                    return False, error_msg, metadata
                else:
                    # Trial passed
                    pass_count += 1
                    metadata["trials_passed"] += 1
                    # if verbose:
                    #     print(f"{info_prefix}[Correctness] ✅ Trial {trial + 1} passed")
                        
            except Exception as e:
                metadata["trials_failed"] += 1
                error_msg = f"Runtime error in trial {trial + 1}: {e}"
                if verbose:
                    print(f"{info_prefix}[Correctness {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] ❌ {error_msg}")
                return False, error_msg, metadata
    
    # Final validation
    if pass_count == num_correct_trials:
        if verbose:
            print(f"{info_prefix}[Correctness {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] ✅ All {pass_count}/{num_correct_trials} trials passed!")
        
        # Cleanup
        graceful_eval_cleanup(context_original, device)
        graceful_eval_cleanup(context_custom, device)
        
        return True, "", metadata
    else:
        error_msg = f"Only {pass_count}/{num_correct_trials} trials passed"
        if verbose:
            print(f"{info_prefix}[Correctness {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] ❌ {error_msg}")
        return False, error_msg, metadata


def eval_kernel_against_ref(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_perf_trials: int = 10,
    verbose: bool = False,
    build_dir: os.PathLike = None,
    device: torch.device = None, # have to run on GPU
    info_string: str = ""
) -> tuple[float | None, str]:
    """
    Evaluate the custom kernel against the original model

    num_correct_trials: number of trials to initialize different random inputs; correctness pass only if all trials pass
    num_perf_trials: run the evalutation many times to take the average
    device: GPU (cuda) device to run the evalutation on
    
    Returns:
        tuple[float | None, str]: (score, message) where score is original_model_time / custom_model_time 
                                  (higher is better, >1.0 means speedup), or None if failed
    """
    # TODO: check device is busy
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"
    torch.set_printoptions(
        precision=4,  # Decimal places
        threshold=10,  # Total number of elements before truncating
        edgeitems=3,  # Number of elem xents at beginning and end of dimensions
        linewidth=80,  # Maximum width before wrapping
    )

    custom_contain_new_streams = False
    if custom_model_src.find("cuda.Stream")!=-1:
        custom_contain_new_streams = True

    # Define beijing_tz at the beginning of the function
    beijing_tz = timezone(timedelta(hours=8))

    # Format info_string for consistent display
    info_prefix = f"[{info_string}] " if info_string else ""

    # set CUDA device
    torch.cuda.set_device(device)

    context = {}

    if verbose:
        print(f"{info_prefix}[Eval] Start Evalulation! on device: {device}")
        print(f"{info_prefix}[Eval] Loading Original Model")

    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        original_model_src, context, info_string=info_string
    )
    set_seed(seed_num)  # set seed for reproducible input
    init_inputs = get_init_inputs()
    init_inputs = [
        x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
    ]

    with torch.no_grad():
        set_seed(seed_num)  # set seed for reproducible weights
        original_model = Model(*init_inputs)
        assert hasattr(original_model, "forward")
        if verbose:
            print(f"{info_prefix}[Eval] Original Model Loaded")
    if verbose:
        print(f"{info_prefix}[Eval] Loading and Compiling New Model with Custom CUDA Kernel")

    metadata = {}  # for storing result metadata
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)  # for debugging

    # this is where compilation happens
    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        # add hash for later to distinguish between multi-turn kernels
        ModelNew = load_custom_model(custom_model_src, context, build_dir, info_string=info_string)
        
        # Debug: Check what load_custom_model returned
        if verbose:
            print(f"{info_prefix}[DEBUG] load_custom_model returned: {ModelNew} (type: {type(ModelNew)})")
        
        # Validate ModelNew before proceeding
        if ModelNew is None:
            print(f"{info_prefix}ERROR: load_custom_model returned None - check the model source code")
            print(f"{info_prefix}The custom model source must define: ModelNew = YourModelClass")
            return None, "ModelNew is None"
            
        torch.cuda.synchronize(device=device)  # not sure if this is too much
    except Exception as e:
        print(
            f"{info_prefix}Failed to compile custom CUDA kernel: Record as compilation failure. \nError: {e}"
        )
        return None, "Failed to compile custom CUDA kernel"

    # at this point we passed compilation
    try:
        with torch.no_grad():
            set_seed(seed_num)  # set seed for reproducible weights
            custom_model = ModelNew(*init_inputs)
            assert hasattr(custom_model, "forward")
            torch.cuda.synchronize(device=device)
        if verbose:
            print(f"{info_prefix}[Eval] New Model with Custom CUDA Kernel Loaded")
    except RuntimeError as e:
        print(
            f"{info_prefix}Failed to load custom CUDA kernel; Compiled but not able to run, count as runtime error. \nError: {e}"
        )
        return None, "Failed to load custom CUDA kernel with New Model"

    # Handle case where num_correct_trials is 0 (skip correctness check)
        
    if verbose:
        print(f"{info_prefix}[Eval] Measuring Performance")

    # Move models to the correct device for performance measurement
    original_model = original_model.to(device)
    custom_model = custom_model.to(device)

    original_times = []
    custom_times = []
    
    # Warmup
    for _ in range(3):
        inputs = get_inputs()
        inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]
        original_model(*inputs)
        custom_model(*inputs)
        torch.cuda.synchronize(device=device)
    
    if verbose:
        print(f"{info_prefix}[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, trials: {num_perf_trials}")

    t1 = time.time()
    with torch.no_grad():
        for trial in range(num_perf_trials):
            # Generate one random input for this trial - SAME input will be used for both models
            inputs = get_inputs()
            inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]            
            # Randomize execution order to eliminate systematic bias
            run_original_first = random.choice([True, False])
            
            # IMPORTANT: Detect model streams to ensure accurate timing
            current_stream = torch.cuda.current_stream(device=device)
            
            # Comprehensive stream detection - find any CUDA streams the models use
            def find_model_streams(model):
                """Find all CUDA streams in a model, regardless of attribute names"""
                streams = []
                
                # Check all attributes of the model
                for attr_name in dir(model):
                    try:
                        attr_value = getattr(model, attr_name)
                        
                        # Check if it's a single CUDA stream
                        if isinstance(attr_value, torch.cuda.Stream):
                            streams.append(attr_value)
                        
                        # Check if it's a list/tuple of CUDA streams
                        elif isinstance(attr_value, (list, tuple)):
                            for item in attr_value:
                                if isinstance(item, torch.cuda.Stream):
                                    streams.append(item)
                        
                        # Check if it's a dict containing CUDA streams
                        elif isinstance(attr_value, dict):
                            for item in attr_value.values():
                                if isinstance(item, torch.cuda.Stream):
                                    streams.append(item)
                                    
                    except (AttributeError, RuntimeError):
                        # Some attributes might not be accessible or might raise errors
                        continue
                
                return streams
            
            # Find streams for both models
            custom_model_streams = find_model_streams(custom_model)
            # Use current stream for timing, but track all model streams for synchronization
            # This ensures we capture all work regardless of which streams the model uses
            original_model_stream = current_stream
            custom_model_stream = current_stream
            
            # Debug info for stream detection
            if verbose and custom_model_streams:
                print(f"{info_prefix}[Stream Detection] Found {len(custom_model_streams)} CUDA streams in custom model")
            
            if run_original_first:
                # Time original model first
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                with torch.cuda.stream(original_model_stream):
                    start_event.record(original_model_stream)
                    original_model(*inputs)
                    
                    # Wait for all model streams to complete before recording end event                    
                    end_event.record(original_model_stream)
                
                torch.cuda.synchronize(device=device)
                original_time = start_event.elapsed_time(end_event)
                
                # Time custom model second
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                with torch.cuda.stream(custom_model_stream):
                    start_event.record(custom_model_stream)
                    custom_model(*inputs)
                    
                    # Wait for all model streams to complete before recording end event
                    if custom_contain_new_streams:
                        for stream in custom_model_streams:
                            custom_model_stream.wait_stream(stream)
                    
                    end_event.record(custom_model_stream)
                
                torch.cuda.synchronize(device=device)
                custom_time = start_event.elapsed_time(end_event)
            else:
                # Time custom model first
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                with torch.cuda.stream(custom_model_stream):
                    start_event.record(custom_model_stream)
                    custom_model(*inputs)
                    
                    # Wait for all model streams to complete before recording end event
                    if custom_contain_new_streams:
                        for stream in custom_model_streams:
                            custom_model_stream.wait_stream(stream)
                    
                    end_event.record(custom_model_stream)
                
                torch.cuda.synchronize(device=device)
                custom_time = start_event.elapsed_time(end_event)
                
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                with torch.cuda.stream(original_model_stream):
                    start_event.record(original_model_stream)
                    original_model(*inputs)
                    
                    # Wait for all model streams to complete before recording end event                    
                    end_event.record(original_model_stream)
                
                torch.cuda.synchronize(device=device)
                original_time = start_event.elapsed_time(end_event)
            
            original_times.append(original_time)
            custom_times.append(custom_time)
    t2 = time.time()

    # Calculate averages and score
    avg_original_time = sum(original_times) / len(original_times)
    avg_custom_time = sum(custom_times) / len(custom_times)
    score = avg_original_time / avg_custom_time
    total_elapsed_time = (sum(original_times) + sum(custom_times)) / 1000.0  # Convert from milliseconds to seconds
    if verbose:
        print(f"{info_prefix}[Results {datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')}] Original avg: {avg_original_time:.3f}ms")
        print(f"{info_prefix}[Results {datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')}] Custom avg: {avg_custom_time:.3f}ms") 
        print(f"{info_prefix}[Results {datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')}] Score (original/custom): {score:.3f}")
        if score > 1.0:
            speedup = score
            print(f"{info_prefix}[Results {datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')}] Speedup: {speedup:.2f}x faster")
        elif score < 1.0:
            slowdown = 1.0 / score
            print(f"{info_prefix}[Results {datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')}] Slowdown: {slowdown:.2f}x slower")
        else:
            print(f"{info_prefix}[Results] Same performance")

    graceful_eval_cleanup(context, device)
    return score, total_elapsed_time, avg_original_time, avg_custom_time, "Success"


################################################################################
# Performance Eval
################################################################################


def fetch_baseline_time(
    level_name: str, problem_id: int, dataset: List[str], baseline_time_filepath: str
) -> Dict:
    """
    Fetch the baseline time from the time
    """
    if not os.path.exists(baseline_time_filepath):
        raise FileNotFoundError(
            f"Baseline time file not found at {baseline_time_filepath}"
        )

    with open(baseline_time_filepath, "r") as f:
        baseline_json = json.load(f)

    problem_name = dataset[problem_id].split("/")[-1]
    baseline_time = baseline_json[level_name].get(problem_name, None)
    return baseline_time


def get_timing_stats(elapsed_times: List[float], device: Optional[torch.device] = None) -> Dict:
    """Get timing statistics from a list of elapsed times.

    Args:
        elapsed_times: List of elapsed times in milliseconds
        device: CUDA device, record device info
    Returns:
        Dict containing mean, std, min, max and num_trials
        all timing are in ms
    """

    stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }

    if device:
        stats["hardware"] = torch.cuda.get_device_name(device=device)
        stats["device"] = str(device)  # for debugging

    return stats


def get_available_gpus():
    """Get list of available GPU device IDs"""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))




def eval_pipeline(
    original_model_src: str,
    custom_model_src: str,
    num_correct_trials: int,
    num_perf_trials: int,
    global_n_trials: int,
    gpu_index: int,
    verbose: bool = False,
    log_path: str = None,
    max_time: float = None,
    use_process_isolation: bool = False,
    info_string = "",
    valid_bar = 0.15
):
    pst_tz = timezone(timedelta(hours=-8))
    
    # Format info_string for consistent display
    info_prefix = f"[{info_string}] " if info_string else ""

    print(f"{info_prefix}[Score {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] median_comparison_pipeline start")
    if log_path is not None:
        log_dir = os.path.dirname(log_path)
        os.makedirs(log_dir, exist_ok=True)
    print(f"{info_prefix}[Score {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] Writing log to {log_path}")
    current_time = datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')

    with open(log_path, "w") as write_log:
        print(f"in log_path open and write {log_path}")
        write_log.write(json.dumps({"info_string": info_string, "start_time": current_time, "code": custom_model_src}) + "\n")
        # write_log.write(json.dumps({"info_string": info_string, "start_time": current_time, "custom_model_src": custom_model_src}) + "\n")
        write_log.flush()

    # step 1: check whether the model can be executed and compiled
    print(f"{info_prefix}[Score {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] step 1: check whether the model can be executed and compiled")
    context = {}
    success_original, error_msg, execution_time = execute_model_with_timeout(
        model_src=original_model_src,
        context=context,
        timeout=30.0,  # 30 seconds should be enough
        use_process_isolation=use_process_isolation,
        info_string=info_string
    )
    if not success_original:
        log_dict_ = {
            "info_string": info_string,
            "error_msg": f"Original model compilation failed: {error_msg}",
            "error": True,
            "done": True
        }
        with open(log_path, "a") as write_log:
            write_log.write(json.dumps(log_dict_) + "\n")
            write_log.flush()
        return None, f"Original model compilation failed: {error_msg}"
    
    success_custom, error_msg, execution_time = execute_model_with_timeout(
        model_src=custom_model_src,
        context={},  # Use fresh context for custom model
        timeout=100,  # Give enough time for CUDA compilation with minimum 30s
        use_process_isolation=use_process_isolation,
        info_string=info_string
    )
    if not success_custom:
        log_dict_ = {
            "info_string": info_string,
            "error_msg": "fail to compile or execute",
            "error": True,
            "done": True
        }
        with open(log_path, "a") as write_log:
            write_log.write(json.dumps(log_dict_) + "\n")
            write_log.flush()
        return None, "Custom model compilation failed"
    else:
        log_dict_ = {
            "info_string": info_string,
            "info": "stage1:Compile Success",
            "time": datetime.now(timezone(timedelta(hours=-8))).strftime('%Y-%m-%d %H:%M:%S'),
            "error": False,
            "done": False
        }
        with open(log_path, "a") as write_log:
            write_log.write(json.dumps(log_dict_) + "\n")
            write_log.flush()

    # step 2: correctness check
    device = torch.device(f'cuda:{gpu_index}')
    time1 = time.time()
    print(f"{info_prefix}[Score {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] step 2: correctness check")
    time1 = time.time()
    correctness_passed, error_msg, correctness_metadata = check_kernel_correctness(
        original_model_src=original_model_src,
        custom_model_src=custom_model_src,
        num_correct_trials=num_correct_trials,
        verbose=verbose,
        device=device,
        info_string=info_string
    )
    time2 = time.time()
    if not correctness_passed:
        log_dict_ = {
            "info_string": info_string,
            "error_msg": error_msg,
            "error": True,
            "done": True
        }
        with open(log_path, "a") as write_log:
            write_log.write(json.dumps(log_dict_) + "\n")
            write_log.flush()
        return None, error_msg
    else:
        log_dict_ = {
            "info_string": info_string,
            "info": "stage3:Correctness Check Success",
            "time": datetime.now(timezone(timedelta(hours=-8))).strftime('%Y-%m-%d %H:%M:%S'),
            "error": False,
            "done": False,
            "duration": time2 - time1,
        }
        with open(log_path, "a") as write_log:
            write_log.write(json.dumps(log_dict_) + "\n")
            write_log.flush()
    
    log_dict_ = {
        "info_string": info_string,
        "info": "stage4:Performance Evaluation",
        "time": datetime.now(timezone(timedelta(hours=-8))).strftime('%Y-%m-%d %H:%M:%S'),
        "error": False,
        "done": False
    }
    with open(log_path, "a") as write_log:
        write_log.write(json.dumps(log_dict_) + "\n")
        write_log.flush()
    scores = []
    list_gpu_execution_time = []
    # Run global_n_trials sequential evaluations
    start_time = time.time()
    print(f"{info_prefix}[Score {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] step 3: performance evaluation")
    for trial in range(global_n_trials):
        print(f"{info_prefix}[Score {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] step 3: performance evaluation, trial {trial + 1}/{global_n_trials}")
        # Run single evaluation
        time1 = time.time()
        score, gpu_execution_time, avg_original_time, avg_custom_time, error_msg = eval_kernel_against_ref(
            original_model_src=original_model_src,
            custom_model_src=custom_model_src,
            seed_num=42 + trial,  # Different seed for each trial
            num_perf_trials=num_perf_trials,
            verbose=False,  # Keep individual trials quiet unless overall verbose
            build_dir=None,
            device=device,
            info_string=info_string
        )
        list_gpu_execution_time.append(gpu_execution_time)
        if score is None:
            error_msg = f"fail to inference"
            log_dict_ = {
                "info_string": info_string,
                "trial": trial,
                "gpu_index": gpu_index,
                "score": score,
                "error_msg": error_msg,
                "error": True,
                "done": True
            }
            with open(log_path, "a") as write_log:
                write_log.write(json.dumps(log_dict_) + "\n")
                write_log.flush()
            return None, error_msg
        time2 = time.time()
        log_dict_ = {
            "info_string": info_string,
            "n_trial": num_perf_trials,
            "trial": trial,
            "gpu_index": gpu_index,
            "score": score,
            "time": datetime.now(timezone(timedelta(hours=-8))).strftime('%Y-%m-%d %H:%M:%S'),
            "gpu_execution_time": gpu_execution_time,
            "ave_gpu_execution_time": gpu_execution_time / num_perf_trials,
            "avg_original_time": avg_original_time,
            "avg_custom_time": avg_custom_time,
            "done": False,
            "duration": time2 - time1,
            "error": False
        }
        with open(log_path, "a") as write_log:
            write_log.write(json.dumps(log_dict_) + "\n")
            write_log.flush()
        scores.append(score)
        
        print(f"{info_prefix}[Score {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] Trial {trial + 1}: {score:.4f} at gpu {gpu_index}")
        
    if len(scores) == 0:
        print(f"{info_prefix}[Score {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] ❌ No trials completed successfully")
        log_dict_empty = {
            "info_string": info_string,
            "error": True,
            "error_msg": "No trials completed successfully",
            "completed_trials": 0,
            "done": True
        }
        with open(log_path, "a") as write_log:
            write_log.write(json.dumps(log_dict_empty) + "\n")
            write_log.flush()
        return None, "No trials completed successfully"
    
    # Calculate median score and apply custom rounding
    # raw_median = float(np.median(scores))
    raw_mean = float(np.mean(scores))
    mean_score = round(raw_mean, 3)

    median_score = float(np.median(scores))

    std = float(np.std(scores))
    
    # Round all scores in the list to 4 decimal places for consistency
    rounded_scores = [round(score, 4) for score in scores]
    
    # Record final elapsed time
    total_elapsed_time = time.time() - start_time
    n_all_trials = num_perf_trials*global_n_trials
    log_dict_ = {
        "info_string": info_string,
        "median_score": median_score,
        "mean_score": mean_score,
        "rounded_scores": rounded_scores,
        "scores_sorted": sorted(scores),
        "completed_trials": len(scores),
        "total_trials": global_n_trials,
        "n_all_trials_trials": n_all_trials,
        "total_elapsed_time": total_elapsed_time,
        "total_gpu_execution_time": sum(list_gpu_execution_time),
        "ave_gpu_execution_time": sum(list_gpu_execution_time)/n_all_trials, 
        "error": False,
        "done": True,
        "scores": [round(score, 4) for score in scores],
        "std": std
    }
    with open(log_path, "a") as write_log:
        write_log.write(json.dumps(log_dict_) + "\n")
        write_log.flush()
    
    if verbose:
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        
        trials_completed = len(scores)
        print(f"\n{info_prefix}[Score] 📊 Results from {trials_completed}/{global_n_trials} trials:")
        print(f"{info_prefix}  - Total time: {total_elapsed_time:.2f}s")
        if max_time is not None and total_elapsed_time >= max_time:
            print(f"{info_prefix}  - Status: TIMEOUT (reached {max_time}s limit)")
        print(f"{info_prefix}  - Scores: {[f'{s:.4f}' for s in scores]}")
        print(f"{info_prefix}  - Raw Median: {raw_median:.4f}")
        print(f"{info_prefix}  - Final Median: {median_score:.2f}")
        print(f"{info_prefix}  - Mean:   {mean_score:.4f}")
        print(f"{info_prefix}  - Std:    {std_score:.4f}")
        print(f"{info_prefix}  - Range:  [{min_score:.4f}, {max_score:.4f}]")
        
        # Stability assessment
        cv = (std_score / mean_score) * 100 if mean_score > 0 else 0
        print(f"{info_prefix}  - CV:     {cv:.2f}% {'(stable)' if cv < 1.0 else '(variable)' if cv < 5.0 else '(unstable)'}")
    
    return median_score, rounded_scores


def load_cuda_file(PATH_TO_CUDA_FILE):
    if not os.path.exists(PATH_TO_CUDA_FILE):
        raise Exception(f"{PATH_TO_CUDA_FILE} not found")
    with open(PATH_TO_CUDA_FILE, "r") as f:
        ref_cuda_file = json.load(f)
    cuda_dict_ = {}
    for level, level_items in ref_cuda_file.items():
        cuda_dict_[int(level)] = {}
        for item in level_items:
            task_id = int(item["task_id"])
            ref_code = item["ref_code"]
            custom_code = item["custom_code"]
            cuda_dict_[int(level)][task_id] = (ref_code, custom_code)
    return cuda_dict_


if __name__ == "__main__":

    PATH_TO_CUDA_FILE = "YOUR_FOLDER/rtx_3090.json"
    cuda_data_folder = os.path.dirname(PATH_TO_CUDA_FILE)

    cuda_dict_ = load_cuda_file(PATH_TO_CUDA_FILE)
    level_id = 3
    task_id = 42
    ref_code, custom_code = cuda_dict_[level_id][task_id]
    print(custom_code)
    output_path = os.path.join(cuda_data_folder, f"{level_id}_{task_id}_eval.json")
    print(f"eval results output to {output_path}")
    eval_pipeline(
        original_model_src=ref_code,
        custom_model_src=custom_code,
        num_correct_trials=100,
        num_perf_trials=100,
        global_n_trials=7,
        gpu_index=0,
        verbose=False,
        log_path=output_path,
        max_time=1800
    )
    print(f"log_path: {output_path}")
    print(f"log_path: {output_path}")
