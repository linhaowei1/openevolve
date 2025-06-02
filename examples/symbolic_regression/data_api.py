import os
import yaml
import numpy as np
import multiprocessing
import importlib.util
from typing import Dict, List, Tuple, Optional, Any
import traceback # Retained for detailed error messages in evaluator and main processing

from bench.datamodules import get_datamodule


def extract_problem_data_from_initialized_dataset(
    initialized_dataset, problem_id: int
) -> Dict[str, Any]:
    """
    Extract data for a specific problem from an initialized dataset.

    Args:
        initialized_dataset: Pre-initialized and setup dataset object
        problem_id: Index of the problem to extract

    Returns:
        Dictionary containing problem data including train/test samples, symbols, and metadata
    """
    problem = initialized_dataset.problems[problem_id]
    gt_eq = problem.gt_equation
    samples = problem.samples

    data = {
        "train": samples["train"],
        "test": samples["test"],
        "ood_test": samples.get("ood_test", None),
        "symbols": gt_eq.symbols,
        "symbol_descs": gt_eq.symbol_descs,
        "symbol_properties": gt_eq.symbol_properties,
        "expression": gt_eq.expression,
        "dataset_identifier": problem.dataset_identifier,
        "equation_idx": problem.equation_idx,
    }
    return data


def create_program(problem: Dict[str, Any]) -> str:
    """
    Create a Python script template for an evolvable symbolic regression program.
    The LLM evolves key functions for equation definition, parameter initialization,
    and the optimization strategy.

    Args:
        problem: Dictionary containing problem data

    Returns:
        Path to the created program file (e.g., initial_program.py)
    """
    problem_dir = f'problems/{problem["dataset_identifier"]}/{problem["equation_idx"]}'

    symbols = problem["symbols"]
    properties = problem["symbol_properties"]
    descs = problem["symbol_descs"]

    input_vars = []
    input_vars_descs = []
    output_var = None
    output_var_desc = "N/A"

    for i, prop in enumerate(properties):
        if prop == "V":
            input_vars.append(symbols[i])
            input_vars_descs.append(descs[i])
        elif prop == "O":
            output_var = symbols[i]
            output_var_desc = descs[i]

    if not output_var:
        raise ValueError("No output variable ('O') found in symbol_properties.")

    x_mapping_comments = ["# Input variable mapping for x (columns of the input matrix):"]
    if not input_vars:
        x_mapping_comments.append("#   No input variables (x will be an (n_samples, 0) matrix).")
    else:
        for i, var_name in enumerate(input_vars):
            x_mapping_comments.append(f"#   x[:, {i}]: {var_name} ({input_vars_descs[i]})")
    x_mapping_str = "\n".join(x_mapping_comments)

    num_features = len(input_vars)
    model_num_params = 10  # Guideline for the number of parameters in symbolic_equation

    input_vars_desc_list = [f"{v} ({input_vars_descs[i]})" for i, v in enumerate(input_vars)]
    input_vars_desc_str = ", ".join(input_vars_desc_list) if input_vars else "None"

    initial_symbolic_equation_body_lines = []
    initial_symbolic_equation_body_lines.append(f"    # LLM: Replace this with your equation using x and params.")
    initial_symbolic_equation_body_lines.append(f"    # x is (n_samples, {num_features}) and params is ({model_num_params},).")
    initial_symbolic_equation_body_lines.append(f"    # Ensure output is (n_samples,).")
    if model_num_params > 0:
        initial_symbolic_equation_body_lines.append(f"    if x.shape[1] > 0: # If there are input features")
        initial_symbolic_equation_body_lines.append(f"        # Example: use first feature and first param. REPLACE THIS!")
        initial_symbolic_equation_body_lines.append(f"        return x[:, 0] * params[0]")
        initial_symbolic_equation_body_lines.append(f"    else: # No input features, return a constant based on first param. REPLACE THIS!")
        initial_symbolic_equation_body_lines.append(f"        return np.full(x.shape[0], params[0])")
    else: # model_num_params == 0
        initial_symbolic_equation_body_lines.append(f"    return np.zeros(x.shape[0]) # No params to use, replace if needed.")
    initial_symbolic_equation_body = "\n".join(initial_symbolic_equation_body_lines)

    program_content = f'''"""
Evolvable program for symbolic regression.
The LLM's task is to evolve the `symbolic_equation`, `initialize_parameters`,
and `optimize_model` functions within the EVOLVE-BLOCK.
The objective for optimization is implicitly Mean Squared Error (MSE).

Target output variable: {output_var} ({output_var_desc})
Input variables (columns of x): {input_vars_desc_str}
Number of input features for symbolic_equation: {num_features}
Guideline for number of parameters (MODEL_NUM_PARAMS): {model_num_params}
"""
import numpy as np
# Note: scipy.optimize.minimize is imported locally in _default_bfgs_optimize
# to allow LLM to remove it if not using that default strategy.

{x_mapping_str}

# EVOLVE-BLOCK-START
# LLM: You have the freedom to design the following:
# 1. `symbolic_equation(x, params)`: The core mathematical model.
# 2. `initialize_parameters()`: How the initial guess for `params` is generated.
#    It should return a NumPy array of shape (MODEL_NUM_PARAMS,).
# 3. `optimize_model(X_train, y_train, initial_params, symbolic_eq_func)`:
#    The strategy to optimize the parameters of `symbolic_eq_func`.
#    It should return (optimized_callable, optimized_params, train_mse, success_flag).
#    A default BFGS-based implementation `_default_bfgs_optimize` is provided.
#    You can replace `_default_bfgs_optimize` or how it's used.

MODEL_NUM_PARAMS = {model_num_params} # Guideline for params array length

def symbolic_equation(x, params):
    """
    The symbolic equation to be discovered/evolved by the LLM.
    It must take input data `x` and parameters `params` and return predictions.

    Args:
        x (np.ndarray): Input data, shape (n_samples, {num_features}).
                        Column order: {', '.join(input_vars) if input_vars else "None"}.
        params (np.ndarray): Parameters, shape (MODEL_NUM_PARAMS,).

    Returns:
        np.ndarray: Predictions, shape (n_samples,).
    """
{initial_symbolic_equation_body}

def initialize_parameters():
    """
    LLM: Define how initial parameters for `symbolic_equation` are generated.
    Should return a np.ndarray of shape (MODEL_NUM_PARAMS,).
    """
    # Default initialization: random values between -1 and 1.
    # LLM can replace this with a more sophisticated strategy.
    return np.random.uniform(-1, 1, MODEL_NUM_PARAMS)

def _objective_mse(params_to_optimize, x_data, y_data, eq_func):
    """
    Internal objective (MSE) for an optimization algorithm like BFGS.
    This is a helper if you use an optimizer that needs an objective function.
    It's kept minimal; complex error handling can be added by LLM if needed.
    """
    try:
        predictions = eq_func(x_data, params_to_optimize)
        # Basic checks for validity of predictions
        if not isinstance(predictions, np.ndarray) or \
           predictions.shape != y_data.shape or \
           np.any(np.isnan(predictions)) or \
           np.any(np.isinf(predictions)):
            return float('inf') # Indicates an invalid prediction state
        mse = np.mean((predictions - y_data)**2)
        return mse
    except Exception:
        # Catch any other errors during equation evaluation
        return float('inf')

def _default_bfgs_optimize(X_train, y_train, initial_params, symbolic_eq_func_to_optimize):
    """
    A default optimization strategy using BFGS.
    LLM: You can replace this function or how it's called in `optimize_model`.
    """
    from scipy.optimize import minimize # Import locally

    # Simplified options for BFGS
    options = {{'maxiter': 500, 'gtol': 1e-6, 'disp': False}}

    result = minimize(
        _objective_mse,
        initial_params,
        args=(X_train, y_train, symbolic_eq_func_to_optimize),
        method='BFGS',
        options=options
    )

    optimized_params = result.x
    final_mse = result.fun if np.isfinite(result.fun) else float('inf')
    
    # Ensure params have the correct shape, even if minimize had issues or
    # if initial_params was not MODEL_NUM_PARAMS.
    # This is a basic safeguard.
    if optimized_params.shape[0] != MODEL_NUM_PARAMS:
        corrected_params = np.full(MODEL_NUM_PARAMS, np.nan) # Start with NaNs or zeros
        p_len = min(optimized_params.shape[0], MODEL_NUM_PARAMS)
        corrected_params[:p_len] = optimized_params[:p_len]
        # If some params are missing (e.g. optimizer reduced dim), fill with from initial or zeros
        if optimized_params.shape[0] < MODEL_NUM_PARAMS:
             init_p_len = min(initial_params.shape[0], MODEL_NUM_PARAMS)
             # try to fill remaining from initial_params if they existed
             if initial_params.shape[0] == MODEL_NUM_PARAMS:
                  corrected_params[optimized_params.shape[0]:] = initial_params[optimized_params.shape[0]:]
             else: # fallback to zeros for the rest
                  corrected_params[optimized_params.shape[0]:] = 0.0


        optimized_params = corrected_params
        # Re-evaluate MSE if params were corrected, to be sure
        final_mse = _objective_mse(optimized_params, X_train, y_train, symbolic_eq_func_to_optimize)


    optimized_callable = lambda x_eval: symbolic_eq_func_to_optimize(x_eval, optimized_params)
    return optimized_callable, optimized_params, final_mse, result.success

def optimize_model(X_train, y_train, initial_params_from_initializer, symbolic_eq_func):
    """
    LLM: Define the main optimization logic here.
    You can use `_default_bfgs_optimize` (as done by default) or implement your own.
    `initial_params_from_initializer` comes from your `initialize_parameters` function.
    `symbolic_eq_func` is your `symbolic_equation` function.

    Returns:
        tuple: (optimized_callable, optimized_params, train_mse, optimization_success_flag)
               - optimized_callable: Function taking X, returns predictions with optimized_params.
               - optimized_params: 1D NumPy array of optimized parameters.
               - train_mse: MSE on training data with optimized_params.
               - optimization_success_flag: Boolean indicating optimization success.
    """
    # By default, use the provided BFGS optimizer.
    # LLM can change this to use a different optimizer, different settings,
    # or even a sequence of optimization steps.
    return _default_bfgs_optimize(X_train, y_train, initial_params_from_initializer, symbolic_eq_func)

# EVOLVE-BLOCK-END

# This part remains fixed (not evolved by the LLM).
# The evaluator will call this `run_search` function.
def run_search(X_train_data, y_train_data):
    """
    This function is called by the evaluator. It executes the symbolic regression
    process defined within the EVOLVE-BLOCK. It calls the LLM-defined
    `initialize_parameters` and `optimize_model` (which uses `symbolic_equation`).

    Args:
        X_train_data (np.ndarray): Training input data of shape (n_samples, {num_features}).
        y_train_data (np.ndarray): Training target data of shape (n_samples,).

    Returns:
        tuple: (callable_optimized_model, optimized_parameters_array, final_train_mse, optimization_succeeded_flag)
    """
    optimized_model_callable = None
    optimized_params_to_return = np.full(MODEL_NUM_PARAMS, np.nan) # Default to NaN array
    mse_to_return = float('inf')
    success_flag = False

    try:
        # Step 1: Get initial parameters from the (evolved) initializer
        current_initial_params = initialize_parameters()

        # Ensure initial_params conform to MODEL_NUM_PARAMS for broader compatibility,
        # especially if the LLM doesn't perfectly align initialize_parameters output
        # with MODEL_NUM_PARAMS but symbolic_equation or default optimizer expects it.
        if not isinstance(current_initial_params, np.ndarray) or current_initial_params.ndim != 1:
            # print(f"Warning: initialize_parameters did not return a 1D numpy array. Defaulting.") # Minimal
            current_initial_params = np.random.uniform(-1, 1, MODEL_NUM_PARAMS) # Fallback
        
        if current_initial_params.shape[0] != MODEL_NUM_PARAMS:
            # print(f"Warning: Initial params shape {{current_initial_params.shape}} not {{MODEL_NUM_PARAMS}}. Adjusting.") # Minimal
            new_initial_params = np.full(MODEL_NUM_PARAMS, 0.0) # Create array of target size
            len_to_copy = min(current_initial_params.shape[0], MODEL_NUM_PARAMS)
            new_initial_params[:len_to_copy] = current_initial_params[:len_to_copy]
            current_initial_params = new_initial_params
        
        optimized_params_to_return = np.copy(current_initial_params) # Initialize fallback params

        # Step 2: Perform optimization using the (evolved) optimization strategy
        # Pass the global symbolic_equation function to it.
        opt_callable, opt_params, opt_mse, opt_success = \
            optimize_model(X_train_data, y_train_data, current_initial_params, symbolic_equation)

        optimized_model_callable = opt_callable
        # Ensure returned params also conform, similar to initial_params logic.
        if isinstance(opt_params, np.ndarray) and opt_params.ndim == 1:
            if opt_params.shape[0] == MODEL_NUM_PARAMS:
                optimized_params_to_return = opt_params
            else:
                # print(f"Warning: Optimized params shape {{opt_params.shape}} not {{MODEL_NUM_PARAMS}}. Adjusting.") # Minimal
                new_opt_params = np.full(MODEL_NUM_PARAMS, np.nan)
                len_to_copy = min(opt_params.shape[0], MODEL_NUM_PARAMS)
                new_opt_params[:len_to_copy] = opt_params[:len_to_copy]
                # If params are missing, they remain NaN or could be filled from initial_params
                optimized_params_to_return = new_opt_params
        # else leave optimized_params_to_return as its initialized value (copy of current_initial_params or NaNs)

        mse_to_return = opt_mse if np.isfinite(opt_mse) else float('inf')
        success_flag = bool(opt_success)

    except Exception as e:
        # This is a critical fallback for run_search itself.
        # Keep print minimal or remove if desired, error will propagate via infinite MSE.
        # print(f"Critical error in run_search orchestration: {{e}}")
        mse_to_return = float('inf')
        success_flag = False
        # optimized_params_to_return is already pre-filled or NaN-filled

    # Final fallback for the callable if it's None from an error path
    if optimized_model_callable is None:
        final_params_for_dummy = optimized_params_to_return # Use the (potentially adjusted) params
        # If params are all NaN (e.g. from very early failure), use zeros for dummy prediction
        if np.all(np.isnan(final_params_for_dummy)):
             final_params_for_dummy = np.zeros(MODEL_NUM_PARAMS)
        
        # Create a dummy callable that uses these params, even if it means predicting a constant
        # or zeros, to ensure a callable is always returned.
        # It uses the `symbolic_equation` with these fallback params.
        # If symbolic_equation itself is broken, this might still error, caught by evaluator.
        try:
            optimized_model_callable = lambda x_eval: symbolic_equation(x_eval, final_params_for_dummy)
            # Test call to see if it raises an immediate error with dummy params
            _ = optimized_model_callable(np.zeros((1, {num_features}))) 
        except Exception:
            # If symbolic_equation is fundamentally broken, return ultra-dummy predicting zeros
            optimized_model_callable = lambda x_eval: np.zeros(x_eval.shape[0])


    return optimized_model_callable, optimized_params_to_return, mse_to_return, success_flag

'''
    os.makedirs(problem_dir, exist_ok=True)
    file_path = os.path.join(problem_dir, "initial_program.py")
    with open(file_path, "w") as f:
        f.write(program_content)
    return file_path


def create_evaluator(problem: Dict[str, Any]) -> str:
    """
    Create an evaluator script for the symbolic regression problem.
    The evaluator runs the evolved program. The evolved program now handles its own
    parameter initialization and optimization. The evaluator computes metrics.

    Args:
        problem: Dictionary containing problem data

    Returns:
        Path to the created evaluator file
    """
    problem_dir = f'problems/{problem["dataset_identifier"]}/{problem["equation_idx"]}'
    os.makedirs(problem_dir, exist_ok=True)

    symbols = problem["symbols"]
    properties = problem["symbol_properties"]
    train_samples = np.asarray(problem["train"])
    test_samples = np.asarray(problem["test"])
    ood_test_samples = problem["ood_test"]
    if ood_test_samples is not None:
        ood_test_samples = np.asarray(ood_test_samples)

    input_indices = [i for i, prop in enumerate(properties) if prop == "V"]
    output_indices = [i for i, prop in enumerate(properties) if prop == "O"]

    if not output_indices:
        raise ValueError("No output variable ('O') found in symbol_properties.")
    if len(output_indices) > 1:
        raise ValueError("Multiple output variables ('O') found. Evaluator supports single output.")
    output_index = output_indices[0]

    if not input_indices:
        X_train = np.empty((len(train_samples), 0))
        X_test = np.empty((len(test_samples), 0))
        X_ood_test = np.empty((len(ood_test_samples), 0)) if ood_test_samples is not None else None
    else:
        X_train = train_samples[:, input_indices]
        X_test = test_samples[:, input_indices]
        X_ood_test = ood_test_samples[:, input_indices] if ood_test_samples is not None else None

    y_train = train_samples[:, output_index]
    y_test = test_samples[:, output_index]
    y_ood_test = ood_test_samples[:, output_index] if ood_test_samples is not None else None

    num_input_features = len(input_indices)
    model_num_params_expected = 10 # Must match MODEL_NUM_PARAMS in initial_program.py

    base_data_path = "./"
    x_train_path = os.path.join(base_data_path, problem_dir, "X_train_for_eval.npy").replace("\\", "/")
    y_train_path = os.path.join(base_data_path, problem_dir, "y_train_for_eval.npy").replace("\\", "/")
    np.save(x_train_path, X_train)
    np.save(y_train_path, y_train)

    x_test_path = os.path.join(base_data_path, problem_dir, "X_test_for_eval.npy").replace("\\", "/")
    y_test_path = os.path.join(base_data_path, problem_dir, "y_test_for_eval.npy").replace("\\", "/")
    np.save(x_test_path, X_test)
    np.save(y_test_path, y_test)

    x_ood_test_path_str = ""
    y_ood_test_path_str = ""
    if X_ood_test is not None and y_ood_test is not None:
        x_ood_test_path_str = os.path.join(base_data_path, problem_dir, "X_ood_test_for_eval.npy").replace("\\", "/")
        y_ood_test_path_str = os.path.join(base_data_path, problem_dir, "y_ood_test_for_eval.npy").replace("\\", "/")
        np.save(x_ood_test_path_str, X_ood_test)
        np.save(y_ood_test_path_str, y_ood_test)

    evaluator_script_content = f'''"""
Evaluator for a symbolic regression model.
It runs an evolved program (specified by `program_path`), which is expected
to perform its own parameter initialization and optimization using training data.
The evaluator then assesses the optimized model.
"""
import os
import sys
import time
import traceback
import importlib.util
import numpy as np
import concurrent.futures

# Expected number of input features for the model's symbolic_equation
NUM_INPUT_FEATURES_EXPECTED = {num_input_features}
# Expected number of parameters for the model (MODEL_NUM_PARAMS in evolved program)
MODEL_NUM_PARAMS_EXPECTED = {model_num_params_expected}

# Paths to data
X_TRAIN_EVAL_PATH = r'{x_train_path}'
Y_TRAIN_EVAL_PATH = r'{y_train_path}'
X_TEST_EVAL_PATH = r'{x_test_path}'
Y_TEST_EVAL_PATH = r'{y_test_path}'
X_OOD_TEST_EVAL_PATH = r'{x_ood_test_path_str}'
Y_OOD_TEST_EVAL_PATH = r'{y_ood_test_path_str}'

EVOLVED_PROGRAM_RUN_SEARCH_TIMEOUT_SECONDS = 600
PREDICTION_TIMEOUT_SECONDS = 60


def run_with_timeout(func, args=(), kwargs={{}}, timeout_seconds=5):
    """Execute a function with a timeout."""
    if timeout_seconds is None or timeout_seconds <= 0:
        return func(*args, **kwargs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            func_name = getattr(func, '__name__', 'Unnamed function')
            raise TimeoutError(f"Function {{func_name}} timed out after {{timeout_seconds}} seconds")
        except Exception as e:
            raise e


def filter_and_convert_metrics(current_metrics_dict):
    """Filter and convert metrics to appropriate types for final reporting."""
    filtered_dict = {{}}
    float_metric_keys = ['combined_score', 'negative_mse'] # ONLY THESE TWO ARE USED

    for key in float_metric_keys:
        if key in current_metrics_dict:
            value = current_metrics_dict[key]
            if value is None:
                continue
            if isinstance(value, (int, float, np.integer, np.floating, bool)):
                try:
                    filtered_dict[key] = float(value)
                except (ValueError, TypeError):
                    pass
    return filtered_dict


def evaluate(program_path):
    """
    Evaluate an evolved model program. The program's `run_search` method
    handles its own initialization and optimization.
    """
    metrics = {{
        'can_run': 0.0,
        'negative_mse': -float('inf'), # Initialize to very bad score
        'raw_mse_train': float('inf'), # MSE from program's own optimization
        # 'mse_train_score': -100.0, # This will be derived from negative_mse/raw_mse_train
        'raw_mse_test': float('inf'),
        'raw_mse_ood_test': float('inf'),
        'num_params': MODEL_NUM_PARAMS_EXPECTED, # Based on expectation
        'combined_score': -100.0, # Initialize to very bad score
        'error_message': None,
        'optimization_success': False, # From program's optimization attempt
        'optimized_params': None # Store the params returned by the program
    }}

    optimized_function_from_program = None

    try:
        X_train = np.load(X_TRAIN_EVAL_PATH)
        y_train = np.load(Y_TRAIN_EVAL_PATH)

        if X_train.ndim != 2 or X_train.shape[1] != NUM_INPUT_FEATURES_EXPECTED:
            metrics['error_message'] = f"Loaded X_train shape {{X_train.shape}} incompatible with expected features {{NUM_INPUT_FEATURES_EXPECTED}}."
            return filter_and_convert_metrics(metrics)
        if y_train.ndim != 1 or X_train.shape[0] != y_train.shape[0]:
            metrics['error_message'] = f"X_train samples {{X_train.shape[0]}} mismatch y_train samples {{y_train.shape[0]}} or y_train not 1D."
            return filter_and_convert_metrics(metrics)

    except Exception as e:
        metrics['error_message'] = f"Failed to load training data: {{str(e)}}."
        return filter_and_convert_metrics(metrics)

    try:
        spec = importlib.util.spec_from_file_location("evolved_model_program", program_path)
        if spec is None or spec.loader is None:
            metrics['error_message'] = f"Could not create import spec for module at {{program_path}}"
            return filter_and_convert_metrics(metrics)

        model_module = importlib.util.module_from_spec(spec)
        sys.modules['evolved_model_program'] = model_module # Add to sys.modules for potential internal imports if any
        spec.loader.exec_module(model_module)
        metrics['can_run'] = 0.2

        if not hasattr(model_module, 'run_search') or not callable(model_module.run_search):
            metrics['error_message'] = "Evolved program is missing a callable 'run_search' function."
            return filter_and_convert_metrics(metrics)

        evolved_run_search = model_module.run_search
        metrics['can_run'] = 0.5

        # `run_search` no longer takes initial_params_guess from evaluator
        result_tuple = run_with_timeout(
            evolved_run_search,
            args=(X_train, y_train), # Only training data is passed
            timeout_seconds=EVOLVED_PROGRAM_RUN_SEARCH_TIMEOUT_SECONDS
        )

        if not (isinstance(result_tuple, tuple) and len(result_tuple) == 4):
            metrics['error_message'] = f"Evolved program's run_search returned unexpected result. Expected tuple of 4, got: {{type(result_tuple)}} len {{len(result_tuple) if isinstance(result_tuple, tuple) else 'N/A'}}"
            return filter_and_convert_metrics(metrics)

        optimized_function_from_program, final_params, train_mse_from_program, opt_success_flag = result_tuple
        metrics['can_run'] = 1.0

        metrics['raw_mse_train'] = float(train_mse_from_program) if np.isfinite(train_mse_from_program) else float('inf')
        
        if isinstance(final_params, np.ndarray):
            metrics['optimized_params'] = final_params.tolist()
            # Update num_params if the evolved program somehow changed it, though it's guided by MODEL_NUM_PARAMS
            # For scoring, we still primarily care about the MSE.
            # metrics['num_params'] = final_params.shape[0] # This could be uncommented if we want to record actual
        else:
            metrics['optimized_params'] = None


        metrics['optimization_success'] = bool(opt_success_flag)

        if not callable(optimized_function_from_program):
            error_addon = "Evolved program's run_search did not return a callable optimized function."
            if metrics['error_message'] is None: metrics['error_message'] = error_addon
            elif len(metrics['error_message']) < 200: metrics['error_message'] += f"; {{error_addon}}"
            optimized_function_from_program = None # Ensure it's None

    except TimeoutError as te:
        metrics['can_run'] = metrics.get('can_run', 0.5) # Could timeout during import or run_search
        metrics['error_message'] = f"Operation involving evolved program timed out: {{str(te)}}"
    except FileNotFoundError:
        metrics['error_message'] = f"Evolved program file not found: {{program_path}}"
        return filter_and_convert_metrics(metrics)
    except Exception as e:
        error_addon = f"Failed to load/execute evolved program: {{str(e)}}. Trace: {{traceback.format_exc()}}"
        if metrics['error_message'] is None: metrics['error_message'] = error_addon
        elif len(metrics['error_message']) < 200 : metrics['error_message'] += f"; Exc: {{str(e)}}"
        # If error happened after can_run was set, keep it, otherwise it's 0.

    # Calculate scores based on raw_mse_train reported by the evolved program
    # This raw_mse_train is what the evolved program's optimize_model function returned.
    if np.isfinite(metrics['raw_mse_train']) and metrics['raw_mse_train'] >= 0:
        # Avoid log(0) or log(negative) if raw_mse_train is extremely small or errant
        safe_mse_for_log = max(metrics['raw_mse_train'], 1e-12)
        metrics['negative_mse'] = -metrics['raw_mse_train']
        # combined_score can be based directly on negative_mse or a scaled version.
        # For simplicity, let's make combined_score directly reflect negative_mse for now,
        # or use the log score as originally intended for mse_train_score.
        # If combined_score is -log10(mse), then very high MSE -> very negative score.
        # mse_train_score = -np.log10(safe_mse_for_log)
        # metrics['mse_train_score'] = mse_train_score
        metrics['combined_score'] = -np.log10(safe_mse_for_log) # Use log score for combined_score
    else:
        metrics['negative_mse'] = -float('inf')
        # metrics['mse_train_score'] = -100.0
        metrics['combined_score'] = -100.0 # Default bad score

    # Test set evaluation (using the callable returned by the evolved program)
    if callable(optimized_function_from_program):
        try:
            X_test = np.load(X_TEST_EVAL_PATH)
            y_test = np.load(Y_TEST_EVAL_PATH)
            if X_test.shape[1] == NUM_INPUT_FEATURES_EXPECTED and X_test.shape[0] == y_test.shape[0]:
                test_predictions = run_with_timeout(optimized_function_from_program, args=(X_test,), timeout_seconds=PREDICTION_TIMEOUT_SECONDS)
                if isinstance(test_predictions, np.ndarray) and test_predictions.shape == y_test.shape and \
                   not (np.any(np.isnan(test_predictions)) or np.any(np.isinf(test_predictions))):
                    metrics['raw_mse_test'] = float(np.mean((test_predictions - y_test)**2))
                else:
                    err_msg = "Test predictions invalid (shape, type, NaN/Inf)."
                    if metrics['error_message'] is None: metrics['error_message'] = err_msg
                    elif len(metrics['error_message']) < 200: metrics['error_message'] += f"; {{err_msg}}"
            else:
                err_msg = "Test data shape mismatch."
                if metrics['error_message'] is None: metrics['error_message'] = err_msg
                elif len(metrics['error_message']) < 200: metrics['error_message'] += f"; {{err_msg}}"
        except FileNotFoundError:
            pass # Test data files might not be found in some contexts, don't overwrite main error
        except TimeoutError:
            err_msg = "Test prediction timed out."
            if metrics['error_message'] is None: metrics['error_message'] = err_msg
            elif len(metrics['error_message']) < 200: metrics['error_message'] += f"; {{err_msg}}"
        except Exception as e:
            err_msg = f"Error during test set prediction: {{str(e)}}"
            if metrics['error_message'] is None: metrics['error_message'] = err_msg
            elif len(metrics['error_message']) < 200: metrics['error_message'] += f"; {{err_msg}}"

        # OOD test set evaluation
        if X_OOD_TEST_EVAL_PATH and Y_OOD_TEST_EVAL_PATH and os.path.exists(X_OOD_TEST_EVAL_PATH):
            try:
                X_ood_test = np.load(X_OOD_TEST_EVAL_PATH)
                y_ood_test = np.load(Y_OOD_TEST_EVAL_PATH)
                if X_ood_test.shape[1] == NUM_INPUT_FEATURES_EXPECTED and X_ood_test.shape[0] == y_ood_test.shape[0]:
                    ood_predictions = run_with_timeout(optimized_function_from_program, args=(X_ood_test,), timeout_seconds=PREDICTION_TIMEOUT_SECONDS)
                    if isinstance(ood_predictions, np.ndarray) and ood_predictions.shape == y_ood_test.shape and \
                       not (np.any(np.isnan(ood_predictions)) or np.any(np.isinf(ood_predictions))):
                        metrics['raw_mse_ood_test'] = float(np.mean((ood_predictions - y_ood_test)**2))
                    # Silently ignore invalid OOD predictions to not clobber more important errors
                # Silently ignore OOD shape mismatches
            except FileNotFoundError:
                pass # OOD files might not exist
            except TimeoutError:
                err_msg = "OOD test prediction timed out."
                if metrics['error_message'] is None: metrics['error_message'] = err_msg
                elif len(metrics['error_message']) < 200: metrics['error_message'] += f"; {{err_msg}}"
            except Exception: # Broad catch for OOD errors
                pass # Silently ignore other OOD errors
    else: # optimized_function_from_program was not callable
        if metrics['error_message'] is None:
             metrics['error_message'] = "Optimized function was not callable; cannot perform test/OOD evaluations."
        elif len(metrics['error_message']) < 150: # Append if space
             metrics['error_message'] += " Optimized func non-callable, no test/OOD eval."


    return filter_and_convert_metrics(metrics)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <path_to_evolved_model_program.py>")
        sys.exit(1)

    program_to_evaluate = sys.argv[1]
    if not os.path.exists(program_to_evaluate):
        print(f"Error: Program file '{{program_to_evaluate}}' not found.")
        sys.exit(1)

    print(f"Evaluating evolved model: {{program_to_evaluate}}")
    print(f"Using NUM_INPUT_FEATURES_EXPECTED = {{NUM_INPUT_FEATURES_EXPECTED}}")
    print(f"Using MODEL_NUM_PARAMS_EXPECTED = {{MODEL_NUM_PARAMS_EXPECTED}}")
    # ... (rest of the __main__ block prints for data paths)

    essential_data_files = [X_TRAIN_EVAL_PATH, Y_TRAIN_EVAL_PATH, X_TEST_EVAL_PATH, Y_TEST_EVAL_PATH]
    for f_path in essential_data_files:
        if not os.path.exists(f_path):
            print(f"Error: Essential data file '{{f_path}}' not found.")
            sys.exit(1)

    start_time = time.time()
    evaluation_results = evaluate(program_to_evaluate)
    end_time = time.time()

    print(f"\\n--- Evaluation Results (took {{end_time - start_time:.2f}}s) ---")
    if evaluation_results:
        for key, value in evaluation_results.items():
            if isinstance(value, float):
                print(f"  {{key}}: {{value:.4g}}")
            elif isinstance(value, list) and value and isinstance(value[0], (float, np.float32, np.float64)): # Check for list of floats
                print(f"  {{key}}: {{[f'{{x:.4g}}' if isinstance(x, (float, np.float32, np.float64)) else str(x) for x in value]}}")
            else:
                print(f"  {{key}}: {{value}}")
    else:
        print("  Evaluation returned no results or an empty dictionary.")


'''
    evaluator_file_path = os.path.join(problem_dir, "evaluator.py")
    with open(evaluator_file_path, "w") as f:
        f.write(evaluator_script_content)

    return evaluator_file_path


def create_config(problem: Dict[str, Any]) -> str:
    """
    Create a YAML configuration file for the symbolic regression task,
    with an updated system prompt for the LLM.

    Args:
        problem: Dictionary containing problem data

    Returns:
        Path to the created configuration file
    """
    problem_dir = f'problems/{problem["dataset_identifier"]}/{problem["equation_idx"]}'
    os.makedirs(problem_dir, exist_ok=True)
    config_file_path = os.path.join(problem_dir, "config.yaml")

    symbols = problem["symbols"]
    properties = problem["symbol_properties"]
    descs = problem["symbol_descs"]

    input_vars_list = []
    output_var_list = []

    for i, prop in enumerate(properties):
        if prop == "V":
            input_vars_list.append(f"{symbols[i]} ({descs[i]})")
        elif prop == "O":
            output_var_list.append(f"{symbols[i]} ({descs[i]})")

    input_vars_str = ", ".join(input_vars_list) if input_vars_list else "None"
    output_var_str = ", ".join(output_var_list) if output_var_list else "None (Error: No output defined!)"

    model_num_params = 10 # Guideline, must match create_program
    num_features_for_prompt = len(input_vars_list)

    system_message = (
        "You are an expert in scientific modeling and Python programming. Your task is to **evolve three key components** of a Python program "
        "to discover a symbolic equation that models a given scientific process. The evaluation is based on Mean Squared Error (MSE) on training data.\n\n"
        "**You need to define these three Python functions within the ## EVOLVE-BLOCK-START ... ## EVOLVE-BLOCK-END markers:**\n\n"
        "1.  **`symbolic_equation(x: np.ndarray, params: np.ndarray) -> np.ndarray`**\n"
        "    - This is the core mathematical model.\n"
        f"    - `x`: 2D NumPy array, shape `(n_samples, {num_features_for_prompt})`. Input features: {input_vars_str}.\n"
        f"    - `params`: 1D NumPy array, shape `({model_num_params},)`. These are the tunable parameters for your equation. Use the global `MODEL_NUM_PARAMS` constant.\n"
        "    - Must return a 1D NumPy array of predictions, shape `(n_samples,)`.\n"
        "    - Aim for physical plausibility, accuracy, and simplicity.\n\n"
        "2.  **`initialize_parameters() -> np.ndarray`**\n"
        "    - This function generates the initial guess for the `params` array used in `symbolic_equation`.\n"
        f"    - Must return a 1D NumPy array of shape `({model_num_params},)` (i.e., matching `MODEL_NUM_PARAMS`).\n"
        "    - Good initial guesses can significantly help the optimization process.\n\n"
        "3.  **`optimize_model(X_train: np.ndarray, y_train: np.ndarray, initial_params: np.ndarray, symbolic_eq_func: callable) -> tuple`**\n"
        "    - This function defines and executes the optimization strategy to find the best `params` for `symbolic_eq_func` (your `symbolic_equation`) using the provided `X_train`, `y_train` data and `initial_params`.\n"
        "    - It must return a tuple: `(optimized_callable, optimized_params, train_mse, optimization_success_flag)`\n"
        "        - `optimized_callable`: A function that takes X (input data) and returns predictions using the `optimized_params` (e.g., `lambda x_eval: symbolic_eq_func(x_eval, optimized_params)`).\n"
        "        - `optimized_params`: The 1D NumPy array of optimized parameters, shape `(MODEL_NUM_PARAMS,)`.\n"
        "        - `train_mse`: The Mean Squared Error achieved on `X_train`, `y_train` with `optimized_params`.\n"
        "        - `optimization_success_flag`: Boolean indicating if your optimization process reported success.\n"
        "    - A default BFGS-based optimization function `_default_bfgs_optimize` is provided as a starting point. You can use it, modify its usage (e.g., change `maxiter`, `gtol`), or replace it entirely with your own custom optimization logic (e.g., different algorithms like L-BFGS-B if bounds are needed, Nelder-Mead, or even a simple gradient descent if you derive gradients).\n\n"
        "**Goal:**\n"
        "Your primary goal is to discover the **correct underlying equation structure** via `symbolic_equation` and effectively **tune its parameters** using `initialize_parameters` and `optimize_model`.\n"
        f"The program will be evaluated based on the training MSE achieved by your `symbolic_equation` with its parameters optimized by your `optimize_model` function. The target output variable is: {output_var_str}.\n\n"
        "**Context & Constraints:**\n"
        "- The `run_search(X_train, y_train)` function (outside the EVOLVE-BLOCK) will call your `initialize_parameters()` and then `optimize_model(..., symbolic_equation)`.\n"
        "- Ensure your functions are robust. Avoid numerical errors (e.g., log(0), division by zero, sqrt of negative) for valid inputs `x` and `params`.\n"
        f"- The constant `MODEL_NUM_PARAMS = {model_num_params}` is defined globally within the script. Your `params` arrays (both initial and optimized) should consistently have this many elements.\n"
        "- Keep the code within the EVOLVE-BLOCK focused. The evaluation framework handles data loading and final metric calculation.\n"
        "- Focus on both **equation discovery** and the **method to initialize and optimize its parameters** effectively.\n"
    )

    config_data = {
        "# Configuration for Symbolic Regression Task": f"{problem['dataset_identifier']}/{problem['equation_idx']}",
        "max_iterations": 100,
        "log_level": "INFO",
        "target_score": "combined_score", # From filtered metrics: 'combined_score' or 'negative_mse'
        "checkpoint_interval": 10,
        "random_seed": 2025,
        "llm": {
            "primary_model": "gpt-4.1", # Example, adjust as needed
            "primary_model_weight": 0.8,
            "secondary_model": "o4-mini", # Example, adjust as needed
            "secondary_model_weight": 0.2,
            "api_base": "https://api.openai.com/v1",
        },
        "prompt": {
            "system_message": system_message,
            "num_top_programs": 4,
            "num_diverse_programs": 2,
            "use_template_stochasticity": True,
        },
        "database": {
            "population_size": 1000,
            "archive_size": 100,
            "num_islands": 5,
            "migration_interval": 50,
            "migration_rate": 0.1,
            "elite_selection_ratio": 0.3,
            "exploitation_ratio": 0.6,
        },
        "evaluator": {
            "timeout": 600,
            "cascade_evaluation": False,
            "cascade_thresholds": [1.0],
            "parallel_evaluations": 4,
            "use_llm_feedback": False,
        },
        "diff_based_evolution": False, 
        "allow_full_rewrites": True, 
    }

    class PreserveNewlinesDumper(yaml.SafeDumper):
        def represent_scalar(self, tag, value, style=None):
            if style is None and isinstance(value, str) and "\n" in value:
                style = "|"
            return super().represent_scalar(tag, value, style)

    with open(config_file_path, "w") as f:
        yaml.dump(
            config_data,
            f,
            Dumper=PreserveNewlinesDumper,
            default_flow_style=False,
            sort_keys=False,
            indent=2,
        )

    return config_file_path


def process_problem(initialized_dataset, problem_id: int, split_name: str) -> str:
    """
    Process a single problem using a pre-initialized dataset.
    Loads specific problem data, creates program, evaluator, and config.
    Skips processing if essential output files already exist.
    """
    try:
        problem_data = extract_problem_data_from_initialized_dataset(
            initialized_dataset, problem_id
        )

        dataset_identifier = problem_data["dataset_identifier"]
        equation_idx = problem_data["equation_idx"]
        problem_dir = os.path.join("problems", dataset_identifier, str(equation_idx))

        essential_files = [
            os.path.join(problem_dir, "initial_program.py"),
            os.path.join(problem_dir, "evaluator.py"),
            os.path.join(problem_dir, "config.yaml"),
            os.path.join("./", problem_dir, "X_train_for_eval.npy"),
            os.path.join("./", problem_dir, "y_train_for_eval.npy"),
            os.path.join("./", problem_dir, "X_test_for_eval.npy"),
            os.path.join("./", problem_dir, "y_test_for_eval.npy"),
        ]

        if problem_data.get("ood_test") is not None:
            essential_files.extend(
                [
                    os.path.join("./", problem_dir, "X_ood_test_for_eval.npy"),
                    os.path.join("./", problem_dir, "y_ood_test_for_eval.npy"),
                ]
            )

        all_files_exist = all(os.path.exists(f) for f in essential_files)

        if all_files_exist:
            return f"Skipped (already processed): problem_id: {problem_id} for split: {split_name} ({dataset_identifier}/{equation_idx})"

        create_program(problem_data)
        create_evaluator(problem_data)
        create_config(problem_data)

        return f"Successfully processed problem_id: {problem_id} for split: {split_name} ({dataset_identifier}/{equation_idx})"

    except Exception as e:
        return f"Error processing problem_id {problem_id} for split {split_name}: {str(e)}\n{traceback.format_exc()}"


def main():
    num_cores_available = os.cpu_count()
    num_processes = min(max(1, (num_cores_available - 1) if num_cores_available else 4), 24) # Cap at 24 or adjust as needed

    print(f"Starting processing with {num_processes} processes...")

    splits_data = {
        "bio_pop_growth": 24,
        "chem_react": 36,
        "matsci": 25,
        "phys_osc": 44,
        # 'lsrtransform': 111
    }

    all_tasks = []

    for split_name, num_problems in splits_data.items():
        print(f"\nInitializing dataset for split: {split_name}...")
        dataset_root_folder = f"dataset/{split_name}" # Assuming dataset is in ./dataset/

        try:
            initialized_dataset = get_datamodule(split_name, dataset_root_folder)
            initialized_dataset.setup()
            print(f"Dataset for {split_name} initialized and setup complete.")

            print(f"Preparing tasks for split: {split_name} ({num_problems} problems)")
            for problem_id in range(num_problems):
                all_tasks.append((initialized_dataset, problem_id, split_name))

        except Exception as e:
            print(
                f"ERROR: Could not initialize or setup dataset for split {split_name}. Skipping this split."
            )
            print(f"Details: {e}")
            traceback.print_exc()
            continue

    if not all_tasks:
        print(
            "No tasks to process. This could be due to errors in dataset initialization. Exiting."
        )
        return

    print(f"\nTotal tasks to process across all successfully initialized splits: {len(all_tasks)}")

    # Use try-finally for pool shutdown even if starmap fails early
    pool = multiprocessing.Pool(processes=num_processes)
    try:
        results = pool.starmap(process_problem, all_tasks)
    finally:
        pool.close()
        pool.join()


    print("\n--- Processing Complete ---")
    success_count = 0
    skipped_count = 0
    error_count = 0

    for result in results:
        print(result) # Print status of each task
        if "Successfully processed" in result:
            success_count += 1
        elif "Skipped" in result:
            skipped_count += 1
        elif "Error processing" in result:
            error_count += 1

    print(f"\nSummary: {success_count} successful, {skipped_count} skipped, {error_count} errors.")
    print("\nAll tasks finished.")


if __name__ == "__main__":
    main()