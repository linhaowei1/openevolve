"""
Evaluator for 1D Reaction-Diffusion PDE Solver with timeout handling
and automatic GPU allocation. (Modified as per request)
"""

import importlib.util
import numpy as np
import time
import os
import signal
import subprocess
import tempfile
import traceback
import sys
import pickle
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Define TimeoutError for handling long executions
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")

# --- GPU Allocation ---

def get_free_gpu():
    """
    Finds the ID of the GPU with the most free memory and lowest utilization.
    Returns its ID as a string, or None if nvidia-smi fails or no GPU is found.
    """
    try:
        command = ['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu', '--format=csv,noheader,nounits']
        output = subprocess.check_output(command, encoding='utf-8').strip()
        gpus = []
        for line in output.split('\n'):
            if line:
                try:
                    index, free_mem, util = line.split(', ')
                    gpus.append({
                        'id': int(index),
                        'free_mem': float(free_mem),
                        'util': float(util)
                    })
                except ValueError:
                    print(f"Warning: Could not parse nvidia-smi line: '{line}'")
                    continue

        if not gpus:
            print("Warning: No GPUs found by nvidia-smi.")
            return None

        best_gpu = min(gpus, key=lambda gpu: (gpu['util'], -gpu['free_mem']))
        print(f"Found GPUs: {gpus}")
        print(f"Selected GPU {best_gpu['id']} (Util: {best_gpu['util']}%, Free: {best_gpu['free_mem']}MiB)")
        return str(best_gpu['id'])

    except FileNotFoundError:
        print("Warning: nvidia-smi command not found. Cannot automatically assign GPU.")
        return None
    except Exception as e:
        print(f"Warning: Error while querying GPUs: {e}")
        return None

# --- Helper functions (to be used inside the subprocess) ---
_SUBPROCESS_HELPERS = """
import numpy as np
import h5py
import time

def compute_rmse(u_computed, u_reference):
    \"\"\"Computes the Root Mean Squared Error (RMSE).\"\"\"
    # Ensure inputs are numpy arrays
    u_computed = np.asarray(u_computed)
    u_reference = np.asarray(u_reference)

    # Check for NaN or Inf values
    if not np.all(np.isfinite(u_computed)):
        print("Warning: NaN or Inf found in computed solution. Returning inf RMSE.")
        return np.inf
    if not np.all(np.isfinite(u_reference)):
        print("Warning: NaN or Inf found in reference solution.")
        return np.inf

    # Calculate RMSE per batch item
    rmse_values = np.sqrt(np.mean((u_computed - u_reference)**2, axis=(1,2)))

    # Average RMSE across the batch
    rmse = np.mean(rmse_values)
    return rmse

def load_data(path, is_h5py=True):
    \"\"\"Loads data from an HDF5 file.\"\"\"
    if is_h5py:
        with h5py.File(path, 'r') as f:
            t_coordinate = np.array(f['t-coordinate'])
            u = np.array(f['tensor'])
            x_coordinate = np.array(f['x-coordinate'])
    else:
        raise NotImplementedError("Only h5py format is supported for now.")

    t_min, t_max = t_coordinate[0], t_coordinate[-1]
    x_min, x_max = x_coordinate[0], x_coordinate[-1]
    return dict(
        tensor=u,
        t_coordinate=t_coordinate,
        x_coordinate=x_coordinate,
        t_min=t_min,
        t_max=t_max,
        x_min=x_min,
        x_max=x_max
    )
"""

# --- Subprocess Execution ---

def run_with_timeout(program_path, dataset_path, nu, rho, timeout_seconds=600):
    """
    Run the PDE evaluation in a separate process with timeout and GPU selection.
    """
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w', encoding='utf-8') as temp_file:
        script = f"""
import sys
import numpy as np
import os
import pickle
import traceback
import h5py
import time
import importlib.util

# Add helper functions
{_SUBPROCESS_HELPERS}

# Main execution block
def run_evaluation(program_path, dataset_path, nu, rho, results_path):
    try:
        # Import the solver
        print(f"Importing solver from: {{program_path}}")
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        solver = program.solver
        print("Solver imported successfully.")

        # Load data
        print(f"Loading data from: {{dataset_path}}")
        data_dict = load_data(dataset_path)
        u = data_dict['tensor']
        t_coordinate = data_dict['t_coordinate']
        x_coordinate = data_dict['x_coordinate']
        print(f"Loaded data with shape: {{u.shape}}")

        # Extract test set
        u0 = u[:, 0]
        u_ref = u[:, :]
        batch_size, N = u0.shape

        # Run solver
        print("Running the solver...")
        start_time = time.time()
        u_batch = solver(u0, t_coordinate, nu, rho)
        end_time = time.time()
        eval_time = end_time - start_time
        print(f"Solver finished in {{eval_time:.2f}}s.")

        # Compute RMSE
        rmse = compute_rmse(u_batch, u_ref)
        print(f"RMSE: {{rmse:.3e}}")

        # No convergence test anymore

        results = {{
            'rmse': rmse,
            'eval_time': eval_time,
            'status': 'success'
        }}

    except Exception as e:
        print(f"Error in subprocess: {{str(e)}}")
        traceback.print_exc()
        results = {{'error': str(e), 'status': 'error', 'traceback': traceback.format_exc(), 'rmse': float('inf')}}

    # Save results
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {{results_path}}")

if __name__ == "__main__":
    prog_path = sys.argv[1]
    data_path = sys.argv[2]
    nu_val = float(sys.argv[3])
    rho_val = float(sys.argv[4])
    res_path = sys.argv[5] # Adjusted index
    run_evaluation(prog_path, data_path, nu_val, rho_val, res_path) # Removed t_max_factor
"""
        temp_file.write(script)
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    gpu_id = get_free_gpu()
    env = os.environ.copy()
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = gpu_id
        print(f"Setting CUDA_VISIBLE_DEVICES={gpu_id} for subprocess.")
    else:
        print("Warning: Running subprocess without setting CUDA_VISIBLE_DEVICES.")

    try:
        cmd = [
            sys.executable, temp_file_path,
            program_path, dataset_path, str(nu), str(rho), results_path # Adjusted command
        ]
        print(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            print(f"Subprocess stdout:\n{stdout.decode(errors='ignore')}")
            if stderr:
                print(f"Subprocess stderr:\n{stderr.decode(errors='ignore')}")

            if exit_code != 0:
                 raise RuntimeError(f"Process exited with code {exit_code}. Stderr: {stderr.decode(errors='ignore')}")

            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)
                if results.get('status') == 'error':
                    raise RuntimeError(f"Program execution failed: {results['error']}\n{results.get('traceback', '')}")
                return results
            else:
                raise RuntimeError(f"Results file not found. Stdout: {stdout.decode(errors='ignore')}, Stderr: {stderr.decode(errors='ignore')}")

        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        if os.path.exists(temp_file_path):
             os.unlink(temp_file_path)
        if os.path.exists(results_path):
             os.unlink(results_path)

# --- Core Evaluation Logic ---

def _core_evaluate(program_path, dataset_path, nu, rho):
    """
    Core evaluation logic.
    """
    try:
        if not os.path.exists(dataset_path):
             print(f"Error: Dataset path {dataset_path} does not exist. Cannot evaluate.")
             return {
                "rmse": float('inf'),
                "negative_rmse": float('-inf'),
                "eval_time": 0.0,
                "validity": 0.0,
                "combined_score": 0.0,
                "error": f"Dataset not found at {dataset_path}"
             }

        results = run_with_timeout(
            program_path, dataset_path, nu, rho, timeout_seconds=2400
        )

        rmse = results.get('rmse', float('inf'))
        eval_time = results.get('eval_time', 0.0)

        validity = 1.0 if rmse != float('inf') and not np.isnan(rmse) else 0.0

        if validity == 0.0:
            negative_rmse = float('-inf')
            combined_score = 0.0
        elif rmse == 0.0:
            negative_rmse = float('inf')
            combined_score = 1.0 # Max score for perfect RMSE
        else:
            negative_rmse = -rmse
            # Define range for -log10(rmse) to map to [0, 1]
            S_min = 0.0 # Corresponds to RMSE = 1.0
            S_max = 8.0 # Corresponds to RMSE = 1e-8 (This is the target for max score)
            # Clip and normalize
            combined_score = np.clip(-np.log10(rmse), S_min, S_max) / S_max
            # Ensure it's within [0, 1]
            combined_score = np.clip(combined_score, 0.0, 1.0)

        print(
            f"Evaluation: RMSE={rmse:.3e}, negative_rmse={negative_rmse:.3f}, "
            f"Time={eval_time:.2f}s, Score={combined_score:.4f}"
        )

        return {
            "rmse": float(rmse),
            "negative_rmse": float(negative_rmse),
            "eval_time": float(eval_time),
            "validity": float(validity),
            "combined_score": float(combined_score),
        }

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        traceback.print_exc()
        return {
            "rmse": float('inf'),
            "negative_rmse": float('-inf'),
            "eval_time": 0.0,
            "validity": 0.0,
            "combined_score": 0.0,
            "error": str(e)
        }

# --- Evaluation Functions (Simplified Interface) ---

def evaluate(program_path):
    """
    Evaluate the PDE solver program with default/full parameters.
    Dataset info is hardcoded.
    """
    print("--- Running Full Evaluation ---")
    dataset_path = 'ReacDiff_Nu0.5_Rho1.0_development.hdf5'
    nu = 0.5
    rho = 1.0
    return _core_evaluate(program_path, dataset_path, nu, rho)

def evaluate_stage1(program_path):
    """
    First stage evaluation - quick check.
    Dataset info is hardcoded.
    """
    print("--- Running Stage 1 Evaluation ---")
    dataset_path = 'ReacDiff_Nu0.5_Rho1.0_development.hdf5'
    nu = 0.5
    rho = 1.0
    # Currently same as full eval, as t_max_conv_factor is removed
    return _core_evaluate(program_path, dataset_path, nu, rho)


def evaluate_stage2(program_path):
    """
    Second stage evaluation - same as full 'evaluate'.
    Dataset info is hardcoded.
    """
    print("--- Running Stage 2 Evaluation ---")
    return evaluate(program_path)

if __name__ == '__main__':
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate 1D Reaction-Diffusion PDE Solver')
    parser.add_argument('program_path', type=str, help='Path to the solver program')
    parser.add_argument('--dataset', type=str, default='ReacDiff_Nu0.5_Rho1.0_development.hdf5',
                        help='Path to the dataset (default: ReacDiff_Nu0.5_Rho1.0_development.hdf5)')
    parser.add_argument('--nu', type=float, default=0.5,
                        help='Nu parameter value (default: 0.5)')
    parser.add_argument('--rho', type=float, default=1.0,
                        help='Rho parameter value (default: 1.0)')
    parser.add_argument('--stage', type=int, choices=[1, 2], default=None,
                        help='Run specific stage evaluation (1 or 2). If not specified, runs full evaluation.')
    
    args = parser.parse_args()
    
    print(f"Evaluating solver: {args.program_path}")
    print(f"Using dataset: {args.dataset}")
    print(f"Parameters: nu={args.nu}, rho={args.rho}")
    print("-" * 60)
    
    # Run evaluation based on stage or full evaluation
    if args.stage == 1:
        # For stage 1, we can override the default dataset path
        original_dataset = 'ReacDiff_Nu0.5_Rho1.0_development.hdf5'
        # Temporarily modify the function to use custom dataset
        results = _core_evaluate(args.program_path, args.dataset, args.nu, args.rho)
        print("\n=== Stage 1 Evaluation Results ===")
    elif args.stage == 2:
        # For stage 2, directly use custom parameters
        results = _core_evaluate(args.program_path, args.dataset, args.nu, args.rho)
        print("\n=== Stage 2 Evaluation Results ===")
    else:
        # Full evaluation with custom parameters
        results = _core_evaluate(args.program_path, args.dataset, args.nu, args.rho)
        print("\n=== Full Evaluation Results ===")
    
    # Print results in a formatted way
    print(f"RMSE: {results['rmse']:.6e}")
    print(f"Negative RMSE: {results['negative_rmse']:.6f}")
    print(f"Evaluation Time: {results['eval_time']:.2f} seconds")
    print(f"Validity: {results['validity']}")
    print(f"Combined Score: {results['combined_score']:.4f}")
    
    if 'error' in results:
        print(f"\nError encountered: {results['error']}")
    
    print("-" * 60)
    
    
    # Exit with appropriate code
    sys.exit(0 if results['validity'] == 1.0 else 1)