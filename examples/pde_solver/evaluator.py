"""
Evaluator for 1D Reaction-Diffusion PDE Solver with cascade evaluation
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
from scipy.interpolate import interp1d


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def compute_nrmse(u_computed, u_reference):
    """Computes the Normalized Root Mean Squared Error (nRMSE) between the computed solution and reference.
    
    Args:
        u_computed (np.ndarray): Computed solution [batch_size, len(t_coordinate), N].
        u_reference (np.ndarray): Reference solution [batch_size, len(t_coordinate), N].
        
    Returns:
        nrmse (np.float32): The normalized RMSE value.
    """
    rmse_values = np.sqrt(np.mean((u_computed - u_reference)**2, axis=(1,2)))
    u_true_norm = np.sqrt(np.mean(u_reference**2, axis=(1,2)))
    nrmse = np.mean(rmse_values / u_true_norm)
    return nrmse


def init(xc, modes=["sin"], u0=1.0, du=0.1):
    """Initializes one or more 1D scalar functions based on specified modes."""
    initial_conditions = []
    for mode in modes:
        assert mode in ["sin", "sinsin", "Gaussian", "react", "possin"], f"mode {mode} not supported!"
        
        if mode == "sin":
            u = u0 * np.sin((xc + 1.0) * np.pi)
        elif mode == "sinsin":
            u = np.sin((xc + 1.0) * np.pi) + du * np.sin((xc + 1.0) * np.pi * 8.0)
        elif mode == "Gaussian":
            t0 = 1.0
            u = np.exp(-(xc**2) * np.pi / (4.0 * t0)) / np.sqrt(2.0 * t0)
        elif mode == "react":
            logu = -0.5 * (xc - np.pi) ** 2 / (0.25 * np.pi) ** 2
            u = np.exp(logu)
        elif mode == "possin":
            u = u0 * np.abs(np.sin((xc + 1.0) * np.pi))
            
        initial_conditions.append(u)
    return np.stack(initial_conditions)


def interpolate_solution(u_fine, x_fine, t_fine, x_coarse, t_coarse):
    """Interpolates the fine solution onto the coarse grid in both space and time."""
    # Interpolate in space
    space_interp_func = interp1d(x_fine, u_fine, axis=2, kind='linear', fill_value="extrapolate")
    u_fine_interp_space = space_interp_func(x_coarse)
    
    # Interpolate in time
    time_interp_func = interp1d(t_fine, u_fine_interp_space, axis=1, kind='linear', fill_value="extrapolate")
    u_fine_interp = time_interp_func(t_coarse)
    
    return u_fine_interp


def compute_error(coarse_tuple, fine_tuple):
    """Computes the error between coarse and fine grid solutions."""
    u_coarse, x_coarse, t_coarse = coarse_tuple
    u_fine, x_fine, t_fine = fine_tuple
    u_fine_interp = interpolate_solution(u_fine, x_fine, t_fine, x_coarse, t_coarse)
    
    # Compute L2 norm error
    error = np.mean(np.linalg.norm(u_coarse - u_fine_interp, axis=(1,2))) / np.sqrt(u_coarse.size)
    return error


def get_x_coordinate(x_min, x_max, nx):
    dx = (x_max - x_min) / nx
    xe = np.linspace(x_min, x_max, nx+1)
    xc = xe[:-1] + 0.5 * dx
    return xc


def get_t_coordinate(t_min, t_max, dt):
    it_tot = int(np.ceil((t_max - t_min) / dt) + 1)
    tc = np.arange(it_tot + 1) * dt
    return tc


def load_data(path):
    """Load data from HDF5 file."""
    with h5py.File(path, 'r') as f:
        t_coordinate = np.array(f['t-coordinate'])
        u = np.array(f['tensor'])
        x_coordinate = np.array(f['x-coordinate'])
    
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


def convergence_test_subprocess(solver_module, nu, rho,
                              nxs=[256, 512, 1024, 2048],
                              dts=[0.01, 0.01, 0.01, 0.01],
                              t_min=0, t_max=2,
                              x_min=-1, x_max=1):
    """Run convergence test on the solver."""
    us = []
    xcs = []
    tcs = []
    
    for nx, dt in zip(nxs, dts):
        tc = get_t_coordinate(t_min, t_max, dt)
        xc = get_x_coordinate(x_min, x_max, nx)
        u0 = init(xc)
        u = solver_module.solver(u0, tc, nu, rho)
        us.append(np.squeeze(np.array(u)))
        xcs.append(np.array(xc))
        tcs.append(np.array(tc))
    
    # Compute errors
    errors = []
    for i in range(len(nxs) - 1):
        coarse_tuple = (us[i], xcs[i], tcs[i])
        fine_tuple = (us[-1], xcs[-1], tcs[-1])
        error = compute_error(coarse_tuple, fine_tuple)
        errors.append(error)
    
    # Calculate average convergence rate
    rates = []
    for i in range(len(nxs) - 2):
        rate = np.log(errors[i] / errors[i+1]) / np.log(nxs[i+1] / nxs[i])
        rates.append(rate)
    
    avg_rate = np.mean(rates) if rates else 0.0
    return avg_rate


def run_with_timeout(program_path, dataset_path, nu, rho, timeout_seconds=600):
    """
    Run the solver in a separate process with timeout.
    
    Args:
        program_path: Path to the solver file
        dataset_path: Path to the dataset file
        nu: Diffusion coefficient
        rho: Reaction coefficient
        timeout_seconds: Maximum execution time
        
    Returns:
        Dictionary with results
    """
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        script = f"""
import sys
import numpy as np
import os
import pickle
import traceback
import h5py
from scipy.interpolate import interp1d

# Add the directory to sys.path
sys.path.insert(0, os.path.dirname('{program_path}'))

print(f"Running solver evaluation...")
print(f"Program path: {program_path}")
print(f"Dataset path: {dataset_path}")
print(f"nu: {nu}, rho: {rho}")

# Helper functions needed by the solver
def get_x_coordinate(x_min, x_max, nx):
    dx = (x_max - x_min) / nx
    xe = np.linspace(x_min, x_max, nx+1)
    xc = xe[:-1] + 0.5 * dx
    return xc

def get_t_coordinate(t_min, t_max, dt):
    it_tot = int(np.ceil((t_max - t_min) / dt) + 1)
    tc = np.arange(it_tot + 1) * dt
    return tc

def init(xc, modes=["sin"], u0=1.0, du=0.1):
    initial_conditions = []
    for mode in modes:
        if mode == "sin":
            u = u0 * np.sin((xc + 1.0) * np.pi)
        elif mode == "sinsin":
            u = np.sin((xc + 1.0) * np.pi) + du * np.sin((xc + 1.0) * np.pi * 8.0)
        elif mode == "Gaussian":
            t0 = 1.0
            u = np.exp(-(xc**2) * np.pi / (4.0 * t0)) / np.sqrt(2.0 * t0)
        elif mode == "react":
            logu = -0.5 * (xc - np.pi) ** 2 / (0.25 * np.pi) ** 2
            u = np.exp(logu)
        elif mode == "possin":
            u = u0 * np.abs(np.sin((xc + 1.0) * np.pi))
        initial_conditions.append(u)
    return np.stack(initial_conditions)

def interpolate_solution(u_fine, x_fine, t_fine, x_coarse, t_coarse):
    space_interp_func = interp1d(x_fine, u_fine, axis=2, kind='linear', fill_value="extrapolate")
    u_fine_interp_space = space_interp_func(x_coarse)
    time_interp_func = interp1d(t_fine, u_fine_interp_space, axis=1, kind='linear', fill_value="extrapolate")
    u_fine_interp = time_interp_func(t_coarse)
    return u_fine_interp

def compute_error(coarse_tuple, fine_tuple):
    u_coarse, x_coarse, t_coarse = coarse_tuple
    u_fine, x_fine, t_fine = fine_tuple
    u_fine_interp = interpolate_solution(u_fine, x_fine, t_fine, x_coarse, t_coarse)
    error = np.mean(np.linalg.norm(u_coarse - u_fine_interp, axis=(1,2))) / np.sqrt(u_coarse.size)
    return error

try:
    # Import the solver
    spec = __import__('importlib.util').util.spec_from_file_location("solver_module", '{program_path}')
    solver_module = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(solver_module)
    
    # Load dataset
    with h5py.File('{dataset_path}', 'r') as f:
        t_coordinate = np.array(f['t-coordinate'])
        u = np.array(f['tensor'])
        x_coordinate = np.array(f['x-coordinate'])
    
    print(f"Loaded data with shape: {{u.shape}}")
    
    # Extract test set
    u0 = u[:, 0]
    u_ref = u[:, :]
    
    # Run solver
    start_time = __import__('time').time()
    u_batch = solver_module.solver(u0, t_coordinate, {nu}, {rho})
    end_time = __import__('time').time()
    solver_time = end_time - start_time
    
    # Compute nRMSE
    u_batch = np.array(u_batch)
    rmse_values = np.sqrt(np.mean((u_batch - u_ref)**2, axis=(1,2)))
    u_true_norm = np.sqrt(np.mean(u_ref**2, axis=(1,2)))
    nrmse = np.mean(rmse_values / u_true_norm)
    
    # Run simplified convergence test
    t_min, t_max = t_coordinate[0], t_coordinate[-1]
    x_min, x_max = x_coordinate[0], x_coordinate[-1]
    
    # Simplified convergence test with fewer resolutions
    nxs = [256, 512, 1024]
    dts = [0.01, 0.01, 0.01]
    us = []
    xcs = []
    tcs = []
    
    for nx, dt in zip(nxs, dts):
        tc = get_t_coordinate(t_min, min(t_max/10, t_max), dt)  # Limit time for speed
        xc = get_x_coordinate(x_min, x_max, nx)
        u0_conv = init(xc)
        u_conv = solver_module.solver(u0_conv, tc, {nu}, {rho})
        us.append(np.squeeze(np.array(u_conv)))
        xcs.append(np.array(xc))
        tcs.append(np.array(tc))
    
    # Compute convergence rate
    errors = []
    for i in range(len(nxs) - 1):
        coarse_tuple = (us[i], xcs[i], tcs[i])
        fine_tuple = (us[-1], xcs[-1], tcs[-1])
        error = compute_error(coarse_tuple, fine_tuple)
        errors.append(error)
    
    if len(errors) >= 2:
        avg_rate = np.log(errors[0] / errors[1]) / np.log(nxs[1] / nxs[0])
    else:
        avg_rate = 0.0
    
    # Save results
    results = {{
        'nrmse': float(nrmse),
        'avg_rate': float(avg_rate),
        'solver_time': float(solver_time),
        'u_batch_shape': u_batch.shape,
        'success': True
    }}
    
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results: nRMSE={{nrmse:.6f}}, avg_rate={{avg_rate:.3f}}, time={{solver_time:.2f}}s")
    
except Exception as e:
    print(f"Error in subprocess: {{str(e)}}")
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error': str(e), 'success': False}}, f)
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name
    
    results_path = f"{temp_file_path}.results"
    
    try:
        # Run the script with timeout
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode
            
            # Print output for debugging
            if stdout:
                print(f"Subprocess stdout: {stdout.decode()}")
            if stderr:
                print(f"Subprocess stderr: {stderr.decode()}")
            
            if exit_code != 0:
                raise RuntimeError(f"Process exited with code {exit_code}")
            
            # Load results
            if os.path.exists(results_path):
                with open(results_path, 'rb') as f:
                    results = pickle.load(f)
                
                if not results.get('success', False):
                    raise RuntimeError(f"Solver execution failed: {results.get('error', 'Unknown error')}")
                
                return results
            else:
                raise RuntimeError("Results file not found")
                
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")
            
    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)


def evaluate(program_path, dataset_path='ReacDiff_Nu0.5_Rho1.0.hdf5',
             nu=0.5, rho=1.0):
    """
    Evaluate the PDE solver on the reaction-diffusion equation.
    
    Args:
        program_path: Path to the solver file containing solver() function
        dataset_path: Path to the HDF5 dataset
        nu: Diffusion coefficient
        rho: Reaction coefficient
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Target values for good performance (based on typical finite difference methods)
    TARGET_NRMSE = 1e-3  # Target nRMSE for accurate solution
    TARGET_CONVERGENCE_RATE = 2.0  # Expected convergence rate for 2nd order methods
    
    try:
        # Run solver with timeout
        results = run_with_timeout(program_path, dataset_path, nu, rho, timeout_seconds=600)
        
        nrmse = results['nrmse']
        avg_rate = results['avg_rate']
        solver_time = results['solver_time']
        
        # Calculate scores
        # nRMSE score: exponential decay, perfect score at TARGET_NRMSE
        nrmse_score = np.exp(-nrmse / TARGET_NRMSE) if nrmse > 0 else 1.0
        
        # Convergence rate score: normalized to TARGET_CONVERGENCE_RATE
        rate_score = min(avg_rate / TARGET_CONVERGENCE_RATE, 1.0) if avg_rate > 0 else 0.0
        
        # Time score: prefer faster solvers (bonus for < 10s)
        time_score = min(10.0 / solver_time, 1.0) if solver_time > 0 else 0.0
        
        # Validity: solution is valid if nRMSE is reasonable
        validity = 1.0 if nrmse < 1.0 else 0.0
        
        # Combined score weighted by importance
        combined_score = (0.5 * nrmse_score + 0.3 * rate_score + 0.2 * time_score) * validity
        
        print(f"Evaluation complete:")
        print(f"  nRMSE: {nrmse:.6f} (target: {TARGET_NRMSE})")
        print(f"  Convergence rate: {avg_rate:.3f} (target: {TARGET_CONVERGENCE_RATE})")
        print(f"  Solver time: {solver_time:.2f}s")
        print(f"  Combined score: {combined_score:.3f}")
        
        return {
            "nrmse": float(nrmse),
            "convergence_rate": float(avg_rate),
            "solver_time": float(solver_time),
            "nrmse_score": float(nrmse_score),
            "rate_score": float(rate_score),
            "time_score": float(time_score),
            "validity": float(validity),
            "combined_score": float(combined_score)
        }
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        traceback.print_exc()
        return {
            "nrmse": float('inf'),
            "convergence_rate": 0.0,
            "solver_time": 0.0,
            "nrmse_score": 0.0,
            "rate_score": 0.0,
            "time_score": 0.0,
            "validity": 0.0,
            "combined_score": 0.0
        }


def evaluate_stage1(program_path, dataset_path='ReacDiff_Nu0.5_Rho1.0.hdf5',
                   nu=0.5, rho=1.0):
    """
    Stage 1: Quick validation check - test on smaller problem size.
    """
    try:
        # Create a minimal test case
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            script = f"""
import sys
import numpy as np
import pickle
import traceback

sys.path.insert(0, os.path.dirname('{program_path}'))

try:
    # Import solver
    spec = __import__('importlib.util').util.spec_from_file_location("solver_module", '{program_path}')
    solver_module = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(solver_module)
    
    # Create small test case
    nx = 64
    nt = 50
    x = np.linspace(-1, 1, nx)
    t = np.linspace(0, 0.1, nt)
    
    # Simple initial condition
    u0 = np.sin((x + 1.0) * np.pi)
    u0 = u0.reshape(1, -1)  # Add batch dimension
    
    # Run solver
    u_result = solver_module.solver(u0, t, {nu}, {rho})
    u_result = np.array(u_result)
    
    # Basic checks
    success = True
    error_msg = ""
    
    # Check shape
    expected_shape = (1, len(t), nx)
    if u_result.shape != expected_shape:
        success = False
        error_msg = f"Wrong shape: {{u_result.shape}} != {{expected_shape}}"
    
    # Check for NaN/Inf
    if np.any(np.isnan(u_result)) or np.any(np.isinf(u_result)):
        success = False
        error_msg = "Solution contains NaN or Inf"
    
    # Check stability (solution shouldn't explode)
    max_val = np.max(np.abs(u_result))
    if max_val > 100:
        success = False
        error_msg = f"Solution appears unstable: max value = {{max_val}}"
    
    results = {{
        'success': success,
        'error_msg': error_msg,
        'max_val': float(max_val),
        'shape': u_result.shape
    }}
    
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
        
except Exception as e:
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'success': False, 'error_msg': str(e)}}, f)
"""
            temp_file.write(script.encode())
            temp_file_path = temp_file.name
        
        results_path = f"{temp_file_path}.results"
        
        # Run with short timeout
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        try:
            stdout, stderr = process.communicate(timeout=60)  # 1 minute timeout for stage 1
            
            if os.path.exists(results_path):
                with open(results_path, 'rb') as f:
                    results = pickle.load(f)
                
                validity = 1.0 if results['success'] else 0.0
                combined_score = validity  # Simple score for stage 1
                
                return {
                    "validity": float(validity),
                    "combined_score": float(combined_score),
                    "error": results.get('error_msg', ''),
                    "stage": 1
                }
            else:
                raise RuntimeError("Stage 1 results not found")
                
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return {
                "validity": 0.0,
                "combined_score": 0.0,
                "error": "Stage 1 timeout",
                "stage": 1
            }
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            if os.path.exists(results_path):
                os.unlink(results_path)
                
    except Exception as e:
        print(f"Stage 1 evaluation failed: {e}")
        traceback.print_exc()
        return {
            "validity": 0.0,
            "combined_score": 0.0,
            "error": str(e),
            "stage": 1
        }


def evaluate_stage2(program_path, dataset_path='ReacDiff_Nu0.5_Rho1.0.hdf5',
                   nu=0.5, rho=1.0):
    """
    Stage 2: Full evaluation on the complete dataset.
    """
    # Run full evaluation
    results = evaluate(program_path, dataset_path, nu, rho)
    results['stage'] = 2
    return results


# Example usage:
if __name__ == "__main__":
    # For testing, create a dummy solver
    test_solver_code = '''
import numpy as np

def solver(u0, t_coordinate, nu, rho):
    """
    Dummy solver for testing - replace with actual PDE solver.
    This should solve: ∂u/∂t = nu * ∂²u/∂x² + rho * u * (1 - u)
    """
    # Get dimensions
    batch_size, nx = u0.shape
    nt = len(t_coordinate)
    
    # Initialize solution array
    u = np.zeros((batch_size, nt, nx))
    u[:, 0, :] = u0
    
    # Simple forward Euler for demonstration
    dt = t_coordinate[1] - t_coordinate[0] if nt > 1 else 0.01
    dx = 2.0 / nx  # Domain is [-1, 1]
    
    for n in range(1, nt):
        for i in range(1, nx-1):
            # Diffusion term (centered difference)
            diff = nu * (u[:, n-1, i+1] - 2*u[:, n-1, i] + u[:, n-1, i-1]) / (dx**2)
            # Reaction term
            react = rho * u[:, n-1, i] * (1 - u[:, n-1, i])
            # Update
            u[:, n, i] = u[:, n-1, i] + dt * (diff + react)
        
        # Boundary conditions (Neumann)
        u[:, n, 0] = u[:, n, 1]
        u[:, n, -1] = u[:, n, -2]
    
    return u
'''
    
    # Save test solver to file
    with open('test_solver.py', 'w') as f:
        f.write(test_solver_code)
    
    # Test the evaluator
    print("Testing Stage 1...")
    stage1_results = evaluate_stage1('test_solver.py')
    print(f"Stage 1 results: {stage1_results}")
    
    if stage1_results['validity'] > 0:
        print("\nTesting Stage 2...")
        stage2_results = evaluate_stage2('test_solver.py')
        print(f"Stage 2 results: {stage2_results}")
    
    # Clean up
    os.unlink('test_solver.py')