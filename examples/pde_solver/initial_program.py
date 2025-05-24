# EVOLVE-BLOCK-START

import numpy as np
import torch

def crank_nicolson_matrix(N, dx, dt, nu):
    """Constructs the A and B matrices for the Crank-Nicolson method."""
    r = nu * dt / (2 * dx**2)
    # Identity matrix
    I = torch.eye(N, device='cuda')
    
    # Off-diagonal shifts (periodic boundary)
    off_diag = -r * torch.roll(I, shifts=1, dims=1)
    off_diag += -r * torch.roll(I, shifts=-1, dims=1)
    
    # A and B matrices
    A = (1 + 2*r) * I + off_diag
    B = (1 - 2*r) * I - off_diag
    return A, B

def apply_reaction_term(u, rho, dt):
    """Applies the reaction term using explicit Euler."""
    return u + dt * rho * u * (1 - u)

def solver(u0_batch, t_coordinate, nu, rho):
    """Solves the 1D reaction diffusion equation for all times in t_coordinate.

    Args:
        u0_batch (np.ndarray): Initial condition [batch_size, N], 
            where batch_size is the number of different initial conditions,
            and N is the number of spatial grid points.
        t_coordinate (np.ndarray): Time coordinates of shape [T+1]. 
            It begins with t_0=0 and follows the time steps t_1, ..., t_T.
        nu (float): The parameter nu in the equation.
        rho (float): The parameter rho in the equation.

    Returns:
        solutions (np.ndarray): Shape [batch_size, T+1, N].
            solutions[:, 0, :] contains the initial conditions (u0_batch),
            solutions[:, i, :] contains the solutions at time t_coordinate[i].
    """
    # Extract the dimensions
    batch_size, N = u0_batch.shape
    T = len(t_coordinate) - 1
    
    # Convert to torch tensors for GPU operations
    u0_batch = torch.tensor(u0_batch, dtype=torch.float32, device='cuda')
    
    # Spatial step
    dx = 1.0 / N
    
    # Internal time step for stability
    dt_internal = 0.1 * dx**2 / nu
    num_internal_steps = int(np.ceil((t_coordinate[1] - t_coordinate[0]) / dt_internal))
    dt_internal = (t_coordinate[1] - t_coordinate[0]) / num_internal_steps
    
    # Precompute Crank-Nicolson matrices
    A, B = crank_nicolson_matrix(N, dx, dt_internal, nu)
    A_inv = torch.linalg.inv(A).to('cuda')
    
    # Initialize solution array
    solutions = torch.zeros((batch_size, T+1, N), device='cuda')
    solutions[:, 0, :] = u0_batch
    
    # Time-stepping loop
    for t in range(1, T+1):
        u = solutions[:, t-1, :].clone()
        for _ in range(num_internal_steps):
            u = apply_reaction_term(u, rho, dt_internal)  # Apply reaction term
            u = torch.matmul(B, u.T).T  # Now u has shape (batch_size, N)
            u = torch.matmul(A_inv, u.T).T  # Apply diffusion term, maintaining correct shape
        solutions[:, t, :] = u
    
    # Move result back to CPU
    return solutions.cpu().numpy()

# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return solver
