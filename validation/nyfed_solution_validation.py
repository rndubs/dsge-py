"""
NYFed Model 1002 Solution Validation Script.

This script provides detailed diagnostics and validation of the NYFed model solution,
including eigenvalue analysis, determinacy checks, and comparison of key properties.
"""

import os
import sys

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dsge.solvers.linear import solve_linear_model
from models.nyfed_model_1002 import create_nyfed_model


def analyze_eigenvalues(solution_info):
    """Analyze eigenvalue properties."""
    eigvals = solution_info["eigenvalues"]


    # Count eigenvalues by type
    abs_eigvals = np.abs(eigvals)

    np.sum(abs_eigvals < 1e-10)
    np.sum((abs_eigvals >= 1e-10) & (abs_eigvals < 1.0))
    np.sum((abs_eigvals >= 0.99) & (abs_eigvals <= 1.01))
    np.sum(abs_eigvals > 1.01)


    # Largest eigenvalues
    sorted_idx = np.argsort(abs_eigvals)[::-1]
    for i in range(min(10, len(eigvals))):
        idx = sorted_idx[i]
        eigvals[idx]

    # Check determinacy
    max_eigval = np.max(abs_eigvals)

    if max_eigval < 0.99 or max_eigval <= 1.01:
        pass
    else:
        pass

    return eigvals


def analyze_solution_matrices(solution, model) -> None:
    """Analyze solution matrix properties."""
    T_matrix = solution.T
    R_matrix = solution.R
    C_vector = solution.C


    # Check sparsity
    np.sum(np.abs(T_matrix) > 1e-10)
    np.sum(np.abs(R_matrix) > 1e-10)


    # Check matrix norms
    np.linalg.norm(T_matrix, ord="fro")
    np.linalg.norm(R_matrix, ord="fro")


    # Check C vector
    np.sum(np.abs(C_vector) > 1e-10)


def compute_and_plot_irfs(solution, model, shocks_to_plot=None):
    """Compute and visualize impulse response functions."""
    H = 40  # horizon (10 years quarterly)
    n_states = model.spec.n_states
    n_shocks = model.spec.n_shocks

    if shocks_to_plot is None:
        # Default: plot monetary policy, technology, and preference shocks
        shocks_to_plot = ["eps_rm", "eps_z", "eps_b"]

    # Variables to plot
    vars_to_plot = ["y", "pi", "R", "c", "i", "L"]

    # Create index mappings
    state_idx = {name: i for i, name in enumerate(model.spec.state_names)}
    shock_idx = {name: i for i, name in enumerate(model.spec.shock_names)}

    # Compute IRFs for each shock
    irfs = {}
    for shock_name in shocks_to_plot:
        if shock_name not in shock_idx:
            continue

        irf = np.zeros((H, n_states))
        shock = np.zeros(n_shocks)
        shock[shock_idx[shock_name]] = 1.0  # One std dev shock

        # Period 0: initial impact
        irf[0] = solution.R @ shock

        # Propagate forward
        for h in range(1, H):
            irf[h] = solution.T @ irf[h - 1]

        irfs[shock_name] = irf

        # Print impact on key variables
        for var in vars_to_plot:
            if var in state_idx:
                irf[0, state_idx[var]]
                irf[np.argmax(np.abs(irf[:, state_idx[var]])), state_idx[var]]

    return irfs, state_idx


def run_simulation(solution, model, T=200, n_sims=1, shock_std=0.01):
    """Run model simulations."""
    n_states = model.spec.n_states
    n_shocks = model.spec.n_shocks

    # Run simulations
    all_sims = []
    for _sim in range(n_sims):
        states = np.zeros((T, n_states))
        shocks = np.random.randn(T, n_shocks) * shock_std

        for t in range(1, T):
            states[t] = solution.C + solution.T @ states[t - 1] + solution.R @ shocks[t]

        all_sims.append(states)

    # Compute statistics
    states = np.mean(all_sims, axis=0)  # Average across simulations


    # Statistics for key variables
    state_idx = {name: i for i, name in enumerate(model.spec.state_names)}

    for var in ["y", "c", "i", "pi", "R", "L"]:
        if var in state_idx:
            np.std(states[:, state_idx[var]])

    # Check for explosiveness
    if np.any(~np.isfinite(states)) or np.max(np.abs(states)) > 100:
        pass
    else:
        pass

    return states


def main():
    """Main validation routine."""
    # Create model
    model = create_nyfed_model()


    # Get system matrices
    mats = model.system_matrices()

    # Solve model
    solution, info = solve_linear_model(
        Gamma0=mats["Gamma0"],
        Gamma1=mats["Gamma1"],
        Psi=mats["Psi"],
        Pi=mats["Pi"],
        n_states=model.spec.n_states,
    )


    # Analyze eigenvalues
    eigvals = analyze_eigenvalues(info)

    # Analyze solution matrices
    analyze_solution_matrices(solution, model)

    # Compute IRFs
    irfs, _state_idx = compute_and_plot_irfs(solution, model)

    # Run simulation
    states = run_simulation(solution, model, T=200, n_sims=10, shock_std=0.01)

    # Summary

    max_eigval = np.max(np.abs(eigvals))


    if max_eigval <= 1.01:
        pass
    else:
        pass



    return solution, model, irfs, states


if __name__ == "__main__":
    solution, model, irfs, states = main()
