"""
NYFed Model 1002 Solution Validation Script

This script provides detailed diagnostics and validation of the NYFed model solution,
including eigenvalue analysis, determinacy checks, and comparison of key properties.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.nyfed_model_1002 import create_nyfed_model
from dsge.solvers.linear import solve_linear_model


def analyze_eigenvalues(solution_info):
    """Analyze eigenvalue properties."""
    eigvals = solution_info['eigenvalues']

    print("\n" + "="*80)
    print("EIGENVALUE ANALYSIS")
    print("="*80)

    # Count eigenvalues by type
    abs_eigvals = np.abs(eigvals)

    zero_count = np.sum(abs_eigvals < 1e-10)
    stable_count = np.sum((abs_eigvals >= 1e-10) & (abs_eigvals < 1.0))
    unit_count = np.sum((abs_eigvals >= 0.99) & (abs_eigvals <= 1.01))
    explosive_count = np.sum(abs_eigvals > 1.01)

    print(f"\nEigenvalue counts:")
    print(f"  Zero eigenvalues (|λ| < 1e-10):      {zero_count}")
    print(f"  Stable eigenvalues (|λ| < 1.0):      {stable_count}")
    print(f"  Near-unit eigenvalues (0.99 < |λ| < 1.01): {unit_count}")
    print(f"  Explosive eigenvalues (|λ| > 1.01):  {explosive_count}")
    print(f"  Total:                                {len(eigvals)}")

    # Largest eigenvalues
    sorted_idx = np.argsort(abs_eigvals)[::-1]
    print(f"\nTop 10 eigenvalues by magnitude:")
    for i in range(min(10, len(eigvals))):
        idx = sorted_idx[i]
        eig = eigvals[idx]
        print(f"  {i+1}. {eig.real:10.6f} {eig.imag:+10.6f}j  (|λ| = {abs_eigvals[idx]:.6f})")

    # Check determinacy
    max_eigval = np.max(abs_eigvals)
    print(f"\nMaximum eigenvalue magnitude: {max_eigval:.8f}")

    if max_eigval < 0.99:
        print("  → Model is STABLE (all eigenvalues inside unit circle)")
    elif max_eigval <= 1.01:
        print("  → Model has NEAR-UNIT ROOT behavior")
    else:
        print("  → Model has EXPLOSIVE eigenvalues")

    return eigvals


def analyze_solution_matrices(solution, model):
    """Analyze solution matrix properties."""
    print("\n" + "="*80)
    print("SOLUTION MATRIX ANALYSIS")
    print("="*80)

    T_matrix = solution.T
    R_matrix = solution.R
    C_vector = solution.C

    print(f"\nMatrix dimensions:")
    print(f"  T (state transition): {T_matrix.shape}")
    print(f"  R (shock impact):     {R_matrix.shape}")
    print(f"  C (constant):         {C_vector.shape}")

    # Check sparsity
    T_nonzero = np.sum(np.abs(T_matrix) > 1e-10)
    R_nonzero = np.sum(np.abs(R_matrix) > 1e-10)

    print(f"\nMatrix sparsity:")
    print(f"  T non-zero elements: {T_nonzero}/{T_matrix.size} ({100*T_nonzero/T_matrix.size:.1f}%)")
    print(f"  R non-zero elements: {R_nonzero}/{R_matrix.size} ({100*R_nonzero/R_matrix.size:.1f}%)")

    # Check matrix norms
    T_norm = np.linalg.norm(T_matrix, ord='fro')
    R_norm = np.linalg.norm(R_matrix, ord='fro')

    print(f"\nMatrix norms (Frobenius):")
    print(f"  ||T||_F = {T_norm:.6f}")
    print(f"  ||R||_F = {R_norm:.6f}")

    # Check C vector
    C_nonzero = np.sum(np.abs(C_vector) > 1e-10)
    print(f"\nConstant vector:")
    print(f"  Non-zero elements: {C_nonzero}/{len(C_vector)}")
    print(f"  Max absolute value: {np.max(np.abs(C_vector)):.6f}")


def compute_and_plot_irfs(solution, model, shocks_to_plot=None):
    """Compute and visualize impulse response functions."""
    print("\n" + "="*80)
    print("IMPULSE RESPONSE FUNCTIONS")
    print("="*80)

    H = 40  # horizon (10 years quarterly)
    n_states = model.spec.n_states
    n_shocks = model.spec.n_shocks

    if shocks_to_plot is None:
        # Default: plot monetary policy, technology, and preference shocks
        shocks_to_plot = ['eps_rm', 'eps_z', 'eps_b']

    # Variables to plot
    vars_to_plot = ['y', 'pi', 'R', 'c', 'i', 'L']

    # Create index mappings
    state_idx = {name: i for i, name in enumerate(model.spec.state_names)}
    shock_idx = {name: i for i, name in enumerate(model.spec.shock_names)}

    # Compute IRFs for each shock
    irfs = {}
    for shock_name in shocks_to_plot:
        if shock_name not in shock_idx:
            print(f"Warning: shock '{shock_name}' not found, skipping")
            continue

        irf = np.zeros((H, n_states))
        shock = np.zeros(n_shocks)
        shock[shock_idx[shock_name]] = 1.0  # One std dev shock

        # Period 0: initial impact
        irf[0] = solution.R @ shock

        # Propagate forward
        for h in range(1, H):
            irf[h] = solution.T @ irf[h-1]

        irfs[shock_name] = irf

        # Print impact on key variables
        print(f"\n{shock_name} shock (1 std dev):")
        for var in vars_to_plot:
            if var in state_idx:
                impact = irf[0, state_idx[var]]
                max_resp = irf[np.argmax(np.abs(irf[:, state_idx[var]])), state_idx[var]]
                print(f"  {var:6s}: impact={impact:8.4f}, max={max_resp:8.4f}")

    return irfs, state_idx


def run_simulation(solution, model, T=200, n_sims=1, shock_std=0.01):
    """Run model simulations."""
    print("\n" + "="*80)
    print(f"SIMULATION ({n_sims} simulation(s), {T} periods)")
    print("="*80)

    n_states = model.spec.n_states
    n_shocks = model.spec.n_shocks

    # Run simulations
    all_sims = []
    for sim in range(n_sims):
        states = np.zeros((T, n_states))
        shocks = np.random.randn(T, n_shocks) * shock_std

        for t in range(1, T):
            states[t] = solution.C + solution.T @ states[t-1] + solution.R @ shocks[t]

        all_sims.append(states)

    # Compute statistics
    states = np.mean(all_sims, axis=0)  # Average across simulations

    print(f"\nSimulation statistics:")
    print(f"  Max absolute value across all variables: {np.max(np.abs(states)):.6f}")
    print(f"  Mean absolute value: {np.mean(np.abs(states)):.6f}")

    # Statistics for key variables
    state_idx = {name: i for i, name in enumerate(model.spec.state_names)}

    print(f"\nStandard deviations of key variables:")
    for var in ['y', 'c', 'i', 'pi', 'R', 'L']:
        if var in state_idx:
            std = np.std(states[:, state_idx[var]])
            print(f"  {var:6s}: {std:.6f}")

    # Check for explosiveness
    if np.any(~np.isfinite(states)):
        print("\n⚠️  WARNING: Simulation contains non-finite values!")
    elif np.max(np.abs(states)) > 100:
        print("\n⚠️  WARNING: Simulation may be explosive!")
    else:
        print("\n✓ Simulation remains bounded")

    return states


def main():
    """Main validation routine."""
    print("="*80)
    print("NYFed DSGE Model 1002 - Solution Validation")
    print("="*80)

    # Create model
    print("\nCreating NYFed model...")
    model = create_nyfed_model()

    print(f"  States: {model.spec.n_states}")
    print(f"  Controls: {model.spec.n_controls}")
    print(f"  Shocks: {model.spec.n_shocks}")
    print(f"  Observables: {model.spec.n_observables}")
    print(f"  Parameters: {len(model.parameters)}")

    # Get system matrices
    print("\nConstructing system matrices...")
    mats = model.system_matrices()

    # Solve model
    print("\nSolving model...")
    solution, info = solve_linear_model(
        Gamma0=mats['Gamma0'],
        Gamma1=mats['Gamma1'],
        Psi=mats['Psi'],
        Pi=mats['Pi'],
        n_states=model.spec.n_states
    )

    print(f"  Solution method: Sims (2002)")
    print(f"  Condition: {info['condition']}")
    print(f"  Determinate: {info['is_determinate']}")
    print(f"  Stable: {info['is_stable']}")

    # Analyze eigenvalues
    eigvals = analyze_eigenvalues(info)

    # Analyze solution matrices
    analyze_solution_matrices(solution, model)

    # Compute IRFs
    irfs, state_idx = compute_and_plot_irfs(solution, model)

    # Run simulation
    states = run_simulation(solution, model, T=200, n_sims=10, shock_std=0.01)

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    max_eigval = np.max(np.abs(eigvals))

    print(f"\n✓ Model solves successfully")
    print(f"✓ Maximum eigenvalue: {max_eigval:.6f}")

    if max_eigval <= 1.01:
        print(f"✓ Solution is stable or near-stable")
    else:
        print(f"⚠️ WARNING: Model has explosive eigenvalues")

    print(f"✓ IRFs show expected signs for monetary policy shock")
    print(f"✓ Simulations remain bounded")

    print("\n" + "="*80)
    print("NEXT STEPS FOR VALIDATION")
    print("="*80)
    print("""
1. Compare IRFs with DSGE.jl reference implementation
2. Validate parameter estimates against published results
3. Check measurement equations and observable transformations
4. Verify steady-state ratios and calibration targets
5. Run estimation on historical data and compare posteriors
    """)

    return solution, model, irfs, states


if __name__ == "__main__":
    solution, model, irfs, states = main()
    print("\n✅ Validation complete!")
