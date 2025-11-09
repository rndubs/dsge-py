"""
NYFed Model 1002 - Validation Notebook with Visualizations

This script creates comprehensive plots for model validation including:
- Impulse response functions for all major shocks
- Eigenvalue distribution
- Simulation paths
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.nyfed_model_1002 import create_nyfed_model
from dsge.solvers.linear import solve_linear_model


def plot_eigenvalues(eigvals, save_path=None):
    """Plot eigenvalue distribution in complex plane."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Complex plane
    ax1.scatter(eigvals.real, eigvals.imag, alpha=0.7, s=50)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'r--', linewidth=2, label='Unit circle')

    ax1.set_xlabel('Real Part', fontsize=12)
    ax1.set_ylabel('Imaginary Part', fontsize=12)
    ax1.set_title('Eigenvalues in Complex Plane', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Magnitude distribution
    abs_eigvals = np.abs(eigvals)
    ax2.hist(abs_eigvals, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(x=1.0, color='r', linestyle='--', linewidth=2, label='Unit circle')
    ax2.set_xlabel('Magnitude |λ|', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Eigenvalue Magnitudes', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved eigenvalue plot to {save_path}")

    return fig


def plot_irfs(irfs, state_idx, vars_to_plot=None, save_path=None):
    """Plot impulse response functions for multiple shocks and variables."""

    if vars_to_plot is None:
        vars_to_plot = ['y', 'c', 'i', 'L', 'pi', 'R', 'w', 'r_k']

    n_vars = len(vars_to_plot)
    n_shocks = len(irfs)

    fig, axes = plt.subplots(n_vars, n_shocks, figsize=(5*n_shocks, 3*n_vars))

    if n_shocks == 1:
        axes = axes.reshape(-1, 1)
    if n_vars == 1:
        axes = axes.reshape(1, -1)

    shock_names = list(irfs.keys())
    H = irfs[shock_names[0]].shape[0]
    quarters = np.arange(H)

    for j, shock_name in enumerate(shock_names):
        irf_data = irfs[shock_name]

        for i, var in enumerate(vars_to_plot):
            ax = axes[i, j]

            if var in state_idx:
                response = irf_data[:, state_idx[var]]
                ax.plot(quarters, response, linewidth=2)
                ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                ax.grid(True, alpha=0.3)

                # Labels
                if i == 0:
                    ax.set_title(f'{shock_name}', fontsize=12, fontweight='bold')
                if j == 0:
                    ax.set_ylabel(f'{var}', fontsize=11)
                if i == n_vars - 1:
                    ax.set_xlabel('Quarters', fontsize=10)

    plt.suptitle('Impulse Response Functions - NYFed Model 1002',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved IRF plot to {save_path}")

    return fig


def plot_simulations(states, state_idx, vars_to_plot=None, save_path=None):
    """Plot simulation paths for key variables."""

    if vars_to_plot is None:
        vars_to_plot = ['y', 'c', 'i', 'pi', 'R', 'L']

    n_vars = len(vars_to_plot)
    T = states.shape[0]
    quarters = np.arange(T)

    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 2.5*n_vars))

    if n_vars == 1:
        axes = [axes]

    for i, var in enumerate(vars_to_plot):
        ax = axes[i]

        if var in state_idx:
            path = states[:, state_idx[var]]
            ax.plot(quarters, path, linewidth=1.5)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3)
            ax.set_ylabel(f'{var}', fontsize=11)

            if i == n_vars - 1:
                ax.set_xlabel('Quarters', fontsize=10)

    plt.suptitle('Simulated Paths - NYFed Model 1002',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved simulation plot to {save_path}")

    return fig


def create_validation_report(solution, model, info, save_dir='validation'):
    """Create comprehensive validation report with all plots."""

    print("\n" + "="*80)
    print("Creating Validation Plots")
    print("="*80)

    # Create output directory
    os.makedirs(save_dir, exist_ok=True)

    # 1. Plot eigenvalues
    print("\n1. Plotting eigenvalue distribution...")
    eigvals = info['eigenvalues']
    fig_eig = plot_eigenvalues(eigvals, save_path=f'{save_dir}/eigenvalues.png')
    plt.close(fig_eig)

    # 2. Compute and plot IRFs
    print("\n2. Computing impulse response functions...")
    shocks_to_analyze = ['eps_rm', 'eps_z', 'eps_b', 'eps_mu', 'eps_g']
    vars_to_plot = ['y', 'c', 'i', 'L', 'pi', 'R']

    H = 40
    n_states = model.spec.n_states
    n_shocks = model.spec.n_shocks

    state_idx = {name: i for i, name in enumerate(model.spec.state_names)}
    shock_idx = {name: i for i, name in enumerate(model.spec.shock_names)}

    irfs = {}
    for shock_name in shocks_to_analyze:
        if shock_name not in shock_idx:
            continue

        irf = np.zeros((H, n_states))
        shock = np.zeros(n_shocks)
        shock[shock_idx[shock_name]] = 1.0

        irf[0] = solution.R @ shock
        for h in range(1, H):
            irf[h] = solution.T @ irf[h-1]

        irfs[shock_name] = irf

    print("\n3. Plotting impulse response functions...")
    fig_irf = plot_irfs(irfs, state_idx, vars_to_plot=vars_to_plot,
                        save_path=f'{save_dir}/irfs.png')
    plt.close(fig_irf)

    # 3. Run and plot simulation
    print("\n4. Running simulation...")
    T = 200
    states = np.zeros((T, n_states))
    shocks = np.random.randn(T, n_shocks) * 0.01

    for t in range(1, T):
        states[t] = solution.C + solution.T @ states[t-1] + solution.R @ shocks[t]

    print("\n5. Plotting simulation paths...")
    fig_sim = plot_simulations(states, state_idx, vars_to_plot=vars_to_plot,
                               save_path=f'{save_dir}/simulation.png')
    plt.close(fig_sim)

    print("\n" + "="*80)
    print("Validation plots saved to:", save_dir)
    print("="*80)

    return irfs, states


def main():
    """Main validation routine with visualizations."""

    print("="*80)
    print("NYFed Model 1002 - Validation with Visualizations")
    print("="*80)

    # Create and solve model
    print("\nCreating and solving NYFed model...")
    model = create_nyfed_model()
    mats = model.system_matrices()

    solution, info = solve_linear_model(
        Gamma0=mats['Gamma0'],
        Gamma1=mats['Gamma1'],
        Psi=mats['Psi'],
        Pi=mats['Pi'],
        n_states=model.spec.n_states
    )

    print(f"\n✓ Model solved successfully")
    print(f"  Max eigenvalue: {np.max(np.abs(info['eigenvalues'])):.6f}")

    # Create validation report
    irfs, states = create_validation_report(solution, model, info)

    print("\n✅ Validation complete! Check the validation/ directory for plots.")

    return solution, model, info, irfs, states


if __name__ == "__main__":
    solution, model, info, irfs, states = main()
