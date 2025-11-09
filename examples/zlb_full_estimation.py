"""
Zero Lower Bound Model - Full Bayesian Estimation Example

This example demonstrates the complete workflow:
1. Simulating data from a ZLB model
2. Filtering with OccBin
3. Estimating parameters using SMC
4. Analyzing posterior distributions

This is a demonstration of Phase 2.3: OccBin Estimation Integration
"""

import numpy as np
import matplotlib.pyplot as plt
from dsge import DSGEModel, ModelSpecification, Parameter, Prior, solve_linear_model
from dsge.solvers.occbin import OccBinSolver, create_zlb_constraint
from dsge.filters.occbin_filter import occbin_filter
from dsge.estimation.occbin_estimation import estimate_occbin


class SimpleZLBModel(DSGEModel):
    """
    Simple ZLB model for demonstration.

    Normal regime: i_t = φ*i_{t-1} + (1-φ)*r* + ε_t
    ZLB regime: i_t = 0

    where i is the nominal interest rate, r* is the natural rate,
    and φ is persistence.
    """

    def __init__(self, at_zlb: bool = False):
        """
        Initialize simple ZLB model.

        Parameters
        ----------
        at_zlb : bool
            Whether the model is in ZLB regime
        """
        spec = ModelSpecification(
            n_states=1,  # i (interest rate)
            n_controls=0,
            n_shocks=1,  # monetary shock
            n_observables=1,  # observe interest rate
            state_names=['i'],
            shock_names=['eps_m'],
            observable_names=['i_obs']
        )
        self.at_zlb = at_zlb
        super().__init__(spec)

    def _setup_parameters(self):
        """Define parameters."""
        self.parameters.add(Parameter(
            name='phi',
            value=0.8,
            description='Interest rate persistence',
            fixed=False,
            bounds=(0.0, 0.99),
            prior=Prior('beta', {'alpha': 16, 'beta': 4})  # mean = 0.8
        ))

        self.parameters.add(Parameter(
            name='r_star',
            value=0.02,
            description='Natural rate (steady state)',
            fixed=True
        ))

        self.parameters.add(Parameter(
            name='sigma_m',
            value=0.01,
            description='Monetary shock std',
            fixed=False,
            bounds=(0.001, 0.1),
            prior=Prior('invgamma', {'shape': 2.0, 'scale': 0.02})
        ))

    def system_matrices(self, params=None):
        """Construct system matrices."""
        if params is not None:
            self.parameters.set_values(params)

        φ = self.parameters['phi']
        r_star = self.parameters['r_star']

        if self.at_zlb:
            # ZLB: i_t = 0
            Γ0 = np.array([[1.0]])
            Γ1 = np.array([[0.0]])
            Ψ = np.array([[0.0]])
        else:
            # Normal: i_t = φ*i_{t-1} + (1-φ)*r* + ε_t
            # Rewrite in deviations from r*: (i_t - r*) = φ*(i_{t-1} - r*) + ε_t
            Γ0 = np.array([[1.0]])
            Γ1 = np.array([[φ]])
            Ψ = np.array([[1.0]])

        Π = np.array([[1e-10]])

        return {
            'Gamma0': Γ0,
            'Gamma1': Γ1,
            'Psi': Ψ,
            'Pi': Π
        }

    def measurement_equation(self, params=None):
        """Observe interest rate."""
        Z = np.eye(1)
        D = np.zeros(1)
        return Z, D

    def shock_covariance(self, params=None):
        """Monetary shock variance."""
        if params is not None:
            self.parameters.set_values(params)
        σ_m = self.parameters['sigma_m']
        return np.array([[σ_m ** 2]])

    def measurement_error_covariance(self, params=None):
        """Tiny measurement error."""
        return np.eye(1) * 1e-8


def simulate_zlb_data(T: int = 100, seed: int = 42) -> tuple:
    """
    Simulate data from ZLB model.

    Returns simulated interest rates and true regime sequence.
    """
    np.random.seed(seed)

    # Create models
    model_normal = SimpleZLBModel(at_zlb=False)
    model_zlb = SimpleZLBModel(at_zlb=True)

    # True parameters
    true_phi = 0.85
    true_sigma = 0.015
    model_normal.parameters['phi'] = true_phi
    model_normal.parameters['sigma_m'] = true_sigma
    model_zlb.parameters['phi'] = true_phi
    model_zlb.parameters['sigma_m'] = true_sigma

    # Solve both regimes
    sys_normal = model_normal.system_matrices()
    solution_normal, _ = solve_linear_model(
        sys_normal['Gamma0'], sys_normal['Gamma1'],
        sys_normal['Psi'], sys_normal['Pi'], n_states=1
    )

    sys_zlb = model_zlb.system_matrices()
    solution_zlb, _ = solve_linear_model(
        sys_zlb['Gamma0'], sys_zlb['Gamma1'],
        sys_zlb['Psi'], sys_zlb['Pi'], n_states=1
    )

    # Create OccBin solver
    zlb_constraint = create_zlb_constraint(variable_index=0, bound=0.0)
    occbin = OccBinSolver(solution_normal, solution_zlb, zlb_constraint)

    # Generate shocks - include a large negative shock to hit ZLB
    shocks = np.random.randn(T, 1) * true_sigma
    shocks[30] = -0.08  # Large negative shock

    # Solve with OccBin
    initial_state = np.array([0.02])  # Start at steady state
    result = occbin.solve(initial_state, shocks, T)

    return result.states, result.regime_sequence, true_phi, true_sigma


def main():
    """Run full ZLB estimation example."""
    print("=" * 70)
    print("ZLB Model - Full Bayesian Estimation with SMC")
    print("=" * 70)

    # Simulate data
    print("\n1. Simulating data from ZLB model...")
    T = 80
    true_data, true_regimes, true_phi, true_sigma = simulate_zlb_data(T=T, seed=42)

    print(f"   Generated {T} observations")
    print(f"   Periods at ZLB: {np.sum(true_regimes == 1)}/{T}")
    print(f"   True parameters:")
    print(f"     phi = {true_phi:.3f}")
    print(f"     sigma_m = {true_sigma:.4f}")

    # Plot simulated data
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].plot(true_data[:, 0], 'b-', linewidth=1.5)
    axes[0].set_ylabel('Interest Rate')
    axes[0].set_title('Simulated Interest Rate (with ZLB)')
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='ZLB')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Shade ZLB periods
    for t in range(T):
        if true_regimes[t] == 1:
            axes[0].axvspan(t-0.5, t+0.5, alpha=0.2, color='red')

    axes[1].plot(true_regimes, 'r-', linewidth=2, drawstyle='steps-post')
    axes[1].set_ylabel('Regime')
    axes[1].set_xlabel('Time')
    axes[1].set_title('True Regime Sequence (0=Normal, 1=ZLB)')
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('examples/zlb_full_estimation_data.png', dpi=150)
    print("   ✓ Data plot saved to examples/zlb_full_estimation_data.png")

    # Bayesian Estimation with SMC
    print("\n2. Bayesian Estimation using Sequential Monte Carlo...")

    # Create models for estimation
    model_M1 = SimpleZLBModel(at_zlb=False)
    model_M2 = SimpleZLBModel(at_zlb=True)

    # Create constraint
    constraint = create_zlb_constraint(variable_index=0, bound=0.0)

    # Run SMC estimation
    print("   Running SMC sampler...")
    print(f"   Using {200} particles, {2} MH steps per stage")

    results = estimate_occbin(
        model_M1=model_M1,
        model_M2=model_M2,
        constraint=constraint,
        data=true_data,
        n_particles=200,
        n_mh_steps=2,
        max_filter_iter=20,
        verbose=True
    )

    print(f"\n   ✓ Estimation complete!")
    print(f"   Log marginal likelihood: {results.log_evidence:.2f}")
    print(f"   Overall acceptance rate: {results.acceptance_rate:.1%}")
    print(f"   Number of tempering stages: {results.n_iterations}")

    # Analyze posterior
    print("\n3. Posterior Analysis...")

    # Compute posterior statistics
    posterior_mean = np.average(results.particles, weights=results.weights, axis=0)
    posterior_std = np.sqrt(np.average(
        (results.particles - posterior_mean)**2,
        weights=results.weights,
        axis=0
    ))

    phi_mean, sigma_mean = posterior_mean
    phi_std, sigma_std = posterior_std

    print(f"\n   Posterior Summary:")
    print(f"   -----------------")
    print(f"   phi:")
    print(f"     True value:       {true_phi:.3f}")
    print(f"     Posterior mean:   {phi_mean:.3f}")
    print(f"     Posterior std:    {phi_std:.3f}")
    print(f"     95% CI:          [{phi_mean - 1.96*phi_std:.3f}, {phi_mean + 1.96*phi_std:.3f}]")
    print(f"\n   sigma_m:")
    print(f"     True value:       {true_sigma:.4f}")
    print(f"     Posterior mean:   {sigma_mean:.4f}")
    print(f"     Posterior std:    {sigma_std:.4f}")
    print(f"     95% CI:          [{sigma_mean - 1.96*sigma_std:.4f}, {sigma_mean + 1.96*sigma_std:.4f}]")

    # Plot posterior distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Phi posterior
    axes[0].hist(results.particles[:, 0], weights=results.weights,
                 bins=30, density=True, alpha=0.6, color='blue', edgecolor='black')
    axes[0].axvline(true_phi, color='red', linestyle='--', linewidth=2, label='True Value')
    axes[0].axvline(phi_mean, color='green', linestyle='-', linewidth=2, label='Posterior Mean')
    axes[0].set_xlabel('φ (Persistence)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Posterior Distribution of φ')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Sigma posterior
    axes[1].hist(results.particles[:, 1], weights=results.weights,
                 bins=30, density=True, alpha=0.6, color='blue', edgecolor='black')
    axes[1].axvline(true_sigma, color='red', linestyle='--', linewidth=2, label='True Value')
    axes[1].axvline(sigma_mean, color='green', linestyle='-', linewidth=2, label='Posterior Mean')
    axes[1].set_xlabel('σ_m (Shock Std Dev)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Posterior Distribution of σ_m')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('examples/zlb_full_estimation_posterior.png', dpi=150)
    print("\n   ✓ Posterior plot saved to examples/zlb_full_estimation_posterior.png")

    # Joint distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(results.particles[:, 0], results.particles[:, 1],
                        c=results.weights, s=30, alpha=0.6, cmap='viridis')
    ax.scatter(true_phi, true_sigma, color='red', s=200, marker='*',
              edgecolor='black', linewidth=2, label='True Values', zorder=10)
    ax.scatter(phi_mean, sigma_mean, color='green', s=100, marker='o',
              edgecolor='black', linewidth=2, label='Posterior Mean', zorder=10)
    ax.set_xlabel('φ (Persistence)')
    ax.set_ylabel('σ_m (Shock Std Dev)')
    ax.set_title('Joint Posterior Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Particle Weight')

    plt.tight_layout()
    plt.savefig('examples/zlb_full_estimation_joint.png', dpi=150)
    print("   ✓ Joint posterior plot saved to examples/zlb_full_estimation_joint.png")

    print("\n" + "=" * 70)
    print("Full Estimation Example Completed Successfully!")
    print("\nKey Results:")
    print(f"  - Recovered phi: {phi_mean:.3f} (true: {true_phi:.3f})")
    print(f"  - Recovered sigma_m: {sigma_mean:.4f} (true: {true_sigma:.4f})")
    print(f"  - Log marginal likelihood: {results.log_evidence:.2f}")
    print(f"  - Acceptance rate: {results.acceptance_rate:.1%}")
    print("\nThis demonstrates successful OccBin estimation integration (Phase 2.3)!")
    print("=" * 70)


if __name__ == '__main__':
    main()
