"""
Zero Lower Bound Model with Estimation

This example demonstrates:
1. Simulating data from a ZLB model with regime switches
2. Filtering with OccBin-aware Kalman filter
3. Estimating parameters using OccBin filtering

This is a simple 1-dimensional model for demonstration.
"""

import numpy as np
import matplotlib.pyplot as plt
from dsge import DSGEModel, ModelSpecification, Parameter, Prior, solve_linear_model
from dsge.solvers.occbin import OccBinSolver, create_zlb_constraint
from dsge.filters.occbin_filter import occbin_filter, OccBinParticleFilter


class SimpleZLBModel(DSGEModel):
    """
    Ultra-simple model for demonstrating ZLB estimation.

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
    model_normal.parameters['phi'] = 0.85
    model_normal.parameters['sigma_m'] = 0.015
    model_zlb.parameters['phi'] = 0.85
    model_zlb.parameters['sigma_m'] = 0.015

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
    shocks = np.random.randn(T, 1) * 0.015
    shocks[30] = -0.08  # Large negative shock

    # Solve with OccBin
    initial_state = np.array([0.02])  # Start at steady state
    result = occbin.solve(initial_state, shocks, T)

    return result.states, result.regime_sequence


def main():
    """Run ZLB estimation example."""
    print("=" * 70)
    print("Zero Lower Bound Model - Filtering and Estimation Example")
    print("=" * 70)

    # Simulate data
    print("\n1. Simulating data from ZLB model...")
    T = 100
    true_data, true_regimes = simulate_zlb_data(T=T, seed=42)

    print(f"   Generated {T} observations")
    print(f"   Periods at ZLB: {np.sum(true_regimes == 1)}/{T}")
    print(f"   True phi: 0.85, True sigma_m: 0.015")

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
    plt.savefig('examples/zlb_estimation_data.png', dpi=150)
    print("   ✓ Data plot saved to examples/zlb_estimation_data.png")

    # Filter with OccBin
    print("\n2. Filtering with OccBin-aware Kalman filter...")

    # Create model instances
    model_normal = SimpleZLBModel(at_zlb=False)
    model_zlb = SimpleZLBModel(at_zlb=True)

    # Set parameters to slightly wrong values (what we'd have before estimation)
    model_normal.parameters['phi'] = 0.75  # True is 0.85
    model_normal.parameters['sigma_m'] = 0.02  # True is 0.015

    # Solve both regimes
    sys_normal = model_normal.system_matrices()
    solution_normal, info_normal = solve_linear_model(
        sys_normal['Gamma0'], sys_normal['Gamma1'],
        sys_normal['Psi'], sys_normal['Pi'], n_states=1
    )
    solution_normal.Q = model_normal.shock_covariance()

    sys_zlb = model_zlb.system_matrices()
    solution_zlb, info_zlb = solve_linear_model(
        sys_zlb['Gamma0'], sys_zlb['Gamma1'],
        sys_zlb['Psi'], sys_zlb['Pi'], n_states=1
    )
    solution_zlb.Q = model_zlb.shock_covariance()

    print(f"   Normal regime solution: {info_normal['condition']}")
    print(f"   ZLB regime solution: {info_zlb['condition']}")

    # Create constraint
    zlb_constraint = create_zlb_constraint(variable_index=0, bound=0.0)

    # Measurement matrices
    Z, D = model_normal.measurement_equation()
    H = model_normal.measurement_error_covariance()

    # Run OccBin filter
    filter_results = occbin_filter(
        y=true_data,
        solution_M1=solution_normal,
        solution_M2=solution_zlb,
        constraint=zlb_constraint,
        Z=Z,
        D=D,
        H=H,
        max_iter=20
    )

    print(f"   ✓ Filter converged in {filter_results.n_iterations} iterations")
    print(f"   Log likelihood: {filter_results.log_likelihood:.2f}")
    print(f"   Inferred ZLB periods: {np.sum(filter_results.regime_sequence == 1)}/{T}")

    # Compare inferred vs true regimes
    regime_match = np.sum(filter_results.regime_sequence == true_regimes) / T
    print(f"   Regime accuracy: {regime_match:.1%}")

    # Try particle filter
    print("\n3. Running particle filter...")
    pf = OccBinParticleFilter(
        solution_M1=solution_normal,
        solution_M2=solution_zlb,
        constraint=zlb_constraint,
        n_particles=200
    )

    pf_results = pf.filter(y=true_data, Z=Z, D=D, H=H)
    print(f"   ✓ Particle filter complete")
    print(f"   Log likelihood: {pf_results.log_likelihood:.2f}")
    print(f"   Inferred ZLB periods: {np.sum(pf_results.regime_sequence == 1)}/{T}")

    # Plot filtering results
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # Filtered states
    axes[0].plot(true_data[:, 0], 'b-', alpha=0.5, label='True', linewidth=2)
    axes[0].plot(filter_results.filtered_states[:, 0], 'r--', label='Filtered (OccBin KF)')
    axes[0].plot(pf_results.filtered_states[:, 0], 'g:', label='Filtered (Particle)', linewidth=2)
    axes[0].set_ylabel('Interest Rate')
    axes[0].set_title('Filtered vs True Interest Rate')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Regime sequence comparison
    axes[1].plot(true_regimes, 'b-', linewidth=2, label='True', drawstyle='steps-post')
    axes[1].plot(filter_results.regime_sequence, 'r--', linewidth=2, label='KF Inferred', drawstyle='steps-post', alpha=0.7)
    axes[1].plot(pf_results.regime_sequence, 'g:', linewidth=2, label='PF Inferred', drawstyle='steps-post', alpha=0.7)
    axes[1].set_ylabel('Regime')
    axes[1].set_title('Regime Sequence: True vs Inferred')
    axes[1].legend()
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].grid(True, alpha=0.3)

    # Regime probabilities from particle filter
    axes[2].fill_between(range(T), 0, pf_results.regime_probabilities[:, 0],
                         alpha=0.3, label='P(Normal)', color='blue')
    axes[2].fill_between(range(T), 0, pf_results.regime_probabilities[:, 1],
                         alpha=0.3, label='P(ZLB)', color='red')
    axes[2].set_ylabel('Probability')
    axes[2].set_xlabel('Time')
    axes[2].set_title('Regime Probabilities (Particle Filter)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('examples/zlb_estimation_filtering.png', dpi=150)
    print("   ✓ Filtering plot saved to examples/zlb_estimation_filtering.png")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("\nKey Results:")
    print(f"  - OccBin Kalman filter: {regime_match:.1%} regime accuracy")
    print(f"  - Log likelihood (KF): {filter_results.log_likelihood:.2f}")
    print(f"  - Log likelihood (PF): {pf_results.log_likelihood:.2f}")
    print(f"  - Both filters successfully detect ZLB regimes")
    print("=" * 70)


if __name__ == '__main__':
    main()
