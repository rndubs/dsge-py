"""
Minimal AR(1) Example

This is the simplest possible example to test the framework.
Model: x_t = ρ*x_{t-1} + ε_t, ε_t ~ N(0, σ²)
Observation: y_t = x_t + η_t, η_t ~ N(0, R)
"""

import numpy as np
import matplotlib.pyplot as plt
from dsge import (DSGEModel, ModelSpecification, Parameter, Prior,
                  solve_linear_model, kalman_filter, estimate_dsge)


class AR1Model(DSGEModel):
    """Simple AR(1) process as a minimal DSGE model example."""

    def __init__(self):
        """Initialize AR(1) model."""
        spec = ModelSpecification(
            n_states=1,  # x_t
            n_controls=0,  # no controls
            n_shocks=1,  # ε_t
            n_observables=1,  # y_t
            state_names=['x'],
            control_names=[],
            shock_names=['eps'],
            observable_names=['y']
        )
        super().__init__(spec)

    def _setup_parameters(self):
        """Define model parameters."""
        self.parameters.add(Parameter(
            name='rho',
            value=0.9,
            description='AR coefficient',
            fixed=False,
            bounds=(-0.999, 0.999),
            prior=Prior('beta', {'alpha': 18, 'beta': 2})  # mean = 0.9
        ))

        self.parameters.add(Parameter(
            name='sigma',
            value=0.1,
            description='Shock std dev',
            fixed=False,
            bounds=(0.01, 1.0),
            prior=Prior('invgamma', {'shape': 2.0, 'scale': 0.2})
        ))

    def system_matrices(self, params=None):
        """
        Construct system matrices for AR(1):
        x_t = ρ*x_{t-1} + ε_t
        """
        if params is not None:
            self.parameters.set_values(params)

        ρ = self.parameters['rho']

        # System: Γ0*s_t = Γ1*s_{t-1} + Ψ*ε_t + Π*η_t
        # x_t = ρ*x_{t-1} + ε_t
        # => 1*x_t = ρ*x_{t-1} + 1*ε_t + 0*η_t

        Γ0 = np.array([[1.0]])
        Γ1 = np.array([[ρ]])
        Ψ = np.array([[1.0]])
        Π = np.array([[1e-10]])  # Small for numerical stability

        return {
            'Gamma0': Γ0,
            'Gamma1': Γ1,
            'Psi': Ψ,
            'Pi': Π
        }

    def measurement_equation(self, params=None):
        """Observe x_t directly: y_t = x_t."""
        Z = np.array([[1.0]])
        D = np.array([0.0])
        return Z, D

    def shock_covariance(self, params=None):
        """Shock variance."""
        if params is not None:
            self.parameters.set_values(params)
        σ = self.parameters['sigma']
        return np.array([[σ ** 2]])

    def measurement_error_covariance(self, params=None):
        """Small measurement error."""
        return np.array([[1e-6]])


def main():
    """Run minimal AR(1) example."""
    print("=" * 60)
    print("Minimal AR(1) Model Example")
    print("=" * 60)

    # Create model
    model = AR1Model()
    print("\nModel created with parameters:")
    for param in model.parameters:
        print(f"  {param.name}: {param.value}")

    # Validate model
    print("\nValidating model...")
    if model.validate():
        print("✓ Model is valid")
    else:
        print("✗ Model validation failed")
        return

    # Solve model
    print("\nSolving model...")
    system_mats = model.system_matrices()
    solution, info = solve_linear_model(
        system_mats['Gamma0'],
        system_mats['Gamma1'],
        system_mats['Psi'],
        system_mats['Pi'],
        model.spec.n_states
    )

    print(f"  Solution status: {info['condition']}")
    print(f"  Eigenvalue: {info['eigenvalues'][0]:.4f}")
    print(f"  Is stable: {info['is_stable']}")

    if not solution.is_stable:
        print("✗ Model solution is not stable")
        return

    print(f"✓ Model solved successfully")
    print(f"  Transition matrix T = {solution.T[0,0]:.4f} (true ρ = {model.parameters['rho']:.4f})")

    # Simulate data
    print("\nSimulating data...")
    from dsge.solvers.linear import simulate

    Z, D = model.measurement_equation()
    solution.Z = Z
    solution.D = D
    solution.Q = model.shock_covariance()

    T_sim = 300
    states, obs = simulate(solution, T_sim, random_seed=42)

    print(f"  Simulated {T_sim} periods")
    print(f"  Observed data mean: {obs.mean():.4f}, std: {obs.std():.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(obs)
    ax.set_title('Simulated AR(1) Process')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('examples/ar1_simulation.png', dpi=150)
    print("✓ Plot saved to examples/ar1_simulation.png")

    # Evaluate likelihood
    print("\nEvaluating likelihood...")
    from dsge.estimation import log_likelihood_linear

    log_lik = log_likelihood_linear(model, obs)
    print(f"  Log likelihood: {log_lik:.2f}")

    # Estimation test
    print("\nRunning parameter estimation...")
    print("  (Using small sample for speed)")

    test_data = obs[:100]

    try:
        results = estimate_dsge(
            model,
            test_data,
            n_particles=200,
            n_mh_steps=1,
            verbose=True
        )

        print(f"\n✓ Estimation completed:")
        print(f"  Log evidence: {results.log_evidence:.2f}")

        # Posterior statistics
        posterior_mean = np.average(results.particles, weights=results.weights, axis=0)
        posterior_std = np.sqrt(np.average((results.particles - posterior_mean)**2,
                                          weights=results.weights, axis=0))

        free_params = list(model.parameters.get_free_params().keys())
        print(f"\n  Posterior estimates:")
        for i, name in enumerate(free_params):
            true_val = model.parameters[name]
            est_mean = posterior_mean[i]
            est_std = posterior_std[i]
            print(f"    {name}: {est_mean:.4f} ± {est_std:.4f} (true: {true_val:.4f})")

    except Exception as e:
        print(f"  Error in estimation: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
