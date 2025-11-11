"""
Simple Real Business Cycle (RBC) Model Example.

This example demonstrates the basic usage of the DSGE framework
with a simple RBC model featuring:
- Capital accumulation
- Technology shocks
- Labor-leisure choice
"""

import matplotlib.pyplot as plt
import numpy as np

from dsge import (
    DSGEModel,
    ModelSpecification,
    Parameter,
    Prior,
    estimate_dsge,
    solve_linear_model,
)


class SimpleRBC(DSGEModel):
    """
    A simple RBC model with capital and labor.

    Log-linearized equilibrium conditions:
    1. Resource constraint: y_t = α*k_{t-1} + (1-α)*n_t + z_t
    2. Labor supply: w_t = ψ*n_t + σ*c_t
    3. Capital Euler: c_t = E_t[c_{t+1}] + r_{t+1}
    4. Production: y_t = α*k_{t-1} + (1-α)*n_t + z_t
    5. Wage: w_t = y_t - n_t
    6. Rental rate: r_t = y_t - k_{t-1}
    7. Capital accumulation: k_t = (1-δ)*k_{t-1} + δ*y_t
    8. Technology: z_t = ρ*z_{t-1} + ε_t

    Variables (in log-deviations from steady state):
    States: k_t (capital), z_t (technology)
    Controls: c_t (consumption), n_t (labor), y_t (output), w_t (wage), r_t (rental)
    Shocks: ε_t (technology shock)
    """

    def __init__(self) -> None:
        """Initialize RBC model."""
        spec = ModelSpecification(
            n_states=2,  # k, z
            n_controls=5,  # c, n, y, w, r
            n_shocks=1,  # technology shock
            n_observables=2,  # observe output and labor
            state_names=["k", "z"],
            control_names=["c", "n", "y", "w", "r"],
            shock_names=["eps_z"],
            observable_names=["y_obs", "n_obs"],
        )
        super().__init__(spec)

    def _setup_parameters(self) -> None:
        """Define model parameters."""
        # Structural parameters
        self.parameters.add(
            Parameter(
                name="alpha",
                value=0.33,
                description="Capital share",
                fixed=False,
                bounds=(0.01, 0.99),
                prior=Prior("beta", {"alpha": 3.3, "beta": 6.7}),  # mean = 0.33
            )
        )

        self.parameters.add(
            Parameter(
                name="beta",
                value=0.99,
                description="Discount factor",
                fixed=True,
                prior=Prior("beta", {"alpha": 99, "beta": 1}),  # mean = 0.99
            )
        )

        self.parameters.add(
            Parameter(name="delta", value=0.025, description="Depreciation rate", fixed=True)
        )

        self.parameters.add(
            Parameter(
                name="psi",
                value=2.0,
                description="Labor disutility",
                fixed=False,
                bounds=(0.1, 10.0),
                prior=Prior("gamma", {"shape": 4.0, "rate": 2.0}),  # mean = 2.0
            )
        )

        self.parameters.add(
            Parameter(
                name="sigma",
                value=1.0,
                description="Risk aversion",
                fixed=True,
                prior=Prior("gamma", {"shape": 1.0, "rate": 1.0}),
            )
        )

        self.parameters.add(
            Parameter(
                name="rho_z",
                value=0.95,
                description="Technology persistence",
                fixed=False,
                bounds=(-0.999, 0.999),
                prior=Prior("beta", {"alpha": 19, "beta": 1}),  # mean = 0.95
            )
        )

        self.parameters.add(
            Parameter(
                name="sigma_z",
                value=0.01,
                description="Technology shock std",
                fixed=False,
                bounds=(0.001, 0.1),
                prior=Prior("invgamma", {"shape": 2.0, "scale": 0.02}),
            )
        )

    def system_matrices(self, params=None):
        """
        Construct linearized system matrices.

        System: Γ0*s_t = Γ1*s_{t-1} + Ψ*ε_t + Π*η_t
        where s_t = [k_t, z_t, c_t, n_t, y_t, w_t, r_t]

        This is a simplified RBC model with static equations.
        """
        if params is not None:
            self.parameters.set_values(params)

        # Get parameter values
        α = self.parameters["alpha"]
        self.parameters["beta"]
        δ = self.parameters["delta"]
        ψ = self.parameters["psi"]
        σ = self.parameters["sigma"]
        ρ = self.parameters["rho_z"]

        n_total = self.spec.n_states + self.spec.n_controls  # 7

        Γ0 = np.zeros((n_total, n_total))
        Γ1 = np.zeros((n_total, n_total))
        Ψ = np.zeros((n_total, self.spec.n_shocks))
        Π = np.zeros((n_total, n_total))

        # Variables: [k, z, c, n, y, w, r]
        # Indices:    [0, 1, 2, 3, 4, 5, 6]

        # Equation 1: Capital law of motion (state)
        # k_t = (1-δ)*k_{t-1} + δ*(y_t - c_t)
        Γ0[0, 0] = 1  # k_t
        Γ1[0, 0] = -(1 - δ)  # k_{t-1}
        Γ0[0, 4] = -δ  # y_t (investment)
        Γ0[0, 2] = δ  # -c_t

        # Equation 2: Technology process (state)
        # z_t = ρ*z_{t-1} + ε_t
        Γ0[1, 1] = 1  # z_t
        Γ1[1, 1] = -ρ  # z_{t-1}
        Ψ[1, 0] = 1  # ε_t

        # Equation 3: Production function (static)
        # y_t = α*k_{t-1} + (1-α)*n_t + z_t
        Γ0[2, 4] = 1  # y_t
        Γ1[2, 0] = -α  # k_{t-1}
        Γ0[2, 3] = -(1 - α)  # n_t
        Γ0[2, 1] = -1  # z_t

        # Equation 4: Marginal product of labor = wage (static)
        # w_t = (1-α)*(y_t - n_t)
        Γ0[3, 5] = 1  # w_t
        Γ0[3, 4] = -(1 - α)  # y_t
        Γ0[3, 3] = 1 - α  # n_t

        # Equation 5: Labor supply (static)
        # w_t = ψ*n_t + σ*c_t
        Γ0[4, 5] = 1  # w_t
        Γ0[4, 3] = -ψ  # n_t
        Γ0[4, 2] = -σ  # c_t

        # Equation 6: Resource constraint (static)
        # c_t = (1-δ)*y_t - δ*k_{t-1}
        Γ0[5, 2] = 1  # c_t
        Γ0[5, 4] = -(1 - δ)  # y_t
        Γ1[5, 0] = -δ  # k_{t-1}

        # Equation 7: Marginal product of capital (static)
        # r_t = α*(y_t - k_{t-1})
        Γ0[6, 6] = 1  # r_t
        Γ0[6, 4] = -α  # y_t
        Γ1[6, 0] = α  # k_{t-1}

        # Expectational errors (none in this simple static version)
        Π = np.eye(n_total) * 1e-6  # Small value for numerical stability

        return {"Gamma0": Γ0, "Gamma1": Γ1, "Psi": Ψ, "Pi": Π}

    def measurement_equation(self, params=None):
        """
        Define measurement equation: observables = Z*states + D.

        We observe output and labor (with measurement error).
        """
        n_total = self.spec.n_states + self.spec.n_controls
        Z = np.zeros((self.spec.n_observables, n_total))

        # Observe output (y_t is index 4)
        Z[0, 4] = 1

        # Observe labor (n_t is index 3)
        Z[1, 3] = 1

        D = np.zeros(self.spec.n_observables)

        return Z, D

    def shock_covariance(self, params=None):
        """Technology shock variance."""
        if params is not None:
            self.parameters.set_values(params)

        σ_z = self.parameters["sigma_z"]
        return np.array([[σ_z**2]])

    def measurement_error_covariance(self, params=None):
        """Small measurement error."""
        return np.eye(self.spec.n_observables) * 1e-6


def main() -> None:
    """Run RBC model example."""
    # Create model
    model = SimpleRBC()
    for _param in model.parameters:
        pass

    # Validate model
    if model.validate():
        pass
    else:
        return

    # Solve model
    system_mats = model.system_matrices()
    solution, _info = solve_linear_model(
        system_mats["Gamma0"],
        system_mats["Gamma1"],
        system_mats["Psi"],
        system_mats["Pi"],
        model.spec.n_states,
    )

    if not solution.is_stable:
        return

    # Simulate data
    from dsge.solvers.linear import simulate

    # Update solution with measurement equation
    Z, D = model.measurement_equation()
    solution.Z = Z
    solution.D = D
    solution.Q = model.shock_covariance()

    T_sim = 200
    _states, obs = simulate(solution, T_sim, random_seed=42)

    # Plot simulated data
    _fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(obs[:, 0])
    axes[0].set_title("Simulated Output")
    axes[0].set_ylabel("Log deviation")
    axes[1].plot(obs[:, 1])
    axes[1].set_title("Simulated Labor")
    axes[1].set_ylabel("Log deviation")
    axes[1].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig("examples/rbc_simulation.png", dpi=150)

    # Likelihood evaluation
    from dsge.estimation import log_likelihood_linear

    log_likelihood_linear(model, obs)

    # Quick estimation test (small sample for speed)

    # Use a small sample for faster testing
    test_data = obs[:50]

    try:
        results = estimate_dsge(
            model,
            test_data,
            n_particles=100,  # Small for speed
            n_mh_steps=1,
            verbose=False,
        )

        # Compute posterior means
        posterior_mean = np.average(results.particles, weights=results.weights, axis=0)
        free_params = list(model.parameters.get_free_params().keys())

        for i, name in enumerate(free_params):
            model.parameters[name]
            posterior_mean[i]

    except Exception:
        pass


if __name__ == "__main__":
    main()
