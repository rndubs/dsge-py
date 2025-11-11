"""
Simple New Keynesian Model with Zero Lower Bound.

This example demonstrates OccBin with a minimal NK model:
- Taylor rule with ZLB constraint
- Cost-push shocks
- Two regimes: normal (ZLB slack) and ZLB binding
"""

import matplotlib.pyplot as plt
import numpy as np

from dsge import DSGEModel, ModelSpecification, Parameter, solve_linear_model
from dsge.solvers.occbin import OccBinSolver, create_zlb_constraint


class SimpleNKModel(DSGEModel):
    """
    Simple 3-equation New Keynesian model.

    Linearized equations:
    1. IS curve: x_t = E_t[x_{t+1}] - σ*(i_t - E_t[π_{t+1}]) + ε_d_t
    2. Phillips curve: π_t = β*E_t[π_{t+1}] + κ*x_t + ε_s_t
    3. Taylor rule: i_t = φ_π*π_t + φ_x*x_t + ε_m_t

    where x = output gap, π = inflation, i = nominal interest rate
    """

    def __init__(self, zlb_binding: bool = False) -> None:
        """
        Initialize NK model.

        Parameters
        ----------
        zlb_binding : bool
            If True, impose ZLB constraint (i >= 0)
        """
        spec = ModelSpecification(
            n_states=3,  # x, π, i
            n_controls=0,  # All variables are states in simple version
            n_shocks=3,  # demand shock, supply shock, monetary shock
            n_observables=3,  # observe all variables
            state_names=["x", "pi", "i"],
            control_names=[],
            shock_names=["eps_d", "eps_s", "eps_m"],
            observable_names=["x_obs", "pi_obs", "i_obs"],
        )
        self.zlb_binding = zlb_binding
        super().__init__(spec)

    def _setup_parameters(self) -> None:
        """Define model parameters."""
        # Structural parameters
        self.parameters.add(
            Parameter(
                name="sigma",
                value=1.0,
                description="Risk aversion / intertemporal elasticity",
                fixed=True,
            )
        )

        self.parameters.add(
            Parameter(name="beta", value=0.99, description="Discount factor", fixed=True)
        )

        self.parameters.add(
            Parameter(name="kappa", value=0.1, description="Slope of Phillips curve", fixed=True)
        )

        self.parameters.add(
            Parameter(
                name="phi_pi",
                value=1.5 if not self.zlb_binding else 0.0,  # No response at ZLB
                description="Taylor rule inflation response",
                fixed=True,
            )
        )

        self.parameters.add(
            Parameter(
                name="phi_x",
                value=0.5 if not self.zlb_binding else 0.0,  # No response at ZLB
                description="Taylor rule output response",
                fixed=True,
            )
        )

        # Shock persistence (simplified - assume iid)
        for shock_name in ["rho_d", "rho_s", "rho_m"]:
            self.parameters.add(
                Parameter(
                    name=shock_name,
                    value=0.0,  # IID shocks
                    description=f"Persistence of {shock_name}",
                    fixed=True,
                )
            )

        # Shock standard deviations
        self.parameters.add(
            Parameter(name="sigma_d", value=0.01, description="Demand shock std", fixed=True)
        )

        self.parameters.add(
            Parameter(name="sigma_s", value=0.01, description="Supply shock std", fixed=True)
        )

        self.parameters.add(
            Parameter(name="sigma_m", value=0.01, description="Monetary shock std", fixed=True)
        )

    def system_matrices(self, params=None):
        """
        Construct linearized system matrices.

        For the NK model with forward-looking variables, we use:
        Γ0*s_t = Γ1*s_{t-1} + Ψ*ε_t + Π*η_t

        where η_t are expectational errors.
        """
        if params is not None:
            self.parameters.set_values(params)

        σ = self.parameters["sigma"]
        β = self.parameters["beta"]
        κ = self.parameters["kappa"]
        φ_π = self.parameters["phi_pi"]
        φ_x = self.parameters["phi_x"]

        n_total = 3  # x, π, i

        Γ0 = np.zeros((n_total, n_total))
        Γ1 = np.zeros((n_total, n_total))
        Ψ = np.zeros((n_total, self.spec.n_shocks))
        Π = np.zeros((n_total, n_total))

        # Variables: [x, π, i]
        # Indices:    [0, 1, 2]

        # Equation 1: IS curve
        # x_t = E_t[x_{t+1}] - σ*(i_t - E_t[π_{t+1}]) + ε_d_t
        # => x_t - E_t[x_{t+1}] + σ*i_t - σ*E_t[π_{t+1}] = ε_d_t
        # => x_t + σ*i_t = E_t[x_{t+1}] + σ*E_t[π_{t+1}] + ε_d_t
        Γ0[0, 0] = 1  # x_t
        Γ0[0, 2] = σ  # i_t
        Π[0, 0] = -1  # -η_x (expectational error for x)
        Π[0, 1] = -σ  # -σ*η_π
        Ψ[0, 0] = 1  # ε_d_t

        # Equation 2: Phillips curve
        # π_t = β*E_t[π_{t+1}] + κ*x_t + ε_s_t
        # => π_t - κ*x_t = β*E_t[π_{t+1}] + ε_s_t
        Γ0[1, 1] = 1  # π_t
        Γ0[1, 0] = -κ  # -κ*x_t
        Π[1, 1] = -β  # -β*η_π
        Ψ[1, 1] = 1  # ε_s_t

        # Equation 3: Taylor rule (or ZLB constraint)
        if self.zlb_binding:
            # At ZLB: i_t = 0
            Γ0[2, 2] = 1
            # Γ1, Π, Ψ all zero for this equation
        else:
            # Taylor rule: i_t = φ_π*π_t + φ_x*x_t + ε_m_t
            Γ0[2, 2] = 1  # i_t
            Γ0[2, 1] = -φ_π  # -φ_π*π_t
            Γ0[2, 0] = -φ_x  # -φ_x*x_t
            Ψ[2, 2] = 1  # ε_m_t

        return {"Gamma0": Γ0, "Gamma1": Γ1, "Psi": Ψ, "Pi": Π}

    def measurement_equation(self, params=None):
        """Observe all variables."""
        Z = np.eye(3)
        D = np.zeros(3)
        return Z, D

    def shock_covariance(self, params=None):
        """Shock covariance matrix."""
        if params is not None:
            self.parameters.set_values(params)

        σ_d = self.parameters["sigma_d"]
        σ_s = self.parameters["sigma_s"]
        σ_m = self.parameters["sigma_m"]

        return np.diag([σ_d**2, σ_s**2, σ_m**2])


def main() -> None:
    """Run ZLB New Keynesian model example."""
    # Create both regimes

    # Regime M1: Normal times (ZLB slack)
    model_M1 = SimpleNKModel(zlb_binding=False)

    # Regime M2: ZLB binding
    model_M2 = SimpleNKModel(zlb_binding=True)

    # Solve both models

    system_M1 = model_M1.system_matrices()
    solution_M1, _info_M1 = solve_linear_model(
        system_M1["Gamma0"],
        system_M1["Gamma1"],
        system_M1["Psi"],
        system_M1["Pi"],
        model_M1.spec.n_states,
    )

    system_M2 = model_M2.system_matrices()
    solution_M2, _info_M2 = solve_linear_model(
        system_M2["Gamma0"],
        system_M2["Gamma1"],
        system_M2["Psi"],
        system_M2["Pi"],
        model_M2.spec.n_states,
    )

    if not (solution_M1.is_stable and solution_M2.is_stable):
        return

    # Create ZLB constraint (interest rate is index 2)
    zlb_constraint = create_zlb_constraint(variable_index=2, bound=0.0)

    # Create OccBin solver
    occbin_solver = OccBinSolver(solution_M1, solution_M2, zlb_constraint)

    # Simulate with a large negative demand shock
    T = 50
    initial_state = np.zeros(3)  # Start at steady state

    shocks = np.zeros((T, 3))
    shocks[0, 0] = -5.0  # Large negative demand shock in period 0

    # Solve OccBin model
    occbin_solution = occbin_solver.solve(initial_state, shocks, T)

    # Count ZLB periods
    np.sum(occbin_solution.regime_sequence == 1)

    # Plot results
    _fig, axes = plt.subplots(4, 1, figsize=(10, 10))

    var_names = ["Output Gap", "Inflation", "Interest Rate"]
    for i in range(3):
        axes[i].plot(occbin_solution.states[:, i], "b-", linewidth=2)
        axes[i].set_ylabel(var_names[i])
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color="k", linestyle="--", alpha=0.5)

        # Shade ZLB periods
        for t in range(T):
            if occbin_solution.regime_sequence[t] == 1:
                axes[i].axvspan(t - 0.5, t + 0.5, alpha=0.2, color="red")

    # Plot regime
    axes[3].plot(occbin_solution.regime_sequence, "r-", linewidth=2, drawstyle="steps-post")
    axes[3].set_ylabel("Regime (0=Normal, 1=ZLB)")
    axes[3].set_xlabel("Time")
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim([-0.1, 1.1])

    plt.tight_layout()
    plt.savefig("examples/zlb_simulation.png", dpi=150)


if __name__ == "__main__":
    main()
