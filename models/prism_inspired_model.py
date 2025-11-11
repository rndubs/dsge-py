"""
PRISM-Inspired Model: Medium-Scale DSGE with Labor Market Search Frictions.

This model implements a medium-scale New Keynesian DSGE model with labor market
search frictions, inspired by the Philadelphia Federal Reserve's PRISM-II model
and based on the academic literature.

Note: The exact specification of PRISM-II is proprietary to the Philadelphia Fed.
This implementation is based on similar models in the academic literature that
incorporate the same key features: labor market search frictions, unemployment,
and nominal/real rigidities.

Model Structure:
---------------
- Households with consumption-leisure choice
- Labor market with search and matching frictions (Mortensen-Pissarides)
- Unemployment as an explicit state variable
- Firms with sticky prices (Calvo pricing)
- Wage bargaining between firms and workers
- Monetary policy via Taylor rule
- Multiple shocks: technology, monetary policy, labor market

Key Features:
------------
1. Labor market frictions with hiring costs
2. Unemployment rate determination via matching function
3. Wage determination via bargaining or sticky wages
4. Nominal price rigidities
5. Standard New Keynesian transmission mechanisms

References
----------
This implementation draws on the following academic literature:

Primary Inspiration:
    Philadelphia Federal Reserve Bank (2020). "Philadelphia Research Intertemporal
    Stochastic Model-II (PRISM-II)." Technical Appendix.
    https://www.philadelphiafed.org/surveys-and-data/macroeconomic-data/prism-ii

    Note: PRISM-II is a medium-scale DSGE model with ~30 equations, labor market
    search frictions, unemployment, and nominal/real rigidities. The exact
    specification is maintained by the Philadelphia Fed's Real-Time Data Research Center.

Theoretical Foundations:
    Blanchard, O. J., & Galí, J. (2010). "Labor Markets and Monetary Policy:
    A New Keynesian Model with Unemployment." American Economic Journal:
    Macroeconomics, 2(2), 1-30.

    Gertler, M., Sala, L., & Trigari, A. (2008). "An Estimated Monetary DSGE
    Model with Unemployment and Staggered Nominal Wage Bargaining." Journal of
    Money, Credit and Banking, 40(8), 1713-1764.

    Christoffel, K., Kuester, K., & Linzert, T. (2009). "The Role of Labor
    Markets for Euro Area Monetary Policy." European Economic Review, 53(8),
    908-936.

Search and Matching Framework:
    Diamond, P. A. (1982). "Aggregate Demand Management in Search Equilibrium."
    Journal of Political Economy, 90(5), 881-894.

    Mortensen, D. T., & Pissarides, C. A. (1994). "Job Creation and Job
    Destruction in the Theory of Unemployment." Review of Economic Studies,
    61(3), 397-415.

    Pissarides, C. A. (2000). "Equilibrium Unemployment Theory" (2nd ed.).
    MIT Press.

Model Simplification:
    This implementation follows the simpler Blanchard-Galí (2010) framework
    rather than the more complex Gertler-Sala-Trigari (2008) staggered wage
    bargaining approach, while maintaining the key features of labor market
    search frictions and unemployment.

Parameter Calibration:
    Based on standard quarterly U.S. data calibrations from the literature:
    - beta = 0.99 (discount factor, ~4% annual rate)
    - sigma = 1.0 (log utility in consumption)
    - theta = 0.75 (Calvo price stickiness, ~4 quarters)
    - kappa = 0.3 (Phillips curve slope)
    - phi_pi = 1.5 (Taylor rule inflation response)
    - phi_y = 0.5 (Taylor rule output response)
    - alpha_m = 0.5 (matching function elasticity)
    - rho_u = 0.05 (exogenous separation rate)

Purpose:
-------
This model serves as an example of integrating labor market frictions into
the DSGE framework. It demonstrates:
- Implementation of search and matching models
- Modeling unemployment dynamics
- Integration with New Keynesian features
- Framework applicability to policy-relevant models

The model can be extended with additional features present in PRISM-II such as:
- More detailed wage determination (staggered bargaining)
- Financial frictions
- Additional sectors
- More shock processes
"""

import numpy as np

from dsge.models.base import DSGEModel, ModelSpecification
from dsge.models.parameters import Parameter


class PRISMInspiredModel(DSGEModel):
    """
    PRISM-inspired medium-scale DSGE model with labor market search frictions.

    Based on Blanchard-Galí (2010) framework with unemployment.

    State variables (24 total):
    - Core variables: c_t, y_t, pi_t, r_t, w_t, n_t, u_t, theta_t, x_t
    - Lags: c_lag, y_lag, pi_lag, r_lag, w_lag, n_lag, u_lag, theta_lag, x_lag
    - Shocks: e_a, e_m, e_s (technology, monetary policy, labor market)
    - Shock lags: e_a_lag, e_m_lag, e_s_lag

    where:
        c_t = consumption (log deviation from SS)
        y_t = output
        pi_t = inflation rate
        r_t = nominal interest rate
        w_t = real wage
        n_t = employment
        u_t = unemployment rate
        theta_t = labor market tightness (vacancies/unemployment)
        x_t = hiring rate

    Shocks (3):
    - eps_a: technology shock
    - eps_m: monetary policy shock
    - eps_s: labor market (separation) shock

    Observables (7):
    - Output growth
    - Consumption growth
    - Inflation
    - Nominal interest rate
    - Real wage growth
    - Employment growth
    - Unemployment rate
    """

    def __init__(self) -> None:
        """Initialize the PRISM-inspired model."""
        # Model dimensions
        n_states = 24  # Core(9) + lags(9) + shocks(3) + shock_lags(3)
        n_controls = 0  # All variables are states
        n_shocks = 3  # Technology, monetary policy, labor market
        n_observables = 7  # Growth rates and levels

        state_names = [
            # Core economic variables
            "c", "y", "pi", "r", "w", "n", "u", "theta", "x",
            # Lags for observables
            "c_lag", "y_lag", "pi_lag", "r_lag", "w_lag", "n_lag",
            "u_lag", "theta_lag", "x_lag",
            # Shock processes
            "e_a", "e_m", "e_s",
            # Shock lags
            "e_a_lag", "e_m_lag", "e_s_lag"
        ]

        shock_names = ["eps_a", "eps_m", "eps_s"]

        observable_names = [
            "obs_dy",    # Output growth
            "obs_dc",    # Consumption growth
            "obs_pi",    # Inflation
            "obs_r",     # Interest rate
            "obs_dw",    # Wage growth
            "obs_dn",    # Employment growth
            "obs_u"      # Unemployment rate
        ]

        spec = ModelSpecification(
            n_states=n_states,
            n_controls=n_controls,
            n_shocks=n_shocks,
            n_observables=n_observables,
            state_names=state_names,
            control_names=[],
            shock_names=shock_names,
            observable_names=observable_names,
        )

        super().__init__(spec)

    def _setup_parameters(self) -> None:
        """Define model parameters."""
        # Household parameters
        self.parameters.add(
            Parameter(
                name="beta",
                value=0.99,
                fixed=True,
                description="Discount factor (quarterly)"
            )
        )

        self.parameters.add(
            Parameter(
                name="sigma",
                value=1.0,
                fixed=False,
                description="Inverse of intertemporal elasticity of substitution"
            )
        )

        # Price setting
        self.parameters.add(
            Parameter(
                name="theta_p",
                value=0.75,
                fixed=False,
                description="Calvo price stickiness parameter"
            )
        )

        self.parameters.add(
            Parameter(
                name="epsilon_p",
                value=6.0,
                fixed=True,
                description="Elasticity of substitution between goods"
            )
        )

        # Labor market parameters
        self.parameters.add(
            Parameter(
                name="alpha_m",
                value=0.5,
                fixed=False,
                description="Matching function elasticity (w.r.t. unemployment)"
            )
        )

        self.parameters.add(
            Parameter(
                name="rho_u",
                value=0.05,
                fixed=False,
                description="Exogenous separation rate"
            )
        )

        self.parameters.add(
            Parameter(
                name="kappa_v",
                value=0.5,
                fixed=False,
                description="Vacancy posting cost"
            )
        )

        self.parameters.add(
            Parameter(
                name="gamma",
                value=0.5,
                fixed=False,
                description="Worker bargaining power"
            )
        )

        # Wage stickiness (can be zero for flexible wages)
        self.parameters.add(
            Parameter(
                name="xi_w",
                value=0.5,
                fixed=False,
                description="Wage adjustment cost parameter"
            )
        )

        # Monetary policy
        self.parameters.add(
            Parameter(
                name="phi_pi",
                value=1.5,
                fixed=False,
                description="Taylor rule response to inflation"
            )
        )

        self.parameters.add(
            Parameter(
                name="phi_y",
                value=0.5,
                fixed=False,
                description="Taylor rule response to output gap"
            )
        )

        self.parameters.add(
            Parameter(
                name="rho_r",
                value=0.75,
                fixed=False,
                description="Interest rate smoothing"
            )
        )

        # Technology
        self.parameters.add(
            Parameter(
                name="alpha",
                value=0.33,
                fixed=True,
                description="Capital share (if capital included)"
            )
        )

        # Shock persistence
        self.parameters.add(
            Parameter(
                name="rho_a",
                value=0.9,
                fixed=False,
                description="Technology shock persistence"
            )
        )

        self.parameters.add(
            Parameter(
                name="rho_m",
                value=0.5,
                fixed=False,
                description="Monetary shock persistence"
            )
        )

        self.parameters.add(
            Parameter(
                name="rho_s",
                value=0.8,
                fixed=False,
                description="Labor market shock persistence"
            )
        )

        # Shock standard deviations
        self.parameters.add(
            Parameter(
                name="sigma_a",
                value=0.01,
                fixed=False,
                description="Technology shock std dev"
            )
        )

        self.parameters.add(
            Parameter(
                name="sigma_m",
                value=0.0025,
                fixed=False,
                description="Monetary policy shock std dev"
            )
        )

        self.parameters.add(
            Parameter(
                name="sigma_s",
                value=0.01,
                fixed=False,
                description="Labor market shock std dev"
            )
        )

    def system_matrices(self, params: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """
        Compute linearized system matrices.

        System: Γ0 * s_t = Γ1 * s_{t-1} + Ψ * ε_t + Π * η_t

        This is a simplified version of the Blanchard-Galí (2010) model.
        The full model would include additional equilibrium conditions for:
        - Job creation condition
        - Wage determination
        - Matching function
        - Unemployment dynamics

        For tractability, we implement a reduced-form version that captures
        the key dynamics while maintaining the framework structure.
        """
        # Get parameter values
        p = self.parameters.to_dict() if params is None else params

        beta = p["beta"]
        sigma = p["sigma"]
        theta_p = p["theta_p"]
        epsilon_p = p["epsilon_p"]
        alpha_m = p["alpha_m"]
        rho_u = p["rho_u"]
        kappa_v = p["kappa_v"]
        gamma = p["gamma"]
        xi_w = p["xi_w"]
        phi_pi = p["phi_pi"]
        phi_y = p["phi_y"]
        rho_r = p["rho_r"]
        alpha = p["alpha"]
        rho_a = p["rho_a"]
        rho_m = p["rho_m"]
        rho_s = p["rho_s"]
        sigma_a = p["sigma_a"]
        sigma_m = p["sigma_m"]
        sigma_s = p["sigma_s"]

        n = self.spec.n_states
        n_shocks = self.spec.n_shocks
        n_eta = 4  # Expectation errors: E[c_t+1], E[pi_t+1], E[n_t+1], E[theta_t+1]

        # Initialize matrices
        Gamma0 = np.zeros((n, n))
        Gamma1 = np.zeros((n, n))
        Psi = np.zeros((n, n_shocks))
        Pi = np.zeros((n, n_eta))

        # State indices
        idx_c, idx_y, idx_pi, idx_r, idx_w, idx_n, idx_u = 0, 1, 2, 3, 4, 5, 6
        idx_theta, idx_x = 7, 8
        idx_c_lag, idx_y_lag, idx_pi_lag, idx_r_lag = 9, 10, 11, 12
        idx_w_lag, idx_n_lag, idx_u_lag, idx_theta_lag, idx_x_lag = 13, 14, 15, 16, 17
        idx_e_a, idx_e_m, idx_e_s = 18, 19, 20
        idx_e_a_lag, idx_e_m_lag, idx_e_s_lag = 21, 22, 23

        # Shock indices
        idx_eps_a, idx_eps_m, idx_eps_s = 0, 1, 2

        # Expectation error indices
        idx_eta_c, idx_eta_pi, idx_eta_n, idx_eta_theta = 0, 1, 2, 3

        # Derived parameters
        kappa = (1 - theta_p) * (1 - beta * theta_p) / theta_p * (epsilon_p - 1)

        # ===================================================================
        # Equation 0: Consumption Euler equation
        # c_t = E_t[c_t+1] - 1/sigma * (r_t - E_t[pi_t+1])
        # ===================================================================
        eq = 0
        Gamma0[eq, idx_c] = 1.0
        Gamma0[eq, idx_r] = 1.0 / sigma
        Pi[eq, idx_eta_c] = -1.0
        Pi[eq, idx_eta_pi] = -1.0 / sigma

        # ===================================================================
        # Equation 1: Phillips curve
        # pi_t = beta * E_t[pi_t+1] + kappa * (w_t + n_t - y_t - e_a_t)
        # (where w_t + n_t - y_t - e_a_t is marginal cost)
        # ===================================================================
        eq = 1
        Gamma0[eq, idx_pi] = 1.0
        Gamma0[eq, idx_w] = -kappa
        Gamma0[eq, idx_n] = -kappa
        Gamma0[eq, idx_y] = kappa
        Gamma0[eq, idx_e_a] = kappa
        Pi[eq, idx_eta_pi] = -beta

        # ===================================================================
        # Equation 2: Production function (simplified)
        # y_t = e_a_t + n_t
        # ===================================================================
        eq = 2
        Gamma0[eq, idx_y] = 1.0
        Gamma0[eq, idx_e_a] = -1.0
        Gamma0[eq, idx_n] = -1.0

        # ===================================================================
        # Equation 3: Goods market clearing
        # y_t = c_t
        # (simplified, assumes no investment, government spending, or vacancies)
        # ===================================================================
        eq = 3
        Gamma0[eq, idx_y] = 1.0
        Gamma0[eq, idx_c] = -1.0

        # ===================================================================
        # Equation 5: Taylor rule
        # r_t = rho_r * r_t-1 + (1-rho_r) * (phi_pi * pi_t + phi_y * y_t) + e_m_t
        # ===================================================================
        eq = 4
        Gamma0[eq, idx_r] = 1.0
        Gamma0[eq, idx_pi] = -(1 - rho_r) * phi_pi
        Gamma0[eq, idx_y] = -(1 - rho_r) * phi_y
        Gamma0[eq, idx_e_m] = -1.0
        Gamma1[eq, idx_r_lag] = rho_r

        # ===================================================================
        # Equation 6: Job creation condition (simplified)
        # kappa_v * theta_t = beta * E_t[(y_t+1 - w_t+1) + (1-rho_u) * kappa_v * theta_t+1]
        # Linearized: theta_t ≈ E_t[y_t+1 - w_t+1] + (1-rho_u) * E_t[theta_t+1]
        # ===================================================================
        eq = 5
        Gamma0[eq, idx_theta] = kappa_v
        Pi[eq, idx_eta_n] = -beta  # Simplified using n instead of (y-w)
        Pi[eq, idx_eta_theta] = -beta * (1 - rho_u) * kappa_v

        # ===================================================================
        # Equation 7: Wage equation (simplified Nash bargaining)
        # w_t = gamma * (y_t - n_t + theta_t) + (1-gamma) * (utility from employment)
        # Simplified: w_t = gamma * y_t + (1-gamma) * MRS + adjustment costs
        # ===================================================================
        eq = 6
        Gamma0[eq, idx_w] = 1.0
        Gamma0[eq, idx_y] = -gamma
        Gamma0[eq, idx_n] = gamma
        Gamma0[eq, idx_theta] = -gamma
        Gamma1[eq, idx_w_lag] = xi_w  # Wage stickiness

        # ===================================================================
        # Equation 8: Matching function
        # x_t = m_0 * u_t^alpha_m * v_t^(1-alpha_m)
        # Log-linearized: x_t = alpha_m * u_t + (1-alpha_m) * v_t
        # And v_t ≈ theta_t + u_t (definition of tightness)
        # So: x_t = alpha_m * u_t + (1-alpha_m) * (theta_t + u_t) = u_t + (1-alpha_m) * theta_t
        # ===================================================================
        eq = 7
        Gamma0[eq, idx_x] = 1.0
        Gamma0[eq, idx_u] = -1.0
        Gamma0[eq, idx_theta] = -(1 - alpha_m)

        # ===================================================================
        # Equation 9: Unemployment dynamics
        # u_t = u_t-1 + rho_u * (1 - u_t-1) - x_t + e_s_t
        # Linearized around steady state: u_t ≈ (1-rho_u) * u_t-1 - x_t + e_s_t
        # ===================================================================
        eq = 8
        Gamma0[eq, idx_u] = 1.0
        Gamma0[eq, idx_x] = 1.0
        Gamma0[eq, idx_e_s] = -1.0
        Gamma1[eq, idx_u_lag] = (1 - rho_u)

        # ===================================================================
        # Equations 10-18: Lag definitions
        # ===================================================================
        lag_vars = [
            (idx_c_lag, idx_c),
            (idx_y_lag, idx_y),
            (idx_pi_lag, idx_pi),
            (idx_r_lag, idx_r),
            (idx_w_lag, idx_w),
            (idx_n_lag, idx_n),
            (idx_u_lag, idx_u),
            (idx_theta_lag, idx_theta),
            (idx_x_lag, idx_x)
        ]

        for i, (lag_idx, var_idx) in enumerate(lag_vars):
            eq = 9 + i
            Gamma0[eq, lag_idx] = 1.0
            Gamma1[eq, var_idx] = 1.0

        # ===================================================================
        # Equations 19-21: Shock processes
        # e_a_t = rho_a * e_a_t-1 + sigma_a * eps_a_t
        # e_m_t = rho_m * e_m_t-1 + sigma_m * eps_m_t
        # e_s_t = rho_s * e_s_t-1 + sigma_s * eps_s_t
        # ===================================================================
        eq = 18
        Gamma0[eq, idx_e_a] = 1.0
        Gamma1[eq, idx_e_a] = rho_a
        Psi[eq, idx_eps_a] = sigma_a

        eq = 19
        Gamma0[eq, idx_e_m] = 1.0
        Gamma1[eq, idx_e_m] = rho_m
        Psi[eq, idx_eps_m] = sigma_m

        eq = 20
        Gamma0[eq, idx_e_s] = 1.0
        Gamma1[eq, idx_e_s] = rho_s
        Psi[eq, idx_eps_s] = sigma_s

        # ===================================================================
        # Equations 22-24: Shock lag definitions
        # ===================================================================
        Gamma0[21, idx_e_a_lag] = 1.0
        Gamma1[21, idx_e_a] = 1.0

        Gamma0[22, idx_e_m_lag] = 1.0
        Gamma1[22, idx_e_m] = 1.0

        Gamma0[23, idx_e_s_lag] = 1.0
        Gamma1[23, idx_e_s] = 1.0

        return {"Gamma0": Gamma0, "Gamma1": Gamma1, "Psi": Psi, "Pi": Pi}

    def measurement_equation(
        self, params: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Measurement equation: observables = Z * states + D.

        Observables:
        1. Output growth: dy_t = y_t - y_t-1
        2. Consumption growth: dc_t = c_t - c_t-1
        3. Inflation: pi_t
        4. Interest rate: r_t
        5. Wage growth: dw_t = w_t - w_t-1
        6. Employment growth: dn_t = n_t - n_t-1
        7. Unemployment rate: u_t
        """
        n_obs = self.spec.n_observables
        n_states = self.spec.n_states

        Z = np.zeros((n_obs, n_states))
        D = np.zeros(n_obs)

        # State indices
        idx_c, idx_y, idx_pi, idx_r, idx_w, idx_n, idx_u = 0, 1, 2, 3, 4, 5, 6
        idx_c_lag, idx_y_lag, idx_pi_lag, idx_r_lag = 9, 10, 11, 12
        idx_w_lag, idx_n_lag = 13, 14

        # Observable 0: Output growth
        Z[0, idx_y] = 1.0
        Z[0, idx_y_lag] = -1.0

        # Observable 1: Consumption growth
        Z[1, idx_c] = 1.0
        Z[1, idx_c_lag] = -1.0

        # Observable 2: Inflation
        Z[2, idx_pi] = 1.0

        # Observable 3: Interest rate
        Z[3, idx_r] = 1.0

        # Observable 4: Wage growth
        Z[4, idx_w] = 1.0
        Z[4, idx_w_lag] = -1.0

        # Observable 5: Employment growth
        Z[5, idx_n] = 1.0
        Z[5, idx_n_lag] = -1.0

        # Observable 6: Unemployment rate
        Z[6, idx_u] = 1.0

        return Z, D

    def steady_state(self, params: np.ndarray | None = None) -> np.ndarray:
        """
        Compute steady state.

        In this log-linearized model, all variables are deviations from steady state,
        so the steady state is zero.

        Note: A full nonlinear model would compute the actual steady-state values
        for employment, unemployment, vacancies, etc.
        """
        return np.zeros(self.spec.n_states)


def create_prism_inspired_model() -> PRISMInspiredModel:
    """Factory function to create a PRISM-inspired model instance."""
    return PRISMInspiredModel()


if __name__ == "__main__":
    # Example usage
    model = create_prism_inspired_model()

    # Test system matrices
    mats = model.system_matrices()
    print(f"Gamma0 shape: {mats['Gamma0'].shape}")
    print(f"Gamma1 shape: {mats['Gamma1'].shape}")
    print(f"Psi shape: {mats['Psi'].shape}")
    print(f"Pi shape: {mats['Pi'].shape}")

    # Test measurement equation
    Z, D = model.measurement_equation()
    print(f"\nMeasurement matrix Z shape: {Z.shape}")
    print(f"Measurement offset D shape: {D.shape}")

    # Display parameters
    print(f"\nModel has {len(model.parameters)} parameters")
    print("Parameter names:", [p.name for p in model.parameters])
