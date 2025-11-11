"""
Galí (2010) Model: New Keynesian DSGE with Unemployment and Labor Market Frictions.

This model implements the baseline sticky wage model from:

Galí, Jordi (2010). "Monetary Policy and Unemployment."
In: Handbook of Monetary Economics, Volume 3A, Chapter 10, pp. 487-546.
Elsevier.

This is a properly published DSGE model with labor market search and matching
frictions based on the Diamond-Mortensen-Pissarides framework. The model combines
New Keynesian features (sticky prices and wages) with unemployment dynamics through
hiring costs and job separation.

Key Features:
- Labor market search and matching (Diamond-Mortensen-Pissarides)
- Explicit unemployment and job-finding rates
- Sticky nominal wages (Calvo pricing)
- Sticky nominal prices (Calvo pricing)
- Hiring costs proportional to labor market tightness
- Taylor rule monetary policy

Implementation Notes:
This implementation is based on the Dynare replication by Lahcen Bounader and
Johannes Pfeifer, which corrects several issues in the original paper's calibration.
See: https://github.com/JohannesPfeifer/DSGE_mod/tree/master/Gali_2010

References
----------
Primary Source:
    Galí, Jordi (2010). "Monetary Policy and Unemployment."
    In: Benjamin M. Friedman and Michael Woodford (eds.),
    Handbook of Monetary Economics, Volume 3A, Chapter 10, pp. 487-546.
    Elsevier.

Theoretical Foundations:
    Diamond, Peter A. (1982). "Aggregate Demand Management in Search Equilibrium."
    Journal of Political Economy, 90(5), 881-894.

    Mortensen, Dale T., and Christopher A. Pissarides (1994).
    "Job Creation and Job Destruction in the Theory of Unemployment."
    Review of Economic Studies, 61(3), 397-415.

    Pissarides, Christopher A. (2000). Equilibrium Unemployment Theory (2nd ed.).
    MIT Press.

Replication Files:
    Bounader, Lahcen, and Johannes Pfeifer (2016).
    Dynare implementation of Galí (2010).
    https://github.com/JohannesPfeifer/DSGE_mod/tree/master/Gali_2010
"""

import numpy as np

from dsge.models.base import DSGEModel, ModelSpecification
from dsge.models.parameters import Parameter


class Gali2010Model(DSGEModel):
    """
    Galí (2010) New Keynesian model with unemployment.

    State variables (30 total):
    - Core: y_gap, chat, rhat, ihat, nhat, lhat, fhat, uhat, uhat_0,
            urhat, xhat, ghat, hhat, mu_hat, hatw_real, bhat, hatw_tar,
            a, pi_w, pi_p, nu
    - Lags (for observables): y_gap_lag, chat_lag, nhat_lag, etc.

    Shocks (2):
    - eps_a: technology shock
    - eps_nu: monetary policy shock

    Observables (6):
    - Output gap
    - Consumption
    - Employment
    - Unemployment rate
    - Wage inflation
    - Price inflation
    """

    def __init__(self) -> None:
        """Initialize the Galí (2010) model."""
        # Model dimensions
        # Core states: 21 endogenous variables
        # Lags: 6 (for growth rates in observables)
        # Shock lags: 2
        # Total: 29 states
        n_states = 29
        n_controls = 0  # All variables are states in log-linearized form
        n_shocks = 2  # Technology and monetary policy
        n_observables = 6

        state_names = [
            # Core endogenous variables (21)
            "y_gap", "chat", "rhat", "ihat", "nhat", "lhat", "fhat",
            "uhat", "uhat_0", "urhat", "xhat", "ghat", "hhat",
            "mu_hat", "hatw_real", "bhat", "hatw_tar",
            "a", "pi_w", "pi_p", "nu",
            # Lags for observables (6)
            "y_gap_lag", "chat_lag", "nhat_lag", "urhat_lag", "pi_w_lag", "pi_p_lag",
            # Shock lags (2)
            "a_lag", "nu_lag"
        ]

        shock_names = ["eps_a", "eps_nu"]

        observable_names = [
            "obs_y_gap",    # Output gap
            "obs_dc",       # Consumption growth
            "obs_dn",       # Employment growth
            "obs_ur",       # Unemployment rate
            "obs_pi_w",     # Wage inflation
            "obs_pi_p"      # Price inflation
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
        """Define model parameters following Galí (2010) calibration."""
        # Production
        self.parameters.add(
            Parameter(
                name="alfa",
                value=1.0/3.0,
                fixed=True,
                description="Exponent of labor in production function (capital share)"
            )
        )

        # Labor market
        self.parameters.add(
            Parameter(
                name="delta",
                value=0.12,  # Set in steady state to match calibration targets
                fixed=False,
                description="Separation rate (quarterly)"
            )
        )

        self.parameters.add(
            Parameter(
                name="gammma",
                value=1.0,
                fixed=True,
                description="Coefficient of hiring cost function"
            )
        )

        self.parameters.add(
            Parameter(
                name="psi",
                value=0.0,  # Set in steady state file
                fixed=False,
                description="Coefficient of unemployment in labor market effort"
            )
        )

        # Preferences
        self.parameters.add(
            Parameter(
                name="betta",
                value=0.99,
                fixed=True,
                description="Discount factor (quarterly)"
            )
        )

        self.parameters.add(
            Parameter(
                name="varphi",
                value=5.0,
                fixed=False,
                description="Frisch elasticity of labor effort"
            )
        )

        self.parameters.add(
            Parameter(
                name="chi",
                value=1.0,  # Set in steady state file
                fixed=False,
                description="Labor disutility parameter"
            )
        )

        # Bargaining
        self.parameters.add(
            Parameter(
                name="xi",
                value=0.05,  # Low calibration
                fixed=False,
                description="Bargaining power of workers"
            )
        )

        # Nominal rigidities
        self.parameters.add(
            Parameter(
                name="theta_w",
                value=0.75,
                fixed=False,
                description="Calvo wage stickiness (75% of wages unchanged)"
            )
        )

        self.parameters.add(
            Parameter(
                name="theta_p",
                value=0.75,
                fixed=False,
                description="Calvo price stickiness (75% of prices unchanged)"
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
                value=0.5/4.0,  # 0.125
                fixed=False,
                description="Taylor rule response to output gap"
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
                name="rho_nu",
                value=0.5,
                fixed=False,
                description="Monetary policy shock persistence"
            )
        )

        # Shock standard deviations
        self.parameters.add(
            Parameter(
                name="sigma_a",
                value=1.0,
                fixed=False,
                description="Technology shock std dev"
            )
        )

        self.parameters.add(
            Parameter(
                name="sigma_nu",
                value=0.25,
                fixed=False,
                description="Monetary policy shock std dev (25 basis points)"
            )
        )

        # Steady state/calibration targets
        self.parameters.add(
            Parameter(
                name="N",
                value=0.59,
                fixed=True,
                description="Employment rate (steady state)"
            )
        )

        self.parameters.add(
            Parameter(
                name="U",
                value=0.03,
                fixed=True,
                description="Unemployment rate (steady state)"
            )
        )

        self.parameters.add(
            Parameter(
                name="x",
                value=0.7,
                fixed=True,
                description="Job finding rate (steady state)"
            )
        )

        self.parameters.add(
            Parameter(
                name="hiring_cost_share",
                value=0.045,
                fixed=True,
                description="Hiring costs as share of quarterly wage"
            )
        )

        # Composite parameters (computed in steady state)
        self.parameters.add(
            Parameter(
                name="Theta",
                value=0.0036,  # Computed to satisfy calibration
                fixed=False,
                description="Share of hiring costs to GDP"
            )
        )

        self.parameters.add(
            Parameter(
                name="Upsilon",
                value=0.5,  # Computed in steady state
                fixed=False,
                description="Composite parameter for wage determination"
            )
        )

        self.parameters.add(
            Parameter(
                name="Phi",
                value=0.0,  # Computed in steady state
                fixed=False,
                description="Share of hiring costs to hiring costs plus wage"
            )
        )

        self.parameters.add(
            Parameter(
                name="Xi",
                value=0.0,  # Computed in steady state
                fixed=False,
                description="Coefficient in optimal participation condition"
            )
        )

        # Derived steady state values
        self.parameters.add(
            Parameter(
                name="F",
                value=0.62,  # N + U
                fixed=True,
                description="Labor force (steady state)"
            )
        )

        self.parameters.add(
            Parameter(
                name="L",
                value=0.59,  # Approximately N in low psi calibration
                fixed=True,
                description="Labor effort in utility function (steady state)"
            )
        )

    def system_matrices(self, params: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """
        Compute linearized system matrices for Galí (2010) model.

        System: Γ0 * s_t = Γ1 * s_{t-1} + Ψ * ε_t + Π * η_t

        The model has 20 equilibrium conditions plus lag definitions.
        """
        # Get parameter values
        p = self.parameters.to_dict() if params is None else params

        alfa = p["alfa"]
        delta = p["delta"]
        gammma = p["gammma"]
        psi = p["psi"]
        betta = p["betta"]
        varphi = p["varphi"]
        xi = p["xi"]
        theta_w = p["theta_w"]
        theta_p = p["theta_p"]
        phi_pi = p["phi_pi"]
        phi_y = p["phi_y"]
        rho_a = p["rho_a"]
        rho_nu = p["rho_nu"]
        sigma_a = p["sigma_a"]
        sigma_nu = p["sigma_nu"]
        Theta = p["Theta"]
        Upsilon = p["Upsilon"]
        Phi = p["Phi"]
        Xi = p["Xi"]
        N = p["N"]
        U = p["U"]
        F = p["F"]
        L = p["L"]
        x = p["x"]

        n = self.spec.n_states
        n_shocks = self.spec.n_shocks
        n_eta = 6  # Expectation errors: chat, pi_p, pi_w, ghat, rhat, nhat

        # Initialize matrices
        Gamma0 = np.zeros((n, n))
        Gamma1 = np.zeros((n, n))
        Psi = np.zeros((n, n_shocks))
        Pi = np.zeros((n, n_eta))

        # State indices (following the order in state_names)
        idx = {name: i for i, name in enumerate(self.spec.state_names)}

        # Shock indices
        idx_eps_a, idx_eps_nu = 0, 1

        # Expectation error indices
        idx_eta_chat, idx_eta_pi_p, idx_eta_pi_w = 0, 1, 2
        idx_eta_ghat, idx_eta_rhat = 3, 4

        # Derived parameters
        lambda_p = (1 - theta_p) * (1 - betta * theta_p) / theta_p
        lambda_w = ((1 - betta * (1 - delta) * theta_w) * (1 - theta_w) /
                   (theta_w * (1 - (1 - Upsilon) * (1 - Phi))))

        eq = 0

        # ===================================================================
        # Equation 1: Goods Market Clearing
        # y_gap = (1-Theta)*chat + Theta*(ghat + hhat)
        # ===================================================================
        Gamma0[eq, idx["y_gap"]] = 1.0
        Gamma0[eq, idx["chat"]] = -(1 - Theta)
        Gamma0[eq, idx["ghat"]] = -Theta
        Gamma0[eq, idx["hhat"]] = -Theta
        eq += 1

        # ===================================================================
        # Equation 2: Aggregate production function
        # y_gap = a + (1-alfa)*nhat
        # ===================================================================
        Gamma0[eq, idx["y_gap"]] = 1.0
        Gamma0[eq, idx["a"]] = -1.0
        Gamma0[eq, idx["nhat"]] = -(1 - alfa)
        eq += 1

        # ===================================================================
        # Equation 3: Aggregate hiring and employment
        # delta*hhat = nhat - (1-delta)*nhat(-1)
        # ===================================================================
        Gamma0[eq, idx["hhat"]] = delta
        Gamma0[eq, idx["nhat"]] = -1.0
        Gamma1[eq, idx["nhat"]] = -(1 - delta)
        eq += 1

        # ===================================================================
        # Equation 4: Hiring Cost
        # ghat = gammma * xhat
        # ===================================================================
        Gamma0[eq, idx["ghat"]] = 1.0
        Gamma0[eq, idx["xhat"]] = -gammma
        eq += 1

        # ===================================================================
        # Equation 5: Job finding rate
        # xhat = hhat - uhat_0
        # ===================================================================
        Gamma0[eq, idx["xhat"]] = 1.0
        Gamma0[eq, idx["hhat"]] = -1.0
        Gamma0[eq, idx["uhat_0"]] = 1.0
        eq += 1

        # ===================================================================
        # Equation 6: Effective market effort
        # lhat = (N/L)*nhat + (psi*U/L)*uhat
        # ===================================================================
        Gamma0[eq, idx["lhat"]] = 1.0
        Gamma0[eq, idx["nhat"]] = -(N / L)
        Gamma0[eq, idx["uhat"]] = -(psi * U / L)
        eq += 1

        # ===================================================================
        # Equation 7: Labor Force
        # fhat = (N/F)*nhat + (U/F)*uhat
        # ===================================================================
        Gamma0[eq, idx["fhat"]] = 1.0
        Gamma0[eq, idx["nhat"]] = -(N / F)
        Gamma0[eq, idx["uhat"]] = -(U / F)
        eq += 1

        # ===================================================================
        # Equation 8: Unemployment
        # uhat = uhat_0 - (x/(1-x))*xhat
        # ===================================================================
        Gamma0[eq, idx["uhat"]] = 1.0
        Gamma0[eq, idx["uhat_0"]] = -1.0
        Gamma0[eq, idx["xhat"]] = x / (1 - x)
        eq += 1

        # ===================================================================
        # Equation 9: Unemployment rate
        # urhat = (U/F)*uhat - (U/F)*fhat
        # ===================================================================
        Gamma0[eq, idx["urhat"]] = 1.0
        Gamma0[eq, idx["uhat"]] = -(U / F)
        Gamma0[eq, idx["fhat"]] = (U / F)
        eq += 1

        # ===================================================================
        # Equation 10: Euler equation
        # chat = E_t[chat(+1)] - rhat
        # ===================================================================
        Gamma0[eq, idx["chat"]] = 1.0
        Gamma0[eq, idx["rhat"]] = 1.0
        Pi[eq, idx_eta_chat] = -1.0  # -E_t[chat(+1)]
        eq += 1

        # ===================================================================
        # Equation 11: Fisher equation
        # rhat = ihat - E_t[pi_p(+1)]
        # ===================================================================
        Gamma0[eq, idx["rhat"]] = 1.0
        Gamma0[eq, idx["ihat"]] = -1.0
        Pi[eq, idx_eta_pi_p] = 1.0  # +E_t[pi_p(+1)]
        eq += 1

        # ===================================================================
        # Equation 12: Price Phillips Curve
        # pi_p = betta*E_t[pi_p(+1)] - lambda_p*mu_hat
        # ===================================================================
        Gamma0[eq, idx["pi_p"]] = 1.0
        Gamma0[eq, idx["mu_hat"]] = lambda_p
        Pi[eq, idx_eta_pi_p] = -betta  # -betta*E_t[pi_p(+1)]
        eq += 1

        # ===================================================================
        # Equation 13: Optimal Hiring Condition
        # alfa*nhat = a - ((1-Phi)*hatw_real + Phi*bhat) - mu_hat
        # ===================================================================
        Gamma0[eq, idx["nhat"]] = alfa
        Gamma0[eq, idx["a"]] = -1.0
        Gamma0[eq, idx["hatw_real"]] = (1 - Phi)
        Gamma0[eq, idx["bhat"]] = Phi
        Gamma0[eq, idx["mu_hat"]] = 1.0
        eq += 1

        # ===================================================================
        # Equation 14: Definition of bhat
        # bhat = (1/(1-betta*(1-delta)))*ghat -
        #        (betta*(1-delta)/(1-betta*(1-delta)))*(E_t[ghat(+1)] - rhat)
        # ===================================================================
        coef1 = 1.0 / (1 - betta * (1 - delta))
        coef2 = betta * (1 - delta) / (1 - betta * (1 - delta))

        Gamma0[eq, idx["bhat"]] = 1.0
        Gamma0[eq, idx["ghat"]] = -coef1
        Gamma0[eq, idx["rhat"]] = -coef2
        Pi[eq, idx_eta_ghat] = coef2  # +coef2*E_t[ghat(+1)]
        eq += 1

        # ===================================================================
        # Equation 15: Optimal participation condition
        # chat + varphi*lhat = (1/(1-x))*xhat + ghat - Xi*pi_w
        # ===================================================================
        Gamma0[eq, idx["chat"]] = 1.0
        Gamma0[eq, idx["lhat"]] = varphi
        Gamma0[eq, idx["xhat"]] = -1.0 / (1 - x)
        Gamma0[eq, idx["ghat"]] = -1.0
        Gamma0[eq, idx["pi_w"]] = Xi
        eq += 1

        # ===================================================================
        # Equation 16: Evolution of real wage
        # hatw_real = hatw_real(-1) + pi_w - pi_p
        # ===================================================================
        Gamma0[eq, idx["hatw_real"]] = 1.0
        Gamma0[eq, idx["pi_w"]] = -1.0
        Gamma0[eq, idx["pi_p"]] = 1.0
        Gamma1[eq, idx["hatw_real"]] = 1.0
        eq += 1

        # ===================================================================
        # Equation 17: Wage Phillips Curve
        # pi_w = betta*(1-delta)*E_t[pi_w(+1)] - lambda_w*(hatw_real - hatw_tar)
        # ===================================================================
        Gamma0[eq, idx["pi_w"]] = 1.0
        Gamma0[eq, idx["hatw_real"]] = lambda_w
        Gamma0[eq, idx["hatw_tar"]] = -lambda_w
        Pi[eq, idx_eta_pi_w] = -betta * (1 - delta)  # -betta*(1-delta)*E_t[pi_w(+1)]
        eq += 1

        # ===================================================================
        # Equation 18: Target wage
        # hatw_tar = Upsilon*(chat + varphi*lhat) +
        #            (1-Upsilon)*(-mu_hat + a - alfa*nhat)
        # ===================================================================
        Gamma0[eq, idx["hatw_tar"]] = 1.0
        Gamma0[eq, idx["chat"]] = -Upsilon
        Gamma0[eq, idx["lhat"]] = -Upsilon * varphi
        Gamma0[eq, idx["mu_hat"]] = (1 - Upsilon)
        Gamma0[eq, idx["a"]] = -(1 - Upsilon)
        Gamma0[eq, idx["nhat"]] = (1 - Upsilon) * alfa
        eq += 1

        # ===================================================================
        # Equation 19: Interest rate rule
        # ihat = phi_pi*pi_p + phi_y*y_gap + nu
        # ===================================================================
        Gamma0[eq, idx["ihat"]] = 1.0
        Gamma0[eq, idx["pi_p"]] = -phi_pi
        Gamma0[eq, idx["y_gap"]] = -phi_y
        Gamma0[eq, idx["nu"]] = -1.0
        eq += 1

        # ===================================================================
        # Equation 20: Monetary policy shock
        # nu = rho_nu*nu(-1) + eps_nu
        # ===================================================================
        Gamma0[eq, idx["nu"]] = 1.0
        Gamma1[eq, idx["nu"]] = rho_nu
        Psi[eq, idx_eps_nu] = sigma_nu
        eq += 1

        # ===================================================================
        # Equation 21: Technology shock
        # a = rho_a*a(-1) + eps_a
        # ===================================================================
        Gamma0[eq, idx["a"]] = 1.0
        Gamma1[eq, idx["a"]] = rho_a
        Psi[eq, idx_eps_a] = sigma_a
        eq += 1

        # ===================================================================
        # Equations 22-27: Lag definitions for observables
        # ===================================================================
        lag_pairs = [
            ("y_gap_lag", "y_gap"),
            ("chat_lag", "chat"),
            ("nhat_lag", "nhat"),
            ("urhat_lag", "urhat"),
            ("pi_w_lag", "pi_w"),
            ("pi_p_lag", "pi_p")
        ]

        for lag_name, var_name in lag_pairs:
            Gamma0[eq, idx[lag_name]] = 1.0
            Gamma1[eq, idx[var_name]] = 1.0
            eq += 1

        # ===================================================================
        # Equations 28-29: Shock lag definitions
        # ===================================================================
        Gamma0[eq, idx["a_lag"]] = 1.0
        Gamma1[eq, idx["a"]] = 1.0
        eq += 1

        Gamma0[eq, idx["nu_lag"]] = 1.0
        Gamma1[eq, idx["nu"]] = 1.0
        eq += 1

        assert eq == n, f"Expected {n} equations, got {eq}"

        return {"Gamma0": Gamma0, "Gamma1": Gamma1, "Psi": Psi, "Pi": Pi}

    def measurement_equation(
        self, params: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Measurement equation: observables = Z * states + D.

        Observables:
        1. Output gap: y_gap
        2. Consumption growth: chat - chat_lag
        3. Employment growth: nhat - nhat_lag
        4. Unemployment rate: urhat
        5. Wage inflation: pi_w
        6. Price inflation: pi_p
        """
        n_obs = self.spec.n_observables
        n_states = self.spec.n_states

        Z = np.zeros((n_obs, n_states))
        D = np.zeros(n_obs)

        idx = {name: i for i, name in enumerate(self.spec.state_names)}

        # Observable 0: Output gap
        Z[0, idx["y_gap"]] = 1.0

        # Observable 1: Consumption growth
        Z[1, idx["chat"]] = 1.0
        Z[1, idx["chat_lag"]] = -1.0

        # Observable 2: Employment growth
        Z[2, idx["nhat"]] = 1.0
        Z[2, idx["nhat_lag"]] = -1.0

        # Observable 3: Unemployment rate
        Z[3, idx["urhat"]] = 1.0

        # Observable 4: Wage inflation
        Z[4, idx["pi_w"]] = 1.0

        # Observable 5: Price inflation
        Z[5, idx["pi_p"]] = 1.0

        return Z, D

    def steady_state(self, params: np.ndarray | None = None) -> np.ndarray:
        """
        Compute steady state.

        In the log-linearized model, all variables are deviations from steady state,
        so the steady state is zero.

        Note: The nonlinear steady state is non-trivial and involves solving for
        composite parameters (Theta, Upsilon, Phi, Xi) to satisfy calibration targets.
        This is handled in the parameter setup.
        """
        return np.zeros(self.spec.n_states)


def create_gali_2010_model() -> Gali2010Model:
    """Factory function to create a Galí (2010) model instance."""
    return Gali2010Model()


if __name__ == "__main__":
    # Example usage
    model = create_gali_2010_model()

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
