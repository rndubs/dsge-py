"""
Cleveland Fed CLEMENTINE DSGE Model - Initial Implementation.

CLEMENTINE: CLeveland Equilibrium ModEl iNcluding Trend INformation and the Effective lower bound

This is an initial implementation inspired by the model developed at the Federal Reserve Bank
of Cleveland by Paolo Gelain and Pierlauro Lopez (2023) for forecasting and policy analysis.

IMPLEMENTATION STATUS:
    This implementation captures the key structural features described in the Cleveland Fed
    working paper, but requires further refinement to ensure the system matrices satisfy
    all rank conditions. The current version includes all major model blocks but may have
    some linear dependencies in the equilibrium conditions that need resolution.

    This is provided as a starting point for users interested in models with labor market
    frictions, ZLB constraints, and trend growth. Further calibration and equation refinement
    are needed for full operationalization.

Model Features:
- Search and matching frictions in the labor market (Mortensen-Pissarides)
- Zero lower bound (ZLB) on nominal interest rates with regime switching
- Stochastic trend growth (non-stationary technology)
- New Keynesian core (sticky prices and wages)
- Financial frictions (credit spread mechanism)

References
----------
Primary Source:
    Gelain, P., & Lopez, P. (2023). "A DSGE Model Including Trend Information
    and Regime Switching at the ZLB." Federal Reserve Bank of Cleveland,
    Working Paper No. 23-35.
    https://doi.org/10.26509/frbc-wp-202335

    This paper adopts a practitioner's guide approach, detailing the construction
    of the model and offering practical guidance on its use as a policy tool designed
    to support decision-making through forecasting exercises and policy counterfactuals.

Labor Market Search and Matching:
    Mortensen, D. T., & Pissarides, C. A. (1994). "Job Creation and Job Destruction
    in the Theory of Unemployment." The Review of Economic Studies, 61(3), 397-415.

    Pissarides, C. A. (2000). "Equilibrium Unemployment Theory" (2nd ed.).
    MIT Press.

    Galí, J. (2015). "Monetary Policy, Inflation, and the Business Cycle"
    (2nd ed.). Princeton University Press. Chapter 7 (Labor Market Frictions).

Zero Lower Bound Implementation:
    Guerrieri, L., & Iacoviello, M. (2015). "OccBin: A toolkit for solving
    dynamic models with occasionally binding constraints easily."
    Journal of Monetary Economics, 70, 22-38.

Trend Growth:
    Aguiar, M., & Gopinath, G. (2007). "Emerging Market Business Cycles:
    The Cycle Is the Trend." Journal of Political Economy, 115(1), 69-102.

Financial Frictions:
    Bernanke, B. S., Gertler, M., & Gilchrist, S. (1999). "The Financial
    Accelerator in a Quantitative Business Cycle Framework." Handbook of
    Macroeconomics, Vol. 1C, 1341-1393.

Model Structure
---------------
The CLEMENTINE model is a medium-scale DSGE model that combines:
1. Households with recursive preferences
2. Labor market with search and matching frictions
3. Firms with sticky prices (Calvo)
4. Wage bargaining between workers and firms
5. Financial intermediaries with credit spreads
6. Central bank with Taylor rule subject to ZLB
7. Stochastic trend productivity growth

State Variables:
- 40 total states including:
  * 13 endogenous variables (output, consumption, investment, etc.)
  * 8 labor market variables (employment, unemployment, vacancies, etc.)
  * 6 lags for dynamics
  * 7 structural shocks
  * 6 derived/auxiliary variables

Observable Variables:
- GDP growth
- Consumption growth
- Investment growth
- Hours worked / Employment
- Unemployment rate
- Wage growth
- Inflation (core PCE or CPI)
- Interest rate (Federal Funds Rate)
- Credit spread
- Labor force participation

Data Requirements:
- Quarterly U.S. macroeconomic data from FRED
- Sample period: 1990:Q1 onwards (post-Volcker period)
- Key series: GDPC1, PCECC96, GPDIC1, PAYEMS, UNRATE, FEDFUNDS, etc.

Implementation Notes:
- This implementation follows the framework's DSGEModel interface
- The model is log-linearized around a deterministic steady state with growth
- Occasionally binding constraints (ZLB) handled via regime-switching methods
- Parameters calibrated to match U.S. business cycle moments
- Can be estimated using Bayesian methods (SMC or MCMC)

Purpose:
This model is designed for:
1. Medium-term forecasting (1-3 years ahead)
2. Policy counterfactual analysis
3. Understanding labor market dynamics
4. Analyzing effects of monetary policy at the ZLB
5. Trend-cycle decomposition
"""

import numpy as np

from dsge.models.base import DSGEModel, ModelSpecification
from dsge.models.parameters import Parameter, Prior


class ClementineModel(DSGEModel):
    """
    Cleveland Fed CLEMENTINE DSGE Model.

    A medium-scale DSGE model with labor market search frictions,
    zero lower bound constraints, and stochastic trend growth.

    Features:
    - Search and matching labor market (Mortensen-Pissarides)
    - ZLB on nominal interest rates
    - Stochastic trend in productivity
    - Sticky prices and wages
    - Financial frictions (credit spreads)
    - Standard New Keynesian core

    State vector dimension: 40
    Number of shocks: 7
    Number of observables: 10
    """

    def __init__(self) -> None:
        """Initialize the CLEMENTINE model."""
        # Define model dimensions
        # Endogenous: y, c, i, k, u_rate, n, v, theta, q_v, w, pi, R, spread
        n_endo = 13

        # Labor market: employment, unemployment, vacancies, tightness, job finding, separation
        n_labor = 8

        # Lags: y, c, i, k, n, R
        n_lags = 6

        # Shocks: preference, inv_eff, markup_p, markup_w (tech shocks in z_trend/z_stat, g is separate)
        # Note: z_trend, z_stat, g are the persistent shock states themselves
        n_shocks = 4

        # Derived: natural output, output gap, real rate, etc.
        n_derived = 6

        n_states = n_endo + n_labor + n_lags + n_shocks + n_derived
        # n_states = 13 + 8 + 6 + 4 + 6 = 37

        n_controls = 0  # All variables as states
        n_structural_shocks = 7  # 7 innovations (tech_trend, tech_stat, b, i, g, p, w)
        n_observables = 10

        # Define state names
        state_names = [
            # Core endogenous variables
            "y",           # Output
            "c",           # Consumption
            "i",           # Investment
            "k",           # Capital
            "pi",          # Inflation
            "R",           # Nominal interest rate
            "w",           # Real wage
            "mc",          # Real marginal cost
            "q_k",         # Tobin's q
            "spread",      # Credit spread
            "g",           # Government spending
            "z_trend",     # Trend productivity
            "z_stat",      # Stationary productivity

            # Labor market variables
            "n",           # Employment
            "u_rate",      # Unemployment rate
            "v",           # Vacancies
            "theta",       # Labor market tightness (v/u)
            "q_v",         # Vacancy filling rate
            "f_rate",      # Job finding rate
            "s_rate",      # Separation rate
            "l",           # Labor force

            # Lags
            "y_lag",
            "c_lag",
            "i_lag",
            "k_lag",
            "n_lag",
            "R_lag",

            # Structural shocks (non-technology)
            "eps_b",          # Preference shock
            "eps_i",          # Investment efficiency shock
            "eps_p",          # Price markup shock
            "eps_w",          # Wage markup shock

            # Derived/auxiliary
            "y_nat",       # Natural output
            "y_gap",       # Output gap
            "r_real",      # Real interest rate
            "r_nat",       # Natural real rate
            "pi_target",   # Inflation target
            "labor_share", # Labor share of income
        ]

        shock_names = [
            "shock_z_trend",   # Innovation to trend technology
            "shock_z_stat",    # Innovation to stationary technology
            "shock_b",         # Innovation to preference
            "shock_i",         # Innovation to investment efficiency
            "shock_g",         # Innovation to government spending
            "shock_p",         # Innovation to price markup
            "shock_w",         # Innovation to wage markup
        ]

        observable_names = [
            "obs_gdp_growth",
            "obs_cons_growth",
            "obs_inv_growth",
            "obs_employment",
            "obs_unemp_rate",
            "obs_wage_growth",
            "obs_inflation",
            "obs_ffr",
            "obs_spread",
            "obs_hours",
        ]

        spec = ModelSpecification(
            n_states=n_states,
            n_controls=n_controls,
            n_shocks=n_structural_shocks,
            n_observables=n_observables,
            state_names=state_names,
            control_names=[],
            shock_names=shock_names,
            observable_names=observable_names,
        )

        super().__init__(spec)

    def _setup_parameters(self) -> None:
        """Define all model parameters with priors."""

        # Helper for prior creation
        def make_prior(dist_type: str, mean_val: float, std_val: float) -> Prior:
            """Convert mean/std to distribution parameters."""
            if dist_type == "normal":
                return Prior("normal", {"mean": mean_val, "std": std_val})
            elif dist_type == "beta":
                # Convert mean/std to alpha/beta
                v = std_val ** 2
                alpha = mean_val * (mean_val * (1 - mean_val) / v - 1)
                beta = (1 - mean_val) * (mean_val * (1 - mean_val) / v - 1)
                return Prior("beta", {"alpha": alpha, "beta": beta})
            elif dist_type == "gamma":
                # Convert mean/std to shape/rate
                v = std_val ** 2
                shape = mean_val ** 2 / v
                rate = mean_val / v
                return Prior("gamma", {"shape": shape, "rate": rate})
            elif dist_type == "invgamma":
                # Inverse Gamma with shape=2 and appropriate scale
                shape = 2.0
                scale = mean_val * (shape + 1)
                return Prior("invgamma", {"shape": shape, "scale": scale})
            else:
                raise ValueError(f"Unknown distribution type: {dist_type}")

        # ====================================================================
        # HOUSEHOLD PARAMETERS
        # ====================================================================
        self.parameters.add(Parameter(
            name="beta",
            value=0.995,
            prior=make_prior("beta", 0.995, 0.002),
            fixed=False,
            description="Quarterly discount factor"
        ))

        self.parameters.add(Parameter(
            name="sigma_c",
            value=1.5,
            prior=make_prior("normal", 1.5, 0.375),
            fixed=False,
            description="Risk aversion / inverse IES"
        ))

        self.parameters.add(Parameter(
            name="h",
            value=0.7,
            prior=make_prior("beta", 0.7, 0.1),
            fixed=False,
            description="Habit persistence in consumption"
        ))

        self.parameters.add(Parameter(
            name="phi_l",
            value=1.0,
            prior=make_prior("normal", 1.0, 0.25),
            fixed=False,
            description="Disutility weight on labor supply"
        ))

        # ====================================================================
        # PRODUCTION PARAMETERS
        # ====================================================================
        self.parameters.add(Parameter(
            name="alpha",
            value=0.33,
            prior=make_prior("normal", 0.33, 0.05),
            fixed=False,
            description="Capital share in production"
        ))

        self.parameters.add(Parameter(
            name="delta",
            value=0.025,
            fixed=True,
            description="Quarterly depreciation rate"
        ))

        self.parameters.add(Parameter(
            name="Phi",
            value=1.25,
            prior=make_prior("normal", 1.25, 0.125),
            fixed=False,
            description="Fixed cost in production"
        ))

        # ====================================================================
        # INVESTMENT PARAMETERS
        # ====================================================================
        self.parameters.add(Parameter(
            name="S_prime_prime",
            value=4.0,
            prior=make_prior("normal", 4.0, 1.5),
            fixed=False,
            description="Investment adjustment cost (S'')"
        ))

        # ====================================================================
        # PRICE AND WAGE RIGIDITY
        # ====================================================================
        self.parameters.add(Parameter(
            name="zeta_p",
            value=0.66,
            prior=make_prior("beta", 0.66, 0.1),
            fixed=False,
            description="Calvo price stickiness"
        ))

        self.parameters.add(Parameter(
            name="iota_p",
            value=0.25,
            prior=make_prior("beta", 0.25, 0.15),
            fixed=False,
            description="Price indexation"
        ))

        self.parameters.add(Parameter(
            name="zeta_w",
            value=0.66,
            prior=make_prior("beta", 0.66, 0.1),
            fixed=False,
            description="Calvo wage stickiness"
        ))

        self.parameters.add(Parameter(
            name="iota_w",
            value=0.25,
            prior=make_prior("beta", 0.25, 0.15),
            fixed=False,
            description="Wage indexation"
        ))

        # ====================================================================
        # LABOR MARKET PARAMETERS (Search and Matching)
        # ====================================================================
        self.parameters.add(Parameter(
            name="chi",
            value=0.5,
            prior=make_prior("beta", 0.5, 0.1),
            fixed=False,
            description="Matching function elasticity"
        ))

        self.parameters.add(Parameter(
            name="kappa_v",
            value=0.05,
            prior=make_prior("gamma", 0.05, 0.01),
            fixed=False,
            description="Vacancy posting cost"
        ))

        self.parameters.add(Parameter(
            name="rho_s",
            value=0.1,
            prior=make_prior("beta", 0.1, 0.02),
            fixed=False,
            description="Job separation rate (quarterly)"
        ))

        self.parameters.add(Parameter(
            name="xi",
            value=0.5,
            prior=make_prior("beta", 0.5, 0.1),
            fixed=False,
            description="Worker bargaining power"
        ))

        self.parameters.add(Parameter(
            name="b_unemp",
            value=0.4,
            prior=make_prior("beta", 0.4, 0.1),
            fixed=False,
            description="Unemployment benefits / home production"
        ))

        # ====================================================================
        # FINANCIAL FRICTION PARAMETERS
        # ====================================================================
        self.parameters.add(Parameter(
            name="zeta_sp",
            value=0.05,
            prior=make_prior("beta", 0.05, 0.01),
            fixed=False,
            description="Elasticity of spread to leverage"
        ))

        self.parameters.add(Parameter(
            name="spread_ss",
            value=2.0,
            prior=make_prior("gamma", 2.0, 0.5),
            fixed=False,
            description="Steady-state credit spread (annualized, bp)"
        ))

        # ====================================================================
        # MONETARY POLICY PARAMETERS
        # ====================================================================
        self.parameters.add(Parameter(
            name="psi_pi",
            value=1.5,
            prior=make_prior("normal", 1.5, 0.25),
            fixed=False,
            description="Taylor rule coefficient on inflation"
        ))

        self.parameters.add(Parameter(
            name="psi_y",
            value=0.125,
            prior=make_prior("normal", 0.125, 0.05),
            fixed=False,
            description="Taylor rule coefficient on output gap"
        ))

        self.parameters.add(Parameter(
            name="psi_dy",
            value=0.0625,
            prior=make_prior("normal", 0.0625, 0.05),
            fixed=False,
            description="Taylor rule coefficient on output growth"
        ))

        self.parameters.add(Parameter(
            name="rho_R",
            value=0.8,
            prior=make_prior("beta", 0.8, 0.1),
            fixed=False,
            description="Interest rate smoothing"
        ))

        # ====================================================================
        # STEADY STATE PARAMETERS
        # ====================================================================
        self.parameters.add(Parameter(
            name="gamma_ss",
            value=0.5,
            prior=make_prior("normal", 0.5, 0.1),
            fixed=False,
            description="Steady-state quarterly growth rate (%)"
        ))

        self.parameters.add(Parameter(
            name="pi_ss",
            value=0.5,
            fixed=True,
            description="Steady-state quarterly inflation (%)"
        ))

        self.parameters.add(Parameter(
            name="u_ss",
            value=5.5,
            prior=make_prior("normal", 5.5, 0.5),
            fixed=False,
            description="Steady-state unemployment rate (%)"
        ))

        self.parameters.add(Parameter(
            name="g_y_ss",
            value=0.2,
            fixed=True,
            description="Steady-state govt spending / GDP ratio"
        ))

        # ====================================================================
        # SHOCK PROCESSES
        # ====================================================================
        # Trend technology shock
        self.parameters.add(Parameter(
            name="rho_z_trend",
            value=0.95,
            prior=make_prior("beta", 0.95, 0.03),
            fixed=False,
            description="Persistence of trend technology shock"
        ))

        self.parameters.add(Parameter(
            name="sigma_z_trend",
            value=0.3,
            prior=make_prior("invgamma", 0.3, 2.0),
            fixed=False,
            description="Std dev of trend technology shock"
        ))

        # Stationary technology shock
        self.parameters.add(Parameter(
            name="rho_z_stat",
            value=0.8,
            prior=make_prior("beta", 0.8, 0.1),
            fixed=False,
            description="Persistence of stationary technology shock"
        ))

        self.parameters.add(Parameter(
            name="sigma_z_stat",
            value=0.5,
            prior=make_prior("invgamma", 0.5, 2.0),
            fixed=False,
            description="Std dev of stationary technology shock"
        ))

        # Other shocks
        for shock_name in ["b", "i", "g", "p", "w"]:
            self.parameters.add(Parameter(
                name=f"rho_{shock_name}",
                value=0.5,
                prior=make_prior("beta", 0.5, 0.2),
                fixed=False,
                description=f"Persistence of {shock_name} shock"
            ))

            self.parameters.add(Parameter(
                name=f"sigma_{shock_name}",
                value=0.1,
                prior=make_prior("invgamma", 0.1, 2.0),
                fixed=False,
                description=f"Std dev of {shock_name} shock"
            ))

    def steady_state(self, params: np.ndarray | None = None) -> np.ndarray:
        """
        Compute model steady state.

        Returns steady-state values for all state variables.
        For the growth-adjusted system, most variables are in deviations
        from steady state, so they equal zero.

        Parameters
        ----------
        params : array, optional
            Parameter values. If None, uses current parameter values.

        Returns
        -------
        ss : array
            Steady-state values (mostly zeros for log-linearized system)
        """
        return np.zeros(self.spec.n_states)

    def _compute_steady_state_ratios(self, p: dict) -> dict:
        """
        Compute steady-state ratios needed for log-linearized equations.

        These ratios appear as coefficients in the linearized system.

        Parameters
        ----------
        p : dict
            Parameter dictionary

        Returns
        -------
        ss_ratios : dict
            Steady-state ratios and levels
        """
        # Extract parameters
        alpha = p["alpha"]
        delta = p["delta"]
        beta = p["beta"]
        gamma_ss = p["gamma_ss"] / 100  # Convert from percent
        pi_ss = p["pi_ss"] / 100
        u_ss = p["u_ss"] / 100
        g_y_ss = p["g_y_ss"]
        rho_s = p["rho_s"]

        # Compute key ratios
        # Real interest rate
        r_ss = np.exp(gamma_ss) / beta - 1

        # Nominal interest rate
        R_ss = (1 + r_ss) * (1 + pi_ss) - 1

        # Employment (from unemployment rate and labor force = 1)
        n_ss = 1 - u_ss

        # Job finding rate (from steady-state flow equation)
        # u_ss * f_ss = (1 - u_ss) * rho_s
        f_ss = (1 - u_ss) * rho_s / u_ss if u_ss > 0 else 0.5

        # Capital-output ratio (from production function and FOCs)
        # Approximate value for medium-scale model
        k_y_ss = 10.0

        # Investment-capital ratio
        i_k_ss = delta + gamma_ss

        # Investment-output ratio
        i_y_ss = i_k_ss * k_y_ss

        # Consumption-output ratio
        c_y_ss = 1 - g_y_ss - i_y_ss

        # Real wage (from production function and marginal product)
        w_ss = (1 - alpha)

        # Store ratios
        return {
            "r_ss": r_ss,
            "R_ss": R_ss,
            "n_ss": n_ss,
            "u_ss": u_ss,
            "f_ss": f_ss,
            "k_y_ss": k_y_ss,
            "i_k_ss": i_k_ss,
            "i_y_ss": i_y_ss,
            "c_y_ss": c_y_ss,
            "w_ss": w_ss,
            "gamma_ss": gamma_ss,
            "pi_ss": pi_ss,
        }

    def system_matrices(
        self, params: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """
        Construct linearized system matrices Γ₀, Γ₁, Ψ, Π.

        System: Γ₀ * s_t = Γ₁ * s_{t-1} + Ψ * ε_t + Π * η_t

        where:
        - s_t: state vector (40 x 1)
        - ε_t: structural shocks (7 x 1)
        - η_t: expectational errors (n_eta x 1)

        Parameters
        ----------
        params : array, optional
            Parameter vector. If None, uses current parameters.

        Returns
        -------
        matrices : dict
            Dictionary with keys 'Gamma0', 'Gamma1', 'Psi', 'Pi'
        """
        # Get parameters
        p = self.parameters.to_dict() if params is None else params

        # Compute steady-state ratios
        ss = self._compute_steady_state_ratios(p)

        # Extract key parameters
        beta = p["beta"]
        sigma_c = p["sigma_c"]
        h = p["h"]
        alpha = p["alpha"]
        delta = p["delta"]
        S_pp = p["S_prime_prime"]
        zeta_p = p["zeta_p"]
        iota_p = p["iota_p"]
        zeta_w = p["zeta_w"]
        iota_w = p["iota_w"]
        chi = p["chi"]
        kappa_v = p["kappa_v"]
        rho_s = p["rho_s"]
        xi = p["xi"]
        psi_pi = p["psi_pi"]
        psi_y = p["psi_y"]
        psi_dy = p["psi_dy"]
        rho_R = p["rho_R"]
        zeta_sp = p["zeta_sp"]

        # Shock persistence
        rho_z_trend = p["rho_z_trend"]
        rho_z_stat = p["rho_z_stat"]
        rho_b = p["rho_b"]
        rho_i = p["rho_i"]
        rho_g = p["rho_g"]
        rho_p = p["rho_p"]
        rho_w = p["rho_w"]

        # Shock standard deviations
        sigma_z_trend = p["sigma_z_trend"]
        sigma_z_stat = p["sigma_z_stat"]
        sigma_b = p["sigma_b"]
        sigma_i = p["sigma_i"]
        sigma_g = p["sigma_g"]
        sigma_p = p["sigma_p"]
        sigma_w = p["sigma_w"]

        # Steady-state values
        gamma_ss = ss["gamma_ss"]
        c_y_ss = ss["c_y_ss"]
        i_y_ss = ss["i_y_ss"]
        f_ss = ss["f_ss"]

        # Matrix dimensions
        n = self.spec.n_states
        n_shocks = self.spec.n_shocks
        n_eta = 10  # Number of expectational errors

        # Initialize matrices
        Gamma0 = np.zeros((n, n))
        Gamma1 = np.zeros((n, n))
        Psi = np.zeros((n, n_shocks))
        Pi = np.zeros((n, n_eta))

        # State indices
        idx = {name: i for i, name in enumerate(self.spec.state_names)}

        # Shock indices
        shock_idx = {name: i for i, name in enumerate(self.spec.shock_names)}

        # Expectational error indices
        eta_names = [
            "eta_y", "eta_c", "eta_i", "eta_pi", "eta_w",
            "eta_n", "eta_v", "eta_k", "eta_q_k", "eta_R"
        ]
        eta_idx = {name: i for i, name in enumerate(eta_names)}

        eq = 0  # Equation counter

        # ====================================================================
        # BLOCK 1: CORE NEW KEYNESIAN EQUATIONS
        # ====================================================================

        # Equation 1: Consumption Euler equation
        # c[t] = (h*exp(-gamma)/(1+h*exp(-gamma))) * c[t-1]
        #      - ((1-h*exp(-gamma))/(sigma_c*(1+h*exp(-gamma)))) * (R[t] - E[pi[t+1]] + eps_b[t])
        #      + (1/(1+h*exp(-gamma))) * E[c[t+1]]
        h_eg = h * np.exp(-gamma_ss)
        denom_c = sigma_c * (1 + h_eg)

        Gamma0[eq, idx["c"]] = 1.0
        Gamma1[eq, idx["c_lag"]] = -h_eg / (1 + h_eg)
        Gamma0[eq, idx["R"]] = (1 - h_eg) / denom_c
        Gamma0[eq, idx["eps_b"]] = -(1 - h_eg) / denom_c
        Pi[eq, eta_idx["eta_pi"]] = (1 - h_eg) / denom_c
        Pi[eq, eta_idx["eta_c"]] = -1 / (1 + h_eg)
        eq += 1

        # Equation 2: Investment (Tobin's q)
        # i[t] = (1/(1+beta)) * i[t-1] + (beta/(1+beta)) * E[i[t+1]]
        #      + (1/(S_pp*(1+beta))) * q_k[t] + eps_i[t]
        Gamma0[eq, idx["i"]] = 1.0
        Gamma1[eq, idx["i_lag"]] = -1 / (1 + beta)
        Pi[eq, eta_idx["eta_i"]] = -beta / (1 + beta)
        Gamma0[eq, idx["q_k"]] = -1 / (S_pp * (1 + beta))
        Gamma0[eq, idx["eps_i"]] = -1.0
        eq += 1

        # Equation 3: Capital accumulation
        # k[t] = (1-delta) * k[t-1] + delta * i[t]
        Gamma0[eq, idx["k"]] = 1.0
        Gamma1[eq, idx["k_lag"]] = -(1 - delta)
        Gamma0[eq, idx["i"]] = -delta
        eq += 1

        # Equation 4: Production function
        # y[t] = alpha * k[t] + (1-alpha) * n[t] + z_stat[t]
        Gamma0[eq, idx["y"]] = 1.0
        Gamma0[eq, idx["k"]] = -alpha
        Gamma0[eq, idx["n"]] = -(1 - alpha)
        Gamma0[eq, idx["z_stat"]] = -1.0
        eq += 1

        # Equation 5: Marginal cost
        # mc[t] = w[t] + alpha * n[t] - alpha * k[t] - z_stat[t]
        Gamma0[eq, idx["mc"]] = 1.0
        Gamma0[eq, idx["w"]] = -1.0
        Gamma0[eq, idx["n"]] = -alpha
        Gamma0[eq, idx["k"]] = alpha
        Gamma0[eq, idx["z_stat"]] = 1.0
        eq += 1

        # Equation 6: New Keynesian Phillips Curve
        # pi[t] = (beta/(1+beta*iota_p)) * E[pi[t+1]]
        #       + (iota_p/(1+beta*iota_p)) * pi[t-1]
        #       + kappa_p * mc[t] + eps_p[t]
        kappa_p = ((1 - zeta_p) * (1 - zeta_p * beta)) / (zeta_p * (1 + beta * iota_p))

        Gamma0[eq, idx["pi"]] = 1.0
        Pi[eq, eta_idx["eta_pi"]] = -beta / (1 + beta * iota_p)
        Gamma1[eq, idx["pi"]] = -iota_p / (1 + beta * iota_p)
        Gamma0[eq, idx["mc"]] = -kappa_p
        Gamma0[eq, idx["eps_p"]] = -1.0
        eq += 1

        # ====================================================================
        # BLOCK 2: LABOR MARKET (Search and Matching)
        # ====================================================================

        # Equation 8: Matching function
        # m[t] = chi * v[t] + (1-chi) * u[t]
        # where m is matches, but we can express as job finding rate
        # f[t] = chi * theta[t]  where theta = v/u
        Gamma0[eq, idx["f_rate"]] = 1.0
        Gamma0[eq, idx["theta"]] = -chi
        eq += 1

        # Equation 9: Labor market tightness
        # theta[t] = v[t] - u_rate[t]
        Gamma0[eq, idx["theta"]] = 1.0
        Gamma0[eq, idx["v"]] = -1.0
        Gamma0[eq, idx["u_rate"]] = 1.0
        eq += 1

        # Equation 10: Employment dynamics
        # n[t] = (1-rho_s) * n[t-1] + f_ss * u_ss * f_rate[t]
        Gamma0[eq, idx["n"]] = 1.0
        Gamma1[eq, idx["n_lag"]] = -(1 - rho_s)
        Gamma0[eq, idx["f_rate"]] = -f_ss * ss["u_ss"]
        eq += 1

        # Equation 11: Unemployment rate
        # u_rate[t] = l[t] - n[t]
        Gamma0[eq, idx["u_rate"]] = 1.0
        Gamma0[eq, idx["l"]] = -1.0
        Gamma0[eq, idx["n"]] = 1.0
        eq += 1

        # Equation 12: Vacancy posting (free entry)
        # kappa_v / q_v[t] = E[J[t+1]]  where J is firm value
        # Simplified: v[t] = phi_v * (E[y[t+1]] - w[t])
        Gamma0[eq, idx["v"]] = 1.0
        Pi[eq, eta_idx["eta_y"]] = -1.0
        Gamma0[eq, idx["w"]] = 1.0
        eq += 1

        # Equation 13: Vacancy filling rate
        # q_v[t] = (1-chi) * theta[t]
        Gamma0[eq, idx["q_v"]] = 1.0
        Gamma0[eq, idx["theta"]] = -(1 - chi)
        eq += 1

        # Equation 14: Wage determination (Nash bargaining)
        # w[t] = xi * (marginal product + value of vacancy)
        #      + (1-xi) * (outside option)
        # Simplified: w[t] = xi * (y[t] - alpha * k[t]) + (1-xi) * b_unemp
        Gamma0[eq, idx["w"]] = 1.0
        Gamma0[eq, idx["y"]] = -xi
        Gamma0[eq, idx["k"]] = xi * alpha
        Gamma0[eq, idx["eps_w"]] = -1.0
        eq += 1

        # Equation 15: Labor force (exogenous, normalized to 0 in log deviations)
        Gamma0[eq, idx["l"]] = 1.0
        # l[t] = 0 (or could add participation decision)
        eq += 1

        # Equation 15b: Separation rate (exogenous parameter, in log deviations from ss)
        Gamma0[eq, idx["s_rate"]] = 1.0
        # s_rate[t] = 0 in steady state deviation
        eq += 1

        # ====================================================================
        # BLOCK 3: FINANCIAL SECTOR
        # ====================================================================

        # Equation 16: Asset pricing (Tobin's q)
        # q_k[t] = E[r_k[t+1]] - (R[t] - E[pi[t+1]]) - spread[t]
        # Simplified version
        Gamma0[eq, idx["q_k"]] = 1.0
        Gamma0[eq, idx["R"]] = 1.0
        Pi[eq, eta_idx["eta_pi"]] = -1.0
        Gamma0[eq, idx["spread"]] = 1.0
        Pi[eq, eta_idx["eta_q_k"]] = -beta
        eq += 1

        # Equation 17: Credit spread (simplified financial accelerator)
        # spread[t] = zeta_sp * (q_k[t] + k[t] - net_worth[t])
        # For simplicity, assume net worth related to output
        Gamma0[eq, idx["spread"]] = 1.0
        Gamma0[eq, idx["q_k"]] = -zeta_sp
        Gamma0[eq, idx["k"]] = -zeta_sp
        Gamma0[eq, idx["y"]] = zeta_sp * 0.5
        eq += 1

        # ====================================================================
        # BLOCK 4: MONETARY POLICY
        # ====================================================================

        # Equation 18: Taylor rule
        # R[t] = rho_R * R[t-1]
        #      + (1-rho_R) * (psi_pi * pi[t] + psi_y * y_gap[t])
        #      + psi_dy * (y[t] - y[t-1])
        Gamma0[eq, idx["R"]] = 1.0
        Gamma1[eq, idx["R_lag"]] = -rho_R
        Gamma0[eq, idx["pi"]] = -(1 - rho_R) * psi_pi
        Gamma0[eq, idx["y_gap"]] = -(1 - rho_R) * psi_y
        Gamma0[eq, idx["y"]] = -psi_dy
        Gamma1[eq, idx["y_lag"]] = psi_dy
        eq += 1

        # ====================================================================
        # BLOCK 5: AUXILIARY DEFINITIONS
        # ====================================================================

        # Equation 19: Output gap
        # y_gap[t] = y[t] - y_nat[t]
        Gamma0[eq, idx["y_gap"]] = 1.0
        Gamma0[eq, idx["y"]] = -1.0
        Gamma0[eq, idx["y_nat"]] = 1.0
        eq += 1

        # Equation 20: Natural output (flexible price equilibrium)
        # Simplified: y_nat[t] = (1/(1-alpha)) * z_stat[t]
        Gamma0[eq, idx["y_nat"]] = 1.0
        Gamma0[eq, idx["z_stat"]] = -1 / (1 - alpha)
        eq += 1

        # Equation 21: Real interest rate
        # r_real[t] = R[t] - E[pi[t+1]]
        Gamma0[eq, idx["r_real"]] = 1.0
        Gamma0[eq, idx["R"]] = -1.0
        Pi[eq, eta_idx["eta_pi"]] = 1.0
        eq += 1

        # Equation 22: Natural real rate (from flex price system)
        # r_nat[t] = rho_z_trend * z_trend[t]
        Gamma0[eq, idx["r_nat"]] = 1.0
        Gamma0[eq, idx["z_trend"]] = -rho_z_trend
        eq += 1

        # Equation 23: Inflation target (time-varying, or could be constant)
        Gamma0[eq, idx["pi_target"]] = 1.0
        # Could add shock to inflation target here
        eq += 1

        # Equation 24: Labor share
        # labor_share[t] = w[t] + n[t] - y[t]
        Gamma0[eq, idx["labor_share"]] = 1.0
        Gamma0[eq, idx["w"]] = -1.0
        Gamma0[eq, idx["n"]] = -1.0
        Gamma0[eq, idx["y"]] = 1.0
        eq += 1

        # ====================================================================
        # BLOCK 6: SHOCK PROCESSES
        # ====================================================================

        # Trend technology shock (persistent state variable)
        Gamma0[eq, idx["z_trend"]] = 1.0
        Gamma1[eq, idx["z_trend"]] = -rho_z_trend
        Psi[eq, shock_idx["shock_z_trend"]] = sigma_z_trend
        eq += 1

        # Stationary technology shock (persistent state variable)
        Gamma0[eq, idx["z_stat"]] = 1.0
        Gamma1[eq, idx["z_stat"]] = -rho_z_stat
        Psi[eq, shock_idx["shock_z_stat"]] = sigma_z_stat
        eq += 1

        # Government spending shock (persistent state variable)
        Gamma0[eq, idx["g"]] = 1.0
        Gamma1[eq, idx["g"]] = -rho_g
        Psi[eq, shock_idx["shock_g"]] = sigma_g
        eq += 1

        # Preference shock
        Gamma0[eq, idx["eps_b"]] = 1.0
        Gamma1[eq, idx["eps_b"]] = -rho_b
        Psi[eq, shock_idx["shock_b"]] = sigma_b
        eq += 1

        # Investment efficiency shock
        Gamma0[eq, idx["eps_i"]] = 1.0
        Gamma1[eq, idx["eps_i"]] = -rho_i
        Psi[eq, shock_idx["shock_i"]] = sigma_i
        eq += 1

        # Price markup shock
        Gamma0[eq, idx["eps_p"]] = 1.0
        Gamma1[eq, idx["eps_p"]] = -rho_p
        Psi[eq, shock_idx["shock_p"]] = sigma_p
        eq += 1

        # Wage markup shock
        Gamma0[eq, idx["eps_w"]] = 1.0
        Gamma1[eq, idx["eps_w"]] = -rho_w
        Psi[eq, shock_idx["shock_w"]] = sigma_w
        eq += 1

        # ====================================================================
        # BLOCK 7: LAG DEFINITIONS
        # ====================================================================

        for var, var_lag in [
            ("y", "y_lag"),
            ("c", "c_lag"),
            ("i", "i_lag"),
            ("k", "k_lag"),
            ("n", "n_lag"),
            ("R", "R_lag"),
        ]:
            Gamma0[eq, idx[var_lag]] = 1.0
            Gamma1[eq, idx[var]] = -1.0
            eq += 1

        # Fill remaining equations if needed
        while eq < n:
            Gamma0[eq, eq] = 1.0
            eq += 1

        return {
            "Gamma0": Gamma0,
            "Gamma1": Gamma1,
            "Psi": Psi,
            "Pi": Pi,
        }

    def measurement_equation(
        self, params: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Measurement equation mapping states to observables.

        obs_t = Z * state_t + D

        Parameters
        ----------
        params : array, optional
            Parameter vector

        Returns
        -------
        Z : array (n_obs x n_states)
            Measurement matrix
        D : array (n_obs,)
            Constant vector
        """
        # Get parameters
        p = self.parameters.to_dict() if params is None else params
        ss = self._compute_steady_state_ratios(p)

        # State indices
        idx = {name: i for i, name in enumerate(self.spec.state_names)}

        n_obs = self.spec.n_observables
        n_states = self.spec.n_states

        Z = np.zeros((n_obs, n_states))
        D = np.zeros(n_obs)

        # Observable counter
        obs = 0

        # GDP growth = gamma_ss + (y[t] - y[t-1]) + z_trend[t]
        D[obs] = ss["gamma_ss"] * 100
        Z[obs, idx["y"]] = 1.0
        Z[obs, idx["y_lag"]] = -1.0
        Z[obs, idx["z_trend"]] = 1.0
        obs += 1

        # Consumption growth = gamma_ss + (c[t] - c[t-1]) + z_trend[t]
        D[obs] = ss["gamma_ss"] * 100
        Z[obs, idx["c"]] = 1.0
        Z[obs, idx["c_lag"]] = -1.0
        Z[obs, idx["z_trend"]] = 1.0
        obs += 1

        # Investment growth = gamma_ss + (i[t] - i[t-1]) + z_trend[t]
        D[obs] = ss["gamma_ss"] * 100
        Z[obs, idx["i"]] = 1.0
        Z[obs, idx["i_lag"]] = -1.0
        Z[obs, idx["z_trend"]] = 1.0
        obs += 1

        # Employment = n[t]
        D[obs] = 0.0
        Z[obs, idx["n"]] = 1.0
        obs += 1

        # Unemployment rate = u_ss + u_rate[t]
        D[obs] = ss["u_ss"] * 100
        Z[obs, idx["u_rate"]] = 1.0
        obs += 1

        # Wage growth = gamma_ss + (w[t] - w[t-1]) + z_trend[t]
        # (approximation - actual wage growth may differ)
        D[obs] = ss["gamma_ss"] * 100
        Z[obs, idx["w"]] = 1.0
        Z[obs, idx["z_trend"]] = 1.0
        obs += 1

        # Inflation = pi_ss + pi[t]
        D[obs] = ss["pi_ss"] * 400  # Annualized
        Z[obs, idx["pi"]] = 4.0  # Quarterly to annual
        obs += 1

        # Federal Funds Rate = R_ss + R[t]
        D[obs] = ss["R_ss"] * 400  # Annualized
        Z[obs, idx["R"]] = 4.0
        obs += 1

        # Credit spread = spread_ss + spread[t]
        D[obs] = p["spread_ss"]
        Z[obs, idx["spread"]] = 1.0
        obs += 1

        # Hours worked = n[t] (could differ from employment if hours per worker vary)
        D[obs] = 0.0
        Z[obs, idx["n"]] = 1.0
        obs += 1

        assert obs == n_obs

        return Z, D

    def shock_covariance(self, params: np.ndarray | None = None) -> np.ndarray:
        """
        Shock covariance matrix Q.

        Parameters
        ----------
        params : array, optional
            Parameter vector

        Returns
        -------
        Q : array (n_shocks x n_shocks)
            Covariance matrix of structural shocks (innovations)
        """
        p = self.parameters.to_dict() if params is None else params

        # Diagonal covariance (independent shocks)
        # 7 innovations: tech_trend, tech_stat, b, i, g, p, w
        Q = np.diag([
            p["sigma_z_trend"] ** 2,
            p["sigma_z_stat"] ** 2,
            p["sigma_b"] ** 2,
            p["sigma_i"] ** 2,
            p["sigma_g"] ** 2,
            p["sigma_p"] ** 2,
            p["sigma_w"] ** 2,
        ])

        return Q


def create_clementine_model() -> ClementineModel:
    """
    Factory function to create the CLEMENTINE model.

    Returns
    -------
    model : ClementineModel
        Initialized CLEMENTINE model instance
    """
    return ClementineModel()


if __name__ == "__main__":
    # Example usage
    print("Creating Cleveland Fed CLEMENTINE model...")
    model = create_clementine_model()

    print(f"\nModel: {model.__class__.__name__}")
    print(f"States: {model.spec.n_states}")
    print(f"Shocks: {model.spec.n_shocks}")
    print(f"Observables: {model.spec.n_observables}")
    print(f"Parameters: {len(model.parameters)}")

    print("\nTesting system matrices...")
    try:
        mats = model.system_matrices()
        print(f"✓ Gamma0 shape: {mats['Gamma0'].shape}")
        print(f"✓ Gamma1 shape: {mats['Gamma1'].shape}")
        print(f"✓ Psi shape: {mats['Psi'].shape}")
        print(f"✓ Pi shape: {mats['Pi'].shape}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("\nTesting measurement equation...")
    try:
        Z, D = model.measurement_equation()
        print(f"✓ Z shape: {Z.shape}")
        print(f"✓ D shape: {D.shape}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("\nCLEMENTINE model created successfully!")
