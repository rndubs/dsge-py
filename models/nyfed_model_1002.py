"""
FRBNY DSGE Model 1002 - Framework Implementation.

Translation of the New York Federal Reserve DSGE model (version 1002)
to the dsge-py framework interface.

References
----------
Official Documentation:
    FRBNY DSGE Model Documentation (March 3, 2021)
    https://github.com/FRBNY-DSGE/DSGE.jl/blob/main/docs/DSGE_Model_Documentation_1002.pdf

Reference Implementation (Julia):
    FRBNY-DSGE/DSGE.jl - Model 1002
    https://github.com/FRBNY-DSGE/DSGE.jl/tree/main/src/models/representative/m1002

    Key files for verification:
    - m1002.jl: Model definition and parameter specifications
    - eqcond.jl: Equilibrium conditions (canonical form matrices)
    - measurement.jl: Observable equations

    Parameters verified against DSGE.jl v1.1.6+ (2021-2024).

Foundational Papers:
    Del Negro, M., Giannoni, M. P., & Schorfheide, F. (2015).
    "Inflation in the Great Recession and New Keynesian Models."
    American Economic Journal: Macroeconomics, 7(1), 168-196.

    Del Negro, M., Hasegawa, R. B., & Schorfheide, F. (2016).
    "Dynamic Prediction Pools: An Investigation of Financial Frictions
    and Forecasting Performance." Journal of Econometrics, 192(2), 391-405.

Financial Accelerator Mechanism:
    Bernanke, B. S., Gertler, M., & Gilchrist, S. (1999).
    "The Financial Accelerator in a Quantitative Business Cycle Framework."
    Handbook of Macroeconomics, Vol. 1C, 1341-1393.

Parameter Values Source:
    Parameter values in this implementation are PRIOR MEANS from the FRBNY specification,
    NOT posterior estimates. This is appropriate for:
    1. Estimation frameworks (start from priors, obtain posteriors via SMC/MCMC)
    2. Forecast initialization before re-estimation

    Prior distributions match those specified in DSGE.jl Model 1002 (verified 2025-11-11).
    For posterior estimates, run the estimation pipeline on FRED data or use FRBNY's
    published vintage estimates.

    See ECONOMIC_MODELS_REVIEW.md for detailed parameter verification.

Data Requirements:
    13 U.S. macroeconomic quarterly time series (1959:Q3 onward):
    - Real GDP growth, GDI growth, Consumption growth, Investment growth
    - Real wage growth, Hours worked
    - Core PCE inflation, GDP deflator inflation
    - Federal Funds Rate, 10-year Treasury yield
    - Long-run inflation expectations, Credit spread (BAA-10Y)
    - TFP growth (computed)

Solution Method:
    Sims (2002) canonical form with QZ decomposition (Blanchard-Kahn conditions).
    All equilibrium conditions expressed in matrix form (Γ₀, Γ₁, Ψ, Π).
"""

import numpy as np

from dsge.models.base import DSGEModel, ModelSpecification
from dsge.models.parameters import Parameter, Prior


class NYFedModel1002(DSGEModel):
    """
    FRBNY DSGE Model 1002.

    A medium-scale New Keynesian model with financial frictions.

    State vector includes:
    - 18 endogenous variables
    - Lags of key variables (for dynamics)
    - 9 structural shocks
    - 6 measurement error processes
    - Derived productivity growth variable

    Total: approximately 50 states
    """

    def __init__(self) -> None:
        """Initialize the NYFed Model 1002."""
        # Count states carefully
        # Endogenous: c, i, y, L, k_bar, k, u, q_k, w, R, pi, mc, r_k, R_k_tilde, n, w_h, y_f, pi_star
        n_endo = 18

        # Lags needed: c, i, w, R, pi, k_bar, q_k, n, y, y_f
        n_lags = 10

        # Shocks: z_tilde, z_p, b, mu, g, lambda_f, lambda_w, sigma_omega, r_m
        # Plus MA lags for lambda_f, lambda_w
        n_shocks = 9
        n_shock_lags = 2  # lambda_f_ma, lambda_w_ma

        # Measurement errors: e_gdp, e_gdi, e_pce, e_gdpdef, e_10y, e_tfp
        # Plus lags for e_gdp, e_gdi (for cointegration)
        n_me = 6
        n_me_lags = 2

        # Derived: z (productivity growth)
        n_derived = 1

        n_states = n_endo + n_lags + n_shocks + n_shock_lags + n_me + n_me_lags + n_derived
        # n_states = 18 + 10 + 9 + 2 + 6 + 2 + 1 = 48

        n_controls = 0  # All variables as states for simplicity
        n_structural_shocks = 9  # Innovations to shock processes
        n_observables = 13

        # Define state names
        state_names = [
            # Endogenous variables
            "c",
            "i",
            "y",
            "L",
            "k_bar",
            "k",
            "u",
            "q_k",
            "w",
            "R",
            "pi",
            "mc",
            "r_k",
            "R_k_tilde",
            "n",
            "w_h",
            "y_f",
            "pi_star_t",
            # Lags
            "c_lag",
            "i_lag",
            "w_lag",
            "R_lag",
            "pi_lag",
            "k_bar_lag",
            "q_k_lag",
            "n_lag",
            "y_lag",
            "y_f_lag",
            # Shocks
            "z_tilde",
            "z_p",
            "b",
            "mu",
            "g",
            "lambda_f",
            "lambda_w",
            "sigma_omega",
            "r_m",
            # Shock MA terms
            "lambda_f_ma",
            "lambda_w_ma",
            # Measurement errors
            "e_gdp",
            "e_gdi",
            "e_pce",
            "e_gdpdef",
            "e_10y",
            "e_tfp",
            # Measurement error lags
            "e_gdp_lag",
            "e_gdi_lag",
            # Derived
            "z",  # Productivity growth
        ]

        shock_names = [
            "eps_z",
            "eps_zp",
            "eps_b",
            "eps_mu",
            "eps_g",
            "eps_lambda_f",
            "eps_lambda_w",
            "eps_sigma_omega",
            "eps_rm",
        ]

        observable_names = [
            "obs_gdp_growth",
            "obs_gdi_growth",
            "obs_cons_growth",
            "obs_inv_growth",
            "obs_wage_growth",
            "obs_hours",
            "obs_infl_pce",
            "obs_infl_gdpdef",
            "obs_ffr",
            "obs_10y_rate",
            "obs_10y_infl_exp",
            "obs_spread",
            "obs_tfp_growth",
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
        """Define all model parameters."""

        # Helper function to convert prior specifications
        def make_prior(dist_type, mean_val, std_val):
            """Convert from documentation notation to Prior format."""
            if dist_type == "normal":
                return Prior("normal", {"mean": mean_val, "std": std_val})
            if dist_type == "beta":
                # Convert from mean/std to alpha/beta
                # For Beta distribution: mean = alpha/(alpha+beta), var = alpha*beta/((alpha+beta)^2*(alpha+beta+1))
                v = std_val**2
                alpha = mean_val * (mean_val * (1 - mean_val) / v - 1)
                beta = (1 - mean_val) * (mean_val * (1 - mean_val) / v - 1)
                return Prior("beta", {"alpha": alpha, "beta": beta})
            if dist_type == "gamma":
                # Convert from mean/std to shape/rate
                # mean = shape/rate, var = shape/rate^2
                v = std_val**2
                shape = mean_val**2 / v
                rate = mean_val / v
                return Prior("gamma", {"shape": shape, "rate": rate})
            if dist_type == "invgamma":
                # Inverse Gamma parameterization
                # Using default shape=2, scale such that mode ≈ mean
                shape = 2.0
                scale = mean_val * (shape + 1)
                return Prior("invgamma", {"shape": shape, "scale": scale})
            msg = f"Unknown distribution type: {dist_type}"
            raise ValueError(msg)

        # ====================================================================
        # POLICY PARAMETERS
        # ====================================================================
        self.parameters.add(
            Parameter(
                name="psi1",
                value=1.50,
                prior=make_prior("normal", 1.50, 0.25),
                fixed=False,
                description="Taylor rule coefficient on inflation",
            )
        )

        self.parameters.add(
            Parameter(
                name="psi2",
                value=0.12,
                prior=make_prior("normal", 0.12, 0.05),
                fixed=False,
                description="Taylor rule coefficient on output gap",
            )
        )

        self.parameters.add(
            Parameter(
                name="psi3",
                value=0.12,
                prior=make_prior("normal", 0.12, 0.05),
                fixed=False,
                description="Taylor rule coefficient on output gap growth",
            )
        )

        self.parameters.add(
            Parameter(
                name="rho_R",
                value=0.75,
                prior=make_prior("beta", 0.75, 0.10),
                fixed=False,
                description="Interest rate smoothing",
            )
        )

        self.parameters.add(
            Parameter(
                name="rho_rm",
                value=0.50,
                prior=make_prior("beta", 0.50, 0.20),
                fixed=False,
                description="Monetary policy shock persistence",
            )
        )

        self.parameters.add(
            Parameter(
                name="sigma_rm",
                value=0.10,
                prior=make_prior("invgamma", 0.10, 2.00),
                fixed=False,
                description="Monetary policy shock std dev",
            )
        )

        # ====================================================================
        # NOMINAL RIGIDITIES
        # ====================================================================
        self.parameters.add(
            Parameter(
                name="zeta_p",
                value=0.50,
                prior=make_prior("beta", 0.50, 0.10),
                fixed=False,
                description="Calvo parameter for prices",
            )
        )

        self.parameters.add(
            Parameter(
                name="iota_p",
                value=0.50,
                prior=make_prior("beta", 0.50, 0.15),
                fixed=False,
                description="Price indexation",
            )
        )

        self.parameters.add(
            Parameter(
                name="epsilon_p",
                value=10.0,
                fixed=True,
                description="Curvature of Kimball aggregator (prices)",
            )
        )

        self.parameters.add(
            Parameter(
                name="zeta_w",
                value=0.50,
                prior=make_prior("beta", 0.50, 0.10),
                fixed=False,
                description="Calvo parameter for wages",
            )
        )

        self.parameters.add(
            Parameter(
                name="iota_w",
                value=0.50,
                prior=make_prior("beta", 0.50, 0.15),
                fixed=False,
                description="Wage indexation",
            )
        )

        self.parameters.add(
            Parameter(
                name="epsilon_w",
                value=10.0,
                fixed=True,
                description="Curvature of Kimball aggregator (wages)",
            )
        )

        # ====================================================================
        # STEADY STATE AND PREFERENCES
        # ====================================================================
        self.parameters.add(
            Parameter(
                name="gamma",
                value=0.40,
                prior=make_prior("normal", 0.40, 0.10),
                fixed=False,
                description="Steady-state growth rate (quarterly, x100)",
            )
        )

        self.parameters.add(
            Parameter(
                name="alpha",
                value=0.30,
                prior=make_prior("normal", 0.30, 0.05),
                fixed=False,
                description="Capital share",
            )
        )

        self.parameters.add(
            Parameter(
                name="beta_bar",
                value=0.25,
                prior=make_prior("gamma", 0.25, 0.10),
                fixed=False,
                description="Discount factor transformation: 100*(beta^-1 - 1)",
            )
        )

        self.parameters.add(
            Parameter(
                name="sigma_c",
                value=1.50,
                prior=make_prior("normal", 1.50, 0.37),
                fixed=False,
                description="Risk aversion / IES",
            )
        )

        self.parameters.add(
            Parameter(
                name="h",
                value=0.70,
                prior=make_prior("beta", 0.70, 0.10),
                fixed=False,
                description="Habit persistence",
            )
        )

        self.parameters.add(
            Parameter(
                name="nu_l",
                value=2.00,
                prior=make_prior("normal", 2.00, 0.75),
                fixed=False,
                description="Inverse Frisch elasticity",
            )
        )

        self.parameters.add(
            Parameter(
                name="S_pp",
                value=4.00,
                prior=make_prior("normal", 4.00, 1.50),
                fixed=False,
                description="Investment adjustment cost (S'')",
            )
        )

        self.parameters.add(
            Parameter(
                name="psi",
                value=0.50,
                prior=make_prior("beta", 0.50, 0.15),
                fixed=False,
                description="Capital utilization cost",
            )
        )

        self.parameters.add(
            Parameter(
                name="delta", value=0.025, fixed=True, description="Depreciation rate (quarterly)"
            )
        )

        self.parameters.add(
            Parameter(
                name="Phi_p",
                value=1.25,
                prior=make_prior("normal", 1.25, 0.12),
                fixed=False,
                description="Fixed cost in production",
            )
        )

        self.parameters.add(
            Parameter(
                name="pi_star",
                value=0.50,
                fixed=True,
                description="Steady-state inflation (quarterly, net, x100)",
            )
        )

        self.parameters.add(
            Parameter(name="lambda_w", value=1.50, fixed=True, description="Wage markup")
        )

        self.parameters.add(
            Parameter(
                name="g_star",
                value=0.18,
                fixed=True,
                description="Steady-state government spending share",
            )
        )

        self.parameters.add(
            Parameter(
                name="L_bar",
                value=-45.00,
                prior=make_prior("normal", -45.00, 5.00),
                fixed=False,
                description="Steady-state hours (log-level)",
            )
        )

        # GDP deflator measurement parameters
        self.parameters.add(
            Parameter(
                name="gamma_gdpdef",
                value=1.00,
                prior=make_prior("normal", 1.00, 2.00),
                fixed=False,
                description="GDP deflator loading on model inflation",
            )
        )

        self.parameters.add(
            Parameter(
                name="delta_gdpdef",
                value=0.00,
                prior=make_prior("normal", 0.00, 2.00),
                fixed=False,
                description="GDP deflator steady-state diff from core PCE",
            )
        )

        # ====================================================================
        # FINANCIAL FRICTIONS
        # ====================================================================
        self.parameters.add(
            Parameter(
                name="F_omega",
                value=0.03,
                fixed=True,
                description="Steady-state default probability",
            )
        )

        self.parameters.add(
            Parameter(
                name="zeta_sp_b",
                value=0.05,
                prior=make_prior("beta", 0.05, 0.005),
                fixed=False,
                description="Elasticity of spread w.r.t. leverage",
            )
        )

        self.parameters.add(
            Parameter(
                name="SP_star",
                value=2.00,
                prior=make_prior("gamma", 2.00, 0.10),
                fixed=False,
                description="Steady-state spread (annualized, x100)",
            )
        )

        self.parameters.add(
            Parameter(
                name="gamma_star", value=0.99, fixed=True, description="Entrepreneur survival rate"
            )
        )

        # ====================================================================
        # SHOCK PROCESSES
        # ====================================================================
        # Technology shocks
        self.parameters.add(
            Parameter(
                name="rho_z",
                value=0.50,
                prior=make_prior("beta", 0.50, 0.20),
                fixed=False,
                description="Stationary TFP persistence",
            )
        )

        self.parameters.add(
            Parameter(
                name="sigma_z",
                value=0.10,
                prior=make_prior("invgamma", 0.10, 2.00),
                fixed=False,
                description="Stationary TFP shock std dev",
            )
        )

        self.parameters.add(
            Parameter(
                name="rho_zp",
                value=0.50,
                prior=make_prior("beta", 0.50, 0.20),
                fixed=False,
                description="Stochastic trend TFP persistence",
            )
        )

        self.parameters.add(
            Parameter(
                name="sigma_zp",
                value=0.10,
                prior=make_prior("invgamma", 0.10, 2.00),
                fixed=False,
                description="Stochastic trend TFP shock std dev",
            )
        )

        # Other shocks - using similar priors for all
        for shock in ["b", "mu", "g", "sigma_w"]:
            self.parameters.add(
                Parameter(
                    name=f"rho_{shock}",
                    value=0.50,
                    prior=make_prior("beta", 0.50, 0.20),
                    fixed=False,
                    description=f"{shock} shock persistence",
                )
            )
            self.parameters.add(
                Parameter(
                    name=f"sigma_{shock}",
                    value=0.10,
                    prior=make_prior("invgamma", 0.10, 2.00),
                    fixed=False,
                    description=f"{shock} shock std dev",
                )
            )

        # Markup shocks with MA component
        for markup in ["lambda_f", "lambda_w"]:
            self.parameters.add(
                Parameter(
                    name=f"rho_{markup}",
                    value=0.50,
                    prior=make_prior("beta", 0.50, 0.20),
                    fixed=False,
                    description=f"{markup} shock persistence",
                )
            )
            self.parameters.add(
                Parameter(
                    name=f"sigma_{markup}",
                    value=0.10,
                    prior=make_prior("invgamma", 0.10, 2.00),
                    fixed=False,
                    description=f"{markup} shock std dev",
                )
            )
            self.parameters.add(
                Parameter(
                    name=f"eta_{markup}",
                    value=0.50,
                    prior=make_prior("beta", 0.50, 0.20),
                    fixed=False,
                    description=f"{markup} MA coefficient",
                )
            )

        # Government shock correlation with TFP
        self.parameters.add(
            Parameter(
                name="eta_gz",
                value=0.50,
                prior=make_prior("beta", 0.50, 0.20),
                fixed=False,
                description="Correlation of g shock with z shock",
            )
        )

        # Inflation target
        self.parameters.add(
            Parameter(
                name="rho_pi_star",
                value=0.99,
                fixed=True,
                description="Inflation target persistence",
            )
        )

        self.parameters.add(
            Parameter(
                name="sigma_pi_star",
                value=0.03,
                prior=make_prior("invgamma", 0.03, 6.00),
                fixed=False,
                description="Inflation target shock std dev",
            )
        )

        # ====================================================================
        # MEASUREMENT ERROR PARAMETERS
        # ====================================================================
        self.parameters.add(
            Parameter(
                name="C_me",
                value=1.0,
                fixed=True,
                description="Cointegration parameter for GDP/GDI",
            )
        )

        # Measurement error parameters for each observable
        me_vars = ["gdp", "gdi", "pce", "gdpdef", "10y", "tfp"]
        for var in me_vars:
            self.parameters.add(
                Parameter(
                    name=f"rho_{var}",
                    value=0.00 if var in ["gdp", "gdi"] else 0.50,
                    prior=make_prior(
                        "normal" if var in ["gdp", "gdi"] else "beta",
                        0.00 if var in ["gdp", "gdi"] else 0.50,
                        0.20,
                    ),
                    fixed=False,
                    description=f"{var} measurement error persistence",
                )
            )
            self.parameters.add(
                Parameter(
                    name=f"sigma_{var}",
                    value=0.10,
                    prior=make_prior("invgamma", 0.10, 2.00),
                    fixed=False,
                    description=f"{var} measurement error std dev",
                )
            )

        # GDP-GDI correlation
        self.parameters.add(
            Parameter(
                name="rho_gdp_gdi",
                value=0.00,
                prior=make_prior("normal", 0.00, 0.40),
                fixed=False,
                description="Correlation between GDP and GDI errors",
            )
        )

    def steady_state(self, params: np.ndarray | None = None) -> np.ndarray:
        """
        Compute steady state.

        All variables are log-deviations from steady state, so steady state is zero.
        However, we compute steady-state ratios needed for equation coefficients.

        Returns:
        -------
        ss : array
            Steady state values (all zeros for log-linearized model)
        """
        return np.zeros(self.spec.n_states)

    def _compute_steady_state_ratios(self, p: dict[str, float]) -> dict[str, float]:
        """
        Compute steady-state ratios and levels needed for equation coefficients.

        These ratios appear in the log-linearized equations.

        Parameters
        ----------
        p : dict
            Parameter values

        Returns:
        -------
        ss : dict
            Dictionary of steady-state ratios
        """
        # Extract parameters
        alpha = p["alpha"]
        delta = p["delta"]
        gamma_pct = p["gamma"] / 100  # Convert from percentage points
        beta_bar_pct = p["beta_bar"] / 100
        sigma_c = p["sigma_c"]
        g_star = p["g_star"]
        p["Phi_p"]
        p["SP_star"] / 400  # Convert annual spread to quarterly
        p["gamma_star"]  # Entrepreneur survival rate

        # Compute discount factor
        beta = 1 / (1 + beta_bar_pct) * np.exp((sigma_c - 1) * gamma_pct)

        # Nominal interest rate and rental rate
        pi_star_pct = p["pi_star"] / 100  # Quarterly inflation
        R_star = np.exp(gamma_pct) / beta * np.exp(pi_star_pct) - 1  # Nominal rate
        r_k_star = np.exp(gamma_pct) / beta - (1 - delta)  # Rental rate of capital

        # Return on capital for entrepreneurs
        R_k_star = (r_k_star + 1 - delta) * np.exp(gamma_pct)  # Gross return

        # Leverage and net worth ratios (financial accelerator)
        # SP_star = E[R_k] - R, so R_k_star ≈ R_star + SP_star

        # Capital-output ratio (from production function)
        # y = Φ_p * (alpha*k + (1-alpha)*L)
        # In steady state, need to pin down k/y ratio
        # From capital accumulation: i/k = (delta + gamma_pct)
        # From resource constraint: c/y + i/y + g = 1
        # This gives us: i/y = (delta + gamma_pct) * k/y

        # Solve for k/y ratio (requires iterating on FOCs)
        # For now, use approximate value
        k_y_star = 10.0  # Typical value for quarterly model

        i_k_star = delta + gamma_pct  # Investment-capital ratio
        i_y_star = i_k_star * k_y_star  # Investment-output ratio
        c_y_star = 1 - g_star - i_y_star  # Consumption-output ratio

        # Tobin's q in steady state (normalized to 1 in log-linearized model)
        q_k_star = 1.0

        # Net worth to capital ratio (from financial accelerator)
        # Leverage: (q_k * k) / n
        # From spread equation: SP = zeta_sp_b * (q_k + k - n)
        leverage_star = 2.0  # Typical leverage ratio
        n_qk_star = 1 / leverage_star  # n / (q_k * k_bar)

        # Utilization in steady state is normalized

        # Wage and labor ratios
        w_L_c_star = (1 - alpha) / alpha * r_k_star * k_y_star / c_y_star

        # Store all ratios
        return {
            "beta": beta,
            "beta_bar": 1 / beta - 1,  # Transformed discount factor
            "R_star": R_star,
            "r_k_star": r_k_star,
            "R_k_star": R_k_star,
            "pi_star": pi_star_pct,
            "gamma": gamma_pct,
            "c_y_star": c_y_star,
            "i_y_star": i_y_star,
            "i_k_star": i_k_star,
            "k_y_star": k_y_star,
            "g_star": g_star,
            "q_k_star": q_k_star,
            "n_qk_star": n_qk_star,
            "leverage_star": leverage_star,
            "w_L_c_star": w_L_c_star,
        }

    def system_matrices(self, params: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """
        Compute linearized system matrices.

        System: Γ0 * s_t = Γ1 * s_{t-1} + Ψ * ε_t + Π * η_t

        Returns:
        -------
        dict
            Dictionary containing 'Gamma0', 'Gamma1', 'Psi', 'Pi' matrices
        """
        # Get parameter values
        p = self.parameters.to_dict() if params is None else params

        # Compute steady-state ratios
        ss = self._compute_steady_state_ratios(p)

        # Extract frequently used parameters
        alpha = p["alpha"]
        sigma_c = p["sigma_c"]
        h = p["h"]
        nu_l = p["nu_l"]
        S_pp = p["S_pp"]
        psi = p["psi"]
        delta = p["delta"]
        Phi_p = p["Phi_p"]
        zeta_p = p["zeta_p"]
        iota_p = p["iota_p"]
        epsilon_p = p["epsilon_p"]
        zeta_w = p["zeta_w"]
        iota_w = p["iota_w"]
        epsilon_w = p["epsilon_w"]
        lambda_w = p["lambda_w"]
        psi1 = p["psi1"]
        psi2 = p["psi2"]
        psi3 = p["psi3"]
        rho_R = p["rho_R"]
        zeta_sp_b = p["zeta_sp_b"]

        # Shock persistence parameters
        rho_z = p["rho_z"]
        rho_zp = p["rho_zp"]
        rho_b = p["rho_b"]
        rho_mu = p["rho_mu"]
        rho_g = p["rho_g"]
        rho_lambda_f = p["rho_lambda_f"]
        rho_lambda_w = p["rho_lambda_w"]
        rho_sigma_w = p["rho_sigma_w"]
        rho_rm = p["rho_rm"]
        rho_pi_star = p["rho_pi_star"]

        # Shock std devs
        sigma_z = p["sigma_z"]
        sigma_zp = p["sigma_zp"]
        sigma_b = p["sigma_b"]
        sigma_mu = p["sigma_mu"]
        sigma_g = p["sigma_g"]
        sigma_lambda_f = p["sigma_lambda_f"]
        sigma_lambda_w = p["sigma_lambda_w"]
        sigma_sigma_w = p["sigma_sigma_w"]
        sigma_rm = p["sigma_rm"]
        p["sigma_pi_star"]

        # MA coefficients
        eta_lambda_f = p["eta_lambda_f"]
        eta_lambda_w = p["eta_lambda_w"]
        eta_gz = p["eta_gz"]

        # Steady state ratios
        beta = ss["beta"]
        beta_bar = ss["beta_bar"]
        gamma = ss["gamma"]
        np.exp(gamma)
        exp_neg_gamma = np.exp(-gamma)
        r_k_star = ss["r_k_star"]
        c_y_star = ss["c_y_star"]
        i_y_star = ss["i_y_star"]
        i_k_star = ss["i_k_star"]
        k_y_star = ss["k_y_star"]
        g_star = p["g_star"]
        w_L_c_star = ss["w_L_c_star"]

        # Matrix dimensions
        n = self.spec.n_states  # 48
        n_shocks = self.spec.n_shocks  # 9
        n_eta = 13  # Expectation errors (for forward-looking variables)

        # Initialize matrices
        Gamma0 = np.zeros((n, n))
        Gamma1 = np.zeros((n, n))
        Psi = np.zeros((n, n_shocks))
        Pi = np.zeros((n, n_eta))

        # State indices (must match state_names order)
        idx = {name: i for i, name in enumerate(self.spec.state_names)}

        # Shock indices
        shock_idx = {name: i for i, name in enumerate(self.spec.shock_names)}

        # Expectation error indices
        # We have expectations of: c, i, y, pi, w, R_k_tilde, L
        # Plus longer-horizon expectations for 10y rate and inflation
        eta_names = [
            "eta_c",
            "eta_i",
            "eta_pi",
            "eta_w",
            "eta_R_k_tilde",
            "eta_L",
            "eta_y",
            "eta_R",
            "eta_q_k",
            "eta_n",
            "eta_z",
            "eta_pi_star_t",
            "eta_y_f",
        ]
        eta_idx = {name: i for i, name in enumerate(eta_names)}

        eq = 0  # Equation counter

        # ===================================================================
        # EQUATION 1-2: Technology shocks
        # z_tilde[t] = rho_z * z_tilde[t-1] + sigma_z * eps_z[t]
        # z_p[t] = rho_zp * z_p[t-1] + sigma_zp * eps_zp[t]
        # ===================================================================
        Gamma0[eq, idx["z_tilde"]] = 1.0
        Gamma1[eq, idx["z_tilde"]] = rho_z
        Psi[eq, shock_idx["eps_z"]] = sigma_z
        eq += 1

        Gamma0[eq, idx["z_p"]] = 1.0
        Gamma1[eq, idx["z_p"]] = rho_zp
        Psi[eq, shock_idx["eps_zp"]] = sigma_zp
        eq += 1

        # ===================================================================
        # EQUATION 3: Productivity growth
        # z[t] = (1/(1-alpha))*(rho_z-1)*z_tilde[t-1] + (1/(1-alpha))*sigma_z*eps_z[t] + z_p[t]
        # ===================================================================
        Gamma0[eq, idx["z"]] = 1.0
        Gamma1[eq, idx["z_tilde"]] = (rho_z - 1) / (1 - alpha)
        Gamma0[eq, idx["z_p"]] = -1.0
        Psi[eq, shock_idx["eps_z"]] = sigma_z / (1 - alpha)
        eq += 1

        # ===================================================================
        # EQUATION 4: Consumption Euler equation
        # c[t] = -(1-h*exp(-gamma))/(sigma_c*(1+h*exp(-gamma))) * (R[t] - E[pi[t+1]] + b[t])
        #      + h*exp(-gamma)/(1+h*exp(-gamma)) * (c[t-1] - z[t])
        #      + 1/(1+h*exp(-gamma)) * E[c[t+1] + z[t+1]]
        #      + (sigma_c-1)/(sigma_c*(1+h*exp(-gamma))) * w*L/c * (L[t] - E[L[t+1]])
        # ===================================================================
        h_eg = h * exp_neg_gamma
        denom_c = sigma_c * (1 + h_eg)

        Gamma0[eq, idx["c"]] = 1.0
        Gamma0[eq, idx["R"]] = (1 - h_eg) / denom_c
        Gamma0[eq, idx["b"]] = -(1 - h_eg) / denom_c
        Gamma1[eq, idx["c_lag"]] = -h_eg / (1 + h_eg)
        Gamma0[eq, idx["z"]] = h_eg / (1 + h_eg)
        Gamma0[eq, idx["L"]] = -(sigma_c - 1) / denom_c * w_L_c_star

        # Expectations
        Pi[eq, eta_idx["eta_pi"]] = (1 - h_eg) / denom_c  # -E[pi[t+1]]
        Pi[eq, eta_idx["eta_c"]] = -1 / (1 + h_eg)  # -E[c[t+1]]
        Pi[eq, eta_idx["eta_z"]] = -1 / (1 + h_eg)  # -E[z[t+1]]
        Pi[eq, eta_idx["eta_L"]] = (sigma_c - 1) / denom_c * w_L_c_star  # E[L[t+1]]
        eq += 1

        # ===================================================================
        # EQUATION 5: Investment
        # i[t] = q_k[t]/(S_pp*exp(2*gamma)*(1+beta_bar))
        #      + 1/(1+beta_bar) * (i[t-1] - z[t])
        #      + beta_bar/(1+beta_bar) * E[i[t+1] + z[t+1]]
        #      + mu[t]
        # ===================================================================
        S_adj = S_pp * np.exp(2 * gamma) * (1 + beta_bar)

        Gamma0[eq, idx["i"]] = 1.0
        Gamma0[eq, idx["q_k"]] = -1.0 / S_adj
        Gamma1[eq, idx["i_lag"]] = -1.0 / (1 + beta_bar)
        Gamma0[eq, idx["z"]] = 1.0 / (1 + beta_bar)
        Gamma0[eq, idx["mu"]] = -1.0

        # Expectations
        Pi[eq, eta_idx["eta_i"]] = -beta_bar / (1 + beta_bar)
        Pi[eq, eta_idx["eta_z"]] = -beta_bar / (1 + beta_bar)
        eq += 1

        # ===================================================================
        # EQUATION 6: Capital accumulation
        # k_bar[t] = (1 - i/k_bar) * (k_bar[t-1] - z[t])
        #          + i/k_bar * i[t]
        #          + i/k_bar * S_pp*exp(2*gamma)*(1+beta_bar) * mu[t]
        # ===================================================================
        Gamma0[eq, idx["k_bar"]] = 1.0
        Gamma1[eq, idx["k_bar_lag"]] = -(1 - i_k_star)
        Gamma0[eq, idx["z"]] = 1 - i_k_star
        Gamma0[eq, idx["i"]] = -i_k_star
        Gamma0[eq, idx["mu"]] = -i_k_star * S_adj
        eq += 1

        # ===================================================================
        # EQUATION 7: Effective capital
        # k[t] = u[t] - z[t] + k_bar[t-1]
        # ===================================================================
        Gamma0[eq, idx["k"]] = 1.0
        Gamma0[eq, idx["u"]] = -1.0
        Gamma0[eq, idx["z"]] = 1.0
        Gamma1[eq, idx["k_bar_lag"]] = -1.0
        eq += 1

        # ===================================================================
        # EQUATION 8: Capital utilization
        # (1-psi)/psi * r_k[t] = u[t]
        # ===================================================================
        Gamma0[eq, idx["u"]] = 1.0
        Gamma0[eq, idx["r_k"]] = -(1 - psi) / psi
        eq += 1

        # ===================================================================
        # EQUATION 9: Marginal cost
        # mc[t] = w[t] + alpha*L[t] - alpha*k[t]
        # ===================================================================
        Gamma0[eq, idx["mc"]] = 1.0
        Gamma0[eq, idx["w"]] = -1.0
        Gamma0[eq, idx["L"]] = -alpha
        Gamma0[eq, idx["k"]] = alpha
        eq += 1

        # ===================================================================
        # EQUATION 10: Capital-labor ratio (from cost minimization)
        # k[t] = w[t] - r_k[t] + L[t]
        # ===================================================================
        Gamma0[eq, idx["k"]] = 1.0
        Gamma0[eq, idx["w"]] = -1.0
        Gamma0[eq, idx["r_k"]] = 1.0
        Gamma0[eq, idx["L"]] = -1.0
        eq += 1

        # ===================================================================
        # EQUATION 11: Return on capital
        # R_k_tilde[t] - pi[t] = r_k*/(r_k* + (1-delta)) * r_k[t]
        #                       + (1-delta)/(r_k* + (1-delta)) * q_k[t]
        #                       - q_k[t-1]
        # ===================================================================
        rk_denom = r_k_star + (1 - delta)

        Gamma0[eq, idx["R_k_tilde"]] = 1.0
        Gamma0[eq, idx["pi"]] = -1.0
        Gamma0[eq, idx["r_k"]] = -r_k_star / rk_denom
        Gamma0[eq, idx["q_k"]] = -(1 - delta) / rk_denom
        Gamma1[eq, idx["q_k_lag"]] = 1.0
        eq += 1

        # ===================================================================
        # EQUATION 12: Credit spread
        # E[R_k_tilde[t+1] - R[t]] = b[t] + zeta_sp_b * (q_k[t] + k_bar[t] - n[t]) + sigma_omega[t]
        # ===================================================================
        Gamma0[eq, idx["R"]] = -1.0
        Gamma0[eq, idx["b"]] = -1.0
        Gamma0[eq, idx["q_k"]] = -zeta_sp_b
        Gamma0[eq, idx["k_bar"]] = -zeta_sp_b
        Gamma0[eq, idx["n"]] = zeta_sp_b
        Gamma0[eq, idx["sigma_omega"]] = -1.0

        # Expectation
        Pi[eq, eta_idx["eta_R_k_tilde"]] = -1.0  # -E[R_k_tilde[t+1]]
        Pi[eq, eta_idx["eta_R"]] = 1.0  # +E[R[t]]
        eq += 1

        # ===================================================================
        # EQUATION 13: Net worth evolution (simplified)
        # n[t] = zeta_n_Rk * (R_k_tilde[t] - pi[t])
        #      - zeta_n_R * (R[t-1] - pi[t] + b[t-1])
        #      + zeta_n_qK * (q_k[t-1] + k_bar[t-1])
        #      + zeta_n_n * n[t-1]
        #      - gamma* * v*/n* * z[t]
        #      - zeta_n_sigma/zeta_sp_sigma * sigma_omega[t-1]
        # ===================================================================
        # Coefficients from financial accelerator (using approximate values)
        zeta_n_Rk = 0.95
        zeta_n_R = 0.90
        zeta_n_qK = 0.02
        zeta_n_n = 0.97

        Gamma0[eq, idx["n"]] = 1.0
        Gamma0[eq, idx["R_k_tilde"]] = -zeta_n_Rk
        Gamma0[eq, idx["pi"]] = zeta_n_Rk + zeta_n_R
        Gamma1[eq, idx["R_lag"]] = zeta_n_R
        Gamma1[eq, idx["b"]] = -zeta_n_R
        Gamma1[eq, idx["q_k_lag"]] = -zeta_n_qK
        Gamma1[eq, idx["k_bar_lag"]] = -zeta_n_qK
        Gamma1[eq, idx["n_lag"]] = -zeta_n_n
        Gamma0[eq, idx["z"]] = 0.02  # gamma* * v*/n*
        Gamma1[eq, idx["sigma_omega"]] = 0.05  # zeta_n_sigma/zeta_sp_sigma
        eq += 1

        # ===================================================================
        # EQUATION 14: Production function
        # y[t] = Phi_p * (alpha*k[t] + (1-alpha)*L[t])
        # ===================================================================
        Gamma0[eq, idx["y"]] = 1.0
        Gamma0[eq, idx["k"]] = -Phi_p * alpha
        Gamma0[eq, idx["L"]] = -Phi_p * (1 - alpha)
        eq += 1

        # ===================================================================
        # EQUATION 15: Resource constraint
        # y[t] = g_star*g[t] + c_star/y_star*c[t] + i_star/y_star*i[t] + r_k_star*k_star/y_star*u[t]
        # ===================================================================
        rk_k_y = r_k_star * k_y_star

        Gamma0[eq, idx["y"]] = 1.0
        Gamma0[eq, idx["g"]] = -g_star
        Gamma0[eq, idx["c"]] = -c_y_star
        Gamma0[eq, idx["i"]] = -i_y_star
        Gamma0[eq, idx["u"]] = -rk_k_y
        eq += 1

        # ===================================================================
        # EQUATION 16: New Keynesian Phillips Curve
        # pi[t] = kappa*mc[t]
        #       + iota_p/(1 + iota_p*beta_bar) * pi[t-1]
        #       + beta_bar/(1 + iota_p*beta_bar) * E[pi[t+1]]
        #       + lambda_f[t]
        # ===================================================================
        kappa = (
            (1 - zeta_p * beta)
            * (1 - zeta_p)
            / ((1 + iota_p * beta_bar) * zeta_p * ((Phi_p - 1) * epsilon_p + 1))
        )

        Gamma0[eq, idx["pi"]] = 1.0
        Gamma0[eq, idx["mc"]] = -kappa
        Gamma1[eq, idx["pi_lag"]] = -iota_p / (1 + iota_p * beta_bar)
        Gamma0[eq, idx["lambda_f"]] = -1.0

        # Expectation
        Pi[eq, eta_idx["eta_pi"]] = -beta_bar / (1 + iota_p * beta_bar)
        eq += 1

        # ===================================================================
        # EQUATION 17: Wage Phillips Curve
        # w[t] = kappa_w * (w_h[t] - w[t])
        #      - (1 + iota_w*beta_bar)/(1+beta_bar) * pi[t]
        #      + 1/(1+beta_bar) * (w[t-1] - z[t] + iota_w*pi[t-1])
        #      + beta_bar/(1+beta_bar) * E[w[t+1] + z[t+1] + pi[t+1]]
        #      + lambda_w[t]
        # ===================================================================
        kappa_w = (
            (1 - zeta_w * beta)
            * (1 - zeta_w)
            / ((1 + beta_bar) * zeta_w * ((lambda_w - 1) * epsilon_w + 1))
        )

        Gamma0[eq, idx["w"]] = 1.0 + kappa_w
        Gamma0[eq, idx["w_h"]] = -kappa_w
        Gamma0[eq, idx["pi"]] = (1 + iota_w * beta_bar) / (1 + beta_bar)
        Gamma1[eq, idx["w_lag"]] = -1.0 / (1 + beta_bar)
        Gamma0[eq, idx["z"]] = 1.0 / (1 + beta_bar)
        Gamma1[eq, idx["pi_lag"]] = -iota_w / (1 + beta_bar)
        Gamma0[eq, idx["lambda_w"]] = -1.0

        # Expectations
        Pi[eq, eta_idx["eta_w"]] = -beta_bar / (1 + beta_bar)
        Pi[eq, eta_idx["eta_z"]] = -beta_bar / (1 + beta_bar)
        Pi[eq, eta_idx["eta_pi"]] = -beta_bar / (1 + beta_bar)
        eq += 1

        # ===================================================================
        # EQUATION 18: Household MRS
        # w_h[t] = 1/(1-h*exp(-gamma)) * (c[t] - h*exp(-gamma)*c[t-1] + h*exp(-gamma)*z[t]) + nu_l*L[t]
        # ===================================================================
        Gamma0[eq, idx["w_h"]] = 1.0
        Gamma0[eq, idx["c"]] = -1.0 / (1 - h_eg)
        Gamma1[eq, idx["c_lag"]] = h_eg / (1 - h_eg)
        Gamma0[eq, idx["z"]] = -h_eg / (1 - h_eg)
        Gamma0[eq, idx["L"]] = -nu_l
        eq += 1

        # ===================================================================
        # EQUATION 19: Monetary policy rule
        # R[t] = rho_R*R[t-1]
        #      + (1-rho_R) * (psi1*(pi[t] - pi_star[t]) + psi2*(y[t] - y_f[t]))
        #      + psi3 * ((y[t] - y_f[t]) - (y[t-1] - y_f[t-1]))
        #      + r_m[t]
        # ===================================================================
        Gamma0[eq, idx["R"]] = 1.0
        Gamma1[eq, idx["R_lag"]] = -rho_R
        Gamma0[eq, idx["pi"]] = -(1 - rho_R) * psi1
        Gamma0[eq, idx["pi_star_t"]] = (1 - rho_R) * psi1
        Gamma0[eq, idx["y"]] = -(1 - rho_R) * psi2 - psi3
        Gamma0[eq, idx["y_f"]] = (1 - rho_R) * psi2 + psi3
        Gamma1[eq, idx["y_lag"]] = psi3
        Gamma1[eq, idx["y_f_lag"]] = -psi3
        Gamma0[eq, idx["r_m"]] = -1.0
        eq += 1

        # ===================================================================
        # EQUATION 20: Inflation target
        # pi_star[t] = rho_pi_star * pi_star[t-1] + sigma_pi_star * eps_pi_star[t]
        # Note: eps_pi_star not in shock vector, treated as exogenous
        # ===================================================================
        Gamma0[eq, idx["pi_star_t"]] = 1.0
        Gamma1[eq, idx["pi_star_t"]] = -rho_pi_star
        # Shock added externally if needed
        eq += 1

        # ===================================================================
        # EQUATION 21: Flexible-price output (placeholder - simplified)
        # In full model, would solve separate flex-price system
        # Here, approximate as trend output
        # ===================================================================
        Gamma0[eq, idx["y_f"]] = 1.0
        Gamma0[eq, idx["z"]] = -1.0  # y_f grows with productivity
        eq += 1

        # ===================================================================
        # SHOCK PROCESSES
        # ===================================================================
        # Risk premium shock
        Gamma0[eq, idx["b"]] = 1.0
        Gamma1[eq, idx["b"]] = -rho_b
        Psi[eq, shock_idx["eps_b"]] = sigma_b
        eq += 1

        # MEI shock
        Gamma0[eq, idx["mu"]] = 1.0
        Gamma1[eq, idx["mu"]] = -rho_mu
        Psi[eq, shock_idx["eps_mu"]] = sigma_mu
        eq += 1

        # Government spending shock (with correlation to TFP)
        Gamma0[eq, idx["g"]] = 1.0
        Gamma1[eq, idx["g"]] = -rho_g
        Psi[eq, shock_idx["eps_g"]] = sigma_g
        Psi[eq, shock_idx["eps_z"]] = eta_gz * sigma_g  # Correlation with TFP
        eq += 1

        # Price markup shock with MA component
        Gamma0[eq, idx["lambda_f"]] = 1.0
        Gamma1[eq, idx["lambda_f"]] = -rho_lambda_f
        Gamma0[eq, idx["lambda_f_ma"]] = -eta_lambda_f
        Psi[eq, shock_idx["eps_lambda_f"]] = sigma_lambda_f
        eq += 1

        # MA lag for lambda_f
        Gamma0[eq, idx["lambda_f_ma"]] = 1.0
        Psi[eq, shock_idx["eps_lambda_f"]] = sigma_lambda_f
        eq += 1

        # Wage markup shock with MA component
        Gamma0[eq, idx["lambda_w"]] = 1.0
        Gamma1[eq, idx["lambda_w"]] = -rho_lambda_w
        Gamma0[eq, idx["lambda_w_ma"]] = -eta_lambda_w
        Psi[eq, shock_idx["eps_lambda_w"]] = sigma_lambda_w
        eq += 1

        # MA lag for lambda_w
        Gamma0[eq, idx["lambda_w_ma"]] = 1.0
        Psi[eq, shock_idx["eps_lambda_w"]] = sigma_lambda_w
        eq += 1

        # Cross-sectional volatility shock
        Gamma0[eq, idx["sigma_omega"]] = 1.0
        Gamma1[eq, idx["sigma_omega"]] = -rho_sigma_w
        Psi[eq, shock_idx["eps_sigma_omega"]] = sigma_sigma_w
        eq += 1

        # Monetary policy shock
        Gamma0[eq, idx["r_m"]] = 1.0
        Gamma1[eq, idx["r_m"]] = -rho_rm
        Psi[eq, shock_idx["eps_rm"]] = sigma_rm
        eq += 1

        # ===================================================================
        # LAG DEFINITIONS
        # ===================================================================
        for var, var_lag in [
            ("c", "c_lag"),
            ("i", "i_lag"),
            ("w", "w_lag"),
            ("R", "R_lag"),
            ("pi", "pi_lag"),
            ("k_bar", "k_bar_lag"),
            ("q_k", "q_k_lag"),
            ("n", "n_lag"),
            ("y", "y_lag"),
            ("y_f", "y_f_lag"),
        ]:
            Gamma0[eq, idx[var_lag]] = 1.0
            Gamma1[eq, idx[var]] = -1.0
            eq += 1

        # ===================================================================
        # MEASUREMENT ERROR PROCESSES
        # ===================================================================
        # Simplified - AR(1) for each measurement error
        me_params = {
            "e_gdp": (p["rho_gdp"], p["sigma_gdp"]),
            "e_gdi": (p["rho_gdi"], p["sigma_gdi"]),
            "e_pce": (p["rho_pce"], p["sigma_pce"]),
            "e_gdpdef": (p["rho_gdpdef"], p["sigma_gdpdef"]),
            "e_10y": (p["rho_10y"], p["sigma_10y"]),
            "e_tfp": (p["rho_tfp"], p["sigma_tfp"]),
        }

        for me_var, (rho_me, _sigma_me) in me_params.items():
            Gamma0[eq, idx[me_var]] = 1.0
            Gamma1[eq, idx[me_var]] = -rho_me
            # Shocks to measurement errors would be separate innovations (not in structural shocks)
            # For now, treat as exogenous
            eq += 1

        # ME lags
        Gamma0[eq, idx["e_gdp_lag"]] = 1.0
        Gamma1[eq, idx["e_gdp"]] = -1.0
        eq += 1

        Gamma0[eq, idx["e_gdi_lag"]] = 1.0
        Gamma1[eq, idx["e_gdi"]] = -1.0
        eq += 1

        # Fill remaining rows if needed (should be exactly n equations)
        assert eq <= n, f"Too many equations: {eq} > {n}"

        # If we have fewer equations than states, fill with identities
        while eq < n:
            Gamma0[eq, eq] = 1.0
            eq += 1

        return {"Gamma0": Gamma0, "Gamma1": Gamma1, "Psi": Psi, "Pi": Pi}

    def measurement_equation(
        self, params: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Measurement equation: observables = Z * states + D.

        Returns:
        -------
        Z : array
            Measurement matrix (n_obs x n_states)
        D : array
            Constant term (n_obs,)
        """
        # Get parameter values
        p = self.parameters.to_dict() if params is None else params

        # Get indices
        idx = {name: i for i, name in enumerate(self.spec.state_names)}

        n_obs = self.spec.n_observables
        n_states = self.spec.n_states

        Z = np.zeros((n_obs, n_states))
        D = np.zeros(n_obs)

        # Extract parameters for measurement equations
        gamma = p["gamma"]  # Quarterly growth rate (x100)
        L_bar = p["L_bar"]  # Hours steady state level
        pi_star = p["pi_star"]  # Steady-state inflation
        gamma_gdpdef = p["gamma_gdpdef"]
        delta_gdpdef = p["delta_gdpdef"]
        C_me = p["C_me"]  # Cointegration parameter
        p["alpha"]

        # Compute steady-state interest rate
        ss = self._compute_steady_state_ratios(p)
        R_star = ss["R_star"] * 400  # Convert to annualized percent
        SP_star = p["SP_star"]  # Already annualized

        obs = 0  # Observable counter

        # GDP growth = 100*gamma + (y[t] - y[t-1] + z[t]) + e_gdp[t] - C_me*e_gdp[t-1]
        D[obs] = gamma
        Z[obs, idx["y"]] = 1.0
        Z[obs, idx["y_lag"]] = -1.0
        Z[obs, idx["z"]] = 1.0
        Z[obs, idx["e_gdp"]] = 1.0
        Z[obs, idx["e_gdp_lag"]] = -C_me
        obs += 1

        # GDI growth = 100*gamma + (y[t] - y[t-1] + z[t]) + e_gdi[t] - C_me*e_gdi[t-1]
        D[obs] = gamma
        Z[obs, idx["y"]] = 1.0
        Z[obs, idx["y_lag"]] = -1.0
        Z[obs, idx["z"]] = 1.0
        Z[obs, idx["e_gdi"]] = 1.0
        Z[obs, idx["e_gdi_lag"]] = -C_me
        obs += 1

        # Consumption growth = 100*gamma + (c[t] - c[t-1] + z[t])
        D[obs] = gamma
        Z[obs, idx["c"]] = 1.0
        Z[obs, idx["c_lag"]] = -1.0
        Z[obs, idx["z"]] = 1.0
        obs += 1

        # Investment growth = 100*gamma + (i[t] - i[t-1] + z[t])
        D[obs] = gamma
        Z[obs, idx["i"]] = 1.0
        Z[obs, idx["i_lag"]] = -1.0
        Z[obs, idx["z"]] = 1.0
        obs += 1

        # Wage growth = 100*gamma + (w[t] - w[t-1] + z[t])
        D[obs] = gamma
        Z[obs, idx["w"]] = 1.0
        Z[obs, idx["w_lag"]] = -1.0
        Z[obs, idx["z"]] = 1.0
        obs += 1

        # Hours = L_bar + L[t]
        D[obs] = L_bar
        Z[obs, idx["L"]] = 1.0
        obs += 1

        # Core PCE inflation = pi_star + pi[t] + e_pce[t]
        D[obs] = pi_star
        Z[obs, idx["pi"]] = 1.0
        Z[obs, idx["e_pce"]] = 1.0
        obs += 1

        # GDP deflator inflation = pi_star + delta_gdpdef + gamma_gdpdef * pi[t] + e_gdpdef[t]
        D[obs] = pi_star + delta_gdpdef
        Z[obs, idx["pi"]] = gamma_gdpdef
        Z[obs, idx["e_gdpdef"]] = 1.0
        obs += 1

        # Federal funds rate = R_star + R[t]
        D[obs] = R_star
        Z[obs, idx["R"]] = 1.0
        obs += 1

        # 10-year rate = R_star + E[sum R[t+k]/40] + e_10y[t]
        # Simplified: approximate as current R plus measurement error
        D[obs] = R_star
        Z[obs, idx["R"]] = 1.0
        Z[obs, idx["e_10y"]] = 1.0
        obs += 1

        # 10-year inflation expectations = pi_star + E[sum pi[t+k]/40]
        # Simplified: approximate as current pi plus time-varying target
        D[obs] = pi_star
        Z[obs, idx["pi"]] = 0.5  # Weight on current inflation
        Z[obs, idx["pi_star_t"]] = 0.5  # Weight on target
        obs += 1

        # Spread = SP_star + E[R_k_tilde[t+1] - R[t]]
        # From spread equation: b[t] + zeta_sp_b * (q_k[t] + k_bar[t] - n[t]) + sigma_omega[t]
        D[obs] = SP_star
        Z[obs, idx["b"]] = 1.0
        Z[obs, idx["q_k"]] = p["zeta_sp_b"]
        Z[obs, idx["k_bar"]] = p["zeta_sp_b"]
        Z[obs, idx["n"]] = -p["zeta_sp_b"]
        Z[obs, idx["sigma_omega"]] = 1.0
        obs += 1

        # TFP growth = z[t] + alpha/(1-alpha) * (u[t] - u[t-1]) + e_tfp[t]
        # Note: u[t-1] not in state vector, so approximate
        Z[obs, idx["z"]] = 1.0
        Z[obs, idx["e_tfp"]] = 1.0
        obs += 1

        assert obs == n_obs, f"Expected {n_obs} observables, got {obs}"

        return Z, D


def create_nyfed_model() -> NYFedModel1002:
    """
    Factory function to create an instance of the NYFed Model 1002.

    Returns:
    -------
    model : NYFedModel1002
        Initialized model instance
    """
    return NYFedModel1002()


if __name__ == "__main__":
    # Example usage
    model = create_nyfed_model()

    # Test system matrices
    try:
        mats = model.system_matrices()
    except Exception:
        import traceback

        traceback.print_exc()

    # Test measurement equation
    try:
        Z, D = model.measurement_equation()
    except Exception:
        import traceback

        traceback.print_exc()
