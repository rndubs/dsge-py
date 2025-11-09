"""
FRBNY DSGE Model 1002

This module implements the New York Federal Reserve DSGE model (version 1002)
as documented in "DSGE Model Documentation" (FRBNY, March 3, 2021).

The model is a medium-scale New Keynesian DSGE model with financial frictions,
based on Smets-Wouters (2007), Christiano et al. (2005), and Bernanke et al. (1999).

Key features:
- Financial accelerator with credit frictions
- Time-varying inflation target
- Habit formation in consumption
- Variable capital utilization
- Investment adjustment costs
- Calvo wage and price rigidities with indexation
- Kimball aggregator for goods and labor
- Anticipated policy shocks (forward guidance)

References:
- Del Negro, M., M. P. Giannoni, and F. Schorfheide (2015)
- Smets, F. and R. Wouters (2007)
- Christiano, L. J., M. Eichenbaum, and C. L. Evans (2005)
- Bernanke, B. S., M. Gertler, and S. Gilchrist (1999)
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from src.dsge.models.base import DSGEModel, Parameter, PriorDistribution


@dataclass
class NYFedModel1002(DSGEModel):
    """
    FRBNY DSGE Model 1002 (March 2021 specification)

    A medium-scale New Keynesian model with financial frictions.
    """

    def __init__(self):
        """Initialize the NYFed Model 1002 with all parameters and equations."""
        super().__init__(name="NYFed_Model_1002")

        # Define parameters with priors from Appendix D of documentation
        self._define_parameters()

        # Define state and control variables
        self._define_variables()

    def _define_parameters(self):
        """Define all model parameters with their priors."""

        # =================================================================
        # POLICY PARAMETERS
        # =================================================================
        self.add_parameter(Parameter(
            name="psi1",
            description="Taylor rule coefficient on inflation",
            prior=PriorDistribution("normal", mean=1.50, std=0.25)
        ))

        self.add_parameter(Parameter(
            name="psi2",
            description="Taylor rule coefficient on output gap",
            prior=PriorDistribution("normal", mean=0.12, std=0.05)
        ))

        self.add_parameter(Parameter(
            name="psi3",
            description="Taylor rule coefficient on output gap growth",
            prior=PriorDistribution("normal", mean=0.12, std=0.05)
        ))

        self.add_parameter(Parameter(
            name="rho_R",
            description="Interest rate smoothing",
            prior=PriorDistribution("beta", mean=0.75, std=0.10)
        ))

        self.add_parameter(Parameter(
            name="rho_rm",
            description="Monetary policy shock persistence",
            prior=PriorDistribution("beta", mean=0.50, std=0.20)
        ))

        self.add_parameter(Parameter(
            name="sigma_rm",
            description="Monetary policy shock std dev",
            prior=PriorDistribution("inv_gamma", mean=0.10, std=2.00)
        ))

        # =================================================================
        # NOMINAL RIGIDITIES
        # =================================================================
        self.add_parameter(Parameter(
            name="zeta_p",
            description="Calvo parameter for prices",
            prior=PriorDistribution("beta", mean=0.50, std=0.10)
        ))

        self.add_parameter(Parameter(
            name="iota_p",
            description="Price indexation",
            prior=PriorDistribution("beta", mean=0.50, std=0.15)
        ))

        self.add_parameter(Parameter(
            name="epsilon_p",
            description="Curvature of Kimball aggregator (prices)",
            value=10.0,  # Fixed
            is_calibrated=True
        ))

        self.add_parameter(Parameter(
            name="zeta_w",
            description="Calvo parameter for wages",
            prior=PriorDistribution("beta", mean=0.50, std=0.10)
        ))

        self.add_parameter(Parameter(
            name="iota_w",
            description="Wage indexation",
            prior=PriorDistribution("beta", mean=0.50, std=0.15)
        ))

        self.add_parameter(Parameter(
            name="epsilon_w",
            description="Curvature of Kimball aggregator (wages)",
            value=10.0,  # Fixed
            is_calibrated=True
        ))

        # =================================================================
        # STEADY STATE AND ENDOGENOUS PROPAGATION
        # =================================================================
        self.add_parameter(Parameter(
            name="gamma",
            description="Steady-state growth rate (quarterly, x100)",
            prior=PriorDistribution("normal", mean=0.40, std=0.10)
        ))

        self.add_parameter(Parameter(
            name="alpha",
            description="Capital share",
            prior=PriorDistribution("normal", mean=0.30, std=0.05)
        ))

        self.add_parameter(Parameter(
            name="beta_bar",
            description="Discount factor transformation",
            prior=PriorDistribution("gamma", mean=0.25, std=0.10),
            transform="beta_transform"  # 100*(beta^-1 - 1)
        ))

        self.add_parameter(Parameter(
            name="sigma_c",
            description="Risk aversion / IES",
            prior=PriorDistribution("normal", mean=1.50, std=0.37)
        ))

        self.add_parameter(Parameter(
            name="h",
            description="Habit persistence",
            prior=PriorDistribution("beta", mean=0.70, std=0.10)
        ))

        self.add_parameter(Parameter(
            name="nu_l",
            description="Inverse Frisch elasticity",
            prior=PriorDistribution("normal", mean=2.00, std=0.75)
        ))

        self.add_parameter(Parameter(
            name="S_double_prime",
            description="Investment adjustment cost",
            prior=PriorDistribution("normal", mean=4.00, std=1.50)
        ))

        self.add_parameter(Parameter(
            name="psi",
            description="Capital utilization cost",
            prior=PriorDistribution("beta", mean=0.50, std=0.15)
        ))

        self.add_parameter(Parameter(
            name="delta",
            description="Depreciation rate",
            value=0.03,  # Fixed quarterly depreciation
            is_calibrated=True
        ))

        self.add_parameter(Parameter(
            name="Phi_p",
            description="Fixed cost in production",
            prior=PriorDistribution("normal", mean=1.25, std=0.12)
        ))

        self.add_parameter(Parameter(
            name="pi_star",
            description="Steady-state inflation (quarterly, net, x100)",
            value=0.50,  # Fixed at 2% annualized
            is_calibrated=True
        ))

        self.add_parameter(Parameter(
            name="lambda_w",
            description="Wage markup",
            value=1.50,  # Fixed
            is_calibrated=True
        ))

        self.add_parameter(Parameter(
            name="g_star",
            description="Steady-state government spending share",
            value=0.18,  # Fixed
            is_calibrated=True
        ))

        self.add_parameter(Parameter(
            name="L_bar",
            description="Steady-state hours (index level)",
            prior=PriorDistribution("normal", mean=-45.00, std=5.00)
        ))

        # GDP deflator measurement equation parameters
        self.add_parameter(Parameter(
            name="gamma_gdpdef",
            description="GDP deflator loading on model inflation",
            prior=PriorDistribution("normal", mean=1.00, std=2.00)
        ))

        self.add_parameter(Parameter(
            name="delta_gdpdef",
            description="GDP deflator steady-state diff from core PCE",
            prior=PriorDistribution("normal", mean=0.00, std=2.00)
        ))

        # =================================================================
        # FINANCIAL FRICTIONS
        # =================================================================
        self.add_parameter(Parameter(
            name="F_omega",
            description="Steady-state default probability",
            value=0.03,  # Fixed at 3%
            is_calibrated=True
        ))

        self.add_parameter(Parameter(
            name="zeta_sp_b",
            description="Elasticity of spread w.r.t. leverage",
            prior=PriorDistribution("beta", mean=0.05, std=0.005)
        ))

        self.add_parameter(Parameter(
            name="SP_star",
            description="Steady-state spread (annualized)",
            prior=PriorDistribution("gamma", mean=2.00, std=0.10)
        ))

        self.add_parameter(Parameter(
            name="gamma_star",
            description="Entrepreneur survival rate",
            value=0.99,  # Fixed
            is_calibrated=True
        ))

        # =================================================================
        # EXOGENOUS SHOCK PROCESSES
        # =================================================================
        # Technology shocks
        self.add_parameter(Parameter(
            name="rho_z",
            description="Stationary TFP persistence",
            prior=PriorDistribution("beta", mean=0.50, std=0.20)
        ))

        self.add_parameter(Parameter(
            name="sigma_z",
            description="Stationary TFP shock std dev",
            prior=PriorDistribution("inv_gamma", mean=0.10, std=2.00)
        ))

        self.add_parameter(Parameter(
            name="rho_zp",
            description="Stochastic trend TFP persistence",
            prior=PriorDistribution("beta", mean=0.50, std=0.20)
        ))

        self.add_parameter(Parameter(
            name="sigma_zp",
            description="Stochastic trend TFP shock std dev",
            prior=PriorDistribution("inv_gamma", mean=0.10, std=2.00)
        ))

        # Risk premium shock
        self.add_parameter(Parameter(
            name="rho_b",
            description="Risk premium shock persistence",
            prior=PriorDistribution("beta", mean=0.50, std=0.20)
        ))

        self.add_parameter(Parameter(
            name="sigma_b",
            description="Risk premium shock std dev",
            prior=PriorDistribution("inv_gamma", mean=0.10, std=2.00)
        ))

        # MEI (marginal efficiency of investment) shock
        self.add_parameter(Parameter(
            name="rho_mu",
            description="MEI shock persistence",
            prior=PriorDistribution("beta", mean=0.50, std=0.20)
        ))

        self.add_parameter(Parameter(
            name="sigma_mu",
            description="MEI shock std dev",
            prior=PriorDistribution("inv_gamma", mean=0.10, std=2.00)
        ))

        # Government spending shock
        self.add_parameter(Parameter(
            name="rho_g",
            description="Government spending shock persistence",
            prior=PriorDistribution("beta", mean=0.50, std=0.20)
        ))

        self.add_parameter(Parameter(
            name="sigma_g",
            description="Government spending shock std dev",
            prior=PriorDistribution("inv_gamma", mean=0.10, std=2.00)
        ))

        self.add_parameter(Parameter(
            name="eta_gz",
            description="Correlation of g shock with z shock",
            prior=PriorDistribution("beta", mean=0.50, std=0.20)
        ))

        # Price markup shock
        self.add_parameter(Parameter(
            name="rho_lambda_f",
            description="Price markup shock persistence",
            prior=PriorDistribution("beta", mean=0.50, std=0.20)
        ))

        self.add_parameter(Parameter(
            name="sigma_lambda_f",
            description="Price markup shock std dev",
            prior=PriorDistribution("inv_gamma", mean=0.10, std=2.00)
        ))

        self.add_parameter(Parameter(
            name="eta_lambda_f",
            description="Price markup MA coefficient",
            prior=PriorDistribution("beta", mean=0.50, std=0.20)
        ))

        # Wage markup shock
        self.add_parameter(Parameter(
            name="rho_lambda_w",
            description="Wage markup shock persistence",
            prior=PriorDistribution("beta", mean=0.50, std=0.20)
        ))

        self.add_parameter(Parameter(
            name="sigma_lambda_w",
            description="Wage markup shock std dev",
            prior=PriorDistribution("inv_gamma", mean=0.10, std=2.00)
        ))

        self.add_parameter(Parameter(
            name="eta_lambda_w",
            description="Wage markup MA coefficient",
            prior=PriorDistribution("beta", mean=0.50, std=0.20)
        ))

        # Cross-sectional volatility shock
        self.add_parameter(Parameter(
            name="rho_sigma_w",
            description="Volatility shock persistence",
            prior=PriorDistribution("beta", mean=0.75, std=0.15)
        ))

        self.add_parameter(Parameter(
            name="sigma_sigma_w",
            description="Volatility shock std dev",
            prior=PriorDistribution("inv_gamma", mean=0.05, std=4.00)
        ))

        # Time-varying inflation target
        self.add_parameter(Parameter(
            name="rho_pi_star",
            description="Inflation target persistence",
            value=0.99,  # Fixed to be highly persistent
            is_calibrated=True
        ))

        self.add_parameter(Parameter(
            name="sigma_pi_star",
            description="Inflation target shock std dev",
            prior=PriorDistribution("inv_gamma", mean=0.03, std=6.00)
        ))

        # =================================================================
        # MEASUREMENT ERROR PARAMETERS
        # =================================================================
        # GDP measurement
        self.add_parameter(Parameter(
            name="C_me",
            description="Cointegration parameter for GDP/GDI",
            value=1.0,  # Fixed
            is_calibrated=True
        ))

        self.add_parameter(Parameter(
            name="rho_gdp",
            description="GDP measurement error persistence",
            prior=PriorDistribution("normal", mean=0.00, std=0.20)
        ))

        self.add_parameter(Parameter(
            name="sigma_gdp",
            description="GDP measurement error std dev",
            prior=PriorDistribution("inv_gamma", mean=0.10, std=2.00)
        ))

        # GDI measurement
        self.add_parameter(Parameter(
            name="rho_gdi",
            description="GDI measurement error persistence",
            prior=PriorDistribution("normal", mean=0.00, std=0.20)
        ))

        self.add_parameter(Parameter(
            name="rho_gdp_gdi",
            description="Correlation between GDP and GDI errors",
            prior=PriorDistribution("normal", mean=0.00, std=0.40)
        ))

        self.add_parameter(Parameter(
            name="sigma_gdi",
            description="GDI measurement error std dev",
            prior=PriorDistribution("inv_gamma", mean=0.10, std=2.00)
        ))

        # Other measurement errors
        self.add_parameter(Parameter(
            name="rho_pce",
            description="PCE measurement error persistence",
            prior=PriorDistribution("beta", mean=0.50, std=0.20)
        ))

        self.add_parameter(Parameter(
            name="sigma_pce",
            description="PCE measurement error std dev",
            prior=PriorDistribution("inv_gamma", mean=0.10, std=2.00)
        ))

        self.add_parameter(Parameter(
            name="rho_gdpdef",
            description="GDP deflator measurement error persistence",
            prior=PriorDistribution("beta", mean=0.50, std=0.20)
        ))

        self.add_parameter(Parameter(
            name="sigma_gdpdef",
            description="GDP deflator measurement error std dev",
            prior=PriorDistribution("inv_gamma", mean=0.10, std=2.00)
        ))

        self.add_parameter(Parameter(
            name="rho_10y",
            description="10-year rate measurement error persistence",
            prior=PriorDistribution("beta", mean=0.50, std=0.20)
        ))

        self.add_parameter(Parameter(
            name="sigma_10y",
            description="10-year rate measurement error std dev",
            prior=PriorDistribution("inv_gamma", mean=0.75, std=2.00)
        ))

        self.add_parameter(Parameter(
            name="rho_tfp",
            description="TFP measurement error persistence",
            prior=PriorDistribution("beta", mean=0.50, std=0.20)
        ))

        self.add_parameter(Parameter(
            name="sigma_tfp",
            description="TFP measurement error std dev",
            prior=PriorDistribution("inv_gamma", mean=0.10, std=2.00)
        ))

    def _define_variables(self):
        """Define state and control variables."""

        # Endogenous state variables (deviations from steady state)
        self.endogenous_states = [
            "c",          # consumption
            "i",          # investment
            "y",          # output
            "L",          # labor/hours
            "k_bar",      # installed capital
            "k",          # effective capital
            "u",          # capital utilization
            "q_k",        # Tobin's q
            "w",          # real wage
            "R",          # nominal interest rate
            "pi",         # inflation
            "mc",         # real marginal cost
            "r_k",        # rental rate of capital
            "R_k_tilde",  # gross nominal return on capital
            "n",          # entrepreneurial net worth
            "w_h",        # household MRS (wage from labor supply)
            "y_f",        # flexible-price output
            "pi_star",    # time-varying inflation target
        ]

        # Exogenous state variables (shocks)
        self.exogenous_states = [
            "z_tilde",    # stationary productivity
            "z_p",        # stochastic trend productivity growth
            "b",          # risk premium shock
            "mu",         # marginal efficiency of investment shock
            "g",          # government spending shock
            "lambda_f",   # price markup shock
            "lambda_w",   # wage markup shock
            "sigma_omega",  # cross-sectional volatility shock
            "r_m",        # monetary policy shock
        ]

        # Measurement error states
        self.measurement_states = [
            "e_gdp",
            "e_gdi",
            "e_pce",
            "e_gdpdef",
            "e_10y",
            "e_tfp",
        ]

        # Observable variables (link to data)
        self.observables = [
            "obs_gdp_growth",
            "obs_gdi_growth",
            "obs_consumption_growth",
            "obs_investment_growth",
            "obs_wage_growth",
            "obs_hours",
            "obs_inflation_pce",
            "obs_inflation_gdpdef",
            "obs_ffr",
            "obs_10y_rate",
            "obs_10y_infl_exp",
            "obs_spread",
            "obs_tfp_growth",
        ]

    def get_log_linearized_equations(self, params: Dict[str, float]) -> Dict[str, str]:
        """
        Return the log-linearized equilibrium conditions.

        This would contain all equations (3)-(22) from the documentation.
        For now, returning a placeholder structure.

        Args:
            params: Dictionary of parameter values

        Returns:
            Dictionary mapping equation names to their symbolic representations
        """
        # This is a placeholder - full implementation would require
        # translating all equations from the PDF
        equations = {}

        # Technology (Equations 3-5)
        equations["z_tilde_process"] = "z_tilde[t] = rho_z * z_tilde[t-1] + sigma_z * eps_z[t]"
        equations["z_p_process"] = "z_p[t] = rho_zp * z_p[t-1] + sigma_zp * eps_zp[t]"
        equations["z_growth"] = "z[t] = (1/(1-alpha))*(rho_z-1)*z_tilde[t-1] + (1/(1-alpha))*sigma_z*eps_z[t] + z_p[t]"

        # Consumption Euler equation (Equation 6)
        equations["consumption_euler"] = """
        c[t] = -(1-h*exp(-gamma))/(sigma_c*(1+h*exp(-gamma))) * (R[t] - E[pi[t+1]] + b[t])
             + h*exp(-gamma)/(1+h*exp(-gamma)) * (c[t-1] - z[t])
             + 1/(1+h*exp(-gamma)) * E[c[t+1] + z[t+1]]
             + (sigma_c-1)/(sigma_c*(1+h*exp(-gamma))) * w_star*L_star/c_star * (L[t] - E[L[t+1]])
        """

        # Investment (Equation 7)
        equations["investment"] = """
        i[t] = q_k[t]/(S_pp*exp(2*gamma)*(1+beta_bar))
             + 1/(1+beta_bar) * (i[t-1] - z[t])
             + beta_bar/(1+beta_bar) * E[i[t+1] + z[t+1]]
             + mu[t]
        """

        # Capital accumulation (Equation 8)
        equations["capital_accumulation"] = """
        k_bar[t] = (1 - i_star/k_bar_star) * (k_bar[t-1] - z[t])
                 + i_star/k_bar_star * i[t]
                 + i_star/k_bar_star * S_pp*exp(2*gamma)*(1+beta_bar) * mu[t]
        """

        # Effective capital (Equation 9)
        equations["effective_capital"] = "k[t] = u[t] - z[t] + k_bar[t-1]"

        # Capital utilization (Equation 10)
        equations["capital_utilization"] = "(1-psi)/psi * r_k[t] = u[t]"

        # Marginal cost (Equation 11)
        equations["marginal_cost"] = "mc[t] = w[t] + alpha*L[t] - alpha*k[t]"

        # Capital-labor ratio (Equation 12)
        equations["capital_labor"] = "k[t] = w[t] - r_k[t] + L[t]"

        # Return on capital (Equation 13)
        equations["return_on_capital"] = """
        R_k_tilde[t] - pi[t] = r_k_star/(r_k_star + (1-delta)) * r_k[t]
                              + (1-delta)/(r_k_star + (1-delta)) * q_k[t]
                              - q_k[t-1]
        """

        # Spread (Equation 14)
        equations["spread"] = """
        E[R_k_tilde[t+1] - R[t]] = b[t] + zeta_sp_b * (q_k[t] + k_bar[t] - n[t]) + sigma_omega_tilde[t]
        """

        # Net worth evolution (Equation 15)
        equations["net_worth"] = """
        n[t] = zeta_n_Rk * (R_k_tilde[t] - pi[t])
             - zeta_n_R * (R[t-1] - pi[t] + b[t-1])
             + zeta_n_qK * (q_k[t-1] + k_bar[t-1])
             + zeta_n_n * n[t-1]
             - gamma_star * v_star/n_star * z[t]
             - zeta_n_sigma_omega/zeta_sp_sigma_omega * sigma_omega_tilde[t-1]
        """

        # Production (Equation 16)
        equations["production"] = "y[t] = Phi_p * (alpha*k[t] + (1-alpha)*L[t])"

        # Resource constraint (Equation 17)
        equations["resource_constraint"] = """
        y[t] = g_star*g[t] + c_star/y_star*c[t] + i_star/y_star*i[t] + r_k_star*k_star/y_star*u[t]
        """

        # Price Phillips curve (Equation 18)
        equations["price_phillips"] = """
        pi[t] = kappa*mc[t]
              + iota_p/(1 + iota_p*beta_bar) * pi[t-1]
              + beta_bar/(1 + iota_p*beta_bar) * E[pi[t+1]]
              + lambda_f[t]
        """

        # Wage Phillips curve (Equation 19)
        equations["wage_phillips"] = """
        w[t] = (1-zeta_w*beta_bar)*(1-zeta_w)/((1+beta_bar)*zeta_w*((lambda_w-1)*epsilon_w+1)) * (w_h[t] - w[t])
             - (1 + iota_w*beta_bar)/(1+beta_bar) * pi[t]
             + 1/(1+beta_bar) * (w[t-1] - z[t] + iota_w*pi[t-1])
             + beta_bar/(1+beta_bar) * E[w[t+1] + z[t+1] + pi[t+1]]
             + lambda_w[t]
        """

        # Household MRS (Equation 20)
        equations["household_mrs"] = """
        w_h[t] = 1/(1-h*exp(-gamma)) * (c[t] - h*exp(-gamma)*c[t-1] + h*exp(-gamma)*z[t]) + nu_l*L[t]
        """

        # Monetary policy rule (Equation 21)
        equations["monetary_policy"] = """
        R[t] = rho_R*R[t-1]
             + (1-rho_R) * (psi1*(pi[t] - pi_star[t]) + psi2*(y[t] - y_f[t]))
             + psi3 * ((y[t] - y_f[t]) - (y[t-1] - y_f[t-1]))
             + r_m[t]
        """

        # Inflation target (Equation 22)
        equations["inflation_target"] = "pi_star[t] = rho_pi_star * pi_star[t-1] + sigma_pi_star * eps_pi_star[t]"

        return equations

    def get_measurement_equations(self, params: Dict[str, float]) -> Dict[str, str]:
        """
        Return the measurement equations linking observables to model variables.

        Based on equations (32) from the documentation.

        Args:
            params: Dictionary of parameter values

        Returns:
            Dictionary mapping observable names to their measurement equations
        """
        measurement = {}

        # GDP growth (equation 32)
        measurement["obs_gdp_growth"] = """
        100*gamma + (y[t] - y[t-1] + z[t]) + e_gdp[t] - C_me*e_gdp[t-1]
        """

        # GDI growth
        measurement["obs_gdi_growth"] = """
        100*gamma + (y[t] - y[t-1] + z[t]) + e_gdi[t] - C_me*e_gdi[t-1]
        """

        # Consumption growth
        measurement["obs_consumption_growth"] = """
        100*gamma + (c[t] - c[t-1] + z[t])
        """

        # Investment growth
        measurement["obs_investment_growth"] = """
        100*gamma + (i[t] - i[t-1] + z[t])
        """

        # Real wage growth
        measurement["obs_wage_growth"] = """
        100*gamma + (w[t] - w[t-1] + z[t])
        """

        # Hours
        measurement["obs_hours"] = "L_bar + L[t]"

        # Core PCE inflation
        measurement["obs_inflation_pce"] = "pi_star + pi[t] + e_pce[t]"

        # GDP deflator inflation
        measurement["obs_inflation_gdpdef"] = """
        pi_star + delta_gdpdef + gamma_gdpdef * pi[t] + e_gdpdef[t]
        """

        # Federal funds rate
        measurement["obs_ffr"] = "R_star + R[t]"

        # 10-year nominal bond yield
        measurement["obs_10y_rate"] = """
        R_star + E[sum_{k=1}^{40} R[t+k] / 40] + e_10y[t]
        """

        # 10-year inflation expectations
        measurement["obs_10y_infl_exp"] = """
        pi_star + E[sum_{k=1}^{40} pi[t+k] / 40]
        """

        # Spread
        measurement["obs_spread"] = "SP_star + E[R_k_tilde[t+1] - R[t]]"

        # TFP growth (demeaned)
        measurement["obs_tfp_growth"] = """
        z[t] + alpha/(1-alpha) * (u[t] - u[t-1]) + e_tfp[t]
        """

        return measurement

    def get_steady_state(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Compute the non-stochastic steady state.

        Args:
            params: Dictionary of parameter values

        Returns:
            Dictionary of steady-state values
        """
        # Extract key parameters
        alpha = params["alpha"]
        delta = params["delta"]
        gamma = params["gamma"] / 100  # Convert from percentage
        beta_param = params["beta_bar"] / 100  # Convert from percentage
        beta = 1 / (1 + beta_param) * np.exp((params["sigma_c"] - 1) * gamma)

        pi_star = params["pi_star"] / 100  # Convert from percentage
        g_star = params["g_star"]
        Phi_p = params["Phi_p"]

        # Compute steady state
        steady_state = {}

        # All deviations from steady state are zero
        for var in self.endogenous_states + self.exogenous_states + self.measurement_states:
            steady_state[var] = 0.0

        # Steady-state levels (not deviations) for reference
        R_star = np.exp(gamma) / beta * np.exp(pi_star) - 1
        r_k_star = np.exp(gamma) / beta - (1 - delta)

        steady_state["R_star"] = R_star
        steady_state["r_k_star"] = r_k_star

        return steady_state


def create_nyfed_model() -> NYFedModel1002:
    """
    Factory function to create an instance of the NYFed Model 1002.

    Returns:
        Initialized NYFedModel1002 instance
    """
    return NYFedModel1002()


if __name__ == "__main__":
    # Example usage
    model = create_nyfed_model()
    print(f"Model: {model.name}")
    print(f"Number of parameters: {len(model.parameters)}")
    print(f"Number of endogenous states: {len(model.endogenous_states)}")
    print(f"Number of exogenous states: {len(model.exogenous_states)}")
    print(f"Number of observables: {len(model.observables)}")
