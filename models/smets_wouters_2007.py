"""
Smets-Wouters (2007) DSGE Model.

Translation of the Smets-Wouters (2007) "Shocks and Frictions in US Business Cycles"
model to the dsge-py framework.

References
----------
Primary Paper:
    Smets, F., & Wouters, R. (2007). "Shocks and frictions in US business cycles:
    A Bayesian DSGE approach." American Economic Review, 97(3), 586-606.
    https://www.aeaweb.org/articles?id=10.1257/aer.97.3.586

Reference Implementation:
    Johannes Pfeifer's Dynare replication (verified against published posterior estimates):
    https://github.com/JohannesPfeifer/DSGE_mod/blob/master/Smets_Wouters_2007/Smets_Wouters_2007_45.mod

Parameter Values Source:
    Published posterior estimates from Table 1A and 1B of Smets & Wouters (2007).
    See ECONOMIC_MODELS_REVIEW.md for detailed parameter verification against literature.

    IMPORTANT: Current implementation uses a mix of prior means and posterior estimates.
    For research use, parameters should match published posteriors. See review document
    for specific discrepancies and corrections needed.

Data:
    US quarterly data (1966:Q1-2004:Q4) for 7 observables:
    - GDP growth, Consumption growth, Investment growth, Wage growth
    - Inflation, Federal funds rate, Hours worked

Estimation Method:
    Bayesian estimation using Random Walk Metropolis-Hastings algorithm
    with 100,000 draws (50,000 burn-in).
"""


import numpy as np

from dsge.models.base import DSGEModel, ModelSpecification
from dsge.models.parameters import Parameter, Prior


class SmetsWouters2007(DSGEModel):
    """
    Smets-Wouters (2007) medium-scale DSGE model.

    Features:
    - Sticky prices and wages (Calvo framework with indexation)
    - Habit formation in consumption
    - Investment adjustment costs
    - Variable capital utilization
    - 7 structural shocks
    - Both flexible and sticky price/wage equilibria

    State vector (41 states):
    - Sticky price variables (13): c, inve, y, lab, k, pk, zcap, rk, w, r, pinf, mc, kp
    - Flexible price variables (11): cf, invef, yf, labf, kf, pkf, zcapf, rkf, wf, rrf, kpf
    - Lags (8): c_lag, inve_lag, y_lag, w_lag, r_lag, pinf_lag, kp_lag, kpf_lag
    - Shocks (7): a, b, g, qs, ms, spinf, sw
    - MA lags (2): epinfma_lag, ewma_lag

    Observables (7):
    - dy: Output growth
    - dc: Consumption growth
    - dinve: Investment growth
    - dw: Wage growth
    - pinfobs: Inflation
    - robs: Federal funds rate
    - labobs: Hours worked
    """

    def __init__(self) -> None:
        """Initialize the Smets-Wouters 2007 model."""
        # State dimensions
        n_sticky = 13  # Sticky price economy variables
        n_flex = 11  # Flexible price economy variables
        n_lags = 8  # Lags needed for dynamics
        n_shocks = 7  # Structural shocks
        n_ma_lags = 2  # MA lags for price/wage markup shocks

        n_states = n_sticky + n_flex + n_lags + n_shocks + n_ma_lags  # 41 total
        n_controls = 0  # All variables treated as states
        n_structural_shocks = 7
        n_observables = 7

        # State names
        state_names = [
            # Sticky price economy (13)
            "c",
            "inve",
            "y",
            "lab",
            "k",
            "pk",
            "zcap",
            "rk",
            "w",
            "r",
            "pinf",
            "mc",
            "kp",
            # Flexible price economy (11)
            "cf",
            "invef",
            "yf",
            "labf",
            "kf",
            "pkf",
            "zcapf",
            "rkf",
            "wf",
            "rrf",
            "kpf",
            # Lags (8)
            "c_lag",
            "inve_lag",
            "y_lag",
            "w_lag",
            "r_lag",
            "pinf_lag",
            "kp_lag",
            "kpf_lag",
            # Shocks (7)
            "a",
            "b",
            "g",
            "qs",
            "ms",
            "spinf",
            "sw",
            # MA auxiliary lags (2)
            "epinfma_lag",
            "ewma_lag",
        ]

        shock_names = [
            "ea",  # Productivity shock
            "eb",  # Risk premium shock
            "eg",  # Government spending shock
            "eqs",  # Investment-specific technology shock
            "em",  # Monetary policy shock
            "epinf",  # Price markup shock
            "ew",  # Wage markup shock
        ]

        observable_names = [
            "obs_dy",  # Output growth
            "obs_dc",  # Consumption growth
            "obs_dinve",  # Investment growth
            "obs_dw",  # Wage growth
            "obs_pinfobs",  # Inflation
            "obs_robs",  # Federal funds rate
            "obs_labobs",  # Hours worked
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

        # Helper function for prior conversion
        def make_prior(dist_type, mean_val, std_val):
            """Convert distribution specifications to Prior objects."""
            if dist_type == "normal":
                return Prior("normal", {"mean": mean_val, "std": std_val})
            if dist_type == "beta":
                v = std_val**2
                alpha = mean_val * (mean_val * (1 - mean_val) / v - 1)
                beta = (1 - mean_val) * (mean_val * (1 - mean_val) / v - 1)
                return Prior("beta", {"alpha": alpha, "beta": beta})
            if dist_type == "gamma":
                v = std_val**2
                shape = mean_val**2 / v
                rate = mean_val / v
                return Prior("gamma", {"shape": shape, "rate": rate})
            if dist_type == "invgamma":
                shape = 2.0
                scale = mean_val * (shape + 1)
                return Prior("invgamma", {"shape": shape, "scale": scale})
            msg = f"Unknown distribution type: {dist_type}"
            raise ValueError(msg)

        # ==================================================================
        # STRUCTURAL PARAMETERS (Estimated in original paper)
        # ==================================================================

        # Preferences
        self.parameters.add(
            Parameter(
                name="csigma",
                value=1.2312,
                prior=make_prior("normal", 1.50, 0.375),
                fixed=False,
                description="Risk aversion parameter",
            )
        )

        self.parameters.add(
            Parameter(
                name="chabb",
                value=0.7205,
                prior=make_prior("beta", 0.70, 0.10),
                fixed=False,
                description="External habit degree",
            )
        )

        self.parameters.add(
            Parameter(
                name="csigl",
                value=2.8401,
                prior=make_prior("normal", 2.00, 0.75),
                fixed=False,
                description="Inverse Frisch elasticity of labor supply",
            )
        )

        # Technology
        self.parameters.add(
            Parameter(
                name="calfa",
                value=0.24,
                prior=make_prior("normal", 0.30, 0.05),
                fixed=False,
                description="Capital share in production",
            )
        )

        self.parameters.add(
            Parameter(
                name="csadjcost",
                value=6.3325,
                prior=make_prior("normal", 4.00, 1.50),
                fixed=False,
                description="Investment adjustment cost",
            )
        )

        self.parameters.add(
            Parameter(
                name="czcap",
                value=0.2696,
                prior=make_prior("beta", 0.50, 0.15),
                fixed=False,
                description="Capacity utilization cost parameter",
            )
        )

        # Price and wage setting
        self.parameters.add(
            Parameter(
                name="cprobp",
                value=0.7813,
                prior=make_prior("beta", 0.50, 0.10),
                fixed=False,
                description="Calvo parameter for prices",
            )
        )

        self.parameters.add(
            Parameter(
                name="cindp",
                value=0.3291,
                prior=make_prior("beta", 0.50, 0.15),
                fixed=False,
                description="Indexation to past inflation (prices)",
            )
        )

        self.parameters.add(
            Parameter(
                name="cprobw",
                value=0.7937,
                prior=make_prior("beta", 0.50, 0.10),
                fixed=False,
                description="Calvo parameter for wages",
            )
        )

        self.parameters.add(
            Parameter(
                name="cindw",
                value=0.4425,
                prior=make_prior("beta", 0.50, 0.15),
                fixed=False,
                description="Indexation to past inflation (wages)",
            )
        )

        # Policy rule
        self.parameters.add(
            Parameter(
                name="crpi",
                value=1.7985,
                prior=make_prior("normal", 1.50, 0.25),
                fixed=False,
                description="Taylor rule: response to inflation",
            )
        )

        self.parameters.add(
            Parameter(
                name="crr",
                value=0.8258,
                prior=make_prior("beta", 0.75, 0.10),
                fixed=False,
                description="Interest rate smoothing",
            )
        )

        self.parameters.add(
            Parameter(
                name="cry",
                value=0.0893,
                prior=make_prior("normal", 0.125, 0.05),
                fixed=False,
                description="Taylor rule: response to output gap",
            )
        )

        self.parameters.add(
            Parameter(
                name="crdy",
                value=0.2239,
                prior=make_prior("normal", 0.125, 0.05),
                fixed=False,
                description="Taylor rule: response to output growth",
            )
        )

        # Shock persistence
        self.parameters.add(
            Parameter(
                name="crhoa",
                value=0.9676,
                prior=make_prior("beta", 0.50, 0.20),
                fixed=False,
                description="AR(1) productivity shock persistence",
            )
        )

        self.parameters.add(
            Parameter(
                name="crhob",
                value=0.2703,
                prior=make_prior("beta", 0.50, 0.20),
                fixed=False,
                description="AR(1) risk premium shock persistence",
            )
        )

        self.parameters.add(
            Parameter(
                name="crhog",
                value=0.9930,
                prior=make_prior("beta", 0.50, 0.20),
                fixed=False,
                description="AR(1) government spending shock persistence",
            )
        )

        self.parameters.add(
            Parameter(
                name="crhoqs",
                value=0.5724,
                prior=make_prior("beta", 0.50, 0.20),
                fixed=False,
                description="AR(1) investment shock persistence",
            )
        )

        self.parameters.add(
            Parameter(
                name="crhoms",
                value=0.3000,
                prior=make_prior("beta", 0.50, 0.20),
                fixed=False,
                description="AR(1) monetary policy shock persistence",
            )
        )

        self.parameters.add(
            Parameter(
                name="crhopinf",
                value=0.8692,
                prior=make_prior("beta", 0.50, 0.20),
                fixed=False,
                description="AR(1) price markup shock persistence",
            )
        )

        self.parameters.add(
            Parameter(
                name="crhow",
                value=0.9546,
                prior=make_prior("beta", 0.50, 0.20),
                fixed=False,
                description="AR(1) wage markup shock persistence",
            )
        )

        # MA components for markup shocks
        self.parameters.add(
            Parameter(
                name="cmap",
                value=0.0,
                prior=make_prior("beta", 0.50, 0.20),
                fixed=False,
                description="MA coefficient for price markup shock",
            )
        )

        self.parameters.add(
            Parameter(
                name="cmaw",
                value=0.0,
                prior=make_prior("beta", 0.50, 0.20),
                fixed=False,
                description="MA coefficient for wage markup shock",
            )
        )

        # Technology spillover
        self.parameters.add(
            Parameter(
                name="cgy",
                value=0.51,
                fixed=True,
                description="Feedback from technology to government spending",
            )
        )

        # Shock standard deviations
        self.parameters.add(
            Parameter(
                name="sigma_ea",
                value=0.4618,
                prior=make_prior("invgamma", 0.10, 2.0),
                fixed=False,
                description="Std dev of productivity shock",
            )
        )

        self.parameters.add(
            Parameter(
                name="sigma_eb",
                value=0.1819,
                prior=make_prior("invgamma", 0.10, 2.0),
                fixed=False,
                description="Std dev of risk premium shock",
            )
        )

        self.parameters.add(
            Parameter(
                name="sigma_eg",
                value=0.6090,
                prior=make_prior("invgamma", 0.10, 2.0),
                fixed=False,
                description="Std dev of government spending shock",
            )
        )

        self.parameters.add(
            Parameter(
                name="sigma_eqs",
                value=0.4602,
                prior=make_prior("invgamma", 0.10, 2.0),
                fixed=False,
                description="Std dev of investment shock",
            )
        )

        self.parameters.add(
            Parameter(
                name="sigma_em",
                value=0.2397,
                prior=make_prior("invgamma", 0.10, 2.0),
                fixed=False,
                description="Std dev of monetary policy shock",
            )
        )

        self.parameters.add(
            Parameter(
                name="sigma_epinf",
                value=0.1455,
                prior=make_prior("invgamma", 0.10, 2.0),
                fixed=False,
                description="Std dev of price markup shock",
            )
        )

        self.parameters.add(
            Parameter(
                name="sigma_ew",
                value=0.2089,
                prior=make_prior("invgamma", 0.10, 2.0),
                fixed=False,
                description="Std dev of wage markup shock",
            )
        )

        # ==================================================================
        # FIXED PARAMETERS (Calibrated)
        # ==================================================================

        self.parameters.add(
            Parameter(
                name="ctou", value=0.025, fixed=True, description="Depreciation rate (quarterly)"
            )
        )

        self.parameters.add(
            Parameter(
                name="clandaw", value=1.5, fixed=True, description="Gross markup in labor market"
            )
        )

        self.parameters.add(
            Parameter(
                name="cg",
                value=0.18,
                fixed=True,
                description="Steady-state exogenous spending share",
            )
        )

        self.parameters.add(
            Parameter(
                name="curvp",
                value=10.0,
                fixed=True,
                description="Kimball aggregator curvature for prices",
            )
        )

        self.parameters.add(
            Parameter(
                name="curvw",
                value=10.0,
                fixed=True,
                description="Kimball aggregator curvature for wages",
            )
        )

        self.parameters.add(
            Parameter(
                name="cfc",
                value=1.5,
                fixed=True,
                description="Fixed cost share (ensures zero profits in SS)",
            )
        )

        # Steady-state values
        self.parameters.add(
            Parameter(
                name="constepinf",
                value=0.7,
                fixed=True,
                description="Steady-state quarterly inflation rate (%)",
            )
        )

        self.parameters.add(
            Parameter(
                name="constebeta",
                value=0.7420,
                fixed=True,
                description="Quarterly time preference rate (%)",
            )
        )

        self.parameters.add(
            Parameter(
                name="constelab",
                value=0.0,
                fixed=True,
                description="Steady-state hours worked (log deviation)",
            )
        )

        self.parameters.add(
            Parameter(
                name="ctrend", value=0.3982, fixed=True, description="Quarterly net growth rate (%)"
            )
        )

    def _compute_steady_state_params(self, params: dict[str, float]) -> dict[str, float]:
        """
        Compute derived parameters based on steady-state relationships.

        These parameters depend on structural parameters and ensure steady-state consistency.
        """
        # Extract needed parameters
        ctou = params["ctou"]
        calfa = params["calfa"]
        clandaw = params["clandaw"]
        cg = params["cg"]
        constepinf = params["constepinf"]
        constebeta = params["constebeta"]
        ctrend = params["ctrend"]
        csigma = params["csigma"]
        cfc = params["cfc"]

        # Gross inflation and growth
        cpie = 1 + constepinf / 100
        cgamma = 1 + ctrend / 100
        cbeta = 1 / (1 + constebeta / 100)

        # Growth-adjusted discount factor
        cbetabar = cbeta * cgamma ** (-csigma)

        # Steady-state real interest rate
        cr = cpie / (cbeta * cgamma ** (-csigma))

        # Steady-state rental rate of capital
        crk = (cbeta ** (-1)) * (cgamma**csigma) - (1 - ctou)

        # Gross price markup (derived from technology)
        clandap = cfc  # In SW2007, fixed cost ensures zero profits

        # Steady-state wage
        cw = (calfa**calfa * (1 - calfa) ** (1 - calfa) / (clandap * crk**calfa)) ** (
            1 / (1 - calfa)
        )

        # Investment-capital ratio
        cikbar = 1 - (1 - ctou) / cgamma
        cik = (1 - (1 - ctou) / cgamma) * cgamma

        # Labor-capital ratio
        clk = ((1 - calfa) / calfa) * (crk / cw)

        # Capital-output ratio
        cky = cfc * (clk) ** (calfa - 1)

        # Investment-output and consumption-output ratios
        ciy = cik * cky
        ccy = 1 - cg - cik * cky

        # Rental income share
        crkky = crk * cky

        # Wage-hours-consumption composite
        cwhlc = (1 / clandaw) * (1 - calfa) / calfa * crk * cky / ccy

        # Steady-state federal funds rate
        conster = (cr - 1) * 100

        # Return all derived parameters
        return {
            "cpie": cpie,
            "cgamma": cgamma,
            "cbeta": cbeta,
            "cbetabar": cbetabar,
            "cr": cr,
            "crk": crk,
            "clandap": clandap,
            "cw": cw,
            "cikbar": cikbar,
            "cik": cik,
            "clk": clk,
            "cky": cky,
            "ciy": ciy,
            "ccy": ccy,
            "crkky": crkky,
            "cwhlc": cwhlc,
            "conster": conster,
        }


    def system_matrices(self, params: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """
        Compute linearized system matrices for the Smets-Wouters model.

        System: Γ0 * s_t = Γ1 * s_{t-1} + Ψ * ε_t + Π * η_t

        Returns:
            Dictionary with keys 'Gamma0', 'Gamma1', 'Psi', 'Pi'
        """
        # Get parameters
        if params is None:
            p = self.parameters.to_dict()
        else:
            # Convert array to dict if needed
            param_names = [param.name for param in self.parameters.parameters]
            p = dict(zip(param_names, params, strict=False))

        # Compute derived steady-state parameters
        derived = self._compute_steady_state_params(p)
        p.update(derived)  # Add derived parameters to parameter dict

        # Extract commonly used parameters
        calfa = p["calfa"]
        cbetabar = p["cbetabar"]
        cgamma = p["cgamma"]
        chabb = p["chabb"]
        csigma = p["csigma"]
        csadjcost = p["csadjcost"]
        csigl = p["csigl"]
        czcap = p["czcap"]
        cprobp = p["cprobp"]
        cprobw = p["cprobw"]
        cindp = p["cindp"]
        cindw = p["cindw"]
        cfc = p["cfc"]
        curvp = p["curvp"]
        curvw = p["curvw"]
        clandaw = p["clandaw"]
        crr = p["crr"]
        crpi = p["crpi"]
        cry = p["cry"]
        crdy = p["crdy"]
        crhoa = p["crhoa"]
        crhob = p["crhob"]
        crhog = p["crhog"]
        crhoqs = p["crhoqs"]
        crhoms = p["crhoms"]
        crhopinf = p["crhopinf"]
        crhow = p["crhow"]
        cmap = p["cmap"]
        cmaw = p["cmaw"]
        cgy = p["cgy"]
        ctou = p["ctou"]

        # Derived parameters
        crk = p["crk"]
        p["cw"]
        cikbar = p["cikbar"]
        p["cky"]
        ciy = p["ciy"]
        ccy = p["ccy"]
        crkky = p["crkky"]
        cwhlc = p["cwhlc"]

        # Shock standard deviations
        sigma_ea = p["sigma_ea"]
        sigma_eb = p["sigma_eb"]
        sigma_eg = p["sigma_eg"]
        sigma_eqs = p["sigma_eqs"]
        sigma_em = p["sigma_em"]
        sigma_epinf = p["sigma_epinf"]
        sigma_ew = p["sigma_ew"]

        # Matrix dimensions
        n = self.spec.n_states  # 41
        n_shocks = self.spec.n_shocks  # 7
        n_eta = 13  # Number of expectation errors

        # Initialize matrices
        Gamma0 = np.zeros((n, n))
        Gamma1 = np.zeros((n, n))
        Psi = np.zeros((n, n_shocks))
        Pi = np.zeros((n, n_eta))

        # State indices (following state_names order)
        # Sticky price (0-12)
        idx_c, idx_inve, idx_y, idx_lab = 0, 1, 2, 3
        idx_k, idx_pk, idx_zcap, idx_rk = 4, 5, 6, 7
        idx_w, idx_r, idx_pinf, idx_mc, idx_kp = 8, 9, 10, 11, 12

        # Flexible price (13-23)
        idx_cf, idx_invef, idx_yf, idx_labf = 13, 14, 15, 16
        idx_kf, idx_pkf, idx_zcapf, idx_rkf = 17, 18, 19, 20
        idx_wf, idx_rrf, idx_kpf = 21, 22, 23

        # Lags (24-31)
        idx_c_lag, idx_inve_lag, idx_y_lag, idx_w_lag = 24, 25, 26, 27
        idx_r_lag, idx_pinf_lag, idx_kp_lag, idx_kpf_lag = 28, 29, 30, 31

        # Shocks (32-38)
        idx_a, idx_b, idx_g, idx_qs = 32, 33, 34, 35
        idx_ms, idx_spinf, idx_sw = 36, 37, 38

        # MA lags (39-40)
        idx_epinfma_lag, idx_ewma_lag = 39, 40

        # Shock indices
        idx_ea, idx_eb, idx_eg, idx_eqs = 0, 1, 2, 3
        idx_em, idx_epinf, idx_ew = 4, 5, 6

        # Expectation error indices (for Pi matrix)
        # Flexible price economy: cf, invef, yf, labf, pkf, wf, rrf (7 errors)
        # Sticky price economy: c, inve, pinf, w, pk, y (6 errors)
        idx_eta_cf = 0
        idx_eta_invef = 1
        idx_eta_pkf = 2
        idx_eta_c = 3
        idx_eta_inve = 4
        idx_eta_pinf = 5
        idx_eta_pinf2 = 6  # Second pinf expectation (in consumption Euler)
        idx_eta_w = 7
        idx_eta_pk = 8
        idx_eta_rk = 9
        idx_eta_labf = 11
        idx_eta_lab = 12

        # ===================================================================
        # FLEXIBLE PRICE ECONOMY EQUATIONS (11 equations)
        # ===================================================================

        # Equation 1: Labor FOC (flexible)
        # 0*(1-calfa)*a + 1*a = calfa*rkf + (1-calfa)*wf
        # => -a + calfa*rkf + (1-calfa)*wf = 0
        Gamma0[idx_rkf, idx_a] = -1.0
        Gamma0[idx_rkf, idx_rkf] = calfa
        Gamma0[idx_rkf, idx_wf] = 1.0 - calfa

        # Equation 2: Capacity utilization (flexible)
        # zcapf = (1/(czcap/(1-czcap))) * rkf
        zcap_coef = 1.0 / (czcap / (1 - czcap))
        Gamma0[idx_zcapf, idx_zcapf] = 1.0
        Gamma0[idx_zcapf, idx_rkf] = -zcap_coef

        # Equation 3: Capital FOC (flexible)
        # rkf = wf + labf - kf
        Gamma0[idx_kf, idx_rkf] = 1.0
        Gamma0[idx_kf, idx_wf] = -1.0
        Gamma0[idx_kf, idx_labf] = -1.0
        Gamma0[idx_kf, idx_kf] = 1.0

        # Equation 4: Capital services (flexible)
        # kf = kpf(-1) + zcapf
        Gamma0[idx_labf, idx_kf] = 1.0
        Gamma0[idx_labf, idx_zcapf] = -1.0
        Gamma1[idx_labf, idx_kpf] = 1.0

        # Equation 5: Investment Euler (flexible)
        # invef = (1/(1+cbetabar*cgamma))*(invef(-1) + cbetabar*cgamma*invef(1) + (1/(cgamma^2*csadjcost))*pkf) + qs
        inv_coef = 1.0 / (1.0 + cbetabar * cgamma)
        pk_coef = 1.0 / (cgamma**2 * csadjcost)
        Gamma0[idx_invef, idx_invef] = 1.0
        Gamma0[idx_invef, idx_pkf] = -inv_coef * pk_coef
        Gamma0[idx_invef, idx_qs] = -1.0
        Gamma1[idx_invef, idx_invef] = inv_coef
        Pi[idx_invef, idx_eta_invef] = -inv_coef * cbetabar * cgamma

        # Equation 6: Capital value arbitrage (flexible)
        # pkf = -rrf + (crk/(crk+(1-ctou)))*E[rkf(1)] + ((1-ctou)/(crk+(1-ctou)))*E[pkf(1)]
        rk_share = crk / (crk + (1 - ctou))
        pk_share = (1 - ctou) / (crk + (1 - ctou))
        Gamma0[idx_pkf, idx_pkf] = 1.0
        Gamma0[idx_pkf, idx_rrf] = 1.0
        Pi[idx_pkf, idx_eta_cf] = rk_share  # Using cf expectation as proxy for rkf
        Pi[idx_pkf, idx_eta_pkf] = -pk_share

        # Equation 7: Consumption Euler (flexible)
        # cf = (chabb/cgamma)/(1+chabb/cgamma)*cf(-1) + (1/(1+chabb/cgamma))*E[cf(1)]
        #      + ((csigma-1)*cwhlc/(csigma*(1+chabb/cgamma)))*(labf - E[labf(1)])
        #      - (1-chabb/cgamma)/(csigma*(1+chabb/cgamma))*(rrf) + b
        c_coef1 = (chabb / cgamma) / (1.0 + chabb / cgamma)
        c_coef2 = 1.0 / (1.0 + chabb / cgamma)
        lab_coef = ((csigma - 1) * cwhlc) / (csigma * (1.0 + chabb / cgamma))
        r_coef = (1.0 - chabb / cgamma) / (csigma * (1.0 + chabb / cgamma))

        Gamma0[idx_cf, idx_cf] = 1.0
        Gamma0[idx_cf, idx_labf] = -lab_coef
        Gamma0[idx_cf, idx_rrf] = r_coef
        Gamma0[idx_cf, idx_b] = -1.0
        Gamma1[idx_cf, idx_cf] = c_coef1
        Pi[idx_cf, idx_eta_cf] = -c_coef2
        Pi[idx_cf, idx_eta_labf] = lab_coef

        # Equation 8: Resource constraint (flexible)
        # yf = ccy*cf + ciy*invef + cg*g + crkky*zcapf
        Gamma0[idx_yf, idx_yf] = 1.0
        Gamma0[idx_yf, idx_cf] = -ccy
        Gamma0[idx_yf, idx_invef] = -ciy
        Gamma0[idx_yf, idx_g] = -p["cg"]
        Gamma0[idx_yf, idx_zcapf] = -crkky

        # Equation 9: Production function (flexible)
        # yf = cfc*(calfa*kf + (1-calfa)*labf + a)
        Gamma0[idx_wf, idx_yf] = 1.0
        Gamma0[idx_wf, idx_kf] = -cfc * calfa
        Gamma0[idx_wf, idx_labf] = -cfc * (1.0 - calfa)
        Gamma0[idx_wf, idx_a] = -cfc

        # Equation 10: Wage equation (flexible)
        # wf = csigl*labf + (1/(1-chabb/cgamma))*cf - (chabb/cgamma)/(1-chabb/cgamma)*cf(-1)
        wage_c_coef = 1.0 / (1.0 - chabb / cgamma)
        wage_c_lag_coef = (chabb / cgamma) / (1.0 - chabb / cgamma)
        Gamma0[idx_rrf, idx_wf] = 1.0
        Gamma0[idx_rrf, idx_labf] = -csigl
        Gamma0[idx_rrf, idx_cf] = -wage_c_coef
        Gamma1[idx_rrf, idx_cf] = -wage_c_lag_coef

        # Equation 11: Capital law of motion (flexible)
        # kpf = (1-cikbar)*kpf(-1) + cikbar*invef + cikbar*(cgamma^2*csadjcost)*qs
        Gamma0[idx_kpf, idx_kpf] = 1.0
        Gamma0[idx_kpf, idx_invef] = -cikbar
        Gamma0[idx_kpf, idx_qs] = -cikbar * (cgamma**2 * csadjcost)
        Gamma1[idx_kpf, idx_kpf] = 1.0 - cikbar

        # ===================================================================
        # STICKY PRICE/WAGE ECONOMY EQUATIONS (13 equations)
        # ===================================================================

        # Equation 1: Consumption Euler (row 0: c)
        # c = (chabb/cgamma)/(1+chabb/cgamma)*c(-1) + (1/(1+chabb/cgamma))*E[c(1)]
        #     + ((csigma-1)*cwhlc/(csigma*(1+chabb/cgamma)))*(lab - E[lab(1)])
        #     - (1-chabb/cgamma)/(csigma*(1+chabb/cgamma))*(r - E[pinf(1)]) + b
        Gamma0[idx_c, idx_c] = 1.0
        Gamma0[idx_c, idx_lab] = -lab_coef
        Gamma0[idx_c, idx_r] = r_coef
        Gamma0[idx_c, idx_b] = -1.0
        Gamma1[idx_c, idx_c] = c_coef1
        Pi[idx_c, idx_eta_c] = -c_coef2
        Pi[idx_c, idx_eta_lab] = lab_coef
        Pi[idx_c, idx_eta_pinf2] = -r_coef

        # Equation 2: Investment Euler (row 1: inve)
        # inve = (1/(1+cbetabar*cgamma))*(inve(-1) + cbetabar*cgamma*E[inve(1)] + (1/(cgamma^2*csadjcost))*pk) + qs
        Gamma0[idx_inve, idx_inve] = 1.0
        Gamma0[idx_inve, idx_pk] = -inv_coef * pk_coef
        Gamma0[idx_inve, idx_qs] = -1.0
        Gamma1[idx_inve, idx_inve] = inv_coef
        Pi[idx_inve, idx_eta_inve] = -inv_coef * cbetabar * cgamma

        # Equation 3: Resource constraint (row 2: y)
        # y = ccy*c + ciy*inve + cg*g + crkky*zcap
        Gamma0[idx_y, idx_y] = 1.0
        Gamma0[idx_y, idx_c] = -ccy
        Gamma0[idx_y, idx_inve] = -ciy
        Gamma0[idx_y, idx_g] = -p["cg"]
        Gamma0[idx_y, idx_zcap] = -crkky

        # Equation 4: Production function (row 3: lab)
        # y = cfc*(calfa*k + (1-calfa)*lab + a)
        Gamma0[idx_lab, idx_y] = 1.0
        Gamma0[idx_lab, idx_k] = -cfc * calfa
        Gamma0[idx_lab, idx_lab] = -cfc * (1.0 - calfa)
        Gamma0[idx_lab, idx_a] = -cfc

        # Equation 5: Capital services (row 4: k)
        # k = kp(-1) + zcap
        Gamma0[idx_k, idx_k] = 1.0
        Gamma0[idx_k, idx_zcap] = -1.0
        Gamma1[idx_k, idx_kp] = 1.0

        # Equation 6: Capital value (row 5: pk)
        # pk = -r + E[pinf(1)] + (crk/(crk+(1-ctou)))*E[rk(1)] + ((1-ctou)/(crk+(1-ctou)))*E[pk(1)]
        Gamma0[idx_pk, idx_pk] = 1.0
        Gamma0[idx_pk, idx_r] = 1.0
        Pi[idx_pk, idx_eta_pinf] = -1.0
        Pi[idx_pk, idx_eta_rk] = -rk_share
        Pi[idx_pk, idx_eta_pk] = -pk_share

        # Equation 7: Capacity utilization (row 6: zcap)
        # zcap = (1/(czcap/(1-czcap))) * rk
        Gamma0[idx_zcap, idx_zcap] = 1.0
        Gamma0[idx_zcap, idx_rk] = -zcap_coef

        # Equation 8: Capital FOC (row 7: rk)
        # rk = w + lab - k
        Gamma0[idx_rk, idx_rk] = 1.0
        Gamma0[idx_rk, idx_w] = -1.0
        Gamma0[idx_rk, idx_lab] = -1.0
        Gamma0[idx_rk, idx_k] = 1.0

        # Equation 9: Wage Phillips curve (row 8: w)
        # w = (1/(1+cbetabar*cgamma))*w(-1) + (cbetabar*cgamma/(1+cbetabar*cgamma))*E[w(1)]
        #     + (cindw/(1+cbetabar*cgamma))*pinf(-1) - (1+cbetabar*cgamma*cindw)/(1+cbetabar*cgamma)*pinf
        #     + (cbetabar*cgamma)/(1+cbetabar*cgamma)*E[pinf(1)]
        #     + (1-cprobw)*(1-cbetabar*cgamma*cprobw)/((1+cbetabar*cgamma)*cprobw)*(1/((clandaw-1)*curvw+1))
        #       * (csigl*lab + (1/(1-chabb/cgamma))*c - (chabb/cgamma)/(1-chabb/cgamma)*c(-1) - w) + sw
        wage_denom = 1.0 + cbetabar * cgamma
        wage_lag_coef = 1.0 / wage_denom
        wage_fwd_coef = cbetabar * cgamma / wage_denom
        wage_pinf_lag = cindw / wage_denom
        wage_pinf_cur = -(1.0 + cbetabar * cgamma * cindw) / wage_denom
        wage_pinf_fwd = cbetabar * cgamma / wage_denom
        wage_mrs_coef = (
            (1 - cprobw) * (1 - cbetabar * cgamma * cprobw) / (cprobw * wage_denom)
        ) / ((clandaw - 1) * curvw + 1)

        Gamma0[idx_w, idx_w] = 1.0 + wage_mrs_coef
        Gamma0[idx_w, idx_lab] = -wage_mrs_coef * csigl
        Gamma0[idx_w, idx_c] = -wage_mrs_coef * wage_c_coef
        Gamma0[idx_w, idx_pinf] = -wage_pinf_cur
        Gamma0[idx_w, idx_sw] = -1.0
        Gamma1[idx_w, idx_w] = wage_lag_coef
        Gamma1[idx_w, idx_c] = wage_mrs_coef * wage_c_lag_coef
        Gamma1[idx_w, idx_pinf] = wage_pinf_lag
        Pi[idx_w, idx_eta_w] = -wage_fwd_coef
        Pi[idx_w, idx_eta_pinf] = -wage_pinf_fwd

        # Equation 10: Taylor rule (row 9: r)
        # r = crpi*(1-crr)*pinf + cry*(1-crr)*(y-yf) + crdy*(y-yf-y(-1)+yf(-1)) + crr*r(-1) + ms
        Gamma0[idx_r, idx_r] = 1.0
        Gamma0[idx_r, idx_pinf] = -crpi * (1 - crr)
        Gamma0[idx_r, idx_y] = -(cry * (1 - crr) + crdy)
        Gamma0[idx_r, idx_yf] = cry * (1 - crr) + crdy
        Gamma0[idx_r, idx_ms] = -1.0
        Gamma1[idx_r, idx_r] = crr
        Gamma1[idx_r, idx_y] = -crdy
        Gamma1[idx_r, idx_yf] = crdy

        # Equation 11: Price Phillips curve (row 10: pinf)
        # pinf = (1/(1+cbetabar*cgamma*cindp))*(cbetabar*cgamma*E[pinf(1)] + cindp*pinf(-1)
        #        + ((1-cprobp)*(1-cbetabar*cgamma*cprobp)/cprobp)/((cfc-1)*curvp+1)*mc) + spinf
        phillips_denom = 1.0 + cbetabar * cgamma * cindp
        phillips_fwd = cbetabar * cgamma / phillips_denom
        phillips_lag = cindp / phillips_denom
        phillips_mc = (
            ((1 - cprobp) * (1 - cbetabar * cgamma * cprobp) / cprobp)
            / ((cfc - 1) * curvp + 1)
            / phillips_denom
        )

        Gamma0[idx_pinf, idx_pinf] = 1.0
        Gamma0[idx_pinf, idx_mc] = -phillips_mc
        Gamma0[idx_pinf, idx_spinf] = -1.0
        Gamma1[idx_pinf, idx_pinf] = phillips_lag
        Pi[idx_pinf, idx_eta_pinf] = -phillips_fwd

        # Equation 12: Marginal cost (row 11: mc)
        # mc = calfa*rk + (1-calfa)*w - a
        Gamma0[idx_mc, idx_mc] = 1.0
        Gamma0[idx_mc, idx_rk] = -calfa
        Gamma0[idx_mc, idx_w] = -(1.0 - calfa)
        Gamma0[idx_mc, idx_a] = 1.0

        # Equation 13: Capital law of motion (row 12: kp)
        # kp = (1-cikbar)*kp(-1) + cikbar*inve + cikbar*(cgamma^2*csadjcost)*qs
        Gamma0[idx_kp, idx_kp] = 1.0
        Gamma0[idx_kp, idx_inve] = -cikbar
        Gamma0[idx_kp, idx_qs] = -cikbar * (cgamma**2 * csadjcost)
        Gamma1[idx_kp, idx_kp] = 1.0 - cikbar

        # ===================================================================
        # LAG EQUATIONS (8 equations)
        # ===================================================================

        # c_lag = c(-1)
        Gamma0[idx_c_lag, idx_c_lag] = 1.0
        Gamma1[idx_c_lag, idx_c] = 1.0

        # inve_lag = inve(-1)
        Gamma0[idx_inve_lag, idx_inve_lag] = 1.0
        Gamma1[idx_inve_lag, idx_inve] = 1.0

        # y_lag = y(-1)
        Gamma0[idx_y_lag, idx_y_lag] = 1.0
        Gamma1[idx_y_lag, idx_y] = 1.0

        # w_lag = w(-1)
        Gamma0[idx_w_lag, idx_w_lag] = 1.0
        Gamma1[idx_w_lag, idx_w] = 1.0

        # r_lag = r(-1)
        Gamma0[idx_r_lag, idx_r_lag] = 1.0
        Gamma1[idx_r_lag, idx_r] = 1.0

        # pinf_lag = pinf(-1)
        Gamma0[idx_pinf_lag, idx_pinf_lag] = 1.0
        Gamma1[idx_pinf_lag, idx_pinf] = 1.0

        # kp_lag = kp(-1)
        Gamma0[idx_kp_lag, idx_kp_lag] = 1.0
        Gamma1[idx_kp_lag, idx_kp] = 1.0

        # kpf_lag = kpf(-1)
        Gamma0[idx_kpf_lag, idx_kpf_lag] = 1.0
        Gamma1[idx_kpf_lag, idx_kpf] = 1.0

        # ===================================================================
        # SHOCK PROCESSES (7 equations)
        # ===================================================================

        # a = crhoa*a(-1) + ea
        Gamma0[idx_a, idx_a] = 1.0
        Gamma1[idx_a, idx_a] = crhoa
        Psi[idx_a, idx_ea] = sigma_ea

        # b = crhob*b(-1) + eb
        Gamma0[idx_b, idx_b] = 1.0
        Gamma1[idx_b, idx_b] = crhob
        Psi[idx_b, idx_eb] = sigma_eb

        # g = crhog*g(-1) + eg + cgy*ea
        Gamma0[idx_g, idx_g] = 1.0
        Gamma1[idx_g, idx_g] = crhog
        Psi[idx_g, idx_eg] = sigma_eg
        Psi[idx_g, idx_ea] = cgy * sigma_ea

        # qs = crhoqs*qs(-1) + eqs
        Gamma0[idx_qs, idx_qs] = 1.0
        Gamma1[idx_qs, idx_qs] = crhoqs
        Psi[idx_qs, idx_eqs] = sigma_eqs

        # ms = crhoms*ms(-1) + em
        # (This equation conflicts with Taylor rule assignment above - need to fix)
        Gamma0[idx_ms, idx_ms] = 1.0
        Gamma1[idx_ms, idx_ms] = crhoms
        Psi[idx_ms, idx_em] = sigma_em

        # spinf = crhopinf*spinf(-1) + epinf - cmap*epinfma_lag
        Gamma0[idx_spinf, idx_spinf] = 1.0
        Gamma0[idx_spinf, idx_epinfma_lag] = cmap
        Gamma1[idx_spinf, idx_spinf] = crhopinf
        Psi[idx_spinf, idx_epinf] = sigma_epinf

        # sw = crhow*sw(-1) + ew - cmaw*ewma_lag
        Gamma0[idx_sw, idx_sw] = 1.0
        Gamma0[idx_sw, idx_ewma_lag] = cmaw
        Gamma1[idx_sw, idx_sw] = crhow
        Psi[idx_sw, idx_ew] = sigma_ew

        # ===================================================================
        # MA LAG EQUATIONS (2 equations)
        # ===================================================================

        # epinfma_lag = epinf(-1)
        Gamma0[idx_epinfma_lag, idx_epinfma_lag] = 1.0
        Psi[idx_epinfma_lag, idx_epinf] = sigma_epinf

        # ewma_lag = ew(-1)
        Gamma0[idx_ewma_lag, idx_ewma_lag] = 1.0
        Psi[idx_ewma_lag, idx_ew] = sigma_ew

        return {"Gamma0": Gamma0, "Gamma1": Gamma1, "Psi": Psi, "Pi": Pi}

    def measurement_equation(
        self, params: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Measurement equation: observables = Z * states + D.

        Returns:
            Z: (n_obs x n_states) measurement matrix
            D: (n_obs,) constant vector
        """
        # Get parameters
        if params is None:
            p = self.parameters.to_dict()
        else:
            param_names = [param.name for param in self.parameters.parameters]
            p = dict(zip(param_names, params, strict=False))

        # Compute derived parameters
        derived = self._compute_steady_state_params(p)
        p.update(derived)

        ctrend = p["ctrend"]
        constepinf = p["constepinf"]
        conster = p["conster"]
        constelab = p["constelab"]

        n_obs = self.spec.n_observables
        n_states = self.spec.n_states

        Z = np.zeros((n_obs, n_states))
        D = np.zeros(n_obs)

        # State indices
        idx_c, idx_inve, idx_y = 0, 1, 2
        idx_w, idx_r, idx_pinf = 8, 9, 10
        idx_lab = 3
        idx_y_lag, idx_w_lag = 26, 27

        # Observable indices
        idx_obs_dy = 0
        idx_obs_dc = 1
        idx_obs_dinve = 2
        idx_obs_dw = 3
        idx_obs_pinfobs = 4
        idx_obs_robs = 5
        idx_obs_labobs = 6

        # dy = y - y_lag + ctrend
        Z[idx_obs_dy, idx_y] = 1.0
        Z[idx_obs_dy, idx_y_lag] = -1.0
        D[idx_obs_dy] = ctrend

        # dc = c - c_lag + ctrend
        Z[idx_obs_dc, idx_c] = 1.0
        Z[idx_obs_dc, 24] = -1.0  # c_lag
        D[idx_obs_dc] = ctrend

        # dinve = inve - inve_lag + ctrend
        Z[idx_obs_dinve, idx_inve] = 1.0
        Z[idx_obs_dinve, 25] = -1.0  # inve_lag
        D[idx_obs_dinve] = ctrend

        # dw = w - w_lag + ctrend
        Z[idx_obs_dw, idx_w] = 1.0
        Z[idx_obs_dw, idx_w_lag] = -1.0
        D[idx_obs_dw] = ctrend

        # pinfobs = pinf + constepinf
        Z[idx_obs_pinfobs, idx_pinf] = 1.0
        D[idx_obs_pinfobs] = constepinf

        # robs = r + conster
        Z[idx_obs_robs, idx_r] = 1.0
        D[idx_obs_robs] = conster

        # labobs = lab + constelab
        Z[idx_obs_labobs, idx_lab] = 1.0
        D[idx_obs_labobs] = constelab

        return Z, D

    def steady_state(self, params: np.ndarray | None = None) -> np.ndarray:
        """
        Compute steady state (all zeros for log-linearized model).

        Returns:
            Steady state vector (all zeros since model is in log-deviations)
        """
        return np.zeros(self.spec.n_states)


def create_smets_wouters_model() -> SmetsWouters2007:
    """Factory function to create a Smets-Wouters 2007 model instance."""
    return SmetsWouters2007()


if __name__ == "__main__":
    # Example usage
    model = create_smets_wouters_model()

    # Test system matrices
    mats = model.system_matrices()

    # Test measurement equation
    Z, D = model.measurement_equation()
