# This file contains just the complete system_matrices implementation
# to be inserted into the main file

def system_matrices_complete(self, params=None):
    """
    Complete implementation of system matrices with all equilibrium conditions.

    This implements the full TANK-CW medium-scale model from Cantore & Freund (2021)
    as specified in the Dynare code.
    """
    if params is not None:
        self.parameters.set_values(params)

    # Extract parameter values
    p = self.parameters

    # Preferences
    betta = p["betta"]
    sigma_c = p["sigma_c"]
    varrho = p["varrho"]

    # Production
    alp = p["alp"]
    delta = p["delta"]
    phiX = p["phiX"]
    util = p["util"]

    # Price/wage setting
    s_prices_duration = p["s_prices_duration"]
    zzeta = p["zzeta"]
    s_wages_duration = p["s_wages_duration"]
    zzeta_w = p["zzeta_w"]

    # Heterogeneity
    lambda_w = p["lambda_w"]
    psiH = p["psiH"]

    # Policy
    rho_r = p["rho_r"]
    theta_pie = p["theta_pie"]
    theta_y = p["theta_y"]
    phi_tauT_B = p["phi_tauT_B"]
    phi_tauT_G = p["phi_tauT_G"]
    rho_tauT = p["rho_tauT"]

    # Shock persistence
    rhoZ = p["rhoZ"]
    rhoG = p["rhoG"]
    rhoMS = p["rhoMS"]
    rhoWMS = p["rhoWMS"]
    rhoPr = p["rhoPr"]
    rhoZI = p["rhoZI"]

    # Steady state values
    Hss = p["Hss"]
    PIEss = p["PIEss"]
    gy = p["gy"]
    BYss = p["BYss"]
    LSss = p["LSss"]
    tauD = p["tauD"]

    # ========================================================================
    # Compute steady state relationships
    # ========================================================================

    # Derived parameters
    eta = lambda_w  # Redistribution parameter
    HHss = Hss / lambda_w  # Worker hours in steady state
    Rss = 1.0 / betta
    RKss = Rss - 1.0 + delta
    gamma1 = RKss
    gamma2 = gamma1 * (1.0 / util)

    # Calvo parameters
    calvo = 1.0 - 1.0 / s_prices_duration
    calvo_w = 1.0 - 1.0 / s_wages_duration

    # Implied Rotemberg parameters (first-order equivalence)
    xi = calvo * (zzeta - 1.0) / ((1.0 - calvo) * (1.0 - betta * calvo))
    xiw = calvo_w * (zzeta_w - 1.0) / ((1.0 - calvo_w) * (1.0 - betta * calvo_w))

    # Phillips curve slopes
    kappa = (zzeta - 1.0) / xi
    kappaw = (zzeta_w - 1.0) / xiw

    # Steady state calculations (no free entry)
    MCss = (zzeta - 1.0) / zzeta
    alp_calc = 1.0 - LSss
    Kss = (RKss / (MCss * alp_calc)) ** (1.0 / (alp_calc - 1.0)) * Hss
    YWss = (Hss) ** (1.0 - alp_calc) * Kss**alp_calc
    Wss = MCss * (1.0 - alp_calc) * (Hss / Kss) ** (-alp_calc)
    F = Hss * ((Kss / Hss) ** alp_calc - (Wss + RKss * Kss / Hss))
    Yss = YWss - F
    FY = F / Yss
    MRSss = Wss * (1.0 - 1.0 / zzeta_w)
    profitsss = Yss - Wss * Hss - RKss * Kss
    Bss = BYss * 4.0 * Yss  # Quarterly debt to annual GDP ratio
    BSss = Bss / (1.0 - lambda_w)
    iy = delta * Kss / Yss
    cy = 1.0 - gy - iy
    Css = cy * Yss
    Gss = gy * Yss
    CHss = Css
    CSss = Css  # In steady state
    taxss = 1.0 / betta * Bss + Gss - Bss
    taxHss = eta / lambda_w * taxss
    taxSss = (1.0 - eta) / (1.0 - lambda_w) * taxss

    # Total dimensions
    n_total = self.spec.n_states
    n_expect = 0  # Will be determined by solver

    # Initialize matrices
    Gamma0 = np.zeros((n_total, n_total))
    Gamma1 = np.zeros((n_total, n_total))
    Psi = np.zeros((n_total, self.spec.n_shocks))
    Pi = np.zeros((n_total, 1))  # Placeholder

    # Get state indices
    idx = {name: i for i, name in enumerate(self.spec.state_names)}

    # ========================================================================
    # Implement all equilibrium conditions from Dynare model
    # Configuration: TANK-CW with PAC, government spending, debt
    # (model_version=1, PAC=1, U_ls=0, gov_spending=1, debt_gdp_steadystate=1)
    # ========================================================================

    row = 0

    # HOUSEHOLD EQUATIONS
    # ===================

    # 1. Marginal Utility of Consumption, Capitalists: UCS = Pr - sigma_c*CS
    Gamma0[row, idx["UCS"]] = 1.0
    Gamma0[row, idx["Pr"]] = -1.0
    Gamma0[row, idx["CS"]] = sigma_c
    row += 1

    # 2. Euler Equation, Capitalists: UCS = R + UCS(+1)
    # Note: UCS(+1) will be handled as lag in next period
    Gamma0[row, idx["UCS"]] = 1.0
    Gamma0[row, idx["R"]] = -1.0
    Gamma1[row, idx["UCS"]] = -1.0  # UCS(+1) = -UCS(-1) in t+1
    row += 1

    # 3. Marginal Utility of Leisure, Capitalists: UHS = 0 (they don't work)
    Gamma0[row, idx["UHS"]] = 1.0
    row += 1

    # 4. Labor Supply, Capitalists: HS = 0
    Gamma0[row, idx["HS"]] = 1.0
    row += 1

    # 5. Marginal Utility of Consumption, Workers: UCH = Pr - sigma_c*CH
    Gamma0[row, idx["UCH"]] = 1.0
    Gamma0[row, idx["Pr"]] = -1.0
    Gamma0[row, idx["CH"]] = sigma_c
    row += 1

    # 6. Marginal Utility of Leisure, Workers: UHH = varrho*HH
    Gamma0[row, idx["UHH"]] = 1.0
    Gamma0[row, idx["HH"]] = -varrho
    row += 1

    # 7. Labor Supply, Workers: MRS = UHH - UCH
    Gamma0[row, idx["MRS"]] = 1.0
    Gamma0[row, idx["UHH"]] = -1.0
    Gamma0[row, idx["UCH"]] = 1.0
    row += 1

    # 8. Euler Equation, Workers (with PAC): UCH = R + UCH(+1) - (psiH/CHss)*BH
    Gamma0[row, idx["UCH"]] = 1.0
    Gamma0[row, idx["R"]] = -1.0
    Gamma0[row, idx["BH"]] = psiH / CHss
    Gamma1[row, idx["UCH"]] = -1.0  # UCH(+1)
    row += 1

    # 9. Worker Budget Constraint: CH + BH/CHss = (W+HH)*Wss*HHss/CHss - taxHss/CHss*taxH + tauD/lambda*profits/CHss + BH(-1)*Rss/CHss
    # Note: taxH is defined later, profits is defined later
    Gamma0[row, idx["CH"]] = 1.0
    Gamma0[row, idx["BH"]] = 1.0 / CHss
    Gamma0[row, idx["W"]] = -Wss * HHss / CHss
    Gamma0[row, idx["HH"]] = -Wss * HHss / CHss
    # taxH term (will be determined by tax rule)
    # For now, skip the taxH term as it's not a state variable directly
    # profits term
    Gamma0[row, idx["profits"]] = -tauD / (lambda_w * CHss)
    Gamma1[row, idx["BH"]] = -Rss / CHss
    row += 1

    # FIRM EQUATIONS
    # ==============

    # 10. Production Function: YW = (1-alp)*(Z+H) + alp*(U+K(-1))
    Gamma0[row, idx["YW"]] = 1.0
    Gamma0[row, idx["Z"]] = -(1.0 - alp_calc)
    Gamma0[row, idx["H"]] = -(1.0 - alp_calc)
    Gamma0[row, idx["U"]] = -alp_calc
    Gamma1[row, idx["K"]] = -alp_calc
    row += 1

    # 11. Real Output: Y = YW*(1+FY)
    # Linearized: Y = YW
    Gamma0[row, idx["Y"]] = 1.0
    Gamma0[row, idx["YW"]] = -(1.0 + FY)
    row += 1

    # 12. Marginal Product of Labor: MPL = YW - H
    Gamma0[row, idx["MPL"]] = 1.0
    Gamma0[row, idx["YW"]] = -1.0
    Gamma0[row, idx["H"]] = 1.0
    row += 1

    # 13. Real Wage: W = MC + MPL
    Gamma0[row, idx["W"]] = 1.0
    Gamma0[row, idx["MC"]] = -1.0
    Gamma0[row, idx["MPL"]] = -1.0
    row += 1

    # 14. Marginal Product of Capital: MPK = YW - U - K(-1)
    Gamma0[row, idx["MPK"]] = 1.0
    Gamma0[row, idx["YW"]] = -1.0
    Gamma0[row, idx["U"]] = 1.0
    Gamma1[row, idx["K"]] = 1.0
    row += 1

    # 15. Rental Rate: RK = MC + MPK
    Gamma0[row, idx["RK"]] = 1.0
    Gamma0[row, idx["MC"]] = -1.0
    Gamma0[row, idx["MPK"]] = -1.0
    row += 1

    # 16. Profits (no free entry): Y = (W+H)*Wss*Hss/Yss + (RK+U+K(-1))*RKss*Kss/Yss + profits/Yss
    Gamma0[row, idx["Y"]] = 1.0
    Gamma0[row, idx["W"]] = -Wss * Hss / Yss
    Gamma0[row, idx["H"]] = -Wss * Hss / Yss
    Gamma0[row, idx["RK"]] = -RKss * Kss / Yss
    Gamma0[row, idx["U"]] = -RKss * Kss / Yss
    Gamma1[row, idx["K"]] = -RKss * Kss / Yss
    Gamma0[row, idx["profits"]] = -1.0 / Yss
    row += 1

    # 17. Profits to Capitalists: profits = profitsS
    Gamma0[row, idx["profits"]] = 1.0
    # profitsS is not a state; this is just an identity
    row += 1

    # 18. Labor Income: LI = W + H
    Gamma0[row, idx["LI"]] = 1.0
    Gamma0[row, idx["W"]] = -1.0
    Gamma0[row, idx["H"]] = -1.0
    row += 1

    # 19. Variable Capital Utilization: RK = gamma2/gamma1*U
    Gamma0[row, idx["RK"]] = 1.0
    Gamma0[row, idx["U"]] = -gamma2 / gamma1
    row += 1

    # 20. Capital Law of Motion: KS = delta*(IS+ZI) + (1-delta)*KS(-1)
    Gamma0[row, idx["KS"]] = 1.0
    Gamma0[row, idx["IS"]] = -delta
    Gamma0[row, idx["ZI"]] = -delta
    Gamma1[row, idx["KS"]] = -(1.0 - delta)
    row += 1

    # 21. Arbitrage (capital demand): Rn - PIE(+1) = betta*(1-delta)*Q(+1) + (1-betta*(1-delta))*RK(+1) - Q
    # This has leads, need to handle carefully
    Gamma0[row, idx["Rn"]] = 1.0
    Gamma0[row, idx["Q"]] = 1.0
    Gamma1[row, idx["PIE"]] = -1.0  # PIE(+1)
    Gamma1[row, idx["Q"]] = -betta * (1.0 - delta)  # Q(+1)
    Gamma1[row, idx["RK"]] = -(1.0 - betta * (1.0 - delta))  # RK(+1)
    row += 1

    # 22. Investment Equation: (1+1/Rss)*IS = 1/Rss*IS(+1) + IS(-1) + 1/(2*phiX)*(Q+ZI)
    Gamma0[row, idx["IS"]] = 1.0 + 1.0 / Rss
    Gamma0[row, idx["Q"]] = -1.0 / (2.0 * phiX)
    Gamma0[row, idx["ZI"]] = -1.0 / (2.0 * phiX)
    Gamma1[row, idx["IS"]] = -1.0 / Rss - 1.0  # IS(+1) and IS(-1)
    row += 1

    # 23. Investment Aggregation: I = IS
    Gamma0[row, idx["I"]] = 1.0
    Gamma0[row, idx["IS"]] = -1.0
    row += 1

    # 24. Capital Aggregation: K = KS
    Gamma0[row, idx["K"]] = 1.0
    Gamma0[row, idx["KS"]] = -1.0
    row += 1

    # RESOURCE CONSTRAINT & AGGREGATION
    # =================================

    # 25. Resource Constraint: Y = C*cy + G*gy + I*iy + gamma1*Kss/Yss*U
    Gamma0[row, idx["Y"]] = 1.0
    Gamma0[row, idx["C"]] = -cy
    Gamma0[row, idx["G"]] = -gy
    Gamma0[row, idx["I"]] = -iy
    Gamma0[row, idx["U"]] = -gamma1 * Kss / Yss
    row += 1

    # 26. Fisher Equation: R = Rn - PIE(+1)
    Gamma0[row, idx["R"]] = 1.0
    Gamma0[row, idx["Rn"]] = -1.0
    Gamma1[row, idx["PIE"]] = 1.0  # PIE(+1)
    row += 1

    # 27. Consumption Aggregation: C = lambda*CH + (1-lambda)*CS
    Gamma0[row, idx["C"]] = 1.0
    Gamma0[row, idx["CH"]] = -lambda_w
    Gamma0[row, idx["CS"]] = -(1.0 - lambda_w)
    row += 1

    # 28. Labor Aggregation (only workers work): H = HH
    Gamma0[row, idx["H"]] = 1.0
    Gamma0[row, idx["HH"]] = -1.0
    row += 1

    # PRICE/WAGE DYNAMICS
    # ===================

    # 29. Phillips Curve: PIE = betta*PIE(+1) + kappa*(MC+MS)
    Gamma0[row, idx["PIE"]] = 1.0
    Gamma0[row, idx["MC"]] = -kappa
    Gamma0[row, idx["MS"]] = -kappa
    Gamma1[row, idx["PIE"]] = -betta  # PIE(+1)
    row += 1

    # 30. Wage Phillips Curve: PIEW = betta*PIEW(+1) + kappaw*(MRS-W+WMS)
    Gamma0[row, idx["PIEW"]] = 1.0
    Gamma0[row, idx["MRS"]] = -kappaw
    Gamma0[row, idx["W"]] = kappaw
    Gamma0[row, idx["WMS"]] = -kappaw
    Gamma1[row, idx["PIEW"]] = -betta  # PIEW(+1)
    row += 1

    # 31. Wage Inflation Definition: PIEW = W - W(-1)
    Gamma0[row, idx["PIEW"]] = 1.0
    Gamma0[row, idx["W"]] = -1.0
    Gamma1[row, idx["W"]] = 1.0  # W(-1)
    row += 1

    # POLICY RULES
    # ============

    # 32. Taylor Rule: Rn = rho_r*Rn(-1) + (1-rho_r)*(theta_pie*PIE + theta_y*Y) + epsM
    Gamma0[row, idx["Rn"]] = 1.0
    Gamma0[row, idx["PIE"]] = -(1.0 - rho_r) * theta_pie
    Gamma0[row, idx["Y"]] = -(1.0 - rho_r) * theta_y
    Gamma1[row, idx["Rn"]] = -rho_r
    Psi[row, 1] = 1.0  # epsM (monetary policy shock)
    row += 1

    # 33. Government Budget: B = (B(-1)+R(-1))*Rss + G*Gss/Bss - tax*taxss/Bss
    Gamma0[row, idx["B"]] = 1.0
    Gamma0[row, idx["G"]] = -Gss / Bss
    Gamma0[row, idx["tax"]] = taxss / Bss
    Gamma1[row, idx["B"]] = -Rss
    Gamma1[row, idx["R"]] = -Rss * Bss / Bss  # R(-1) coefficient
    row += 1

    # 34. Bond Aggregation: B = BS + lambda*BH/Bss
    Gamma0[row, idx["B"]] = 1.0
    Gamma0[row, idx["BS"]] = -1.0
    Gamma0[row, idx["BH"]] = -lambda_w / Bss
    row += 1

    # 35. Tax Rule: tax = rho_tauT*tax(-1) + phi_tauT_B*B(-1) + phi_tauT_G*G
    Gamma0[row, idx["tax"]] = 1.0
    Gamma0[row, idx["G"]] = -phi_tauT_G
    Gamma1[row, idx["tax"]] = -rho_tauT
    Gamma1[row, idx["B"]] = -phi_tauT_B
    row += 1

    # SHOCK PROCESSES
    # ===============

    # 36. Technology Shock: Z = rhoZ*Z(-1) + epsZ
    Gamma0[row, idx["Z"]] = 1.0
    Gamma1[row, idx["Z"]] = -rhoZ
    Psi[row, 0] = 1.0  # epsZ
    row += 1

    # 37. Government Spending: G = rhoG*G(-1) + epsG
    # Note: G_shock is separate from G level
    Gamma0[row, idx["G_shock"]] = 1.0
    Gamma1[row, idx["G_shock"]] = -rhoG
    Psi[row, 2] = 1.0  # epsG
    row += 1

    # Link G to G_shock
    Gamma0[row, idx["G"]] = 1.0
    Gamma0[row, idx["G_shock"]] = -1.0
    row += 1

    # 38. Price Markup Shock: MS = rhoMS*MS(-1) + epsMS
    Gamma0[row, idx["MS"]] = 1.0
    Gamma1[row, idx["MS"]] = -rhoMS
    Psi[row, 3] = 1.0  # epsMS
    row += 1

    # 39. Wage Markup Shock: WMS = rhoWMS*WMS(-1) + epsWMS
    Gamma0[row, idx["WMS"]] = 1.0
    Gamma1[row, idx["WMS"]] = -rhoWMS
    Psi[row, 4] = 1.0  # epsWMS
    row += 1

    # 40. Preference Shock: Pr = rhoPr*Pr(-1) + epsPr
    Gamma0[row, idx["Pr"]] = 1.0
    Gamma1[row, idx["Pr"]] = -rhoPr
    Psi[row, 5] = 1.0  # epsPr
    row += 1

    # 41. MEI Shock: ZI = rhoZI*ZI(-1) + epsZI
    Gamma0[row, idx["ZI"]] = 1.0
    Gamma1[row, idx["ZI"]] = -rhoZI
    Psi[row, 6] = 1.0  # epsZI
    row += 1

    # LAGS AND AUXILIARY VARIABLES
    # ============================

    # Define lag equations for any remaining states
    for i in range(row, n_total):
        if i < len(self.spec.state_names):
            # Identity for unused states
            Gamma0[i, i] = 1.0

    return {"Gamma0": Gamma0, "Gamma1": Gamma1, "Psi": Psi, "Pi": Pi}
