# St. Louis Fed DSGE Model

## Overview

This document provides technical documentation for the St. Louis Fed DSGE model implementation in the dsge-py framework.

The St. Louis Fed DSGE model is a medium-scale Two-Agent New Keynesian (TANK) model that extends the standard DSGE framework to include household heterogeneity and an explicit fiscal sector. It was developed by Miguel Faria-e-Castro at the Federal Reserve Bank of St. Louis.

## Model Features

### 1. **Household Heterogeneity**

Unlike representative agent models, this model features two types of households:

- **Workers (W)**: Constitute ~80% of the population. They supply labor, consume, and can save in government bonds but face portfolio adjustment costs when changing their bond holdings. This gives them a higher marginal propensity to consume (MPC) than capitalists.

- **Capitalists (C/S)**: Constitute ~20% of the population. They do not supply labor but own all capital and are the residual claimants to firm profits. They can freely adjust their portfolios without costs, making them the "savers" in the economy.

### 2. **Explicit Fiscal Sector**

The model includes a rich fiscal block with:
- Government spending (G)
- Lump-sum taxes (tax, taxH, taxS)
- Government debt (B, with separate holdings BH and BS)
- Fiscal policy rules that respond to debt and output

### 3. **Standard New Keynesian Features**

- **Sticky Prices**: Calvo price setting with duration of ~3.5 quarters
- **Sticky Wages**: Calvo wage setting with duration of ~3.5 quarters
- **Capital Accumulation**: Investment with adjustment costs and variable utilization
- **Monetary Policy**: Taylor rule with interest rate smoothing

## Model Structure

### State Vector (49 states)

The model's state vector consists of:

1. **Real Economy Variables** (20):
   - Consumption: C (aggregate), CH (workers), CS (capitalists)
   - Investment: I (aggregate), IS (capitalists)
   - Output: Y, YW (wholesale)
   - Labor: H (aggregate), HH (workers), HS (capitalists)
   - Prices: W (wage), R (real rate), Rn (nominal rate), PIE (inflation), PIEW (wage inflation)
   - Capital: K (aggregate), KS (capitalists), Q (Tobin's Q), U (utilization)
   - Costs: MC (marginal cost)

2. **Fiscal Variables** (5):
   - G (government spending), B (debt), BS (capitalist bonds), BH (worker bonds), tax

3. **Marginal Utilities** (4):
   - UCS, UCH, UHH, UHS

4. **Additional Variables** (6):
   - MPL, MPK, RK, MRS, profits, LI

5. **Lags** (7):
   - C_lag, I_lag, W_lag, K_lag, Q_lag, B_lag, PIE_lag

6. **Shocks** (7):
   - Z (technology), MS (price markup), WMS (wage markup), Pr (preference),
     ZI (MEI), G_shock, M_shock

### Observable Variables (13)

The model maps to 13 quarterly US macroeconomic time series:

1. **dy**: GDP growth (GDPC1)
2. **dc**: Consumption growth (PCECC96)
3. **dinve**: Investment growth (GPDIC1)
4. **dg**: Government spending growth (GCEC1)
5. **hours**: Hours worked (HOANBS)
6. **dw**: Real wage growth (COMPNFB / GDP deflator)
7. **infl**: Core PCE inflation (PCEPILFE)
8. **ffr**: Federal Funds Rate (FEDFUNDS)
9. **r10y**: 10-Year Treasury Rate (GS10)
10. **infl_exp**: Inflation expectations (EXPINF10YR)
11. **ls**: Labor share (LABSHPUSA156NRUG)
12. **debt_gdp**: Debt-to-GDP ratio (GFDGDPA188S)
13. **tax_gdp**: Tax revenue-to-GDP ratio (FGRECPT)

See `data/stlouisfed_fred_mapping.py` for complete FRED series mappings and transformations.

### Shocks (7)

1. **epsZ**: Technology shock (affects productivity)
2. **epsM**: Monetary policy shock (affects interest rate)
3. **epsG**: Government spending shock
4. **epsMS**: Price markup shock (cost-push inflation)
5. **epsWMS**: Wage markup shock
6. **epsPr**: Preference shock (affects consumption demand)
7. **epsZI**: Marginal efficiency of investment shock

## Parameters

### Key Structural Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| betta | 0.99 | Discount factor (quarterly) |
| sigma_c | 1.0 | Risk aversion / inverse IES |
| varrho | 1.0 | Inverse Frisch elasticity |
| alp | 0.33 | Capital share |
| delta | 0.025 | Depreciation rate (quarterly) |
| phiX | 2.0 | Investment adjustment cost |
| util | 0.495 | Capital utilization parameter |

### Heterogeneity Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| lambda_w | 0.7967 | Share of workers in population |
| psiH | 0.0742 | Portfolio adjustment cost for workers |

### Policy Parameters

**Monetary Policy (Taylor Rule):**
- rho_r = 0.7 (interest rate smoothing)
- theta_pie = 1.5 (response to inflation)
- theta_y = 0.125 (response to output)

**Fiscal Policy:**
- phi_tauT_B = 0.33 (tax response to debt)
- phi_tauT_G = 0.1 (tax response to spending)

### Steady State Calibration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Hss | 0.33 | Steady state hours |
| PIEss | 1.0 | Steady state gross inflation |
| gy | 0.20 | Government spending/GDP ratio |
| BYss | 0.57 | Debt to quarterly GDP ratio |
| LSss | 0.67 | Steady state labor share |

## References

### Primary Documentation

**St. Louis Fed Working Paper:**
- Faria-e-Castro, Miguel (2024). "The St. Louis Fed DSGE Model."
  Federal Reserve Bank of St. Louis Working Paper 2024-014.
  https://s3.amazonaws.com/real.stlouisfed.org/wp/2024/2024-014.pdf

### Base Model

**Theoretical Foundation:**
- Cantore, Cristiano & Freund, Lukas B. (2021). "Workers, capitalists, and the government:
  fiscal policy and income (re)distribution." *Journal of Monetary Economics*, 119, 58-74.
  https://doi.org/10.1016/j.jmoneco.2021.01.003

### Reference Implementation

**Dynare Code:**
- Cantore-Freund TANK Model Replication
  https://github.com/ccantore/TANK-CW_Replication

## Usage Example

```python
from models.stlouisfed_dsge import create_stlouisfed_dsge
from dsge.solvers.linear import solve_linear_model

# Create model
model = create_stlouisfed_dsge()

# Get system matrices
matrices = model.system_matrices()

# Note: Full model solution requires complete equilibrium system implementation
# Current version provides framework structure with simplified equations
```

## Data Requirements

### Estimation Period
- Recommended: 1960:Q1 onwards
- Some series (e.g., Core PCE inflation, inflation expectations) have shorter histories
- Kalman filter can be used to infer missing early observations

### Data Sources
All data from FRED (Federal Reserve Economic Data):
- Real activity: GDP, Consumption, Investment, Government spending
- Labor market: Hours worked, Wage growth
- Prices: Core PCE inflation, Inflation expectations
- Interest rates: Federal Funds Rate, 10-Year Treasury
- Fiscal: Labor share, Debt/GDP, Tax revenue/GDP

## Implementation Status

### ✅ Completed
- Model specification (states, shocks, observables)
- Parameter definitions with Bayesian priors
- FRED data mappings
- Measurement equations
- Basic framework structure
- Comprehensive test suite (20 tests passing)

### ⚠️ Partial Implementation
- System matrices (equilibrium conditions): Basic structure implemented, requires full equation system from Dynare code
- Solution: Requires complete system_matrices implementation

### 📋 Next Steps

To make this model fully operational for estimation:

1. **Complete Equilibrium Conditions**: Translate all ~40-50 equilibrium equations from the Cantore-Freund Dynare code (`tank_cw_ms.mod`) into the `system_matrices` method.

2. **Verify Steady State**: Ensure all steady state relationships match the Dynare implementation.

3. **Test Solution**: Verify that the model solves and produces stable dynamics.

4. **Validate IRFs**: Compare impulse responses to those from the Dynare code.

5. **Prepare Data**: Download FRED series and apply transformations.

6. **Run Estimation**: Use SMC estimation framework to estimate parameters.

## Model Equations (Simplified)

The full model consists of ~40+ equilibrium conditions. Key equations include:

### Euler Equations
- Capitalists: `UCS_t = R_t + E_t[UCS_{t+1}]`
- Workers: `UCH_t = R_t + E_t[UCH_{t+1}] - (psiH/CHss)*BH_t`

### Production
- `YW_t = (1-α)(Z_t + H_t) + α(U_t + K_{t-1})`

### Price Setting (Phillips Curve)
- `PIE_t = β*E_t[PIE_{t+1}] + κ*(MC_t + MS_t)`

### Wage Setting
- `PIEW_t = β*E_t[PIEW_{t+1}] + κ_w*(MRS_t - W_t + WMS_t)`

### Monetary Policy (Taylor Rule)
- `Rn_t = ρ_r*Rn_{t-1} + (1-ρ_r)*(θ_π*PIE_t + θ_y*Y_t) + epsM_t`

### Fiscal Policy
- Government budget: `B_t = (B_{t-1} + R_{t-1})*Rss + G_t*Gss/Bss - tax_t*taxss/Bss`
- Tax rule: `tax_t = ρ_τ*tax_{t-1} + φ_B*B_{t-1} + φ_G*G_t`

### Resource Constraint
- `Y_t = C_t*cy + G_t*gy + I_t*iy + γ_1*Kss/Yss*U_t`

### Aggregation
- `C_t = λ*CH_t + (1-λ)*CS_t`
- `B_t = BS_t + λ*BH_t/Bss`

*Note: Full equation system available in `/tmp/TANK-CW_Replication/TANK models/Dynare master codes/medium scale model/tank_cw_ms.mod`*

## Differences from Other Models

### vs. NY Fed Model 1002
- **Heterogeneity**: TANK (2 agents) vs. Representative agent
- **Fiscal sector**: Explicit vs. Minimal
- **Financial frictions**: None vs. Financial accelerator
- **Focus**: Fiscal policy effects via MPC differences vs. Financial stability

### vs. Smets-Wouters (2007)
- **Heterogeneity**: 2 agents vs. Representative agent
- **Fiscal sector**: Explicit vs. Exogenous
- **Size**: Medium-scale vs. Medium-scale
- **Observables**: 13 (including fiscal) vs. 7 (macro only)

## Technical Notes

1. **Portfolio Adjustment Costs**: The parameter `psiH` controls how costly it is for workers to adjust bond holdings. Higher values make workers more "hand-to-mouth" like.

2. **Redistribution**: The parameter `eta` (set equal to `lambda_w`) controls how tax burdens are distributed between workers and capitalists.

3. **Debt/GDP Ratio**: `BYss = 0.57` refers to quarterly debt-to-GDP. Annualized this is ~2.28, matching US federal debt levels.

4. **Measurement Error**: The model includes measurement error on the 10-year rate (term premium) and inflation expectations to capture features not in the model.

## Contact & Support

For questions about this implementation:
- Open an issue in the repository
- Refer to the original papers for theoretical details
- Check the Dynare replication code for equation-level verification

---

**Last Updated**: 2025-11-11
**Status**: Framework implementation complete, full equilibrium system requires completion
**Version**: 0.1.0
