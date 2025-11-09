# NYFed DSGE Model 1002 - Translation Documentation

## Overview

This directory contains the Python translation of the New York Federal Reserve DSGE Model (version 1002) as documented in the March 3, 2021 specification.

**Source**: FRBNY DSGE Model Documentation (DSGE_Model_Documentation_1002.pdf)

**Original Implementation**: [FRBNY-DSGE/DSGE.jl](https://github.com/FRBNY-DSGE/DSGE.jl) (Julia)

## Model Characteristics

### Model Type
- **Scale**: Medium-scale New Keynesian DSGE model
- **Sectors**: One sector
- **Agent Types**: 8 classes (households, firms, entrepreneurs, banks, government)

### Key Features

1. **Financial Frictions**
   - Financial accelerator mechanism (Bernanke-Gertler-Gilchrist 1999)
   - Entrepreneurial leverage and credit spreads
   - Time-varying cross-sectional volatility

2. **Nominal Rigidities**
   - Calvo price and wage stickiness with indexation
   - Kimball aggregator for goods and labor (more flexible than Dixit-Stiglitz)

3. **Real Rigidities**
   - Habit formation in consumption
   - Investment adjustment costs
   - Variable capital utilization

4. **Trend and Growth**
   - Deterministic and stochastic productivity trends
   - Balanced growth path

5. **Monetary Policy**
   - Generalized Taylor rule with smoothing
   - Time-varying inflation target
   - Support for anticipated policy shocks (forward guidance)

## Model Structure

### Endogenous Variables (18)
- `c`: Consumption
- `i`: Investment
- `y`: Output
- `L`: Labor/Hours worked
- `k_bar`: Installed capital stock
- `k`: Effective capital (with utilization)
- `u`: Capital utilization rate
- `q_k`: Tobin's q (value of capital)
- `w`: Real wage
- `R`: Nominal interest rate
- `pi`: Inflation
- `mc`: Real marginal cost
- `r_k`: Rental rate of capital
- `R_k_tilde`: Gross nominal return on capital for entrepreneurs
- `n`: Entrepreneurial net worth
- `w_h`: Household marginal rate of substitution (labor supply)
- `y_f`: Flexible-price output (for policy rule)
- `pi_star`: Time-varying inflation target

### Exogenous Shocks (9)
1. `z_tilde`: Stationary productivity shock
2. `z_p`: Stochastic trend productivity growth shock
3. `b`: Risk premium shock
4. `mu`: Marginal efficiency of investment (MEI) shock
5. `g`: Government spending shock
6. `lambda_f`: Price markup shock (ARMA)
7. `lambda_w`: Wage markup shock (ARMA)
8. `sigma_omega`: Cross-sectional volatility shock (financial)
9. `r_m`: Monetary policy shock (with anticipated components)

### Observable Variables (13)
1. GDP growth
2. GDI growth
3. Consumption growth
4. Investment growth
5. Real wage growth
6. Hours worked
7. Core PCE inflation
8. GDP deflator inflation
9. Federal Funds Rate (FFR)
10. 10-year Treasury yield
11. 10-year inflation expectations
12. Credit spread (Baa-Treasury)
13. Total Factor Productivity (TFP) growth

### Parameters (~70+)

**Policy Parameters (6)**:
- Taylor rule coefficients (ψ₁, ψ₂, ψ₃)
- Interest rate smoothing (ρᵣ)
- Monetary shock persistence and volatility

**Nominal Rigidities (6)**:
- Calvo parameters for prices and wages (ζₚ, ζ_w)
- Indexation parameters (ιₚ, ι_w)
- Kimball curvature parameters (εₚ, ε_w)

**Steady State & Preferences (11)**:
- Steady-state growth rate (γ)
- Capital share (α)
- Discount factor (β)
- Risk aversion (σ_c)
- Habit persistence (h)
- Labor supply elasticity (ν_l)
- Investment adjustment cost (S'')
- Capital utilization cost (ψ)
- Depreciation rate (δ)
- Steady-state inflation (π*)
- Production fixed cost (Φₚ)

**Financial Frictions (4)**:
- Steady-state default probability (F(ω̄))
- Spread elasticity (ζ_sp,b)
- Steady-state spread (SP*)
- Entrepreneur survival rate (γ*)

**Shock Processes (18+ parameters)**:
- Persistence parameters (ρ) for each shock
- Standard deviations (σ) for each shock
- MA coefficients for markup shocks (η)

**Measurement Errors (12+ parameters)**:
- Persistence and volatility for each observable's measurement error
- Correlation between GDP and GDI errors

## Log-Linearized Equilibrium Conditions

The model consists of approximately 20+ log-linear equations around the balanced growth path:

### Technology and Growth (Equations 3-5)
```
z̃ₜ = ρ_z z̃ₜ₋₁ + σ_z εₜᶻ

zₜᵖ = ρ_zp zₜ₋₁ᵖ + σ_zp εₜᶻᵖ

zₜ = 1/(1-α)(ρ_z-1)z̃ₜ₋₁ + 1/(1-α)σ_z εₜᶻ + zₜᵖ
```

### Household Behavior (Equations 6, 20)
**Consumption Euler Equation**:
```
cₜ = -(1-he⁻ᵞ)/(σ_c(1+he⁻ᵞ))(Rₜ - E[πₜ₊₁] + bₜ)
    + he⁻ᵞ/(1+he⁻ᵞ)(cₜ₋₁ - zₜ)
    + 1/(1+he⁻ᵞ)E[cₜ₊₁ + zₜ₊₁]
    + (σ_c-1)/(σ_c(1+he⁻ᵞ)) w*L*/c* (Lₜ - E[Lₜ₊₁])
```

**Labor Supply (MRS)**:
```
wₜʰ = 1/(1-he⁻ᵞ)(cₜ - he⁻ᵞcₜ₋₁ + he⁻ᵞzₜ) + ν_l Lₜ
```

### Investment and Capital (Equations 7-10)
**Investment Demand**:
```
iₜ = qₜᵏ/(S''e²ᵞ(1+β̄)) + 1/(1+β̄)(iₜ₋₁ - zₜ) + β̄/(1+β̄)E[iₜ₊₁ + zₜ₊₁] + μₜ
```

**Capital Accumulation**:
```
k̄ₜ = (1 - i*/k̄*)(k̄ₜ₋₁ - zₜ) + i*/k̄* iₜ + i*/k̄* S''e²ᵞ(1+β̄)μₜ
```

**Effective Capital**:
```
kₜ = uₜ - zₜ + k̄ₜ₋₁
```

**Capital Utilization**:
```
(1-ψ)/ψ rₜᵏ = uₜ
```

### Production (Equations 11-12, 16)
**Marginal Cost**:
```
mcₜ = wₜ + αLₜ - αkₜ
```

**Capital-Labor Ratio**:
```
kₜ = wₜ - rₜᵏ + Lₜ
```

**Production Function**:
```
yₜ = Φₚ(αkₜ + (1-α)Lₜ)
```

### Financial Frictions (Equations 13-15)
**Return on Capital**:
```
R̃ₜᵏ - πₜ = rₖ*/(rₖ*+(1-δ)) rₜᵏ + (1-δ)/(rₖ*+(1-δ)) qₜᵏ - qₜ₋₁ᵏ
```

**Credit Spread**:
```
E[R̃ₜ₊₁ᵏ - Rₜ] = bₜ + ζ_sp,b(qₜᵏ + k̄ₜ - nₜ) + σ̃_ω,t
```

**Net Worth Evolution**:
```
nₜ = ζ_n,R̃ᵏ(R̃ₜᵏ - πₜ) - ζ_n,R(Rₜ₋₁ - πₜ + bₜ₋₁) + ζ_n,qK(qₜ₋₁ᵏ + k̄ₜ₋₁)
    + ζ_n,n nₜ₋₁ - γ*v*/n* zₜ - ζ_n,σω/ζ_sp,σω σ̃_ω,t-1
```

### Equilibrium (Equation 17)
**Resource Constraint**:
```
yₜ = g* gₜ + c*/y* cₜ + i*/y* iₜ + rₖ*k*/y* uₜ
```

### Price and Wage Setting (Equations 18-19)
**New Keynesian Phillips Curve**:
```
πₜ = κ mcₜ + ιₚ/(1+ιₚβ̄) πₜ₋₁ + β̄/(1+ιₚβ̄) E[πₜ₊₁] + λₜᶠ
```
where κ = (1-ζₚβ̄)(1-ζₚ)/((1+ιₚβ̄)ζₚ((Φₚ-1)εₚ+1))

**Wage Phillips Curve**:
```
wₜ = (1-ζ_wβ̄)(1-ζ_w)/((1+β̄)ζ_w((λ_w-1)ε_w+1)) (wₜʰ - wₜ)
    - (1+ι_wβ̄)/(1+β̄) πₜ
    + 1/(1+β̄)(wₜ₋₁ - zₜ + ι_w πₜ₋₁)
    + β̄/(1+β̄)E[wₜ₊₁ + zₜ₊₁ + πₜ₊₁] + λₜʷ
```

### Monetary Policy (Equations 21-22)
**Taylor Rule**:
```
Rₜ = ρ_R Rₜ₋₁ + (1-ρ_R)(ψ₁(πₜ - πₜ*) + ψ₂(yₜ - yₜᶠ))
    + ψ₃((yₜ - yₜᶠ) - (yₜ₋₁ - yₜ₋₁ᶠ)) + rₜᵐ
```

**Time-Varying Inflation Target**:
```
πₜ* = ρ_π* πₜ₋₁* + σ_π* εₜᵖⁱ*
```

## Measurement Equations

Observable variables are linked to model states through measurement equations (Equation system 32):

```python
GDP growth     = 100γ + (yₜ - yₜ₋₁ + zₜ) + eₜᵍᵈᵖ - C_me eₜ₋₁ᵍᵈᵖ
GDI growth     = 100γ + (yₜ - yₜ₋₁ + zₜ) + eₜᵍᵈⁱ - C_me eₜ₋₁ᵍᵈⁱ
Cons growth    = 100γ + (cₜ - cₜ₋₁ + zₜ)
Inv growth     = 100γ + (iₜ - iₜ₋₁ + zₜ)
Wage growth    = 100γ + (wₜ - wₜ₋₁ + zₜ)
Hours          = L̄ + Lₜ
Core PCE Infl  = π* + πₜ + eₜᵖᶜᵉ
GDP Def Infl   = π* + δ_gdpdef + γ_gdpdef πₜ + eₜᵍᵈᵖᵈᵉᶠ
FFR            = R* + Rₜ
10y Rate       = R* + Eₜ[∑ᵏ₌₁⁴⁰ Rₜ₊ₖ/40] + eₜ¹⁰ʸ
10y Infl Exp   = π* + Eₜ[∑ᵏ₌₁⁴⁰ πₜ₊ₖ/40]
Spread         = SP* + Eₜ[R̃ₜ₊₁ᵏ - Rₜ]
TFP growth     = zₜ + α/(1-α)(uₜ - uₜ₋₁) + eₜᵗᶠᵖ
```

## Implementation Status

### ✅ Completed
- [x] Parameter definitions with priors (all ~70 parameters)
- [x] Variable definitions (states, controls, observables)
- [x] Symbolic equation representation
- [x] Measurement equation specification
- [x] Documentation of model structure

### 🚧 In Progress
- [ ] Matrix form equilibrium conditions for solver
- [ ] Steady-state computation functions
- [ ] Integration with linear solver
- [ ] Integration with Kalman filter

### ⏳ To Do
- [ ] Full model testing
- [ ] Calibration utilities
- [ ] Comparison with DSGE.jl output
- [ ] Impulse response function validation
- [ ] Estimation example

## Key Differences from Julia Implementation

1. **Matrix Representation**: The Python implementation uses explicit matrix form for the linear solver, while DSGE.jl uses a more symbolic approach.

2. **Prior Distributions**: Implemented using our `PriorDistribution` class rather than Distributions.jl.

3. **State Space**: Our implementation directly specifies states and observables, while DSGE.jl infers some structure from model definition.

4. **Anticipated Shocks**: The Python implementation will handle anticipated shocks through augmented state vector (per documentation Appendix B).

## References

### Primary Documentation
- FRBNY DSGE Model Documentation (March 3, 2021)
- Available at: https://github.com/FRBNY-DSGE/DSGE.jl/blob/main/docs/DSGE_Model_Documentation_1002.pdf

### Key Papers
- Del Negro, M., M. P. Giannoni, and F. Schorfheide (2015). "Inflation in the Great Recession and New Keynesian Models." *American Economic Journal: Macroeconomics*, 7(1), 168-196.
- Smets, F. and R. Wouters (2007). "Shocks and Frictions in US Business Cycles: A Bayesian DSGE Approach." *American Economic Review*, 97(3), 586-606.
- Christiano, L. J., M. Eichenbaum, and C. L. Evans (2005). "Nominal Rigidities and the Dynamic Effects of a Shock to Monetary Policy." *Journal of Political Economy*, 113(1), 1-45.
- Bernanke, B. S., M. Gertler, and S. Gilchrist (1999). "The Financial Accelerator in a Quantitative Business Cycle Framework." *Handbook of Macroeconomics*, Vol. 1C, 1341-1393.

## Usage Example

```python
from models.nyfed_model_1002 import create_nyfed_model

# Create model instance
model = create_nyfed_model()

# Access parameters
params = {p.name: p.prior.mean for p in model.parameters.values()}

# Get equations
equations = model.get_log_linearized_equations(params)

# Get measurement system
measurements = model.get_measurement_equations(params)

# Compute steady state
ss = model.get_steady_state(params)

print(f"Model: {model.name}")
print(f"Parameters: {len(model.parameters)}")
print(f"States: {len(model.endogenous_states + model.exogenous_states)}")
print(f"Observables: {len(model.observables)}")
```

## Next Steps

1. **Complete Matrix Implementation**: Translate symbolic equations into Γ₀, Γ₁, Ψ, Π matrices for Sims (2002) solver
2. **Steady State Computation**: Implement full non-stochastic steady-state calculation
3. **Testing**: Create unit tests comparing with known solutions
4. **Validation**: Compare impulse responses with DSGE.jl
5. **Estimation**: Run SMC estimation on US macro data
6. **Documentation**: Complete API documentation and tutorials

## Translation Notes

- All variables are expressed as log-deviations from steady state (except where noted)
- Growth rates use the convention that γ is quarterly, annualized rates are 4×quarterly
- Measurement equations include both structural measurement errors and bridging equations
- The model includes COVID-19 specific shocks (see Appendix D) which can be set to zero for pre-2020 estimation

## Contact & Contributions

This translation is part of the `dsge-py` project. For questions or contributions, please refer to the main project README.
