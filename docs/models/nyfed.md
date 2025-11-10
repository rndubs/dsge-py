# NYFed DSGE Model 1002

The New York Federal Reserve DSGE Model (version 1002) is a medium-scale macroeconomic model used for policy analysis and forecasting. This implementation closely follows the specification documented by the FRBNY research team.

## Model Overview

### Characteristics

- **Scale**: Medium-scale New Keynesian DSGE
- **Sectors**: Single sector with multiple agent types
- **Agent Types**: Households, firms, entrepreneurs, financial intermediaries, government
- **Primary Use**: Monetary policy analysis, macroeconomic forecasting, scenario analysis

### Key Features

**1. Financial Frictions**
- Financial accelerator mechanism (Bernanke, Gertler & Gilchrist 1999)
- Entrepreneurial leverage and default risk
- Credit spreads endogenous to economic conditions
- Time-varying cross-sectional volatility

**2. Nominal Rigidities**
- Calvo price stickiness with partial indexation
- Calvo wage stickiness with partial indexation
- Kimball aggregator (more flexible than standard Dixit-Stiglitz)

**3. Real Rigidities**
- Internal habit formation in consumption
- Investment adjustment costs
- Variable capital utilization
- Fixed costs in production

**4. Monetary Policy**
- Generalized Taylor rule with interest rate smoothing
- Response to inflation and output gap
- Time-varying inflation target
- Support for forward guidance (anticipated shocks)

## Model Structure

### Dimensions

- **Parameters**: 67 (structural + shock + measurement)
- **State Variables**: 48 total
  - 18 endogenous variables
  - 10 lags
  - 9 structural shocks
  - 2 shock MA components
  - 6 measurement errors
  - 2 measurement error lags
  - 1 derived variable
- **Observables**: 13 macroeconomic time series
- **Shocks**: 9 structural + measurement errors

### Endogenous Variables

| Variable | Description |
|----------|-------------|
| `c` | Consumption |
| `i` | Investment |
| `y` | Output |
| `L` | Labor / Hours worked |
| `k_bar` | Installed capital stock |
| `k` | Effective capital (with utilization) |
| `u` | Capital utilization rate |
| `q_k` | Tobin's q (value of capital) |
| `w` | Real wage |
| `R` | Nominal interest rate |
| `pi` | Inflation |
| `mc` | Real marginal cost |
| `r_k` | Rental rate of capital |
| `R_k_tilde` | Gross nominal return on capital |
| `n` | Entrepreneurial net worth |
| `w_h` | Household marginal rate of substitution |
| `y_f` | Flexible-price output |
| `pi_star` | Time-varying inflation target |

### Structural Shocks

| Shock | Description | Persistence |
|-------|-------------|-------------|
| `z_tilde` | Stationary productivity | AR(1) |
| `z_p` | Stochastic trend growth | AR(1) |
| `b` | Risk premium (demand) | AR(1) |
| `mu` | Marginal efficiency of investment | AR(1) |
| `g` | Government spending | AR(1) |
| `lambda_f` | Price markup | ARMA(1,1) |
| `lambda_w` | Wage markup | ARMA(1,1) |
| `sigma_omega` | Financial volatility | AR(1) |
| `r_m` | Monetary policy | AR(1) + anticipated |

### Observables

The model maps to 13 U.S. macroeconomic data series:

| Observable | FRED Code | Transformation |
|------------|-----------|----------------|
| GDP growth | GDPC1 | Quarterly annualized growth |
| GDI growth | GDI | Quarterly annualized growth |
| Consumption growth | PCECC96 | Quarterly annualized growth |
| Investment growth | GPDIC1 | Quarterly annualized growth |
| Real wage growth | COMPRNFB / GDPDEF | Real deflation + growth |
| Hours worked | HOANBS | Level (deviation from mean) |
| Core PCE inflation | PCEPILFE | Year-over-year |
| GDP deflator inflation | GDPDEF | Year-over-year |
| Federal Funds Rate | FEDFUNDS | Level (annualized) |
| 10-year Treasury | GS10 | Level (annualized) |
| 10-year inflation expectations | Computed | From surveys |
| Credit spread | BAA - GS10 | Spread |
| TFP growth | Computed | From growth accounting |

See `data/fred_series_mapping.py` for complete mapping details.

## Key Equations

### Household Sector

**Consumption Euler Equation:**
```
c_t = -(1-h*e^(-γ))/(σ_c(1+h*e^(-γ))) * (R_t - E[π_{t+1}] + b_t)
    + h*e^(-γ)/(1+h*e^(-γ)) * (c_{t-1} - z_t)
    + 1/(1+h*e^(-γ)) * E[c_{t+1} + z_{t+1}]
    + ...
```

where:
- `h`: Habit persistence
- `σ_c`: Risk aversion
- `γ`: Steady-state growth rate
- `b_t`: Risk premium shock

**Labor Supply:**
```
w_t^h = 1/(1-h*e^(-γ)) * (c_t - h*e^(-γ)*c_{t-1} + h*e^(-γ)*z_t) + ν_l * L_t
```

where `ν_l` is the inverse Frisch elasticity.

### Firms

**Production Function:**
```
y_t = Φ_p * (α*k_t + (1-α)*L_t)
```

**Marginal Cost:**
```
mc_t = w_t + α*L_t - α*k_t
```

**New Keynesian Phillips Curve:**
```
π_t = κ*mc_t + ι_p/(1+ι_p*β)*π_{t-1} + β/(1+ι_p*β)*E[π_{t+1}] + λ_t^f
```

where:
- `κ`: Slope parameter (function of Calvo parameter ζ_p)
- `ι_p`: Price indexation
- `λ_t^f`: Price markup shock

### Investment and Capital

**Investment Demand:**
```
i_t = q_t^k/(S''*e^(2γ)*(1+β)) + 1/(1+β)*(i_{t-1} - z_t)
    + β/(1+β)*E[i_{t+1} + z_{t+1}] + μ_t
```

**Capital Accumulation:**
```
k̄_t = (1 - i*/k̄*)*(k̄_{t-1} - z_t) + i*/k̄**i_t + ...
```

**Capital Utilization:**
```
(1-ψ)/ψ * r_t^k = u_t
```

### Financial Sector

**Return on Capital:**
```
R̃_t^k - π_t = r_k*/(r_k*+(1-δ))*r_t^k + (1-δ)/(r_k*+(1-δ))*q_t^k - q_{t-1}^k
```

**Credit Spread (External Finance Premium):**
```
E[R̃_{t+1}^k - R_t] = b_t + ζ_{sp,b}*(q_t^k + k̄_t - n_t) + σ̃_{ω,t}
```

This is the key financial friction: the spread depends on leverage ratio (q_k + k̄ - n).

**Net Worth Evolution:**
```
n_t = ζ_{n,R̃^k}*(R̃_t^k - π_t) - ζ_{n,R}*(R_{t-1} - π_t + b_{t-1})
    + ζ_{n,qK}*(q_{t-1}^k + k̄_{t-1}) + ζ_{n,n}*n_{t-1} - ...
```

### Monetary Policy

**Taylor Rule:**
```
R_t = ρ_R*R_{t-1} + (1-ρ_R)*(ψ_1*(π_t - π_t*) + ψ_2*(y_t - y_t^f))
    + ψ_3*((y_t - y_t^f) - (y_{t-1} - y_{t-1}^f)) + r_t^m
```

where:
- `ρ_R`: Interest rate smoothing
- `ψ_1`: Response to inflation gap (typically > 1)
- `ψ_2`: Response to output gap
- `ψ_3`: Response to output gap growth
- `π_t*`: Time-varying inflation target

**Inflation Target:**
```
π_t* = ρ_{π*}*π_{t-1}* + σ_{π*}*ε_t^{π*}
```

### Resource Constraint

```
y_t = g**g_t + c*/y**c_t + i*/y**i_t + r_k**k*/y**u_t
```

## Parameter Calibration

### Fixed Parameters (Calibrated)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `β` | 0.9975 | Quarterly discount factor |
| `δ` | 0.025 | Quarterly depreciation rate |
| `α` | 0.33 | Capital share |
| `γ` | 0.004 | Quarterly trend growth rate |
| `Φ_p` | 1.0 | Fixed costs in production |
| ... | ... | (See model code for complete list) |

### Estimated Parameters (with priors)

**Nominal Rigidities:**
- `ζ_p`: Price stickiness ~ Beta(0.65, 0.10)
- `ζ_w`: Wage stickiness ~ Beta(0.65, 0.10)
- `ι_p`: Price indexation ~ Beta(0.15, 0.10)
- `ι_w`: Wage indexation ~ Beta(0.15, 0.10)

**Monetary Policy:**
- `ψ_1`: Inflation response ~ Normal(1.5, 0.25)
- `ψ_2`: Output response ~ Gamma(0.5, 0.25)
- `ρ_R`: Smoothing ~ Beta(0.8, 0.10)

**Household Preferences:**
- `σ_c`: Risk aversion ~ Normal(1.5, 0.35)
- `h`: Habit persistence ~ Beta(0.7, 0.10)
- `ν_l`: Labor supply elasticity ~ Normal(2.0, 0.75)

**Investment:**
- `S''`: Investment adjustment cost ~ Normal(4.0, 1.5)

**Financial:**
- `ζ_{sp,b}`: Spread elasticity ~ Beta(0.05, 0.01)

See `models/nyfed_model_1002.py` for the complete list with prior specifications.

## Usage

### Loading the Model

```python
from models.nyfed_model_1002 import create_nyfed_model

# Create with default calibration
model = create_nyfed_model()

# Or with custom parameters
params = {
    'psi1': 1.8,      # More aggressive inflation targeting
    'psi2': 0.3,      # Less output stabilization
    # ...
}
model = create_nyfed_model(params)
```

### Solving

```python
from dsge.solvers.linear import solve_linear_model

mats = model.system_matrices()

solution, info = solve_linear_model(
    Gamma0=mats['Gamma0'],
    Gamma1=mats['Gamma1'],
    Psi=mats['Psi'],
    Pi=mats['Pi'],
    n_states=model.spec.n_states
)

print(f"Stable: {solution.is_stable}")
print(f"Max eigenvalue: {np.max(np.abs(solution.eigenvalues)):.4f}")
```

### Estimation

See the [User Guide](../user-guide/using-models.md#working-with-the-nyfed-model) for complete estimation examples.

**Quick start:**

```python
from examples.estimate_nyfed_model import estimate_nyfed_model
import pandas as pd

# Load data
data = pd.read_csv('data/nyfed_data.csv')

# Estimate subset of parameters (faster)
results = estimate_nyfed_model(
    data=data,
    subset_params=True,    # Estimate 10 key parameters
    n_particles=1000,
    n_phi=100
)
```

### Forecasting

```python
from dsge.forecasting import forecast_observables, compute_forecast_bands

# Get measurement system
Z, D = model.measurement_equation()

# Initial state (from Kalman smoother)
x_T = filtered_states[-1]

# 20-quarter forecast
forecast_result = forecast_observables(
    T=solution.T,
    R=solution.R,
    C=solution.C,
    Z=Z,
    D=D,
    x_T=x_T,
    horizon=20,
    n_paths=1000
)

# Uncertainty bands
bands = compute_forecast_bands(forecast_result.paths)
lower_90, upper_90 = bands[0.90]
```

## Data Requirements

### Required FRED Series

The model requires the following data series for estimation:

1. **GDP growth**: GDPC1 (Real GDP)
2. **GDI growth**: GDI (Gross Domestic Income)
3. **Consumption growth**: PCECC96 (Real Personal Consumption)
4. **Investment growth**: GPDIC1 (Real Private Investment)
5. **Wage growth**: COMPRNFB (Compensation) + GDPDEF (Deflator)
6. **Hours**: HOANBS (Hours, Nonfarm Business)
7. **Core PCE inflation**: PCEPILFE (Core PCE Price Index)
8. **GDP deflator**: GDPDEF (GDP Deflator)
9. **Federal Funds Rate**: FEDFUNDS (Effective Federal Funds Rate)
10. **10-year Treasury**: GS10 (10-Year Treasury Rate)
11. **Inflation expectations**: Survey-based (SPF or Michigan)
12. **Credit spread**: BAA - GS10
13. **TFP**: Computed from growth accounting

### Data Download

```python
from data.download_nyfed_data import download_all_nyfed_data

# Download and transform all series
data = download_all_nyfed_data(
    start_date='1990-01-01',
    end_date='2020-12-31',
    save_path='data/nyfed_data.csv'
)
```

**Note**: Requires free FRED API key from https://fred.stlouisfed.org/

## Validation

The model has been validated against:

1. **Solution properties**: Stable eigenvalues, determinacy
2. **Impulse responses**: Economically sensible signs and magnitudes
3. **Simulations**: Bounded, stationary dynamics
4. **Steady state**: Consistent with calibration targets

See `validation/VALIDATION_REPORT.md` for detailed validation results.

**Key validation metrics:**
- Maximum eigenvalue: 1.002 (expected for growth model)
- Monetary policy shock: Contractionary effects on output and inflation ✓
- Technology shock: Expansionary effects ✓
- Simulation: Bounded over 200 periods ✓

## References

### Primary Documentation

- FRBNY DSGE Model Documentation (March 3, 2021)
- [DSGE.jl Repository](https://github.com/FRBNY-DSGE/DSGE.jl)

### Key Papers

1. **Del Negro, M., Giannoni, M. P., & Schorfheide, F. (2015)**. "Inflation in the Great Recession and New Keynesian Models." *American Economic Journal: Macroeconomics*, 7(1), 168-196.

2. **Bernanke, B. S., Gertler, M., & Gilchrist, S. (1999)**. "The Financial Accelerator in a Quantitative Business Cycle Framework." *Handbook of Macroeconomics*, Vol. 1C, 1341-1393.

3. **Smets, F., & Wouters, R. (2007)**. "Shocks and Frictions in US Business Cycles: A Bayesian DSGE Approach." *American Economic Review*, 97(3), 586-606.

4. **Christiano, L. J., Eichenbaum, M., & Evans, C. L. (2005)**. "Nominal Rigidities and the Dynamic Effects of a Shock to Monetary Policy." *Journal of Political Economy*, 113(1), 1-45.

## Implementation Notes

### Julia to Python Translation

The Python implementation follows the Julia specification closely but with some adaptations:

1. **Matrix representation**: Explicit Γ₀, Γ₁, Ψ, Π matrices for the linear solver
2. **Prior distributions**: Using framework's `Prior` class
3. **State organization**: Explicit state vector ordering for clarity
4. **Measurement errors**: Included in state vector for filtering

### Performance Considerations

- **Solution time**: < 1 second at calibrated parameters
- **Estimation time**: 5-15 minutes (subset), 1-2 hours (full) on modern CPU
- **Memory**: ~1 GB for 1000-particle SMC estimation

### Extensions

The model can be extended to include:
- Additional sectors (housing, foreign)
- Alternative monetary policy rules
- Fiscal policy with government debt
- Time-varying volatility
- Anticipated shocks (forward guidance)

## Next Steps

- **[Using Models Guide](../user-guide/using-models.md)**: Detailed estimation and forecasting examples
- **[API Reference](../api.md)**: Complete function documentation
- **[Creating Models Guide](../user-guide/creating-models.md)**: Build your own model
