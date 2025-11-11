# Cleveland Fed CLEMENTINE Model

The CLEMENTINE (CLeveland Equilibrium ModEl iNcluding Trend INformation and the Effective lower bound) model is a medium-scale DSGE model developed at the Federal Reserve Bank of Cleveland for macroeconomic forecasting and policy analysis.

## Model Overview

### Characteristics

- **Scale**: Medium-scale New Keynesian DSGE with labor market frictions
- **Sectors**: Single sector with households, firms, labor market, financial intermediaries, and government
- **Agent Types**: Households, searching workers/unemployed, firms, financial intermediaries, central bank
- **Primary Use**: Medium-term forecasting, policy counterfactuals, labor market analysis, ZLB scenarios

### Key Features

**1. Labor Market Search and Matching**
- Mortensen-Pissarides search and matching framework
- Endogenous unemployment and vacancies
- Job finding and separation rates
- Nash bargaining for wage determination
- Labor market tightness dynamics

**2. Zero Lower Bound (ZLB)**
- Regime-switching framework for ZLB constraint
- Perfect foresight or stochastic regime transitions
- Occasionally binding constraint on nominal interest rates
- Suitable for analysis of unconventional monetary policy

**3. Stochastic Trend Growth**
- Non-stationary technology shock (trend)
- Stationary technology shock (cycle)
- Balanced growth path with detrended system
- Trend-cycle decomposition capability

**4. Standard New Keynesian Core**
- Calvo price and wage rigidities
- Partial indexation to past inflation
- Monopolistic competition
- Investment adjustment costs
- Habit formation in consumption

**5. Financial Frictions**
- Credit spreads responsive to leverage
- Simplified financial accelerator mechanism
- Tobin's q for capital valuation
- Borrowing constraints

## Model Structure

### Dimensions

- **Parameters**: 41 total (structural + calibrated + shock parameters)
- **State Variables**: 40 total
  - 13 core endogenous variables
  - 8 labor market variables
  - 6 lags
  - 7 structural shocks
  - 6 auxiliary/derived variables
- **Observables**: 10 macroeconomic time series
- **Shocks**: 7 structural shocks

### Core Endogenous Variables

| Variable | Description |
|----------|-------------|
| `y` | Output |
| `c` | Consumption |
| `i` | Investment |
| `k` | Capital stock |
| `pi` | Inflation |
| `R` | Nominal interest rate |
| `w` | Real wage |
| `mc` | Real marginal cost |
| `q_k` | Tobin's q (price of capital) |
| `spread` | Credit spread |
| `g` | Government spending |
| `z_trend` | Trend productivity |
| `z_stat` | Stationary productivity |

### Labor Market Variables

| Variable | Description |
|----------|-------------|
| `n` | Employment |
| `u_rate` | Unemployment rate |
| `v` | Vacancies |
| `theta` | Labor market tightness (v/u) |
| `q_v` | Vacancy filling rate |
| `f_rate` | Job finding rate |
| `s_rate` | Job separation rate |
| `l` | Labor force |

### Structural Shocks

| Shock | Description | Persistence (Prior Mean) |
|-------|-------------|--------------------------|
| `eps_z_trend` | Trend technology | 0.95 |
| `eps_z_stat` | Stationary technology | 0.80 |
| `eps_b` | Preference / risk premium | 0.50 |
| `eps_i` | Investment efficiency / MEI | 0.50 |
| `eps_g` | Government spending | 0.50 |
| `eps_p` | Price markup | 0.50 |
| `eps_w` | Wage markup | 0.50 |

### Observables

The model maps to 10 U.S. macroeconomic data series:

| Observable | FRED Code (Suggested) | Transformation |
|------------|----------------------|----------------|
| GDP growth | GDPC1 | Quarterly log diff × 100 |
| Consumption growth | PCECC96 | Quarterly log diff × 100 |
| Investment growth | GPDIC1 | Quarterly log diff × 100 |
| Employment | PAYEMS or CE16OV | Log deviation from trend |
| Unemployment rate | UNRATE | Level (percent) |
| Wage growth | COMPRNFB / GDPDEF | Real wage, log diff × 100 |
| Inflation | PCEPILFE or CPIAUCSL | Quarterly log diff × 400 |
| Interest rate | FEDFUNDS | Level (annualized %) |
| Credit spread | BAA10Y or similar | Level (basis points) |
| Hours worked | HOANBS | Log deviation from trend |

See `data/fred_series_mapping.py` for detailed data transformations.

## Key Equations

### Household Sector

**Consumption Euler Equation:**
```
c_t = (h*exp(-γ))/(1+h*exp(-γ)) * c_{t-1}
    - (1-h*exp(-γ))/(σ_c*(1+h*exp(-γ))) * (R_t - E[π_{t+1}] + b_t)
    + 1/(1+h*exp(-γ)) * E[c_{t+1}]
```

where:
- `h`: Habit persistence parameter
- `σ_c`: Risk aversion / inverse of IES
- `γ`: Steady-state growth rate
- `b_t`: Preference shock

### Firms

**Production Function:**
```
y_t = α*k_t + (1-α)*n_t + z_t
```

**New Keynesian Phillips Curve:**
```
π_t = β/(1+β*ι_p) * E[π_{t+1}]
    + ι_p/(1+β*ι_p) * π_{t-1}
    + κ_p * mc_t + ε_t^p
```

where `κ_p = ((1-ζ_p)*(1-ζ_p*β))/(ζ_p*(1+β*ι_p))`

### Labor Market

**Matching Function:**
```
m_t = A * u_t^{1-χ} * v_t^χ
```

**Job Finding Rate:**
```
f_t = m_t / u_t = A * θ_t^χ
```

where `θ_t = v_t/u_t` is labor market tightness

**Employment Dynamics:**
```
n_t = (1-ρ_s)*n_{t-1} + f_t*u_{t-1}
```

**Unemployment:**
```
u_t = l_t - n_t
```

**Nash Bargaining Wage:**
```
w_t = ξ*(MPL_t + κ_v*θ_t) + (1-ξ)*b_{unemp}
```

where:
- `ξ`: Worker bargaining power
- `MPL_t`: Marginal product of labor
- `κ_v`: Vacancy posting cost
- `b_{unemp}`: Unemployment benefits / home production

### Investment and Capital

**Investment Equation:**
```
i_t = 1/(1+β) * i_{t-1} + β/(1+β) * E[i_{t+1}]
    + 1/(S''*(1+β)) * q_t^k + ε_t^i
```

**Capital Accumulation:**
```
k_t = (1-δ)*k_{t-1} + δ*i_t
```

### Financial Sector

**Tobin's q:**
```
q_t^k = E[r_{t+1}^k] - (R_t - E[π_{t+1}]) - spread_t
```

**Credit Spread:**
```
spread_t = ζ_{sp}*(q_t^k + k_t - n_t)
```

This captures the financial accelerator: higher leverage (q_k*k relative to net worth n) increases spreads.

### Monetary Policy

**Taylor Rule:**
```
R_t = ρ_R*R_{t-1}
    + (1-ρ_R)*(ψ_π*π_t + ψ_y*ỹ_t)
    + ψ_{Δy}*(y_t - y_{t-1})
```

**With ZLB:**
```
R_t = max{0, R_t^Taylor}
```

Implemented via regime switching (OccBin framework).

### Trend Growth

**Trend Technology:**
```
z_t^{trend} = ρ_{z,trend}*z_{t-1}^{trend} + σ_{z,trend}*ε_t^{z,trend}
```

**Stationary Technology:**
```
z_t^{stat} = ρ_{z,stat}*z_{t-1}^{stat} + σ_{z,stat}*ε_t^{z,stat}
```

Total productivity: `z_t = z_t^{trend} + z_t^{stat}`

## Parameter Calibration

### Fixed Parameters (Calibrated)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `β` | 0.995 | Quarterly discount factor (prior mean) |
| `δ` | 0.025 | Quarterly depreciation rate |
| `α` | 0.33 | Capital share |
| `γ_ss` | 0.5 | Steady-state quarterly growth (%) |
| `π_ss` | 0.5 | Steady-state quarterly inflation (%) |
| `g_y_ss` | 0.2 | Govt spending / GDP ratio |

### Estimated Parameters (with Priors)

**Household:**
- `σ_c`: Risk aversion ~ Normal(1.5, 0.375)
- `h`: Habit persistence ~ Beta(0.7, 0.1)

**Production:**
- `α`: Capital share ~ Normal(0.33, 0.05)
- `Φ`: Fixed cost ~ Normal(1.25, 0.125)

**Rigidities:**
- `ζ_p`: Price stickiness ~ Beta(0.66, 0.1)
- `ι_p`: Price indexation ~ Beta(0.25, 0.15)
- `ζ_w`: Wage stickiness ~ Beta(0.66, 0.1)
- `ι_w`: Wage indexation ~ Beta(0.25, 0.15)

**Labor Market:**
- `χ`: Matching elasticity ~ Beta(0.5, 0.1)
- `κ_v`: Vacancy cost ~ Gamma(0.05, 0.01)
- `ξ`: Bargaining power ~ Beta(0.5, 0.1)
- `ρ_s`: Separation rate ~ Beta(0.1, 0.02)

**Monetary Policy:**
- `ψ_π`: Inflation response ~ Normal(1.5, 0.25)
- `ψ_y`: Output gap response ~ Normal(0.125, 0.05)
- `ρ_R`: Interest smoothing ~ Beta(0.8, 0.1)

See `models/clementine_model.py` for complete parameter specifications with priors.

## Usage

### Loading the Model

```python
from models.clementine_model import create_clementine_model

# Create model with default calibration
model = create_clementine_model()

# Inspect model
print(f"States: {model.spec.n_states}")
print(f"Observables: {model.spec.n_observables}")
print(f"Parameters: {len(model.parameters)}")
```

### Solving the Model

```python
from dsge.solvers.linear import solve_linear_model

# Get system matrices
mats = model.system_matrices()

# Solve linear system
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

### Solving with ZLB (OccBin)

```python
from dsge.solvers.occbin import OccBinSolver, Constraint

# Create two regime models
model_M1 = create_clementine_model()  # Normal regime
model_M2 = create_clementine_model()  # ZLB regime

# Solve both regimes
solution_M1, _ = solve_linear_model(...)  # Normal
solution_M2, _ = solve_linear_model(...)  # ZLB (R=0)

# Define ZLB constraint
zlb_constraint = Constraint(
    variable_index=5,  # R is 6th state (index 5)
    bound=0.0,
    binding_direction="lower"
)

# Create OccBin solver
solver = OccBinSolver(solution_M1, solution_M2, zlb_constraint)

# Simulate large negative shock
initial_state = np.zeros(40)
shocks = np.zeros((50, 7))
shocks[0, 2] = -3.0  # Large negative preference shock

result = solver.solve(initial_state, shocks, T=50)
```

### Estimation

```python
from dsge.estimation import estimate_model
import pandas as pd

# Load data
data = pd.read_csv('data/clementine_data.csv')

# Prepare observables (10 series)
obs_data = data[[
    'gdp_growth', 'cons_growth', 'inv_growth',
    'employment', 'unemp_rate', 'wage_growth',
    'inflation', 'ffr', 'spread', 'hours'
]].values

# Estimate with SMC
results = estimate_model(
    model=model,
    data=obs_data,
    n_particles=1000,
    n_phi=100,
    method='smc'
)

# View posterior estimates
print(results.posterior_mean)
print(results.posterior_std)
```

### Forecasting

```python
from dsge.forecasting import forecast_observables

# Get measurement system
Z, D = model.measurement_equation()

# Initial state from filtered data
x_T = filtered_states[-1]

# Generate 20-quarter forecast
forecast = forecast_observables(
    T=solution.T,
    R=solution.R,
    C=solution.C,
    Z=Z,
    D=D,
    x_T=x_T,
    horizon=20,
    n_paths=1000
)

# Plot with uncertainty bands
import matplotlib.pyplot as plt
mean_forecast = forecast.mean(axis=0)
lower_90 = np.percentile(forecast, 5, axis=0)
upper_90 = np.percentile(forecast, 95, axis=0)

plt.plot(mean_forecast[:, 0], label='GDP Growth Forecast')
plt.fill_between(range(20), lower_90[:, 0], upper_90[:, 0], alpha=0.3)
plt.legend()
plt.show()
```

## Data Requirements

### Required FRED Series

The model requires 10 quarterly U.S. macroeconomic series:

1. **Real GDP**: GDPC1
2. **Real Consumption**: PCECC96
3. **Real Investment**: GPDIC1
4. **Employment**: PAYEMS (monthly → quarterly average) or CE16OV
5. **Unemployment Rate**: UNRATE (monthly → quarterly average)
6. **Compensation/Wages**: COMPRNFB (nominal) deflated by GDPDEF
7. **Core PCE Inflation**: PCEPILFE
8. **Federal Funds Rate**: FEDFUNDS (monthly → quarterly average)
9. **Credit Spread**: BAA10Y or compute as (BAMLC0A0CM - GS10)
10. **Hours Worked**: HOANBS (Index)

### Data Download

```python
from data.download_clementine_data import download_all_clementine_data

# Requires FRED API key
data = download_all_clementine_data(
    start_date='1990-01-01',
    end_date='2023-12-31',
    save_path='data/clementine_data.csv'
)
```

### Sample Period

Recommended: **1990:Q1 - Present**
- Post-Volcker period with stable monetary policy
- Captures Great Recession (2008-2009) with ZLB episode
- Includes COVID-19 pandemic (2020-2021)

## Model Applications

### 1. Medium-Term Forecasting

The CLEMENTINE model is designed for forecasting 1-3 years ahead with explicit modeling of:
- Labor market dynamics (employment, unemployment, participation)
- Trend vs. cycle decomposition
- Financial conditions via credit spreads

### 2. Monetary Policy Analysis

**Taylor Rule vs. Optimal Policy:**
- Compare historical policy with model-implied optimal policy
- Evaluate alternative policy rules
- Assess welfare implications

**ZLB Episodes:**
- Analyze 2008-2015 ZLB period
- Simulate forward guidance scenarios
- Evaluate unconventional policy effectiveness

### 3. Labor Market Counterfactuals

**Unemployment Dynamics:**
- Decompose unemployment into cyclical vs. structural components
- Analyze job finding vs. separation margins
- Evaluate labor market policies (e.g., UI extensions)

**Wage Dynamics:**
- Understand wage rigidity effects
- Analyze wage-price spirals
- Evaluate labor market tightness effects on inflation

### 4. Trend-Cycle Decomposition

**Growth Slowdown:**
- Separate trend growth shocks from cyclical fluctuations
- Analyze productivity trends
- Forecast potential output

### 5. Financial Conditions Analysis

**Credit Spreads:**
- Monitor financial stress indicators
- Link credit conditions to real activity
- Evaluate financial-real feedback loops

## Validation

The model has been validated for:

1. **Solution Properties**: Stable eigenvalues, Blanchard-Kahn conditions satisfied
2. **Impulse Responses**: Economically sensible signs and magnitudes
3. **Simulations**: Bounded, stationary dynamics around balanced growth path
4. **Steady State**: Consistent with calibration targets (u_ss, growth, etc.)

**Key validation metrics:**
- Maximum eigenvalue < 1 (for detrended system)
- Monetary policy shock: Contractionary effects ✓
- Technology shock: Expansionary effects with employment lag ✓
- Labor market dynamics match flows ✓

## References

### Primary Source

**Gelain, P., & Lopez, P. (2023)**. "A DSGE Model Including Trend Information and Regime Switching at the ZLB." *Federal Reserve Bank of Cleveland, Working Paper No. 23-35*.
- https://doi.org/10.26509/frbc-wp-202335
- https://www.clevelandfed.org/publications/working-paper/2023/wp-2335-dsge-model

This paper adopts a practitioner's guide approach, detailing the construction of the model and offering practical guidance on its use as a policy tool.

### Methodological Foundations

1. **Pissarides, C. A. (2000)**. *Equilibrium Unemployment Theory* (2nd ed.). MIT Press.

2. **Galí, J. (2015)**. *Monetary Policy, Inflation, and the Business Cycle* (2nd ed.). Princeton University Press. Chapter 7.

3. **Guerrieri, L., & Iacoviello, M. (2015)**. "OccBin: A toolkit for solving dynamic models with occasionally binding constraints easily." *Journal of Monetary Economics*, 70, 22-38.

4. **Aguiar, M., & Gopinath, G. (2007)**. "Emerging Market Business Cycles: The Cycle Is the Trend." *Journal of Political Economy*, 115(1), 69-102.

5. **Bernanke, B. S., Gertler, M., & Gilchrist, S. (1999)**. "The Financial Accelerator in a Quantitative Business Cycle Framework." *Handbook of Macroeconomics*, Vol. 1C, 1341-1393.

## Implementation Notes

### Design Choices

1. **Simplified Financial Sector**: The model uses a reduced-form credit spread equation rather than explicit bank balance sheets
2. **Labor Force**: Treated as exogenous; could be endogenized with participation decision
3. **Government**: Simple exogenous spending process; no fiscal policy interactions
4. **Wage Determination**: Nash bargaining rather than Calvo wage rigidity in matching sector

### Performance Considerations

- **Solution time**: < 1 second for linear solution
- **Estimation time**: 10-30 minutes (SMC with 1000 particles)
- **Memory**: ~500 MB for typical estimation

### Extensions

Possible model extensions:
- Endogenous labor force participation
- Multiple sectors (goods vs. services, tradable vs. non-tradable)
- Open economy features (trade, exchange rate)
- Richer financial sector with bank capital
- Fiscal policy with government debt
- Heterogeneous agents

## Next Steps

- **[Using Models Guide](../user-guide/using-models.md)**: Estimation and forecasting workflows
- **[API Reference](../api.md)**: Complete function documentation
- **[Creating Models Guide](../user-guide/creating-models.md)**: Build your own model
- **[OccBin Tutorial](../user-guide/occbin.md)**: Handle ZLB constraints

## Contact

For questions about the CLEMENTINE model:
- **Original authors**: Paolo Gelain and Pierlauro Lopez (Federal Reserve Bank of Cleveland)
- **Implementation**: This Python implementation follows the framework's interface
- **Issues**: Report at https://github.com/rndubs/dsge-py/issues
