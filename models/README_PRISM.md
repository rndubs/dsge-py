# PRISM-Inspired Model: Medium-Scale DSGE with Labor Market Search Frictions

## Overview

This model implements a medium-scale New Keynesian DSGE model with labor market search frictions, inspired by the Philadelphia Federal Reserve's PRISM-II model. Since the exact PRISM-II specification is proprietary to the Philadelphia Fed, this implementation is based on similar models in the academic literature that incorporate the same key features.

## Important Note

**This is NOT the official PRISM-II model.** The exact specification of PRISM-II is proprietary to the Federal Reserve Bank of Philadelphia's Research Department. This implementation represents a similar class of models based on publicly available academic research that shares the key features of PRISM-II: labor market search frictions, unemployment dynamics, and nominal/real rigidities.

## Model Features

### Core Components

1. **Labor Market Search and Matching**
   - Mortensen-Pissarides search framework
   - Explicit unemployment dynamics
   - Labor market tightness (vacancies/unemployment ratio)
   - Hiring costs and job creation conditions

2. **New Keynesian Features**
   - Sticky prices (Calvo pricing)
   - Consumption Euler equation
   - Taylor rule monetary policy
   - Phillips curve

3. **Wage Determination**
   - Nash bargaining between workers and firms
   - Optional wage adjustment costs (sticky wages)

4. **Stochastic Shocks**
   - Technology shocks
   - Monetary policy shocks
   - Labor market shocks (separation rate)

## Model Structure

### State Variables (24 total)

#### Core Economic Variables (9):
- `c`: Consumption (log deviation from steady state)
- `y`: Output
- `pi`: Inflation rate
- `r`: Nominal interest rate
- `w`: Real wage
- `n`: Employment
- `u`: Unemployment rate
- `theta`: Labor market tightness (v/u ratio)
- `x`: Hiring rate

#### Lags (9):
Used for computing growth rates in observables

#### Shock Processes (6):
- Current and lagged values of: technology, monetary policy, labor market shocks

### Exogenous Shocks (3)

1. **Technology shock** (`eps_a`): Productivity innovations
2. **Monetary policy shock** (`eps_m`): Unexpected policy rate changes
3. **Labor market shock** (`eps_s`): Changes in separation rate

### Observable Variables (7)

1. **Output growth**: Quarter-over-quarter log difference
2. **Consumption growth**: Quarter-over-quarter log difference
3. **Inflation**: Annualized rate
4. **Nominal interest rate**: Annualized
5. **Real wage growth**: Quarter-over-quarter log difference
6. **Employment growth**: Quarter-over-quarter log difference
7. **Unemployment rate**: Level

## Key Parameters

### Household Preferences
- `beta = 0.99`: Discount factor (implies ~4% annual rate)
- `sigma = 1.0`: Inverse of intertemporal elasticity of substitution

### Price Setting
- `theta_p = 0.75`: Calvo parameter (avg price duration = 4 quarters)
- `epsilon_p = 6.0`: Elasticity of substitution between goods

### Labor Market
- `alpha_m = 0.5`: Matching function elasticity
- `rho_u = 0.05`: Quarterly exogenous separation rate
- `kappa_v = 0.5`: Vacancy posting cost
- `gamma = 0.5`: Worker bargaining power
- `xi_w = 0.5`: Wage adjustment cost

### Monetary Policy (Taylor Rule)
- `phi_pi = 1.5`: Response to inflation (satisfies Taylor principle)
- `phi_y = 0.5`: Response to output gap
- `rho_r = 0.75`: Interest rate smoothing

### Shock Processes
- `rho_a = 0.9`: Technology shock persistence
- `rho_m = 0.5`: Monetary shock persistence
- `rho_s = 0.8`: Labor market shock persistence
- `sigma_a = 0.01`: Technology shock std dev
- `sigma_m = 0.0025`: Monetary shock std dev
- `sigma_s = 0.01`: Labor market shock std dev

## Theoretical Foundations

This model draws from the following academic literature:

### Primary Inspiration

**Philadelphia Federal Reserve Bank (2020).** "Philadelphia Research Intertemporal Stochastic Model-II (PRISM-II)." Technical Appendix.
- Available at: https://www.philadelphiafed.org/surveys-and-data/macroeconomic-data/prism-ii
- Medium-scale DSGE model (~30 equations) with labor market search frictions and unemployment
- Maintained by Philadelphia Fed's Real-Time Data Research Center
- Contact: Keith Sill, Associate Director of Research

### Theoretical Framework

**Blanchard, O. J., & Galí, J. (2010).** "Labor Markets and Monetary Policy: A New Keynesian Model with Unemployment." *American Economic Journal: Macroeconomics*, 2(2), 1-30.
- Combines New Keynesian nominal rigidities with Diamond-Mortensen-Pissarides labor market frictions
- Derives utility-based model with unemployment
- Analyzes monetary policy trade-offs with unemployment

**Gertler, M., Sala, L., & Trigari, A. (2008).** "An Estimated Monetary DSGE Model with Unemployment and Staggered Nominal Wage Bargaining." *Journal of Money, Credit and Banking*, 40(8), 1713-1764.
- Medium-scale estimated model with search frictions
- Staggered Nash wage bargaining
- Bayesian estimation on U.S. data

**Christoffel, K., Kuester, K., & Linzert, T. (2009).** "The Role of Labor Markets for Euro Area Monetary Policy." *European Economic Review*, 53(8), 908-936.
- Labor market frictions in policy analysis
- Quantitative assessment of unemployment dynamics

### Search and Matching Framework

**Diamond, P. A. (1982).** "Aggregate Demand Management in Search Equilibrium." *Journal of Political Economy*, 90(5), 881-894.

**Mortensen, D. T., & Pissarides, C. A. (1994).** "Job Creation and Job Destruction in the Theory of Unemployment." *Review of Economic Studies*, 61(3), 397-415.

**Pissarides, C. A. (2000).** "Equilibrium Unemployment Theory" (2nd ed.). MIT Press.

## Implementation Details

### Simplifications

This implementation follows the simpler Blanchard-Galí (2010) framework rather than the more complex Gertler-Sala-Trigari (2008) staggered wage bargaining, while maintaining the key features:

1. **Search and matching** via Mortensen-Pissarides framework
2. **Unemployment** as explicit state variable
3. **Job creation** through vacancy posting
4. **Wage bargaining** between workers and firms
5. **Nominal rigidities** via Calvo pricing

### Potential Extensions

The model can be extended to more closely match PRISM-II by adding:

1. **Staggered wage bargaining** (Gertler-Sala-Trigari 2008)
2. **Capital accumulation** and investment decisions
3. **Financial frictions** (credit spreads, financial accelerator)
4. **Additional sectors** (government, trade)
5. **More detailed shock processes** (preference shocks, investment shocks, etc.)
6. **Richer wage dynamics** (real wage rigidities, wage indexation)

## Usage Example

```python
from models.prism_inspired_model import create_prism_inspired_model
from src.dsge.solvers.linear import solve_linear_model

# Create model instance
model = create_prism_inspired_model()

# Get system matrices
mats = model.system_matrices()

# Solve the model
solution, info = solve_linear_model(
    Gamma0=mats["Gamma0"],
    Gamma1=mats["Gamma1"],
    Psi=mats["Psi"],
    Pi=mats["Pi"],
    n_states=model.spec.n_states
)

# Check solution
print(f"Model is {'stable' if solution.is_stable else 'unstable'}")

# Simulate the model
import numpy as np

T = 200  # periods
n_states = model.spec.n_states
n_shocks = model.spec.n_shocks

states = np.zeros((T, n_states))
shocks = np.random.randn(T, n_shocks) * 0.001

for t in range(1, T):
    states[t] = solution.C + solution.T @ states[t-1] + solution.R @ shocks[t]

# Extract key variables
consumption = states[:, 0]
output = states[:, 1]
unemployment = states[:, 6]
```

## Data Sources

For estimation, the model's observables can be mapped to standard U.S. macroeconomic time series:

| Observable | FRED Series | Transformation |
|------------|-------------|----------------|
| Output growth | GDPC1 | Log difference × 100 |
| Consumption growth | PCECC96 | Log difference × 100 |
| Inflation | PCECTPI | Log difference × 400 (annualized) |
| Interest rate | FEDFUNDS | Level / 4 (quarterly) |
| Wage growth | COMPRNFB | Log difference × 100 |
| Employment growth | CE16OV or PAYEMS | Log difference × 100 |
| Unemployment rate | UNRATE | Level |

See `data/prism_fred_mapping.py` for detailed data loading utilities.

## Estimation

The model can be estimated using Bayesian methods (SMC or MCMC) on U.S. quarterly data. Typical estimation period: 1984:Q1 - present (Great Moderation onwards).

For estimation with this framework:

```python
from src.dsge.estimation.smc import SMCSampler

# Define priors for parameters
# Run SMC estimation
# Analyze posterior distributions
```

## References

See the module docstring in `prism_inspired_model.py` for complete references.

## Contact

For questions about the official PRISM-II model, contact:
- Keith Sill, Federal Reserve Bank of Philadelphia
- Email: Via Philadelphia Fed Research Department
- Website: https://www.philadelphiafed.org/surveys-and-data/macroeconomic-data/prism-ii

For questions about this implementation, refer to the main repository documentation.

## Disclaimer

This implementation is for research and educational purposes. It is not the official PRISM-II model and has not been validated by the Philadelphia Federal Reserve. Results from this model should not be used for policy decisions.

## License

See main repository LICENSE file.
