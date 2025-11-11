# Galí (2010) Model: New Keynesian DSGE with Unemployment

## Overview

This is an implementation of the baseline sticky wage model from:

**Galí, Jordi (2010). "Monetary Policy and Unemployment."**
In: Benjamin M. Friedman and Michael Woodford (eds.),
*Handbook of Monetary Economics*, Volume 3A, Chapter 10, pp. 487-546.
Elsevier.

This is a **properly published DSGE model** with complete equation specifications. It combines New Keynesian features (sticky prices and wages) with labor market search and matching frictions based on the Diamond-Mortensen-Pissarides framework.

## Why This Model?

Unlike the previous "PRISM-inspired" model that was based on incomplete information, the Galí (2010) model:

- ✅ Has **complete published equations** in a peer-reviewed handbook chapter
- ✅ Has **full replication files** available (Dynare implementation by Pfeifer & Bounader)
- ✅ Is **widely cited** and used in academic research (~3,000+ Google Scholar citations)
- ✅ Has **calibration based on U.S. data**
- ✅ Includes the same key features: labor market search frictions, unemployment dynamics, hiring costs

## Model Features

### Core Components

1. **Labor Market Search and Matching**
   - Diamond-Mortensen-Pissarides framework
   - Explicit job-finding and separation rates
   - Hiring costs proportional to labor market tightness
   - Unemployment as an endogenous state variable

2. **Nominal Rigidities**
   - Sticky nominal wages (Calvo pricing with θ_w = 0.75)
   - Sticky nominal prices (Calvo pricing with θ_p = 0.75)
   - Wage Phillips Curve
   - Price Phillips Curve

3. **Household Sector**
   - Consumption-saving decision (Euler equation)
   - Labor force participation choice
   - Utility from consumption and disutility from labor effort

4. **Firm Sector**
   - Production function: Y = A * N^(1-α)
   - Optimal hiring decision
   - Price-setting under Calvo framework

5. **Monetary Policy**
   - Taylor rule: i = φ_π * π + φ_y * y_gap
   - Responds to inflation and output gap

### State Variables (29 total)

**Core Endogenous (21):**
- `y_gap`: Output gap
- `chat`: Consumption (log deviation)
- `rhat`: Real interest rate
- `ihat`: Nominal interest rate
- `nhat`: Employment
- `lhat`: Labor effort
- `fhat`: Labor force
- `uhat`: Unemployment
- `uhat_0`: Unemployment at beginning of period
- `urhat`: Unemployment rate
- `xhat`: Job-finding rate
- `ghat`: Hiring costs
- `hhat`: New hiring
- `mu_hat`: Price markup
- `hatw_real`: Real wage
- `bhat`: Composite auxiliary variable
- `hatw_tar`: Target wage (Nash bargaining)
- `a`: Technology shock process
- `pi_w`: Wage inflation
- `pi_p`: Price inflation
- `nu`: Monetary policy shock

**Lags (8):** For computing growth rates in observables

### Exogenous Shocks (2)

1. **Technology shock** (`eps_a`): AR(1) with persistence ρ_a = 0.9
2. **Monetary policy shock** (`eps_nu`): AR(1) with persistence ρ_ν = 0.5

### Observable Variables (6)

1. **Output gap** (`y_gap`)
2. **Consumption growth** (Δc)
3. **Employment growth** (Δn)
4. **Unemployment rate** (u)
5. **Wage inflation** (π_w)
6. **Price inflation** (π_p)

## Calibration

The model is calibrated to match U.S. quarterly data (from p. 515-516):

| Parameter | Value | Description |
|-----------|-------|-------------|
| N | 0.59 | Employment rate (steady state) |
| U | 0.03 | Unemployment rate (steady state) |
| x | 0.7 | Job-finding rate (quarterly) |
| α | 1/3 | Capital share (labor exponent) |
| β | 0.99 | Discount factor (~4% annual rate) |
| φ | 5.0 | Frisch elasticity of labor supply |
| θ_w | 0.75 | Wage stickiness (avg duration 4 quarters) |
| θ_p | 0.75 | Price stickiness (avg duration 4 quarters) |
| γ | 1.0 | Hiring cost elasticity |
| ξ | 0.05 | Bargaining power of workers |
| φ_π | 1.5 | Taylor rule inflation coefficient |
| φ_y | 0.125 | Taylor rule output coefficient |
| ρ_a | 0.9 | Technology shock persistence |
| ρ_ν | 0.5 | Monetary shock persistence |

**Calibration Targets:**
- Hiring costs = 4.5% of quarterly wage
- Separation rate δ computed from steady state consistency
- Composite parameters (Θ, Υ, Φ, Ξ) computed to satisfy steady state

## Key Equilibrium Conditions

The model has **20 equilibrium conditions** (see model file for complete log-linearized system):

1. **Goods Market Clearing**: Output = Consumption + Hiring Costs
2. **Production Function**: Technology and employment determine output
3. **Employment Dynamics**: New hiring offsets separations
4. **Hiring Cost Function**: Costs increase with job-finding rate
5. **Job-Finding Rate**: Hiring relative to unemployment
6. **Labor Effort**: Combination of employment and unemployment
7. **Labor Force**: Sum of employment and unemployment
8. **Unemployment Evolution**: Separations minus new hires
9. **Unemployment Rate**: Unemployment relative to labor force
10. **Euler Equation**: Intertemporal consumption smoothing
11. **Fisher Equation**: Real rate = nominal rate - expected inflation
12. **Price Phillips Curve**: Sticky price adjustment
13. **Optimal Hiring**: Marginal product = wage + hiring costs
14. **Hiring Cost Present Value**: Forward-looking hiring decision
15. **Participation Condition**: Labor force participation optimality
16. **Real Wage Evolution**: Nominal wage growth - inflation
17. **Wage Phillips Curve**: Sticky wage adjustment
18. **Target Wage**: Nash bargaining outcome
19. **Taylor Rule**: Monetary policy response function
20. **Shock Processes**: AR(1) for technology and policy shocks

## Usage Example

```python
from models.gali_2010_unemployment import create_gali_2010_model
from src.dsge.solvers.linear import solve_linear_model

# Create model instance
model = create_gali_2010_model()

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
print(f"Stable: {solution.is_stable}")
print(f"Max eigenvalue: {np.max(np.abs(np.linalg.eigvals(solution.T))):.4f}")

# Simulate the model
import numpy as np

T = 200  # periods
states = np.zeros((T, model.spec.n_states))
shocks = np.random.randn(T, model.spec.n_shocks) * 0.01

for t in range(1, T):
    states[t] = solution.C + solution.T @ states[t-1] + solution.R @ shocks[t]

# Extract unemployment rate
idx_ur = model.spec.state_names.index("urhat")
unemployment_rate = states[:, idx_ur]
```

## Data Sources

For estimation, map observables to FRED series:

| Observable | FRED Series | Transformation |
|------------|-------------|----------------|
| Output gap | GDPC1 | HP-filtered or estimated |
| Consumption growth | PCECC96 | Log difference |
| Employment growth | CE16OV | Log difference |
| Unemployment rate | UNRATE | Level |
| Wage inflation | COMPRNFB | Log difference |
| Price inflation | PCECTPI | Log difference |

## Implementation Notes

This implementation is based on the Dynare replication by Lahcen Bounader and Johannes Pfeifer, which:

1. **Corrects calibration inconsistencies** in the original paper (see mod file notes)
2. **Uses full unemployment rate definition** rather than approximation
3. **Computes composite parameters** to satisfy calibration targets
4. **Handles steady state** with proper normalization

## Replication

This model can replicate Figures 2A, 2B, 3A, and 3B from Galí (2010) showing impulse responses to:
- Monetary policy shocks
- Technology shocks

The model produces the expected qualitative responses:
- **Monetary contraction**: ↑ rates, ↓ output, ↓ inflation, ↑ unemployment
- **Positive technology**: ↑ output, ↓ inflation (for strong enough response)

## References

### Primary Source

Galí, Jordi (2010). "Monetary Policy and Unemployment."
In: Benjamin M. Friedman and Michael Woodford (eds.),
*Handbook of Monetary Economics*, Volume 3A, Chapter 10, pp. 487-546. Elsevier.

### Theoretical Foundations

Diamond, Peter A. (1982). "Aggregate Demand Management in Search Equilibrium."
*Journal of Political Economy*, 90(5), 881-894.

Mortensen, Dale T., and Christopher A. Pissarides (1994).
"Job Creation and Job Destruction in the Theory of Unemployment."
*Review of Economic Studies*, 61(3), 397-415.

Pissarides, Christopher A. (2000). *Equilibrium Unemployment Theory* (2nd ed.).
MIT Press.

### Replication Code

Bounader, Lahcen, and Johannes Pfeifer (2016).
Dynare implementation of Galí (2010).
https://github.com/JohannesPfeifer/DSGE_mod/tree/master/Gali_2010

## Extensions

Possible extensions to this baseline model:

1. **Capital accumulation** - Add investment and capital stock
2. **Multiple sectors** - Differentiate tradable/non-tradable
3. **Financial frictions** - Credit constraints, financial accelerator
4. **Staggered wage bargaining** - Follow Gertler, Sala & Trigari (2008)
5. **Additional shocks** - Preference, investment, risk premium shocks
6. **OccBin** - Occasionally binding zero lower bound
7. **Open economy** - Trade and exchange rates

## License

The model implementation follows the published equations from Galí (2010).
The Dynare replication code is GPL v3 licensed (Bounader & Pfeifer, 2016).
See repository LICENSE file for this implementation.
