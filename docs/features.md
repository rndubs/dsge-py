# Features Overview

DSGE-PY provides a complete toolkit for DSGE modeling. This page summarizes the key features and links to detailed API documentation.

## Model Specification

### Abstract Base Class

The {py:class}`dsge.models.base.DSGEModel` class provides a clean interface for defining DSGE models.

**Key Components:**
- {py:class}`~dsge.models.base.ModelSpecification`: Defines dimensions (states, shocks, observables)
- {py:class}`~dsge.models.parameters.Parameter`: Individual parameter with bounds and priors
- {py:class}`~dsge.models.parameters.ParameterSet`: Collection of parameters with value management

**Required Methods:**
- `system_matrices()`: Return Γ₀, Γ₁, Ψ, Π matrices for the linear solver
- `measurement_equation()`: Return Z matrix and D vector linking states to observables
- `shock_covariance()`: Return Q matrix for structural shock covariances

### Prior Distributions

The {py:class}`dsge.models.parameters.Prior` class supports common Bayesian priors:

| Distribution | Parameters | Use Case |
|--------------|-----------|----------|
| **Normal** | `mean`, `std` | Unbounded parameters |
| **Beta** | `alpha`, `beta` | Parameters in (0,1) like persistence |
| **Gamma** | `shape`, `rate` | Positive parameters like standard deviations |
| **Inverse Gamma** | `shape`, `scale` | Variance parameters |
| **Uniform** | `lower`, `upper` | Bounded parameters with flat prior |

**Example:**
```python
from dsge.models.parameters import Prior, Parameter

# AR(1) persistence with Beta prior
rho = Parameter(
    name='rho',
    value=0.9,
    fixed=False,
    prior=Prior('beta', {'alpha': 18, 'beta': 2})
)
```

See {py:class}`dsge.models.parameters.Prior` for full API.

## Solution Methods

### Linear Solver

The {py:func}`dsge.solvers.linear.solve_linear_model` function implements the Blanchard-Kahn solution method using generalized Schur (QZ) decomposition.

**Algorithm:**
1. Form canonical system: Γ₀ x_t = Γ₁ x_{t-1} + Ψ ε_t + Π η_t
2. Compute QZ decomposition
3. Reorder eigenvalues (stable vs. unstable)
4. Check Blanchard-Kahn conditions (determinacy)
5. Compute policy functions

**Returns:**
- {py:class}`~dsge.solvers.linear.LinearSolution`: State-space representation
  - `T`: Transition matrix (n_states × n_states)
  - `R`: Shock impact matrix (n_states × n_shocks)
  - `C`: Constant term (n_states)
  - `is_stable`: Boolean indicating stability
  - `eigenvalues`: All eigenvalues for diagnostics

**Features:**
- Handles pure state models (no expectational errors)
- Provides detailed eigenvalue diagnostics
- Validates determinacy conditions
- Numerically stable implementation

See {py:func}`dsge.solvers.linear.solve_linear_model` for full API.

### OccBin Solver

The {py:class}`dsge.solvers.occbin.OccBinSolver` implements the Guerrieri-Iacoviello (2015) algorithm for occasionally binding constraints.

**Use Cases:**
- Zero Lower Bound on nominal interest rates
- Occasionally binding credit constraints
- Capacity constraints
- Any model with regime-dependent dynamics

**Algorithm:**
1. Solve two regimes (constraint slack M1, constraint binding M2)
2. Guess regime sequence
3. Simulate under piecewise-linear policy functions
4. Check if constraint conditions are satisfied
5. Update regime guess and iterate until convergence

**Components:**
- {py:class}`~dsge.solvers.occbin.OccBinConstraint`: Defines the constraint
- {py:class}`~dsge.solvers.occbin.OccBinSolution`: Solution with regime sequence
- {py:func}`~dsge.solvers.occbin.create_zlb_constraint`: Helper for ZLB

**Example:**
```python
from dsge.solvers.occbin import OccBinSolver, create_zlb_constraint

# Define constraint: nominal rate >= 0
zlb_constraint = create_zlb_constraint(
    variable_index=2,  # Interest rate position in state vector
    bound=0.0
)

# Create solver with two regime solutions
solver = OccBinSolver(solution_M1, solution_M2, zlb_constraint)

# Solve for specific initial state and shocks
result = solver.solve(initial_state, shocks, T=50)

print(f"Converged: {result.converged}")
print(f"Periods at ZLB: {sum(result.regime_sequence == 1)}")
```

See {py:class}`dsge.solvers.occbin.OccBinSolver` for full API.

## State-Space Filtering

### Kalman Filter

The {py:func}`dsge.filters.kalman.kalman_filter` function implements the standard Kalman filter for likelihood evaluation and state inference.

**Measurement Equation:**
```
y_t = Z x_t + D + v_t,  v_t ~ N(0, H)
```

**State Equation:**
```
x_t = T x_t-1 + R ε_t,  ε_t ~ N(0, Q)
```

**Returns:**
- {py:class}`~dsge.filters.kalman.KalmanResults`:
  - `log_likelihood`: Log likelihood for parameter estimation
  - `filtered_states`: E[x_t | y_1:t]
  - `predicted_states`: E[x_t | y_1:t-1]
  - `filtered_covariances`: Var[x_t | y_1:t]

**Features:**
- Handles missing data automatically (NaN values)
- Numerically stable recursions
- Steady-state initialization via discrete Lyapunov equation

**Kalman Smoother:**

The {py:func}`dsge.filters.kalman.kalman_smoother` function implements the Rauch-Tung-Striebel (RTS) backward smoother.

**Returns:**
- Smoothed states: E[x_t | y_1:T] for all t
- Smoothed covariances: Var[x_t | y_1:T]

See {py:func}`dsge.filters.kalman.kalman_filter` and {py:func}`dsge.filters.kalman.kalman_smoother` for full API.

### OccBin Filtering

Two approaches for filtering models with occasionally binding constraints:

#### 1. Perfect Foresight Filter

{py:func}`dsge.filters.occbin_filter.occbin_filter` extends the Kalman filter to handle regime switching with perfect foresight.

**Algorithm:**
1. For each time period, infer the regime sequence
2. Apply regime-appropriate solution matrices
3. Update Kalman filter recursions
4. Iterate until regime sequence converges

**Use When:**
- Regimes are observable or highly persistent
- Computational speed is critical
- Perfect foresight assumption is reasonable

#### 2. Particle Filter

{py:class}`dsge.filters.occbin_filter.OccBinParticleFilter` tracks regime probabilities explicitly using particle filtering.

**Algorithm:**
1. Propagate particles through regime-switching dynamics
2. Weight particles by observation likelihood
3. Resample when effective sample size drops
4. Track regime probabilities across particles

**Use When:**
- Regime uncertainty is important
- Need probabilistic regime inference
- Multiple regime switches possible

See {py:mod}`dsge.filters.occbin_filter` for full API.

## Bayesian Estimation

### Sequential Monte Carlo (SMC)

The {py:class}`dsge.estimation.smc.SMCSampler` implements the Herbst-Schorfheide algorithm for Bayesian parameter estimation.

**Algorithm:**
1. Initialize particles from prior distribution
2. Gradually increase data weight via tempering: p(θ)^(1-φ) p(y|θ)^φ
3. At each stage:
   - Correct: Reweight particles by likelihood
   - Resample if ESS too low
   - Mutate: Metropolis-Hastings steps

**Advantages over MCMC:**
- Parallelizable (across particles)
- Adaptive tempering avoids getting stuck
- Computes marginal likelihood (for model comparison)
- Works well with multimodal posteriors

**Configuration:**
```python
from dsge.estimation import estimate_dsge

results = estimate_dsge(
    model=my_model,
    data=observed_data,
    n_particles=1000,      # More particles = better accuracy
    n_phi=100,             # More stages = slower but more stable
    n_mh_steps=1,          # MH steps per stage
    verbose=True
)
```

**Returns:**
- {py:class}`~dsge.estimation.smc.SMCResults`:
  - `particles`: Posterior particle cloud (n_particles × n_params)
  - `weights`: Normalized particle weights
  - `log_evidence`: Log marginal likelihood
  - `ess_history`: Effective sample size over stages
  - `acceptance_rates`: MH acceptance rates

**Diagnostics:**
- ESS should remain above 50% of n_particles
- Acceptance rates typically 20-40%
- Log evidence used for model comparison (higher is better)

See {py:class}`dsge.estimation.smc.SMCSampler` for full API.

### OccBin Estimation

{py:func}`dsge.estimation.occbin_estimation.estimate_occbin` integrates OccBin filtering with SMC estimation.

**Features:**
- Regime-aware likelihood evaluation
- Works with both perfect foresight and particle filter approaches
- Estimates parameters governing regime-switching behavior

**Example:**
```python
from dsge.estimation.occbin_estimation import estimate_occbin

results = estimate_occbin(
    model_M1=normal_model,
    model_M2=zlb_model,
    constraint=zlb_constraint,
    data=observed_data,
    n_particles=1000,
    use_particle_filter=True  # For regime uncertainty
)
```

See {py:mod}`dsge.estimation.occbin_estimation` for full API.

## Forecasting

The {py:mod}`dsge.forecasting.forecast` module provides multiple forecasting approaches.

### Unconditional Forecasts

{py:func}`~dsge.forecasting.forecast.forecast_observables` generates forecasts from a given initial state.

**Features:**
- Multiple simulation paths for uncertainty quantification
- Confidence bands (68%, 90%, 95%)
- Optional conditional forecasts (constrain future observables)

**Example:**
```python
from dsge.forecasting import forecast_observables, compute_forecast_bands

result = forecast_observables(
    T=solution.T,
    R=solution.R,
    C=solution.C,
    Z=Z,
    D=D,
    x_T=initial_state,
    horizon=20,       # 5 years ahead (quarterly)
    n_paths=1000,
    seed=42
)

# Compute confidence bands
bands = compute_forecast_bands(result.paths)
lower_90, upper_90 = bands[0.90]
```

### Conditional Forecasts

{py:func}`~dsge.forecasting.forecast.conditional_forecast` allows constraining specific observables.

**Use Case:** "What is the path of inflation if GDP grows at 2% for the next 4 quarters?"

**Example:**
```python
conditions = {
    0: {0: 2.0},  # GDP growth = 2% in Q1
    1: {0: 2.0},  # GDP growth = 2% in Q2
    # ... etc
}

conditional_result = conditional_forecast(
    T=solution.T, R=solution.R, C=solution.C, Z=Z, D=D,
    x_T=initial_state,
    horizon=20,
    conditions=conditions
)
```

### Posterior Forecasts

{py:func}`~dsge.forecasting.forecast.forecast_from_posterior` incorporates parameter uncertainty from estimation.

**Features:**
- Draws parameter values from posterior distribution
- Generates forecasts for each draw
- Uncertainty bands reflect both parameter and shock uncertainty

**Example:**
```python
forecast_result = forecast_from_posterior(
    posterior_samples=posterior_particles,
    posterior_weights=particle_weights,
    model=my_model,
    x_T=initial_state,
    horizon=20,
    n_forecast_paths=100,
    n_posterior_draws=100
)
```

See {py:mod}`dsge.forecasting.forecast` for full API.

## Data Management

### FRED Data Loader

The {py:mod}`dsge.data.fred_loader` module provides utilities for downloading and transforming macroeconomic data from FRED (Federal Reserve Economic Data).

**Features:**
- Download time series by FRED code
- Automatic frequency conversion (monthly → quarterly)
- Common transformations:
  - Growth rates (annualized)
  - Inflation rates
  - Real deflation
  - Log differences

**Configuration:**

Set up your FRED API key (free from https://fred.stlouisfed.org/):

```bash
# In .env file
FRED_API_KEY=your_key_here
```

**Example:**
```python
from dsge.data.fred_loader import download_fred_series, compute_growth_rate

# Download GDP
gdp = download_fred_series(
    series_id='GDPC1',
    start_date='1990-01-01',
    end_date='2020-12-31'
)

# Compute annualized quarterly growth rate
gdp_growth = compute_growth_rate(gdp, annualize=True)
```

See {py:mod}`dsge.data.fred_loader` for full API.

## Implemented Models

### Simple New Keynesian Model

A 3-equation pedagogical model for learning:
- Output gap (IS curve)
- Inflation (Phillips curve)
- Interest rate (Taylor rule)

**Location:** `models.simple_nk_model.SimpleNKModel`

### NYFed DSGE Model 1002

Medium-scale model used by the Federal Reserve Bank of New York:
- 67 parameters
- 13 observables
- Financial frictions (Bernanke-Gertler-Gilchrist)
- Nominal and real rigidities
- 9 structural shocks

**Location:** `models.nyfed_model_1002.NYFedModel1002`

See [NYFed Model Documentation](models/nyfed.md) for details.

### Smets-Wouters (2007)

Canonical medium-scale DSGE model:
- 41 parameters
- 7 observables
- Sticky prices and wages
- Capital accumulation

**Location:** `models.smets_wouters_2007.SmetsWouters2007`

## Performance Characteristics

### Linear Models

- AR(1) solution: < 0.01s
- Medium-scale model (40 states): < 0.1s
- Kalman filter (100 periods): < 0.01s

### Estimation

- SMC (1000 particles, 100 stages, 10 parameters): 5-10 minutes
- SMC (2000 particles, 200 stages, 67 parameters): 1-2 hours

*Timings on modern CPU (Intel i7 or equivalent)*

### Future Optimizations

Phase 5 of the project plan includes:
- Parallel particle evaluation
- JAX/Numba acceleration
- Boehl's fast OccBin method (~1500x speedup)

## Next Steps

- **[User Guide - Using Models](user-guide/using-models.md)**: Learn how to work with existing models
- **[User Guide - Creating Models](user-guide/creating-models.md)**: Build your own DSGE model
- **[API Reference](api.md)**: Detailed API documentation
