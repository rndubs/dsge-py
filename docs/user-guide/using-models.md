# Using Existing Models

This guide walks you through using the DSGE models included in the package, from basic simulation to full Bayesian estimation and forecasting.

## Quick Start: Simple AR(1) Example

Let's start with the simplest possible model to understand the complete workflow.

### Step 1: Import and Create Model

```python
import numpy as np
from dsge import DSGEModel, ModelSpecification, Parameter, Prior
from dsge import solve_linear_model, estimate_dsge

# Define a simple AR(1) model: x_t = ρ x_{t-1} + ε_t
class AR1Model(DSGEModel):
    def __init__(self):
        spec = ModelSpecification(
            n_states=1,
            n_controls=0,
            n_shocks=1,
            n_observables=1,
            state_names=['x'],
            shock_names=['eps'],
            observable_names=['y']
        )
        super().__init__(spec)

    def _setup_parameters(self):
        # Persistence parameter with Beta prior
        self.parameters.add(Parameter(
            name='rho',
            value=0.9,
            fixed=False,
            prior=Prior('beta', {'alpha': 18, 'beta': 2})
        ))

        # Shock standard deviation with Inverse Gamma prior
        self.parameters.add(Parameter(
            name='sigma',
            value=0.1,
            fixed=False,
            prior=Prior('invgamma', {'shape': 2.0, 'scale': 0.2})
        ))

    def system_matrices(self, params=None):
        if params is not None:
            self.parameters.set_values(params)

        rho = self.parameters['rho']

        return {
            'Gamma0': np.array([[1.0]]),              # x_t
            'Gamma1': np.array([[rho]]),              # ρ x_{t-1}
            'Psi': np.array([[1.0]]),                 # ε_t
            'Pi': np.array([[1e-10]])                 # Expectational error (none)
        }

    def measurement_equation(self, params=None):
        # y_t = x_t (perfect observation)
        Z = np.array([[1.0]])
        D = np.array([0.0])
        return Z, D

    def shock_covariance(self, params=None):
        if params is not None:
            self.parameters.set_values(params)

        sigma = self.parameters['sigma']
        return np.array([[sigma**2]])

# Create model instance
model = AR1Model()
```

### Step 2: Solve the Model

```python
# Get system matrices at current parameter values
system_mats = model.system_matrices()

# Solve using Blanchard-Kahn method
solution, info = solve_linear_model(
    Gamma0=system_mats['Gamma0'],
    Gamma1=system_mats['Gamma1'],
    Psi=system_mats['Psi'],
    Pi=system_mats['Pi'],
    n_states=1
)

print(f"Solution stable: {solution.is_stable}")
print(f"Transition matrix T: {solution.T}")
print(f"Shock matrix R: {solution.R}")
```

**Output:**
```
Solution stable: True
Transition matrix T: [[0.9]]
Shock matrix R: [[1.0]]
```

### Step 3: Simulate Data

```python
from dsge.solvers.linear import simulate

# Add measurement equation to solution
solution.Z = np.array([[1.0]])
solution.D = np.array([0.0])
solution.Q = model.shock_covariance()

# Simulate 200 periods
states, obs = simulate(solution, n_periods=200, random_seed=42)

print(f"States shape: {states.shape}")    # (200, 1)
print(f"Observables shape: {obs.shape}")  # (200, 1)
```

### Step 4: Estimate Parameters

```python
# Run Sequential Monte Carlo estimation
results = estimate_dsge(
    model=model,
    data=obs,
    n_particles=500,
    n_phi=50,
    n_mh_steps=1,
    verbose=True
)

# Extract posterior estimates
posterior_mean = np.average(
    results.particles,
    weights=results.weights,
    axis=0
)

print(f"\nEstimation Results:")
print(f"Log evidence: {results.log_evidence:.2f}")
print(f"\nParameter estimates (true → posterior mean):")
print(f"  rho: 0.90 → {posterior_mean[0]:.3f}")
print(f"  sigma: 0.10 → {posterior_mean[1]:.3f}")
```

**Output:**
```
Stage 10/50: phi=0.200, ESS=425/500, accept=0.25
Stage 20/50: phi=0.400, ESS=398/500, accept=0.22
...
Stage 50/50: phi=1.000, ESS=412/500, accept=0.18

Estimation Results:
Log evidence: 145.23

Parameter estimates (true → posterior mean):
  rho: 0.90 → 0.897
  sigma: 0.10 → 0.103
```

## Working with the NYFed Model

The NYFed DSGE Model 1002 is a production-ready medium-scale model with 67 parameters and 13 observables.

### Loading the Model

```python
from models.nyfed_model_1002 import create_nyfed_model

# Create model with default calibration
model = create_nyfed_model()

print(f"Model: {model.name}")
print(f"Parameters: {len(model.parameters)}")
print(f"States: {model.spec.n_states}")
print(f"Observables: {model.spec.n_observables}")
```

### Solving the Model

```python
from dsge.solvers.linear import solve_linear_model

# Get system matrices
mats = model.system_matrices()

# Solve
solution, info = solve_linear_model(
    Gamma0=mats['Gamma0'],
    Gamma1=mats['Gamma1'],
    Psi=mats['Psi'],
    Pi=mats['Pi'],
    n_states=model.spec.n_states
)

print(f"Solution stable: {solution.is_stable}")
print(f"Max eigenvalue: {np.max(np.abs(solution.eigenvalues)):.4f}")
```

### Generating Synthetic Data

For testing without real data:

```python
from examples.generate_nyfed_synthetic_data import generate_synthetic_data

# Generate 200 quarters of data
data = generate_synthetic_data(
    T=200,
    shock_std=0.01,
    seed=42,
    save_path='data/nyfed_synthetic_data.csv'
)

print(f"Generated {len(data)} observations")
print(f"Variables: {list(data.columns)}")
```

### Estimation

Due to computational intensity, estimation can focus on a subset of key parameters:

```python
# Define subset of parameters to estimate (10 most important)
subset_params = [
    'psi1', 'psi2', 'rho_R',      # Monetary policy
    'sigma_c', 'h',                # Preferences
    'rho_z', 'rho_b',              # Shock persistence
    'sigma_r_m', 'sigma_z', 'sigma_b'  # Shock volatilities
]

# Run estimation on subset
from examples.estimate_nyfed_model import estimate_nyfed_subset

results = estimate_nyfed_subset(
    data_path='data/nyfed_synthetic_data.csv',
    param_subset=subset_params,
    n_particles=1000,
    n_phi=100,
    verbose=True
)
```

**For full estimation:**

```python
# Estimate all 67 parameters (requires significant computation)
results = estimate_nyfed_model(
    data_path='data/nyfed_data.csv',
    n_particles=2000,
    n_phi=200,
    save_dir='results/nyfed_estimation'
)
```

### Forecasting

#### Unconditional Forecast

```python
from dsge.forecasting import forecast_observables, compute_forecast_bands

# Get measurement equation
Z, D = model.measurement_equation()

# Initial state (e.g., from Kalman filter)
x_T = np.zeros(model.spec.n_states)

# Generate 20-quarter forecast
forecast_result = forecast_observables(
    T=solution.T,
    R=solution.R,
    C=solution.C,
    Z=Z,
    D=D,
    x_T=x_T,
    horizon=20,
    n_paths=1000,
    seed=42
)

# Compute uncertainty bands
bands = compute_forecast_bands(forecast_result.paths)

# Extract 90% bands
lower_90, upper_90 = bands[0.90]

# Plot forecast for GDP growth (first observable)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
horizon = np.arange(1, 21)
plt.plot(horizon, forecast_result.mean[:, 0], 'b-', label='Mean forecast')
plt.fill_between(horizon, lower_90[:, 0], upper_90[:, 0],
                 alpha=0.3, label='90% confidence')
plt.xlabel('Quarters ahead')
plt.ylabel('GDP Growth (%)')
plt.title('GDP Growth Forecast')
plt.legend()
plt.grid(True)
plt.show()
```

#### Conditional Forecast

Impose constraints on future observables:

```python
from dsge.forecasting import conditional_forecast

# Condition: GDP grows at 2% for first 4 quarters
conditions = {
    0: {0: 2.0},  # Quarter 1
    1: {0: 2.0},  # Quarter 2
    2: {0: 2.0},  # Quarter 3
    3: {0: 2.0},  # Quarter 4
}

conditional_result = conditional_forecast(
    T=solution.T,
    R=solution.R,
    C=solution.C,
    Z=Z,
    D=D,
    x_T=x_T,
    horizon=20,
    conditions=conditions,
    n_paths=1000
)

print(f"Conditional GDP growth Q1: {conditional_result.mean[0, 0]:.2f}%")
# Output: ~2.0% (matches constraint)
```

#### Forecast with Parameter Uncertainty

Incorporate estimation uncertainty:

```python
from dsge.forecasting import forecast_from_posterior
import pandas as pd

# Load posterior samples from estimation
posterior_df = pd.read_csv('results/nyfed_estimation/posterior_samples.csv')
posterior_samples = posterior_df.drop('weight', axis=1).values
weights = posterior_df['weight'].values

# Generate forecast
forecast_result = forecast_from_posterior(
    posterior_samples=posterior_samples,
    posterior_weights=weights,
    model=model,
    x_T=x_T,
    horizon=20,
    n_forecast_paths=100,
    n_posterior_draws=100,
    seed=42
)

# Uncertainty bands now reflect both parameter and shock uncertainty
bands = compute_forecast_bands(forecast_result.paths)
lower_90, upper_90 = bands[0.90]
```

## Working with OccBin Models

Models with occasionally binding constraints require the OccBin solver.

### Example: Zero Lower Bound

```python
from dsge.solvers.occbin import OccBinSolver, create_zlb_constraint

# Define NK model in two regimes
class SimpleNKModel:
    # ... (see models/simple_nk_model.py)
    pass

# Regime 1: Normal monetary policy (Taylor rule active)
model_M1 = SimpleNKModel(zlb_binding=False)
solution_M1, _ = solve_linear_model(...)

# Regime 2: ZLB binding (interest rate fixed at 0)
model_M2 = SimpleNKModel(zlb_binding=True)
solution_M2, _ = solve_linear_model(...)

# Define ZLB constraint
zlb_constraint = create_zlb_constraint(
    variable_index=2,  # Interest rate position in state vector
    bound=0.0          # ZLB at 0%
)

# Create OccBin solver
solver = OccBinSolver(solution_M1, solution_M2, zlb_constraint)

# Solve for large negative demand shock
initial_state = np.zeros(model_M1.spec.n_states)
shocks = np.zeros((50, model_M1.spec.n_shocks))
shocks[0, 0] = -3.0  # Large negative demand shock

result = solver.solve(initial_state, shocks, T=50)

print(f"Converged: {result.converged}")
print(f"Periods at ZLB: {np.sum(result.regime_sequence == 1)}")
print(f"Regime sequence: {result.regime_sequence[:10]}")
```

### OccBin Filtering

```python
from dsge.filters.occbin_filter import occbin_filter

# Filter observed data through OccBin model
Z, D = model_M1.measurement_equation()
H = np.eye(model_M1.spec.n_observables) * 0.01  # Measurement error

results = occbin_filter(
    y=observed_data,
    solution_M1=solution_M1,
    solution_M2=solution_M2,
    constraint=zlb_constraint,
    Z=Z,
    D=D,
    H=H
)

print(f"Log likelihood: {results.log_likelihood:.2f}")
print(f"Regime accuracy: {np.mean(results.regime_sequence == true_regimes):.1%}")
```

### OccBin Estimation

```python
from dsge.estimation.occbin_estimation import estimate_occbin

# Estimate parameters of ZLB model
results = estimate_occbin(
    model_M1=model_M1,
    model_M2=model_M2,
    constraint=zlb_constraint,
    data=observed_data,
    n_particles=1000,
    n_phi=100,
    use_particle_filter=True  # Account for regime uncertainty
)

print(f"Log evidence: {results.log_evidence:.2f}")
```

## Downloading Real Data

### FRED Data Setup

Get a free API key from https://fred.stlouisfed.org/docs/api/api_key.html

Create `.env` file:
```bash
FRED_API_KEY=your_key_here
```

### Download Data for NYFed Model

```python
from data.download_nyfed_data import download_all_nyfed_data

# Download all 13 observables
data = download_all_nyfed_data(
    start_date='1990-01-01',
    end_date='2020-12-31',
    save_path='data/nyfed_data.csv'
)

print(f"Downloaded {len(data)} quarters")
print(f"Series: {list(data.columns)}")
```

The script automatically:
- Downloads from FRED
- Converts to quarterly frequency
- Applies transformations (growth rates, inflation)
- Aligns all series
- Saves to CSV

### Manual Data Download

```python
from dsge.data.fred_loader import download_fred_series, compute_growth_rate

# Download GDP
gdp = download_fred_series('GDPC1', '1990-01-01', '2020-12-31')

# Compute annualized quarterly growth rate
gdp_growth = compute_growth_rate(gdp, annualize=True)
```

## Best Practices

### Estimation

1. **Start Small**: Test with subset of parameters before full estimation
2. **Check Convergence**: Monitor ESS and acceptance rates
3. **Multiple Runs**: Run estimation multiple times with different seeds
4. **Prior Sensitivity**: Test different prior specifications

### Forecasting

1. **Validate Historically**: Compare forecasts to realized data
2. **Check Stability**: Ensure forecasts converge to steady state
3. **Scenario Analysis**: Use conditional forecasts for policy scenarios
4. **Uncertainty**: Always report confidence bands

### Performance

1. **Parallel Estimation**: Future versions will support parallel SMC
2. **Reduce Particles**: Start with fewer particles for testing
3. **Subset Parameters**: Estimate key parameters first
4. **Cache Solutions**: Save solved models to avoid re-solving

## Troubleshooting

### "No particles with finite likelihood"

**Cause**: All parameter draws produce unstable solutions or very poor fit

**Solutions:**
- Check model solution at prior means
- Relax prior distributions
- Reduce number of parameters being estimated
- Verify data is properly scaled

### "Very low ESS"

**Cause**: Particles have very different likelihoods

**Solutions:**
- Increase number of particles
- Increase number of tempering stages
- Check for outliers in data

### "Forecast explodes"

**Cause**: Unstable eigenvalues in solution

**Solutions:**
- Check eigenvalues: `print(solution.eigenvalues)`
- Verify parameter values are reasonable
- Check model specification for errors

## Next Steps

- **[Creating Custom Models](creating-models.md)**: Build your own DSGE model
- **[NYFed Model Details](../models/nyfed.md)**: Deep dive into the NYFed model
- **[API Reference](../api.md)**: Detailed API documentation
