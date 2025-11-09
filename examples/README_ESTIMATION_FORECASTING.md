# NYFed Model 1002 - Estimation & Forecasting Guide

This directory contains examples for estimating and forecasting with the NYFed DSGE Model 1002.

## Quick Start

### 1. Generate Synthetic Data

```bash
# Generate 200 quarters of synthetic data
uv run python examples/generate_nyfed_synthetic_data.py --periods 200 --shock-std 0.01

# Output: data/nyfed_synthetic_data.csv
```

### 2. Estimate Model (Optional - computationally intensive)

```bash
# Estimate subset of parameters (faster)
uv run python examples/estimate_nyfed_model.py --particles 500 --stages 50

# Estimate all parameters (slower)
uv run python examples/estimate_nyfed_model.py --full --particles 2000 --stages 200

# Output: results/nyfed_estimation/
#   - posterior_samples.csv
#   - posterior_summary.csv
```

### 3. Generate Forecasts

```bash
# Unconditional forecast (without parameter uncertainty)
uv run python examples/forecast_nyfed_model.py --horizon 20 --paths 1000

# Conditional forecast with posterior uncertainty
uv run python examples/forecast_nyfed_model.py --use-posterior --horizon 20

# Output: results/nyfed_forecasts/
#   - forecast_mean.csv
#   - forecast_lower_68.csv, forecast_upper_68.csv
#   - forecast_lower_90.csv, forecast_upper_90.csv
#   - forecast_plot.png
```

## Examples

### Example 1: Synthetic Data Generation

```python
from examples.generate_nyfed_synthetic_data import generate_synthetic_data

# Generate data
data = generate_synthetic_data(
    T=200,              # 50 years of quarterly data
    shock_std=0.01,     # Standard deviation of shocks
    seed=42,            # Reproducibility
    save_path='data/nyfed_synthetic_data.csv'
)

print(f"Generated {len(data)} observations")
print(f"Variables: {list(data.columns)}")
```

### Example 2: Model Estimation

```python
from examples.estimate_nyfed_model import estimate_nyfed_model

# Run estimation
results = estimate_nyfed_model(
    data_path='data/nyfed_synthetic_data.csv',
    n_particles=1000,          # SMC particles
    n_phi=100,                 # Tempering stages
    save_dir='results/nyfed_estimation',
    subset_params=True,        # Estimate subset for speed
    verbose=True
)

# Access results
posterior_mean = results['posterior_mean']
posterior_std = results['posterior_std']
log_evidence = results['log_evidence']

print(f"Log marginal likelihood: {log_evidence:.2f}")
for i, param in enumerate(results['param_names']):
    print(f"{param}: {posterior_mean[i]:.4f} ± {posterior_std[i]:.4f}")
```

### Example 3: Unconditional Forecasting

```python
from dsge.forecasting import forecast_observables
from models.nyfed_model_1002 import create_nyfed_model
from dsge.solvers.linear import solve_linear_model
import numpy as np

# Create and solve model
model = create_nyfed_model()
mats = model.system_matrices()

solution, info = solve_linear_model(
    Gamma0=mats['Gamma0'],
    Gamma1=mats['Gamma1'],
    Psi=mats['Psi'],
    Pi=mats['Pi'],
    n_states=model.spec.n_states
)

# Get measurement equation
Z, D = model.measurement_equation()

# Initial state (e.g., from Kalman filter)
x_T = np.zeros(model.spec.n_states)

# Generate forecast
forecast_result = forecast_observables(
    T=solution.T,
    R=solution.R,
    C=solution.C,
    Z=Z,
    D=D,
    x_T=x_T,
    horizon=20,         # 5 years ahead
    n_paths=1000,
    seed=42
)

# Access forecasts
forecast_mean = forecast_result.mean  # (horizon x n_obs)
forecast_paths = forecast_result.paths  # (n_paths x horizon x n_obs)
forecast_bands = forecast_result.bands  # {level: (lower, upper)}

# 90% uncertainty band
lower_90, upper_90 = forecast_bands[0.90]
```

### Example 4: Conditional Forecasting

```python
from dsge.forecasting import conditional_forecast

# Condition on specific observable values
# E.g., fix GDP growth at 2% for first 4 quarters
conditions = {
    0: {0: 2.0},  # obs_gdp_growth = 2% in Q1
    1: {0: 2.0},  # obs_gdp_growth = 2% in Q2
    2: {0: 2.0},  # obs_gdp_growth = 2% in Q3
    3: {0: 2.0},  # obs_gdp_growth = 2% in Q4
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
    n_paths=1000,
    seed=42
)

# Conditional forecast respects constraints
print(f"Q1 forecast: {conditional_result.mean[0, 0]:.2f}%")  # ≈ 2.0%
```

### Example 5: Forecasting with Parameter Uncertainty

```python
from dsge.forecasting import forecast_from_posterior
import pandas as pd

# Load posterior samples from estimation
posterior_df = pd.read_csv('results/nyfed_estimation/posterior_samples.csv')
posterior_samples = posterior_df.drop('weight', axis=1).values
weights = posterior_df['weight'].values

# Forecast incorporating parameter uncertainty
forecast_result = forecast_from_posterior(
    posterior_samples=posterior_samples,
    posterior_weights=weights,
    model=model,
    x_T=x_T,
    horizon=20,
    n_forecast_paths=100,      # Paths per parameter draw
    n_posterior_draws=100,      # Number of parameter draws
    seed=42
)

# Uncertainty bands account for both parameter and shock uncertainty
lower_90, upper_90 = forecast_result.bands[0.90]
```

## Estimation Settings

### SMC Configuration

**For testing (fast)**:
- Particles: 500-1000
- Tempering stages: 50-100
- Subset of parameters: ~10 key parameters
- Runtime: ~5-15 minutes

**For production (accurate)**:
- Particles: 2000-5000
- Tempering stages: 200-500
- All parameters with priors
- Runtime: 1-4 hours (depending on hardware)

### Parameter Subsets

**Subset estimation** (for testing):
- Policy: psi_1, psi_2, rho_r
- Preferences: sigma_c, h
- Shocks: rho_z, rho_b, sigma_r, sigma_z, sigma_b

**Full estimation**:
- All 67 model parameters with prior distributions
- See `models/nyfed_model_1002.py` for complete list

## Forecasting Options

### Uncertainty Sources

1. **Shock uncertainty only** (unconditional forecast)
   - Uses calibrated/estimated parameters
   - Uncertainty from structural shocks only

2. **Parameter + shock uncertainty** (posterior forecast)
   - Samples from posterior distribution
   - Accounts for parameter estimation uncertainty
   - Wider confidence bands

### Forecast Horizons

- **Short-term** (1-8 quarters): Most reliable
- **Medium-term** (2-5 years): Moderate uncertainty
- **Long-term** (5+ years): Wide uncertainty, convergence to steady state

### Confidence Bands

- **68%**: Inner band (1 standard deviation)
- **90%**: Outer band (common in policy reports)
- **95%**: Widest band (conservative)

## Output Files

### Estimation Output

- `posterior_samples.csv`: Full posterior sample (particles × parameters)
- `posterior_summary.csv`: Mean, std dev for each parameter
- Columns: parameter names + weights

### Forecast Output

- `forecast_mean.csv`: Point forecast (horizon × observables)
- `forecast_lower_XX.csv`: Lower XX% confidence bound
- `forecast_upper_XX.csv`: Upper XX% confidence bound
- `forecast_plot.png`: Visualization of key variables

## Performance Notes

### Memory Requirements

- **Synthetic data generation**: < 100 MB
- **Estimation** (1000 particles): ~500 MB - 1 GB
- **Forecasting** (1000 paths): < 500 MB

### Computational Time (approximate)

**On modern CPU (e.g., Intel i7)**:
- Data generation: < 10 seconds
- Subset estimation (1000 particles, 100 stages): 5-10 minutes
- Full estimation (2000 particles, 200 stages): 1-2 hours
- Forecasting (1000 paths): < 30 seconds

**Parallel execution**:
- SMC can be parallelized (future enhancement)
- Forecast paths are independent (can parallelize)

## Validation

### Posterior Checks

After estimation, validate:

1. **Convergence**: ESS > 50% of particles
2. **Acceptance rates**: 20-40% in RWMH mutation
3. **Posterior vs. Prior**: Parameters should update from data
4. **Log evidence**: Compare models (higher is better)

```python
results = estimate_nyfed_model(...)

# Check diagnostics
print(f"Final ESS: {results.get('ess', 'N/A')}")
print(f"Log evidence: {results['log_evidence']:.2f}")

# Compare posterior to prior
for i, param in enumerate(results['param_names']):
    prior_val = model.parameters[param].value
    post_val = results['posterior_mean'][i]
    print(f"{param}: {prior_val:.4f} → {post_val:.4f}")
```

### Forecast Validation

Check forecast sensibility:

1. **Stationarity**: Forecasts converge to steady state
2. **Uncertainty**: Bands widen then stabilize
3. **Historical fit**: Model captures data dynamics
4. **IRF consistency**: Shocks have expected effects

## Troubleshooting

### Estimation Issues

**Problem**: "No particles with finite likelihood"
- **Solution**: Check model solution stability
- **Solution**: Relax prior distributions
- **Solution**: Reduce number of parameters

**Problem**: "Very low ESS (<10%)"
- **Solution**: Increase number of particles
- **Solution**: Increase tempering stages
- **Solution**: Adjust tempering schedule

**Problem**: "Slow convergence"
- **Solution**: Reduce parameter count (subset)
- **Solution**: Better prior specifications
- **Solution**: Check for identification issues

### Forecasting Issues

**Problem**: "Forecast explodes"
- **Solution**: Check solution stability (eigenvalues)
- **Solution**: Verify model specification
- **Solution**: Use more reasonable initial state

**Problem**: "Very wide uncertainty bands"
- **Solution**: Normal for long horizons
- **Solution**: Consider conditional forecast
- **Solution**: Check shock standard deviations

## References

1. **SMC Estimation**: Herbst & Schorfheide (2015) "Bayesian Estimation of DSGE Models"
2. **NYFed Model**: Del Negro et al. (2015) "Inflation in the Great Recession and New Keynesian Models"
3. **Forecasting**: Schorfheide & Song (2015) "Real-Time Forecasting with a Mixed-Frequency VAR"
4. **DSGE.jl**: FRBNY reference implementation (https://github.com/FRBNY-DSGE/DSGE.jl)

## See Also

- `tests/test_forecasting.py`: Forecasting unit tests
- `validation/VALIDATION_REPORT.md`: Model solution validation
- `data/README.md`: Data documentation
- `PLAN.md`: Overall project plan
