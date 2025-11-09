# Data Download Instructions

## Status

The data loading infrastructure is **complete and tested**, but actual data download requires a **FRED API key** (free from https://fred.stlouisfed.org/).

## What's Implemented

✅ **FRED Series Mapping** (`fred_series_mapping.py`)
- All 13 observable variables mapped to FRED series codes
- Transformation specifications for each variable
- Alternative series for robustness checks

✅ **Data Loading Module** (`src/dsge/data/fred_loader.py`)
- Functions to download data from FRED
- Automatic transformation application (growth rates, deflation, etc.)
- Data validation and quality checks
- Quarterly frequency conversion

✅ **Comprehensive Tests** (`tests/test_data_loading.py`)
- 25 unit tests covering all functionality
- All tests passing ✓
- Tests for transformations, frequency conversion, validation

✅ **Documentation** (`data/README.md`)
- Complete documentation of all 13 observables
- Data sources and transformations
- Usage examples and best practices

✅ **Download Script** (`data/download_nyfed_data.py`)
- Command-line tool for data download
- Automatic validation
- CSV export

## To Download Data

### Option 1: Get Free FRED API Key (Recommended)

1. Go to https://fred.stlouisfed.org/
2. Create a free account
3. Request API key at https://fred.stlouisfed.org/docs/api/api_key.html
4. Download data:

```bash
# Set API key
export FRED_API_KEY=your_key_here

# Download data
uv run python data/download_nyfed_data.py --validate

# Or with custom date range
uv run python data/download_nyfed_data.py --start 1960-01-01 --end 2024-01-01 --validate
```

### Option 2: Use Synthetic Data for Testing

For testing the estimation framework without real data, you can create synthetic data:

```python
import numpy as np
import pandas as pd
from models.nyfed_model_1002 import create_nyfed_model
from dsge.solvers.linear import solve_linear_model

# Create model
model = create_nyfed_model()

# Solve model
mats = model.system_matrices()
solution, info = solve_linear_model(
    Gamma0=mats['Gamma0'],
    Gamma1=mats['Gamma1'],
    Psi=mats['Psi'],
    Pi=mats['Pi'],
    n_states=model.spec.n_states
)

# Simulate synthetic data
T = 200  # 50 years of quarterly data
n_shocks = model.spec.n_shocks
n_states = model.spec.n_states

states = np.zeros((T, n_states))
shocks = np.random.randn(T, n_shocks) * 0.01

for t in range(1, T):
    states[t] = solution.T @ states[t-1] + solution.R @ shocks[t]

# Generate observables
Z, D = model.measurement_equation()
observables = states @ Z.T + D

# Create DataFrame
dates = pd.date_range('1960-01-01', periods=T, freq='Q')
data = pd.DataFrame(
    observables,
    index=dates,
    columns=model.spec.observable_names
)

# Save synthetic data
data.to_csv('data/nyfed_synthetic_data.csv')
print(f"✓ Synthetic data saved to data/nyfed_synthetic_data.csv")
```

### Option 3: Use Pre-Downloaded Data

If someone on your team has already downloaded the data:

```python
import pandas as pd

# Load pre-downloaded data
data = pd.read_csv('data/nyfed_data.csv', index_col=0, parse_dates=True)
```

## Testing Without Data

All tests in `tests/test_data_loading.py` work **without a FRED API key** because they use mock data. You can verify the data infrastructure works:

```bash
# Run all data tests (no API key needed)
uv run pytest tests/test_data_loading.py -v

# Should see: 25 passed
```

## Next Steps

Once data is downloaded (or synthetic data is created):

1. **Task 3.4**: Run estimation on the data
2. **Validate results**: Compare with published FRBNY estimates
3. **Task 3.5**: Generate forecasts

## Available Test Data

For immediate testing, the following is available:

- **Simple NK Model**: Uses own synthetic data (see `examples/zlb_full_estimation.py`)
- **AR(1) Model**: Tests pass with synthetic data
- **OccBin ZLB Model**: Has built-in synthetic data generation

## Questions?

- For FRED API issues: See https://fred.stlouisfed.org/docs/api/
- For transformation questions: See `data/README.md`
- For code examples: See `tests/test_data_loading.py`
