# NYFed DSGE Model 1002 - Data Documentation

This directory contains data specifications and loading utilities for the NYFed DSGE Model 1002.

## Overview

The model uses **13 observable variables** from FRED (Federal Reserve Economic Data) spanning the period from 1960-Q1 to present. All data is quarterly frequency.

## Observable Variables and FRED Series

### 1. GDP Growth (`obs_gdp_growth`)
- **FRED Code**: `GDPC1`
- **Description**: Real Gross Domestic Product
- **Units**: Billions of Chained 2017 Dollars, SAAR
- **Transformation**: Quarterly growth rate (annualized)
  - Formula: `400 * (log(GDPC1_t) - log(GDPC1_{t-1}))`
  - Interpretation: Annualized percent change in real GDP

### 2. GDI Growth (`obs_gdi_growth`)
- **FRED Code**: `GDI` (deflated by `GDPDEF`)
- **Description**: Gross Domestic Income
- **Units**: Billions of Dollars, SAAR → deflated to real terms
- **Transformation**: Real quarterly growth rate
  - Formula: `400 * (log(GDI_t/GDPDEF_t) - log(GDI_{t-1}/GDPDEF_{t-1}))`
  - Note: GDI has measurement errors (hence the e_gdi measurement error term)

### 3. Consumption Growth (`obs_cons_growth`)
- **FRED Code**: `PCECC96`
- **Description**: Real Personal Consumption Expenditures
- **Units**: Billions of Chained 2017 Dollars, SAAR
- **Transformation**: Quarterly growth rate
  - Formula: `400 * (log(PCECC96_t) - log(PCECC96_{t-1}))`

### 4. Investment Growth (`obs_inv_growth`)
- **FRED Code**: `GPDIC1`
- **Description**: Real Gross Private Domestic Investment
- **Units**: Billions of Chained 2017 Dollars, SAAR
- **Transformation**: Quarterly growth rate
  - Formula: `400 * (log(GPDIC1_t) - log(GPDIC1_{t-1}))`
  - Includes: Fixed investment + change in private inventories

### 5. Wage Growth (`obs_wage_growth`)
- **FRED Code**: `COMPNFB` (deflated by `GDPDEF`)
- **Description**: Nonfarm Business Sector: Compensation Per Hour
- **Units**: Index 2012=100, SA → deflated to real terms
- **Transformation**: Real quarterly growth rate
  - Formula: `400 * (log(COMPNFB_t/GDPDEF_t) - log(COMPNFB_{t-1}/GDPDEF_{t-1}))`

### 6. Hours Worked (`obs_hours`)
- **FRED Code**: `HOANBS`
- **Description**: Nonfarm Business Sector: Hours of All Persons
- **Units**: Index 2012=100, SA
- **Transformation**: Log level
  - Formula: `log(HOANBS_t)`
  - Note: Often demeaned or adjusted for population

### 7. Core PCE Inflation (`obs_infl_pce`)
- **FRED Code**: `PCEPILFE`
- **Description**: Core PCE Price Index (Excluding Food and Energy)
- **Units**: Index 2017=100, SA
- **Transformation**: Quarterly inflation rate (annualized)
  - Formula: `400 * (log(PCEPILFE_t) - log(PCEPILFE_{t-1}))`
  - This is the Fed's preferred inflation measure

### 8. GDP Deflator Inflation (`obs_infl_gdpdef`)
- **FRED Code**: `GDPDEF`
- **Description**: GDP Implicit Price Deflator
- **Units**: Index 2017=100, SA
- **Transformation**: Quarterly inflation rate (annualized)
  - Formula: `400 * (log(GDPDEF_t) - log(GDPDEF_{t-1}))`
  - Broader inflation measure than PCE

### 9. Federal Funds Rate (`obs_ffr`)
- **FRED Code**: `FEDFUNDS`
- **Description**: Effective Federal Funds Rate
- **Units**: Percent, NSA
- **Transformation**: Quarterly average
  - Formula: Average of monthly FEDFUNDS over quarter
  - Note: Already in annualized percent (e.g., 2.5 = 2.5%)

### 10. 10-Year Treasury Rate (`obs_10y_rate`)
- **FRED Code**: `GS10`
- **Description**: 10-Year Treasury Constant Maturity Rate
- **Units**: Percent, NSA
- **Transformation**: Quarterly average
  - Formula: Average of monthly GS10 over quarter

### 11. 10-Year Inflation Expectations (`obs_10y_infl_exp`)
- **FRED Code**: `EXPINF10YR` or `T10YIE`
- **Description**: 10-Year Ahead Inflation Expectations
- **Units**: Percent
- **Transformation**: Quarterly average
  - Primary: University of Michigan Survey
  - Alternative: 10-Year Breakeven Inflation Rate (T10YIE)
  - Alternative: Survey of Professional Forecasters (SPF)

### 12. Credit Spread (`obs_spread`)
- **FRED Code**: `BAA10Y` (or compute as BAMLC0A4CBBB - GS10)
- **Description**: Baa Corporate Bond Yield - 10Y Treasury Spread
- **Units**: Percent, NSA
- **Transformation**: Quarterly average
  - Formula: Average of (Baa rate - GS10) over quarter
  - Measures credit risk premium

### 13. TFP Growth (`obs_tfp_growth`)
- **FRED Code**: `TFPKQ`
- **Description**: Total Factor Productivity (Utilization-Adjusted)
- **Units**: Index 2009=100, SA
- **Transformation**: Quarterly growth rate
  - Formula: `400 * (log(TFPKQ_t) - log(TFPKQ_{t-1}))`
  - Source: John Fernald (SF Fed)
  - Note: Utilization-adjusted TFP is preferred

## Data Transformations

All growth rates are computed as log differences and expressed in annualized percent:
- **Quarterly Growth Rate**: `400 * (log(X_t) - log(X_{t-1}))`
  - Factor of 400 = 100 (percent) × 4 (annualize)
- **Real Growth Rate**: First deflate by price index, then compute growth
- **Inflation Rate**: Same formula as growth rate, applied to price indices

## Estimation Period

Standard estimation periods used in literature:
- **Full sample**: 1960-Q1 to present
- **Great Moderation**: 1984-Q1 to 2007-Q4
- **Post-Financial Crisis**: 2008-Q1 to present
- **Recent**: 2000-Q1 to present

## Data Quality Notes

### Measurement Errors
The model includes measurement error terms for:
- **GDP growth** (`e_gdp`): NIPA revisions and measurement issues
- **GDI growth** (`e_gdi`): Known discrepancy between GDP and GDI
- **Core PCE inflation** (`e_pce`): Short-run noise in price indices
- **GDP deflator** (`e_gdpdef`): Different price aggregation
- **10-year rate** (`e_10y`): Term premium shocks
- **TFP growth** (`e_tfp`): TFP measurement error

### Cointegration
GDP and GDI are cointegrated in the long run, so the model includes:
- Cointegration parameter `C_me` in measurement equations
- This helps with identification and improves filtering

### Missing Data
Some series may have missing observations:
- **TFP**: Fernald series starts in 1947, may have gaps
- **Inflation expectations**: Survey data may be sparse before 1990s
- **Credit spreads**: May be affected by financial crises

## Using the Data

### Quick Start

```python
from dsge.data import load_nyfed_data

# Download and transform all 13 observables
# Requires FRED API key (free from https://fred.stlouisfed.org/)
data = load_nyfed_data(
    start_date='1960-01-01',
    end_date='2024-01-01',
    api_key='your_fred_api_key',
    save_path='data/nyfed_data.csv'
)

print(data.head())
print(data.describe())
```

### Get Free FRED API Key

1. Go to https://fred.stlouisfed.org/
2. Create a free account
3. Request an API key at https://fred.stlouisfed.org/docs/api/api_key.html
4. Set environment variable: `export FRED_API_KEY=your_key_here`

### Alternative: Use Pre-Downloaded Data

If you don't have a FRED API key, you can use pre-processed data files:

```python
import pandas as pd

# Load pre-downloaded data
data = pd.read_csv('data/nyfed_data_1960_2024.csv', index_col=0, parse_dates=True)
```

## Data Validation

After loading data, always validate:

```python
from dsge.data import validate_data

# Check for missing values, outliers, etc.
validation_results = validate_data(data, verbose=True)

# Check date range
print(f"Data span: {data.index[0]} to {data.index[-1]}")
print(f"Observations: {len(data)}")

# Check for issues
if validation_results['missing_count'].sum() > 0:
    print("⚠️ Missing data detected!")
```

## References

1. **FRBNY DSGE Model Documentation** (March 3, 2021)
   - Complete model specification and data sources

2. **DSGE.jl** - Julia implementation by FRBNY
   - https://github.com/FRBNY-DSGE/DSGE.jl
   - Reference implementation for validation

3. **FRED API Documentation**
   - https://fred.stlouisfed.org/docs/api/fred/

4. **Fernald TFP Data**
   - https://www.frbsf.org/economic-research/indicators-data/total-factor-productivity-tfp/
   - Quarterly utilization-adjusted TFP

## File Structure

```
data/
├── README.md                          # This file
├── fred_series_mapping.py             # Series specifications
├── nyfed_data_1960_2024.csv          # Pre-downloaded data (if available)
└── download_nyfed_data.py            # Data download script
```

## Support

For questions about:
- **Data series**: Consult FRED documentation
- **Transformations**: See fred_series_mapping.py
- **Data issues**: Check NYFED DSGE Model Documentation
- **Code issues**: See tests/test_data_loading.py for examples
