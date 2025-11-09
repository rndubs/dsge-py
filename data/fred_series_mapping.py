"""
FRED Series Mapping for NYFed DSGE Model 1002

This module defines the mapping between model observables and FRED data series,
along with the necessary transformations.

Reference: FRBNY DSGE Model Documentation (March 3, 2021)
Data period: 1960-Q1 to present (quarterly data)
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class SeriesSpec:
    """Specification for a FRED data series."""
    fred_code: str
    description: str
    units: str
    transformation: str
    notes: Optional[str] = None


# FRED Series Mapping for NYFed Model 1002
# Based on DSGE.jl data specification and FRBNY documentation
FRED_SERIES_MAP: Dict[str, SeriesSpec] = {
    # 1. GDP Growth (Real GDP, quarterly growth rate)
    'obs_gdp_growth': SeriesSpec(
        fred_code='GDPC1',
        description='Real Gross Domestic Product',
        units='Billions of Chained 2017 Dollars, Seasonally Adjusted Annual Rate',
        transformation='quarterly_growth_rate',
        notes='Take log difference and multiply by 100 (annualized quarterly growth)'
    ),

    # 2. GDI Growth (Real GDI, quarterly growth rate)
    'obs_gdi_growth': SeriesSpec(
        fred_code='GDI',  # Nominal GDI
        description='Gross Domestic Income',
        units='Billions of Dollars, Seasonally Adjusted Annual Rate',
        transformation='real_quarterly_growth_rate',
        notes='Deflate by GDP deflator, then compute growth rate. Note: GDI has measurement errors'
    ),

    # 3. Consumption Growth (Real Personal Consumption Expenditures)
    'obs_cons_growth': SeriesSpec(
        fred_code='PCECC96',
        description='Real Personal Consumption Expenditures',
        units='Billions of Chained 2017 Dollars, Seasonally Adjusted Annual Rate',
        transformation='quarterly_growth_rate',
        notes='Total personal consumption expenditures (goods + services)'
    ),

    # 4. Investment Growth (Real Gross Private Domestic Investment)
    'obs_inv_growth': SeriesSpec(
        fred_code='GPDIC1',
        description='Real Gross Private Domestic Investment',
        units='Billions of Chained 2017 Dollars, Seasonally Adjusted Annual Rate',
        transformation='quarterly_growth_rate',
        notes='Fixed investment + change in inventories'
    ),

    # 5. Wage Growth (Compensation per Hour, nonfarm business sector)
    'obs_wage_growth': SeriesSpec(
        fred_code='COMPNFB',  # Nominal compensation
        description='Nonfarm Business Sector: Compensation Per Hour',
        units='Index 2012=100, Seasonally Adjusted',
        transformation='real_quarterly_growth_rate',
        notes='Deflate by GDP deflator, then compute growth rate'
    ),

    # 6. Hours Worked (Total Hours, nonfarm business sector)
    'obs_hours': SeriesSpec(
        fred_code='HOANBS',
        description='Nonfarm Business Sector: Hours of All Persons',
        units='Index 2012=100, Seasonally Adjusted',
        transformation='log_level',
        notes='Take natural log and demean. Per capita adjustment may be applied'
    ),

    # 7. Core PCE Inflation (quarterly, annualized)
    'obs_infl_pce': SeriesSpec(
        fred_code='PCEPILFE',
        description='Personal Consumption Expenditures Excluding Food and Energy (Chain-Type Price Index)',
        units='Index 2017=100, Seasonally Adjusted',
        transformation='quarterly_inflation_rate',
        notes='Core PCE, preferred Fed inflation measure'
    ),

    # 8. GDP Deflator Inflation
    'obs_infl_gdpdef': SeriesSpec(
        fred_code='GDPDEF',
        description='Gross Domestic Product: Implicit Price Deflator',
        units='Index 2017=100, Seasonally Adjusted',
        transformation='quarterly_inflation_rate',
        notes='Broader inflation measure than PCE'
    ),

    # 9. Federal Funds Rate (quarterly average, annualized)
    'obs_ffr': SeriesSpec(
        fred_code='FEDFUNDS',
        description='Federal Funds Effective Rate',
        units='Percent, Not Seasonally Adjusted',
        transformation='quarterly_average',
        notes='Monthly data, take quarterly average. Already in annualized percent.'
    ),

    # 10. 10-Year Treasury Rate (quarterly average, annualized)
    'obs_10y_rate': SeriesSpec(
        fred_code='GS10',
        description='Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity',
        units='Percent, Not Seasonally Adjusted',
        transformation='quarterly_average',
        notes='Monthly data, take quarterly average'
    ),

    # 11. 10-Year Inflation Expectations
    'obs_10y_infl_exp': SeriesSpec(
        fred_code='EXPINF10YR',  # Survey of Professional Forecasters
        description='University of Michigan: Inflation Expectation (10-Year ahead)',
        units='Percent',
        transformation='quarterly_average',
        notes='Survey-based measure. Alternative: SPF 10-year expectations'
    ),

    # 12. Credit Spread (Baa corporate - 10-year Treasury)
    'obs_spread': SeriesSpec(
        fred_code='BAA10Y',  # Computed as BAMLC0A4CBBB - GS10
        description='Moodys Seasoned Baa Corporate Bond Yield Relative to 10-Year Treasury',
        units='Percent, Not Seasonally Adjusted',
        transformation='quarterly_average',
        notes='Credit risk premium. Computed as Baa rate minus 10Y Treasury'
    ),

    # 13. TFP Growth (Total Factor Productivity)
    'obs_tfp_growth': SeriesSpec(
        fred_code='TFPKQ',  # San Francisco Fed TFP series
        description='Total Factor Productivity (Adjusted for Utilization)',
        units='Index 2009=100, Seasonally Adjusted',
        transformation='quarterly_growth_rate',
        notes='Utilization-adjusted TFP from John Fernald (SF Fed)'
    ),
}


# Alternative series for robustness checks
ALTERNATIVE_SERIES: Dict[str, Dict[str, str]] = {
    'obs_gdi_growth': {
        'alt_1': 'A261RX1Q020SBEA',  # Real GDI (if available)
        'note': 'Direct real GDI series may not be available for full sample'
    },
    'obs_10y_infl_exp': {
        'alt_1': 'T10YIE',  # 10-Year Breakeven Inflation Rate
        'alt_2': 'MICH',    # Michigan Survey (median)
        'note': 'Multiple sources for inflation expectations'
    },
    'obs_spread': {
        'alt_1': 'BAMLC0A4CBBB',  # Baa corporate yield
        'note': 'Subtract GS10 to get spread'
    },
    'obs_tfp_growth': {
        'alt_1': 'GDPC1',  # Can use Solow residual if TFP not available
        'note': 'Fernald TFP is preferred; can construct from production function'
    }
}


# Data transformations
TRANSFORMATION_FUNCTIONS = {
    'quarterly_growth_rate': '100 * (log(X_t) - log(X_{t-1}))',
    'real_quarterly_growth_rate': '100 * (log(X_t/P_t) - log(X_{t-1}/P_{t-1}))',
    'quarterly_inflation_rate': '400 * (log(P_t) - log(P_{t-1}))',  # Annualized
    'quarterly_average': 'mean(X_t over quarter)',
    'log_level': 'log(X_t)',
    'level': 'X_t'
}


def get_series_spec(observable_name: str) -> SeriesSpec:
    """Get the FRED series specification for an observable."""
    if observable_name not in FRED_SERIES_MAP:
        raise ValueError(f"Unknown observable: {observable_name}")
    return FRED_SERIES_MAP[observable_name]


def get_all_fred_codes() -> Dict[str, str]:
    """Get mapping from observable names to FRED series codes."""
    return {obs: spec.fred_code for obs, spec in FRED_SERIES_MAP.items()}


def get_transformation(observable_name: str) -> str:
    """Get the transformation type for an observable."""
    spec = get_series_spec(observable_name)
    return spec.transformation
