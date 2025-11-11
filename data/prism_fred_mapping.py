"""
FRED Data Mapping for PRISM-Inspired Model.

This module defines the mapping between the PRISM-inspired model's observable
variables and corresponding FRED (Federal Reserve Economic Data) series.

The model has 7 observable variables that can be matched to standard U.S.
macroeconomic time series.

References
----------
Federal Reserve Bank of St. Louis. "FRED Economic Data."
https://fred.stlouisfed.org/

For FRED API access, obtain a free API key at:
https://fred.stlouisfed.org/docs/api/api_key.html
"""

from dataclasses import dataclass


@dataclass
class FREDSeries:
    """Information about a FRED data series."""

    fred_code: str
    description: str
    frequency: str
    units: str
    transformation: str
    seasonal_adjustment: str


# PRISM-Inspired Model Observable Variables Mapping
PRISM_OBSERVABLE_MAPPING = {
    # Observable 1: Output Growth (dy)
    "obs_dy": FREDSeries(
        fred_code="GDPC1",
        description="Real Gross Domestic Product",
        frequency="Quarterly",
        units="Billions of Chained 2017 Dollars",
        transformation="Log difference × 100 (quarter-over-quarter growth rate)",
        seasonal_adjustment="Seasonally Adjusted Annual Rate"
    ),

    # Observable 2: Consumption Growth (dc)
    "obs_dc": FREDSeries(
        fred_code="PCECC96",
        description="Real Personal Consumption Expenditures",
        frequency="Quarterly",
        units="Billions of Chained 2017 Dollars",
        transformation="Log difference × 100 (quarter-over-quarter growth rate)",
        seasonal_adjustment="Seasonally Adjusted Annual Rate"
    ),

    # Observable 3: Inflation (pi)
    "obs_pi": FREDSeries(
        fred_code="PCECTPI",
        description="Personal Consumption Expenditures: Chain-type Price Index",
        frequency="Quarterly",
        units="Index 2017=100",
        transformation="Log difference × 400 (annualized inflation rate)",
        seasonal_adjustment="Seasonally Adjusted"
    ),

    # Observable 4: Nominal Interest Rate (r)
    "obs_r": FREDSeries(
        fred_code="FEDFUNDS",
        description="Federal Funds Effective Rate",
        frequency="Monthly → Quarterly (average)",
        units="Percent per annum",
        transformation="Level / 4 (convert annual to quarterly)",
        seasonal_adjustment="Not Seasonally Adjusted"
    ),

    # Observable 5: Real Wage Growth (dw)
    "obs_dw": FREDSeries(
        fred_code="COMPRNFB",
        description="Nonfarm Business Sector: Real Compensation Per Hour",
        frequency="Quarterly",
        units="Index 2017=100",
        transformation="Log difference × 100 (quarter-over-quarter growth rate)",
        seasonal_adjustment="Seasonally Adjusted"
    ),

    # Observable 6: Employment Growth (dn)
    "obs_dn": FREDSeries(
        fred_code="CE16OV",  # Alternative: PAYEMS (nonfarm payrolls)
        description="Civilian Employment Level (or Total Nonfarm Payrolls)",
        frequency="Monthly → Quarterly (average or end-of-period)",
        units="Thousands of Persons",
        transformation="Log difference × 100 (quarter-over-quarter growth rate)",
        seasonal_adjustment="Seasonally Adjusted"
    ),

    # Observable 7: Unemployment Rate (u)
    "obs_u": FREDSeries(
        fred_code="UNRATE",
        description="Unemployment Rate",
        frequency="Monthly → Quarterly (average)",
        units="Percent",
        transformation="Level (no transformation needed)",
        seasonal_adjustment="Seasonally Adjusted"
    ),
}


# Alternative series that could be used
ALTERNATIVE_SERIES = {
    "employment": {
        "PAYEMS": "All Employees, Total Nonfarm (thousands)",
        "CE16OV": "Civilian Employment (thousands)",
        "JTSJOL": "Job Openings: Total Nonfarm (thousands)",
    },

    "wages": {
        "COMPRNFB": "Real Compensation Per Hour, Nonfarm Business",
        "AHETPI": "Average Hourly Earnings of Production and Nonsupervisory Employees",
        "CES0500000003": "Average Hourly Earnings, Total Private",
    },

    "output": {
        "GDPC1": "Real GDP",
        "GDPDEF": "GDP Deflator",
        "INDPRO": "Industrial Production Index",
    },

    "inflation": {
        "PCECTPI": "PCE Price Index",
        "CPIAUCSL": "Consumer Price Index",
        "CPILFESL": "CPI Less Food and Energy",
    },

    "interest_rate": {
        "FEDFUNDS": "Federal Funds Rate",
        "GS3M": "3-Month Treasury Bill Rate",
        "DGS10": "10-Year Treasury Constant Maturity Rate",
    },

    "labor_market": {
        "UNRATE": "Unemployment Rate",
        "CIVPART": "Labor Force Participation Rate",
        "U6RATE": "Total Unemployed Plus Marginally Attached Plus Part Time",
        "JTSJOL": "Job Openings",
        "JTSQUR": "Quits Rate",
        "JTSLDR": "Layoffs and Discharges Rate",
    },
}


def get_fred_series_list() -> list[str]:
    """
    Get list of FRED series codes for PRISM model observables.

    Returns
    -------
    list[str]
        List of FRED series codes in order of observables.
    """
    return [series.fred_code for series in PRISM_OBSERVABLE_MAPPING.values()]


def get_transformation_description(observable: str) -> str:
    """
    Get transformation description for a given observable.

    Parameters
    ----------
    observable : str
        Observable variable name (e.g., 'obs_dy').

    Returns
    -------
    str
        Description of how to transform the raw FRED data.
    """
    if observable in PRISM_OBSERVABLE_MAPPING:
        return PRISM_OBSERVABLE_MAPPING[observable].transformation
    else:
        raise ValueError(f"Unknown observable: {observable}")


def print_data_summary() -> None:
    """Print summary of FRED data mapping."""
    print("=" * 80)
    print("PRISM-Inspired Model: FRED Data Mapping")
    print("=" * 80)
    print()

    for i, (obs_name, series) in enumerate(PRISM_OBSERVABLE_MAPPING.items(), 1):
        print(f"{i}. {obs_name}")
        print(f"   FRED Code: {series.fred_code}")
        print(f"   Description: {series.description}")
        print(f"   Transformation: {series.transformation}")
        print()

    print("=" * 80)
    print(f"Total observables: {len(PRISM_OBSERVABLE_MAPPING)}")
    print("=" * 80)


if __name__ == "__main__":
    print_data_summary()

    print("\n\nFRED Series List:")
    print(get_fred_series_list())

    print("\n\nExample Transformation:")
    print(f"Output growth: {get_transformation_description('obs_dy')}")
