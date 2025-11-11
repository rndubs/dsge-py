"""
FRED Series Mapping for St. Louis Fed DSGE Model.

This module defines the mapping between the St. Louis Fed DSGE model's 13 observable
variables and their corresponding FRED (Federal Reserve Economic Data) series codes,
along with the required transformations.

References:
    Faria-e-Castro, Miguel (2024). "The St. Louis Fed DSGE Model."
    Federal Reserve Bank of St. Louis Working Paper 2024-014.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class FREDSeriesMapping:
    """Mapping for a single observable variable to FRED series."""

    observable_name: str
    fred_code: str
    description: str
    transformation: str
    frequency: str
    units: str
    notes: str = ""


# St. Louis Fed DSGE Model Observable Variables (13 total)
STLOUISFED_FRED_MAPPINGS: List[FREDSeriesMapping] = [
    # ========================================================================
    # REAL ACTIVITY VARIABLES
    # ========================================================================
    FREDSeriesMapping(
        observable_name="dy",
        fred_code="GDPC1",
        description="Real Gross Domestic Product",
        transformation="400 * (log(GDP_t) - log(GDP_{t-1}))",
        frequency="Quarterly",
        units="Billions of Chained 2017 Dollars, Seasonally Adjusted Annual Rate",
        notes="Annualized quarterly growth rate in percent",
    ),
    FREDSeriesMapping(
        observable_name="dc",
        fred_code="PCECC96",
        description="Real Personal Consumption Expenditures",
        transformation="400 * (log(C_t) - log(C_{t-1}))",
        frequency="Quarterly",
        units="Billions of Chained 2017 Dollars, Seasonally Adjusted Annual Rate",
        notes="Annualized quarterly growth rate in percent",
    ),
    FREDSeriesMapping(
        observable_name="dinve",
        fred_code="GPDIC1",
        description="Real Gross Private Domestic Investment",
        transformation="400 * (log(I_t) - log(I_{t-1}))",
        frequency="Quarterly",
        units="Billions of Chained 2017 Dollars, Seasonally Adjusted Annual Rate",
        notes="Annualized quarterly growth rate in percent",
    ),
    FREDSeriesMapping(
        observable_name="dg",
        fred_code="GCEC1",
        description="Real Government Consumption Expenditures and Gross Investment",
        transformation="400 * (log(G_t) - log(G_{t-1}))",
        frequency="Quarterly",
        units="Billions of Chained 2017 Dollars, Seasonally Adjusted Annual Rate",
        notes="Annualized quarterly growth rate in percent",
    ),
    # ========================================================================
    # LABOR MARKET VARIABLES
    # ========================================================================
    FREDSeriesMapping(
        observable_name="hours",
        fred_code="HOANBS",
        description="Nonfarm Business Sector: Hours of All Persons",
        transformation="log(hours_t) - log(hours_mean)",
        frequency="Quarterly",
        units="Index 2017=100, Seasonally Adjusted",
        notes="Log deviation from mean. Available from 1964:Q1 onwards.",
    ),
    FREDSeriesMapping(
        observable_name="dw",
        fred_code="COMPNFB",
        description="Nonfarm Business Sector: Compensation Per Hour",
        transformation="400 * (log(W_t) - log(W_{t-1}) - log(GDP_deflator_t) + log(GDP_deflator_{t-1}))",
        frequency="Quarterly",
        units="Index 2017=100, Seasonally Adjusted",
        notes="Real wage growth (deflated by GDP deflator), annualized",
    ),
    # ========================================================================
    # PRICE AND INFLATION VARIABLES
    # ========================================================================
    FREDSeriesMapping(
        observable_name="infl",
        fred_code="PCEPILFE",
        description="Personal Consumption Expenditures Excluding Food and Energy (Core PCE)",
        transformation="400 * (log(PCE_t) - log(PCE_{t-1}))",
        frequency="Quarterly",
        units="Index 2017=100, Seasonally Adjusted",
        notes="Annualized Core PCE inflation. Available from 1979:Q4 onwards; earlier values inferred via Kalman filter.",
    ),
    FREDSeriesMapping(
        observable_name="infl_exp",
        fred_code="EXPINF10YR",
        description="University of Michigan: Inflation Expectation (10-Year)",
        transformation="No transformation (already in percent)",
        frequency="Quarterly (aggregated from monthly)",
        units="Percent",
        notes="10-year ahead inflation expectations. Measurement error included to capture deviations from rational expectations.",
    ),
    # ========================================================================
    # INTEREST RATE VARIABLES
    # ========================================================================
    FREDSeriesMapping(
        observable_name="ffr",
        fred_code="FEDFUNDS",
        description="Effective Federal Funds Rate",
        transformation="Quarterly average",
        frequency="Monthly → Quarterly",
        units="Percent per annum",
        notes="Policy rate, aggregated to quarterly frequency",
    ),
    FREDSeriesMapping(
        observable_name="r10y",
        fred_code="GS10",
        description="10-Year Treasury Constant Maturity Rate",
        transformation="Quarterly average",
        frequency="Monthly → Quarterly",
        units="Percent per annum",
        notes="Measurement error included to capture time-varying term premium not in model",
    ),
    # ========================================================================
    # FISCAL VARIABLES
    # ========================================================================
    FREDSeriesMapping(
        observable_name="ls",
        fred_code="LABSHPUSA156NRUG",
        description="Labor Share for United States",
        transformation="log(LS_t) - log(LS_mean)",
        frequency="Annual → Quarterly (interpolated)",
        units="Percent of GDP",
        notes="Labor compensation as share of GDP. Log deviation from mean.",
    ),
    FREDSeriesMapping(
        observable_name="debt_gdp",
        fred_code="GFDGDPA188S",
        description="Federal Debt: Total Public Debt as Percent of Gross Domestic Product",
        transformation="log(Debt_t / 4) - log(GDP_t)",
        frequency="Annual → Quarterly",
        units="Percent of GDP",
        notes="Federal debt to GDP ratio (quarterly). Convert annual debt to quarterly equivalent.",
    ),
    FREDSeriesMapping(
        observable_name="tax_gdp",
        fred_code="FGRECPT",
        description="Federal Government Current Tax Receipts",
        transformation="log(Tax_t) - log(GDP_t)",
        frequency="Quarterly",
        units="Billions of Dollars, Seasonally Adjusted Annual Rate",
        notes="Tax revenue to GDP ratio in logs",
    ),
]


def get_fred_code_dict() -> Dict[str, str]:
    """Return dictionary mapping observable names to FRED codes."""
    return {mapping.observable_name: mapping.fred_code for mapping in STLOUISFED_FRED_MAPPINGS}


def get_transformation_dict() -> Dict[str, str]:
    """Return dictionary mapping observable names to transformation formulas."""
    return {
        mapping.observable_name: mapping.transformation for mapping in STLOUISFED_FRED_MAPPINGS
    }


def print_data_summary():
    """Print a formatted summary of all FRED series mappings."""
    print("=" * 80)
    print("ST. LOUIS FED DSGE MODEL - FRED DATA SERIES MAPPINGS")
    print("=" * 80)
    print(f"\nTotal Observables: {len(STLOUISFED_FRED_MAPPINGS)}\n")

    for i, mapping in enumerate(STLOUISFED_FRED_MAPPINGS, 1):
        print(f"{i}. {mapping.observable_name.upper()}")
        print(f"   FRED Code: {mapping.fred_code}")
        print(f"   Description: {mapping.description}")
        print(f"   Transformation: {mapping.transformation}")
        print(f"   Frequency: {mapping.frequency}")
        print(f"   Units: {mapping.units}")
        if mapping.notes:
            print(f"   Notes: {mapping.notes}")
        print()


if __name__ == "__main__":
    print_data_summary()
