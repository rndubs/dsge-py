"""
FRED Data Loader for DSGE Estimation

This module provides utilities for downloading and transforming economic data
from FRED (Federal Reserve Economic Data) for use in DSGE model estimation.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
import warnings
from pathlib import Path


def download_fred_series(
    series_id: str,
    start_date: str = '1960-01-01',
    end_date: Optional[str] = None,
    api_key: Optional[str] = None
) -> pd.Series:
    """
    Download a single series from FRED.

    Parameters
    ----------
    series_id : str
        FRED series identifier (e.g., 'GDPC1')
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format (default: today)
    api_key : str, optional
        FRED API key. If None, will try to use fredapi with environment variable

    Returns
    -------
    pd.Series
        Downloaded series with datetime index
    """
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError(
            "fredapi package required for FRED data download. "
            "Install with: uv add fredapi"
        )

    # Initialize FRED API
    if api_key is None:
        # Try to get from environment variable
        import os
        api_key = os.environ.get('FRED_API_KEY')
        if api_key is None:
            warnings.warn(
                "No FRED API key provided. Set FRED_API_KEY environment variable "
                "or pass api_key parameter. You can get a free API key at "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )
            # Return empty series if no API key
            return pd.Series(dtype=float)

    fred = Fred(api_key=api_key)

    # Download series
    try:
        series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        return series
    except Exception as e:
        warnings.warn(f"Failed to download {series_id}: {str(e)}")
        return pd.Series(dtype=float)


def to_quarterly(series: pd.Series, agg_method: str = 'mean') -> pd.Series:
    """
    Convert monthly or daily data to quarterly frequency.

    Parameters
    ----------
    series : pd.Series
        Input series with datetime index
    agg_method : str
        Aggregation method: 'mean', 'last', 'sum'

    Returns
    -------
    pd.Series
        Quarterly series
    """
    if agg_method == 'mean':
        return series.resample('Q').mean()
    elif agg_method == 'last':
        return series.resample('Q').last()
    elif agg_method == 'sum':
        return series.resample('Q').sum()
    else:
        raise ValueError(f"Unknown aggregation method: {agg_method}")


def compute_growth_rate(series: pd.Series, annualize: bool = True) -> pd.Series:
    """
    Compute quarterly growth rate.

    Growth rate = 100 * (log(X_t) - log(X_{t-1}))

    Parameters
    ----------
    series : pd.Series
        Input series (levels)
    annualize : bool
        If True, multiply by 4 to annualize

    Returns
    -------
    pd.Series
        Growth rate in percent
    """
    log_series = np.log(series)
    growth = 100 * log_series.diff()

    if annualize:
        growth = growth * 4

    return growth


def compute_inflation_rate(price_index: pd.Series, annualize: bool = True) -> pd.Series:
    """
    Compute quarterly inflation rate.

    Inflation = 100 * (log(P_t) - log(P_{t-1}))

    Parameters
    ----------
    price_index : pd.Series
        Price index series
    annualize : bool
        If True, multiply by 4 to get annualized rate

    Returns
    -------
    pd.Series
        Inflation rate in percent
    """
    return compute_growth_rate(price_index, annualize=annualize)


def compute_real_series(nominal: pd.Series, deflator: pd.Series) -> pd.Series:
    """
    Deflate a nominal series using a price deflator.

    Real = Nominal / Deflator * 100

    Parameters
    ----------
    nominal : pd.Series
        Nominal series
    deflator : pd.Series
        Price deflator (index form)

    Returns
    -------
    pd.Series
        Real series
    """
    return (nominal / deflator) * 100


def transform_series(
    series: pd.Series,
    transformation: str,
    deflator: Optional[pd.Series] = None
) -> pd.Series:
    """
    Apply transformation to a series.

    Parameters
    ----------
    series : pd.Series
        Input series
    transformation : str
        Transformation type:
        - 'quarterly_growth_rate': log difference * 100
        - 'real_quarterly_growth_rate': deflate then growth rate
        - 'quarterly_inflation_rate': inflation rate (annualized)
        - 'quarterly_average': convert to quarterly if needed
        - 'log_level': natural log
        - 'level': no transformation
    deflator : pd.Series, optional
        Price deflator for real transformations

    Returns
    -------
    pd.Series
        Transformed series
    """
    if transformation == 'quarterly_growth_rate':
        return compute_growth_rate(series, annualize=True)

    elif transformation == 'real_quarterly_growth_rate':
        if deflator is None:
            raise ValueError("Deflator required for real_quarterly_growth_rate transformation")
        real_series = compute_real_series(series, deflator)
        return compute_growth_rate(real_series, annualize=True)

    elif transformation == 'quarterly_inflation_rate':
        return compute_inflation_rate(series, annualize=True)

    elif transformation == 'quarterly_average':
        # If monthly, convert to quarterly
        freq = pd.infer_freq(series.index)
        if freq and ('M' in freq or 'D' in freq):
            return to_quarterly(series, agg_method='mean')
        return series

    elif transformation == 'log_level':
        return np.log(series)

    elif transformation == 'level':
        return series

    else:
        raise ValueError(f"Unknown transformation: {transformation}")


def load_nyfed_data(
    start_date: str = '1960-01-01',
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Load and transform all data for NYFed DSGE Model 1002.

    Parameters
    ----------
    start_date : str
        Start date for data download
    end_date : str, optional
        End date for data download
    api_key : str, optional
        FRED API key
    save_path : str, optional
        Path to save processed data (CSV format)

    Returns
    -------
    pd.DataFrame
        Dataframe with all 13 observables as columns, quarterly frequency
    """
    # Import the series mapping
    import sys
    from pathlib import Path

    # Add data directory to path
    data_dir = Path(__file__).parent.parent.parent / 'data'
    sys.path.insert(0, str(data_dir))

    try:
        from fred_series_mapping import FRED_SERIES_MAP, get_series_spec
    except ImportError:
        raise ImportError("Could not import fred_series_mapping. Check data/ directory.")

    print(f"Loading NYFed DSGE Model 1002 data from FRED...")
    print(f"Period: {start_date} to {end_date or 'present'}")
    print(f"Number of observables: {len(FRED_SERIES_MAP)}")

    # Download raw series
    raw_data = {}

    # First, download GDP deflator (needed for real transformations)
    print("\n1. Downloading GDP deflator...")
    gdpdef = download_fred_series('GDPDEF', start_date, end_date, api_key)
    gdpdef_q = to_quarterly(gdpdef, 'mean') if not gdpdef.empty else pd.Series()

    # Download all series
    for i, (obs_name, spec) in enumerate(FRED_SERIES_MAP.items(), 1):
        print(f"\n{i}. Downloading {obs_name}: {spec.fred_code}")
        print(f"   Description: {spec.description}")

        series = download_fred_series(spec.fred_code, start_date, end_date, api_key)

        if series.empty:
            print(f"   ⚠️  Warning: No data for {spec.fred_code}")
            raw_data[obs_name] = pd.Series(dtype=float)
            continue

        # Convert to quarterly if needed
        freq = pd.infer_freq(series.index)
        if freq and ('M' in freq or 'D' in freq):
            print(f"   Converting from {freq} to quarterly...")
            series = to_quarterly(series, 'mean')

        raw_data[obs_name] = series
        print(f"   ✓ Downloaded: {len(series)} observations")

    # Apply transformations
    print("\n" + "="*80)
    print("Applying transformations...")
    print("="*80)

    transformed_data = {}

    for obs_name, series in raw_data.items():
        if series.empty:
            transformed_data[obs_name] = series
            continue

        spec = get_series_spec(obs_name)
        print(f"\n{obs_name}:")
        print(f"  Transformation: {spec.transformation}")

        # Apply transformation
        try:
            if spec.transformation == 'real_quarterly_growth_rate':
                transformed = transform_series(series, spec.transformation, deflator=gdpdef_q)
            else:
                transformed = transform_series(series, spec.transformation)

            transformed_data[obs_name] = transformed
            print(f"  ✓ Transformed: {len(transformed.dropna())} valid observations")

        except Exception as e:
            print(f"  ⚠️  Error transforming {obs_name}: {str(e)}")
            transformed_data[obs_name] = pd.Series(dtype=float)

    # Combine into DataFrame
    df = pd.DataFrame(transformed_data)

    # Align to common index (intersection of all series)
    df = df.dropna()

    print("\n" + "="*80)
    print("Data Summary")
    print("="*80)
    print(f"Start date: {df.index[0] if len(df) > 0 else 'N/A'}")
    print(f"End date: {df.index[-1] if len(df) > 0 else 'N/A'}")
    print(f"Total observations: {len(df)}")
    print(f"Variables: {len(df.columns)}")

    # Print basic statistics
    if len(df) > 0:
        print("\nSummary statistics:")
        print(df.describe())

    # Save if requested
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path)
        print(f"\n✓ Data saved to: {save_path}")

    return df


def validate_data(df: pd.DataFrame, verbose: bool = True) -> Dict[str, any]:
    """
    Validate data quality and report issues.

    Parameters
    ----------
    df : pd.DataFrame
        Data to validate
    verbose : bool
        Print validation report

    Returns
    -------
    dict
        Validation results
    """
    results = {
        'n_obs': len(df),
        'n_vars': len(df.columns),
        'missing_count': df.isnull().sum().to_dict(),
        'inf_count': np.isinf(df).sum().to_dict(),
        'mean': df.mean().to_dict(),
        'std': df.std().to_dict(),
        'min': df.min().to_dict(),
        'max': df.max().to_dict(),
    }

    if verbose:
        print("\n" + "="*80)
        print("Data Validation Report")
        print("="*80)

        print(f"\nObservations: {results['n_obs']}")
        print(f"Variables: {results['n_vars']}")

        print("\nMissing values:")
        for var, count in results['missing_count'].items():
            if count > 0:
                print(f"  {var}: {count} ({100*count/results['n_obs']:.1f}%)")

        print("\nInfinite values:")
        for var, count in results['inf_count'].items():
            if count > 0:
                print(f"  {var}: {count}")

        print("\nVariable ranges:")
        for var in df.columns:
            print(f"  {var}: [{results['min'][var]:.2f}, {results['max'][var]:.2f}]")

    return results
