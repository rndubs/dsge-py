"""
FRED Data Loader for DSGE Estimation.

This module provides utilities for downloading and transforming economic data
from FRED (Federal Reserve Economic Data) for use in DSGE model estimation.
"""

import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas import DatetimeIndex


def download_fred_series(
    series_id: str,
    start_date: str = "1960-01-01",
    end_date: str | None = None,
    api_key: str | None = None,
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

    Returns:
    -------
    pd.Series
        Downloaded series with datetime index
    """
    try:
        from fredapi import Fred
    except ImportError:
        msg = "fredapi package required for FRED data download. Install with: uv add fredapi"
        raise ImportError(
            msg
        )

    # Initialize FRED API
    if api_key is None:
        # Try to get from settings (which loads from .env or environment variable)
        from dsge.config import get_fred_api_key

        api_key = get_fred_api_key()
        if api_key is None:
            warnings.warn(
                "No FRED API key provided. Set FRED_API_KEY in .env file or "
                "environment variable, or pass api_key parameter. "
                "You can get a free API key at "
                "https://fred.stlouisfed.org/docs/api/api_key.html", stacklevel=2
            )
            # Return empty series if no API key
            return pd.Series(dtype=float)

    fred = Fred(api_key=api_key)

    # Download series
    try:
        return fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    except Exception as e:
        warnings.warn(f"Failed to download {series_id}: {e!s}", stacklevel=2)
        return pd.Series(dtype=float)


def to_quarterly(series: pd.Series, agg_method: str = "mean") -> pd.Series:
    """
    Convert monthly or daily data to quarterly frequency.

    Parameters
    ----------
    series : pd.Series
        Input series with datetime index
    agg_method : str
        Aggregation method: 'mean', 'last', 'sum'

    Returns:
    -------
    pd.Series
        Quarterly series
    """
    if agg_method == "mean":
        return series.resample("Q").mean()
    if agg_method == "last":
        return series.resample("Q").last()
    if agg_method == "sum":
        return series.resample("Q").sum()
    msg = f"Unknown aggregation method: {agg_method}"
    raise ValueError(msg)


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

    Returns:
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

    Returns:
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

    Returns:
    -------
    pd.Series
        Real series
    """
    # Normalize both series to use the same quarter period (to handle different
    # quarter-end conventions like Q vs QE or start-of-quarter vs end-of-quarter)
    if isinstance(nominal.index, pd.DatetimeIndex) and isinstance(deflator.index, pd.DatetimeIndex):
        # Convert both to PeriodIndex with quarterly frequency for alignment
        nominal_period = nominal.copy()
        deflator_period = deflator.copy()

        nominal_period.index = nominal.index.to_period("Q")
        deflator_period.index = deflator.index.to_period("Q")

        # Align by period (this will match 2019Q1 regardless of whether it's 2019-01-01 or 2019-03-31)
        aligned_nominal, aligned_deflator = nominal_period.align(deflator_period, join="inner")

        if len(aligned_nominal) == 0:
            warnings.warn(
                f"No overlapping quarters between nominal series and deflator. "
                f"Nominal quarters: {nominal_period.index[[0, -1]].tolist() if len(nominal_period) > 0 else 'empty'}, "
                f"Deflator quarters: {deflator_period.index[[0, -1]].tolist() if len(deflator_period) > 0 else 'empty'}", stacklevel=2
            )
            return pd.Series(dtype=float)

        # Compute real series
        real_series = (aligned_nominal / aligned_deflator) * 100

        # Convert back to timestamp index (using end of quarter)
        real_series.index = real_series.index.to_timestamp(how="end")

        return real_series
    # Fallback for non-datetime indices
    aligned_nominal, aligned_deflator = nominal.align(deflator, join="inner")
    if len(aligned_nominal) == 0:
        return pd.Series(dtype=float)
    return (aligned_nominal / aligned_deflator) * 100


def transform_series(
    series: pd.Series, transformation: str, deflator: pd.Series | None = None
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

    Returns:
    -------
    pd.Series
        Transformed series
    """
    if transformation == "quarterly_growth_rate":
        return compute_growth_rate(series, annualize=True)

    if transformation == "real_quarterly_growth_rate":
        if deflator is None:
            msg = "Deflator required for real_quarterly_growth_rate transformation"
            raise ValueError(msg)
        if deflator.empty:
            warnings.warn("Deflator is empty, returning empty series for real transformation", stacklevel=2)
            return pd.Series(dtype=float)
        real_series = compute_real_series(series, deflator)
        return compute_growth_rate(real_series, annualize=True)

    if transformation == "quarterly_inflation_rate":
        return compute_inflation_rate(series, annualize=True)

    if transformation == "quarterly_average":
        # If monthly, convert to quarterly
        freq = pd.infer_freq(cast("DatetimeIndex", series.index))
        if freq and ("M" in freq or "D" in freq):
            return to_quarterly(series, agg_method="mean")
        return series

    if transformation == "log_level":
        return np.log(series)

    if transformation == "level":
        return series

    msg = f"Unknown transformation: {transformation}"
    raise ValueError(msg)


def load_nyfed_data(
    start_date: str = "1960-01-01",
    end_date: str | None = None,
    api_key: str | None = None,
    save_path: str | None = None,
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

    Returns:
    -------
    pd.DataFrame
        Dataframe with all 13 observables as columns, quarterly frequency
    """
    # Import the series mapping
    import sys

    # Add data directory to path
    # Path(__file__) = .../src/dsge/data/fred_loader.py
    # .parent.parent.parent.parent = project root
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    sys.path.insert(0, str(data_dir))

    try:
        from fred_series_mapping import FRED_SERIES_MAP, get_series_spec  # type: ignore[import-untyped]
    except ImportError:
        msg = "Could not import fred_series_mapping. Check data/ directory."
        raise ImportError(msg)


    # Download raw series
    raw_data = {}

    # First, download GDP deflator (needed for real transformations)
    gdpdef = download_fred_series("GDPDEF", start_date, end_date, api_key)
    gdpdef_q = pd.Series() if gdpdef.empty else to_quarterly(gdpdef, "mean")

    # Download all series
    for _i, (obs_name, spec) in enumerate(FRED_SERIES_MAP.items(), 1):

        series = download_fred_series(spec.fred_code, start_date, end_date, api_key)

        if series.empty:
            raw_data[obs_name] = pd.Series(dtype=float)
            continue


        # Convert to quarterly if needed
        freq = pd.infer_freq(cast("DatetimeIndex", series.index))
        if freq and ("M" in freq or "D" in freq or "B" in freq):
            series = to_quarterly(series, "mean")
        elif (
            len(series) > 10
        ):  # If we have many observations but freq not detected, likely daily/monthly
            series = to_quarterly(series, "mean")

        raw_data[obs_name] = series

    # Apply transformations

    transformed_data = {}

    for obs_name, series in raw_data.items():
        if series.empty:
            transformed_data[obs_name] = series
            continue

        spec = get_series_spec(obs_name)

        # Apply transformation
        try:
            if spec.transformation == "real_quarterly_growth_rate":
                transformed = transform_series(series, spec.transformation, deflator=gdpdef_q)
            else:
                transformed = transform_series(series, spec.transformation)

            transformed_data[obs_name] = transformed

        except Exception:
            transformed_data[obs_name] = pd.Series(dtype=float)

    # Normalize all series to use the same quarter period before combining
    # (to handle QS-OCT vs Q-DEC differences)
    normalized_data = {}
    for obs_name, series in transformed_data.items():
        if not series.empty and isinstance(series.index, pd.DatetimeIndex):
            # Convert to PeriodIndex then back to timestamp with consistent convention
            series_normalized = series.copy()
            series_normalized.index = series.index.to_period("Q").to_timestamp(how="end")
            normalized_data[obs_name] = series_normalized
        else:
            normalized_data[obs_name] = series

    # Combine into DataFrame
    df = pd.DataFrame(normalized_data)

    # Ensure index is DatetimeIndex (not just Index)
    if len(df) > 0 and not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)

    # Remove columns that are entirely NaN (e.g., series that failed to download)
    df = df.dropna(axis=1, how="all")

    # Align to common index (intersection of all series with data)
    df = df.dropna()


    # Print basic statistics
    if len(df) > 0:
        pass

    # Save if requested
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path)

    return df


def validate_data(df: pd.DataFrame, verbose: bool = True) -> dict[str, Any]:
    """
    Validate data quality and report issues.

    Parameters
    ----------
    df : pd.DataFrame
        Data to validate
    verbose : bool
        Print validation report

    Returns:
    -------
    dict
        Validation results
    """
    results = {
        "n_obs": len(df),
        "n_vars": len(df.columns),
        "missing_count": df.isnull().sum().to_dict(),
        "inf_count": np.isinf(df).sum().to_dict(),
        "mean": df.mean().to_dict(),
        "std": df.std().to_dict(),
        "min": df.min().to_dict(),
        "max": df.max().to_dict(),
    }

    if verbose:


        for _var, count in cast("dict[str, Any]", results["missing_count"]).items():
            if count > 0:
                pass

        for _var, count in cast("dict[str, Any]", results["inf_count"]).items():
            if count > 0:
                pass

        for _var in df.columns:
            pass

    return results
