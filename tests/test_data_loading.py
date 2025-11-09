"""
Tests for data loading and transformation utilities.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dsge.data.fred_loader import (
    to_quarterly,
    compute_growth_rate,
    compute_inflation_rate,
    compute_real_series,
    transform_series,
    validate_data,
)


@pytest.fixture
def sample_monthly_series():
    """Create a sample monthly time series."""
    dates = pd.date_range('2020-01-01', periods=12, freq='M')
    values = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111])
    return pd.Series(values, index=dates)


@pytest.fixture
def sample_quarterly_series():
    """Create a sample quarterly time series."""
    dates = pd.date_range('2020-01-01', periods=8, freq='Q')
    values = np.array([100, 102, 104, 106, 108, 110, 112, 114])
    return pd.Series(values, index=dates)


@pytest.fixture
def sample_price_index():
    """Create a sample price index series."""
    dates = pd.date_range('2020-01-01', periods=8, freq='Q')
    # Inflation of about 2% per quarter
    values = np.array([100.0, 102.0, 104.04, 106.12, 108.24, 110.41, 112.62, 114.87])
    return pd.Series(values, index=dates)


class TestFrequencyConversion:
    """Tests for frequency conversion functions."""

    def test_monthly_to_quarterly_mean(self, sample_monthly_series):
        """Test converting monthly to quarterly using mean."""
        quarterly = to_quarterly(sample_monthly_series, agg_method='mean')

        # Should have 4 quarters
        assert len(quarterly) == 4

        # Check that it's quarterly frequency (QE-DEC in newer pandas, Q-DEC in older)
        assert quarterly.index.freqstr in ['Q-DEC', 'QE-DEC']

        # Check first quarter average (Jan, Feb, Mar): (100 + 101 + 102) / 3
        np.testing.assert_almost_equal(quarterly.iloc[0], 101.0, decimal=5)

    def test_monthly_to_quarterly_last(self, sample_monthly_series):
        """Test converting monthly to quarterly using last."""
        quarterly = to_quarterly(sample_monthly_series, agg_method='last')

        # Should take last month of each quarter
        assert len(quarterly) == 4
        np.testing.assert_almost_equal(quarterly.iloc[0], 102.0, decimal=5)  # March

    def test_invalid_aggregation_method(self, sample_monthly_series):
        """Test that invalid aggregation method raises error."""
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            to_quarterly(sample_monthly_series, agg_method='invalid')


class TestGrowthRates:
    """Tests for growth rate calculations."""

    def test_compute_growth_rate_basic(self, sample_quarterly_series):
        """Test basic growth rate computation."""
        growth = compute_growth_rate(sample_quarterly_series, annualize=False)

        # First value should be NaN (no previous value)
        assert pd.isna(growth.iloc[0])

        # Check second value: 100 * (log(102) - log(100))
        expected = 100 * (np.log(102) - np.log(100))
        np.testing.assert_almost_equal(growth.iloc[1], expected, decimal=5)

    def test_compute_growth_rate_annualized(self, sample_quarterly_series):
        """Test annualized growth rate."""
        growth = compute_growth_rate(sample_quarterly_series, annualize=True)

        # Annualized should be 4x quarterly
        growth_q = compute_growth_rate(sample_quarterly_series, annualize=False)
        np.testing.assert_array_almost_equal(
            growth.dropna().values,
            (growth_q.dropna() * 4).values,
            decimal=5
        )

    def test_constant_series_zero_growth(self):
        """Test that constant series has zero growth rate."""
        dates = pd.date_range('2020-01-01', periods=5, freq='Q')
        constant_series = pd.Series([100, 100, 100, 100, 100], index=dates)

        growth = compute_growth_rate(constant_series, annualize=False)

        # All growth rates should be zero (except first which is NaN)
        np.testing.assert_array_almost_equal(
            growth.dropna().values,
            np.zeros(4),
            decimal=10
        )


class TestInflationRate:
    """Tests for inflation rate calculations."""

    def test_compute_inflation_rate(self, sample_price_index):
        """Test inflation rate computation."""
        inflation = compute_inflation_rate(sample_price_index, annualize=True)

        # First value should be NaN
        assert pd.isna(inflation.iloc[0])

        # Check approximate quarterly inflation rate
        # With 2% per quarter, annualized should be about 8%
        # 400 * log(102/100) ≈ 7.92%
        expected = 400 * np.log(1.02)
        np.testing.assert_almost_equal(inflation.iloc[1], expected, decimal=2)

    def test_inflation_rate_is_growth_rate(self, sample_price_index):
        """Test that inflation rate equals growth rate for price index."""
        inflation = compute_inflation_rate(sample_price_index, annualize=True)
        growth = compute_growth_rate(sample_price_index, annualize=True)

        np.testing.assert_array_almost_equal(
            inflation.dropna().values,
            growth.dropna().values,
            decimal=10
        )


class TestRealSeries:
    """Tests for real series calculations."""

    def test_compute_real_series(self):
        """Test deflating nominal series."""
        dates = pd.date_range('2020-01-01', periods=4, freq='Q')

        # Nominal series growing at 10% per quarter
        nominal = pd.Series([100, 110, 121, 133.1], index=dates)

        # Price deflator growing at 5% per quarter
        deflator = pd.Series([100, 105, 110.25, 115.76], index=dates)

        real = compute_real_series(nominal, deflator)

        # Real series should be roughly constant (nominal growth ≈ inflation)
        # Actually growing at about (1.10/1.05 - 1) ≈ 4.76% per quarter
        expected_first = 100.0
        expected_second = (110 / 105) * 100

        np.testing.assert_almost_equal(real.iloc[0], expected_first, decimal=2)
        np.testing.assert_almost_equal(real.iloc[1], expected_second, decimal=2)


class TestTransformations:
    """Tests for the main transform_series function."""

    def test_quarterly_growth_rate_transformation(self, sample_quarterly_series):
        """Test quarterly growth rate transformation."""
        result = transform_series(sample_quarterly_series, 'quarterly_growth_rate')

        # Should match compute_growth_rate output
        expected = compute_growth_rate(sample_quarterly_series, annualize=True)
        pd.testing.assert_series_equal(result, expected)

    def test_real_quarterly_growth_rate_transformation(self):
        """Test real quarterly growth rate transformation."""
        dates = pd.date_range('2020-01-01', periods=4, freq='Q')
        nominal = pd.Series([100, 110, 120, 130], index=dates)
        deflator = pd.Series([100, 105, 110, 115], index=dates)

        result = transform_series(
            nominal,
            'real_quarterly_growth_rate',
            deflator=deflator
        )

        # Should deflate first, then compute growth rate
        real = compute_real_series(nominal, deflator)
        expected = compute_growth_rate(real, annualize=True)

        pd.testing.assert_series_equal(result, expected)

    def test_real_transformation_without_deflator_raises_error(self, sample_quarterly_series):
        """Test that real transformation without deflator raises error."""
        with pytest.raises(ValueError, match="Deflator required"):
            transform_series(sample_quarterly_series, 'real_quarterly_growth_rate')

    def test_quarterly_inflation_rate_transformation(self, sample_price_index):
        """Test inflation rate transformation."""
        result = transform_series(sample_price_index, 'quarterly_inflation_rate')

        expected = compute_inflation_rate(sample_price_index, annualize=True)
        pd.testing.assert_series_equal(result, expected)

    def test_log_level_transformation(self, sample_quarterly_series):
        """Test log level transformation."""
        result = transform_series(sample_quarterly_series, 'log_level')

        expected = np.log(sample_quarterly_series)
        pd.testing.assert_series_equal(result, expected)

    def test_level_transformation(self, sample_quarterly_series):
        """Test level transformation (no-op)."""
        result = transform_series(sample_quarterly_series, 'level')

        pd.testing.assert_series_equal(result, sample_quarterly_series)

    def test_quarterly_average_on_monthly(self, sample_monthly_series):
        """Test quarterly average on monthly data."""
        result = transform_series(sample_monthly_series, 'quarterly_average')

        # Should convert to quarterly
        assert len(result) == 4

    def test_quarterly_average_on_quarterly(self, sample_quarterly_series):
        """Test quarterly average on already quarterly data."""
        result = transform_series(sample_quarterly_series, 'quarterly_average')

        # Should return as-is (check values, not dtype which may differ due to resampling)
        pd.testing.assert_series_equal(result, sample_quarterly_series, check_dtype=False)

    def test_unknown_transformation_raises_error(self, sample_quarterly_series):
        """Test that unknown transformation raises error."""
        with pytest.raises(ValueError, match="Unknown transformation"):
            transform_series(sample_quarterly_series, 'invalid_transformation')


class TestDataValidation:
    """Tests for data validation functions."""

    def test_validate_clean_data(self):
        """Test validation of clean data."""
        dates = pd.date_range('2020-01-01', periods=10, freq='Q')
        df = pd.DataFrame({
            'var1': np.random.randn(10),
            'var2': np.random.randn(10),
        }, index=dates)

        results = validate_data(df, verbose=False)

        assert results['n_obs'] == 10
        assert results['n_vars'] == 2
        assert all(count == 0 for count in results['missing_count'].values())
        assert all(count == 0 for count in results['inf_count'].values())

    def test_validate_data_with_missing(self):
        """Test validation detects missing values."""
        dates = pd.date_range('2020-01-01', periods=10, freq='Q')
        df = pd.DataFrame({
            'var1': [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
            'var2': [1, 2, 3, 4, np.nan, 6, 7, 8, 9, np.nan],
        }, index=dates)

        results = validate_data(df, verbose=False)

        assert results['missing_count']['var1'] == 1
        assert results['missing_count']['var2'] == 2

    def test_validate_data_with_inf(self):
        """Test validation detects infinite values."""
        dates = pd.date_range('2020-01-01', periods=10, freq='Q')
        df = pd.DataFrame({
            'var1': [1, 2, np.inf, 4, 5, 6, 7, 8, 9, 10],
            'var2': [1, 2, 3, 4, -np.inf, 6, 7, 8, 9, 10],
        }, index=dates)

        results = validate_data(df, verbose=False)

        assert results['inf_count']['var1'] == 1
        assert results['inf_count']['var2'] == 1

    def test_validate_data_statistics(self):
        """Test that validation computes correct statistics."""
        dates = pd.date_range('2020-01-01', periods=5, freq='Q')
        df = pd.DataFrame({
            'var1': [1, 2, 3, 4, 5],
        }, index=dates)

        results = validate_data(df, verbose=False)

        assert results['mean']['var1'] == 3.0
        assert results['min']['var1'] == 1.0
        assert results['max']['var1'] == 5.0
        np.testing.assert_almost_equal(results['std']['var1'], np.std([1, 2, 3, 4, 5], ddof=1), decimal=5)


class TestSeriesMapping:
    """Tests for FRED series mapping."""

    def test_all_observables_have_mapping(self):
        """Test that all NYFed observables have FRED mappings."""
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))
        from fred_series_mapping import FRED_SERIES_MAP

        # Should have 13 observables
        assert len(FRED_SERIES_MAP) == 13

        # Check key observables exist
        required_obs = [
            'obs_gdp_growth', 'obs_gdi_growth', 'obs_cons_growth', 'obs_inv_growth',
            'obs_wage_growth', 'obs_hours', 'obs_infl_pce', 'obs_infl_gdpdef',
            'obs_ffr', 'obs_10y_rate', 'obs_10y_infl_exp', 'obs_spread', 'obs_tfp_growth'
        ]

        for obs in required_obs:
            assert obs in FRED_SERIES_MAP, f"Missing observable: {obs}"

    def test_series_spec_has_required_fields(self):
        """Test that each series spec has all required fields."""
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))
        from fred_series_mapping import FRED_SERIES_MAP

        for obs_name, spec in FRED_SERIES_MAP.items():
            assert hasattr(spec, 'fred_code'), f"{obs_name} missing fred_code"
            assert hasattr(spec, 'description'), f"{obs_name} missing description"
            assert hasattr(spec, 'units'), f"{obs_name} missing units"
            assert hasattr(spec, 'transformation'), f"{obs_name} missing transformation"

            # FRED codes should be non-empty strings
            assert isinstance(spec.fred_code, str) and len(spec.fred_code) > 0

    def test_transformations_are_valid(self):
        """Test that all transformations are valid."""
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))
        from fred_series_mapping import FRED_SERIES_MAP, TRANSFORMATION_FUNCTIONS

        valid_transformations = set(TRANSFORMATION_FUNCTIONS.keys())

        for obs_name, spec in FRED_SERIES_MAP.items():
            assert spec.transformation in valid_transformations, \
                f"{obs_name} has invalid transformation: {spec.transformation}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])
