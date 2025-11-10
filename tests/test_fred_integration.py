"""
Integration tests for FRED API data downloading.

These tests require a valid FRED API key to run. They will be skipped
if no API key is found in the environment variables or .env file.

To run these tests, set FRED_API_KEY in your environment or create a .env file:
    export FRED_API_KEY="your_api_key_here"
    # or
    echo "FRED_API_KEY=your_api_key_here" > .env
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dsge.config import get_fred_api_key, get_settings
from dsge.data.fred_loader import (
    download_fred_series,
    load_nyfed_data,
)


# Fixture to check if FRED API key is available
@pytest.fixture(scope="module")
def fred_api_key():
    """Get FRED API key from environment/settings."""
    return get_fred_api_key()


@pytest.fixture(scope="module")
def has_fred_api_key(fred_api_key):
    """Check if FRED API key is available."""
    return fred_api_key is not None and len(fred_api_key) > 0


# Mark to skip tests if no API key
skip_if_no_api_key = pytest.mark.skipif(
    get_fred_api_key() is None,
    reason="FRED API key not found in environment. Set FRED_API_KEY to run this test.",
)


class TestFREDConfig:
    """Tests for FRED configuration loading."""

    def test_settings_load(self) -> None:
        """Test that settings can be loaded."""
        settings = get_settings()
        assert settings is not None

    def test_fred_api_key_type(self, fred_api_key) -> None:
        """Test that FRED API key is correct type."""
        # Key can be None or a string
        assert fred_api_key is None or isinstance(fred_api_key, str)

    def test_get_fred_api_key_function(self) -> None:
        """Test that get_fred_api_key returns consistent value."""
        key1 = get_fred_api_key()
        key2 = get_fred_api_key()
        assert key1 == key2


class TestFREDDownloadBasic:
    """Basic tests for FRED download functionality (without API calls)."""

    def test_download_without_api_key_returns_empty(self, monkeypatch) -> None:
        """Test that download without API key returns empty series."""
        # Remove API key from environment
        monkeypatch.delenv("FRED_API_KEY", raising=False)

        # Reload settings without .env file
        from dsge.config import reload_settings

        reload_settings(env_file=None)

        # Try to download without providing API key
        # (will fail gracefully if no key in environment)
        with pytest.warns(UserWarning, match="No FRED API key provided"):
            series = download_fred_series("GDPC1", api_key=None)
            # Should return empty series if no key available
            assert isinstance(series, pd.Series)
            assert len(series) == 0

        # Restore settings with .env file for subsequent tests
        reload_settings()


@skip_if_no_api_key
class TestFREDDownloadIntegration:
    """Integration tests that actually download data from FRED.

    These tests require a valid FRED API key and will be skipped if not available.
    """

    def test_download_gdp_series(self, fred_api_key) -> None:
        """Test downloading real GDP series."""
        series = download_fred_series(
            "GDPC1", start_date="2020-01-01", end_date="2020-12-31", api_key=fred_api_key
        )

        # Check that we got data
        assert isinstance(series, pd.Series)
        assert len(series) > 0

        # Check index is datetime
        assert isinstance(series.index, pd.DatetimeIndex)

        # Check values are numeric
        assert pd.api.types.is_numeric_dtype(series)

        # GDP should be positive
        assert (series > 0).all()

    def test_download_inflation_series(self, fred_api_key) -> None:
        """Test downloading PCE inflation series."""
        series = download_fred_series(
            "PCEPI", start_date="2020-01-01", end_date="2020-12-31", api_key=fred_api_key
        )

        assert isinstance(series, pd.Series)
        assert len(series) > 0
        assert (series > 0).all()  # Price index should be positive

    def test_download_interest_rate_series(self, fred_api_key) -> None:
        """Test downloading federal funds rate."""
        series = download_fred_series(
            "FEDFUNDS", start_date="2020-01-01", end_date="2020-12-31", api_key=fred_api_key
        )

        assert isinstance(series, pd.Series)
        assert len(series) > 0
        assert (series >= 0).all()  # Interest rates should be non-negative

    def test_download_with_date_range(self, fred_api_key) -> None:
        """Test that date range filtering works."""
        series = download_fred_series(
            "GDPC1", start_date="2015-01-01", end_date="2015-12-31", api_key=fred_api_key
        )

        assert len(series) > 0

        # All dates should be in 2015
        assert series.index.year.min() == 2015
        assert series.index.year.max() == 2015

    def test_download_invalid_series_returns_empty(self, fred_api_key) -> None:
        """Test that invalid series ID returns empty series with warning."""
        with pytest.warns(UserWarning, match="Failed to download"):
            series = download_fred_series(
                "INVALID_SERIES_XYZ123", start_date="2020-01-01", api_key=fred_api_key
            )

            assert isinstance(series, pd.Series)
            assert len(series) == 0

    def test_download_uses_environment_key_if_none_provided(self, fred_api_key) -> None:
        """Test that API key from environment is used when None provided."""
        # This should use the key from environment/settings
        series = download_fred_series(
            "GDPC1",
            start_date="2020-01-01",
            end_date="2020-12-31",
            api_key=None,  # Should fall back to environment
        )

        # Should succeed if environment key is valid
        assert isinstance(series, pd.Series)
        assert len(series) > 0


@skip_if_no_api_key
class TestNYFedDataLoading:
    """Integration tests for loading complete NYFed DSGE model dataset.

    These tests require a valid FRED API key and will be skipped if not available.
    Note: These tests may be slow as they download 13 series from FRED.
    """

    @pytest.mark.slow
    def test_load_nyfed_data_short_sample(self, fred_api_key, tmp_path) -> None:
        """Test loading NYFed data for a short sample period."""
        # Use a short time period to speed up test
        save_path = tmp_path / "nyfed_data_test.csv"

        df = load_nyfed_data(
            start_date="2019-01-01",
            end_date="2020-12-31",
            api_key=fred_api_key,
            save_path=str(save_path),
        )

        # Check that we got a DataFrame
        assert isinstance(df, pd.DataFrame)

        # Should have some observations
        assert len(df) > 0

        # Check that file was saved
        assert save_path.exists()

        # Load saved file and verify it matches
        df_loaded = pd.read_csv(save_path, index_col=0, parse_dates=True)
        pd.testing.assert_frame_equal(df, df_loaded)

    @pytest.mark.slow
    def test_load_nyfed_data_has_expected_columns(self, fred_api_key) -> None:
        """Test that loaded data has expected observable columns."""
        df = load_nyfed_data(start_date="2019-01-01", end_date="2020-12-31", api_key=fred_api_key)

        # Core observables that should always be available
        core_columns = [
            "obs_gdp_growth",
            "obs_gdi_growth",
            "obs_cons_growth",
            "obs_inv_growth",
            "obs_wage_growth",
            "obs_hours",
            "obs_infl_pce",
            "obs_infl_gdpdef",
            "obs_ffr",
            "obs_10y_rate",
            "obs_10y_infl_exp",
            "obs_spread",
        ]

        # Optional observables that may not be available (e.g., obs_tfp_growth/TFPKQ doesn't exist in FRED)

        # Check core columns are present
        for col in core_columns:
            assert col in df.columns, f"Missing core column: {col}"

        # Should have at least the core columns
        assert len(df.columns) >= len(core_columns)

    @pytest.mark.slow
    def test_load_nyfed_data_quarterly_frequency(self, fred_api_key) -> None:
        """Test that loaded data is at quarterly frequency."""
        df = load_nyfed_data(start_date="2019-01-01", end_date="2020-12-31", api_key=fred_api_key)

        if len(df) > 0:
            # Check frequency (should be quarterly)
            freq = pd.infer_freq(df.index)
            # Could be 'Q-DEC', 'QE-DEC', 'Q', or 'QE' depending on pandas version
            assert freq is None or "Q" in freq, f"Expected quarterly frequency, got: {freq}"

    @pytest.mark.slow
    def test_load_nyfed_data_no_missing_values(self, fred_api_key) -> None:
        """Test that loaded data has no missing values (after alignment)."""
        df = load_nyfed_data(start_date="2019-01-01", end_date="2020-12-31", api_key=fred_api_key)

        # After alignment, there should be no missing values
        assert df.isnull().sum().sum() == 0

    @pytest.mark.slow
    def test_load_nyfed_data_no_infinite_values(self, fred_api_key) -> None:
        """Test that loaded data has no infinite values."""
        df = load_nyfed_data(start_date="2019-01-01", end_date="2020-12-31", api_key=fred_api_key)

        # Check for infinite values
        assert not np.isinf(df.values).any()

    @pytest.mark.slow
    def test_load_nyfed_data_reasonable_ranges(self, fred_api_key) -> None:
        """Test that loaded data values are in reasonable ranges."""
        df = load_nyfed_data(start_date="2019-01-01", end_date="2020-12-31", api_key=fred_api_key)

        if len(df) > 0:
            # Growth rates should typically be between -50% and +50% annualized
            growth_cols = [col for col in df.columns if "growth" in col]
            for col in growth_cols:
                assert df[col].abs().max() < 100, f"{col} has unreasonable values"

            # Interest rates should be between -5% and 20%
            rate_cols = [col for col in df.columns if "rate" in col or "ffr" in col]
            for col in rate_cols:
                assert df[col].min() > -10, f"{col} has unreasonable negative values"
                assert df[col].max() < 30, f"{col} has unreasonably high values"


class TestFREDIntegrationWithoutKey:
    """Tests for FRED functionality when API key is not available."""

    def test_settings_without_key(self, monkeypatch) -> None:
        """Test that settings work even without API key."""
        # Remove API key from environment
        monkeypatch.delenv("FRED_API_KEY", raising=False)

        # Reload settings without .env file
        from dsge.config import reload_settings

        settings = reload_settings(env_file=None)

        # Should still load, just with None for API key
        assert settings.fred_api_key is None

        # Restore settings with .env file for subsequent tests
        reload_settings()

    def test_download_without_key_fails_gracefully(self, monkeypatch) -> None:
        """Test that download fails gracefully without API key."""
        # Remove API key
        monkeypatch.delenv("FRED_API_KEY", raising=False)

        # Reload settings without .env file to ensure no API key is available
        from dsge.config import reload_settings

        reload_settings(env_file=None)

        # Should warn and return empty series
        with pytest.warns(UserWarning):
            series = download_fred_series("GDPC1")
            assert isinstance(series, pd.Series)
            assert len(series) == 0

        # Restore settings with .env file for subsequent tests
        reload_settings()


# Pytest configuration for this module
def pytest_configure(config) -> None:
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
