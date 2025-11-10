"""Tests for forecasting utilities."""

import os
import sys

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dsge.forecasting import (
    compute_forecast_bands,
    conditional_forecast,
    forecast_observables,
    forecast_states,
)


@pytest.fixture
def simple_ar1_system():
    """Create a simple AR(1) state space system."""
    # x_t = 0.9 * x_{t-1} + eps_t
    # y_t = x_t + noise
    T = np.array([[0.9]])
    R = np.array([[1.0]])
    C = np.array([0.0])
    Z = np.array([[1.0]])
    D = np.array([0.0])

    x0 = np.array([0.0])

    return {"T": T, "R": R, "C": C, "Z": Z, "D": D, "x0": x0}


@pytest.fixture
def var_system():
    """Create a simple VAR(1) system."""
    # Two-variable VAR
    T = np.array([[0.8, 0.1], [0.1, 0.7]])
    R = np.array([[1.0, 0.0], [0.0, 1.0]])
    C = np.zeros(2)
    Z = np.eye(2)
    D = np.zeros(2)

    x0 = np.zeros(2)

    return {"T": T, "R": R, "C": C, "Z": Z, "D": D, "x0": x0}


class TestStateForecast:
    """Tests for state forecasting."""

    def test_forecast_states_shape(self, simple_ar1_system) -> None:
        """Test that forecast_states returns correct shapes."""
        sys = simple_ar1_system

        result = forecast_states(
            T=sys["T"], R=sys["R"], C=sys["C"], x_T=sys["x0"], horizon=10, n_paths=100, seed=42
        )

        # Mean forecast should be (horizon x n_states)
        assert result.mean.shape == (10, 1)

        # Paths should be (n_paths x horizon x n_states)
        assert result.paths.shape == (100, 10, 1)

        # Bands should be dict with (lower, upper) tuples
        assert isinstance(result.bands, dict)
        assert 0.90 in result.bands
        assert 0.68 in result.bands

        # Each band should have (horizon x n_states) shape
        lower, upper = result.bands[0.90]
        assert lower.shape == (10, 1)
        assert upper.shape == (10, 1)

    def test_forecast_states_ar1_decay(self, simple_ar1_system) -> None:
        """Test that AR(1) forecast decays as expected."""
        sys = simple_ar1_system

        # Start from x_0 = 1.0
        x0 = np.array([1.0])

        result = forecast_states(
            T=sys["T"],
            R=sys["R"],
            C=sys["C"],
            x_T=x0,
            horizon=10,
            n_paths=10000,  # Many paths for accurate mean
            seed=42,
        )

        # For AR(1) with phi=0.9 and no shocks (on average),
        # mean forecast should be approximately 0.9^h * x_0
        for h in range(10):
            expected = 0.9**h * x0[0]
            actual = result.mean[h, 0]

            # Allow some tolerance due to random shocks
            assert abs(actual - expected) < 0.1, (
                f"At horizon {h}, expected {expected:.3f}, got {actual:.3f}"
            )

    def test_forecast_reproducibility(self, simple_ar1_system) -> None:
        """Test that forecasts are reproducible with same seed."""
        sys = simple_ar1_system

        result1 = forecast_states(
            T=sys["T"], R=sys["R"], C=sys["C"], x_T=sys["x0"], horizon=5, n_paths=10, seed=42
        )

        result2 = forecast_states(
            T=sys["T"], R=sys["R"], C=sys["C"], x_T=sys["x0"], horizon=5, n_paths=10, seed=42
        )

        # Results should be identical
        np.testing.assert_array_equal(result1.paths, result2.paths)
        np.testing.assert_array_equal(result1.mean, result2.mean)


class TestObservableForecast:
    """Tests for observable forecasting."""

    def test_forecast_observables_shape(self, simple_ar1_system) -> None:
        """Test observable forecast shapes."""
        sys = simple_ar1_system

        result = forecast_observables(
            T=sys["T"],
            R=sys["R"],
            C=sys["C"],
            Z=sys["Z"],
            D=sys["D"],
            x_T=sys["x0"],
            horizon=10,
            n_paths=100,
            seed=42,
        )

        n_obs = sys["Z"].shape[0]

        # Check shapes
        assert result.mean.shape == (10, n_obs)
        assert result.paths.shape == (100, 10, n_obs)

        # Check bands
        lower, upper = result.bands[0.90]
        assert lower.shape == (10, n_obs)
        assert upper.shape == (10, n_obs)

    def test_forecast_observables_matches_states(self, simple_ar1_system) -> None:
        """Test that observable forecast matches state forecast."""
        sys = simple_ar1_system

        # For this system, Z = I and D = 0, so observables = states
        state_result = forecast_states(
            T=sys["T"], R=sys["R"], C=sys["C"], x_T=sys["x0"], horizon=10, n_paths=100, seed=42
        )

        obs_result = forecast_observables(
            T=sys["T"],
            R=sys["R"],
            C=sys["C"],
            Z=sys["Z"],
            D=sys["D"],
            x_T=sys["x0"],
            horizon=10,
            n_paths=100,
            seed=42,
        )

        # Observables should match states
        np.testing.assert_array_almost_equal(obs_result.mean, state_result.mean, decimal=10)

    def test_forecast_with_measurement_error(self, simple_ar1_system) -> None:
        """Test forecast with measurement error."""
        sys = simple_ar1_system

        # Forecast with measurement error
        meas_cov = np.array([[0.1]])

        result_with_error = forecast_observables(
            T=sys["T"],
            R=sys["R"],
            C=sys["C"],
            Z=sys["Z"],
            D=sys["D"],
            x_T=sys["x0"],
            horizon=10,
            n_paths=100,
            measurement_cov=meas_cov,
            seed=42,
        )

        # Forecast without measurement error
        result_no_error = forecast_observables(
            T=sys["T"],
            R=sys["R"],
            C=sys["C"],
            Z=sys["Z"],
            D=sys["D"],
            x_T=sys["x0"],
            horizon=10,
            n_paths=100,
            seed=42,
        )

        # With measurement error, variance should be larger
        var_with_error = np.var(result_with_error.paths, axis=0)
        var_no_error = np.var(result_no_error.paths, axis=0)

        # Check that variance increased (at least for some periods)
        assert np.any(var_with_error > var_no_error)


class TestUncertaintyBands:
    """Tests for uncertainty band computation."""

    def test_compute_forecast_bands_shape(self) -> None:
        """Test that bands have correct shape."""
        # Create sample paths (100 paths, 10 periods, 2 variables)
        paths = np.random.randn(100, 10, 2)

        bands = compute_forecast_bands(paths, confidence_levels=[0.68, 0.90, 0.95])

        # Check that all confidence levels are present
        assert len(bands) == 3
        assert 0.68 in bands
        assert 0.90 in bands
        assert 0.95 in bands

        # Check shapes
        for _level, (lower, upper) in bands.items():
            assert lower.shape == (10, 2)
            assert upper.shape == (10, 2)

    def test_forecast_bands_ordering(self) -> None:
        """Test that uncertainty bands are properly ordered."""
        paths = np.random.randn(1000, 10, 1)

        bands = compute_forecast_bands(paths, confidence_levels=[0.68, 0.90])

        lower_68, upper_68 = bands[0.68]
        lower_90, upper_90 = bands[0.90]

        # 90% band should be wider than 68% band
        assert np.all(lower_90 <= lower_68)
        assert np.all(upper_90 >= upper_68)

    def test_forecast_bands_contain_mean(self) -> None:
        """Test that mean is within uncertainty bands."""
        paths = np.random.randn(1000, 10, 2)
        mean = np.mean(paths, axis=0)

        bands = compute_forecast_bands(paths, confidence_levels=[0.90])

        lower, upper = bands[0.90]

        # Mean should be within 90% bands (approximately, allowing tolerance for sampling)
        within_bands = np.sum((mean >= lower) & (mean <= upper))
        total_elements = mean.size

        # At least 80% should be within bands (conservative due to sampling variance)
        assert within_bands / total_elements > 0.8


class TestConditionalForecast:
    """Tests for conditional forecasting."""

    def test_conditional_forecast_basic(self, simple_ar1_system) -> None:
        """Test basic conditional forecast."""
        sys = simple_ar1_system

        # Condition on observable being close to 0.5 in first period
        conditions = {0: {0: 0.5}}

        result = conditional_forecast(
            T=sys["T"],
            R=sys["R"],
            C=sys["C"],
            Z=sys["Z"],
            D=sys["D"],
            x_T=sys["x0"],
            horizon=5,
            conditions=conditions,
            n_paths=100,
            seed=42,
        )

        # Check that first period mean is close to condition
        assert abs(result.mean[0, 0] - 0.5) < 0.05

    def test_conditional_forecast_shape(self, var_system) -> None:
        """Test conditional forecast with multiple variables."""
        sys = var_system

        # Use more modest conditions (closer to expected values)
        # Start from x0 = [0, 0], so early forecasts will be close to 0
        conditions = {
            0: {0: 0.1},  # Constrain first variable to small value in period 0
        }

        result = conditional_forecast(
            T=sys["T"],
            R=sys["R"],
            C=sys["C"],
            Z=sys["Z"],
            D=sys["D"],
            x_T=sys["x0"],
            horizon=10,
            conditions=conditions,
            n_paths=50,  # Fewer paths for testing
            seed=42,
        )

        # Check shapes
        assert result.mean.shape == (10, 2)
        assert result.paths.shape[1:] == (10, 2)

        # Check that condition is approximately satisfied
        assert abs(result.mean[0, 0] - 0.1) < 0.05


class TestForecastIntegration:
    """Integration tests for forecasting."""

    def test_forecast_with_var_system(self, var_system) -> None:
        """Test full forecast workflow with VAR system."""
        sys = var_system

        # Set initial state
        x0 = np.array([1.0, -0.5])

        # Generate forecast
        result = forecast_observables(
            T=sys["T"],
            R=sys["R"],
            C=sys["C"],
            Z=sys["Z"],
            D=sys["D"],
            x_T=x0,
            horizon=20,
            n_paths=500,
            seed=42,
        )

        # Check that forecast eventually converges to zero (stable system)
        final_mean = result.mean[-1]
        assert np.all(np.abs(final_mean) < 0.5)  # Should be close to steady state

        # Check that uncertainty bands widen over time (initially)
        lower_68, upper_68 = result.bands[0.68]

        width_period_0 = upper_68[0] - lower_68[0]
        width_period_5 = upper_68[5] - lower_68[5]

        # Early periods should have wider bands than initial
        assert np.all(width_period_5 >= width_period_0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
