"""Tests for Kalman filter."""

import numpy as np
import pytest
from dsge.filters.kalman import kalman_filter, kalman_smoother


def test_kalman_filter_ar1():
    """Test Kalman filter on AR(1) model."""
    # Simulate AR(1) data
    T_periods = 100
    rho = 0.9
    sigma = 1.0

    np.random.seed(42)
    states = np.zeros((T_periods, 1))
    for t in range(1, T_periods):
        states[t] = rho * states[t-1] + np.random.normal(0, sigma, 1)

    obs = states + np.random.normal(0, 0.1, (T_periods, 1))

    # Filter
    T = np.array([[rho]])
    R = np.array([[1.0]])
    Q = np.array([[sigma**2]])
    Z = np.array([[1.0]])
    D = np.array([0.0])
    H = np.array([[0.01]])

    results = kalman_filter(obs, T, R, Q, Z, D, H)

    assert results.log_likelihood < 0  # Negative log likelihood
    assert np.isfinite(results.log_likelihood)
    assert results.filtered_states.shape == (T_periods, 1)
    assert results.predicted_states.shape == (T_periods, 1)


def test_kalman_filter_missing_data():
    """Test Kalman filter with missing observations."""
    T_periods = 50
    rho = 0.8
    sigma = 1.0

    np.random.seed(42)
    states = np.zeros((T_periods, 1))
    for t in range(1, T_periods):
        states[t] = rho * states[t-1] + np.random.normal(0, sigma, 1)

    obs = states + np.random.normal(0, 0.1, (T_periods, 1))

    # Add missing values
    obs[10:15] = np.nan

    # Filter
    T = np.array([[rho]])
    R = np.array([[1.0]])
    Q = np.array([[sigma**2]])
    Z = np.array([[1.0]])
    D = np.array([0.0])
    H = np.array([[0.01]])

    results = kalman_filter(obs, T, R, Q, Z, D, H)

    assert np.isfinite(results.log_likelihood)
    # Filtered states should be finite even for missing obs
    assert np.isfinite(results.filtered_states).all()


def test_kalman_smoother():
    """Test Kalman smoother."""
    T_periods = 50
    rho = 0.9
    sigma = 1.0

    np.random.seed(42)
    states = np.zeros((T_periods, 1))
    for t in range(1, T_periods):
        states[t] = rho * states[t-1] + np.random.normal(0, sigma, 1)

    obs = states + np.random.normal(0, 0.1, (T_periods, 1))

    # Filter
    T = np.array([[rho]])
    R = np.array([[1.0]])
    Q = np.array([[sigma**2]])
    Z = np.array([[1.0]])
    D = np.array([0.0])
    H = np.array([[0.01]])

    filter_results = kalman_filter(obs, T, R, Q, Z, D, H)

    # Smooth
    smoothed_states, smoothed_cov = kalman_smoother(filter_results, T, R, Q)

    assert smoothed_states.shape == (T_periods, 1)
    assert smoothed_cov.shape == (T_periods, 1, 1)
    assert np.isfinite(smoothed_states).all()
