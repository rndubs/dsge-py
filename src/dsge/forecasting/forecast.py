"""Forecasting functions for DSGE models."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ForecastResult:
    """Container for forecast results."""

    mean: np.ndarray
    bands: dict[float, tuple[np.ndarray, np.ndarray]] | None = None
    paths: np.ndarray | None = None


def forecast_states(
    T: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    x_T: np.ndarray,
    horizon: int,
    n_paths: int = 1000,
    shock_cov: np.ndarray | None = None,
    seed: int | None = None,
) -> ForecastResult:
    """
    Forecast state variables from a DSGE model.

    Parameters
    ----------
    T : np.ndarray
        State transition matrix (n_states x n_states)
    R : np.ndarray
        Shock impact matrix (n_states x n_shocks)
    C : np.ndarray
        Constant vector (n_states,)
    x_T : np.ndarray
        Initial state at time T (n_states,)
    horizon : int
        Forecast horizon in periods
    n_paths : int
        Number of simulation paths for uncertainty bands
    shock_cov : np.ndarray, optional
        Shock covariance matrix (default: identity)
    seed : int, optional
        Random seed for reproducibility

    Returns:
    -------
    ForecastResult
        Forecast results with mean, bands, and paths
    """
    if seed is not None:
        np.random.seed(seed)

    n_states = T.shape[0]
    n_shocks = R.shape[1]

    if shock_cov is None:
        shock_cov = np.eye(n_shocks)

    # Generate shock paths
    shocks = np.random.multivariate_normal(
        mean=np.zeros(n_shocks), cov=shock_cov, size=(n_paths, horizon)
    )

    # Simulate paths
    paths = np.zeros((n_paths, horizon, n_states))

    for i in range(n_paths):
        x = x_T.copy()

        for h in range(horizon):
            x = C + T @ x + R @ shocks[i, h]
            paths[i, h] = x

    # Compute mean forecast
    mean_forecast = np.mean(paths, axis=0)

    # Compute uncertainty bands
    percentiles = [0.05, 0.16, 0.84, 0.95]  # 90% and 68% bands
    bands = {}

    for p in percentiles:
        bands[p] = np.percentile(paths, p * 100, axis=0)

    # Convert to lower/upper format
    band_dict = {
        0.90: (bands[0.05], bands[0.95]),
        0.68: (bands[0.16], bands[0.84]),
    }

    return ForecastResult(mean=mean_forecast, bands=band_dict, paths=paths)


def forecast_observables(
    T: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    Z: np.ndarray,
    D: np.ndarray,
    x_T: np.ndarray,
    horizon: int,
    n_paths: int = 1000,
    shock_cov: np.ndarray | None = None,
    measurement_cov: np.ndarray | None = None,
    seed: int | None = None,
) -> ForecastResult:
    """
    Forecast observable variables from a DSGE model.

    Parameters
    ----------
    T, R, C : np.ndarray
        State space matrices
    Z : np.ndarray
        Measurement matrix (n_obs x n_states)
    D : np.ndarray
        Measurement constant (n_obs,)
    x_T : np.ndarray
        Initial state at time T
    horizon : int
        Forecast horizon
    n_paths : int
        Number of simulation paths
    shock_cov : np.ndarray, optional
        Structural shock covariance
    measurement_cov : np.ndarray, optional
        Measurement error covariance (default: zeros)
    seed : int, optional
        Random seed

    Returns:
    -------
    ForecastResult
        Observable forecast with uncertainty
    """
    # Forecast states
    state_forecast = forecast_states(T, R, C, x_T, horizon, n_paths, shock_cov, seed)

    n_obs = Z.shape[0]

    # Convert state paths to observable paths
    obs_paths = np.zeros((n_paths, horizon, n_obs))

    # Ensure paths are available (they will be since forecast_states always returns paths)
    assert state_forecast.paths is not None, "State forecast paths should not be None"

    for i in range(n_paths):
        for h in range(horizon):
            obs_paths[i, h] = Z @ state_forecast.paths[i, h] + D

            # Add measurement error if specified
            if measurement_cov is not None:
                obs_paths[i, h] += np.random.multivariate_normal(np.zeros(n_obs), measurement_cov)

    # Compute statistics
    mean_forecast = np.mean(obs_paths, axis=0)

    percentiles = [0.05, 0.16, 0.84, 0.95]
    bands = {}

    for p in percentiles:
        bands[p] = np.percentile(obs_paths, p * 100, axis=0)

    band_dict = {
        0.90: (bands[0.05], bands[0.95]),
        0.68: (bands[0.16], bands[0.84]),
    }

    return ForecastResult(mean=mean_forecast, bands=band_dict, paths=obs_paths)


def conditional_forecast(
    T: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    Z: np.ndarray,
    D: np.ndarray,
    x_T: np.ndarray,
    horizon: int,
    conditions: dict[int, dict[int, float]],
    n_paths: int = 1000,
    shock_cov: np.ndarray | None = None,
    seed: int | None = None,
) -> ForecastResult:
    """
    Generate conditional forecasts with specified observable paths.

    Conditional forecasting constrains certain observables to follow
    specified paths while allowing others to vary.

    Parameters
    ----------
    T, R, C, Z, D : np.ndarray
        State space matrices
    x_T : np.ndarray
        Initial state
    horizon : int
        Forecast horizon
    conditions : dict
        Dictionary mapping time periods to observable constraints
        Format: {period: {obs_idx: value, ...}, ...}
        Example: {0: {0: 2.5}, 1: {0: 2.6}} constrains obs 0 in periods 0-1
    n_paths : int
        Number of simulation paths
    shock_cov : np.ndarray, optional
        Shock covariance
    seed : int, optional
        Random seed

    Returns:
    -------
    ForecastResult
        Conditional forecast results

    Notes:
    -----
    This uses a simple rejection sampling approach. For more efficient
    conditional forecasting, use the Kalman smoother or
    importance sampling methods.
    """
    if seed is not None:
        np.random.seed(seed)

    # First generate unconditional forecast to get shocks
    unconditional = forecast_observables(
        T, R, C, Z, D, x_T, horizon, n_paths * 10, shock_cov, None, seed
    )

    # Filter paths that satisfy conditions (approximately)
    tolerance = 0.01  # Allow small deviation from conditions

    valid_paths = []

    # Ensure paths are available
    assert unconditional.paths is not None, "Unconditional forecast paths should not be None"

    for i in range(unconditional.paths.shape[0]):
        path = unconditional.paths[i]
        is_valid = True

        for t, obs_constraints in conditions.items():
            if t >= horizon:
                continue

            for obs_idx, target_value in obs_constraints.items():
                if abs(path[t, obs_idx] - target_value) > tolerance:
                    is_valid = False
                    break

            if not is_valid:
                break

        if is_valid:
            valid_paths.append(path)

        if len(valid_paths) >= n_paths:
            break

    if len(valid_paths) == 0:
        msg = "No paths satisfy the conditions. Try relaxing constraints or increasing tolerance."
        raise ValueError(
            msg
        )

    # Convert to array
    valid_paths = np.array(valid_paths[:n_paths])

    # Compute statistics
    mean_forecast = np.mean(valid_paths, axis=0)

    percentiles = [0.05, 0.16, 0.84, 0.95]
    bands = {}

    for p in percentiles:
        bands[p] = np.percentile(valid_paths, p * 100, axis=0)

    band_dict = {
        0.90: (bands[0.05], bands[0.95]),
        0.68: (bands[0.16], bands[0.84]),
    }

    return ForecastResult(mean=mean_forecast, bands=band_dict, paths=valid_paths)


def compute_forecast_bands(
    forecast_paths: np.ndarray, confidence_levels: list | None = None
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """
    Compute forecast uncertainty bands from simulation paths.

    Parameters
    ----------
    forecast_paths : np.ndarray
        Array of forecast paths (n_paths x horizon x n_vars)
    confidence_levels : list
        Confidence levels for bands (e.g., [0.68, 0.90, 0.95])

    Returns:
    -------
    dict
        Dictionary mapping confidence levels to (lower, upper) bands
    """
    if confidence_levels is None:
        confidence_levels = [0.68, 0.9, 0.95]
    bands = {}

    for level in confidence_levels:
        alpha = 1 - level
        lower_p = (alpha / 2) * 100
        upper_p = (1 - alpha / 2) * 100

        lower = np.percentile(forecast_paths, lower_p, axis=0)
        upper = np.percentile(forecast_paths, upper_p, axis=0)

        bands[level] = (lower, upper)

    return bands


def forecast_from_posterior(
    posterior_samples: np.ndarray,
    posterior_weights: np.ndarray,
    model,
    x_T: np.ndarray,
    horizon: int,
    n_forecast_paths: int = 100,
    n_posterior_draws: int = 100,
    seed: int | None = None,
) -> ForecastResult:
    """
    Generate forecasts incorporating parameter uncertainty from posterior.

    Parameters
    ----------
    posterior_samples : np.ndarray
        Posterior parameter samples (n_samples x n_params)
    posterior_weights : np.ndarray
        Weights for posterior samples
    model : DSGEModel
        DSGE model instance
    x_T : np.ndarray
        Initial state
    horizon : int
        Forecast horizon
    n_forecast_paths : int
        Number of forecast paths per parameter draw
    n_posterior_draws : int
        Number of posterior draws to use
    seed : int, optional
        Random seed

    Returns:
    -------
    ForecastResult
        Forecast incorporating parameter uncertainty
    """
    if seed is not None:
        np.random.seed(seed)

    # Sample from posterior
    indices = np.random.choice(
        len(posterior_samples),
        size=n_posterior_draws,
        p=posterior_weights / np.sum(posterior_weights),
    )

    param_draws = posterior_samples[indices]

    # Store all forecast paths
    all_paths = []

    for i, params in enumerate(param_draws):
        # Solve model at these parameters
        from dsge.solvers.linear import solve_linear_model

        mats = model.system_matrices(params)

        solution, _info = solve_linear_model(
            Gamma0=mats["Gamma0"],
            Gamma1=mats["Gamma1"],
            Psi=mats["Psi"],
            Pi=mats["Pi"],
            n_states=model.spec.n_states,
        )

        if solution is None:
            continue

        # Get measurement equation
        Z, D = model.measurement_equation(params)

        # Generate forecasts
        forecast = forecast_observables(
            T=solution.T,
            R=solution.R,
            C=solution.C,
            Z=Z,
            D=D,
            x_T=x_T,
            horizon=horizon,
            n_paths=n_forecast_paths,
            seed=seed + i if seed is not None else None,
        )

        all_paths.append(forecast.paths)

    # Combine all paths
    combined_paths = np.concatenate(all_paths, axis=0)

    # Compute statistics
    mean_forecast = np.mean(combined_paths, axis=0)

    bands = compute_forecast_bands(combined_paths)

    return ForecastResult(mean=mean_forecast, bands=bands, paths=combined_paths)
