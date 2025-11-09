"""
Kalman filter and smoother for linear state space models.

Implements standard Kalman filtering and smoothing algorithms
for likelihood evaluation and state inference in DSGE models.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class KalmanFilter:
    """
    Results from Kalman filtering.

    Attributes
    ----------
    log_likelihood : float
        Log likelihood of the data
    filtered_states : array (T x n_states)
        Filtered state estimates E[s_t | y_{1:t}]
    filtered_covariances : array (T x n_states x n_states)
        Filtered state covariance P_t|t
    predicted_states : array (T x n_states)
        Predicted state estimates E[s_t | y_{1:t-1}]
    predicted_covariances : array (T x n_states x n_states)
        Predicted state covariance P_t|t-1
    forecast_errors : array (T x n_obs)
        One-step ahead forecast errors v_t
    forecast_error_covariances : array (T x n_obs x n_obs)
        Forecast error covariance F_t
    """
    log_likelihood: float
    filtered_states: np.ndarray
    filtered_covariances: np.ndarray
    predicted_states: np.ndarray
    predicted_covariances: np.ndarray
    forecast_errors: np.ndarray
    forecast_error_covariances: np.ndarray


def kalman_filter(y: np.ndarray,
                  T: np.ndarray,
                  R: np.ndarray,
                  Q: np.ndarray,
                  Z: np.ndarray,
                  D: np.ndarray,
                  H: np.ndarray,
                  a0: Optional[np.ndarray] = None,
                  P0: Optional[np.ndarray] = None) -> KalmanFilter:
    """
    Kalman filter for linear state space model.

    State equation:    s_t = T * s_{t-1} + R * ε_t,  ε_t ~ N(0, Q)
    Observation eq:    y_t = Z * s_t + D + η_t,      η_t ~ N(0, H)

    Parameters
    ----------
    y : array (T x n_obs)
        Observed data
    T : array (n_states x n_states)
        State transition matrix
    R : array (n_states x n_shocks)
        Shock loading matrix
    Q : array (n_shocks x n_shocks)
        Shock covariance matrix
    Z : array (n_obs x n_states)
        Measurement matrix
    D : array (n_obs,)
        Measurement constant
    H : array (n_obs x n_obs)
        Measurement error covariance
    a0 : array (n_states,), optional
        Initial state mean. Default: zero
    P0 : array (n_states x n_states), optional
        Initial state covariance. Default: solve Lyapunov equation

    Returns
    -------
    KalmanFilter
        Filtering results including log likelihood
    """
    T_periods, n_obs = y.shape
    n_states = T.shape[0]

    # Initialize storage
    filtered_states = np.zeros((T_periods, n_states))
    filtered_covariances = np.zeros((T_periods, n_states, n_states))
    predicted_states = np.zeros((T_periods, n_states))
    predicted_covariances = np.zeros((T_periods, n_states, n_states))
    forecast_errors = np.zeros((T_periods, n_obs))
    forecast_error_covariances = np.zeros((T_periods, n_obs, n_obs))

    # Initial conditions
    if a0 is None:
        a0 = np.zeros(n_states)
    if P0 is None:
        # Solve discrete Lyapunov equation for unconditional variance
        RQR = R @ Q @ R.T
        P0 = solve_discrete_lyapunov(T, RQR)

    # Initial predicted state
    a = a0.copy()
    P = P0.copy()

    log_likelihood = 0.0

    # Kalman filter iterations
    for t in range(T_periods):
        # Store predicted state and covariance
        predicted_states[t] = a
        predicted_covariances[t] = P

        # Handle missing data
        y_t = y[t]
        if np.any(np.isnan(y_t)):
            # If observation is missing, skip update step
            filtered_states[t] = a
            filtered_covariances[t] = P
            forecast_errors[t] = np.nan
            forecast_error_covariances[t] = np.nan
        else:
            # Forecast error
            v = y_t - (Z @ a + D)
            forecast_errors[t] = v

            # Forecast error covariance
            F = Z @ P @ Z.T + H
            forecast_error_covariances[t] = F

            # Kalman gain
            try:
                F_inv = np.linalg.inv(F)
                K = P @ Z.T @ F_inv

                # Update state estimate
                a_update = a + K @ v
                P_update = P - K @ Z @ P

                filtered_states[t] = a_update
                filtered_covariances[t] = P_update

                # Update log likelihood
                sign, logdet = np.linalg.slogdet(F)
                if sign <= 0:
                    log_likelihood = -np.inf
                else:
                    log_likelihood += -0.5 * (n_obs * np.log(2 * np.pi) +
                                             logdet +
                                             v.T @ F_inv @ v)

                a = a_update
                P = P_update
            except np.linalg.LinAlgError:
                # Singular forecast error covariance
                filtered_states[t] = a
                filtered_covariances[t] = P
                log_likelihood = -np.inf

        # Predict next period
        a = T @ a
        P = T @ P @ T.T + R @ Q @ R.T

        # Ensure P remains symmetric
        P = 0.5 * (P + P.T)

    return KalmanFilter(
        log_likelihood=log_likelihood,
        filtered_states=filtered_states,
        filtered_covariances=filtered_covariances,
        predicted_states=predicted_states,
        predicted_covariances=predicted_covariances,
        forecast_errors=forecast_errors,
        forecast_error_covariances=forecast_error_covariances
    )


def kalman_smoother(filter_results: KalmanFilter,
                   T: np.ndarray,
                   R: np.ndarray,
                   Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kalman smoother (Rauch-Tung-Striebel backward recursion).

    Computes smoothed state estimates E[s_t | y_{1:T}].

    Parameters
    ----------
    filter_results : KalmanFilter
        Results from forward Kalman filter
    T : array (n_states x n_states)
        State transition matrix
    R : array (n_states x n_shocks)
        Shock loading matrix
    Q : array (n_shocks x n_shocks)
        Shock covariance matrix

    Returns
    -------
    smoothed_states : array (T x n_states)
        Smoothed state estimates
    smoothed_covariances : array (T x n_states x n_states)
        Smoothed state covariances
    """
    T_periods, n_states = filter_results.filtered_states.shape

    smoothed_states = np.zeros((T_periods, n_states))
    smoothed_covariances = np.zeros((T_periods, n_states, n_states))

    # Initialize with filtered values at T
    smoothed_states[-1] = filter_results.filtered_states[-1]
    smoothed_covariances[-1] = filter_results.filtered_covariances[-1]

    # Backward recursion
    for t in range(T_periods - 2, -1, -1):
        a_t = filter_results.filtered_states[t]
        P_t = filter_results.filtered_covariances[t]
        a_t1 = filter_results.predicted_states[t + 1]
        P_t1 = filter_results.predicted_covariances[t + 1]

        # Smoother gain
        try:
            J_t = P_t @ T.T @ np.linalg.inv(P_t1)

            # Smoothed state
            smoothed_states[t] = a_t + J_t @ (smoothed_states[t + 1] - a_t1)

            # Smoothed covariance
            smoothed_covariances[t] = P_t + J_t @ (smoothed_covariances[t + 1] - P_t1) @ J_t.T

            # Ensure symmetry
            smoothed_covariances[t] = 0.5 * (smoothed_covariances[t] + smoothed_covariances[t].T)
        except np.linalg.LinAlgError:
            # Singular covariance matrix
            smoothed_states[t] = a_t
            smoothed_covariances[t] = P_t

    return smoothed_states, smoothed_covariances


def solve_discrete_lyapunov(A: np.ndarray, Q: np.ndarray, method: str = 'direct') -> np.ndarray:
    """
    Solve the discrete Lyapunov equation: X = A * X * A' + Q.

    Parameters
    ----------
    A : array (n x n)
        Transition matrix
    Q : array (n x n)
        Covariance matrix
    method : str
        Solution method ('direct' or 'iterative')

    Returns
    -------
    X : array (n x n)
        Solution to Lyapunov equation
    """
    if method == 'direct':
        try:
            from scipy.linalg import solve_discrete_lyapunov as scipy_solve
            return scipy_solve(A, Q)
        except:
            # Fall back to iterative method
            method = 'iterative'

    if method == 'iterative':
        # Iterative solution
        n = A.shape[0]
        X = Q.copy()
        for _ in range(100):  # Max iterations
            X_new = A @ X @ A.T + Q
            if np.allclose(X, X_new, atol=1e-8):
                return X_new
            X = X_new
        return X

    raise ValueError(f"Unknown method: {method}")
