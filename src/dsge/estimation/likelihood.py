"""
Likelihood evaluation for DSGE models.
"""

import numpy as np
from typing import Optional
from ..solvers.linear import solve_linear_model, LinearSolution
from ..filters.kalman import kalman_filter
from ..models.base import DSGEModel


def log_likelihood_linear(model: DSGEModel,
                          data: np.ndarray,
                          params: Optional[np.ndarray] = None) -> float:
    """
    Evaluate log likelihood for a linear DSGE model.

    Parameters
    ----------
    model : DSGEModel
        DSGE model specification
    data : array (T x n_obs)
        Observed data
    params : array, optional
        Parameter values. If None, use current model parameters.

    Returns
    -------
    log_likelihood : float
        Log likelihood of the data given the model and parameters
    """
    # Update parameters
    if params is not None:
        model.parameters.set_values(params)

    try:
        # Get system matrices
        system_mats = model.system_matrices()

        # Solve the model
        solution, info = solve_linear_model(
            system_mats['Gamma0'],
            system_mats['Gamma1'],
            system_mats['Psi'],
            system_mats['Pi'],
            model.spec.n_states
        )

        # Check if solution is stable
        if not solution.is_stable:
            return -np.inf

        # Get measurement equation
        Z, D = model.measurement_equation()
        solution.Z = Z
        solution.D = D

        # Get covariance matrices
        Q = model.shock_covariance()
        H = model.measurement_error_covariance()

        # Run Kalman filter
        kf_results = kalman_filter(
            y=data,
            T=solution.T,
            R=solution.R,
            Q=Q,
            Z=Z,
            D=D,
            H=H
        )

        # Add prior to likelihood
        log_prior = model.parameters.log_prior()

        return kf_results.log_likelihood + log_prior

    except Exception as e:
        # If any error occurs (numerical issues, etc.), return -inf
        return -np.inf
