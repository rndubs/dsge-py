"""
Likelihood evaluation for DSGE models.
"""

import numpy as np
from typing import Optional
from ..solvers.linear import solve_linear_model, LinearSolution
from ..solvers.occbin import OccBinConstraint
from ..filters.kalman import kalman_filter
from ..filters.occbin_filter import occbin_filter
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


def log_likelihood_occbin(model_M1: DSGEModel,
                          model_M2: DSGEModel,
                          constraint: OccBinConstraint,
                          data: np.ndarray,
                          params: Optional[np.ndarray] = None,
                          max_iter: int = 50) -> float:
    """
    Evaluate log likelihood for an OccBin DSGE model with regime switching.

    This uses the OccBin Kalman filter to handle occasionally binding constraints.

    Parameters
    ----------
    model_M1 : DSGEModel
        Reference regime model (constraint slack)
    model_M2 : DSGEModel
        Alternative regime model (constraint binding)
    constraint : OccBinConstraint
        Constraint specification
    data : array (T x n_obs)
        Observed data
    params : array, optional
        Parameter values. If None, use current model parameters.
    max_iter : int
        Maximum OccBin filter iterations

    Returns
    -------
    log_likelihood : float
        Log likelihood of the data given the model and parameters
    """
    # Update parameters (both models share same parameters)
    if params is not None:
        model_M1.parameters.set_values(params)
        model_M2.parameters.set_values(params)

    try:
        # Get system matrices for both regimes
        system_mats_M1 = model_M1.system_matrices()
        system_mats_M2 = model_M2.system_matrices()

        # Solve both regimes
        solution_M1, info_M1 = solve_linear_model(
            system_mats_M1['Gamma0'],
            system_mats_M1['Gamma1'],
            system_mats_M1['Psi'],
            system_mats_M1['Pi'],
            model_M1.spec.n_states
        )

        solution_M2, info_M2 = solve_linear_model(
            system_mats_M2['Gamma0'],
            system_mats_M2['Gamma1'],
            system_mats_M2['Psi'],
            system_mats_M2['Pi'],
            model_M2.spec.n_states
        )

        # Check if both solutions are stable
        if not solution_M1.is_stable or not solution_M2.is_stable:
            return -np.inf

        # Get measurement equation (same for both regimes)
        Z, D = model_M1.measurement_equation()

        # Get covariance matrices
        Q_M1 = model_M1.shock_covariance()
        Q_M2 = model_M2.shock_covariance()
        H = model_M1.measurement_error_covariance()

        # Set Q matrices in solutions
        solution_M1.Q = Q_M1
        solution_M2.Q = Q_M2

        # Run OccBin filter
        filter_results = occbin_filter(
            y=data,
            solution_M1=solution_M1,
            solution_M2=solution_M2,
            constraint=constraint,
            Z=Z,
            D=D,
            H=H,
            max_iter=max_iter
        )

        # Check if filter converged
        if filter_results.n_iterations >= max_iter:
            # Did not converge
            return -np.inf

        # Add prior to likelihood
        log_prior = model_M1.parameters.log_prior()

        return filter_results.log_likelihood + log_prior

    except Exception as e:
        # If any error occurs (numerical issues, etc.), return -inf
        return -np.inf
