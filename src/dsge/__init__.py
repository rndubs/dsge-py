"""
DSGE-PY: A Python framework for estimating Dynamic Stochastic General Equilibrium models.

This package provides tools for:
- Specifying DSGE models
- Solving linear(ized) models
- Bayesian estimation via Sequential Monte Carlo
- OccBin solver for occasionally binding constraints
"""

__version__ = "0.1.0"

from .config import Settings, get_fred_api_key, get_settings
from .estimation import SMCSampler, estimate_dsge, log_likelihood_linear
from .filters import KalmanFilter, kalman_filter, kalman_smoother
from .models import DSGEModel, ModelSpecification, Parameter, ParameterSet, Prior
from .solvers import LinearSolution, solve_linear_model

__all__ = [
    "DSGEModel",
    "KalmanFilter",
    "LinearSolution",
    "ModelSpecification",
    "Parameter",
    "ParameterSet",
    "Prior",
    "SMCSampler",
    "Settings",
    "estimate_dsge",
    "get_fred_api_key",
    "get_settings",
    "kalman_filter",
    "kalman_smoother",
    "log_likelihood_linear",
    "solve_linear_model",
]
