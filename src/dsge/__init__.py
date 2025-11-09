"""
DSGE-PY: A Python framework for estimating Dynamic Stochastic General Equilibrium models.

This package provides tools for:
- Specifying DSGE models
- Solving linear(ized) models
- Bayesian estimation via Sequential Monte Carlo
- OccBin solver for occasionally binding constraints
"""

__version__ = "0.1.0"

from .models import DSGEModel, ModelSpecification, Parameter, ParameterSet, Prior
from .solvers import solve_linear_model, LinearSolution
from .filters import KalmanFilter, kalman_filter, kalman_smoother
from .estimation import estimate_dsge, SMCSampler, log_likelihood_linear
from .config import Settings, get_settings, get_fred_api_key

__all__ = [
    'DSGEModel',
    'ModelSpecification',
    'Parameter',
    'ParameterSet',
    'Prior',
    'solve_linear_model',
    'LinearSolution',
    'KalmanFilter',
    'kalman_filter',
    'kalman_smoother',
    'estimate_dsge',
    'SMCSampler',
    'log_likelihood_linear',
    'Settings',
    'get_settings',
    'get_fred_api_key',
]
