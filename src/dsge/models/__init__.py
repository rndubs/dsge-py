"""
DSGE Model Specification Module

This module provides the base classes and utilities for specifying DSGE models.
"""

from .base import DSGEModel, ModelSpecification
from .parameters import Parameter, ParameterSet, Prior

__all__ = ['DSGEModel', 'ModelSpecification', 'Parameter', 'ParameterSet', 'Prior']
